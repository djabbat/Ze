package processor

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"ze/internal/utils"
)

var (
	ErrCounterNotFound  = errors.New("counter not found")
	ErrMaxCounters      = errors.New("maximum counters reached")
	ErrInvalidFileSize  = errors.New("invalid file size")
	ErrInvalidPath      = errors.New("invalid file path")
)

const (
	counterEntrySize = 16 // 4 (ID) + 4 (data) + 4 (value) + 4 (matches)
	maxCounters      = 1 << 24
)

type Counter struct {
	ID      uint32
	Data    [4]byte
	Value   uint32
	Matches uint32
}

type CounterSystem struct {
	filepath        string
	matchesPath     string
	counters        []Counter
	resetThreshold  uint32
	mu              sync.RWMutex
	logger          *utils.Logger
	config          *Config
	chunkCounter    int
	totalMatches    uint64
	lastID          uint32
}

func NewCounterSystem(countersPath, matchesPath string, initialSize int, threshold uint32, logger *utils.Logger, config *Config) (*CounterSystem, error) {
	if initialSize <= 0 || initialSize > maxCounters {
		return nil, fmt.Errorf("invalid initial size: %d (must be 1-%d)", initialSize, maxCounters)
	}

	if err := os.MkdirAll(filepath.Dir(countersPath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %v", err)
	}

	matchesFile, err := os.OpenFile(matchesPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to create matches file: %v", err)
	}
	matchesFile.Close()

	cs := &CounterSystem{
		filepath:       countersPath,
		matchesPath:    matchesPath,
		counters:       make([]Counter, 0, initialSize),
		resetThreshold: threshold,
		logger:         logger,
		config:         config,
		lastID:         config.Processing.InitialID,
	}

	if err := cs.load(); err != nil {
		if err := cs.initializeFile(); err != nil {
			return nil, fmt.Errorf("failed to initialize counter system: %v", err)
		}
	}

	return cs, nil
}

func (cs *CounterSystem) initializeFile() error {
	file, err := os.Create(cs.filepath)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()
	return nil
}

func (cs *CounterSystem) ProcessCrumb(crumb [4]byte) error {
    cs.mu.Lock()
    defer cs.mu.Unlock()

    if err := cs.checkReset(); err != nil {
        return err
    }

    crumbID := binary.BigEndian.Uint32(crumb[:])
    found := false

    // Сначала проверяем в актуализационной зоне
    actualizationBoundary := int(float64(len(cs.counters)) * cs.config.Processing.Actualization)
    for i := 0; i < len(cs.counters); i++ {
        if cs.counters[i].ID == crumbID {
            found = true
            if i < actualizationBoundary {
                cs.counters[i].Value += cs.config.Processing.PredictIncrement
            } else {
                cs.counters[i].Value += cs.config.Processing.Increment
            }
            cs.counters[i].Matches++
            cs.totalMatches++
            
            // Записываем совпадение в файл
            if err := cs.logMatch(crumb, cs.counters[i].Value); err != nil {
                cs.logger.Error("Failed to log match: %v", err)
            }
            break
        }
    }

    if !found {
        if len(cs.counters) >= maxCounters {
            return ErrMaxCounters
        }
        cs.counters = append(cs.counters, Counter{
            ID:    crumbID,
            Data:  crumb,
            Value: cs.config.Processing.Increment,
        })
    }

    return cs.save()
}

func (cs *CounterSystem) logMatch(crumb [4]byte, value uint32) error {
    file, err := os.OpenFile(cs.matchesPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
    if err != nil {
        return err
    }
    defer file.Close()

    // Записываем ID (crumb) и значение счетчика
    if err := binary.Write(file, binary.BigEndian, crumb); err != nil {
        return err
    }
    if err := binary.Write(file, binary.BigEndian, value); err != nil {
        return err
    }
    return nil
}

func (cs *CounterSystem) IncrementChunkCounter() error {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	cs.chunkCounter++

	if cs.chunkCounter >= cs.config.Processing.FiltrationPeriod {
		cs.chunkCounter = 0
		return cs.filterCounters(cs.config.Processing.FiltrationValue)
	}
	return nil
}

func (cs *CounterSystem) checkReset() error {
	for _, counter := range cs.counters {
		if counter.Value >= cs.resetThreshold {
			cs.logger.Info("Reset threshold reached (%d), resetting counters", cs.resetThreshold)
			return cs.resetAllCounters()
		}
	}
	return nil
}

func (cs *CounterSystem) resetAllCounters() error {
	for i := range cs.counters {
		cs.counters[i].Value /= 2
		cs.counters[i].Matches = 0
	}
	return cs.save()
}

func (cs *CounterSystem) filterCounters(count int) error {
	if count <= 0 || len(cs.counters) <= count {
		return nil
	}

	sort.Slice(cs.counters, func(i, j int) bool {
		return cs.counters[i].Value < cs.counters[j].Value
	})

	cs.counters = cs.counters[count:]
	return cs.save()
}

func (cs *CounterSystem) load() error {
	file, err := os.Open(cs.filepath)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	stat, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat file: %v", err)
	}

	if stat.Size()%counterEntrySize != 0 {
		return ErrInvalidFileSize
	}

	for {
		var counter Counter
		if err := binary.Read(file, binary.BigEndian, &counter.ID); err != nil {
			break
		}
		if err := binary.Read(file, binary.BigEndian, &counter.Data); err != nil {
			break
		}
		if err := binary.Read(file, binary.BigEndian, &counter.Value); err != nil {
			break
		}
		if err := binary.Read(file, binary.BigEndian, &counter.Matches); err != nil {
			break
		}

		cs.counters = append(cs.counters, counter)
		if counter.ID > cs.lastID {
			cs.lastID = counter.ID
		}
	}
	return nil
}

func (cs *CounterSystem) save() error {
    tmpPath := cs.filepath + ".tmp"
    file, err := os.Create(tmpPath)
    if err != nil {
        return fmt.Errorf("failed to create temp file: %v", err)
    }
    defer file.Close()

    // Записываем количество счетчиков
    if err := binary.Write(file, binary.LittleEndian, uint32(len(cs.counters))); err != nil {
        return err
    }

    // Записываем каждый счетчик
    for _, counter := range cs.counters {
        // ID (uint32)
        if err := binary.Write(file, binary.LittleEndian, counter.ID); err != nil {
            return err
        }
        // Data (4 байта)
        if err := binary.Write(file, binary.LittleEndian, counter.Data); err != nil {
            return err
        }
        // Value (uint32)
        if err := binary.Write(file, binary.LittleEndian, counter.Value); err != nil {
            return err
        }
        // Matches (uint32)
        if err := binary.Write(file, binary.LittleEndian, counter.Matches); err != nil {
            return err
        }
    }

    if err := os.Rename(tmpPath, cs.filepath); err != nil {
        return fmt.Errorf("failed to rename file: %v", err)
    }
    return nil
}