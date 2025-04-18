package processor

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"ze/internal/utils"
)

const (
	counterEntrySize = 8 // 4 байта data + 4 байта value
	maxCounters      = 65536
)

var (
	ErrCounterNotFound  = errors.New("counter not found")
	ErrMaxCounters      = errors.New("maximum counters reached")
	ErrInvalidFileSize  = errors.New("invalid file size")
	ErrInvalidPath      = errors.New("invalid file path")
)

type CounterSystem struct {
	filepath       string
	counters       []Counter
	resetThreshold uint32
	mu             sync.RWMutex
	logger         *utils.Logger
}

func NewCounterSystem(filepath string, initialSize int, threshold uint32, logger *utils.Logger) (*CounterSystem, error) {
	if initialSize <= 0 || initialSize > maxCounters {
		return nil, fmt.Errorf("invalid initial size: %d (must be 1-%d)", initialSize, maxCounters)
	}

	cs := &CounterSystem{
		filepath:       filepath,
		counters:       make([]Counter, 0, initialSize),
		resetThreshold: threshold,
		logger:         logger,
	}

	if err := cs.initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize counter system: %v", err)
	}

	return cs, nil
}

func (cs *CounterSystem) initialize() error {
	cs.logger.Debug("Initializing counter system at %s", cs.filepath)

	if err := os.MkdirAll(filepath.Dir(cs.filepath), 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	if _, err := os.Stat(cs.filepath); os.IsNotExist(err) {
		return cs.initializeFile()
	}
	return cs.load()
}

func (cs *CounterSystem) initializeFile() error {
	cs.logger.Debug("Creating new counter file: %s", cs.filepath)

	file, err := os.Create(cs.filepath)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	for i := 0; i < cap(cs.counters); i++ {
		var crumb [4]byte
		binary.BigEndian.PutUint32(crumb[:], uint32(i))

		counter := Counter{
			Data:  crumb,
			Value: 1,
		}

		if err := binary.Write(file, binary.BigEndian, counter.Data); err != nil {
			return fmt.Errorf("failed to write data: %v", err)
		}
		if err := binary.Write(file, binary.BigEndian, counter.Value); err != nil {
			return fmt.Errorf("failed to write value: %v", err)
		}

		cs.counters = append(cs.counters, counter)
	}

	cs.logger.Info("Initialized %d counters in %s", len(cs.counters), cs.filepath)
	return nil
}

func (cs *CounterSystem) load() error {
	cs.logger.Debug("Loading existing counter file: %s", cs.filepath)

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

	var data [4]byte
	var value uint32

	for {
		if err := binary.Read(file, binary.BigEndian, &data); err != nil {
			break
		}
		if err := binary.Read(file, binary.BigEndian, &value); err != nil {
			break
		}

		cs.counters = append(cs.counters, Counter{
			Data:  data,
			Value: value,
		})
	}

	cs.logger.Info("Loaded %d counters from %s", len(cs.counters), cs.filepath)
	return nil
}

func (cs *CounterSystem) ProcessCrumb(crumb [4]byte, increment uint32) error {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	cs.logger.Debug("Processing crumb: %v with increment %d", crumb, increment)

	for i := range cs.counters {
		if cs.counters[i].Data == crumb {
			oldValue := cs.counters[i].Value
			cs.counters[i].Increment(increment)
			cs.logger.Debug("Counter updated: %v %d -> %d", crumb, oldValue, cs.counters[i].Value)
			
			if err := cs.checkReset(); err != nil {
				return err
			}
			return cs.save()
		}
	}

	if len(cs.counters) >= maxCounters {
		return ErrMaxCounters
	}

	cs.counters = append(cs.counters, Counter{
		Data:  crumb,
		Value: increment,
	})
	
	cs.logger.Debug("New counter created: %v = %d", crumb, increment)
	return cs.save()
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
		oldValue := cs.counters[i].Value
		cs.counters[i].Reset()
		cs.logger.Debug("Counter reset: %d -> %d", oldValue, cs.counters[i].Value)
	}
	return cs.save()
}

func (cs *CounterSystem) save() error {
	tmpPath := cs.filepath + ".tmp"
	file, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %v", err)
	}

	for _, counter := range cs.counters {
		if err := binary.Write(file, binary.BigEndian, counter.Data); err != nil {
			file.Close()
			os.Remove(tmpPath)
			return fmt.Errorf("failed to write data: %v", err)
		}
		if err := binary.Write(file, binary.BigEndian, counter.Value); err != nil {
			file.Close()
			os.Remove(tmpPath)
			return fmt.Errorf("failed to write value: %v", err)
		}
	}

	if err := file.Close(); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to close file: %v", err)
	}

	if err := os.Rename(tmpPath, cs.filepath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to rename file: %v", err)
	}

	cs.logger.Debug("Saved %d counters to %s", len(cs.counters), cs.filepath)
	return nil
}