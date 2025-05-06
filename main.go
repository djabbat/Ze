package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"sync/atomic"
	"syscall"
	"time"
)

// Config holds the application configuration
type Config struct {
	CrumbSize          int     `json:"CrumbSize"`
	ChunkSize          int     `json:"ChunkSize"`
	CounterValue       int     `json:"CounterValue"`
	PredictIncrement   int     `json:"PredictIncrement"`
	Increment          int     `json:"Increment"`
	ActualizationValue float64 `json:"ActualizationValue"`
	FiltrationValue    float64 `json:"FiltrationValue"`
	InputFiles         []string `json:"InputFiles"`
	OutputFolder       string   `json:"OutputFolder"`
	MatchSuffix        string   `json:"MatchSuffix"`
}

// Global variables
var (
	config         Config
	totalCrumbs    uint64
	totalMatches   uint64
	interrupted    uint32
	logger         *log.Logger
)

func init() {
	// Initialize logger
	logger = log.New(os.Stdout, "", log.LstdFlags|log.Lshortfile)
}

func loadConfig() error {
	file, err := os.Open("config.json")
	if err != nil {
		return fmt.Errorf("error opening config file: %v", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&config); err != nil {
		return fmt.Errorf("error decoding config: %v", err)
	}

	// Validate required parameters
	if config.CrumbSize <= 0 || config.ChunkSize <= 0 || config.CounterValue <= 0 ||
		config.PredictIncrement <= 0 || config.Increment <= 0 || config.ActualizationValue <= 0 ||
		config.ActualizationValue > 1 || config.FiltrationValue <= 0 || config.FiltrationValue >= 1 {
		return fmt.Errorf("invalid configuration values")
	}

	return nil
}

func setupSignalHandler() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		atomic.StoreUint32(&interrupted, 1)
		logger.Println("Interrupt signal received, shutting down...")
	}()
}

func splitCrumbs(chunk []byte, reverse bool) []uint32 {
	// Padding with zeros if needed
	padding := (config.CrumbSize - len(chunk)%config.CrumbSize) % config.CrumbSize
	padded := make([]byte, len(chunk)+padding)
	copy(padded, chunk)

	var crumbs []uint32

	if reverse {
		// Process from end
		for i := len(padded) - config.CrumbSize; i >= -config.CrumbSize; i -= config.CrumbSize {
			var crumb []byte
			if i < 0 {
				crumb = padded[0:config.CrumbSize]
			} else {
				crumb = padded[i : i+config.CrumbSize]
			}

			var value uint32
			switch config.CrumbSize {
			case 1:
				value = uint32(crumb[0])
			case 2:
				value = uint32(binary.BigEndian.Uint16(crumb))
			case 4:
				value = binary.BigEndian.Uint32(crumb)
			default:
				// For larger crumb sizes, we'll just take the first 4 bytes
				if len(crumb) > 4 {
					crumb = crumb[:4]
				}
				for len(crumb) < 4 {
					crumb = append(crumb, 0)
				}
				value = binary.BigEndian.Uint32(crumb)
			}
			crumbs = append(crumbs, value)
		}
	} else {
		// Process from start
		for i := 0; i < len(padded); i += config.CrumbSize {
			end := i + config.CrumbSize
			if end > len(padded) {
				end = len(padded)
			}
			crumb := padded[i:end]

			var value uint32
			switch config.CrumbSize {
			case 1:
				value = uint32(crumb[0])
			case 2:
				value = uint32(binary.BigEndian.Uint16(crumb))
			case 4:
				value = binary.BigEndian.Uint32(crumb)
			default:
				// For larger crumb sizes, we'll just take the first 4 bytes
				if len(crumb) > 4 {
					crumb = crumb[:4]
				}
				for len(crumb) < 4 {
					crumb = append(crumb, 0)
				}
				value = binary.BigEndian.Uint32(crumb)
			}
			crumbs = append(crumbs, value)
		}
	}

	return crumbs
}

func thresholdCheck(counters map[uint32]int) bool {
	thresholdReached := false
	for _, v := range counters {
		if v >= config.CounterValue {
			thresholdReached = true
			break
		}
	}

	if thresholdReached {
		for k := range counters {
			counters[k] /= 2
		}
	}

	return thresholdReached
}

func processCrumb(counters map[uint32]int, crumb uint32, processorName string) {
	atomic.AddUint64(&totalCrumbs, 1)

	// Check for interrupt
	if atomic.LoadUint32(&interrupted) == 1 {
		return
	}

	// Threshold check
	thresholdCheck(counters)

	// Check for match
	if count, exists := counters[crumb]; exists {
		atomic.AddUint64(&totalMatches, 1)
		if count > config.CounterValue/2 {
			counters[crumb] += config.PredictIncrement
		} else {
			counters[crumb] += config.Increment
		}
	} else {
		counters[crumb] = config.Increment
	}

	// Perform filtration less frequently
	if totalCrumbs%10000 == 0 && len(counters) > 1000 {
		filterCounters(counters, processorName)
	}
}

func filterCounters(counters map[uint32]int, processorName string) {
	if len(counters) == 0 {
		return
	}

	// Get all values and sort them to find the threshold
	values := make([]int, 0, len(counters))
	for _, v := range counters {
		values = append(values, v)
	}
	sort.Ints(values)

	thresholdIndex := int(math.Floor(float64(len(values)) * config.FiltrationValue))
	if thresholdIndex >= len(values) {
		thresholdIndex = len(values) - 1
	}
	threshold := values[thresholdIndex]

	// Remove counters below the threshold
	for k, v := range counters {
		if v <= threshold {
			delete(counters, k)
		}
	}

	logger.Printf("[%s] Filtration: removed %d counters (remaining: %d)\n", 
		processorName, len(values)-len(counters), len(counters))
}

func saveState(filePath string, counters map[uint32]int) error {
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return err
	}

	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	for k, v := range counters {
		if err := binary.Write(file, binary.BigEndian, k); err != nil {
			return err
		}
		if err := binary.Write(file, binary.BigEndian, uint32(v)); err != nil {
			return err
		}
	}

	logger.Printf("State saved to %s (%d entries)\n", filePath, len(counters))
	return nil
}

func saveMatchStats(filePath string, matches, crumbs uint64) error {
	if err := os.MkdirAll(filepath.Dir(filePath), 0755); err != nil {
		return err
	}

	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Write total matches (key=0)
	if err := binary.Write(file, binary.BigEndian, uint32(0)); err != nil {
		return err
	}
	if err := binary.Write(file, binary.BigEndian, uint32(matches)); err != nil {
		return err
	}

	// Write total crumbs (key=1)
	if err := binary.Write(file, binary.BigEndian, uint32(1)); err != nil {
		return err
	}
	if err := binary.Write(file, binary.BigEndian, uint32(crumbs)); err != nil {
		return err
	}

	return nil
}

func processFile(filePath string) error {
	if atomic.LoadUint32(&interrupted) == 1 {
		return nil
	}

	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("error opening file %s: %v", filePath, err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return fmt.Errorf("error getting file info for %s: %v", filePath, err)
	}

	base := filepath.Base(filePath)
	outBegin := filepath.Join(config.OutputFolder, base[:len(base)-len(filepath.Ext(base))]+"_begin.bin")
	outInverse := filepath.Join(config.OutputFolder, base[:len(base)-len(filepath.Ext(base))]+"_inverse.bin")
	outMatch := filepath.Join(config.OutputFolder, base[:len(base)-len(filepath.Ext(base))]+config.MatchSuffix)

	countersBegin := make(map[uint32]int)
	countersInverse := make(map[uint32]int)
	fileCrumbs := uint64(0)
	lastLogTime := time.Now()

	logger.Printf("Processing file: %s (size: %.2f MB)\n", filePath, float64(fileInfo.Size())/1024/1024)

	chunk := make([]byte, config.ChunkSize)
	for {
		if atomic.LoadUint32(&interrupted) == 1 {
			break
		}

		n, err := file.Read(chunk)
		if err != nil && err != io.EOF {
			return fmt.Errorf("error reading file %s: %v", filePath, err)
		}
		if n == 0 {
			break
		}

		// Process chunk
		actualChunk := chunk[:n]
		for _, crumb := range splitCrumbs(actualChunk, false) {
			processCrumb(countersBegin, crumb, "beginning")
			fileCrumbs++
		}

		for _, crumb := range splitCrumbs(actualChunk, true) {
			processCrumb(countersInverse, crumb, "inverse")
			fileCrumbs++
		}

		// Log progress no more than once every 5 seconds
		if time.Since(lastLogTime) > 5*time.Second {
			pos, _ := file.Seek(0, io.SeekCurrent)
			progress := float64(pos) / float64(fileInfo.Size()) * 100
			logger.Printf("Progress: %.1f%%\n", progress)
			lastLogTime = time.Now()
		}
	}

	if atomic.LoadUint32(&interrupted) == 0 {
		if err := saveState(outBegin, countersBegin); err != nil {
			return fmt.Errorf("error saving beginning counters: %v", err)
		}
		if err := saveState(outInverse, countersInverse); err != nil {
			return fmt.Errorf("error saving inverse counters: %v", err)
		}
		if err := saveMatchStats(outMatch, atomic.LoadUint64(&totalMatches), atomic.LoadUint64(&totalCrumbs)); err != nil {
			return fmt.Errorf("error saving match stats: %v", err)
		}

		logger.Printf("File processed: %s\n", filePath)
		logger.Printf("  Total crumbs: %d\n", fileCrumbs)
		logger.Printf("  Matches: %d\n", atomic.LoadUint64(&totalMatches))
	}

	return nil
}

func main() {
	setupSignalHandler()

	logger.Println("=== Processing started ===")
	startTime := time.Now()

	if err := loadConfig(); err != nil {
		logger.Fatalf("Cannot continue without valid configuration: %v", err)
	}

	var exitCode int
	for _, filePath := range config.InputFiles {
		if atomic.LoadUint32(&interrupted) == 1 {
			break
		}

		if err := processFile(filePath); err != nil {
			logger.Printf("Error processing file %s: %v\n", filePath, err)
			exitCode = 1
		}
	}

	duration := time.Since(startTime)
	logger.Printf("Total execution time: %.2f seconds\n", duration.Seconds())
	logger.Printf("Total crumbs processed: %d\n", atomic.LoadUint64(&totalCrumbs))
	logger.Printf("Total matches found: %d\n", atomic.LoadUint64(&totalMatches))

	os.Exit(exitCode)
}