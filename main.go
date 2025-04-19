package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"ze/internal/processor"
	"ze/internal/utils"
)

var (
	stopFlag         = false
	stopMutex        sync.Mutex
	visualizationCmd *exec.Cmd
)

func main() {
	_ = os.Remove("data/stop.flag")

	config, err := processor.LoadConfig("config.yaml")
	if err != nil {
		fmt.Printf("Error loading config: %v\n", err)
		os.Exit(1)
	}

	logger := utils.NewLogger("ze.log")
	defer func() {
		if err := logger.Close(); err != nil {
			fmt.Printf("Error closing logger: %v\n", err)
		}
	}()

	if err := startVisualization(logger); err != nil {
		logger.Error("Failed to start visualization: %v", err)
	}

	mode, filename, err := parseArgs()
	if err != nil {
		logger.Error(err.Error())
		printUsage()
		os.Exit(1)
	}

	if err := os.MkdirAll("data", 0755); err != nil {
		logger.Error("Failed to create data directory: %v", err)
		os.Exit(1)
	}

	beginningProc := processor.NewBeginningProcessor(logger, config)
	inverseProc := processor.NewInverseProcessor(logger, config)

	switch mode {
	case "f":
		if err := runSystem(mode, filename, beginningProc, inverseProc, logger, config); err != nil {
			logger.Error("System error: %v", err)
			os.Exit(1)
		}
	case "r":
		if err := runSystem(mode, filename, beginningProc, inverseProc, logger, config); err != nil {
			logger.Error("System error: %v", err)
			os.Exit(1)
		}
	default:
		logger.Error("Unknown mode: %s", mode)
		printUsage()
		os.Exit(1)
	}

	defer func() {
		_ = os.Remove("data/stop.flag")
		stopVisualization(logger)
	}()
}

func startVisualization(logger *utils.Logger) error {
	scriptPath := "visualisation/simple_visualization.py"
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return fmt.Errorf("visualization script not found at %s", scriptPath)
	}

	cmd := exec.Command("python3", scriptPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start visualization: %v", err)
	}

	visualizationCmd = cmd
	logger.Info("Visualization started (PID: %d)", cmd.Process.Pid)
	return nil
}

func stopVisualization(logger *utils.Logger) {
	if visualizationCmd != nil && visualizationCmd.Process != nil {
		if err := visualizationCmd.Process.Kill(); err != nil {
			logger.Error("Failed to stop visualization: %v", err)
		} else {
			logger.Info("Visualization stopped")
		}
	}
}

func parseArgs() (string, string, error) {
	flag.Parse()
	args := flag.Args()

	if len(args) < 1 {
		return "", "", fmt.Errorf("missing mode argument (use 'f' or 'r')")
	}

	mode := args[0]
	switch mode {
	case "f":
		if len(args) < 2 {
			return "", "", fmt.Errorf("file mode requires filename")
		}
		filename := args[1]
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			return "", "", fmt.Errorf("file not found: %s", filename)
		}
		return mode, filename, nil
	case "r":
		return mode, "", nil
	default:
		return "", "", fmt.Errorf("invalid mode: %s (use 'f' or 'r')", mode)
	}
}

func printUsage() {
	fmt.Println("Usage:")
	fmt.Println("  File mode:   ./ze f <filename>")
	fmt.Println("  Stream mode: ./ze r")
}

func processFile(filename string, dataChan chan<- []byte, done <-chan struct{}, 
	logger *utils.Logger, chunkSize int) error {
	defer close(dataChan)
	logger.Info("Starting file processing: %s", filename)

	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("error opening file: %w", err)
	}
	defer file.Close()

	buffer := make([]byte, chunkSize)
	totalBytes := 0

	for {
		select {
		case <-done:
			logger.Info("File processing interrupted")
			return nil
		default:
			n, err := file.Read(buffer)
			if err != nil {
				if err == io.EOF {
					logger.Info("File processing completed, total bytes: %d", totalBytes)
					return nil
				}
				return fmt.Errorf("error reading file: %w", err)
			}

			if n == 0 {
				continue
			}

			totalBytes += n
			data := make([]byte, n)
			copy(data, buffer[:n])

			select {
			case dataChan <- data:
				logger.Debug("Sent chunk of %d bytes (total: %d)", n, totalBytes)
			case <-done:
				logger.Info("File processing interrupted during send")
				return nil
			}
		}
	}
}

func runSystem(mode, filename string, beginningProc *processor.BeginningProcessor,
	inverseProc *processor.InverseProcessor, logger *utils.Logger, config *processor.Config) error {

	logger.Info("Starting system in %s mode", mode)
	defer logger.Info("System stopped")

	stopMutex.Lock()
	stopFlag = false
	stopMutex.Unlock()

	dataChan := make(chan []byte, config.Processing.ChannelBuffer)
	done := make(chan struct{})
	var once sync.Once

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		logger.Info("Received signal: %v", sig)
		once.Do(func() {
			if err := os.WriteFile("data/stop.flag", []byte("stop"), 0644); err != nil {
				logger.Error("Failed to create stop flag: %v", err)
			}
			close(done)
		})
	}()

	var wg sync.WaitGroup
	wg.Add(2)

	startProcessor := func(name string, p interface{ Process(<-chan []byte, <-chan struct{}) }) {
		defer wg.Done()
		logger.Info("Starting %s processor", name)
		defer logger.Info("%s processor stopped", name)

		defer func() {
			if r := recover(); r != nil {
				logger.Error("%s processor panic: %v", name, r)
			}
		}()

		p.Process(dataChan, done)
	}

	go startProcessor("Beginning", beginningProc)
	go startProcessor("Inverse", inverseProc)

	var processErr error
	switch mode {
	case "f":
		logger.Info("Processing file: %s", filename)
		processErr = processFile(filename, dataChan, done, logger, config.GetChunkSize())
	case "r":
		logger.Info("Processing random stream")
		processErr = processStream(dataChan, done, logger, config.GetChunkSize())
	default:
		return fmt.Errorf("unknown mode: %s", mode)
	}

	wg.Wait()
	return processErr
}

func processStream(dataChan chan<- []byte, done <-chan struct{}, 
	logger *utils.Logger, chunkSize int) error {
	defer close(dataChan)
	logger.Info("Starting stream processing")

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			logger.Info("Stream processing interrupted")
			return nil
		case <-ticker.C:
			data := make([]byte, chunkSize)
			for i := range data {
				data[i] = byte(i % 256)
			}

			select {
			case dataChan <- data:
				logger.Debug("Sent generated chunk of %d bytes", len(data))
			case <-done:
				logger.Info("Stream processing interrupted during send")
				return nil
			}
		}
	}
}

// ... остальные функции (parseArgs, printUsage, startVisualization, stopVisualization) без изменений ...