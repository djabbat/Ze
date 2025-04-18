package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"
	"io"
	"net/http"
	"errors"

	"ze/internal/processor"
	"ze/internal/utils"
)

var (
	stopFlag   = false
	stopMutex  sync.Mutex
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
	case "radio":
		if err := runRadioSystem(beginningProc, inverseProc, logger, config); err != nil {
			logger.Error("Radio system error: %v", err)
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

func runRadioSystem(beginningProc *processor.BeginningProcessor, inverseProc *processor.InverseProcessor, 
	logger *utils.Logger, config *processor.Config) error {
	logger.Info("Starting radio streaming mode")
	defer logger.Info("Radio system stopped")

	stopMutex.Lock()
	stopFlag = false
	stopMutex.Unlock()

	dataChan := make(chan []byte, 10)
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

	processErr := processRadioStream(dataChan, done, logger, config.GetChunkSize())

	if processErr != nil {
		logger.Error("Radio processing error: %v", processErr)
	}

	wg.Wait()
	return nil
}

func processRadioStream(dataChan chan<- []byte, done <-chan struct{}, 
	logger *utils.Logger, chunkSize int) error {
	defer close(dataChan)
	logger.Info("Starting radio stream processing")

	config, err := processor.LoadConfig("config.yaml")
	if err != nil {
		return fmt.Errorf("error loading config: %w", err)
	}

	if len(config.Radio.Stations) == 0 {
		return errors.New("no radio stations configured")
	}

	currentStation := 0
	var client *http.Client
	var resp *http.Response
	var stream io.ReadCloser

	buffer := make([]byte, chunkSize)
	reconnectTimeout := config.Radio.ReconnectTimeout

	for {
		stopMutex.Lock()
		if stopFlag {
			stopMutex.Unlock()
			logger.Info("Radio processing interrupted by stop flag")
			return nil
		}
		stopMutex.Unlock()

		select {
		case <-done:
			if stream != nil {
				stream.Close()
			}
			if resp != nil {
				resp.Body.Close()
			}
			logger.Info("Radio processing interrupted")
			return nil
		default:
			if client == nil || resp == nil || stream == nil {
				station := config.Radio.Stations[currentStation]
				logger.Info("Connecting to radio station: %s", station.URL)

				client = &http.Client{Timeout: reconnectTimeout}
				var err error
				resp, err = client.Get(station.URL)
				if err != nil {
					logger.Error("Failed to connect to %s: %v", station.URL, err)
					currentStation = (currentStation + 1) % len(config.Radio.Stations)
					time.Sleep(reconnectTimeout)
					continue
				}

				if resp.StatusCode != http.StatusOK {
					logger.Error("Bad status from %s: %s", station.URL, resp.Status)
					resp.Body.Close()
					currentStation = (currentStation + 1) % len(config.Radio.Stations)
					time.Sleep(reconnectTimeout)
					continue
				}

				stream = resp.Body
				logger.Info("Successfully connected to %s", station.URL)
			}

			n, err := stream.Read(buffer)
			if err != nil {
				logger.Error("Error reading from stream %s: %v", config.Radio.Stations[currentStation].URL, err)
				stream.Close()
				resp.Body.Close()
				client = nil
				resp = nil
				stream = nil
				currentStation = (currentStation + 1) % len(config.Radio.Stations)
				time.Sleep(reconnectTimeout)
				continue
			}

			if n == 0 {
				continue
			}

			dataCopy := make([]byte, n)
			copy(dataCopy, buffer[:n])

			select {
			case dataChan <- dataCopy:
				logger.Debug("Sent radio chunk of %d bytes", n)
			case <-done:
				stream.Close()
				resp.Body.Close()
				logger.Info("Radio processing interrupted during send")
				return nil
			}
		}
	}
}

func startVisualization(logger *utils.Logger) error {
	cmd := exec.Command("python3", "visualisation/simple_visualization.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	visualizationCmd = cmd
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start visualization: %v", err)
	}
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

func runSystem(mode, filename string, beginningProc *processor.BeginningProcessor,
	inverseProc *processor.InverseProcessor, logger *utils.Logger, config *processor.Config) error {

	logger.Info("Starting system in %s mode", mode)
	defer logger.Info("System stopped")

	stopMutex.Lock()
	stopFlag = false
	stopMutex.Unlock()

	dataChan := make(chan []byte, 10)
	done := make(chan struct{})
	var once sync.Once

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		logger.Info("Received signal: %v", sig)
		once.Do(func() {
			// Создаем stop.flag перед закрытием канала
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

	if processErr != nil {
		logger.Error("Processing error: %v", processErr)
	}

	wg.Wait()
	return nil
}

func processFile(filename string, dataChan chan<- []byte, done <-chan struct{}, logger *utils.Logger, chunkSize int) error {
	defer close(dataChan)
	logger.Info("Starting file processing: %s", filename)

	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("error opening file: %w", err)
	}
	defer file.Close()

	totalBytes := 0
	buffer := make([]byte, chunkSize)
	for {
		stopMutex.Lock()
		if stopFlag {
			stopMutex.Unlock()
			logger.Info("File processing interrupted by stop flag")
			return nil
		}
		stopMutex.Unlock()

		select {
		case <-done:
			logger.Info("File processing interrupted")
			return nil
		default:
			n, err := file.Read(buffer)
			if err != nil {
				if err == io.EOF {
					logger.Info("File processing completed (EOF reached), total bytes: %d", totalBytes)
					return nil
				}
				return fmt.Errorf("error reading file: %w", err)
			}

			if n == 0 {
				continue
			}

			totalBytes += n
			logger.Debug("Read %d bytes (total: %d)", n, totalBytes)

			dataCopy := make([]byte, n)
			copy(dataCopy, buffer[:n])

			select {
			case dataChan <- dataCopy:
				logger.Debug("Sent chunk of %d bytes", n)
			case <-done:
				logger.Info("File processing interrupted during send")
				return nil
			}
		}
	}
}

func processStream(dataChan chan<- []byte, done <-chan struct{}, 
	logger *utils.Logger, chunkSize int) error {
	defer close(dataChan)
	logger.Info("Starting stream processing")

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		stopMutex.Lock()
		if stopFlag {
			stopMutex.Unlock()
			logger.Info("Stream processing interrupted by stop flag")
			return nil
		}
		stopMutex.Unlock()

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

func parseArgs() (string, string, error) {
	flag.Parse()
	args := flag.Args()

	if len(args) < 1 {
		return "", "", fmt.Errorf("missing mode argument (use 'f', 'r' or 'radio')")
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
	case "radio":
		return mode, "", nil
	default:
		return "", "", fmt.Errorf("invalid mode: %s (use 'f', 'r' or 'radio')", mode)
	}
}

func printUsage() {
	fmt.Println("Usage:")
	fmt.Println("  File mode:   ./ze f <filename>")
	fmt.Println("  Stream mode: ./ze r")
	fmt.Println("  Radio mode:  ./ze radio")
	fmt.Println("\nOptions:")
	flag.PrintDefaults()
}