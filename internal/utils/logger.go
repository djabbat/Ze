package utils

import (
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

type Logger struct {
	fileLogger *log.Logger
	consoleLog *log.Logger
	file       *os.File
	mu         sync.Mutex
}

func NewLogger(filename string) *Logger {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}

	return &Logger{
		fileLogger: log.New(file, "", log.LstdFlags),
		consoleLog: log.New(os.Stdout, "", log.LstdFlags),
		file:       file,
	}
}

func (l *Logger) Close() error {
	return l.file.Close()
}

func (l *Logger) log(level, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logEntry := fmt.Sprintf("[%s] %s: %s", timestamp, level, msg)

	l.mu.Lock()
	defer l.mu.Unlock()

	l.consoleLog.Println(logEntry)
	l.fileLogger.Println(logEntry)
}

func (l *Logger) Info(format string, args ...interface{}) {
	l.log("INFO", format, args...)
}

func (l *Logger) Error(format string, args ...interface{}) {
	l.log("ERROR", format, args...)
}

func (l *Logger) Debug(format string, args ...interface{}) {
	l.log("DEBUG", format, args...)
}

func (l *Logger) Warn(format string, args ...interface{}) {
	l.log("WARN", format, args...)
}