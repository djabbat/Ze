package utils

import (
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

const maxLogSize = 50 * 1024 * 1024 // 50 MB

type Logger struct {
	fileLogger  *log.Logger
	consoleLog  *log.Logger
	file        *os.File
	mu          sync.Mutex
	debug       bool
	currentSize int64
	logFilePath string
}

func NewLogger(filename string) *Logger {
	file, size, err := openLogFile(filename)
	if err != nil {
		log.Fatalf("Failed to open log file: %v", err)
	}

	return &Logger{
		fileLogger:  log.New(file, "", log.LstdFlags),
		consoleLog:  log.New(os.Stdout, "", log.LstdFlags),
		file:        file,
		debug:       false,
		currentSize: size,
		logFilePath: filename,
	}
}

func openLogFile(filename string) (*os.File, int64, error) {
	// Проверяем существование файла
	_, err := os.Stat(filename)
	if os.IsNotExist(err) {
		file, err := os.Create(filename)
		return file, 0, err
	}
	if err != nil {
		return nil, 0, err
	}

	// Открываем для проверки размера
	file, err := os.OpenFile(filename, os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return nil, 0, err
	}

	// Получаем текущий размер
	fileInfo, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, 0, err
	}

	// Если превышен размер, создаем новый
	if fileInfo.Size() >= maxLogSize {
		file.Close()
		err = os.Remove(filename)
		if err != nil {
			return nil, 0, err
		}
		file, err = os.Create(filename)
		return file, 0, err
	}

	return file, fileInfo.Size(), nil
}

func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.file.Close()
}

func (l *Logger) log(level, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logEntry := fmt.Sprintf("[%s] %s: %s\n", timestamp, level, msg)

	l.mu.Lock()
	defer l.mu.Unlock()

	// Вывод в консоль
	l.consoleLog.Print(logEntry)

	// Проверка размера и ротация
	if l.currentSize+int64(len(logEntry)) >= maxLogSize {
		l.file.Close()
		file, err := os.Create(l.logFilePath)
		if err != nil {
			l.consoleLog.Printf("Failed to rotate log file: %v", err)
			return
		}
		l.file = file
		l.fileLogger = log.New(file, "", log.LstdFlags)
		l.currentSize = 0
	}

	// Запись в файл
	_, err := l.file.WriteString(logEntry)
	if err != nil {
		l.consoleLog.Printf("Failed to write to log file: %v", err)
	} else {
		l.currentSize += int64(len(logEntry))
	}
}

func (l *Logger) SetDebug(debug bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.debug = debug
}

func (l *Logger) IsDebug() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.debug
}

func (l *Logger) Info(format string, args ...interface{}) {
	l.log("INFO", format, args...)
}

func (l *Logger) Error(format string, args ...interface{}) {
	l.log("ERROR", format, args...)
}

func (l *Logger) Debug(format string, args ...interface{}) {
	if l.IsDebug() {
		l.log("DEBUG", format, args...)
	}
}

func (l *Logger) Warn(format string, args ...interface{}) {
	l.log("WARN", format, args...)
}