package processor

import (
	"math/rand"
	"ze/internal/utils"
	"time"
)

type BeginningProcessor struct {
	logger *utils.Logger
	config *Config
}

func NewBeginningProcessor(logger *utils.Logger, config *Config) *BeginningProcessor {
	return &BeginningProcessor{
		logger: logger,
		config: config,
	}
}

func (p *BeginningProcessor) Process(dataChan <-chan []byte, done <-chan struct{}) {
	p.logger.Info("Starting BeginningProcessor")
	defer p.logger.Info("BeginningProcessor stopped")

	cs, err := NewCounterSystem(
		"data/beginning.bin",
		1<<p.config.Processing.CrumbSize,
		p.config.Processing.CounterValue,
		p.logger,
	)
	if err != nil {
		p.logger.Error("Failed to initialize counter system: %v", err)
		return
	}
	defer func() {
		if err := cs.save(); err != nil {
			p.logger.Error("Failed to save counters: %v", err)
		}
	}()

	var chunkBuffer []byte
	crumbSize := p.config.Processing.CrumbSize
	p.logger.Debug("Crumb size: %d", crumbSize)

	// Таймер для проверки зависаний
	activityTimer := time.NewTimer(5 * time.Second)
	defer activityTimer.Stop()

	for {
		select {
		case <-done:
			p.finalizeProcessing(cs, chunkBuffer, crumbSize)
			return
		case data, ok := <-dataChan:
			if !ok {
				p.finalizeProcessing(cs, chunkBuffer, crumbSize)
				return
			}
			p.logger.Debug("Received chunk: %d bytes", len(data))
			
			// Сброс таймера при получении данных
			if !activityTimer.Stop() {
				<-activityTimer.C
			}
			activityTimer.Reset(5 * time.Second)
			
			chunkBuffer = append(chunkBuffer, data...)
			chunkBuffer = p.processBuffer(cs, chunkBuffer, crumbSize)
			
		case <-activityTimer.C:
			p.logger.Warn("No activity for 5 seconds, stopping processor")
			p.finalizeProcessing(cs, chunkBuffer, crumbSize)
			return
		}
	}
}

func (p *BeginningProcessor) processBuffer(cs *CounterSystem, buffer []byte, crumbSize int) []byte {
	if len(buffer) < crumbSize {
		p.logger.Debug("Buffer too small (%d < %d)", len(buffer), crumbSize)
		return buffer
	}

	for i := 0; i <= len(buffer)-crumbSize; i += crumbSize {
		var crumb [4]byte
		copy(crumb[:], buffer[i:i+crumbSize])
		p.logger.Debug("Processing crumb at offset %d: %v", i, crumb)

		increment := p.getIncrement()
		if err := cs.ProcessCrumb(crumb, increment); err != nil {
			p.logger.Error("Crumb processing error: %v", err)
		}
	}

	return buffer[len(buffer)-(len(buffer)%crumbSize):]
}

func (p *BeginningProcessor) finalizeProcessing(cs *CounterSystem, buffer []byte, crumbSize int) {
	if len(buffer) > 0 {
		p.logger.Debug("Finalizing with %d bytes remaining", len(buffer))
		padded := make([]byte, crumbSize)
		copy(padded, buffer)
		var crumb [4]byte
		copy(crumb[:], padded)
		if err := cs.ProcessCrumb(crumb, p.getIncrement()); err != nil {
			p.logger.Error("Final crumb processing error: %v", err)
		}
	}
}

func (p *BeginningProcessor) getIncrement() uint32 {
	if rand.Float64() < p.config.Processing.Actualization {
		return p.config.Processing.Increment
	}
	return p.config.Processing.PredictIncrement
}