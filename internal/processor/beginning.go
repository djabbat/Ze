package processor

import (
	"errors"
	"time"
	"ze/internal/utils"
)

type BeginningProcessor struct {
	logger *utils.Logger
	config *Config
}

func NewBeginningProcessor(logger *utils.Logger, config *Config) *BeginningProcessor {
	logger.SetDebug(config.Logging.DebugCounters)
	return &BeginningProcessor{
		logger: logger,
		config: config,
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

		if p.config.Logging.DebugCounters {
			p.logger.Debug("Processing crumb at offset %d: %v", i, crumb)
		}

		if err := cs.ProcessCrumb(crumb); err != nil {
			if errors.Is(err, ErrMaxCounters) {
				p.logger.Warn("Max counters reached, cannot add new counter for %v", crumb)
			} else {
				p.logger.Error("Crumb processing error: %v", err)
			}
		}
	}

	if p.config.Logging.DebugCounters {
		p.logger.Debug("Finished processing buffer, total counters: %d", len(cs.counters))
	}

	if err := cs.IncrementChunkCounter(); err != nil {
		p.logger.Error("Filtration error: %v", err)
	}

	return buffer[len(buffer)-(len(buffer)%crumbSize):]
}

func (p *BeginningProcessor) Process(dataChan <-chan []byte, done <-chan struct{}) {
	p.logger.Info("Starting BeginningProcessor")
	defer p.logger.Info("BeginningProcessor stopped")

	cs, err := NewCounterSystem(
		"data/beginning.bin",
		"data/beginning_matches.bin",
		1<<p.config.Processing.CrumbSize,
		p.config.Processing.CounterValue,
		p.logger,
		p.config,
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
	activityTimer := time.NewTimer(p.config.Processing.ActivityTimeout)
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

			if !activityTimer.Stop() {
				select {
				case <-activityTimer.C:
				default:
				}
			}
			activityTimer.Reset(p.config.Processing.ActivityTimeout)

			chunkBuffer = append(chunkBuffer, data...)
			chunkBuffer = p.processBuffer(cs, chunkBuffer, crumbSize)

		case <-activityTimer.C:
			p.logger.Warn("No activity for %v, stopping processor", p.config.Processing.ActivityTimeout)
			p.finalizeProcessing(cs, chunkBuffer, crumbSize)
			return
		}
	}
}

func (p *BeginningProcessor) finalizeProcessing(cs *CounterSystem, buffer []byte, crumbSize int) {
	if len(buffer) > 0 {
		padded := make([]byte, crumbSize)
		copy(padded, buffer)
		var crumb [4]byte
		copy(crumb[:], padded)
		if err := cs.ProcessCrumb(crumb); err != nil {
			p.logger.Error("Final crumb processing error: %v", err)
		}
	}
}