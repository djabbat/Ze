package processor

import (
	"os"
	"errors"
	"time"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Processing struct {
		CrumbSize        int     `yaml:"crumb_size"`
		ChunkPower       int     `yaml:"chunk_power"`
		CounterValue     uint32  `yaml:"counter_value"`
		PredictIncrement uint32  `yaml:"predict_increment"`
		Actualization    float64 `yaml:"actualization_value"`
		Increment        uint32  `yaml:"increment"`
	} `yaml:"processing"`
	Radio struct {
		Stations []struct {
			URL        string `yaml:"url"`
			BufferSize int    `yaml:"buffer_size"`
		} `yaml:"stations"`
		DefaultBuffer     int           `yaml:"default_buffer"`
		ReconnectTimeout  time.Duration `yaml:"reconnect_timeout"`
	} `yaml:"radio"`
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	if config.Processing.CrumbSize < 2 || config.Processing.CrumbSize > 8 {
		return nil, ErrInvalidCrumbSize
	}
	if config.Processing.Actualization <= 0 || config.Processing.Actualization >= 1 {
		return nil, ErrInvalidActualization
	}

	return &config, nil
}

func (c *Config) GetChunkSize() int {
    return c.Processing.ChunkPower
}

var (
    ErrInvalidCrumbSize    = errors.New("invalid crumb size")
    ErrInvalidActualization = errors.New("invalid actualization mode")
)