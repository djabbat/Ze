package processor

import (
	"time"
	"gopkg.in/yaml.v3"
	"os"
)

type Config struct {
	Processing struct {
		CrumbSize        int           `yaml:"crumb_size"`
		ChunkPower       int           `yaml:"chunk_power"`
		CounterValue     uint32        `yaml:"counter_value"`
		PredictIncrement uint32        `yaml:"predict_increment"`
		Actualization    float64       `yaml:"actualization_value"`
		Increment        uint32        `yaml:"increment"`
		InitialCounters  int           `yaml:"initial_counters"`
		InitialID        uint32        `yaml:"initial_id"`
		FiltrationValue  int           `yaml:"filtration_value"`
		FiltrationPeriod int           `yaml:"filtration_period"`
		ChannelBuffer    int           `yaml:"channel_buffer"`
		ActivityTimeout  time.Duration `yaml:"activity_timeout"`
	} `yaml:"processing"`

	Radio struct {
		Stations []struct {
			URL        string `yaml:"url"`
			BufferSize int    `yaml:"buffer_size"`
		} `yaml:"stations"`
		DefaultBuffer     int           `yaml:"default_buffer"`
		ReconnectTimeout  time.Duration `yaml:"reconnect_timeout"`
	} `yaml:"radio"`

	Logging struct {
		Level         string `yaml:"level"`
		DebugCounters bool   `yaml:"debug_counters"`
	} `yaml:"logging"`
}

func (c *Config) GetChunkSize() int {
	return c.Processing.ChunkPower
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
	return &config, nil
}