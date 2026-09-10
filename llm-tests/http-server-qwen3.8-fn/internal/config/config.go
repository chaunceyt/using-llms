// Package config loads application configuration from the environment.
package config

import (
	"fmt"
	"time"
)

type Config struct {
	Addr              string
	ReadTimeout       time.Duration
	WriteTimeout      time.Duration
	IdleTimeout       time.Duration
	ReadHeaderTimeout time.Duration
	ShutdownTimeout   time.Duration
	LogLevel          string
}

func (c Config) Validate() error {
	if c.Addr == "" {
		return fmt.Errorf("config: addr must not be empty")
	}
	if c.ShutdownTimeout <= 0 {
		return fmt.Errorf("config: shutdown timeout must be positive")
	}
	return nil
}

func Load() (Config, error) {
	c := Config{
		Addr:              getEnv("ADDR", ":8080"),
		ReadTimeout:       getDuration("READ_TIMEOUT", 10*time.Second),
		WriteTimeout:      getDuration("WRITE_TIMEOUT", 15*time.Second),
		IdleTimeout:       getDuration("IDLE_TIMEOUT", 60*time.Second),
		ReadHeaderTimeout: getDuration("READ_HEADER_TIMEOUT", 5*time.Second),
		ShutdownTimeout:   getDuration("SHUTDOWN_TIMEOUT", 20*time.Second),
		LogLevel:          getEnv("LOG_LEVEL", "info"),
	}
	if err := c.Validate(); err != nil {
		return Config{}, err
	}
	return c, nil
}
