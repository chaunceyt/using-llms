// Package config loads and validates server configuration from the
// environment. It favours explicit, validated values with sane defaults.
package config

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// Config holds all runtime configuration for the HTTP server.
type Config struct {
	// Addr is the host:port the server listens on.
	Addr string
	// ReadHeaderTimeout bounds how long the server waits to read request headers.
	ReadHeaderTimeout time.Duration
	// ReadTimeout bounds reading the entire request body.
	ReadTimeout time.Duration
	// WriteTimeout bounds writing the response.
	WriteTimeout time.Duration
	// IdleTimeout bounds how long a keep-alive connection is held idle.
	IdleTimeout time.Duration
	// ShutdownTimeout is how long graceful shutdown waits before forcing exit.
	ShutdownTimeout time.Duration
	// LogLevel controls the verbosity of structured logging.
	LogLevel string
}

const (
	defaultAddr       = ":8080"
	defaultReadHeader = 5 * time.Second
	defaultRead       = 10 * time.Second
	defaultWrite      = 10 * time.Second
	defaultIdle       = 60 * time.Second
	defaultShutdown   = 15 * time.Second
	defaultLogLevel   = "info"
)

// Load reads configuration from the environment, applying defaults for any
// unset values. Invalid or malformed values return an error so that
// misconfiguration fails fast at startup rather than silently at runtime.
func Load() (*Config, error) {
	readHeaderTimeout, err := durationEnvOr("HTTP_READ_HEADER_TIMEOUT", defaultReadHeader)
	if err != nil {
		return nil, err
	}
	readTimeout, err := durationEnvOr("HTTP_READ_TIMEOUT", defaultRead)
	if err != nil {
		return nil, err
	}
	writeTimeout, err := durationEnvOr("HTTP_WRITE_TIMEOUT", defaultWrite)
	if err != nil {
		return nil, err
	}
	idleTimeout, err := durationEnvOr("HTTP_IDLE_TIMEOUT", defaultIdle)
	if err != nil {
		return nil, err
	}
	shutdownTimeout, err := durationEnvOr("HTTP_SHUTDOWN_TIMEOUT", defaultShutdown)
	if err != nil {
		return nil, err
	}

	cfg := &Config{
		Addr:              envOr("HTTP_ADDR", defaultAddr),
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,
		ShutdownTimeout:   shutdownTimeout,
		LogLevel:          strings.ToLower(envOr("HTTP_LOG_LEVEL", defaultLogLevel)),
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// Validate checks that configuration values are usable.
func (c *Config) Validate() error {
	switch c.LogLevel {
	case "debug", "info", "warn", "error":
	default:
		return fmt.Errorf("config: invalid log level %q (want debug|info|warn|error)", c.LogLevel)
	}

	for name, d := range map[string]time.Duration{
		"HTTP_READ_HEADER_TIMEOUT": c.ReadHeaderTimeout,
		"HTTP_READ_TIMEOUT":        c.ReadTimeout,
		"HTTP_WRITE_TIMEOUT":       c.WriteTimeout,
		"HTTP_IDLE_TIMEOUT":        c.IdleTimeout,
		"HTTP_SHUTDOWN_TIMEOUT":    c.ShutdownTimeout,
	} {
		if d <= 0 {
			return fmt.Errorf("config: %s must be positive, got %s", name, d)
		}
	}

	return nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func durationEnvOr(key string, fallback time.Duration) (time.Duration, error) {
	v := os.Getenv(key)
	if v == "" {
		return fallback, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("config: invalid %s %q: %w", key, v, err)
	}
	return d, nil
}
