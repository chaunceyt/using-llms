// Package config defines the server's runtime configuration and the logic
// for loading it from the process environment and command-line flags.
package config

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"time"
)

// Config holds the runtime configuration for the HTTP server.
type Config struct {
	// Environment selects the deployment context (e.g. "development",
	// "production"). It influences logging behavior and other environment-
	// specific decisions.
	Environment string
	// Addr is the network address the server listens on, e.g. ":8080".
	Addr string
	// ReadTimeout is the maximum duration for reading the entire request,
	// including the body.
	ReadTimeout time.Duration
	// WriteTimeout is the maximum duration before timing out writes of the
	// response.
	WriteTimeout time.Duration
	// IdleTimeout is the maximum amount of time to wait for the next request
	// when keep-alives are enabled.
	IdleTimeout time.Duration
	// ShutdownTimeout bounds how long graceful shutdown waits for in-flight
	// requests to complete.
	ShutdownTimeout time.Duration
	// LogLevel is the minimum level of structured log output.
	LogLevel slog.Level
}

// FlagOverrides carries values supplied via command-line flags that take
// precedence over environment variables.
type FlagOverrides struct {
	Addr     string
	LogLevel string
}

// Default returns the default configuration values.
func Default() Config {
	return Config{
		Environment:     "development",
		Addr:            ":8080",
		ReadTimeout:     15 * time.Second,
		WriteTimeout:    15 * time.Second,
		IdleTimeout:     60 * time.Second,
		ShutdownTimeout: 10 * time.Second,
		LogLevel:        slog.LevelInfo,
	}
}

// FromEnv builds a Config from environment variables. Unset variables keep
// their defaults; present-but-invalid variables are reported as errors.
func FromEnv() (Config, error) {
	cfg := Default()
	var errs []error

	cfg.Environment = getString("APP_ENV", cfg.Environment)
	cfg.Addr = getString("APP_ADDR", cfg.Addr)

	cfg.ReadTimeout = getDuration("APP_READ_TIMEOUT", cfg.ReadTimeout, &errs)
	cfg.WriteTimeout = getDuration("APP_WRITE_TIMEOUT", cfg.WriteTimeout, &errs)
	cfg.IdleTimeout = getDuration("APP_IDLE_TIMEOUT", cfg.IdleTimeout, &errs)
	cfg.ShutdownTimeout = getDuration("APP_SHUTDOWN_TIMEOUT", cfg.ShutdownTimeout, &errs)

	cfg.LogLevel = getLevel("APP_LOG_LEVEL", cfg.LogLevel, &errs)

	return cfg, errors.Join(errs...)
}

// ApplyFlags applies command-line flag overrides on top of the current values.
// Empty flag values are ignored so that env/config defaults are preserved.
func (c *Config) ApplyFlags(o FlagOverrides) {
	if o.Addr != "" {
		c.Addr = o.Addr
	}
	if o.LogLevel != "" {
		var lv slog.Level
		if err := lv.UnmarshalText([]byte(o.LogLevel)); err == nil {
			c.LogLevel = lv
		}
	}
}

func getString(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getDuration(key string, def time.Duration, errs *[]error) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		*errs = append(*errs, fmt.Errorf("%s: invalid duration %q: %w", key, v, err))
		return def
	}
	return d
}

func getLevel(key string, def slog.Level, errs *[]error) slog.Level {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	var lv slog.Level
	if err := lv.UnmarshalText([]byte(v)); err != nil {
		*errs = append(*errs, fmt.Errorf("%s: invalid level %q: %w", key, v, err))
		return def
	}
	return lv
}
