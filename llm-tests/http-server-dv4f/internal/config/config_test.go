package config

import (
	"os"
	"testing"
	"time"
)

func TestLoadDefaults(t *testing.T) {
	for _, key := range []string{
		"HTTP_ADDR", "HTTP_READ_HEADER_TIMEOUT", "HTTP_READ_TIMEOUT",
		"HTTP_WRITE_TIMEOUT", "HTTP_IDLE_TIMEOUT", "HTTP_SHUTDOWN_TIMEOUT",
		"HTTP_LOG_LEVEL",
	} {
		os.Unsetenv(key)
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if cfg.Addr != defaultAddr {
		t.Errorf("Addr = %q, want %q", cfg.Addr, defaultAddr)
	}
	if cfg.LogLevel != "info" {
		t.Errorf("LogLevel = %q, want info", cfg.LogLevel)
	}
	if cfg.ShutdownTimeout != 15*time.Second {
		t.Errorf("ShutdownTimeout = %s, want 15s", cfg.ShutdownTimeout)
	}
}

func TestLoadFromEnv(t *testing.T) {
	t.Cleanup(func() {
		os.Unsetenv("HTTP_ADDR")
		os.Unsetenv("HTTP_LOG_LEVEL")
		os.Unsetenv("HTTP_SHUTDOWN_TIMEOUT")
	})

	os.Setenv("HTTP_ADDR", "127.0.0.1:9090")
	os.Setenv("HTTP_LOG_LEVEL", "DEBUG")
	os.Setenv("HTTP_SHUTDOWN_TIMEOUT", "5s")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}
	if cfg.Addr != "127.0.0.1:9090" {
		t.Errorf("Addr = %q, want 127.0.0.1:9090", cfg.Addr)
	}
	if cfg.LogLevel != "debug" {
		t.Errorf("LogLevel = %q, want debug (lowercased)", cfg.LogLevel)
	}
	if cfg.ShutdownTimeout != 5*time.Second {
		t.Errorf("ShutdownTimeout = %s, want 5s", cfg.ShutdownTimeout)
	}
}

func TestLoadRejectsInvalidValues(t *testing.T) {
	t.Cleanup(func() { os.Unsetenv("HTTP_SHUTDOWN_TIMEOUT") })

	os.Setenv("HTTP_SHUTDOWN_TIMEOUT", "not-a-duration")
	if _, err := Load(); err == nil {
		t.Error("Load() expected error for malformed duration, got nil")
	}

	os.Setenv("HTTP_SHUTDOWN_TIMEOUT", "5s")
	os.Setenv("HTTP_LOG_LEVEL", "bogus")
	t.Cleanup(func() { os.Unsetenv("HTTP_LOG_LEVEL") })
	if _, err := Load(); err == nil {
		t.Error("Load() expected error for invalid log level, got nil")
	}
}
