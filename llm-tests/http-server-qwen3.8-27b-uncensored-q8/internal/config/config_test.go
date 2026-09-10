package config

import (
	"testing"
	"time"
)

// clearOverrides unsets every environment variable the config reads so tests
// start from a known-clean state.
func clearOverrides(t *testing.T) {
	t.Helper()
	for _, key := range []string{
		"APP_ENV",
		"APP_ADDR",
		"APP_READ_TIMEOUT",
		"APP_WRITE_TIMEOUT",
		"APP_IDLE_TIMEOUT",
		"APP_SHUTDOWN_TIMEOUT",
		"APP_LOG_LEVEL",
	} {
		t.Setenv(key, "")
	}
}

func TestFromEnvDefaults(t *testing.T) {
	clearOverrides(t)

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv() error: %v", err)
	}
	want := Default()
	if cfg != want {
		t.Errorf("FromEnv() = %+v, want default %+v", cfg, want)
	}
}

func TestFromEnvOverrides(t *testing.T) {
	clearOverrides(t)
	t.Setenv("APP_ENV", "production")
	t.Setenv("APP_ADDR", ":9090")
	t.Setenv("APP_READ_TIMEOUT", "30s")
	t.Setenv("APP_LOG_LEVEL", "debug")

	cfg, err := FromEnv()
	if err != nil {
		t.Fatalf("FromEnv() error: %v", err)
	}
	if cfg.Environment != "production" {
		t.Errorf("Environment = %q, want production", cfg.Environment)
	}
	if cfg.Addr != ":9090" {
		t.Errorf("Addr = %q, want :9090", cfg.Addr)
	}
	if cfg.ReadTimeout != 30*time.Second {
		t.Errorf("ReadTimeout = %v, want 30s", cfg.ReadTimeout)
	}
	if cfg.LogLevel.String() != "DEBUG" {
		t.Errorf("LogLevel = %q, want DEBUG", cfg.LogLevel.String())
	}
	// Untouched values keep their defaults.
	if cfg.WriteTimeout != Default().WriteTimeout {
		t.Errorf("WriteTimeout = %v, want default %v", cfg.WriteTimeout, Default().WriteTimeout)
	}
}

func TestFromEnvInvalidDuration(t *testing.T) {
	clearOverrides(t)
	t.Setenv("APP_READ_TIMEOUT", "not-a-duration")

	if _, err := FromEnv(); err == nil {
		t.Error("expected error for invalid duration, got nil")
	}
}

func TestFromEnvInvalidLevel(t *testing.T) {
	clearOverrides(t)
	t.Setenv("APP_LOG_LEVEL", "verbose")

	if _, err := FromEnv(); err == nil {
		t.Error("expected error for invalid level, got nil")
	}
}

func TestApplyFlags(t *testing.T) {
	cfg := Default()
	cfg.ApplyFlags(FlagOverrides{Addr: ":1234", LogLevel: "error"})

	if cfg.Addr != ":1234" {
		t.Errorf("Addr = %q, want :1234", cfg.Addr)
	}
	if cfg.LogLevel.String() != "ERROR" {
		t.Errorf("LogLevel = %q, want ERROR", cfg.LogLevel.String())
	}
}

func TestApplyFlagsIgnoresEmpty(t *testing.T) {
	cfg := Default()
	cfg.ApplyFlags(FlagOverrides{})

	if cfg != Default() {
		t.Errorf("ApplyFlags with empty overrides should not change config; got %+v", cfg)
	}
}
