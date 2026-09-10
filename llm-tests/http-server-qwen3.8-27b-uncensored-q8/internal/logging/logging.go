// Package logging configures the application's structured logger.
package logging

import (
	"log/slog"
	"os"
)

// New builds a structured logger that writes to stderr. JSON output is used
// in production for machine parsing; a human-readable text format is used
// elsewhere.
func New(level slog.Level, environment string) *slog.Logger {
	opts := &slog.HandlerOptions{
		Level:     level,
		AddSource: false,
	}
	if environment == "production" {
		return slog.New(slog.NewJSONHandler(os.Stderr, opts))
	}
	return slog.New(slog.NewTextHandler(os.Stderr, opts))
}
