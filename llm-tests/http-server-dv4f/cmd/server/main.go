// Command server runs the http-server-dv4f HTTP service.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"http-server-dv4f/internal/config"
	"http-server-dv4f/internal/server"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		slog.Error("failed to load configuration", "error", err)
		os.Exit(1)
	}

	logger := newLogger(cfg.LogLevel)
	slog.SetDefault(logger)

	srv := server.New(cfg, logger)

	// Run the server in the background and listen for a shutdown signal.
	errCh := make(chan error, 1)
	go func() {
		logger.Info("server listening", "addr", cfg.Addr)
		errCh <- srv.ListenAndServe()
	}()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	select {
	case err := <-errCh:
		// ListenAndServe only returns a non-nil error, so a nil here means the
		// channel was never written to, which cannot happen.
		if !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	case <-ctx.Done():
		logger.Info("shutdown signal received")
		// Use a fresh background context: the signal context is already
		// canceled here, which would short-circuit Shutdown's own deadline.
		if err := server.Shutdown(context.Background(), srv, cfg.ShutdownTimeout); err != nil {
			logger.Error("graceful shutdown failed", "error", err)
			os.Exit(1)
		}
		logger.Info("server stopped cleanly")
	}
}

// newLogger builds a structured logger at the configured level writing to
// stdout with human-friendly text formatting by default.
func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}

	opts := &slog.HandlerOptions{Level: lvl}
	return slog.New(slog.NewTextHandler(os.Stdout, opts))
}
