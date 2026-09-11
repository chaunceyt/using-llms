package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/example/http-server-qwen3.8-fn/internal/config"
	"github.com/example/http-server-qwen3.8-fn/internal/server"
	"github.com/example/http-server-qwen3.8-fn/internal/user"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}

	logger := newLogger(cfg.LogLevel)
	slog.SetDefault(logger)

	store := user.NewMemoryStore()
	srv := server.New(cfg, store, logger)

	return srv.Run(context.Background(), cfg.ShutdownTimeout)
}

func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(level)); err != nil {
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl}))
}
