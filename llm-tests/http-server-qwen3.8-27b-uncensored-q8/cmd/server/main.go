// Command server is the entry point for the HTTP server. It acts as the
// composition root: it loads configuration, wires the application layers
// together, and runs the server with graceful shutdown.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"example.com/httpserver/internal/config"
	"example.com/httpserver/internal/handler"
	"example.com/httpserver/internal/logging"
	"example.com/httpserver/internal/repository"
	"example.com/httpserver/internal/server"
	"example.com/httpserver/internal/service"
	"example.com/httpserver/internal/version"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.FromEnv()
	if err != nil {
		return fmt.Errorf("load config: %w", err)
	}
	cfg.ApplyFlags(parseFlags())

	logger := logging.New(cfg.LogLevel, cfg.Environment)
	slog.SetDefault(logger)

	logger.Info("starting server",
		"version", version.Version,
		"environment", cfg.Environment,
	)

	// Composition root: build the dependency graph bottom-up.
	repo := repository.NewMemoryItemRepository()
	itemsSvc := service.NewItemService(repo)
	deps := handler.NewDeps(itemsSvc, &cfg)

	router := server.NewRouter(deps, logger)
	srv := server.New(cfg, router, logger)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	return srv.Run(ctx)
}

// parseFlags returns the values of command-line flags that override
// environment configuration.
func parseFlags() config.FlagOverrides {
	var f config.FlagOverrides
	flag.StringVar(&f.Addr, "addr", "", "listen address (overrides APP_ADDR)")
	flag.StringVar(&f.LogLevel, "log-level", "", "log level: debug, info, warn, error (overrides APP_LOG_LEVEL)")
	flag.Parse()
	return f
}
