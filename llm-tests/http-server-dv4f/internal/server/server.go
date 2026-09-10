// Package server wires the HTTP routing, middleware chain, and lifecycle of
// the net/http server together.
package server

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"http-server-dv4f/internal/config"
	"http-server-dv4f/internal/handlers"
	"http-server-dv4f/internal/middleware"
)

// New builds the fully-configured *http.Server with all routes and middleware
// applied. The logger is used by both the access log and recovery middleware.
func New(cfg *config.Config, logger *slog.Logger) *http.Server {
	mux := http.NewServeMux()

	// Health and readiness probes bypass per-request access-log noise but still
	// flow through request-ID and recovery handling.
	mux.Handle("GET /healthz", handlers.Health())
	mux.Handle("GET /readyz", handlers.Ready())

	// Versioned API group.
	mux.Handle("/api/v1/", http.StripPrefix("/api/v1", handlers.NewAPI(logger)))

	// WithRequestID must wrap AccessLog so the request ID is present in the
	// context by the time the access log runs; Recover stays outermost to catch
	// panics from any middleware or handler.
	handler := middleware.Recover(logger)(
		middleware.WithRequestID(
			middleware.AccessLog(logger)(mux),
		),
	)

	return &http.Server{
		Addr:              cfg.Addr,
		Handler:           handler,
		ReadHeaderTimeout: cfg.ReadHeaderTimeout,
		ReadTimeout:       cfg.ReadTimeout,
		WriteTimeout:      cfg.WriteTimeout,
		IdleTimeout:       cfg.IdleTimeout,
	}
}

// Shutdown gracefully stops the server, waiting up to shutdownTimeout for
// in-flight requests to complete. If that deadline passes it force-closes open
// connections and returns the forced-close error.
func Shutdown(ctx context.Context, srv *http.Server, shutdownTimeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, shutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return srv.Close()
		}
		return err
	}
	return nil
}
