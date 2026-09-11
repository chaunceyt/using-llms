// Package server builds the HTTP handler tree and manages the server's
// lifecycle, including graceful shutdown.
package server

import (
	"context"
	"log/slog"
	"net/http"

	"example.com/httpserver/internal/config"
)

// Server wraps an *http.Server with lifecycle management.
type Server struct {
	cfg        config.Config
	logger     *slog.Logger
	httpServer *http.Server
}

// New builds a Server around the provided handler using the timeouts and
// address from cfg.
func New(cfg config.Config, h http.Handler, logger *slog.Logger) *Server {
	return &Server{
		cfg:    cfg,
		logger: logger,
		httpServer: &http.Server{
			Addr:         cfg.Addr,
			Handler:      h,
			ReadTimeout:  cfg.ReadTimeout,
			WriteTimeout: cfg.WriteTimeout,
			IdleTimeout:  cfg.IdleTimeout,
		},
	}
}

// Run serves HTTP requests until ctx is canceled, then performs a graceful
// shutdown. It returns the first error encountered while serving or shutting
// down, or nil for a clean stop.
func (s *Server) Run(ctx context.Context) error {
	errCh := make(chan error, 1)
	go func() {
		s.logger.Info("server listening", "address", s.cfg.Addr)
		if err := s.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
	}()

	var runErr error
	select {
	case runErr = <-errCh:
		// The listener failed (e.g. address in use); skip the graceful wait.
	case <-ctx.Done():
		// Shutting down on signal or parent context cancellation.
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), s.cfg.ShutdownTimeout)
	defer cancel()
	if err := s.httpServer.Shutdown(shutdownCtx); err != nil && runErr == nil {
		runErr = err
	}

	s.logger.Info("server stopped")
	return runErr
}
