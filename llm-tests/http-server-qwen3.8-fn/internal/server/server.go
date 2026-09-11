// Package server wires HTTP routes, middleware, and lifecycle management.
package server

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os/signal"
	"syscall"
	"time"

	"github.com/example/http-server-qwen3.8-fn/internal/config"
	"github.com/example/http-server-qwen3.8-fn/internal/handler"
	"github.com/example/http-server-qwen3.8-fn/internal/user"
)

type Server struct {
	httpServer *http.Server
	logger     *slog.Logger
	shutdown   func(ctx context.Context) error
}

func New(cfg config.Config, store user.Store, logger *slog.Logger) *Server {
	mux := newMux(store, logger)

	chain := Chain(mux,
		requestID,
		recoverer(logger),
		logging(logger),
		timeout(cfg.WriteTimeout),
	)

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           chain,
		ReadTimeout:       cfg.ReadTimeout,
		WriteTimeout:      cfg.WriteTimeout,
		IdleTimeout:       cfg.IdleTimeout,
		ReadHeaderTimeout: cfg.ReadHeaderTimeout,
	}

	return &Server{httpServer: srv, logger: logger, shutdown: srv.Shutdown}
}

func newMux(store user.Store, logger *slog.Logger) *http.ServeMux {
	mux := http.NewServeMux()

	health := handler.NewHealth(logger)
	users := handler.NewUsers(store, logger)

	mux.HandleFunc("GET /healthz", health.Liveness)
	mux.HandleFunc("GET /readyz", health.Readiness)

	mux.HandleFunc("GET /api/v1/users", users.List)
	mux.HandleFunc("POST /api/v1/users", users.Create)
	mux.HandleFunc("GET /api/v1/users/{id}", users.Get)
	mux.HandleFunc("DELETE /api/v1/users/{id}", users.Delete)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"error":"not found"}`))
	})

	return mux
}

// Run starts the server and blocks until ctx is cancelled or SIGINT/SIGTERM
// is received, then shuts down gracefully within shutdownTimeout.
func (s *Server) Run(ctx context.Context, shutdownTimeout time.Duration) error {
	ctx, stop := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		s.logger.Info("server starting", slog.String("addr", s.httpServer.Addr))
		if err := s.httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
			return
		}
		errCh <- nil
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}

	s.logger.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), shutdownTimeout)
	defer cancel()
	if err := s.shutdown(shutdownCtx); err != nil {
		s.logger.Error("graceful shutdown failed", slog.Any("error", err))
		return err
	}
	s.logger.Info("server stopped")
	return nil
}
