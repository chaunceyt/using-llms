package handler

import (
	"log/slog"
	"net/http"
)

type Health struct {
	logger *slog.Logger
}

func NewHealth(logger *slog.Logger) *Health {
	return &Health{logger: logger}
}

// Liveness reports that the process is running.
func (h *Health) Liveness(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, h.logger, http.StatusOK, map[string]string{"status": "ok"})
}

// Readiness reports that the service can accept traffic. Checks such as
// database connectivity belong here.
func (h *Health) Readiness(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, h.logger, http.StatusOK, map[string]string{"status": "ready"})
}
