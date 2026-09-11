// Package handler contains the HTTP transport layer.
package handler

import (
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"

	"github.com/example/http-server-qwen3.8-fn/internal/user"
)

type errorResponse struct {
	Error string `json:"error"`
}

func writeJSON(w http.ResponseWriter, logger *slog.Logger, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if v == nil {
		return
	}
	if err := json.NewEncoder(w).Encode(v); err != nil {
		logger.Error("failed to encode response", slog.Any("error", err))
	}
}

func writeError(w http.ResponseWriter, logger *slog.Logger, status int, msg string) {
	writeJSON(w, logger, status, errorResponse{Error: msg})
}

func writeInternalError(w http.ResponseWriter, r *http.Request, logger *slog.Logger, err error) {
	logger.Error("internal error",
		slog.String("path", r.URL.Path),
		slog.Any("error", err),
	)
	writeError(w, logger, http.StatusInternalServerError, "internal server error")
}

// mapStoreError translates domain errors into HTTP status codes.
func mapStoreError(w http.ResponseWriter, r *http.Request, logger *slog.Logger, err error) {
	switch {
	case errors.Is(err, user.ErrNotFound):
		writeError(w, logger, http.StatusNotFound, "resource not found")
	default:
		writeInternalError(w, r, logger, err)
	}
}
