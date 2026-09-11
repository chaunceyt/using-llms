package handler

import (
	"encoding/json"
	"errors"
	"net/http"

	"example.com/httpserver/internal/service"
)

// writeJSON writes v to the response as JSON with the given status code. A nil
// v writes only the headers (e.g. for 204 No Content).
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if v != nil {
		// Encoding to an already-headed ResponseWriter should not fail for the
		// small, serializable values used here.
		_ = json.NewEncoder(w).Encode(v)
	}
}

// writeError writes a consistent JSON error body.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// writeServiceError maps a service-layer error to an appropriate HTTP status
// and message.
func writeServiceError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, service.ErrNotFound):
		writeError(w, http.StatusNotFound, "resource not found")
	case errors.Is(err, service.ErrInvalidName):
		writeError(w, http.StatusUnprocessableEntity, "invalid item name")
	default:
		writeError(w, http.StatusInternalServerError, "internal server error")
	}
}
