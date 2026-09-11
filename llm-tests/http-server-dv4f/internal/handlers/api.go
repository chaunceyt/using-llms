package handlers

import (
	"log/slog"
	"net/http"
)

// Greeting is the response shape for the greeting endpoint.
type Greeting struct {
	Message string `json:"message"`
	Name    string `json:"name"`
}

// NewAPI returns the versioned API handler group. The logger is injected so
// handlers can emit structured, request-scoped logs.
func NewAPI(logger *slog.Logger) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /hello", func(w http.ResponseWriter, r *http.Request) {
		name := r.URL.Query().Get("name")
		if name == "" {
			name = "world"
		}

		logger.Info("greeting requested",
			"method", r.Method,
			"path", r.URL.Path,
			"name", name,
		)

		writeJSON(w, http.StatusOK, Greeting{
			Message: "Hello, " + name + "!",
			Name:    name,
		})
	})

	return mux
}
