package handler

import (
	"net/http"

	"example.com/httpserver/internal/version"
)

// HealthHandler handles liveness, readiness, and version endpoints.
type HealthHandler struct {
	deps Deps
}

// NewHealthHandler returns a HealthHandler backed by the given dependencies.
func NewHealthHandler(deps Deps) *HealthHandler {
	return &HealthHandler{deps: deps}
}

// Liveness handles GET /healthz. It reports that the process is up.
func (h *HealthHandler) Liveness(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// Readiness handles GET /readyz. In a real service this would verify that
// dependencies (database, cache, etc.) are available.
func (h *HealthHandler) Readiness(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// Version handles GET /version.
func (h *HealthHandler) Version(w http.ResponseWriter, r *http.Request) {
	env := "unknown"
	if h.deps.Config != nil {
		env = h.deps.Config.Environment
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"version":     version.Version,
		"environment": env,
	})
}
