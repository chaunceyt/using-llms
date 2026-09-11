// Package handlers contains the HTTP request handlers for the server.
package handlers

import (
	"net/http"
)

// Health returns a 200 with a JSON status body. It is used by load balancers
// and orchestrators to determine that the process is alive.
func Health() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	}
}

// Ready returns a 200 once the service is able to accept traffic. It is
// intentionally separate from Health so that liveness and readiness probes can
// diverge (e.g. while warming caches or waiting for dependencies).
func Ready() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
	}
}

// writeJSON encodes value as a JSON response with the given status code.
// It is safe to call concurrently and sets a Content-Type header.
func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if err := jsonEncode(w, value); err != nil {
		// Too late to change the status code once headers are sent, so just log.
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
	}
}
