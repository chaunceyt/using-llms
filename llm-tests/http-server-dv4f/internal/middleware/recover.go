package middleware

import (
	"log/slog"
	"net/http"
)

// Recover catches panics from downstream handlers, logs them, and returns a
// 500 to the client instead of crashing the process or leaking a connection.
func Recover(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					logger.Error("panic recovered",
						"panic", rec,
						"path", r.URL.Path,
						// Recover wraps WithRequestID, so the context here lacks the
						// ID; fall back to the response header which is already set.
						"request_id", requestIDOrHeader(r.Context(), w),
					)
					http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}
