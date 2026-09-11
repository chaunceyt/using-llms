package middleware

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
)

// RequestIDHeader is the HTTP header used to carry a request identifier.
const RequestIDHeader = "X-Request-Id"

type requestIDKey struct{}

// RequestID ensures every request carries an identifier. If the incoming
// request already has an X-Request-Id header it is preserved; otherwise a new
// one is generated. The ID is stored in the request context and echoed back in
// the response.
func RequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get(RequestIDHeader)
		if id == "" {
			id = generateRequestID()
		}
		w.Header().Set(RequestIDHeader, id)
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// RequestIDFromContext returns the request ID stored in ctx by RequestID.
func RequestIDFromContext(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(requestIDKey{}).(string)
	return id, ok
}

func generateRequestID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		// rand.Read only fails on a broken OS; fall back to a non-empty value.
		return "unknown"
	}
	return hex.EncodeToString(b)
}
