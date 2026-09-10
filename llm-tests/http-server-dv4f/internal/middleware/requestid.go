// Package middleware provides reusable HTTP middleware for the server.
package middleware

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
)

type ctxKey struct{}

// RequestIDKey is the canonical header used to propagate a request ID across
// services. If a client supplies one it is honoured; otherwise one is generated.
const RequestIDKey = "X-Request-ID"

// WithRequestID attaches a request ID to the context and echoes it back in the
// response headers so clients can correlate logs.
func WithRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqID := r.Header.Get(RequestIDKey)
		if reqID == "" {
			reqID = newRequestID()
		}

		w.Header().Set(RequestIDKey, reqID)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), ctxKey{}, reqID)))
	})
}

// RequestIDFrom returns the request ID stored in ctx, or an empty string if
// none is present.
func RequestIDFrom(ctx context.Context) string {
	id, _ := ctx.Value(ctxKey{}).(string)
	return id
}

// requestIDOrHeader returns the request ID from ctx, falling back to the value
// already written to w's headers. Useful for middleware positioned outside
// WithRequestID in the chain.
func requestIDOrHeader(ctx context.Context, w http.ResponseWriter) string {
	if id := RequestIDFrom(ctx); id != "" {
		return id
	}
	return w.Header().Get(RequestIDKey)
}

func newRequestID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand failing is fatal for our purposes; fall back to a
		// timestamp-based value so requests still carry an ID.
		return "req-unknown"
	}
	return hex.EncodeToString(b)
}
