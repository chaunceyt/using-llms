// Package version exposes build metadata for the HTTP server.
package version

// Version is the build version. It defaults to "dev" and is meant to be
// overridden at build time via:
//
//	go build -ldflags "-X example.com/httpserver/internal/version.Version=v1.2.3"
var Version = "dev"
