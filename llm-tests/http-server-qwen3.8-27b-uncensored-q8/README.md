# http-server

A well-structured HTTP server written in idiomatic Go. It demonstrates a clean
layered architecture, dependency injection, structured logging, middleware,
graceful shutdown, and a small but complete test suite — using only the Go
standard library.

## Requirements

- Go 1.22 or newer (uses the method-aware `http.ServeMux` routing patterns
  introduced in Go 1.22).

## Running

```sh
make run                 # go run ./cmd/server
# or
make build && ./bin/server
```

The server listens on `:8080` by default.

To override the address or log level via flags (which take precedence over
environment variables):

```sh
go run ./cmd/server -addr :9090 -log-level debug
```

## Configuration

Configuration is loaded from environment variables; command-line flags override
them. Unset variables fall back to sensible defaults.

| Variable             | Flag        | Default    | Description                                    |
|----------------------|-------------|------------|------------------------------------------------|
| `APP_ENV`            | —           | `development` | Deployment environment (affects log format) |
| `APP_ADDR`           | `-addr`     | `:8080`    | Listen address                                 |
| `APP_READ_TIMEOUT`   | —           | `15s`      | Request read timeout                           |
| `APP_WRITE_TIMEOUT`  | —           | `15s`      | Response write timeout                         |
| `APP_IDLE_TIMEOUT`   | —           | `60s`      | Idle keep-alive timeout                        |
| `APP_SHUTDOWN_TIMEOUT` | —         | `10s`      | Graceful shutdown deadline                     |
| `APP_LOG_LEVEL`      | `-log-level`| `info`     | `debug`, `info`, `warn`, or `error`            |

Durations accept any `time.ParseDuration` value (e.g. `30s`, `1m`). In
`production` the logger emits JSON; otherwise it emits a readable text format.

## HTTP API

| Method   | Path          | Description                       | Success |
|----------|---------------|-----------------------------------|---------|
| `GET`    | `/`           | Index listing available endpoints | `200`   |
| `GET`    | `/healthz`    | Liveness probe                    | `200`   |
| `GET`    | `/readyz`     | Readiness probe                   | `200`   |
| `GET`    | `/version`    | Build version and environment     | `200`   |
| `GET`    | `/items`      | List all items                    | `200`   |
| `POST`   | `/items`      | Create an item (`{"name": "..."}`)| `201`   |
| `GET`    | `/items/{id}` | Get an item by ID                 | `200`   |
| `DELETE` | `/items/{id}` | Delete an item by ID              | `204`   |

Every response includes an `X-Request-Id` header (generated if not supplied on
the request). Errors return a JSON body of the form `{"error": "..."}`.

### Examples

```sh
curl -s localhost:8080/healthz
# {"status":"ok"}

curl -s -X POST localhost:8080/items -d '{"name":"widget"}'
# {"id":"1","name":"widget","created_at":"..."}

curl -s localhost:8080/items/1
# {"id":"1","name":"widget","created_at":"..."}

curl -s -X DELETE localhost:8080/items/1 -o /dev/null -w '%{http_code}\n'
# 204
```

## Project structure

```
.
├── cmd/
│   └── server/
│       └── main.go            # Composition root: config, wiring, lifecycle
├── internal/
│   ├── config/                # Configuration loading (env + flags)
│   ├── logging/               # Structured logger setup (slog)
│   ├── models/                # Domain entities
│   ├── repository/            # Persistence (interface + in-memory impl)
│   ├── service/               # Business logic
│   ├── handler/               # HTTP handlers and responses
│   ├── middleware/            # Cross-cutting handlers (request ID, logging, recovery)
│   ├── server/                # Router + HTTP server lifecycle
│   └── version/               # Build version (settable via -ldflags)
├── Makefile
├── go.mod
└── README.md
```

### Architecture

The application follows a request flow of **Handler → Service → Repository**:

- **Handler** (`internal/handler`) — translates HTTP to the service layer,
  decodes/encodes JSON, and maps service errors to HTTP status codes.
- **Service** (`internal/service`) — holds business logic and validation;
  returns sentinel errors.
- **Repository** (`internal/repository`) — an `ItemRepository` interface with a
  concurrency-safe in-memory implementation. Swap in a database-backed
  implementation without touching the layers above.

`cmd/server/main.go` is the single **composition root** that wires these
together, keeping lower layers free of construction logic.

### Notable practices

- **Standard library only** — no external runtime dependencies.
- **Method-aware routing** via the Go 1.22 `http.ServeMux` patterns
  (e.g. `GET /items/{id}`), with `405`/`404` handled automatically.
- **Graceful shutdown** on `SIGINT`/`SIGTERM` with a bounded drain period, plus
  read/write/idle timeouts on the `http.Server`.
- **Structured logging** with `log/slog`; request IDs propagate through
  `context` for correlation.
- **Middleware chain** (request ID → logging → recovery) with panics converted
  to `500` responses.
- **Dependency injection** so each layer is independently testable.

## Development

```sh
make test        # go test ./...
make test-race   # go test -race ./...
make vet         # go vet ./...
make fmt         # gofmt -w .
make tidy        # go mod tidy
```
