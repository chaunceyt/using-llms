# http-server-qwen3.8-fn

A well-architected HTTP server in Go using only the standard library (Go 1.22+ pattern routing, `log/slog`).

## Layout

```
cmd/server/            Thin entrypoint: wire dependencies, run, exit code handling
internal/config/       Env-based configuration with defaults and validation
internal/server/       http.Server construction, middleware chain, graceful shutdown
internal/handler/      HTTP transport layer: routes' handlers, JSON helpers
internal/user/         Domain model + Store interface + in-memory implementation
```

Dependencies point inward: `handler` depends on the `user.Store` interface, never a concrete store.

## Run

```sh
make run          # or: go run ./cmd/server
make test
```

## API

| Method | Path                | Description       |
|--------|---------------------|-------------------|
| GET    | /healthz            | Liveness probe    |
| GET    | /readyz             | Readiness probe   |
| GET    | /api/v1/users       | List users        |
| POST   | /api/v1/users       | Create user       |
| GET    | /api/v1/users/{id}  | Get user          |
| DELETE | /api/v1/users/{id}  | Delete user       |

```sh
curl -s localhost:8080/api/v1/users -d '{"name":"Ada","email":"ada@example.com"}'
```

## Configuration

| Env var             | Default | Description                  |
|---------------------|---------|------------------------------|
| ADDR                | :8080   | Listen address               |
| LOG_LEVEL           | info    | debug, info, warn, error     |
| READ_TIMEOUT        | 10s     | Read body timeout            |
| WRITE_TIMEOUT       | 15s     | Write timeout                |
| IDLE_TIMEOUT        | 60s     | Keep-alive idle timeout      |
| READ_HEADER_TIMEOUT | 5s      | Header read timeout (slowloris) |
| SHUTDOWN_TIMEOUT    | 20s     | Graceful shutdown deadline   |

## Practices applied

- Graceful shutdown on SIGINT/SIGTERM with deadline (`signal.NotifyContext` + `Server.Shutdown`).
- All server timeouts set (slowloris protection); request body size limit.
- Middleware chain: request IDs, panic recovery with stack logging, structured access logs, per-request timeout.
- Structured JSON logging via `log/slog`.
- Strict JSON decoding (unknown fields rejected), body limits, input validation.
- Thin `main`; everything else in `internal/` so packages cannot leak outside the module.
- Table-driven tests, `go vet`, `gofmt`.
