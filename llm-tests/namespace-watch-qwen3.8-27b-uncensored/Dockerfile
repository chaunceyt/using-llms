# syntax=docker/dockerfile:1

# --- Build stage ---
FROM golang:1.22 AS build
WORKDIR /src

# Cache module downloads.
COPY go.mod go.sum ./
RUN go mod download

# Build a static binary.
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/namespace-watcher .

# --- Runtime stage ---
# distroless/static:nonroot is a minimal, rootless base ideal for a static Go
# binary that only talks to the API server.
FROM gcr.io/distroless/static:nonroot
COPY --from=build /out/namespace-watcher /namespace-watcher
USER 65532:65532
ENTRYPOINT ["/namespace-watcher"]
