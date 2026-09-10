APP     := server
BIN     := bin/$(APP)
VERSION ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
LDFLAGS := -X example.com/httpserver/internal/version.Version=$(VERSION)

.PHONY: all build test test-race run vet fmt tidy clean

all: build

## build: compile the server binary into bin/
build:
	go build -ldflags "$(LDFLAGS)" -o $(BIN) ./cmd/server

## test: run all tests
test:
	go test ./...

## test-race: run all tests with the race detector
test-race:
	go test -race ./...

## run: run the server from source
run:
	go run ./cmd/server

## vet: run go vet
vet:
	go vet ./...

## fmt: format all Go source files
fmt:
	gofmt -l -w .

## tidy: sync go.mod/go.sum with imports
tidy:
	go mod tidy

## clean: remove build artifacts
clean:
	rm -rf bin
