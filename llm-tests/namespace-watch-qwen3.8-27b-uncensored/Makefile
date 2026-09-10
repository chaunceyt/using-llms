GO ?= go
BINARY ?= bin/namespace-watcher

.PHONY: all build test vet run clean

all: build

build:
	$(GO) build -trimpath -ldflags="-s -w" -o $(BINARY) .

test:
	$(GO) test ./...

vet:
	$(GO) vet ./...

run:
	$(GO) run .

clean:
	rm -rf bin
