BINARY := bin/server
GO ?= go

.PHONY: build run test vet fmt clean

build:
	$(GO) build -o $(BINARY) ./cmd/server

run: build
	HTTP_ADDR=:8080 ./$(BINARY)

test:
	$(GO) test ./... -race -count=1

vet:
	$(GO) vet ./...

fmt:
	gofmt -w .

clean:
	rm -rf bin coverage.out
