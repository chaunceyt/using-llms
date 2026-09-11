FROM golang:1.22 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /namespace-watch ./cmd/namespace-watch

FROM gcr.io/distroless/static:nonroot
COPY --from=build /namespace-watch /namespace-watch
ENTRYPOINT ["/namespace-watch"]
