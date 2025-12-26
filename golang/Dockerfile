FROM golang:1.24-alpine AS builder

# required for fetching some modules and compiling C-go if needed
RUN apk add --no-cache git build-base

WORKDIR /app

# Copy go.mod/go.sum early to utilize Docker layer caching.
COPY go.mod go.sum ./
RUN go mod download

# Copy rest of source
COPY . .

# Build static binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o /bot .

# Final stage: small runtime image
FROM alpine:3.18
RUN apk add --no-cache ca-certificates

# Copy binary produced by builder
COPY --from=builder /bot /usr/local/bin/bot

# Set working dir and default DOC_ROOT location (volume mount)
WORKDIR /app
ENV DOC_ROOT=/files

VOLUME ["/files"]

ENTRYPOINT ["/usr/local/bin/bot"]
