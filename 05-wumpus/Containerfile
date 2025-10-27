# Build stage
FROM golang:1.24-alpine AS builder

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod ./

# Download dependencies
RUN go mod download

# Copy source code
COPY README.md ./
COPY *.go ./

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o wumpus-server .

# Runtime stage
FROM scratch

# Copy the binary from builder
COPY --from=builder /app/wumpus-server /wumpus-server
# Copy the README for reference
COPY --from=builder /app/README.md /README.md

# Expose the port the server runs on
EXPOSE 8080

# Run the server
ENTRYPOINT ["/wumpus-server"]
