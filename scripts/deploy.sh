#!/bin/bash

# AgloK23 Deployment Script
set -e

echo "Starting AgloK23 deployment..."

# Check if required environment variables are set
if [ ! -f .env ]; then
    echo "Error: .env file not found. Please copy .env.template to .env and configure it."
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
if ! command_exists docker; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! command_exists docker-compose; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

# Parse command line arguments
ENVIRONMENT=${1:-development}
BUILD_FLAG=${2:-}

echo "Deploying for environment: $ENVIRONMENT"

# Select appropriate docker-compose file
if [ "$ENVIRONMENT" = "production" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
else
    COMPOSE_FILE="docker-compose.yml"
fi

# Build images if requested
if [ "$BUILD_FLAG" = "--build" ] || [ "$BUILD_FLAG" = "-b" ]; then
    echo "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --no-cache
fi

# Stop existing containers
echo "Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down

# Start services
echo "Starting services..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 30

# Check service health
echo "Checking service health..."
services=("redis" "postgres" "timescaledb")

for service in "${services[@]}"; do
    if docker-compose -f $COMPOSE_FILE ps $service | grep -q "Up"; then
        echo "✓ $service is running"
    else
        echo "✗ $service failed to start"
        docker-compose -f $COMPOSE_FILE logs $service
        exit 1
    fi
done

# Initialize database if it's a fresh deployment
if [ "$ENVIRONMENT" = "development" ] || [ "$BUILD_FLAG" = "--init-db" ]; then
    echo "Initializing database..."
    docker-compose -f $COMPOSE_FILE exec -T postgres psql -U postgres -d algok23 -f /docker-entrypoint-initdb.d/init_db.sql || true
fi

echo "Deployment completed successfully!"
echo ""
echo "Services are running at:"
echo "  - Application: http://localhost:8000"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - MLflow: http://localhost:5000"
echo "  - Prometheus: http://localhost:9090"
if [ "$ENVIRONMENT" = "development" ]; then
    echo "  - Jupyter: http://localhost:8888 (token: algok23)"
fi
echo ""
echo "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "To stop services: docker-compose -f $COMPOSE_FILE down"
