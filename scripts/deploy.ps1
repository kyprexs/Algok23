# AgloK23 PowerShell Deployment Script
param(
    [string]$Environment = "development",
    [switch]$Build,
    [switch]$InitDb
)

Write-Host "Starting AgloK23 deployment..." -ForegroundColor Green

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Error: .env file not found. Please copy .env.template to .env and configure it." -ForegroundColor Red
    exit 1
}

# Function to check if a command exists
function Test-Command($command) {
    try {
        Get-Command $command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check dependencies
if (-not (Test-Command "docker")) {
    Write-Host "Error: Docker is not installed" -ForegroundColor Red
    exit 1
}

if (-not (Test-Command "docker-compose")) {
    Write-Host "Error: Docker Compose is not installed" -ForegroundColor Red
    exit 1
}

Write-Host "Deploying for environment: $Environment" -ForegroundColor Yellow

# Select appropriate docker-compose file
$ComposeFile = if ($Environment -eq "production") { "docker-compose.prod.yml" } else { "docker-compose.yml" }

# Build images if requested
if ($Build) {
    Write-Host "Building Docker images..." -ForegroundColor Yellow
    docker-compose -f $ComposeFile build --no-cache
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to build Docker images" -ForegroundColor Red
        exit 1
    }
}

# Stop existing containers
Write-Host "Stopping existing containers..." -ForegroundColor Yellow
docker-compose -f $ComposeFile down

# Start services
Write-Host "Starting services..." -ForegroundColor Yellow
docker-compose -f $ComposeFile up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to start services" -ForegroundColor Red
    exit 1
}

# Wait for services to be healthy
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check service health
Write-Host "Checking service health..." -ForegroundColor Yellow
$services = @("redis", "postgres", "timescaledb")

foreach ($service in $services) {
    $status = docker-compose -f $ComposeFile ps $service
    if ($status -match "Up") {
        Write-Host "✓ $service is running" -ForegroundColor Green
    }
    else {
        Write-Host "✗ $service failed to start" -ForegroundColor Red
        docker-compose -f $ComposeFile logs $service
        exit 1
    }
}

# Initialize database if requested or in development
if ($Environment -eq "development" -or $InitDb) {
    Write-Host "Initializing database..." -ForegroundColor Yellow
    docker-compose -f $ComposeFile exec -T postgres psql -U postgres -d algok23 -f /docker-entrypoint-initdb.d/init_db.sql
}

Write-Host "Deployment completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Services are running at:" -ForegroundColor Cyan
Write-Host "  - Application: http://localhost:8000" -ForegroundColor White
Write-Host "  - Grafana: http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host "  - MLflow: http://localhost:5000" -ForegroundColor White
Write-Host "  - Prometheus: http://localhost:9090" -ForegroundColor White

if ($Environment -eq "development") {
    Write-Host "  - Jupyter: http://localhost:8888 (token: algok23)" -ForegroundColor White
}

Write-Host ""
Write-Host "To view logs: docker-compose -f $ComposeFile logs -f" -ForegroundColor Yellow
Write-Host "To stop services: docker-compose -f $ComposeFile down" -ForegroundColor Yellow
