# BioNeuro-Olfactory-Fusion Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the BioNeuro-Olfactory-Fusion neuromorphic gas detection system in production environments.

## Prerequisites

### Hardware Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.4 GHz
- RAM: 8 GB
- Storage: 50 GB SSD
- Network: 100 Mbps

**Recommended for Production:**
- CPU: 8 cores, 3.0 GHz
- RAM: 32 GB
- Storage: 200 GB NVMe SSD
- Network: 1 Gbps
- GPU: NVIDIA GPU with CUDA support (optional, for acceleration)

**Neuromorphic Hardware (Optional):**
- Intel Loihi chip
- BrainScaleS system
- SpiNNaker board

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- Python 3.9+
- PostgreSQL 13+
- Redis 6.0+

## Deployment Options

### 1. Docker Compose Deployment (Recommended)

#### Production Deployment

```bash
# Clone the repository
git clone https://github.com/danieleschmidt/bioneuro-olfactory-fusion.git
cd bioneuro-olfactory-fusion

# Copy and configure environment variables
cp .env.example .env
# Edit .env file with your configuration

# Deploy production stack
make deploy-prod

# Verify deployment
make health-check
```

#### Development Deployment

```bash
# Start development environment
make dev-up

# Run with hot reload
make dev-watch

# Stop development environment
make dev-down
```

### 2. Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster 1.20+
- kubectl configured
- Helm 3.0+

```bash
# Create namespace
kubectl create namespace bioneuro-olfactory

# Deploy with Helm
helm install bioneuro-olfactory ./charts/bioneuro-olfactory \
  --namespace bioneuro-olfactory \
  --values ./charts/bioneuro-olfactory/values-production.yaml

# Check deployment status
kubectl get pods -n bioneuro-olfactory
```

### 3. Cloud Deployments

#### AWS ECS Deployment

```bash
# Configure AWS CLI
aws configure

# Deploy to ECS
make deploy-aws-ecs

# Monitor deployment
aws ecs describe-services --cluster bioneuro-cluster --services bioneuro-service
```

#### Azure Container Instances

```bash
# Login to Azure
az login

# Deploy to ACI
make deploy-azure-aci

# Check status
az container show --resource-group bioneuro-rg --name bioneuro-container
```

#### Google Cloud Run

```bash
# Configure gcloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy to Cloud Run
make deploy-gcp-run

# Check deployment
gcloud run services describe bioneuro-service --region=us-central1
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=bioneuro_olfactory
DB_USER=postgres
DB_PASSWORD=secure_password

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=secure_redis_password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here
API_DEBUG=false

# Neuromorphic Hardware
ENABLE_LOIHI=false
ENABLE_BRAINSCALES=false
ENABLE_SPINNAKER=false

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
METRICS_COLLECTION_INTERVAL=30

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=alerts@yourcompany.com
EMAIL_PASSWORD=app_password

# Cloud Storage
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=bioneuro-data

# Security
ENABLE_HTTPS=true
SSL_CERT_PATH=/etc/ssl/certs/bioneuro.crt
SSL_KEY_PATH=/etc/ssl/private/bioneuro.key
```

### Database Migration

```bash
# Run database migrations
make db-migrate

# Create initial admin user
make create-admin-user

# Load sample data (optional)
make load-sample-data
```

### SSL/TLS Configuration

#### Using Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Configure nginx to use certificates
# Certificates will be automatically renewed
```

#### Using Self-Signed Certificates

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout bioneuro.key -out bioneuro.crt -days 365 -nodes

# Move certificates to appropriate location
sudo mv bioneuro.crt /etc/ssl/certs/
sudo mv bioneuro.key /etc/ssl/private/
sudo chmod 600 /etc/ssl/private/bioneuro.key
```

## Monitoring Setup

### Prometheus Configuration

Prometheus is automatically configured with the Docker Compose deployment. Access the Prometheus UI at `http://localhost:9090`.

### Grafana Dashboard

1. Access Grafana at `http://localhost:3000`
2. Login with admin/admin (change password on first login)
3. Import the pre-configured dashboard from `monitoring/grafana/dashboards/`

### Health Checks

The system provides several health check endpoints:

```bash
# API health check
curl http://localhost:8000/health

# Database connectivity
curl http://localhost:8000/health/database

# Redis connectivity
curl http://localhost:8000/health/redis

# Neuromorphic hardware status
curl http://localhost:8000/health/hardware
```

## Backup and Recovery

### Database Backup

```bash
# Create database backup
make backup-db

# Restore from backup
make restore-db BACKUP_FILE=backup_20231201_120000.sql
```

### Complete System Backup

```bash
# Backup entire system including data volumes
make backup-system

# Restore system from backup
make restore-system BACKUP_FILE=system_backup_20231201.tar.gz
```

## Scaling

### Horizontal Scaling

#### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml bioneuro

# Scale API service
docker service scale bioneuro_api=3
```

#### Kubernetes Scaling

```bash
# Scale deployment
kubectl scale deployment bioneuro-api --replicas=3 -n bioneuro-olfactory

# Enable horizontal pod autoscaling
kubectl autoscale deployment bioneuro-api --cpu-percent=50 --min=2 --max=10 -n bioneuro-olfactory
```

### Vertical Scaling

Update resource limits in `docker-compose.prod.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Security Hardening

### Network Security

```bash
# Configure firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API (if needed)
```

### Application Security

1. **Enable HTTPS**: Always use SSL/TLS in production
2. **Secure Headers**: Configure security headers in nginx
3. **Rate Limiting**: Enable rate limiting for API endpoints
4. **Authentication**: Use strong JWT tokens and refresh mechanism
5. **Input Validation**: All inputs are validated using Pydantic models

### Container Security

```bash
# Scan images for vulnerabilities
make security-scan

# Update base images regularly
make update-base-images
```

## Performance Optimization

### Database Optimization

```sql
-- Create indexes for better query performance
CREATE INDEX CONCURRENTLY idx_detections_timestamp ON detections(timestamp);
CREATE INDEX CONCURRENTLY idx_detections_gas_type ON detections(gas_type);
CREATE INDEX CONCURRENTLY idx_sensor_readings_timestamp ON sensor_readings(timestamp);
```

### Caching Configuration

Redis is used for caching frequently accessed data:

```python
# Cache configuration in settings
CACHE_TTL = 300  # 5 minutes
CACHE_KEY_PREFIX = "bioneuro:"
```

### Neuromorphic Hardware Optimization

```bash
# Enable hardware acceleration
export ENABLE_LOIHI=true
export LOIHI_DEVICE_ID=0

# Optimize spike encoding parameters
export SPIKE_ENCODING_RATE=1000
export MEMBRANE_TAU=20.0
```

## Troubleshooting

### Common Issues

#### API Service Won't Start

```bash
# Check logs
docker logs bioneuro-api

# Common causes:
# 1. Database connection failed
# 2. Redis connection failed
# 3. Missing environment variables
# 4. Port already in use
```

#### High Memory Usage

```bash
# Monitor memory usage
docker stats

# Optimize memory:
# 1. Reduce batch sizes
# 2. Enable garbage collection tuning
# 3. Increase available memory
```

#### Slow Detection Performance

```bash
# Check system metrics
curl http://localhost:8000/metrics

# Performance tuning:
# 1. Enable GPU acceleration
# 2. Optimize neural network parameters
# 3. Use neuromorphic hardware
# 4. Tune spike encoding parameters
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# Set debug mode
export API_DEBUG=true
export LOG_LEVEL=DEBUG

# Restart services
make restart
```

### Log Analysis

```bash
# View application logs
make logs

# View specific service logs
docker logs bioneuro-api

# Follow logs in real-time
docker logs -f bioneuro-api
```

## Maintenance

### Regular Maintenance Tasks

1. **Daily**:
   - Monitor system health
   - Check error logs
   - Verify backup completion

2. **Weekly**:
   - Update security patches
   - Review performance metrics
   - Clean up old data

3. **Monthly**:
   - Update base images
   - Review and rotate secrets
   - Performance optimization review

### Update Procedure

```bash
# Backup before update
make backup-system

# Update to latest version
git pull origin main
make update-images
make restart

# Verify update
make health-check
```

## Support

For technical support and issues:

1. Check the troubleshooting section above
2. Review logs for error messages
3. Consult the API documentation at `/docs`
4. Submit issues on GitHub repository

## Appendix

### Port Reference

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | Main API service |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboard |
| Nginx | 80/443 | Reverse proxy |

### Useful Commands

```bash
# Quick deployment
make deploy-prod

# System health check
make health-check

# View system status
make status

# Clean up resources
make clean

# Reset everything
make reset
```