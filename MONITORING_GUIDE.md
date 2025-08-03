# BioNeuro-Olfactory-Fusion Monitoring Guide

## Overview

This guide covers comprehensive monitoring and observability for the neuromorphic gas detection system, including metrics collection, alerting, and performance analysis.

## Monitoring Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│    Grafana      │
│   (Metrics)     │    │   (Collection)  │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Logs       │    │   AlertManager  │    │    Jaeger       │
│   (Structured)  │    │   (Alerting)    │    │   (Tracing)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Metrics

### Neuromorphic Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `neural_spike_rate_hz` | Gauge | Neural spike rate | Hz |
| `membrane_potential_variance` | Summary | Membrane potential variance | mV |
| `kenyon_cell_sparsity_ratio` | Gauge | Kenyon cell sparsity | Ratio |
| `detection_latency_seconds` | Histogram | Detection latency | Seconds |
| `detection_confidence` | Histogram | Detection confidence score | 0-1 |
| `sensor_drift_percentage` | Gauge | Sensor drift | Percentage |
| `false_positive_rate` | Gauge | False positive rate | Ratio |

### System Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `system_cpu_usage_percent` | Gauge | CPU usage | Percentage |
| `system_memory_usage_percent` | Gauge | Memory usage | Percentage |
| `system_temperature_celsius` | Gauge | System temperature | Celsius |
| `gas_detections_total` | Counter | Total gas detections | Count |
| `api_requests_total` | Counter | API requests | Count |
| `api_request_duration_seconds` | Histogram | API request duration | Seconds |

## Prometheus Configuration

### Prometheus Rules

Create `/monitoring/prometheus/rules.yml`:

```yaml
groups:
  - name: bioneuro_alerts
    rules:
      # High gas concentration alert
      - alert: HighGasConcentration
        expr: gas_concentration_ppm > 1000
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "High gas concentration detected"
          description: "Gas concentration of {{ $value }}ppm detected, exceeding safety threshold"

      # System overheating
      - alert: SystemOverheating
        expr: system_temperature_celsius > 70
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "System temperature high"
          description: "System temperature is {{ $value }}°C"

      # High false positive rate
      - alert: HighFalsePositiveRate
        expr: false_positive_rate > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High false positive rate"
          description: "False positive rate is {{ $value }}, may need calibration"

      # API service down
      - alert: APIServiceDown
        expr: up{job="bioneuro-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API service is down"
          description: "The BioNeuro API service is not responding"

      # Database connection issues
      - alert: DatabaseConnectionFailed
        expr: database_connections_active == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
          description: "No active database connections detected"

      # High memory usage
      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"

      # Sensor drift detection
      - alert: SensorDrift
        expr: sensor_drift_percentage > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Sensor drift detected"
          description: "Sensor drift is {{ $value }}%, calibration may be needed"

      # Low detection confidence
      - alert: LowDetectionConfidence
        expr: avg_over_time(detection_confidence[10m]) < 0.7
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low detection confidence"
          description: "Average detection confidence over 10m is {{ $value }}"
```

### Scrape Configuration

Update `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'bioneuro-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

## Grafana Dashboards

### Main Dashboard

Create a comprehensive dashboard with the following panels:

#### Gas Detection Overview
- Current gas concentrations (gauge)
- Detection rate over time (time series)
- Detection confidence distribution (histogram)
- Active alerts (stat panel)

#### Neuromorphic Activity
- Spike rate visualization (time series)
- Membrane potential variance (time series)
- Kenyon cell sparsity (gauge)
- Spatial activity patterns (heatmap)

#### System Health
- CPU and memory usage (time series)
- Temperature monitoring (gauge)
- Disk usage (gauge)
- Network I/O (time series)

#### Performance Metrics
- API response times (time series)
- Request rate (stat)
- Error rate (stat)
- Database query performance (time series)

### Dashboard JSON

```json
{
  "dashboard": {
    "id": null,
    "title": "BioNeuro Olfactory Fusion - Overview",
    "tags": ["bioneuro", "neuromorphic", "gas-detection"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Gas Concentration",
        "type": "stat",
        "targets": [
          {
            "expr": "gas_concentration_ppm",
            "legendFormat": "{{gas_type}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "ppm",
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 500},
                {"color": "red", "value": 1000}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Detection Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(gas_detections_total[5m])",
            "legendFormat": "{{gas_type}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Neural Spike Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "neural_spike_rate_hz",
            "legendFormat": "Spike Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "hertz"
          }
        }
      },
      {
        "id": 4,
        "title": "System Temperature",
        "type": "gauge",
        "targets": [
          {
            "expr": "system_temperature_celsius",
            "legendFormat": "Temperature"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "celsius",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "blue", "value": 0},
                {"color": "green", "value": 30},
                {"color": "yellow", "value": 60},
                {"color": "red", "value": 80}
              ]
            }
          }
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## AlertManager Configuration

### alertmanager.yml

```yaml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourcompany.com'
  smtp_auth_username: 'alerts@yourcompany.com'
  smtp_auth_password: 'app_password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    email_configs:
      - to: 'admin@yourcompany.com'
        subject: 'BioNeuro Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          {{ end }}

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@yourcompany.com'
        subject: 'CRITICAL: BioNeuro Alert'
        body: |
          CRITICAL ALERT DETECTED
          
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Time: {{ .StartsAt }}
          {{ end }}
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#alerts'
        title: 'Critical BioNeuro Alert'
        text: |
          {{ range .Alerts }}
          🚨 *{{ .Annotations.summary }}*
          {{ .Annotations.description }}
          {{ end }}

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@yourcompany.com'
        subject: 'Warning: BioNeuro Alert'
        body: |
          {{ range .Alerts }}
          Warning: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
```

## Logging Configuration

### Structured Logging

The application uses structured logging with the following format:

```json
{
  "timestamp": "2023-12-01T12:00:00Z",
  "level": "INFO",
  "logger": "bioneuro.detection",
  "message": "Gas detection event",
  "gas_type": "methane",
  "concentration": 450.5,
  "confidence": 0.92,
  "location": "sensor_array_01",
  "trace_id": "abc123def456"
}
```

### Log Aggregation

#### ELK Stack Integration

```yaml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline:ro
    ports:
      - "5044:5044"

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
```

#### Fluentd Configuration

```ruby
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<filter bioneuro.**>
  @type parser
  key_name log
  reserve_data true
  <parse>
    @type json
  </parse>
</filter>

<match bioneuro.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name bioneuro-logs
  type_name _doc
  <buffer>
    flush_interval 5s
  </buffer>
</match>
```

## Performance Monitoring

### Custom Metrics Collection

```python
from bioneuro_olfactory.monitoring import metrics_collector, performance_profiler

# Record detection event
metrics_collector.record_detection(
    gas_type="methane",
    concentration=450.5,
    confidence=0.92,
    latency=0.045
)

# Record neuromorphic metrics
neuromorphic_metrics = NeuromorphicMetrics(
    spike_rate=125.0,
    membrane_potential_variance=0.15,
    kenyon_cell_sparsity=0.05,
    detection_latency_ms=45.0,
    confidence_score=0.92,
    gas_concentration=450.5,
    sensor_drift=1.2,
    false_positive_rate=0.03,
    detection_accuracy=0.94,
    energy_consumption_watts=15.8
)
metrics_collector.record_neuromorphic_metrics(neuromorphic_metrics)

# Profile function performance
@performance_profiler.profile_function("spike_encoding")
def encode_spikes(data):
    # Your spike encoding logic
    pass
```

### Query Examples

#### Prometheus Queries

```promql
# Average detection latency over 5 minutes
avg_over_time(detection_latency_seconds[5m])

# Gas detection rate by type
rate(gas_detections_total[5m]) by (gas_type)

# 95th percentile API response time
histogram_quantile(0.95, api_request_duration_seconds)

# System resource utilization
(system_cpu_usage_percent + system_memory_usage_percent) / 2

# False positive rate trend
increase(false_positive_rate[1h])
```

## Health Checks

### Application Health Endpoints

```python
from fastapi import FastAPI
from bioneuro_olfactory.monitoring import metrics_collector

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system metrics."""
    health_status = metrics_collector.get_health_status()
    return health_status

@app.get("/health/database")
async def database_health():
    """Database connectivity check."""
    try:
        # Test database connection
        result = await database.fetch_one("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/health/neuromorphic")
async def neuromorphic_health():
    """Neuromorphic hardware status."""
    hardware_status = {
        "loihi": check_loihi_status(),
        "brainscales": check_brainscales_status(),
        "spinnaker": check_spinnaker_status()
    }
    return {"status": "healthy", "hardware": hardware_status}
```

### External Monitoring

#### Uptime Monitoring

```bash
#!/bin/bash
# Simple uptime check script

URL="http://localhost:8000/health"
TIMEOUT=10

response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT $URL)

if [ $response -eq 200 ]; then
    echo "Service is UP"
    exit 0
else
    echo "Service is DOWN (HTTP $response)"
    exit 1
fi
```

## Troubleshooting

### Common Monitoring Issues

#### High Memory Usage in Prometheus

```yaml
# Reduce retention period
command:
  - '--storage.tsdb.retention.time=7d'
  - '--storage.tsdb.retention.size=10GB'
```

#### Missing Metrics

1. Check scrape targets: `http://localhost:9090/targets`
2. Verify metric names in application code
3. Check network connectivity between services

#### Grafana Dashboard Not Loading

1. Verify Prometheus data source configuration
2. Check dashboard queries for syntax errors
3. Ensure proper time range selection

### Performance Optimization

#### Prometheus Optimization

```yaml
global:
  scrape_interval: 30s  # Increase interval for less load
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'bioneuro-api'
    scrape_interval: 15s  # Critical metrics more frequently
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000']
```

#### Grafana Optimization

1. Use appropriate time ranges
2. Limit number of series in queries
3. Use recording rules for complex queries
4. Enable query caching

## Security Considerations

### Metrics Security

1. **Authentication**: Secure Prometheus and Grafana with authentication
2. **Network Security**: Use internal networks for metrics collection
3. **Data Sensitivity**: Avoid exposing sensitive data in metrics
4. **Access Control**: Implement role-based access for monitoring tools

### Example Security Configuration

```yaml
# Prometheus with basic auth
prometheus:
  web.config.file: /etc/prometheus/web.yml

# web.yml
basic_auth_users:
  admin: $2b$12$hNf2lSsxfm0.i4a.1kVpSOVyBCfIB51VRjgBUyv6kdnyTlgWj81Ay
```

## Maintenance

### Regular Tasks

1. **Daily**: Review critical alerts and system health
2. **Weekly**: Analyze performance trends and optimize queries
3. **Monthly**: Update monitoring configurations and dashboards
4. **Quarterly**: Review and update alerting thresholds

### Backup and Recovery

```bash
# Backup Prometheus data
docker exec prometheus tar -czf /tmp/prometheus-backup.tar.gz /prometheus

# Backup Grafana dashboards
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
     http://localhost:3000/api/search?type=dash-db | \
     jq -r '.[].uri' | \
     xargs -I {} curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
     http://localhost:3000/api/dashboards/{} > dashboards-backup.json
```