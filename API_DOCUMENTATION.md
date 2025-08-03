# BioNeuro-Olfactory-Fusion API Documentation

## Overview

The BioNeuro-Olfactory-Fusion API provides comprehensive endpoints for neuromorphic gas detection, sensor management, and system monitoring. This RESTful API is built with FastAPI and provides both HTTP endpoints and WebSocket connections for real-time data.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com/api/v1`

## Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Obtaining a Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "your-username",
  "password": "your-password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## Gas Detection Endpoints

### Analyze Gas Sample

Analyze a gas sample using the neuromorphic detection system.

```http
POST /detection/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "sensor_data": {
    "chemical_sensors": [
      {"sensor_id": "mq2_01", "value": 450.5, "unit": "ppm"},
      {"sensor_id": "mq7_01", "value": 12.3, "unit": "ppm"}
    ],
    "audio_features": {
      "frequency_spectrum": [0.1, 0.2, 0.3, ...],
      "temporal_features": [0.5, 0.6, 0.7, ...]
    }
  },
  "environmental_conditions": {
    "temperature": 22.5,
    "humidity": 65.0,
    "pressure": 1013.25,
    "wind_speed": 2.1
  },
  "location": {
    "zone": "industrial_area_1",
    "coordinates": {"lat": 40.7128, "lng": -74.0060}
  }
}
```

**Response:**
```json
{
  "detection_id": "det_2023120112345",
  "timestamp": "2023-12-01T12:34:56Z",
  "results": {
    "detected_gases": [
      {
        "gas_type": "methane",
        "concentration": 450.5,
        "unit": "ppm",
        "confidence": 0.92,
        "risk_level": "medium"
      }
    ],
    "overall_confidence": 0.92,
    "detection_latency_ms": 45.2,
    "neuromorphic_activity": {
      "spike_rate": 125.0,
      "membrane_potential_variance": 0.15,
      "kenyon_cell_sparsity": 0.05
    }
  },
  "alerts": [
    {
      "level": "warning",
      "message": "Methane concentration approaching safety threshold",
      "threshold": 500.0,
      "current": 450.5
    }
  ]
}
```

### Get Detection History

Retrieve historical detection data with filtering options.

```http
GET /detection/history?start_time=2023-12-01T00:00:00Z&end_time=2023-12-01T23:59:59Z&gas_type=methane&limit=100
Authorization: Bearer <token>
```

**Response:**
```json
{
  "detections": [
    {
      "detection_id": "det_2023120112345",
      "timestamp": "2023-12-01T12:34:56Z",
      "gas_type": "methane",
      "concentration": 450.5,
      "confidence": 0.92,
      "location": "industrial_area_1",
      "alert_triggered": true
    }
  ],
  "pagination": {
    "total": 1250,
    "page": 1,
    "per_page": 100,
    "pages": 13
  },
  "statistics": {
    "total_detections": 1250,
    "unique_gas_types": 5,
    "average_confidence": 0.89,
    "alert_rate": 0.15
  }
}
```

### Real-time Detection Stream

WebSocket endpoint for real-time detection data.

```javascript
const ws = new WebSocket('ws://localhost:8000/detection/stream');

ws.onopen = function(event) {
  console.log('Connected to detection stream');
  
  // Subscribe to specific gas types
  ws.send(JSON.stringify({
    type: 'subscribe',
    gas_types: ['methane', 'carbon_monoxide'],
    location: 'industrial_area_1'
  }));
};

ws.onmessage = function(event) {
  const detection = JSON.parse(event.data);
  console.log('New detection:', detection);
};
```

## Sensor Management

### Register Sensor

Register a new sensor in the system.

```http
POST /sensors/register
Authorization: Bearer <token>
Content-Type: application/json

{
  "sensor_id": "mq2_05",
  "sensor_type": "chemical",
  "gas_types": ["methane", "lpg", "propane"],
  "location": {
    "zone": "warehouse_b",
    "coordinates": {"lat": 40.7128, "lng": -74.0060}
  },
  "calibration_data": {
    "baseline_resistance": 10000,
    "sensitivity": 0.5,
    "last_calibrated": "2023-12-01T08:00:00Z"
  },
  "specifications": {
    "manufacturer": "Winsen",
    "model": "MQ-2",
    "detection_range": {"min": 300, "max": 10000, "unit": "ppm"},
    "response_time": 10,
    "operating_voltage": 5.0
  }
}
```

**Response:**
```json
{
  "sensor_id": "mq2_05",
  "status": "registered",
  "registration_time": "2023-12-01T14:30:00Z",
  "next_calibration": "2023-12-08T08:00:00Z"
}
```

### Get Sensor Status

```http
GET /sensors/mq2_05/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "sensor_id": "mq2_05",
  "status": "active",
  "health": "good",
  "last_reading": {
    "timestamp": "2023-12-01T14:29:45Z",
    "value": 340.2,
    "unit": "ppm"
  },
  "drift_analysis": {
    "current_drift": 1.2,
    "trend": "stable",
    "calibration_recommended": false
  },
  "performance_metrics": {
    "uptime_percentage": 99.8,
    "response_time_ms": 12.5,
    "error_rate": 0.001
  }
}
```

### Calibrate Sensor

```http
POST /sensors/mq2_05/calibrate
Authorization: Bearer <token>
Content-Type: application/json

{
  "calibration_type": "zero_point",
  "reference_conditions": {
    "temperature": 22.0,
    "humidity": 50.0,
    "pressure": 1013.25
  },
  "reference_gas": {
    "type": "clean_air",
    "concentration": 0.0
  }
}
```

## Neuromorphic Hardware

### Hardware Status

Get status of neuromorphic hardware components.

```http
GET /hardware/status
Authorization: Bearer <token>
```

**Response:**
```json
{
  "loihi": {
    "available": true,
    "status": "online",
    "utilization": 45.2,
    "temperature": 42.0,
    "cores_active": 128,
    "power_consumption": 2.5
  },
  "brainscales": {
    "available": false,
    "status": "offline",
    "reason": "not_connected"
  },
  "spinnaker": {
    "available": true,
    "status": "online",
    "boards_active": 4,
    "neurons_active": 16384,
    "synapses_active": 1048576
  }
}
```

### Execute Neuromorphic Computation

```http
POST /hardware/loihi/execute
Authorization: Bearer <token>
Content-Type: application/json

{
  "network_config": {
    "input_neurons": 64,
    "hidden_neurons": 512,
    "output_neurons": 8
  },
  "input_spikes": {
    "spike_times": [0.001, 0.002, 0.005, ...],
    "neuron_indices": [0, 1, 3, ...]
  },
  "execution_time": 1.0,
  "recording_options": {
    "record_spikes": true,
    "record_membrane_potential": true,
    "record_synaptic_current": false
  }
}
```

## System Monitoring

### System Health

```http
GET /health
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T14:30:00Z",
  "components": {
    "api": {"status": "healthy", "response_time": 45},
    "database": {"status": "healthy", "connections": 8},
    "redis": {"status": "healthy", "memory_usage": "256MB"},
    "neuromorphic_hardware": {"status": "healthy", "devices": 2}
  },
  "system_metrics": {
    "cpu_usage": 35.2,
    "memory_usage": 67.8,
    "disk_usage": 42.1,
    "temperature": 45.0
  }
}
```

### Performance Metrics

```http
GET /metrics
Authorization: Bearer <token>
```

Returns Prometheus-formatted metrics for monitoring systems.

### Get System Logs

```http
GET /logs?level=error&start_time=2023-12-01T00:00:00Z&limit=100
Authorization: Bearer <token>
```

**Response:**
```json
{
  "logs": [
    {
      "timestamp": "2023-12-01T14:25:30Z",
      "level": "ERROR",
      "logger": "bioneuro.detection",
      "message": "Sensor calibration drift detected",
      "details": {
        "sensor_id": "mq2_03",
        "drift_percentage": 5.2,
        "recommended_action": "recalibrate"
      }
    }
  ],
  "total": 156,
  "pagination": {
    "page": 1,
    "per_page": 100,
    "total_pages": 2
  }
}
```

## Configuration Management

### Get System Configuration

```http
GET /config
Authorization: Bearer <token>
```

**Response:**
```json
{
  "detection_thresholds": {
    "methane": 500.0,
    "carbon_monoxide": 50.0,
    "hydrogen_sulfide": 10.0
  },
  "neuromorphic_settings": {
    "spike_encoding_rate": 1000,
    "membrane_tau": 20.0,
    "kenyon_sparsity": 0.05
  },
  "alert_settings": {
    "email_enabled": true,
    "slack_enabled": true,
    "sms_enabled": false
  },
  "sensor_settings": {
    "calibration_interval_days": 7,
    "drift_threshold": 5.0,
    "auto_calibration": true
  }
}
```

### Update Configuration

```http
PUT /config
Authorization: Bearer <token>
Content-Type: application/json

{
  "detection_thresholds": {
    "methane": 450.0
  },
  "alert_settings": {
    "sms_enabled": true
  }
}
```

## Data Export

### Export Detection Data

```http
POST /export/detections
Authorization: Bearer <token>
Content-Type: application/json

{
  "format": "csv",
  "date_range": {
    "start": "2023-12-01T00:00:00Z",
    "end": "2023-12-01T23:59:59Z"
  },
  "filters": {
    "gas_types": ["methane", "carbon_monoxide"],
    "confidence_threshold": 0.8,
    "locations": ["industrial_area_1"]
  },
  "include_fields": [
    "timestamp", "gas_type", "concentration", 
    "confidence", "location", "neuromorphic_data"
  ]
}
```

**Response:**
```json
{
  "export_id": "exp_2023120114300001",
  "status": "processing",
  "estimated_completion": "2023-12-01T14:35:00Z",
  "download_url": null
}
```

### Download Export

```http
GET /export/exp_2023120114300001/download
Authorization: Bearer <token>
```

Returns the exported file based on the requested format.

## WebSocket Events

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Event Types

#### Detection Event
```json
{
  "type": "detection",
  "data": {
    "detection_id": "det_2023120112345",
    "gas_type": "methane",
    "concentration": 450.5,
    "confidence": 0.92,
    "timestamp": "2023-12-01T12:34:56Z"
  }
}
```

#### Alert Event
```json
{
  "type": "alert",
  "data": {
    "alert_id": "alert_2023120112346",
    "level": "critical",
    "message": "Gas concentration exceeds safety threshold",
    "gas_type": "carbon_monoxide",
    "concentration": 75.0,
    "threshold": 50.0,
    "location": "warehouse_b"
  }
}
```

#### System Event
```json
{
  "type": "system",
  "data": {
    "event_type": "sensor_offline",
    "sensor_id": "mq2_03",
    "timestamp": "2023-12-01T12:35:00Z",
    "details": "Sensor not responding to health check"
  }
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid sensor data format",
    "details": {
      "field": "sensor_data.chemical_sensors",
      "issue": "Expected array, got string"
    },
    "timestamp": "2023-12-01T14:30:00Z",
    "request_id": "req_2023120114300001"
  }
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Anonymous requests**: 100 requests per hour
- **Authenticated requests**: 1000 requests per hour
- **Detection analysis**: 60 requests per minute
- **Real-time stream**: 1 connection per user

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1701434400
```

## SDK Examples

### Python SDK

```python
from bioneuro_client import BioNeuroClient

# Initialize client
client = BioNeuroClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Analyze gas sample
result = client.detection.analyze({
    "sensor_data": {
        "chemical_sensors": [
            {"sensor_id": "mq2_01", "value": 450.5, "unit": "ppm"}
        ]
    },
    "location": {"zone": "industrial_area_1"}
})

print(f"Detected: {result.detected_gases[0].gas_type}")
print(f"Confidence: {result.overall_confidence}")

# Real-time detection stream
for detection in client.detection.stream():
    print(f"New detection: {detection.gas_type} at {detection.concentration}ppm")
```

### JavaScript SDK

```javascript
import { BioNeuroClient } from '@bioneuro/client';

const client = new BioNeuroClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Analyze gas sample
const result = await client.detection.analyze({
  sensorData: {
    chemicalSensors: [
      { sensorId: 'mq2_01', value: 450.5, unit: 'ppm' }
    ]
  },
  location: { zone: 'industrial_area_1' }
});

console.log(`Detected: ${result.detectedGases[0].gasType}`);

// Real-time stream
client.detection.stream()
  .on('detection', (detection) => {
    console.log(`New detection: ${detection.gasType}`);
  })
  .on('alert', (alert) => {
    console.log(`Alert: ${alert.message}`);
  });
```

## Testing

### Postman Collection

A Postman collection is available at `/docs/postman/BioNeuro-API.postman_collection.json` with pre-configured requests for all endpoints.

### cURL Examples

```bash
# Get authentication token
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Analyze gas sample
curl -X POST "http://localhost:8000/detection/analyze" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"sensor_data": {"chemical_sensors": [{"sensor_id": "mq2_01", "value": 450.5}]}}'

# Get system health
curl -X GET "http://localhost:8000/health" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Support

- **API Documentation**: Available at `/docs` (Swagger UI) and `/redoc` (ReDoc)
- **GitHub Issues**: Submit bugs and feature requests
- **Support Email**: support@yourcompany.com

## Changelog

### v1.2.0 (2023-12-01)
- Added neuromorphic hardware endpoints
- Improved real-time streaming performance
- Enhanced error handling and validation

### v1.1.0 (2023-11-15)
- Added sensor calibration endpoints
- Implemented data export functionality
- Added WebSocket support for real-time data

### v1.0.0 (2023-11-01)
- Initial API release
- Core detection functionality
- Basic sensor management