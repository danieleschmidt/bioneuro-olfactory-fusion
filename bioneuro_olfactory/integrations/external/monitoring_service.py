"""External monitoring service integrations for comprehensive system observability."""

import os
import logging
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MetricData:
    """Structured metric data for external monitoring."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    unit: Optional[str] = None
    description: Optional[str] = None


class MonitoringService:
    """Central monitoring service for external integrations."""
    
    def __init__(self):
        self.datadog_service = DatadogMonitoringService()
        self.newrelic_service = NewRelicMonitoringService()
        self.cloudwatch_service = CloudWatchMonitoringService()
        self.custom_webhook_service = CustomWebhookMonitoringService()
        
        # Configure which services are enabled
        self.enabled_services = []
        if os.getenv('DATADOG_API_KEY'):
            self.enabled_services.append(self.datadog_service)
        if os.getenv('NEWRELIC_LICENSE_KEY'):
            self.enabled_services.append(self.newrelic_service)
        if os.getenv('AWS_ACCESS_KEY_ID'):
            self.enabled_services.append(self.cloudwatch_service)
        if os.getenv('MONITORING_WEBHOOK_URL'):
            self.enabled_services.append(self.custom_webhook_service)
    
    async def send_metrics(self, metrics: List[MetricData]) -> Dict[str, bool]:
        """Send metrics to all enabled monitoring services."""
        results = {}
        
        for service in self.enabled_services:
            try:
                service_name = service.__class__.__name__
                success = await service.send_metrics(metrics)
                results[service_name] = success
                
                if success:
                    logger.info(f"Metrics sent successfully to {service_name}")
                else:
                    logger.error(f"Failed to send metrics to {service_name}")
                    
            except Exception as e:
                logger.error(f"Error sending metrics to {service.__class__.__name__}: {e}")
                results[service.__class__.__name__] = False
        
        return results
    
    async def send_neuromorphic_metrics(self, network_activity: Dict[str, Any], experiment_id: int):
        """Send neuromorphic-specific metrics."""
        timestamp = datetime.now()
        
        metrics = [
            MetricData(
                name="bioneuro.network.sparsity",
                value=network_activity['kenyon_sparsity']['sparsity_ratio'].item(),
                timestamp=timestamp,
                labels={'experiment_id': str(experiment_id), 'layer': 'kenyon'},
                unit="ratio",
                description="Kenyon cell layer sparsity ratio"
            ),
            MetricData(
                name="bioneuro.network.firing_rate",
                value=network_activity['projection_rates'].mean().item(),
                timestamp=timestamp,
                labels={'experiment_id': str(experiment_id), 'layer': 'projection'},
                unit="hz",
                description="Average projection neuron firing rate"
            ),
            MetricData(
                name="bioneuro.network.active_neurons",
                value=network_activity['kenyon_sparsity']['active_cells'].item(),
                timestamp=timestamp,
                labels={'experiment_id': str(experiment_id), 'layer': 'kenyon'},
                unit="count",
                description="Number of active Kenyon cells"
            )
        ]
        
        # Add temporal dynamics if available
        if 'temporal_dynamics' in network_activity:
            temporal = network_activity['temporal_dynamics']
            metrics.extend([
                MetricData(
                    name="bioneuro.network.burst_frequency",
                    value=temporal.get('burst_frequency', 0.0),
                    timestamp=timestamp,
                    labels={'experiment_id': str(experiment_id)},
                    unit="hz",
                    description="Network burst frequency"
                ),
                MetricData(
                    name="bioneuro.network.synchrony",
                    value=temporal.get('synchrony_index', 0.0),
                    timestamp=timestamp,
                    labels={'experiment_id': str(experiment_id)},
                    unit="ratio",
                    description="Network synchrony index"
                )
            ])
        
        return await self.send_metrics(metrics)
    
    async def send_detection_metrics(self, predictions: List[Dict], processing_time: float, experiment_id: int):
        """Send gas detection performance metrics."""
        timestamp = datetime.now()
        
        metrics = [
            MetricData(
                name="bioneuro.detection.processing_time",
                value=processing_time,
                timestamp=timestamp,
                labels={'experiment_id': str(experiment_id)},
                unit="ms",
                description="Detection processing time"
            ),
            MetricData(
                name="bioneuro.detection.predictions_count",
                value=len(predictions),
                timestamp=timestamp,
                labels={'experiment_id': str(experiment_id)},
                unit="count",
                description="Number of gas predictions"
            )
        ]
        
        # Add per-gas confidence metrics
        for pred in predictions:
            if pred['confidence'] > 0.1:  # Only significant detections
                metrics.append(
                    MetricData(
                        name="bioneuro.detection.confidence",
                        value=pred['confidence'],
                        timestamp=timestamp,
                        labels={
                            'experiment_id': str(experiment_id),
                            'gas_type': pred['gas_type']
                        },
                        unit="ratio",
                        description=f"Detection confidence for {pred['gas_type']}"
                    )
                )
                
                metrics.append(
                    MetricData(
                        name="bioneuro.detection.concentration",
                        value=pred['concentration_estimate'],
                        timestamp=timestamp,
                        labels={
                            'experiment_id': str(experiment_id),
                            'gas_type': pred['gas_type']
                        },
                        unit="ppm",
                        description=f"Estimated concentration for {pred['gas_type']}"
                    )
                )
        
        return await self.send_metrics(metrics)


class DatadogMonitoringService:
    """Datadog monitoring integration."""
    
    def __init__(self):
        self.api_key = os.getenv('DATADOG_API_KEY', '')
        self.app_key = os.getenv('DATADOG_APP_KEY', '')
        self.api_url = 'https://api.datadoghq.com/api/v1/series'
        self.service_name = 'bioneuro-olfactory-fusion'
    
    async def send_metrics(self, metrics: List[MetricData]) -> bool:
        """Send metrics to Datadog."""
        if not self.api_key:
            logger.warning("Datadog API key not configured")
            return False
        
        try:
            payload = {
                'series': [
                    {
                        'metric': metric.name,
                        'points': [[int(metric.timestamp.timestamp()), metric.value]],
                        'tags': [f"{k}:{v}" for k, v in metric.labels.items()],
                        'host': os.getenv('HOSTNAME', 'bioneuro-system'),
                        'type': 'gauge'
                    }
                    for metric in metrics
                ]
            }
            
            headers = {
                'DD-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status == 202:
                        logger.info(f"Sent {len(metrics)} metrics to Datadog")
                        return True
                    else:
                        logger.error(f"Datadog API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send metrics to Datadog: {e}")
            return False


class NewRelicMonitoringService:
    """New Relic monitoring integration."""
    
    def __init__(self):
        self.license_key = os.getenv('NEWRELIC_LICENSE_KEY', '')
        self.api_url = 'https://metric-api.newrelic.com/metric/v1'
    
    async def send_metrics(self, metrics: List[MetricData]) -> bool:
        """Send metrics to New Relic."""
        if not self.license_key:
            logger.warning("New Relic license key not configured")
            return False
        
        try:
            # New Relic expects metrics in specific format
            nr_metrics = []
            for metric in metrics:
                nr_metrics.append({
                    'name': metric.name,
                    'type': 'gauge',
                    'value': metric.value,
                    'timestamp': int(metric.timestamp.timestamp() * 1000),  # milliseconds
                    'attributes': {
                        **metric.labels,
                        'service.name': 'bioneuro-olfactory-fusion',
                        'unit': metric.unit or 'unknown'
                    }
                })
            
            payload = [{
                'common': {
                    'timestamp': int(datetime.now().timestamp() * 1000),
                    'interval.ms': 10000,
                    'attributes': {
                        'service.name': 'bioneuro-olfactory-fusion'
                    }
                },
                'metrics': nr_metrics
            }]
            
            headers = {
                'Api-Key': self.license_key,
                'Content-Type': 'application/json',
                'Content-Encoding': 'gzip'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=payload, headers=headers) as response:
                    if response.status == 202:
                        logger.info(f"Sent {len(metrics)} metrics to New Relic")
                        return True
                    else:
                        logger.error(f"New Relic API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send metrics to New Relic: {e}")
            return False


class CloudWatchMonitoringService:
    """AWS CloudWatch monitoring integration."""
    
    def __init__(self):
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.namespace = 'BioNeuro/GasDetection'
        # AWS credentials should be configured via IAM roles or environment variables
    
    async def send_metrics(self, metrics: List[MetricData]) -> bool:
        """Send metrics to CloudWatch."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            cloudwatch = boto3.client('cloudwatch', region_name=self.region)
            
            # CloudWatch accepts up to 20 metrics per request
            chunk_size = 20
            for i in range(0, len(metrics), chunk_size):
                chunk = metrics[i:i + chunk_size]
                
                metric_data = []
                for metric in chunk:
                    dimensions = [
                        {'Name': k, 'Value': v} 
                        for k, v in metric.labels.items()
                    ]
                    
                    metric_data.append({
                        'MetricName': metric.name.replace('.', '/'),  # CloudWatch naming
                        'Dimensions': dimensions,
                        'Value': metric.value,
                        'Unit': self._get_cloudwatch_unit(metric.unit),
                        'Timestamp': metric.timestamp
                    })
                
                cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=metric_data
                )
            
            logger.info(f"Sent {len(metrics)} metrics to CloudWatch")
            return True
            
        except ImportError:
            logger.error("boto3 not installed - CloudWatch integration disabled")
            return False
        except ClientError as e:
            logger.error(f"CloudWatch API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send metrics to CloudWatch: {e}")
            return False
    
    def _get_cloudwatch_unit(self, unit: Optional[str]) -> str:
        """Convert unit to CloudWatch standard unit."""
        unit_mapping = {
            'ms': 'Milliseconds',
            'hz': 'Count/Second',
            'ppm': 'Count',
            'ratio': 'Percent',
            'count': 'Count'
        }
        return unit_mapping.get(unit, 'None')


class CustomWebhookMonitoringService:
    """Custom webhook monitoring service for proprietary systems."""
    
    def __init__(self):
        self.webhook_url = os.getenv('MONITORING_WEBHOOK_URL', '')
        self.auth_token = os.getenv('MONITORING_WEBHOOK_TOKEN', '')
    
    async def send_metrics(self, metrics: List[MetricData]) -> bool:
        """Send metrics to custom webhook endpoint."""
        if not self.webhook_url:
            logger.warning("Monitoring webhook URL not configured")
            return False
        
        try:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'source': 'bioneuro-olfactory-fusion',
                'metrics': [
                    {
                        'name': metric.name,
                        'value': metric.value,
                        'timestamp': metric.timestamp.isoformat(),
                        'labels': metric.labels,
                        'unit': metric.unit,
                        'description': metric.description
                    }
                    for metric in metrics
                ]
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'BioNeuro-Monitoring/1.0'
            }
            
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Sent {len(metrics)} metrics to custom webhook")
                        return True
                    else:
                        logger.error(f"Webhook API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send metrics to webhook: {e}")
            return False


class HealthCheckService:
    """Health check service for external monitoring integration."""
    
    def __init__(self):
        self.monitoring_service = MonitoringService()
    
    async def send_system_health_metrics(self, health_data: Dict[str, Any]):
        """Send system health metrics to monitoring services."""
        timestamp = datetime.now()
        
        metrics = [
            MetricData(
                name="bioneuro.system.health_status",
                value=1.0 if health_data['status'] == 'healthy' else 0.0,
                timestamp=timestamp,
                labels={'status': health_data['status']},
                unit="boolean",
                description="Overall system health status"
            )
        ]
        
        # Component health metrics
        for component, status in health_data.get('checks', {}).items():
            is_healthy = 1.0 if 'healthy' in status else 0.0
            metrics.append(
                MetricData(
                    name="bioneuro.component.health",
                    value=is_healthy,
                    timestamp=timestamp,
                    labels={'component': component},
                    unit="boolean",
                    description=f"Health status of {component} component"
                )
            )
        
        await self.monitoring_service.send_metrics(metrics)
    
    async def send_alert_metrics(self, alert_stats: Dict[str, Any]):
        """Send alert statistics to monitoring services."""
        timestamp = datetime.now()
        
        metrics = [
            MetricData(
                name="bioneuro.alerts.total_24h",
                value=alert_stats['total_alerts_24h'],
                timestamp=timestamp,
                labels={},
                unit="count",
                description="Total alerts in last 24 hours"
            ),
            MetricData(
                name="bioneuro.alerts.active",
                value=alert_stats['active_alerts'],
                timestamp=timestamp,
                labels={},
                unit="count",
                description="Currently active alerts"
            )
        ]
        
        # Severity breakdown
        for severity, count in alert_stats['severity_breakdown_24h'].items():
            metrics.append(
                MetricData(
                    name="bioneuro.alerts.by_severity",
                    value=count,
                    timestamp=timestamp,
                    labels={'severity': severity},
                    unit="count",
                    description=f"Alerts with {severity} severity in last 24h"
                )
            )
        
        # Gas type breakdown
        for gas_type, count in alert_stats['gas_type_breakdown_24h'].items():
            metrics.append(
                MetricData(
                    name="bioneuro.alerts.by_gas_type",
                    value=count,
                    timestamp=timestamp,
                    labels={'gas_type': gas_type},
                    unit="count",
                    description=f"Alerts for {gas_type} in last 24h"
                )
            )
        
        await self.monitoring_service.send_metrics(metrics)