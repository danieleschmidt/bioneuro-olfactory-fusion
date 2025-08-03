"""Alert management system for gas detection notifications."""

import os
import asyncio
import logging
import smtplib
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    GITHUB = "github"


@dataclass
class AlertRule:
    """Configuration for alert rules."""
    gas_type: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    confidence_threshold: float = 0.8
    time_window_seconds: int = 60
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    cooldown_seconds: int = 300  # 5 minutes
    escalation_enabled: bool = True
    
    def get_severity(self, concentration: float, confidence: float) -> AlertSeverity:
        """Determine alert severity based on concentration and confidence."""
        if confidence < self.confidence_threshold:
            return AlertSeverity.INFO
        
        if concentration >= self.emergency_threshold:
            return AlertSeverity.EMERGENCY
        elif concentration >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        elif concentration >= self.warning_threshold:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO


@dataclass
class Alert:
    """Alert data structure."""
    alert_id: str
    timestamp: datetime
    gas_type: str
    concentration: float
    confidence: float
    severity: AlertSeverity
    location: Optional[str] = None
    experiment_id: Optional[int] = None
    sensor_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'gas_type': self.gas_type,
            'concentration': self.concentration,
            'confidence': self.confidence,
            'severity': self.severity.value,
            'location': self.location,
            'experiment_id': self.experiment_id,
            'sensor_data': self.sensor_data,
            'metadata': self.metadata
        }


class NotificationService:
    """Base class for notification services."""
    
    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send notification for alert."""
        raise NotImplementedError


class EmailNotificationService(NotificationService):
    """Email notification service."""
    
    def __init__(self):
        self.smtp_host = os.getenv('EMAIL_SMTP_HOST', 'localhost')
        self.smtp_port = int(os.getenv('EMAIL_SMTP_PORT', '587'))
        self.username = os.getenv('EMAIL_USERNAME', '')
        self.password = os.getenv('EMAIL_PASSWORD', '')
        self.from_email = os.getenv('EMAIL_FROM', 'alerts@bioneuro.com')
    
    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send email notification."""
        try:
            recipients = config.get('recipients', [])
            if not recipients:
                logger.warning("No email recipients configured")
                return False
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = self._create_subject(alert)
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            
            # Create HTML and text versions
            text_content = self._create_text_content(alert)
            html_content = self._create_html_content(alert)
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.username and self.password:
                    server.starttls()
                    server.login(self.username, self.password)
                
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_subject(self, alert: Alert) -> str:
        """Create email subject line."""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🔴"
        }
        
        emoji = severity_emoji.get(alert.severity, "📊")
        return f"{emoji} Gas Detection Alert: {alert.gas_type.upper()} - {alert.severity.value.upper()}"
    
    def _create_text_content(self, alert: Alert) -> str:
        """Create plain text email content."""
        return f"""
Gas Detection Alert

Alert ID: {alert.alert_id}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Severity: {alert.severity.value.upper()}

Gas Information:
- Type: {alert.gas_type}
- Concentration: {alert.concentration:.1f} ppm
- Confidence: {alert.confidence:.1%}
- Location: {alert.location or 'Unknown'}

Recommended Actions:
{'- IMMEDIATE EVACUATION REQUIRED' if alert.severity == AlertSeverity.EMERGENCY else ''}
{'- Increase ventilation and monitor closely' if alert.severity == AlertSeverity.CRITICAL else ''}
{'- Continue monitoring with increased frequency' if alert.severity == AlertSeverity.WARNING else ''}

System Information:
- Experiment ID: {alert.experiment_id or 'N/A'}
- Detection System: BioNeuro-Olfactory-Fusion

This is an automated alert from the neuromorphic gas detection system.
For immediate assistance, contact the safety team.
        """.strip()
    
    def _create_html_content(self, alert: Alert) -> str:
        """Create HTML email content."""
        severity_colors = {
            AlertSeverity.INFO: "#2196F3",
            AlertSeverity.WARNING: "#FF9800", 
            AlertSeverity.CRITICAL: "#F44336",
            AlertSeverity.EMERGENCY: "#9C27B0"
        }
        
        color = severity_colors.get(alert.severity, "#757575")
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h1 style="margin: 0; font-size: 24px;">Gas Detection Alert</h1>
                    <p style="margin: 5px 0 0 0; font-size: 18px;">{alert.severity.value.upper()}</p>
                </div>
                
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h2 style="color: {color}; margin-top: 0;">Alert Details</h2>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr><td style="padding: 8px; font-weight: bold;">Alert ID:</td><td style="padding: 8px;">{alert.alert_id}</td></tr>
                        <tr><td style="padding: 8px; font-weight: bold;">Timestamp:</td><td style="padding: 8px;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>
                        <tr><td style="padding: 8px; font-weight: bold;">Gas Type:</td><td style="padding: 8px;">{alert.gas_type.upper()}</td></tr>
                        <tr><td style="padding: 8px; font-weight: bold;">Concentration:</td><td style="padding: 8px;">{alert.concentration:.1f} ppm</td></tr>
                        <tr><td style="padding: 8px; font-weight: bold;">Confidence:</td><td style="padding: 8px;">{alert.confidence:.1%}</td></tr>
                        <tr><td style="padding: 8px; font-weight: bold;">Location:</td><td style="padding: 8px;">{alert.location or 'Unknown'}</td></tr>
                    </table>
                </div>
                
                <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: #856404; margin-top: 0;">Recommended Actions</h3>
                    <ul style="margin: 0; padding-left: 20px;">
                        {"<li style='color: #721c24; font-weight: bold;'>IMMEDIATE EVACUATION REQUIRED</li>" if alert.severity == AlertSeverity.EMERGENCY else ""}
                        {"<li>Increase ventilation and monitor closely</li>" if alert.severity == AlertSeverity.CRITICAL else ""}
                        {"<li>Continue monitoring with increased frequency</li>" if alert.severity == AlertSeverity.WARNING else ""}
                        <li>Verify detection with secondary sensors</li>
                        <li>Review safety protocols</li>
                    </ul>
                </div>
                
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px;">
                    <h3 style="color: #1565c0; margin-top: 0;">System Information</h3>
                    <p><strong>Detection System:</strong> BioNeuro-Olfactory-Fusion Neuromorphic Network</p>
                    <p><strong>Experiment ID:</strong> {alert.experiment_id or 'N/A'}</p>
                    <p style="margin-bottom: 0;"><em>This is an automated alert. For immediate assistance, contact the safety team.</em></p>
                </div>
            </div>
        </body>
        </html>
        """


class SlackNotificationService(NotificationService):
    """Slack notification service."""
    
    def __init__(self):
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
    
    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send Slack notification."""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            # Create Slack message
            payload = self._create_slack_payload(alert, config)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _create_slack_payload(self, alert: Alert, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Slack message payload."""
        severity_colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9f00",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.EMERGENCY: "#800080"
        }
        
        severity_emojis = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.EMERGENCY: ":red_circle:"
        }
        
        color = severity_colors.get(alert.severity, "#808080")
        emoji = severity_emojis.get(alert.severity, ":bell:")
        
        channel = config.get('channel', '#alerts')
        
        return {
            "channel": channel,
            "username": "BioNeuro Gas Detection",
            "icon_emoji": ":microscope:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} Gas Detection Alert: {alert.gas_type.upper()}",
                    "title_link": config.get('dashboard_url', ''),
                    "text": f"*{alert.severity.value.upper()}* - Concentration: {alert.concentration:.1f} ppm",
                    "fields": [
                        {
                            "title": "Gas Type",
                            "value": alert.gas_type,
                            "short": True
                        },
                        {
                            "title": "Concentration",
                            "value": f"{alert.concentration:.1f} ppm",
                            "short": True
                        },
                        {
                            "title": "Confidence",
                            "value": f"{alert.confidence:.1%}",
                            "short": True
                        },
                        {
                            "title": "Location",
                            "value": alert.location or "Unknown",
                            "short": True
                        }
                    ],
                    "footer": "BioNeuro-Olfactory-Fusion",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }


class WebhookNotificationService(NotificationService):
    """Generic webhook notification service."""
    
    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        webhook_url = config.get('url', '')
        if not webhook_url:
            logger.warning("Webhook URL not configured")
            return False
        
        try:
            payload = {
                'alert': alert.to_dict(),
                'config': config,
                'timestamp': datetime.now().isoformat()
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'BioNeuro-Olfactory-Fusion/1.0'
            }
            
            # Add authentication if configured
            if 'auth_token' in config:
                headers['Authorization'] = f"Bearer {config['auth_token']}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url, 
                    json=payload, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if 200 <= response.status < 300:
                        logger.info(f"Webhook notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Webhook notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class AlertManager:
    """Central alert management system."""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_services: Dict[NotificationChannel, NotificationService] = {
            NotificationChannel.EMAIL: EmailNotificationService(),
            NotificationChannel.SLACK: SlackNotificationService(),
            NotificationChannel.WEBHOOK: WebhookNotificationService()
        }
        
        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Load default rules
        self._load_default_alert_rules()
    
    def _load_default_alert_rules(self):
        """Load default alert rules for common gases."""
        default_rules = [
            AlertRule(
                gas_type="methane",
                warning_threshold=2500,  # 50% LEL
                critical_threshold=5000,  # 100% LEL
                emergency_threshold=10000,  # 200% LEL
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                gas_type="carbon_monoxide",
                warning_threshold=50,   # OSHA PEL
                critical_threshold=100,  # OSHA STEL
                emergency_threshold=200,  # IDLH
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
            ),
            AlertRule(
                gas_type="ammonia",
                warning_threshold=25,   # OSHA PEL
                critical_threshold=50,   # OSHA STEL
                emergency_threshold=100,  # IDLH
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
            ),
            AlertRule(
                gas_type="propane",
                warning_threshold=5000,  # 50% LEL
                critical_threshold=10000,  # 100% LEL
                emergency_threshold=20000,  # 200% LEL
                notification_channels=[NotificationChannel.EMAIL]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.gas_type] = rule
    
    def add_alert_rule(self, rule: AlertRule):
        """Add or update alert rule."""
        self.alert_rules[rule.gas_type] = rule
        logger.info(f"Added alert rule for {rule.gas_type}")
    
    async def process_detection(
        self,
        gas_type: str,
        concentration: float,
        confidence: float,
        location: Optional[str] = None,
        experiment_id: Optional[int] = None,
        sensor_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Alert]:
        """Process gas detection and trigger alerts if necessary."""
        
        # Check if we have rules for this gas type
        if gas_type not in self.alert_rules:
            logger.warning(f"No alert rules configured for gas type: {gas_type}")
            return None
        
        rule = self.alert_rules[gas_type]
        severity = rule.get_severity(concentration, confidence)
        
        # Check if alert should be triggered
        if severity == AlertSeverity.INFO:
            return None
        
        # Check cooldown period
        last_alert_key = f"{gas_type}_{location or 'unknown'}"
        if last_alert_key in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[last_alert_key]
            if time_since_last.total_seconds() < rule.cooldown_seconds:
                logger.debug(f"Alert suppressed due to cooldown: {gas_type}")
                return None
        
        # Create alert
        alert = Alert(
            alert_id=self._generate_alert_id(),
            timestamp=datetime.now(),
            gas_type=gas_type,
            concentration=concentration,
            confidence=confidence,
            severity=severity,
            location=location,
            experiment_id=experiment_id,
            sensor_data=sensor_data,
            metadata=metadata
        )
        
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        self.last_alert_times[last_alert_key] = alert.timestamp
        
        # Send notifications
        await self._send_notifications(alert, rule)
        
        logger.info(f"Alert triggered: {alert.alert_id} - {gas_type} {severity.value}")
        return alert
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule):
        """Send notifications for alert."""
        notification_config = self._get_notification_config()
        
        # Send notifications to configured channels
        tasks = []
        for channel in rule.notification_channels:
            if channel in self.notification_services:
                service = self.notification_services[channel]
                config = notification_config.get(channel.value, {})
                
                task = asyncio.create_task(
                    service.send_notification(alert, config)
                )
                tasks.append((channel, task))
        
        # Wait for all notifications to complete
        for channel, task in tasks:
            try:
                success = await task
                if success:
                    logger.info(f"Notification sent via {channel.value} for alert {alert.alert_id}")
                else:
                    logger.error(f"Failed to send notification via {channel.value} for alert {alert.alert_id}")
            except Exception as e:
                logger.error(f"Error sending notification via {channel.value}: {e}")
    
    def _get_notification_config(self) -> Dict[str, Dict[str, Any]]:
        """Get notification configuration from environment."""
        return {
            'email': {
                'recipients': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',')
            },
            'slack': {
                'channel': os.getenv('SLACK_ALERT_CHANNEL', '#alerts'),
                'dashboard_url': os.getenv('DASHBOARD_URL', '')
            },
            'webhook': {
                'url': os.getenv('ALERT_WEBHOOK_URL', ''),
                'auth_token': os.getenv('WEBHOOK_AUTH_TOKEN', '')
            }
        }
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return f"alert_{uuid.uuid4().hex[:8]}"
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        return list(self.active_alerts.values())
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            if not alert.metadata:
                alert.metadata = {}
            alert.metadata['acknowledged'] = True
            alert.metadata['acknowledged_by'] = user
            alert.metadata['acknowledged_at'] = datetime.now().isoformat()
            
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        
        return False
    
    def clear_alert(self, alert_id: str, user: str = "system") -> bool:
        """Clear an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            if not alert.metadata:
                alert.metadata = {}
            alert.metadata['cleared'] = True
            alert.metadata['cleared_by'] = user
            alert.metadata['cleared_at'] = datetime.now().isoformat()
            
            logger.info(f"Alert {alert_id} cleared by {user}")
            return True
        
        return False
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        # Count alerts by severity in last 24 hours
        recent_alerts = [a for a in self.alert_history if a.timestamp >= last_24h]
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        gas_type_counts = {}
        
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
            gas_type_counts[alert.gas_type] = gas_type_counts.get(alert.gas_type, 0) + 1
        
        return {
            'total_alerts_24h': len(recent_alerts),
            'active_alerts': len(self.active_alerts),
            'severity_breakdown_24h': severity_counts,
            'gas_type_breakdown_24h': gas_type_counts,
            'total_alerts_all_time': len(self.alert_history)
        }


# Global alert manager instance
_alert_manager = None

def get_alert_manager() -> AlertManager:
    """Get singleton alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager