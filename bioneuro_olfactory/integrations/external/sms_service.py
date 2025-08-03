"""SMS notification service for critical gas detection alerts."""

import os
import logging
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..notifications.alert_manager import NotificationService, Alert

logger = logging.getLogger(__name__)


class SMSNotificationService(NotificationService):
    """SMS notification service using Twilio API."""
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID', '')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN', '')
        self.from_phone = os.getenv('TWILIO_FROM_PHONE', '')
        self.api_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        
        if not all([self.account_sid, self.auth_token, self.from_phone]):
            logger.warning("SMS service not fully configured. Check Twilio credentials.")
    
    async def send_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send SMS notification for critical alerts."""
        
        # Only send SMS for critical/emergency alerts
        if alert.severity.value not in ['critical', 'emergency']:
            logger.debug(f"Skipping SMS for {alert.severity.value} alert")
            return True
        
        recipients = config.get('phone_numbers', [])
        if not recipients:
            logger.warning("No SMS recipients configured")
            return False
        
        if not all([self.account_sid, self.auth_token, self.from_phone]):
            logger.error("SMS service not configured")
            return False
        
        message_text = self._create_sms_message(alert)
        
        try:
            success_count = 0
            for phone_number in recipients:
                if await self._send_sms(phone_number, message_text):
                    success_count += 1
                    logger.info(f"SMS sent to {phone_number} for alert {alert.alert_id}")
                else:
                    logger.error(f"Failed to send SMS to {phone_number}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to send SMS notifications: {e}")
            return False
    
    async def _send_sms(self, to_phone: str, message: str) -> bool:
        """Send individual SMS message."""
        try:
            data = {
                'From': self.from_phone,
                'To': to_phone,
                'Body': message
            }
            
            auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)
            
            async with aiohttp.ClientSession(auth=auth) as session:
                async with session.post(self.api_url, data=data) as response:
                    if response.status == 201:
                        return True
                    else:
                        logger.error(f"Twilio API error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    def _create_sms_message(self, alert: Alert) -> str:
        """Create SMS message text."""
        severity_emoji = {
            'critical': '=¨',
            'emergency': '=4'
        }
        
        emoji = severity_emoji.get(alert.severity.value, ' ')
        
        return (
            f"{emoji} GAS ALERT {emoji}\n"
            f"Type: {alert.gas_type.upper()}\n"
            f"Level: {alert.concentration:.0f} ppm\n"
            f"Severity: {alert.severity.value.upper()}\n"
            f"Location: {alert.location or 'Unknown'}\n"
            f"Time: {alert.timestamp.strftime('%H:%M')}\n"
            f"Alert ID: {alert.alert_id[:8]}\n"
            f"{'EVACUATE IMMEDIATELY' if alert.severity.value == 'emergency' else 'Verify and respond'}"
        )


class SlackIntegrationService:
    """Enhanced Slack integration with additional features."""
    
    def __init__(self):
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
        self.bot_token = os.getenv('SLACK_BOT_TOKEN', '')
        self.channel_alerts = os.getenv('SLACK_CHANNEL_ALERTS', '#alerts')
        self.channel_general = os.getenv('SLACK_CHANNEL_GENERAL', '#general')
    
    async def send_thread_update(self, alert: Alert, thread_ts: str, update_message: str) -> bool:
        """Send threaded update to existing alert."""
        if not self.bot_token:
            logger.warning("Slack bot token not configured for threaded updates")
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {self.bot_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'channel': self.channel_alerts,
                'thread_ts': thread_ts,
                'text': update_message,
                'unfurl_links': False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://slack.com/api/chat.postMessage',
                    headers=headers,
                    json=payload
                ) as response:
                    result = await response.json()
                    return result.get('ok', False)
                    
        except Exception as e:
            logger.error(f"Failed to send Slack thread update: {e}")
            return False
    
    async def create_incident_channel(self, alert: Alert) -> Optional[str]:
        """Create dedicated incident channel for emergency alerts."""
        if alert.severity.value != 'emergency':
            return None
        
        if not self.bot_token:
            logger.warning("Slack bot token not configured for channel creation")
            return None
        
        try:
            channel_name = f"incident-{alert.gas_type}-{alert.timestamp.strftime('%Y%m%d-%H%M')}"
            
            headers = {
                'Authorization': f'Bearer {self.bot_token}',
                'Content-Type': 'application/json'
            }
            
            # Create channel
            create_payload = {
                'name': channel_name,
                'is_private': False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://slack.com/api/conversations.create',
                    headers=headers,
                    json=create_payload
                ) as response:
                    result = await response.json()
                    
                    if result.get('ok'):
                        channel_id = result['channel']['id']
                        
                        # Post initial incident details
                        incident_message = self._create_incident_message(alert)
                        
                        message_payload = {
                            'channel': channel_id,
                            'text': incident_message,
                            'unfurl_links': False
                        }
                        
                        await session.post(
                            'https://slack.com/api/chat.postMessage',
                            headers=headers,
                            json=message_payload
                        )
                        
                        logger.info(f"Created incident channel: {channel_name}")
                        return channel_id
                    else:
                        logger.error(f"Failed to create Slack channel: {result.get('error')}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error creating incident channel: {e}")
            return None
    
    def _create_incident_message(self, alert: Alert) -> str:
        """Create incident channel initial message."""
        return f"""
=¨ **EMERGENCY GAS DETECTION INCIDENT** =¨

**Alert Details:**
" Gas Type: {alert.gas_type.upper()}
" Concentration: {alert.concentration:.1f} ppm
" Confidence: {alert.confidence:.1%}
" Location: {alert.location or 'Unknown'}
" Detection Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
" Alert ID: {alert.alert_id}

**Immediate Actions Required:**
1. =4 EVACUATE AREA IMMEDIATELY
2. =Þ Contact emergency services if needed
3. = Verify with secondary detection methods
4. =Ë Follow emergency response protocols
5. =Ý Document all actions taken

**Incident Commander:** _To be assigned_
**Status:** ACTIVE

This channel will be used to coordinate response efforts.
        """.strip()


class TeamsIntegrationService:
    """Microsoft Teams integration for enterprise environments."""
    
    def __init__(self):
        self.webhook_url = os.getenv('TEAMS_WEBHOOK_URL', '')
    
    async def send_teams_notification(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send notification to Microsoft Teams."""
        if not self.webhook_url:
            logger.warning("Teams webhook URL not configured")
            return False
        
        try:
            card = self._create_teams_card(alert, config)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=card) as response:
                    if response.status == 200:
                        logger.info(f"Teams notification sent for alert {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"Teams notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            return False
    
    def _create_teams_card(self, alert: Alert, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Microsoft Teams adaptive card."""
        severity_colors = {
            'info': '0078D4',
            'warning': 'FF8C00',
            'critical': 'FF0000',
            'emergency': '800080'
        }
        
        color = severity_colors.get(alert.severity.value, '808080')
        
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": f"Gas Detection Alert: {alert.gas_type}",
            "sections": [
                {
                    "activityTitle": f"=¨ Gas Detection Alert: {alert.gas_type.upper()}",
                    "activitySubtitle": f"Severity: {alert.severity.value.upper()}",
                    "activityImage": "https://example.com/gas-detector-icon.png",
                    "facts": [
                        {
                            "name": "Gas Type",
                            "value": alert.gas_type
                        },
                        {
                            "name": "Concentration",
                            "value": f"{alert.concentration:.1f} ppm"
                        },
                        {
                            "name": "Confidence",
                            "value": f"{alert.confidence:.1%}"
                        },
                        {
                            "name": "Location",
                            "value": alert.location or "Unknown"
                        },
                        {
                            "name": "Detection Time",
                            "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
                        },
                        {
                            "name": "Alert ID",
                            "value": alert.alert_id
                        }
                    ],
                    "markdown": True
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Dashboard",
                    "targets": [
                        {
                            "os": "default",
                            "uri": config.get('dashboard_url', 'http://localhost:3000')
                        }
                    ]
                },
                {
                    "@type": "OpenUri",
                    "name": "Acknowledge Alert",
                    "targets": [
                        {
                            "os": "default",
                            "uri": f"http://localhost:8000/api/v1/alerts/{alert.alert_id}/acknowledge"
                        }
                    ]
                }
            ]
        }


class PagerDutyIntegrationService:
    """PagerDuty integration for enterprise alerting."""
    
    def __init__(self):
        self.integration_key = os.getenv('PAGERDUTY_INTEGRATION_KEY', '')
        self.api_url = 'https://events.pagerduty.com/v2/enqueue'
    
    async def send_pagerduty_alert(self, alert: Alert, config: Dict[str, Any]) -> bool:
        """Send alert to PagerDuty."""
        if not self.integration_key:
            logger.warning("PagerDuty integration key not configured")
            return False
        
        # Only trigger PagerDuty for critical/emergency alerts
        if alert.severity.value not in ['critical', 'emergency']:
            return True
        
        try:
            event = self._create_pagerduty_event(alert)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, json=event) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty alert sent for {alert.alert_id}")
                        return True
                    else:
                        logger.error(f"PagerDuty alert failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    def _create_pagerduty_event(self, alert: Alert) -> Dict[str, Any]:
        """Create PagerDuty event payload."""
        return {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "dedup_key": f"gas-alert-{alert.alert_id}",
            "payload": {
                "summary": f"Gas Detection Alert: {alert.gas_type} - {alert.severity.value}",
                "source": "BioNeuro-Olfactory-Fusion",
                "severity": "critical" if alert.severity.value == "emergency" else "warning",
                "component": "gas-detector",
                "group": "safety-systems",
                "class": "gas-detection",
                "custom_details": {
                    "gas_type": alert.gas_type,
                    "concentration_ppm": alert.concentration,
                    "confidence": alert.confidence,
                    "location": alert.location,
                    "detection_time": alert.timestamp.isoformat(),
                    "alert_id": alert.alert_id,
                    "experiment_id": alert.experiment_id
                }
            },
            "client": "BioNeuro Gas Detection System",
            "client_url": "http://localhost:8000/api/v1/alerts"
        }