"""Comprehensive audit management for compliance and regulatory requirements.

This module provides audit trail functionality to track all system activities,
configuration changes, and security events for compliance reporting.
"""

import json
import logging
import sqlite3
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import threading
import queue
import time

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


class AuditEventType(Enum):
    """Types of audit events."""
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    
    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    ACCESS_DENIED = "access_denied"
    
    # Sensor events
    SENSOR_CALIBRATION = "sensor_calibration"
    SENSOR_FAILURE = "sensor_failure"
    SENSOR_MAINTENANCE = "sensor_maintenance"
    
    # Detection events
    GAS_DETECTION = "gas_detection"
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"
    FALSE_ALARM = "false_alarm"
    
    # Security events
    SECURITY_BREACH = "security_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    
    # Compliance events
    CALIBRATION_DUE = "calibration_due"
    MAINTENANCE_DUE = "maintenance_due"
    CERTIFICATION_EXPIRY = "certification_expiry"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    level: AuditLevel
    description: str
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    affected_resource: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Generate checksum for integrity verification."""
        if self.checksum is None:
            data = {
                'event_id': self.event_id,
                'timestamp': self.timestamp.isoformat(),
                'event_type': self.event_type.value,
                'level': self.level.value,
                'description': self.description,
                'user_id': self.user_id,
                'source_ip': self.source_ip,
                'affected_resource': self.affected_resource,
                'metadata': self.metadata
            }
            self.checksum = hashlib.sha256(
                json.dumps(data, sort_keys=True).encode()
            ).hexdigest()


class AuditStorage:
    """Handles persistent storage of audit events."""
    
    def __init__(self, db_path: str = "/var/lib/bioneuro/audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize audit database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    description TEXT NOT NULL,
                    user_id TEXT,
                    source_ip TEXT,
                    affected_resource TEXT,
                    metadata TEXT,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events (event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_level ON audit_events (level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events (user_id)")
            
    def store_event(self, event: AuditEvent) -> bool:
        """Store audit event in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, level, description,
                        user_id, source_ip, affected_resource, metadata,
                        checksum, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.level.value,
                    event.description,
                    event.user_id,
                    event.source_ip,
                    event.affected_resource,
                    json.dumps(event.metadata),
                    event.checksum,
                    datetime.now(timezone.utc).isoformat()
                ))
            return True
        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            return False
            
    def get_events(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        level: Optional[AuditLevel] = None,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Query audit events from database."""
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        if level:
            query += " AND level = ?"
            params.append(level.value)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        events = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                for row in cursor.fetchall():
                    event = AuditEvent(
                        event_id=row['event_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        event_type=AuditEventType(row['event_type']),
                        level=AuditLevel(row['level']),
                        description=row['description'],
                        user_id=row['user_id'],
                        source_ip=row['source_ip'],
                        affected_resource=row['affected_resource'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        checksum=row['checksum']
                    )
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            
        return events
        
    def verify_integrity(self, event: AuditEvent) -> bool:
        """Verify audit event integrity using checksum."""
        # Temporarily remove checksum to recalculate
        original_checksum = event.checksum
        event.checksum = None
        event.__post_init__()  # Recalculate checksum
        
        is_valid = event.checksum == original_checksum
        event.checksum = original_checksum  # Restore original
        
        return is_valid
        
    def cleanup_old_events(self, retention_days: int = 2555):  # 7 years default
        """Clean up old audit events beyond retention period."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                logger.info(f"Cleaned up {deleted_count} old audit events")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old audit events: {e}")
            return 0


class AuditManager:
    """Manages audit events for compliance and security monitoring."""
    
    def __init__(self, storage: Optional[AuditStorage] = None):
        self.storage = storage or AuditStorage()
        self._event_queue = queue.Queue(maxsize=1000)
        self._worker_thread = None
        self._running = False
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'events_logged': 0,
            'events_processed': 0,
            'events_failed': 0,
            'start_time': datetime.now(timezone.utc)
        }
        
        # Start background processing
        self.start_processing()
        
    def start_processing(self):
        """Start background audit event processing."""
        if not self._running:
            self._running = True
            self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self._worker_thread.start()
            logger.info("Audit event processing started")
            
    def stop_processing(self):
        """Stop background audit event processing."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info("Audit event processing stopped")
        
    def _process_events(self):
        """Background thread to process audit events."""
        while self._running:
            try:
                # Get event from queue with timeout
                event = self._event_queue.get(timeout=1.0)
                
                # Store event
                if self.storage.store_event(event):
                    self._stats['events_processed'] += 1
                else:
                    self._stats['events_failed'] += 1
                    
                self._event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
                self._stats['events_failed'] += 1
                
    def log_event(
        self,
        event_type: Union[str, AuditEventType],
        description: str,
        user_id: Optional[str] = None,
        level: AuditLevel = AuditLevel.INFO,
        source_ip: Optional[str] = None,
        affected_resource: Optional[str] = None,
        **metadata
    ) -> str:
        """Log an audit event.
        
        Args:
            event_type: Type of audit event
            description: Human-readable description
            user_id: ID of user associated with event
            level: Severity level of event
            source_ip: Source IP address
            affected_resource: Resource affected by event
            **metadata: Additional metadata
            
        Returns:
            Event ID
        """
        # Convert string event type to enum
        if isinstance(event_type, str):
            try:
                event_type = AuditEventType(event_type)
            except ValueError:
                event_type = AuditEventType.USER_ACTION
                
        # Create audit event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            level=level,
            description=description,
            user_id=user_id,
            source_ip=source_ip,
            affected_resource=affected_resource,
            metadata=metadata
        )
        
        # Queue for background processing
        try:
            self._event_queue.put_nowait(event)
            self._stats['events_logged'] += 1
            
            # Log critical events immediately 
            if level in [AuditLevel.CRITICAL, AuditLevel.SECURITY]:
                logger.critical(f"AUDIT: {description}")
                
        except queue.Full:
            logger.warning("Audit event queue full, dropping event")
            self._stats['events_failed'] += 1
            
        return event.event_id
        
    def log_system_event(self, event_type: AuditEventType, description: str, **metadata):
        """Log system-related audit event."""
        return self.log_event(event_type, description, level=AuditLevel.INFO, **metadata)
        
    def log_security_event(self, description: str, user_id: Optional[str] = None, source_ip: Optional[str] = None, **metadata):
        """Log security-related audit event."""
        return self.log_event(
            AuditEventType.SECURITY_BREACH,
            description,
            user_id=user_id,
            level=AuditLevel.SECURITY,
            source_ip=source_ip,
            **metadata
        )
        
    def log_user_action(self, action: str, user_id: str, source_ip: Optional[str] = None, **metadata):
        """Log user action audit event."""
        return self.log_event(
            AuditEventType.USER_ACTION,
            action,
            user_id=user_id,
            source_ip=source_ip,
            **metadata
        )
        
    def log_sensor_event(self, event_type: AuditEventType, sensor_id: str, description: str, **metadata):
        """Log sensor-related audit event."""
        return self.log_event(
            event_type,
            description,
            affected_resource=sensor_id,
            **metadata
        )
        
    def get_events(self, **kwargs) -> List[AuditEvent]:
        """Get audit events with filtering."""
        return self.storage.get_events(**kwargs)
        
    def get_security_events(self, hours: int = 24) -> List[AuditEvent]:
        """Get security events from last N hours."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return self.get_events(
            start_time=start_time,
            level=AuditLevel.SECURITY
        )
        
    def get_user_activity(self, user_id: str, hours: int = 24) -> List[AuditEvent]:
        """Get user activity from last N hours."""
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return self.get_events(
            start_time=start_time,
            user_id=user_id
        )
        
    def generate_audit_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        events = self.get_events(start_time=start_time, end_time=end_time)
        
        # Analyze events
        event_counts = {}
        level_counts = {}
        user_activity = {}
        
        for event in events:
            # Count by event type
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Count by level
            level = event.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # Count user activity
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
                
        return {
            'report_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'total_events': len(events),
            'event_type_breakdown': event_counts,
            'severity_level_breakdown': level_counts,
            'user_activity_summary': user_activity,
            'security_events': len([e for e in events if e.level == AuditLevel.SECURITY]),
            'critical_events': len([e for e in events if e.level == AuditLevel.CRITICAL]),
            'most_active_users': sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
    def verify_audit_integrity(self, event_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Verify integrity of audit events."""
        if event_ids:
            events = []
            for event_id in event_ids:
                event_list = self.get_events(limit=1)  # Would need event_id parameter
                events.extend(event_list)
        else:
            # Verify recent events
            events = self.get_events(limit=100)
            
        integrity_results = {}
        for event in events:
            is_valid = self.storage.verify_integrity(event)
            integrity_results[event.event_id] = is_valid
            
        return integrity_results
        
    def get_status(self) -> Dict[str, Any]:
        """Get audit manager status."""
        with self._lock:
            return {
                'running': self._running,
                'queue_size': self._event_queue.qsize(),
                'statistics': self._stats.copy(),
                'uptime_seconds': (datetime.now(timezone.utc) - self._stats['start_time']).total_seconds()
            }
            
    def cleanup_old_events(self, retention_days: int = 2555):
        """Clean up old audit events."""
        return self.storage.cleanup_old_events(retention_days)


# Global audit manager instance
default_audit_manager = AuditManager()

def log_audit_event(event_type: str, description: str, user_id: str = None, **metadata) -> str:
    """Global function to log audit events."""
    return default_audit_manager.log_event(event_type, description, user_id, **metadata)