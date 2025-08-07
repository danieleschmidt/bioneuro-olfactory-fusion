"""Compliance and regulatory framework for neuromorphic gas detection.

This module provides comprehensive compliance management including:
- Regulatory standard compliance (OSHA, NIOSH, EU-ATEX, etc.)
- Audit trail and documentation
- Certification management
- Data retention and archival
- Incident reporting and investigation
"""

from .audit_manager import AuditManager, AuditEvent, AuditLevel
from .certification_manager import CertificationManager, Certificate, CertificationType
from .incident_reporter import IncidentReporter, Incident, IncidentSeverity
from .data_retention import DataRetentionManager, RetentionPolicy, ArchivalStrategy

__all__ = [
    'AuditManager',
    'AuditEvent', 
    'AuditLevel',
    'CertificationManager',
    'Certificate',
    'CertificationType',
    'IncidentReporter',
    'Incident',
    'IncidentSeverity',
    'DataRetentionManager',
    'RetentionPolicy',
    'ArchivalStrategy'
]

# Global compliance managers
_audit_manager = AuditManager()
_certification_manager = CertificationManager()
_incident_reporter = IncidentReporter()
_retention_manager = DataRetentionManager()

# Convenience functions
def log_audit_event(event_type: str, description: str, user_id: str = None, **metadata):
    """Log an audit event."""
    return _audit_manager.log_event(event_type, description, user_id, **metadata)

def get_compliance_status():
    """Get overall compliance status."""
    return {
        'audit_status': _audit_manager.get_status(),
        'certification_status': _certification_manager.get_status(),
        'incident_status': _incident_reporter.get_status(),
        'retention_status': _retention_manager.get_status()
    }