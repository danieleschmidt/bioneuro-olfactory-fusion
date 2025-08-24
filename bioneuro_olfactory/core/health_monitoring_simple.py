"""Simple health monitoring for Generation 2 validation."""

from typing import Dict, Any, List, Callable
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    status: str
    score: float
    details: Dict[str, Any]


class SystemHealthChecker:
    """Simple system health checker."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_function
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        components = {}
        overall_status = "healthy"
        
        for name, check_func in self.health_checks.items():
            try:
                component_health = check_func()
                if isinstance(component_health, ComponentHealth):
                    components[name] = {
                        "status": component_health.status,
                        "score": component_health.score,
                        "details": component_health.details
                    }
                    if component_health.status not in ["healthy", "ok"]:
                        overall_status = "degraded"
                else:
                    components[name] = {"status": "unknown", "score": 0.0}
                    overall_status = "degraded"
            except Exception as e:
                components[name] = {
                    "status": "error",
                    "score": 0.0,
                    "details": {"error": str(e)}
                }
                overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "components": components,
            "timestamp": datetime.now().isoformat(),
            "total_components": len(self.health_checks)
        }


class HealthMonitor:
    """Simple health monitor."""
    
    def __init__(self):
        self.checker = SystemHealthChecker()
        self.health_history = []
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return self.checker.check_system_health()
    
    def register_check(self, name: str, check_function: Callable):
        """Register a health check."""
        self.checker.register_health_check(name, check_function)