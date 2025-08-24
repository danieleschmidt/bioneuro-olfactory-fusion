"""Simple enhanced logging for Generation 2 validation."""

import logging
import json
import time
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict, deque
import sys


class EnhancedLogger:
    """Enhanced logger with structured logging and metrics integration."""
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        log_file: Optional[str] = None,
        enable_structured_logging: bool = True,
        metrics_collector: Optional['MetricsCollector'] = None
    ):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.metrics_collector = metrics_collector or MetricsCollector()
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if enable_structured_logging:
            console_handler.setFormatter(StructuredLogFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        extra = kwargs.get('extra', {})
        self.metrics_collector.increment_counter("log_messages")
        self.logger.info(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        extra = kwargs.get('extra', {})
        self.metrics_collector.increment_counter("log_messages")
        self.logger.debug(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        extra = kwargs.get('extra', {})
        self.metrics_collector.increment_counter("log_messages")
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        extra = kwargs.get('extra', {})
        self.metrics_collector.increment_counter("log_messages")
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        extra = kwargs.get('extra', {})
        self.metrics_collector.increment_counter("log_messages")
        self.logger.critical(message, extra=extra)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logger metrics."""
        return self.metrics_collector.get_metrics_summary()


class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_data["data"] = record.structured_data
        
        return json.dumps(log_data, separators=(',', ':'), default=str)


class MetricsCollector:
    """Simple metrics collection."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric."""
        with self.lock:
            self.gauges[name] = value
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Increment counter."""
        with self.lock:
            self.counters[name] += 1
    
    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timing."""
        with self.lock:
            self.histograms[name].append(duration_ms)
    
    def get_metrics_summary(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {}
            }
            
            for name, values in self.histograms.items():
                if values:
                    summary["histograms"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values)
                    }
            
            return summary
    
    def clear_metrics(self):
        """Clear metrics."""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()


# Global instances
_default_logger = None
_default_metrics = None


def get_default_logger() -> EnhancedLogger:
    """Get default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = EnhancedLogger("bioneuro_olfactory")
    return _default_logger


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector."""
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = MetricsCollector()
    return _default_metrics