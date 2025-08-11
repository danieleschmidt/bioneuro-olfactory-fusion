"""Enhanced structured logging system with rotation and performance metrics."""

import logging
import logging.handlers
import json
import sys
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import queue
import os
import gzip
import shutil
from collections import defaultdict, deque
import traceback
import inspect
import uuid
from contextlib import contextmanager
import psutil
import socket

# Try to import colored logging if available
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class LogLevel(Enum):
    """Extended log levels for the BioNeuro system."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 55  # Custom level for security events
    AUDIT = 60     # Custom level for audit events


class LogFormat(Enum):
    """Supported log formats."""
    JSON = "json"
    TEXT = "text"
    COLORED = "colored"
    STRUCTURED = "structured"


@dataclass
class LogMetrics:
    """Log metrics and statistics."""
    total_logs: int = 0
    logs_by_level: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    logs_by_module: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_count: int = 0
    warning_count: int = 0
    average_log_size: float = 0.0
    logs_per_second: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_logs": self.total_logs,
            "logs_by_level": dict(self.logs_by_level),
            "logs_by_module": dict(self.logs_by_module),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "average_log_size": self.average_log_size,
            "logs_per_second": self.logs_per_second,
            "last_reset": self.last_reset.isoformat()
        }


@dataclass
class ContextData:
    """Context data for structured logging."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "operation": self.operation,
            "component": self.component,
            "tags": self.tags,
            **self.custom_fields
        }


class StructuredFormatter(logging.Formatter):
    """Enhanced structured JSON formatter with context awareness."""
    
    def __init__(
        self, 
        include_trace: bool = True,
        include_system_info: bool = False,
        sensitive_fields: Optional[List[str]] = None
    ):
        super().__init__()
        self.include_trace = include_trace
        self.include_system_info = include_system_info
        self.sensitive_fields = sensitive_fields or ["password", "token", "secret", "key", "auth"]
        self.hostname = socket.gethostname()
        self.process_id = os.getpid()
        
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "thread": record.thread,
            "thread_name": record.threadName,
            "process_id": self.process_id,
            "hostname": self.hostname
        }
        
        # Add system information if enabled
        if self.include_system_info:
            log_data["system_info"] = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "timestamp": datetime.now().isoformat()
            }
            
        # Add context data if available
        if hasattr(record, 'context_data'):
            context = record.context_data
            if isinstance(context, ContextData):
                log_data["context"] = context.to_dict()
            elif isinstance(context, dict):
                log_data["context"] = context
                
        # Add structured data if available
        if hasattr(record, 'structured_data'):
            structured = record.structured_data
            if isinstance(structured, dict):
                # Sanitize sensitive fields
                sanitized_data = self._sanitize_sensitive_data(structured)
                log_data["data"] = sanitized_data
                
        # Add exception information if present
        if record.exc_info and self.include_trace:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
            
        # Add stack trace for errors if enabled
        if record.levelno >= logging.ERROR and self.include_trace:
            if not record.exc_info:  # Only add if not already present
                log_data["stack_trace"] = self._get_caller_stack()
                
        return json.dumps(log_data, separators=(',', ':'), default=str)
        
    def _sanitize_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive fields from log data."""
        if not isinstance(data, dict):
            return data
            
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_sensitive_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
                
        return sanitized
        
    def _get_caller_stack(self, skip_frames: int = 8) -> List[str]:
        """Get caller stack information."""
        stack = traceback.extract_stack()
        # Skip formatter frames and return relevant stack
        return [f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in stack[:-skip_frames]]


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development."""
    
    def __init__(self):
        if COLORLOG_AVAILABLE:
            super().__init__()
            self.colored_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                    'SECURITY': 'magenta',
                    'AUDIT': 'blue'
                }
            )
        else:
            super().__init__(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            self.colored_formatter = None
            
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors if available."""
        if self.colored_formatter:
            return self.colored_formatter.format(record)
        else:
            return super().format(record)


class PerformanceTrackingFilter(logging.Filter):
    """Filter that tracks logging performance metrics."""
    
    def __init__(self):
        super().__init__()
        self.metrics = LogMetrics()
        self.log_sizes = deque(maxlen=1000)  # Track last 1000 log sizes
        self.log_times = deque(maxlen=1000)  # Track last 1000 log times
        self.lock = threading.Lock()
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Track metrics for each log record."""
        with self.lock:
            now = time.time()
            
            # Update metrics
            self.metrics.total_logs += 1
            self.metrics.logs_by_level[record.levelname] += 1
            self.metrics.logs_by_module[record.module] += 1
            
            if record.levelno >= logging.ERROR:
                self.metrics.error_count += 1
            elif record.levelno >= logging.WARNING:
                self.metrics.warning_count += 1
                
            # Estimate log size
            log_size = len(record.getMessage()) + 100  # Rough estimate with overhead
            self.log_sizes.append(log_size)
            self.log_times.append(now)
            
            # Calculate average log size
            if self.log_sizes:
                self.metrics.average_log_size = sum(self.log_sizes) / len(self.log_sizes)
                
            # Calculate logs per second (last minute)
            cutoff_time = now - 60
            recent_logs = sum(1 for t in self.log_times if t > cutoff_time)
            self.metrics.logs_per_second = recent_logs / 60.0
            
        return True
        
    def get_metrics(self) -> LogMetrics:
        """Get current metrics."""
        with self.lock:
            return LogMetrics(
                total_logs=self.metrics.total_logs,
                logs_by_level=dict(self.metrics.logs_by_level),
                logs_by_module=dict(self.metrics.logs_by_module),
                error_count=self.metrics.error_count,
                warning_count=self.metrics.warning_count,
                average_log_size=self.metrics.average_log_size,
                logs_per_second=self.metrics.logs_per_second,
                last_reset=self.metrics.last_reset
            )
            
    def reset_metrics(self):
        """Reset metrics counters."""
        with self.lock:
            self.metrics = LogMetrics()
            self.log_sizes.clear()
            self.log_times.clear()


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler for high-performance logging."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 10000):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.stop_event = threading.Event()
        self.dropped_logs = 0
        self.worker_thread.start()
        
    def emit(self, record: logging.LogRecord):
        """Add record to queue for async processing."""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped_logs += 1
            # Optionally log to stderr about dropped logs
            if self.dropped_logs % 100 == 1:  # Log every 100 dropped logs
                print(f"Warning: Dropped {self.dropped_logs} log messages due to full queue", 
                      file=sys.stderr)
                
    def _worker(self):
        """Worker thread to process log records."""
        while not self.stop_event.is_set():
            try:
                record = self.queue.get(timeout=1.0)
                self.target_handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async log worker: {e}", file=sys.stderr)
                
    def close(self):
        """Close handler and wait for queue to empty."""
        self.stop_event.set()
        
        # Process remaining items in queue
        while not self.queue.empty():
            try:
                record = self.queue.get_nowait()
                self.target_handler.emit(record)
            except queue.Empty:
                break
            except Exception:
                pass
                
        self.worker_thread.join(timeout=5.0)
        self.target_handler.close()
        super().close()


class CompressedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating file handler with automatic compression of old logs."""
    
    def __init__(
        self, 
        filename: str, 
        mode: str = 'a',
        maxBytes: int = 100 * 1024 * 1024,  # 100MB default
        backupCount: int = 10,
        encoding: str = 'utf-8',
        delay: bool = False,
        compress_backups: bool = True
    ):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress_backups = compress_backups
        
    def doRollover(self):
        """Do a rollover with optional compression."""
        super().doRollover()
        
        if self.compress_backups and self.backupCount > 0:
            # Compress the most recent backup
            recent_backup = f"{self.baseFilename}.1"
            if os.path.exists(recent_backup):
                compressed_name = f"{recent_backup}.gz"
                
                try:
                    with open(recent_backup, 'rb') as f_in:
                        with gzip.open(compressed_name, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Remove uncompressed file
                    os.remove(recent_backup)
                    
                    # Update backup file names to account for compression
                    for i in range(2, self.backupCount + 1):
                        old_name = f"{self.baseFilename}.{i}"
                        new_name = f"{self.baseFilename}.{i}.gz"
                        
                        if os.path.exists(old_name):
                            os.rename(old_name, new_name)
                            
                except Exception as e:
                    print(f"Failed to compress log backup: {e}", file=sys.stderr)


class LoggingManager:
    """Comprehensive logging manager with advanced features."""
    
    def __init__(
        self,
        app_name: str = "bioneuro_olfactory",
        base_log_dir: str = "/var/log/bioneuro",
        default_level: LogLevel = LogLevel.INFO,
        enable_async: bool = True,
        enable_metrics: bool = True,
        enable_compression: bool = True
    ):
        self.app_name = app_name
        self.base_log_dir = Path(base_log_dir)
        self.default_level = default_level
        self.enable_async = enable_async
        self.enable_metrics = enable_metrics
        self.enable_compression = enable_compression
        
        # Create log directory
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.context_stack: Dict[str, List[ContextData]] = defaultdict(list)
        self.performance_filter: Optional[PerformanceTrackingFilter] = None
        self.lock = threading.Lock()
        
        # Register custom log levels
        self._register_custom_levels()
        
        # Setup performance tracking
        if self.enable_metrics:
            self.performance_filter = PerformanceTrackingFilter()
            
        # Setup default loggers
        self._setup_default_loggers()
        
    def _register_custom_levels(self):
        """Register custom log levels."""
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        logging.addLevelName(LogLevel.SECURITY.value, "SECURITY")
        logging.addLevelName(LogLevel.AUDIT.value, "AUDIT")
        
    def _setup_default_loggers(self):
        """Setup default loggers for the application."""
        # Main application logger
        self.create_logger(
            name=self.app_name,
            log_file=self.base_log_dir / "application.log",
            format_type=LogFormat.STRUCTURED,
            level=self.default_level
        )
        
        # Security logger
        self.create_logger(
            name=f"{self.app_name}.security",
            log_file=self.base_log_dir / "security.log",
            format_type=LogFormat.STRUCTURED,
            level=LogLevel.INFO,
            max_bytes=50 * 1024 * 1024,  # 50MB for security logs
            backup_count=20
        )
        
        # Audit logger
        self.create_logger(
            name=f"{self.app_name}.audit",
            log_file=self.base_log_dir / "audit.log",
            format_type=LogFormat.STRUCTURED,
            level=LogLevel.AUDIT,
            max_bytes=100 * 1024 * 1024,  # 100MB for audit logs
            backup_count=50
        )
        
        # Performance logger
        self.create_logger(
            name=f"{self.app_name}.performance",
            log_file=self.base_log_dir / "performance.log",
            format_type=LogFormat.STRUCTURED,
            level=LogLevel.INFO
        )
        
        # Error logger
        self.create_logger(
            name=f"{self.app_name}.errors",
            log_file=self.base_log_dir / "errors.log",
            format_type=LogFormat.STRUCTURED,
            level=LogLevel.ERROR,
            max_bytes=100 * 1024 * 1024,
            backup_count=30
        )
        
    def create_logger(
        self,
        name: str,
        log_file: Optional[Path] = None,
        format_type: LogFormat = LogFormat.STRUCTURED,
        level: LogLevel = LogLevel.INFO,
        max_bytes: int = 100 * 1024 * 1024,
        backup_count: int = 10,
        enable_console: bool = False
    ) -> logging.Logger:
        """Create a new logger with specified configuration."""
        with self.lock:
            if name in self.loggers:
                return self.loggers[name]
                
            logger = logging.getLogger(name)
            logger.setLevel(level.value)
            
            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                
            # Create file handler if log file specified
            if log_file:
                if self.enable_compression:
                    file_handler = CompressedRotatingFileHandler(
                        str(log_file),
                        maxBytes=max_bytes,
                        backupCount=backup_count,
                        compress_backups=True
                    )
                else:
                    file_handler = logging.handlers.RotatingFileHandler(
                        str(log_file),
                        maxBytes=max_bytes,
                        backupCount=backup_count
                    )
                    
                # Set formatter based on format type
                if format_type == LogFormat.STRUCTURED or format_type == LogFormat.JSON:
                    file_handler.setFormatter(StructuredFormatter(
                        include_trace=True,
                        include_system_info=True
                    ))
                else:
                    file_handler.setFormatter(logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                    ))
                    
                # Wrap in async handler if enabled
                if self.enable_async:
                    async_handler = AsyncLogHandler(file_handler)
                    logger.addHandler(async_handler)
                    self.handlers[f"{name}_file_async"] = async_handler
                else:
                    logger.addHandler(file_handler)
                    self.handlers[f"{name}_file"] = file_handler
                    
            # Create console handler if enabled
            if enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                
                if format_type == LogFormat.COLORED:
                    console_handler.setFormatter(ColoredConsoleFormatter())
                elif format_type == LogFormat.STRUCTURED:
                    console_handler.setFormatter(StructuredFormatter(include_trace=False))
                else:
                    console_handler.setFormatter(logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    ))
                    
                logger.addHandler(console_handler)
                self.handlers[f"{name}_console"] = console_handler
                
            # Add performance filter if enabled
            if self.enable_metrics and self.performance_filter:
                logger.addFilter(self.performance_filter)
                
            # Store logger
            self.loggers[name] = logger
            
            return logger
            
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get logger by name or return default application logger."""
        if name is None:
            name = self.app_name
            
        if name not in self.loggers:
            # Create logger on demand
            return self.create_logger(name)
            
        return self.loggers[name]
        
    @contextmanager
    def logging_context(self, context_data: Union[ContextData, Dict[str, Any]]):
        """Context manager for structured logging with context data."""
        thread_id = threading.get_ident()
        
        # Convert dict to ContextData if needed
        if isinstance(context_data, dict):
            context_data = ContextData(**context_data)
            
        # Push context onto stack
        self.context_stack[thread_id].append(context_data)
        
        try:
            yield
        finally:
            # Pop context from stack
            if self.context_stack[thread_id]:
                self.context_stack[thread_id].pop()
                
    def log_with_context(
        self,
        logger_name: str,
        level: LogLevel,
        message: str,
        structured_data: Optional[Dict[str, Any]] = None,
        context_data: Optional[ContextData] = None,
        **kwargs
    ):
        """Log with structured data and context."""
        logger = self.get_logger(logger_name)
        
        # Get current context if not provided
        thread_id = threading.get_ident()
        if context_data is None and self.context_stack[thread_id]:
            context_data = self.context_stack[thread_id][-1]
            
        # Create log record
        record = logging.LogRecord(
            name=logger.name,
            level=level.value,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        # Add context and structured data
        if context_data:
            record.context_data = context_data
            
        if structured_data:
            record.structured_data = structured_data
            
        # Add caller information
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            record.pathname = caller_frame.f_code.co_filename
            record.lineno = caller_frame.f_lineno
            record.funcName = caller_frame.f_code.co_name
            record.module = os.path.splitext(os.path.basename(record.pathname))[0]
            
        logger.handle(record)
        
    def trace(self, message: str, logger_name: Optional[str] = None, **kwargs):
        """Log trace message."""
        self.log_with_context(logger_name or self.app_name, LogLevel.TRACE, message, **kwargs)
        
    def debug(self, message: str, logger_name: Optional[str] = None, **kwargs):
        """Log debug message."""
        self.log_with_context(logger_name or self.app_name, LogLevel.DEBUG, message, **kwargs)
        
    def info(self, message: str, logger_name: Optional[str] = None, **kwargs):
        """Log info message."""
        self.log_with_context(logger_name or self.app_name, LogLevel.INFO, message, **kwargs)
        
    def warning(self, message: str, logger_name: Optional[str] = None, **kwargs):
        """Log warning message."""
        self.log_with_context(logger_name or self.app_name, LogLevel.WARNING, message, **kwargs)
        
    def error(self, message: str, logger_name: Optional[str] = None, **kwargs):
        """Log error message."""
        # Also log to error logger
        self.log_with_context(logger_name or self.app_name, LogLevel.ERROR, message, **kwargs)
        if logger_name != f"{self.app_name}.errors":
            self.log_with_context(f"{self.app_name}.errors", LogLevel.ERROR, message, **kwargs)
            
    def critical(self, message: str, logger_name: Optional[str] = None, **kwargs):
        """Log critical message."""
        self.log_with_context(logger_name or self.app_name, LogLevel.CRITICAL, message, **kwargs)
        if logger_name != f"{self.app_name}.errors":
            self.log_with_context(f"{self.app_name}.errors", LogLevel.CRITICAL, message, **kwargs)
            
    def security(self, message: str, **kwargs):
        """Log security message."""
        self.log_with_context(f"{self.app_name}.security", LogLevel.SECURITY, message, **kwargs)
        
    def audit(self, message: str, **kwargs):
        """Log audit message."""
        self.log_with_context(f"{self.app_name}.audit", LogLevel.AUDIT, message, **kwargs)
        
    def performance(self, message: str, **kwargs):
        """Log performance message."""
        self.log_with_context(f"{self.app_name}.performance", LogLevel.INFO, message, **kwargs)
        
    def get_metrics(self) -> Optional[LogMetrics]:
        """Get logging performance metrics."""
        if self.performance_filter:
            return self.performance_filter.get_metrics()
        return None
        
    def reset_metrics(self):
        """Reset logging performance metrics."""
        if self.performance_filter:
            self.performance_filter.reset_metrics()
            
    def export_logs(
        self, 
        output_file: Path,
        logger_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level_filter: Optional[LogLevel] = None
    ):
        """Export logs to a file with optional filtering."""
        # This is a placeholder for log export functionality
        # In a real implementation, this would read from log files and apply filters
        pass
        
    def setup_log_rotation_schedule(
        self,
        rotation_time: str = "midnight",
        rotation_interval: int = 1
    ):
        """Setup automatic log rotation schedule."""
        # Replace file handlers with timed rotating handlers
        for name, logger in self.loggers.items():
            for handler in logger.handlers[:]:
                if isinstance(handler, (logging.handlers.RotatingFileHandler, CompressedRotatingFileHandler)):
                    # Get current configuration
                    filename = handler.baseFilename
                    level = handler.level
                    formatter = handler.formatter
                    
                    # Remove old handler
                    logger.removeHandler(handler)
                    
                    # Create timed rotating handler
                    timed_handler = logging.handlers.TimedRotatingFileHandler(
                        filename,
                        when=rotation_time,
                        interval=rotation_interval,
                        backupCount=30,
                        encoding='utf-8'
                    )
                    
                    timed_handler.setLevel(level)
                    timed_handler.setFormatter(formatter)
                    
                    # Wrap in async handler if needed
                    if self.enable_async:
                        async_handler = AsyncLogHandler(timed_handler)
                        logger.addHandler(async_handler)
                    else:
                        logger.addHandler(timed_handler)
                        
    def cleanup(self):
        """Cleanup logging resources."""
        with self.lock:
            # Close all handlers
            for handler in self.handlers.values():
                try:
                    handler.close()
                except Exception as e:
                    print(f"Error closing handler: {e}", file=sys.stderr)
                    
            # Clear internal state
            self.loggers.clear()
            self.handlers.clear()
            self.context_stack.clear()


# Global logging manager instance
_logging_manager = None
_logging_manager_lock = threading.Lock()


def get_logging_manager() -> LoggingManager:
    """Get thread-safe global logging manager instance."""
    global _logging_manager
    
    if _logging_manager is None:
        with _logging_manager_lock:
            if _logging_manager is None:
                _logging_manager = LoggingManager()
                
    return _logging_manager


def configure_logging_manager(
    app_name: str = "bioneuro_olfactory",
    base_log_dir: str = "/var/log/bioneuro",
    default_level: LogLevel = LogLevel.INFO,
    enable_async: bool = True,
    enable_metrics: bool = True,
    enable_compression: bool = True
) -> LoggingManager:
    """Configure global logging manager with specific settings."""
    global _logging_manager
    
    with _logging_manager_lock:
        _logging_manager = LoggingManager(
            app_name=app_name,
            base_log_dir=base_log_dir,
            default_level=default_level,
            enable_async=enable_async,
            enable_metrics=enable_metrics,
            enable_compression=enable_compression
        )
        
    return _logging_manager


# Convenience functions for structured logging
def log_with_context(
    message: str,
    level: LogLevel = LogLevel.INFO,
    logger_name: Optional[str] = None,
    **kwargs
):
    """Log with automatic context detection."""
    manager = get_logging_manager()
    manager.log_with_context(logger_name or manager.app_name, level, message, **kwargs)


def trace(message: str, **kwargs):
    """Log trace message."""
    manager = get_logging_manager()
    manager.trace(message, **kwargs)


def debug(message: str, **kwargs):
    """Log debug message."""
    manager = get_logging_manager()
    manager.debug(message, **kwargs)


def info(message: str, **kwargs):
    """Log info message."""
    manager = get_logging_manager()
    manager.info(message, **kwargs)


def warning(message: str, **kwargs):
    """Log warning message."""
    manager = get_logging_manager()
    manager.warning(message, **kwargs)


def error(message: str, **kwargs):
    """Log error message."""
    manager = get_logging_manager()
    manager.error(message, **kwargs)


def critical(message: str, **kwargs):
    """Log critical message."""
    manager = get_logging_manager()
    manager.critical(message, **kwargs)


def security(message: str, **kwargs):
    """Log security message."""
    manager = get_logging_manager()
    manager.security(message, **kwargs)


def audit(message: str, **kwargs):
    """Log audit message."""
    manager = get_logging_manager()
    manager.audit(message, **kwargs)


def performance(message: str, **kwargs):
    """Log performance message."""
    manager = get_logging_manager()
    manager.performance(message, **kwargs)


@contextmanager
def logging_context(**context_data):
    """Context manager for structured logging."""
    manager = get_logging_manager()
    with manager.logging_context(context_data):
        yield


# Performance timing decorator
def log_execution_time(
    logger_name: Optional[str] = None,
    level: LogLevel = LogLevel.INFO,
    include_args: bool = False
):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func.__name__
            
            context_data = {
                "operation": function_name,
                "component": func.__module__
            }
            
            if include_args:
                context_data["args_count"] = len(args)
                context_data["kwargs_keys"] = list(kwargs.keys())
                
            manager = get_logging_manager()
            
            try:
                with manager.logging_context(context_data):
                    result = func(*args, **kwargs)
                    
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                
                manager.performance(
                    f"Function {function_name} executed successfully",
                    structured_data={
                        "execution_time_ms": execution_time,
                        "function_name": function_name,
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                manager.error(
                    f"Function {function_name} failed",
                    structured_data={
                        "execution_time_ms": execution_time,
                        "function_name": function_name,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                
                raise
                
        return wrapper
    return decorator


# Initialize on import
get_logging_manager()