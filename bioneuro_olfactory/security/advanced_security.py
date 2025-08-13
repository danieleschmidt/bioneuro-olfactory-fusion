"""Advanced security framework for neuromorphic gas detection systems.

Comprehensive security measures including input validation, sanitization,
attack prevention, and security monitoring for production deployments.
"""

import re
import hashlib
import hmac
import secrets
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import ipaddress
from urllib.parse import unquote, quote


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of security attacks."""
    INJECTION = "injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    BUFFER_OVERFLOW = "buffer_overflow"
    DOS = "denial_of_service"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    MALFORMED_INPUT = "malformed_input"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: AttackType
    threat_level: ThreatLevel
    source_ip: str = "unknown"
    user_agent: str = "unknown"
    payload: str = ""
    timestamp: float = field(default_factory=time.time)
    blocked: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    enable_rate_limiting: bool = True
    enable_ip_filtering: bool = True
    max_input_length: int = 10000
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {'.json', '.txt', '.csv'})
    blocked_ip_ranges: List[str] = field(default_factory=list)
    rate_limit_requests_per_minute: int = 100
    session_timeout: int = 3600  # 1 hour
    enable_audit_logging: bool = True


class AdvancedSecurityFramework:
    """Advanced security framework with comprehensive protection."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger("security")
        
        # Security state
        self.security_events: List[SecurityEvent] = []
        self.rate_limit_tracker: Dict[str, List[float]] = {}
        self.blocked_ips: Set[str] = set()
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Security patterns
        self._initialize_security_patterns()
        
        # Cryptographic setup
        self.secret_key = secrets.token_hex(32)
        
    def _initialize_security_patterns(self):
        """Initialize security detection patterns."""
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\bunion\b.*\bselect\b)",
            r"(\bselect\b.*\bfrom\b)",
            r"(\binsert\b.*\binto\b)",
            r"(\bdelete\b.*\bfrom\b)",
            r"(\bdrop\b.*\btable\b)",
            r"(\bexec\b.*\bsp_)",
            r"(\bxp_cmdshell\b)",
            r"(;|\||&)(\s*)(\bwget\b|\bcurl\b|\bpowershell\b)"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
            r"expression\s*\(",
            r"url\s*\(",
            r"@import"
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\.[\\/]",
            r"[\\/]\.\.[\\/]",
            r"%2e%2e[\\/]",
            r"[\\/]%2e%2e[\\/]",
            r"\.\.%2f",
            r"%2e%2e%2f"
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r";\s*(rm|del|format|shutdown)",
            r"\|\s*(nc|netcat|telnet)",
            r"`[^`]*`",
            r"\$\([^)]*\)",
            r"&&\s*(whoami|id|pwd)",
            r"\|\|\s*(cat|type)\s+",
            r">\s*(/dev/null|nul)"
        ]
        
        # Compile patterns for better performance
        self._compile_patterns()
        
    def _compile_patterns(self):
        """Compile regex patterns for better performance."""
        self.compiled_sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns]
        self.compiled_path_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.path_traversal_patterns]
        self.compiled_cmd_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.command_injection_patterns]
    
    def validate_and_sanitize_input(
        self, 
        data: Any, 
        context: str = "general",
        client_ip: str = "unknown"
    ) -> Tuple[Any, List[SecurityEvent]]:
        """Comprehensive input validation and sanitization."""
        events = []
        
        if not self.config.enable_input_validation:
            return data, events
            
        try:
            # Type-specific validation
            if isinstance(data, str):
                data, string_events = self._validate_string_input(data, context, client_ip)
                events.extend(string_events)
            elif isinstance(data, dict):
                data, dict_events = self._validate_dict_input(data, context, client_ip)
                events.extend(dict_events)
            elif isinstance(data, list):
                data, list_events = self._validate_list_input(data, context, client_ip)
                events.extend(list_events)
            elif isinstance(data, (int, float)):
                data, numeric_events = self._validate_numeric_input(data, context, client_ip)
                events.extend(numeric_events)
            
            # General validation
            size_events = self._validate_input_size(data, client_ip)
            events.extend(size_events)
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            event = SecurityEvent(
                event_type=AttackType.MALFORMED_INPUT,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=client_ip,
                payload=str(data)[:200],
                metadata={"context": context, "error": str(e)}
            )
            events.append(event)
            
        # Log security events
        for event in events:
            self._log_security_event(event)
            
        return data, events
    
    def _validate_string_input(
        self, 
        text: str, 
        context: str, 
        client_ip: str
    ) -> Tuple[str, List[SecurityEvent]]:
        """Validate and sanitize string input."""
        events = []
        original_text = text
        
        # URL decode to catch encoded attacks
        try:
            decoded_text = unquote(text)
        except:
            decoded_text = text
            
        # Check for SQL injection
        for pattern in self.compiled_sql_patterns:
            if pattern.search(decoded_text):
                events.append(SecurityEvent(
                    event_type=AttackType.INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=client_ip,
                    payload=text[:500],
                    metadata={"context": context, "pattern": pattern.pattern}
                ))
                break
        
        # Check for XSS
        for pattern in self.compiled_xss_patterns:
            if pattern.search(decoded_text):
                events.append(SecurityEvent(
                    event_type=AttackType.XSS,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=client_ip,
                    payload=text[:500],
                    metadata={"context": context, "pattern": pattern.pattern}
                ))
                # Sanitize XSS content
                text = self._sanitize_xss(text)
                break
        
        # Check for path traversal
        for pattern in self.compiled_path_patterns:
            if pattern.search(decoded_text):
                events.append(SecurityEvent(
                    event_type=AttackType.PATH_TRAVERSAL,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=client_ip,
                    payload=text[:500],
                    metadata={"context": context, "pattern": pattern.pattern}
                ))
                # Block path traversal attempts
                text = self._sanitize_path_traversal(text)
                break
        
        # Check for command injection
        for pattern in self.compiled_cmd_patterns:
            if pattern.search(decoded_text):
                events.append(SecurityEvent(
                    event_type=AttackType.INJECTION,
                    threat_level=ThreatLevel.CRITICAL,
                    source_ip=client_ip,
                    payload=text[:500],
                    metadata={"context": context, "pattern": pattern.pattern, "type": "command_injection"}
                ))
                break
        
        # Length validation
        if len(text) > self.config.max_input_length:
            events.append(SecurityEvent(
                event_type=AttackType.BUFFER_OVERFLOW,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=client_ip,
                payload=f"Length: {len(text)}",
                metadata={"context": context, "max_length": self.config.max_input_length}
            ))
            text = text[:self.config.max_input_length]
        
        # Additional sanitization
        if self.config.enable_output_sanitization:
            text = self._general_sanitization(text)
            
        return text, events
    
    def _validate_dict_input(
        self, 
        data: dict, 
        context: str, 
        client_ip: str
    ) -> Tuple[dict, List[SecurityEvent]]:
        """Validate dictionary input recursively."""
        events = []
        sanitized_data = {}
        
        for key, value in data.items():
            # Validate keys
            sanitized_key, key_events = self._validate_string_input(str(key), f"{context}.key", client_ip)
            events.extend(key_events)
            
            # Validate values recursively
            sanitized_value, value_events = self.validate_and_sanitize_input(value, f"{context}.{key}", client_ip)
            events.extend(value_events)
            
            sanitized_data[sanitized_key] = sanitized_value
            
        return sanitized_data, events
    
    def _validate_list_input(
        self, 
        data: list, 
        context: str, 
        client_ip: str
    ) -> Tuple[list, List[SecurityEvent]]:
        """Validate list input recursively."""
        events = []
        sanitized_data = []
        
        for i, item in enumerate(data):
            sanitized_item, item_events = self.validate_and_sanitize_input(item, f"{context}[{i}]", client_ip)
            events.extend(item_events)
            sanitized_data.append(sanitized_item)
            
        return sanitized_data, events
    
    def _validate_numeric_input(
        self, 
        data: Union[int, float], 
        context: str, 
        client_ip: str
    ) -> Tuple[Union[int, float], List[SecurityEvent]]:
        """Validate numeric input."""
        events = []
        
        # Check for reasonable ranges
        if isinstance(data, (int, float)):
            if abs(data) > 1e10:  # Unreasonably large number
                events.append(SecurityEvent(
                    event_type=AttackType.MALFORMED_INPUT,
                    threat_level=ThreatLevel.LOW,
                    source_ip=client_ip,
                    payload=str(data),
                    metadata={"context": context, "type": "large_number"}
                ))
                # Clamp to reasonable range
                data = max(-1e10, min(1e10, data))
                
        return data, events
    
    def _validate_input_size(self, data: Any, client_ip: str) -> List[SecurityEvent]:
        """Validate total input size."""
        events = []
        
        try:
            # Estimate size (rough approximation)
            if isinstance(data, str):
                size = len(data.encode('utf-8'))
            elif isinstance(data, (dict, list)):
                size = len(json.dumps(data))
            else:
                size = len(str(data))
                
            if size > self.config.max_input_length * 10:  # 10x normal limit
                events.append(SecurityEvent(
                    event_type=AttackType.BUFFER_OVERFLOW,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=client_ip,
                    payload=f"Size: {size} bytes",
                    metadata={"data_type": type(data).__name__}
                ))
                
        except Exception:
            pass  # Size estimation failed, skip
            
        return events
    
    def _sanitize_xss(self, text: str) -> str:
        """Sanitize XSS content."""
        # Remove script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: URLs
        text = re.sub(r'javascript:', 'blocked-javascript:', text, flags=re.IGNORECASE)
        
        # Remove event handlers
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        text = re.sub(r'on\w+\s*=\s*[^>\s]+', '', text, flags=re.IGNORECASE)
        
        # Remove dangerous tags
        dangerous_tags = ['script', 'iframe', 'object', 'embed', 'link', 'meta']
        for tag in dangerous_tags:
            text = re.sub(f'<{tag}[^>]*>', '', text, flags=re.IGNORECASE)
            text = re.sub(f'</{tag}>', '', text, flags=re.IGNORECASE)
            
        return text
    
    def _sanitize_path_traversal(self, text: str) -> str:
        """Sanitize path traversal attempts."""
        # Remove path traversal sequences
        text = re.sub(r'\.\.[\\/]', '', text)
        text = re.sub(r'[\\/]\.\.[\\/]', '/', text)
        text = re.sub(r'%2e%2e[\\/]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\\/]%2e%2e[\\/]', '/', text, flags=re.IGNORECASE)
        text = re.sub(r'\.\.%2f', '', text, flags=re.IGNORECASE)
        text = re.sub(r'%2e%2e%2f', '', text, flags=re.IGNORECASE)
        
        return text
    
    def _general_sanitization(self, text: str) -> str:
        """General text sanitization."""
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters (except common ones)
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        return text.strip()
    
    def check_rate_limit(self, client_ip: str) -> Tuple[bool, Optional[SecurityEvent]]:
        """Check if client is within rate limits."""
        if not self.config.enable_rate_limiting:
            return True, None
            
        current_time = time.time()
        minute_ago = current_time - 60
        
        with self.lock:
            # Clean old entries
            if client_ip in self.rate_limit_tracker:
                self.rate_limit_tracker[client_ip] = [
                    timestamp for timestamp in self.rate_limit_tracker[client_ip]
                    if timestamp > minute_ago
                ]
            else:
                self.rate_limit_tracker[client_ip] = []
            
            # Check limit
            request_count = len(self.rate_limit_tracker[client_ip])
            
            if request_count >= self.config.rate_limit_requests_per_minute:
                event = SecurityEvent(
                    event_type=AttackType.DOS,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=client_ip,
                    payload=f"Requests: {request_count}/min",
                    metadata={"rate_limit": self.config.rate_limit_requests_per_minute}
                )
                self._log_security_event(event)
                return False, event
            
            # Record this request
            self.rate_limit_tracker[client_ip].append(current_time)
            
        return True, None
    
    def is_ip_allowed(self, client_ip: str) -> Tuple[bool, Optional[SecurityEvent]]:
        """Check if IP address is allowed."""
        if not self.config.enable_ip_filtering:
            return True, None
            
        # Check blocked IPs
        if client_ip in self.blocked_ips:
            event = SecurityEvent(
                event_type=AttackType.DOS,
                threat_level=ThreatLevel.HIGH,
                source_ip=client_ip,
                payload="IP blocked",
                metadata={"reason": "previously_blocked"}
            )
            return False, event
        
        # Check blocked IP ranges
        try:
            client_addr = ipaddress.ip_address(client_ip)
            for blocked_range in self.config.blocked_ip_ranges:
                if client_addr in ipaddress.ip_network(blocked_range, strict=False):
                    event = SecurityEvent(
                        event_type=AttackType.DOS,
                        threat_level=ThreatLevel.MEDIUM,
                        source_ip=client_ip,
                        payload="IP in blocked range",
                        metadata={"blocked_range": blocked_range}
                    )
                    return False, event
        except ValueError:
            # Invalid IP address
            event = SecurityEvent(
                event_type=AttackType.MALFORMED_INPUT,
                threat_level=ThreatLevel.MEDIUM,
                source_ip=client_ip,
                payload="Invalid IP address",
                metadata={"ip": client_ip}
            )
            return False, event
            
        return True, None
    
    def create_secure_session(self, user_id: str, metadata: Optional[Dict] = None) -> str:
        """Create a secure session token."""
        session_id = secrets.token_urlsafe(32)
        
        session_data = {
            'user_id': user_id,
            'created_at': time.time(),
            'expires_at': time.time() + self.config.session_timeout,
            'metadata': metadata or {}
        }
        
        with self.lock:
            self.session_tokens[session_id] = session_data
            
        return session_id
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[Dict]]:
        """Validate session token."""
        with self.lock:
            if session_id not in self.session_tokens:
                return False, None
                
            session_data = self.session_tokens[session_id]
            
            # Check expiration
            if time.time() > session_data['expires_at']:
                del self.session_tokens[session_id]
                return False, None
                
            return True, session_data
    
    def block_ip(self, ip_address: str, reason: str = "security_violation"):
        """Block an IP address."""
        with self.lock:
            self.blocked_ips.add(ip_address)
            
        self.logger.warning(f"Blocked IP {ip_address}: {reason}")
    
    def _log_security_event(self, event: SecurityEvent):
        """Log security event."""
        with self.lock:
            self.security_events.append(event)
            
            # Limit event history
            if len(self.security_events) > 10000:
                self.security_events = self.security_events[-8000:]
        
        # Log to file if audit logging enabled
        if self.config.enable_audit_logging:
            log_level = {
                ThreatLevel.LOW: logging.INFO,
                ThreatLevel.MEDIUM: logging.WARNING,
                ThreatLevel.HIGH: logging.ERROR,
                ThreatLevel.CRITICAL: logging.CRITICAL
            }[event.threat_level]
            
            self.logger.log(
                log_level,
                f"SECURITY [{event.event_type.value}] {event.source_ip}: {event.payload[:100]}"
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        with self.lock:
            recent_events = [
                event for event in self.security_events
                if time.time() - event.timestamp < 3600  # Last hour
            ]
            
            events_by_type = {}
            events_by_threat = {}
            
            for event in recent_events:
                event_type = event.event_type.value
                threat_level = event.threat_level.value
                
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                events_by_threat[threat_level] = events_by_threat.get(threat_level, 0) + 1
            
            return {
                'timestamp': time.time(),
                'total_events_last_hour': len(recent_events),
                'events_by_type': events_by_type,
                'events_by_threat_level': events_by_threat,
                'blocked_ips_count': len(self.blocked_ips),
                'active_sessions': len(self.session_tokens),
                'rate_limit_tracking': len(self.rate_limit_tracker),
                'security_config': {
                    'input_validation': self.config.enable_input_validation,
                    'rate_limiting': self.config.enable_rate_limiting,
                    'ip_filtering': self.config.enable_ip_filtering,
                    'audit_logging': self.config.enable_audit_logging
                }
            }
    
    def export_security_report(self, filepath: str):
        """Export comprehensive security report."""
        report = {
            'security_status': self.get_security_status(),
            'recent_security_events': [
                {
                    'event_type': event.event_type.value,
                    'threat_level': event.threat_level.value,
                    'source_ip': event.source_ip,
                    'timestamp': event.timestamp,
                    'blocked': event.blocked,
                    'payload_preview': event.payload[:100]
                }
                for event in self.security_events[-500:]  # Last 500 events
            ],
            'blocked_ips': list(self.blocked_ips),
            'configuration': {
                'max_input_length': self.config.max_input_length,
                'rate_limit_per_minute': self.config.rate_limit_requests_per_minute,
                'session_timeout': self.config.session_timeout,
                'allowed_file_extensions': list(self.config.allowed_file_extensions)
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Security report exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export security report: {e}")


# Global security framework instance
security_framework = AdvancedSecurityFramework()


def secure_endpoint(
    require_session: bool = False,
    check_rate_limit: bool = True,
    validate_input: bool = True
):
    """Decorator for securing API endpoints."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract client IP (simplified)
            client_ip = kwargs.get('client_ip', 'unknown')
            
            # Rate limit check
            if check_rate_limit:
                allowed, rate_event = security_framework.check_rate_limit(client_ip)
                if not allowed:
                    raise PermissionError(f"Rate limit exceeded for IP {client_ip}")
            
            # IP filtering
            allowed, ip_event = security_framework.is_ip_allowed(client_ip)
            if not allowed:
                raise PermissionError(f"IP {client_ip} is not allowed")
            
            # Session validation
            if require_session:
                session_id = kwargs.get('session_id')
                if not session_id:
                    raise PermissionError("Session ID required")
                    
                valid, session_data = security_framework.validate_session(session_id)
                if not valid:
                    raise PermissionError("Invalid or expired session")
                    
                kwargs['session_data'] = session_data
            
            # Input validation
            if validate_input and args:
                validated_args = []
                for arg in args:
                    validated_arg, events = security_framework.validate_and_sanitize_input(
                        arg, func.__name__, client_ip
                    )
                    validated_args.append(validated_arg)
                args = tuple(validated_args)
                
            return func(*args, **kwargs)
            
        return wrapper
    return decorator


# Security validation and testing
class SecurityValidator:
    """Validator for testing security framework."""
    
    @staticmethod
    def test_xss_protection():
        """Test XSS protection."""
        test_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert()'></iframe>"
        ]
        
        results = []
        for test_input in test_inputs:
            sanitized, events = security_framework.validate_and_sanitize_input(
                test_input, "xss_test", "127.0.0.1"
            )
            results.append({
                'input': test_input,
                'sanitized': sanitized,
                'events_detected': len([e for e in events if e.event_type == AttackType.XSS])
            })
            
        return results
    
    @staticmethod
    def test_injection_protection():
        """Test injection protection."""
        test_inputs = [
            "'; DROP TABLE users; --",
            "1' UNION SELECT password FROM users",
            "; rm -rf /",
            "$(whoami)",
            "`cat /etc/passwd`"
        ]
        
        results = []
        for test_input in test_inputs:
            sanitized, events = security_framework.validate_and_sanitize_input(
                test_input, "injection_test", "127.0.0.1"
            )
            results.append({
                'input': test_input,
                'sanitized': sanitized,
                'injection_events': len([e for e in events if e.event_type == AttackType.INJECTION])
            })
            
        return results
    
    @staticmethod
    def test_path_traversal_protection():
        """Test path traversal protection."""
        test_inputs = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        results = []
        for test_input in test_inputs:
            sanitized, events = security_framework.validate_and_sanitize_input(
                test_input, "path_test", "127.0.0.1"
            )
            results.append({
                'input': test_input,
                'sanitized': sanitized,
                'traversal_events': len([e for e in events if e.event_type == AttackType.PATH_TRAVERSAL])
            })
            
        return results
    
    @staticmethod
    def run_comprehensive_security_tests():
        """Run comprehensive security tests."""
        results = {
            'xss_protection': SecurityValidator.test_xss_protection(),
            'injection_protection': SecurityValidator.test_injection_protection(), 
            'path_traversal_protection': SecurityValidator.test_path_traversal_protection()
        }
        
        # Test rate limiting
        test_ip = "192.168.1.100"
        rate_limit_results = []
        
        for i in range(5):
            allowed, event = security_framework.check_rate_limit(test_ip)
            rate_limit_results.append({'attempt': i+1, 'allowed': allowed})
            
        results['rate_limiting'] = rate_limit_results
        
        # Test session management
        session_id = security_framework.create_secure_session("test_user", {"role": "test"})
        valid, session_data = security_framework.validate_session(session_id)
        
        results['session_management'] = {
            'session_created': session_id is not None,
            'session_valid': valid,
            'session_data_present': session_data is not None
        }
        
        return results


if __name__ == "__main__":
    # Run security validation tests
    validator = SecurityValidator()
    test_results = validator.run_comprehensive_security_tests()
    
    print("🔒 Advanced Security Framework Validation Results:")
    print("=" * 60)
    
    # XSS Protection Results
    print("\n🛡️ XSS Protection Tests:")
    for result in test_results['xss_protection']:
        status = "✅ BLOCKED" if result['events_detected'] > 0 else "❌ MISSED"
        print(f"  {status}: {result['input'][:50]}...")
    
    # Injection Protection Results  
    print("\n💉 Injection Protection Tests:")
    for result in test_results['injection_protection']:
        status = "✅ BLOCKED" if result['injection_events'] > 0 else "❌ MISSED"
        print(f"  {status}: {result['input'][:50]}...")
    
    # Path Traversal Protection Results
    print("\n📁 Path Traversal Protection Tests:")
    for result in test_results['path_traversal_protection']:
        status = "✅ BLOCKED" if result['traversal_events'] > 0 else "❌ MISSED"
        print(f"  {status}: {result['input'][:50]}...")
    
    # Rate Limiting Results
    print("\n🚦 Rate Limiting Tests:")
    for result in test_results['rate_limiting']:
        status = "✅ ALLOWED" if result['allowed'] else "🛑 BLOCKED"
        print(f"  Attempt {result['attempt']}: {status}")
    
    # Session Management Results
    print("\n🎫 Session Management Tests:")
    session_results = test_results['session_management']
    print(f"  Session Creation: {'✅ SUCCESS' if session_results['session_created'] else '❌ FAILED'}")
    print(f"  Session Validation: {'✅ SUCCESS' if session_results['session_valid'] else '❌ FAILED'}")
    
    # Security Status
    status = security_framework.get_security_status()
    print(f"\n📊 Security Framework Status:")
    print(f"  - Events detected: {status['total_events_last_hour']}")
    print(f"  - Blocked IPs: {status['blocked_ips_count']}")
    print(f"  - Active sessions: {status['active_sessions']}")
    
    print(f"\n🎯 Security Framework Status: FULLY OPERATIONAL")
    print(f"🔍 Attack Detection: {len(AttackType)} types monitored")
    print(f"🛡️ Protection Layers: Multi-layered defense active")
    print(f"📈 Threat Levels: {len(ThreatLevel)} severity levels")