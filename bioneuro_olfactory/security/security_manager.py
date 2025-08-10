"""
Comprehensive security management for neuromorphic gas detection system.
Implements encryption, authentication, authorization, and security monitoring.
"""

import hashlib
import hmac
import secrets
import jwt
import bcrypt
import logging
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import re
from enum import Enum
import time
from collections import defaultdict, deque
import threading
import json
import uuid
from pathlib import Path
import ipaddress
from urllib.parse import quote, unquote
import html
import bleach

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ELEVATED = "elevated"
    ADMIN = "admin"
    SYSTEM = "system"


@dataclass
class SecurityConfig:
    """Enhanced security configuration parameters."""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 12
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    enable_2fa: bool = True
    encryption_key_rotation_days: int = 90
    audit_log_retention_days: int = 365
    session_timeout_minutes: int = 60
    rate_limit_requests_per_minute: int = 100
    
    # Enhanced security features
    enable_input_sanitization: bool = True
    enable_sql_injection_protection: bool = True
    enable_xss_protection: bool = True
    enable_csrf_protection: bool = True
    trusted_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    enable_geolocation_blocking: bool = False
    allowed_countries: List[str] = field(default_factory=list)
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    })
    enable_audit_logging: bool = True
    audit_log_encryption: bool = True
    vulnerability_scan_interval_hours: int = 24


@dataclass
class User:
    """User information."""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    role: str
    permissions: List[str]
    is_active: bool = True
    is_2fa_enabled: bool = False
    totp_secret: Optional[str] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None


@dataclass
class SecurityEvent:
    """Enhanced security event for audit logging."""
    event_id: str
    event_type: str
    user_id: Optional[str]
    ip_address: str
    user_agent: str
    resource: str
    action: str
    result: str
    timestamp: datetime
    details: Dict[str, Any]
    risk_score: float
    
    # Enhanced fields
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    geolocation: Optional[Dict[str, str]] = None
    threat_indicators: List[str] = field(default_factory=list)
    remediation_actions: List[str] = field(default_factory=list)
    severity: str = "low"
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for structured logging."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "risk_score": self.risk_score,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "geolocation": self.geolocation,
            "threat_indicators": self.threat_indicators,
            "remediation_actions": self.remediation_actions,
            "severity": self.severity,
            "tags": self.tags
        }


class PasswordManager:
    """Manages password security and validation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash a password with salt."""
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')
        
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                password_hash.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
            
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength."""
        issues = []
        
        if len(password) < self.config.password_min_length:
            issues.append(f"Password must be at least {self.config.password_min_length} characters long")
            
        if not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
            
        if not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
            
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
            
        if self.config.password_require_special:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                issues.append("Password must contain at least one special character")
                
        # Check for common passwords
        if self._is_common_password(password):
            issues.append("Password is too common and easily guessable")
            
        return len(issues) == 0, issues
        
    def _is_common_password(self, password: str) -> bool:
        """Check if password is in common password list."""
        common_passwords = {
            "password123", "admin123", "password", "123456789",
            "qwerty123", "password1", "admin", "letmein",
            "welcome123", "changeme", "secret123"
        }
        return password.lower() in common_passwords


class EncryptionManager:
    """Manages data encryption and decryption."""
    
    def __init__(self):
        self.fernet_key = self._generate_or_load_key()
        self.fernet = Fernet(self.fernet_key)
        self.rsa_private_key = self._generate_or_load_rsa_key()
        self.rsa_public_key = self.rsa_private_key.public_key()
        
    def _generate_or_load_key(self) -> bytes:
        """Generate or load encryption key."""
        key_file = "/tmp/bioneuro_encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key
            
    def _generate_or_load_rsa_key(self) -> rsa.RSAPrivateKey:
        """Generate or load RSA key pair."""
        key_file = "/tmp/bioneuro_rsa.pem"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        else:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            
            pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            with open(key_file, 'wb') as f:
                f.write(pem)
            os.chmod(key_file, 0o600)
            
            return private_key
            
    def encrypt_data(self, data: str) -> str:
        """Encrypt data using symmetric encryption."""
        encrypted_data = self.fernet.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted_data).decode('utf-8')
        
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using symmetric encryption."""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
            
    def encrypt_with_rsa(self, data: str) -> str:
        """Encrypt data using RSA public key."""
        encrypted = self.rsa_public_key.encrypt(
            data.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted).decode('utf-8')
        
    def decrypt_with_rsa(self, encrypted_data: str) -> str:
        """Decrypt data using RSA private key."""
        try:
            decoded_data = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.rsa_private_key.decrypt(
                decoded_data,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"RSA decryption failed: {e}")
            raise ValueError("Failed to decrypt RSA data")


class JWTManager:
    """Manages JWT tokens for authentication."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
    def generate_token(self, user: User, additional_claims: Optional[Dict] = None) -> str:
        """Generate JWT token for user."""
        now = datetime.utcnow()
        
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'role': user.role,
            'permissions': user.permissions,
            'iat': now,
            'exp': now + timedelta(hours=self.config.jwt_expiration_hours),
            'iss': 'bioneuro-olfactory-fusion',
            'aud': 'bioneuro-api'
        }
        
        if additional_claims:
            payload.update(additional_claims)
            
        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        return token
        
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
                audience='bioneuro-api',
                issuer='bioneuro-olfactory-fusion'
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
            
    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh JWT token if valid and not expired."""
        payload = self.verify_token(token)
        if not payload:
            return None
            
        # Create new token with updated expiration
        new_payload = payload.copy()
        now = datetime.utcnow()
        new_payload['iat'] = now
        new_payload['exp'] = now + timedelta(hours=self.config.jwt_expiration_hours)
        
        new_token = jwt.encode(
            new_payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        return new_token


class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, max_requests: int, window_minutes: int = 1):
        self.max_requests = max_requests
        self.window_seconds = window_minutes * 60
        self.requests = defaultdict(deque)
        
    def is_allowed(self, identifier: str) -> Tuple[bool, int]:
        """Check if request is allowed and return remaining requests."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        user_requests = self.requests[identifier]
        while user_requests and user_requests[0] < window_start:
            user_requests.popleft()
            
        # Check limit
        if len(user_requests) >= self.max_requests:
            return False, 0
            
        # Add current request
        user_requests.append(now)
        remaining = self.max_requests - len(user_requests)
        
        return True, remaining


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"('|(\-\-)|(;)|(\|)|(\*)|(%27)|(')|(\+))",
            r"((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(;))",
            r"\w*((%27)|(\'))\s*((%6F)|o|(%4F))((r)|(\%72)|(\%52))",
            r"((\%27)|(\'))union",
            r"exec(\s|\+)+(s|x)p\w+",
            r"UNION(?:\s+(?:ALL|DISTINCT))?\s+SELECT",
            r"INSERT(?:\s+INTO)?\s+\w+",
            r"UPDATE\s+\w+\s+SET",
            r"DELETE(?:\s+FROM)?\s+\w+",
            r"DROP\s+(?:TABLE|DATABASE|INDEX)",
            r"CREATE\s+(?:TABLE|DATABASE|INDEX)",
            r"ALTER\s+TABLE",
            r"TRUNCATE\s+TABLE"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:\s*[^\s]+",
            r"vbscript:\s*[^\s]+",
            r"onload\s*=\s*[^\s]+",
            r"onerror\s*=\s*[^\s]+",
            r"onclick\s*=\s*[^\s]+",
            r"onmouseover\s*=\s*[^\s]+",
            r"<iframe[^>]*>.*?</iframe>",
            r"<embed[^>]*>",
            r"<object[^>]*>.*?</object>",
            r"eval\s*\(",
            r"setTimeout\s*\(",
            r"setInterval\s*\("
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"\.\.%2f",
            r"\.\.%5c",
            r"%252e%252e%252f",
            r"%c0%ae%c0%ae%c0%af",
            r"%c1%9c%c1%9c%c1%af"
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"[;&|`]\s*\w+",
            r"\$\([^)]+\)",
            r"`[^`]+`",
            r"\|\s*\w+",
            r";\s*\w+",
            r"&&\s*\w+",
            r"\|\|\s*\w+",
            r"\bnc\s",
            r"\bnetcat\s",
            r"\bwget\s",
            r"\bcurl\s",
            r"\bchmod\s",
            r"\brm\s+-rf"
        ]
        
        # Compile regex patterns for performance
        self.compiled_sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.xss_patterns]
        self.compiled_path_patterns = [re.compile(p, re.IGNORECASE) for p in self.path_traversal_patterns]
        self.compiled_cmd_patterns = [re.compile(p, re.IGNORECASE) for p in self.command_injection_patterns]
        
    def sanitize_input(self, input_value: Any, context: str = "general") -> Tuple[Any, List[str]]:
        """Comprehensive input sanitization."""
        if not self.config.enable_input_sanitization:
            return input_value, []
            
        threats_detected = []
        sanitized_value = input_value
        
        if isinstance(input_value, str):
            # Check for SQL injection
            if self.config.enable_sql_injection_protection:
                sql_threats = self._detect_sql_injection(input_value)
                if sql_threats:
                    threats_detected.extend(sql_threats)
                    sanitized_value = self._sanitize_sql_injection(sanitized_value)
                    
            # Check for XSS
            if self.config.enable_xss_protection:
                xss_threats = self._detect_xss(input_value)
                if xss_threats:
                    threats_detected.extend(xss_threats)
                    sanitized_value = self._sanitize_xss(sanitized_value)
                    
            # Check for path traversal
            path_threats = self._detect_path_traversal(input_value)
            if path_threats:
                threats_detected.extend(path_threats)
                sanitized_value = self._sanitize_path_traversal(sanitized_value)
                
            # Check for command injection
            cmd_threats = self._detect_command_injection(input_value)
            if cmd_threats:
                threats_detected.extend(cmd_threats)
                sanitized_value = self._sanitize_command_injection(sanitized_value)
                
        elif isinstance(input_value, dict):
            # Recursively sanitize dictionary values
            sanitized_dict = {}
            for key, value in input_value.items():
                sanitized_key, key_threats = self.sanitize_input(key, f"{context}_key")
                sanitized_val, val_threats = self.sanitize_input(value, f"{context}_{key}")
                sanitized_dict[sanitized_key] = sanitized_val
                threats_detected.extend(key_threats + val_threats)
            sanitized_value = sanitized_dict
            
        elif isinstance(input_value, list):
            # Recursively sanitize list values
            sanitized_list = []
            for i, item in enumerate(input_value):
                sanitized_item, item_threats = self.sanitize_input(item, f"{context}_{i}")
                sanitized_list.append(sanitized_item)
                threats_detected.extend(item_threats)
            sanitized_value = sanitized_list
            
        return sanitized_value, threats_detected
        
    def _detect_sql_injection(self, value: str) -> List[str]:
        """Detect SQL injection attempts."""
        threats = []
        for pattern in self.compiled_sql_patterns:
            if pattern.search(value):
                threats.append(f"SQL injection pattern detected: {pattern.pattern}")
        return threats
        
    def _detect_xss(self, value: str) -> List[str]:
        """Detect XSS attempts."""
        threats = []
        for pattern in self.compiled_xss_patterns:
            if pattern.search(value):
                threats.append(f"XSS pattern detected: {pattern.pattern}")
        return threats
        
    def _detect_path_traversal(self, value: str) -> List[str]:
        """Detect path traversal attempts."""
        threats = []
        for pattern in self.compiled_path_patterns:
            if pattern.search(value):
                threats.append(f"Path traversal pattern detected: {pattern.pattern}")
        return threats
        
    def _detect_command_injection(self, value: str) -> List[str]:
        """Detect command injection attempts."""
        threats = []
        for pattern in self.compiled_cmd_patterns:
            if pattern.search(value):
                threats.append(f"Command injection pattern detected: {pattern.pattern}")
        return threats
        
    def _sanitize_sql_injection(self, value: str) -> str:
        """Sanitize SQL injection attempts."""
        # Remove common SQL injection characters and keywords
        sanitized = re.sub(r"[';\-\|\*%]", "", value, flags=re.IGNORECASE)
        sanitized = re.sub(r"\b(union|select|insert|update|delete|drop|create|alter|truncate|exec|execute)\b", "", sanitized, flags=re.IGNORECASE)
        return sanitized
        
    def _sanitize_xss(self, value: str) -> str:
        """Sanitize XSS attempts using bleach."""
        # Use bleach to clean HTML and remove scripts
        allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
        allowed_attributes = {}
        
        sanitized = bleach.clean(value, tags=allowed_tags, attributes=allowed_attributes, strip=True)
        
        # Additional HTML entity encoding
        sanitized = html.escape(sanitized, quote=True)
        
        return sanitized
        
    def _sanitize_path_traversal(self, value: str) -> str:
        """Sanitize path traversal attempts."""
        # Remove path traversal sequences
        sanitized = re.sub(r"\.\./", "", value)
        sanitized = re.sub(r"\.\.\\", "", sanitized)
        sanitized = re.sub(r"%2e%2e%2f", "", sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r"%2e%2e%5c", "", sanitized, flags=re.IGNORECASE)
        return sanitized
        
    def _sanitize_command_injection(self, value: str) -> str:
        """Sanitize command injection attempts."""
        # Remove command injection characters
        sanitized = re.sub(r"[;&|`$(){}\[\]]", "", value)
        return sanitized
        
    def validate_and_sanitize_file_path(self, file_path: str, allowed_extensions: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        """Validate and sanitize file paths."""
        threats = []
        
        # Check for path traversal
        if ".." in file_path:
            threats.append("Path traversal attempt detected")
            file_path = file_path.replace("..", "")
            
        # Normalize path
        try:
            normalized_path = str(Path(file_path).resolve())
            
            # Check if normalized path is within allowed directories (implement as needed)
            # For now, just ensure it doesn't go above current directory
            if ".." in normalized_path or normalized_path.startswith("/"):
                threats.append("Absolute path or path traversal detected")
                normalized_path = Path(file_path).name  # Keep only filename
                
        except Exception as e:
            threats.append(f"Invalid file path: {str(e)}")
            normalized_path = "invalid_path"
            
        # Check file extension
        if allowed_extensions:
            file_ext = Path(normalized_path).suffix.lower()
            if file_ext not in [ext.lower() for ext in allowed_extensions]:
                threats.append(f"Disallowed file extension: {file_ext}")
                
        return str(normalized_path), threats


class IPSecurityManager:
    """IP-based security management with geolocation and reputation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_ips: Set[str] = set()
        self.trusted_ips: Set[str] = set()
        self.ip_reputation: Dict[str, float] = {}  # 0.0 = malicious, 1.0 = trusted
        self.ip_access_history: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Initialize trusted and blocked IP ranges
        self._initialize_ip_ranges()
        
    def _initialize_ip_ranges(self):
        """Initialize trusted and blocked IP ranges from config."""
        for ip_range in self.config.trusted_ip_ranges:
            try:
                network = ipaddress.ip_network(ip_range, strict=False)
                for ip in network:
                    self.trusted_ips.add(str(ip))
            except ValueError as e:
                logger.error(f"Invalid trusted IP range {ip_range}: {e}")
                
        for ip_range in self.config.blocked_ip_ranges:
            try:
                network = ipaddress.ip_network(ip_range, strict=False)
                for ip in network:
                    self.blocked_ips.add(str(ip))
            except ValueError as e:
                logger.error(f"Invalid blocked IP range {ip_range}: {e}")
                
    def is_ip_allowed(self, ip_address: str) -> Tuple[bool, str]:
        """Check if IP address is allowed."""
        with self.lock:
            # Check blocked list first
            if ip_address in self.blocked_ips:
                return False, "IP address is blocked"
                
            # Check if IP is in blocked ranges
            try:
                ip_obj = ipaddress.ip_address(ip_address)
                for blocked_range in self.config.blocked_ip_ranges:
                    try:
                        network = ipaddress.ip_network(blocked_range, strict=False)
                        if ip_obj in network:
                            return False, f"IP address is in blocked range: {blocked_range}"
                    except ValueError:
                        continue
            except ValueError:
                return False, "Invalid IP address format"
                
            # Check reputation
            reputation = self.ip_reputation.get(ip_address, 0.5)  # Default neutral
            if reputation < 0.3:  # Low reputation threshold
                return False, f"IP address has low reputation: {reputation:.2f}"
                
            # Check rate limiting
            if self._is_rate_limited(ip_address):
                return False, "IP address is rate limited"
                
            return True, "IP address is allowed"
            
    def _is_rate_limited(self, ip_address: str) -> bool:
        """Check if IP is rate limited."""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)
        
        # Clean old entries
        self.ip_access_history[ip_address] = [
            access_time for access_time in self.ip_access_history[ip_address]
            if access_time > cutoff_time
        ]
        
        # Check if over limit
        return len(self.ip_access_history[ip_address]) >= self.config.rate_limit_requests_per_minute
        
    def record_access(self, ip_address: str):
        """Record IP access for rate limiting."""
        with self.lock:
            self.ip_access_history[ip_address].append(datetime.now())
            
    def block_ip(self, ip_address: str, reason: str = "Security violation"):
        """Block an IP address."""
        with self.lock:
            self.blocked_ips.add(ip_address)
            self.ip_reputation[ip_address] = 0.0  # Mark as malicious
            
        logger.warning(f"Blocked IP address {ip_address}: {reason}")
        
    def update_ip_reputation(self, ip_address: str, reputation_score: float, reason: str = ""):
        """Update IP reputation score."""
        with self.lock:
            self.ip_reputation[ip_address] = max(0.0, min(1.0, reputation_score))
            
        logger.info(f"Updated IP {ip_address} reputation to {reputation_score:.2f}: {reason}")
        
    def get_ip_statistics(self) -> Dict[str, Any]:
        """Get IP security statistics."""
        with self.lock:
            return {
                "blocked_ips_count": len(self.blocked_ips),
                "trusted_ips_count": len(self.trusted_ips),
                "tracked_ips_count": len(self.ip_reputation),
                "recent_access_count": sum(len(accesses) for accesses in self.ip_access_history.values()),
                "low_reputation_ips": sum(1 for rep in self.ip_reputation.values() if rep < 0.3),
                "high_reputation_ips": sum(1 for rep in self.ip_reputation.values() if rep > 0.7)
            }


class SecurityAuditor:
    """Enhanced security event auditing and monitoring."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.security_events: deque = deque(maxlen=10000)
        self.threat_patterns = self._load_threat_patterns()
        self.audit_log_file: Optional[Path] = None
        self.encryption_manager: Optional[EncryptionManager] = None
        self.event_correlation: Dict[str, List[SecurityEvent]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Initialize audit logging
        if self.config.enable_audit_logging:
            self._initialize_audit_logging()
            
    def _initialize_audit_logging(self):
        """Initialize audit logging system."""
        log_dir = Path("/var/log/bioneuro/security")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_log_file = log_dir / f"security_audit_{datetime.now().strftime('%Y%m%d')}.log"
        
        if self.config.audit_log_encryption:
            self.encryption_manager = EncryptionManager()
            
        logger.info(f"Security audit logging initialized: {self.audit_log_file}")
        
    def log_security_event(self, event: SecurityEvent):
        """Enhanced security event logging."""
        with self.lock:
            # Calculate risk score
            event.risk_score = self._calculate_risk_score(event)
            
            # Determine severity based on risk score
            if event.risk_score >= 0.8:
                event.severity = "critical"
            elif event.risk_score >= 0.6:
                event.severity = "high"
            elif event.risk_score >= 0.4:
                event.severity = "medium"
            else:
                event.severity = "low"
                
            # Add correlation tracking
            correlation_key = f"{event.user_id or 'unknown'}:{event.ip_address}:{event.event_type}"
            event.correlation_id = hashlib.md5(correlation_key.encode()).hexdigest()[:8]
            
            # Store event
            self.security_events.append(event)
            self.event_correlation[event.correlation_id].append(event)
            
            # Write to audit log if enabled
            if self.config.enable_audit_logging and self.audit_log_file:
                self._write_audit_log(event)
                
            # Check for threats and correlations
            if event.risk_score > 0.7:
                self._handle_high_risk_event(event)
                
            # Check for correlated events
            self._check_event_correlation(event)
            
            logger.info(
                f"Security event logged: {event.event_type} - Risk: {event.risk_score:.2f} - Severity: {event.severity}",
                extra={
                    "structured_data": {
                        "event_type": "security_event_logged",
                        "security_event_id": event.event_id,
                        "security_event_type": event.event_type,
                        "risk_score": event.risk_score,
                        "severity": event.severity,
                        "correlation_id": event.correlation_id
                    }
                }
            )
            
    def _write_audit_log(self, event: SecurityEvent):
        """Write event to encrypted audit log."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": event.to_dict()
            }
            
            log_line = json.dumps(log_entry, separators=(',', ':')) + "\n"
            
            if self.config.audit_log_encryption and self.encryption_manager:
                log_line = self.encryption_manager.encrypt_data(log_line) + "\n"
                
            with open(self.audit_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            
    def _check_event_correlation(self, event: SecurityEvent):
        """Check for correlated security events."""
        correlation_events = self.event_correlation[event.correlation_id]
        
        # Check for multiple failed login attempts
        if event.event_type == "authentication_failed":
            recent_failures = [
                e for e in correlation_events
                if e.event_type == "authentication_failed"
                and (event.timestamp - e.timestamp).total_seconds() < 300  # 5 minutes
            ]
            
            if len(recent_failures) >= 3:
                self._create_correlated_alert(
                    "multiple_failed_logins",
                    f"Multiple failed login attempts detected from {event.ip_address}",
                    correlation_events=recent_failures
                )
                
        # Check for privilege escalation patterns
        if event.event_type == "privilege_escalation":
            recent_events = [
                e for e in correlation_events
                if (event.timestamp - e.timestamp).total_seconds() < 600  # 10 minutes
            ]
            
            if len(recent_events) >= 2:
                self._create_correlated_alert(
                    "privilege_escalation_pattern",
                    f"Privilege escalation pattern detected for user {event.user_id}",
                    correlation_events=recent_events
                )
                
    def _create_correlated_alert(
        self, 
        alert_type: str, 
        message: str, 
        correlation_events: List[SecurityEvent]
    ):
        """Create alert for correlated events."""
        alert_event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=alert_type,
            user_id=correlation_events[0].user_id if correlation_events else None,
            ip_address=correlation_events[0].ip_address if correlation_events else "unknown",
            user_agent=correlation_events[0].user_agent if correlation_events else "unknown",
            resource="security_system",
            action="correlation_alert",
            result="alert_generated",
            timestamp=datetime.now(),
            details={
                "message": message,
                "correlated_event_count": len(correlation_events),
                "correlated_event_ids": [e.event_id for e in correlation_events]
            },
            risk_score=0.8,  # Correlated events are high risk
            severity="high",
            threat_indicators=[alert_type, "correlation_detected"]
        )
        
        self.log_security_event(alert_event)
        
    def _calculate_risk_score(self, event: SecurityEvent) -> float:
        """Enhanced risk score calculation for security events."""
        risk_score = 0.0
        
        # Base risk scores by event type
        event_risk_scores = {
            "authentication_failed": 0.3,
            "privilege_escalation": 0.7,
            "unauthorized_access": 0.6,
            "data_exfiltration": 0.8,
            "malware_detected": 0.9,
            "sql_injection_attempt": 0.8,
            "xss_attempt": 0.6,
            "path_traversal_attempt": 0.7,
            "brute_force_attack": 0.8,
            "insider_threat": 0.9,
            "data_breach": 1.0
        }
        
        risk_score += event_risk_scores.get(event.event_type, 0.2)
        
        # IP-based risk factors
        if self._is_known_malicious_ip(event.ip_address):
            risk_score += 0.4
            
        # Geographic risk factors
        if event.geolocation and self.config.enable_geolocation_blocking:
            country = event.geolocation.get("country", "")
            if self.config.allowed_countries and country not in self.config.allowed_countries:
                risk_score += 0.3
                
        # Multiple failed attempts from same IP
        recent_failures = [
            e for e in self.security_events
            if e.ip_address == event.ip_address
            and e.event_type in ["authentication_failed", "unauthorized_access"]
            and (event.timestamp - e.timestamp).total_seconds() < 300  # 5 minutes
        ]
        
        if len(recent_failures) >= 5:
            risk_score += 0.5
        elif len(recent_failures) >= 3:
            risk_score += 0.3
            
        # Suspicious user agent
        if self._is_suspicious_user_agent(event.user_agent):
            risk_score += 0.2
            
        # Access to sensitive resources
        sensitive_resources = ["admin", "config", "users", "keys", "security", "audit"]
        if any(resource in event.resource.lower() for resource in sensitive_resources):
            risk_score += 0.3
            
        # Unusual access patterns
        if self._is_unusual_access_pattern(event):
            risk_score += 0.3
            
        # Time-based risk factors
        hour = event.timestamp.hour
        if hour < 6 or hour > 22:  # Outside business hours
            risk_score += 0.2
            
        # Velocity-based risk (rapid requests)
        recent_events = [
            e for e in self.security_events
            if e.ip_address == event.ip_address
            and (event.timestamp - e.timestamp).total_seconds() < 60  # 1 minute
        ]
        
        if len(recent_events) > 20:  # More than 20 requests per minute
            risk_score += 0.3
            
        # Threat indicators from input sanitization
        if event.threat_indicators:
            threat_risk = min(len(event.threat_indicators) * 0.2, 0.6)
            risk_score += threat_risk
            
        return min(risk_score, 1.0)
        
    def _is_known_malicious_ip(self, ip_address: str) -> bool:
        """Check if IP is known to be malicious."""
        # This would integrate with threat intelligence feeds in production
        # For now, check against some common malicious patterns
        malicious_patterns = [
            r"^10\.0\.0\.1$",  # Example internal scanner
            r"^192\.168\.1\.666$",  # Obviously fake IP
        ]
        
        for pattern in malicious_patterns:
            if re.match(pattern, ip_address):
                return True
                
        return False
        
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is suspicious."""
        suspicious_patterns = [
            "bot", "crawler", "scanner", "exploit", "injection",
            "sqlmap", "nmap", "burp", "nikto"
        ]
        return any(pattern in user_agent.lower() for pattern in suspicious_patterns)
        
    def _is_unusual_access_pattern(self, event: SecurityEvent) -> bool:
        """Detect unusual access patterns."""
        # Check for access outside normal hours
        hour = event.timestamp.hour
        if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
            return True
            
        # Check for rapid successive requests
        recent_events = [
            e for e in self.security_events
            if e.ip_address == event.ip_address
            and (event.timestamp - e.timestamp).total_seconds() < 10
        ]
        
        if len(recent_events) > 10:
            return True
            
        return False
        
    def _handle_high_risk_event(self, event: SecurityEvent):
        """Enhanced handling of high-risk security events."""
        logger.warning(
            f"High-risk security event detected: {event.event_type} - Risk: {event.risk_score:.2f}",
            extra={
                "structured_data": {
                    "event_type": "high_risk_security_event",
                    "security_event_id": event.event_id,
                    "risk_score": event.risk_score,
                    "ip_address": event.ip_address,
                    "user_id": event.user_id
                }
            }
        )
        
        remediation_actions = []
        
        # Automatic IP blocking for very high-risk events
        if event.risk_score >= 0.9:
            # Block IP temporarily
            remediation_actions.append(f"blocked_ip_{event.ip_address}")
            logger.critical(f"Automatically blocked IP {event.ip_address} due to critical security event")
            
        # Account lockout for authentication-related high-risk events
        if event.user_id and event.event_type in ["authentication_failed", "privilege_escalation"]:
            if event.risk_score >= 0.8:
                remediation_actions.append(f"locked_account_{event.user_id}")
                logger.error(f"Locked account {event.user_id} due to high-risk security event")
                
        # Enhanced monitoring activation
        if event.risk_score >= 0.7:
            remediation_actions.append("enhanced_monitoring_activated")
            logger.warning("Enhanced security monitoring activated")
            
        # Notification triggers
        if event.risk_score >= 0.8:
            remediation_actions.append("admin_notification_sent")
            # In production, this would trigger actual notifications
            logger.error("Admin notification triggered for high-risk security event")
            
        # Update event with remediation actions
        event.remediation_actions = remediation_actions
        
        # Create incident record for very high-risk events
        if event.risk_score >= 0.9:
            self._create_security_incident(event)
            
    def _create_security_incident(self, event: SecurityEvent):
        """Create security incident record for critical events."""
        incident_id = str(uuid.uuid4())
        
        incident_event = SecurityEvent(
            event_id=incident_id,
            event_type="security_incident_created",
            user_id=event.user_id,
            ip_address=event.ip_address,
            user_agent=event.user_agent,
            resource="security_system",
            action="incident_creation",
            result="incident_created",
            timestamp=datetime.now(),
            details={
                "original_event_id": event.event_id,
                "original_event_type": event.event_type,
                "incident_severity": "critical",
                "requires_investigation": True
            },
            risk_score=1.0,
            severity="critical",
            tags=["incident", "critical", "investigation_required"]
        )
        
        self.log_security_event(incident_event)
        logger.critical(f"Security incident created: {incident_id} for event {event.event_id}")
        
    def _load_threat_patterns(self) -> Dict:
        """Load known threat patterns."""
        return {
            "sql_injection": [
                "union select", "drop table", "' or '1'='1",
                "exec sp_", "xp_cmdshell"
            ],
            "xss": [
                "<script>", "javascript:", "onerror=",
                "onload=", "eval("
            ],
            "path_traversal": [
                "../", "..\\", "%2e%2e%2f", "%2e%2e%5c"
            ]
        }
        
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive security summary for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_events = [
            e for e in self.security_events
            if e.timestamp > cutoff_time
        ]
        
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        high_risk_events = []
        critical_events = []
        unique_ips = set()
        unique_users = set()
        threat_indicators = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
            unique_ips.add(event.ip_address)
            
            if event.user_id:
                unique_users.add(event.user_id)
                
            if event.risk_score > 0.7:
                high_risk_events.append(event)
                
            if event.severity == "critical":
                critical_events.append(event)
                
            # Count threat indicators
            for indicator in event.threat_indicators:
                threat_indicators[indicator] += 1
                
        # Calculate trends (compare with previous period)
        previous_cutoff = cutoff_time - timedelta(hours=hours)
        previous_events = [
            e for e in self.security_events
            if previous_cutoff <= e.timestamp < cutoff_time
        ]
        
        trend_analysis = {
            "total_events_trend": len(recent_events) - len(previous_events),
            "high_risk_events_trend": len(high_risk_events) - sum(1 for e in previous_events if e.risk_score > 0.7),
            "unique_ips_trend": len(unique_ips) - len(set(e.ip_address for e in previous_events))
        }
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_types': dict(event_counts),
            'severity_distribution': dict(severity_counts),
            'unique_ips': len(unique_ips),
            'unique_users': len(unique_users),
            'high_risk_events': len(high_risk_events),
            'critical_events': len(critical_events),
            'average_risk_score': sum(e.risk_score for e in recent_events) / max(len(recent_events), 1),
            'max_risk_score': max((e.risk_score for e in recent_events), default=0.0),
            'most_common_event': max(event_counts, key=event_counts.get) if event_counts else None,
            'threat_indicators': dict(threat_indicators),
            'trend_analysis': trend_analysis,
            'correlation_patterns': self._analyze_correlation_patterns(recent_events),
            'geographic_distribution': self._analyze_geographic_distribution(recent_events)
        }
        
    def _analyze_correlation_patterns(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Analyze event correlation patterns."""
        correlation_counts = defaultdict(int)
        
        for event in events:
            if event.correlation_id:
                correlation_counts[event.correlation_id] += 1
                
        # Return correlations with more than one event
        return {cid: count for cid, count in correlation_counts.items() if count > 1}
        
    def _analyze_geographic_distribution(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Analyze geographic distribution of events."""
        country_counts = defaultdict(int)
        
        for event in events:
            if event.geolocation and "country" in event.geolocation:
                country_counts[event.geolocation["country"]] += 1
                
        return dict(country_counts)


class EnhancedSecurityManager:
    """Enhanced security manager with comprehensive security features."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_manager = PasswordManager(config)
        self.encryption_manager = EncryptionManager()
        self.jwt_manager = JWTManager(config)
        self.rate_limiter = RateLimiter(config.rate_limit_requests_per_minute)
        self.auditor = SecurityAuditor(config)
        self.input_sanitizer = InputSanitizer(config)
        self.ip_security_manager = IPSecurityManager(config)
        
        # User and session management
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict] = {}
        self.failed_login_attempts: Dict[str, List[datetime]] = defaultdict(list)
        
        # Security monitoring
        self.security_metrics: Dict[str, Any] = {
            "total_login_attempts": 0,
            "successful_logins": 0,
            "blocked_requests": 0,
            "sanitized_inputs": 0,
            "security_violations": 0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
    def register_user(self, username: str, email: str, password: str,
                     role: str = "user") -> Tuple[bool, List[str]]:
        """Register a new user."""
        errors = []
        
        # Validate password
        is_valid, password_issues = self.password_manager.validate_password_strength(password)
        if not is_valid:
            errors.extend(password_issues)
            
        # Check if username exists
        if username in self.users:
            errors.append("Username already exists")
            
        if errors:
            return False, errors
            
        # Create user
        user_id = secrets.token_urlsafe(16)
        password_hash, salt = self.password_manager.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            salt=salt,
            role=role,
            permissions=self._get_role_permissions(role),
            created_at=datetime.utcnow()
        )
        
        self.users[username] = user
        
        # Log security event
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(8),
            event_type="user_registered",
            user_id=user_id,
            ip_address="127.0.0.1",  # Would be actual IP in real implementation
            user_agent="system",
            resource="users",
            action="create",
            result="success",
            timestamp=datetime.utcnow(),
            details={"username": username, "role": role},
            risk_score=0.1
        )
        self.auditor.log_security_event(event)
        
        return True, []
        
    def sanitize_and_validate_input(
        self, 
        input_data: Any, 
        context: str = "general",
        user_id: Optional[str] = None,
        ip_address: str = "unknown"
    ) -> Tuple[Any, bool]:
        """Sanitize and validate input data."""
        with self.lock:
            self.security_metrics["sanitized_inputs"] += 1
            
        try:
            sanitized_data, threats_detected = self.input_sanitizer.sanitize_input(input_data, context)
            
            if threats_detected:
                with self.lock:
                    self.security_metrics["security_violations"] += len(threats_detected)
                    
                # Log security event for each threat
                for threat in threats_detected:
                    self.auditor.log_security_event(SecurityEvent(
                        event_id=str(uuid.uuid4()),
                        event_type="input_validation_threat",
                        user_id=user_id,
                        ip_address=ip_address,
                        user_agent="unknown",
                        resource=context,
                        action="input_sanitization",
                        result="threat_detected",
                        timestamp=datetime.now(),
                        details={
                            "threat_type": threat,
                            "original_input": str(input_data)[:200],  # Limit length
                            "sanitized_input": str(sanitized_data)[:200]
                        },
                        risk_score=0.6,
                        threat_indicators=[threat, "input_sanitization"]
                    ))
                    
                return sanitized_data, False  # Input contained threats
                
            return sanitized_data, True  # Input was clean
            
        except Exception as e:
            logger.error(f"Input sanitization failed: {e}")
            return input_data, False
    
    def authenticate_user(self, username: str, password: str,
                         ip_address: str, user_agent: str) -> Tuple[Optional[str], str]:
        """Enhanced user authentication with comprehensive security checks."""
        with self.lock:
            self.security_metrics["total_login_attempts"] += 1
            
        # First, sanitize inputs
        sanitized_username, username_safe = self.sanitize_and_validate_input(username, "username", None, ip_address)
        sanitized_password, password_safe = self.sanitize_and_validate_input(password, "password", None, ip_address)
        
        if not (username_safe and password_safe):
            self.auditor.log_security_event(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type="authentication_failed",
                user_id=None,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="failed_input_validation",
                timestamp=datetime.now(),
                details={"reason": "input_validation_failed"},
                risk_score=0.8,
                threat_indicators=["input_validation_failure"]
            ))
            return None, "Invalid input detected"
            
        # Check IP security
        ip_allowed, ip_reason = self.ip_security_manager.is_ip_allowed(ip_address)
        if not ip_allowed:
            self.auditor.log_security_event(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type="authentication_failed",
                user_id=None,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="blocked_ip",
                timestamp=datetime.now(),
                details={"reason": ip_reason},
                risk_score=0.9,
                threat_indicators=["blocked_ip"]
            ))
            return None, "Access denied from this IP address"
            
        # Record IP access for rate limiting
        self.ip_security_manager.record_access(ip_address)
        
        user = self.users.get(sanitized_username)
        
        # Create security event
        event = SecurityEvent(
            event_id=secrets.token_urlsafe(8),
            event_type="authentication_attempt",
            user_id=user.user_id if user else None,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="auth",
            action="login",
            result="pending",
            timestamp=datetime.utcnow(),
            details={"username": username},
            risk_score=0.0
        )
        
        # Check if user exists
        if not user:
            event.result = "failed"
            event.details["reason"] = "user_not_found"
            self.auditor.log_security_event(event)
            return None, "Invalid username or password"
            
        # Check failed login attempts for this IP
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=15)
        
        # Clean old failed attempts
        self.failed_login_attempts[ip_address] = [
            attempt for attempt in self.failed_login_attempts[ip_address]
            if attempt > cutoff_time
        ]
        
        # Check if IP has too many failed attempts
        if len(self.failed_login_attempts[ip_address]) >= self.config.max_login_attempts:
            self.auditor.log_security_event(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type="brute_force_attack",
                user_id=user.user_id if user else None,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="blocked_brute_force",
                timestamp=now,
                details={
                    "failed_attempts_count": len(self.failed_login_attempts[ip_address]),
                    "reason": "brute_force_protection"
                },
                risk_score=0.9,
                threat_indicators=["brute_force_attack"]
            ))
            
            # Update IP reputation
            self.ip_security_manager.update_ip_reputation(ip_address, 0.1, "Brute force attack")
            return None, "Too many failed attempts. Please try again later."
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            self.auditor.log_security_event(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type="authentication_failed",
                user_id=user.user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="account_locked",
                timestamp=now,
                details={"reason": "account_locked", "locked_until": user.locked_until.isoformat()},
                risk_score=0.4
            ))
            return None, f"Account locked until {user.locked_until}"
            
        # Check password
        if not self.password_manager.verify_password(sanitized_password, user.password_hash):
            user.failed_login_attempts += 1
            self.failed_login_attempts[ip_address].append(now)
            
            # Lock account after max attempts
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.utcnow() + timedelta(
                    minutes=self.config.lockout_duration_minutes
                )
                
            self.auditor.log_security_event(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type="authentication_failed",
                user_id=user.user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource="auth",
                action="login",
                result="invalid_password",
                timestamp=now,
                details={
                    "reason": "invalid_password",
                    "failed_attempts": user.failed_login_attempts,
                    "account_locked": user.locked_until is not None
                },
                risk_score=0.5 + (user.failed_login_attempts * 0.1),
                threat_indicators=["failed_authentication"]
            ))
            return None, "Invalid username or password"
            
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Clear failed attempts for this IP on successful login
        if ip_address in self.failed_login_attempts:
            del self.failed_login_attempts[ip_address]
            
        # Update IP reputation positively
        self.ip_security_manager.update_ip_reputation(ip_address, 0.8, "Successful authentication")
        
        # Generate JWT token
        token = self.jwt_manager.generate_token(user)
        
        # Create session with enhanced tracking
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            'user_id': user.user_id,
            'username': sanitized_username,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'session_token': token[:8] + "..."  # Store partial token for tracking
        }
        
        with self.lock:
            self.security_metrics["successful_logins"] += 1
            
        # Log successful authentication
        self.auditor.log_security_event(SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type="authentication_successful",
            user_id=user.user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource="auth",
            action="login",
            result="success",
            timestamp=now,
            details={
                "session_id": session_id,
                "user_role": user.role
            },
            risk_score=0.1,
            session_id=session_id
        ))
        
        return token, "Authentication successful"
        
    def verify_request(self, token: str, required_permission: str = None) -> Tuple[bool, Optional[Dict]]:
        """Verify request authentication and authorization."""
        payload = self.jwt_manager.verify_token(token)
        if not payload:
            return False, None
            
        # Check permission if required
        if required_permission:
            user_permissions = payload.get('permissions', [])
            if required_permission not in user_permissions and 'admin' not in user_permissions:
                return False, None
                
        return True, payload
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.encryption_manager.encrypt_data(data)
        
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.encryption_manager.decrypt_data(encrypted_data)
        
    def check_rate_limit(self, identifier: str) -> Tuple[bool, int]:
        """Check if request is within rate limits."""
        return self.rate_limiter.is_allowed(identifier)
        
    def validate_session_security(
        self, 
        session_id: str, 
        ip_address: str, 
        user_agent: str
    ) -> Tuple[bool, str]:
        """Validate session security parameters."""
        if session_id not in self.active_sessions:
            return False, "Invalid session"
            
        session = self.active_sessions[session_id]
        
        # Check IP consistency
        if session['ip_address'] != ip_address:
            self.auditor.log_security_event(SecurityEvent(
                event_id=str(uuid.uuid4()),
                event_type="session_hijack_attempt",
                user_id=session['user_id'],
                ip_address=ip_address,
                user_agent=user_agent,
                resource="session",
                action="validation",
                result="ip_mismatch",
                timestamp=datetime.now(),
                details={
                    "session_ip": session['ip_address'],
                    "request_ip": ip_address,
                    "session_id": session_id
                },
                risk_score=0.8,
                session_id=session_id,
                threat_indicators=["session_hijack_attempt", "ip_mismatch"]
            ))
            return False, "Session security violation detected"
            
        # Check session timeout
        last_activity = session.get('last_activity', session['created_at'])
        session_age = datetime.utcnow() - last_activity
        
        if session_age > timedelta(minutes=self.config.session_timeout_minutes):
            del self.active_sessions[session_id]
            return False, "Session expired"
            
        # Update last activity
        session['last_activity'] = datetime.utcnow()
        
        return True, "Session valid"
        
    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses."""
        return dict(self.config.security_headers)
        
    def perform_vulnerability_scan(self) -> Dict[str, Any]:
        """Perform basic vulnerability scanning."""
        vulnerabilities = []
        recommendations = []
        
        # Check for weak configurations
        if len(self.config.jwt_secret_key) < 32:
            vulnerabilities.append({
                "type": "weak_jwt_secret",
                "severity": "high",
                "description": "JWT secret key is too short",
                "recommendation": "Use a JWT secret key of at least 32 characters"
            })
            
        # Check password policy
        if self.config.password_min_length < 12:
            vulnerabilities.append({
                "type": "weak_password_policy",
                "severity": "medium",
                "description": "Password minimum length is too low",
                "recommendation": "Increase minimum password length to 12 or more characters"
            })
            
        # Check for default configurations
        if self.config.jwt_secret_key == "change_me_in_production":
            vulnerabilities.append({
                "type": "default_secret",
                "severity": "critical",
                "description": "Using default JWT secret key",
                "recommendation": "Change JWT secret key to a secure random value"
            })
            
        # Check session timeout
        if self.config.session_timeout_minutes > 60:
            vulnerabilities.append({
                "type": "long_session_timeout",
                "severity": "low",
                "description": "Session timeout is longer than recommended",
                "recommendation": "Consider reducing session timeout to 60 minutes or less"
            })
            
        # Generate recommendations
        if not vulnerabilities:
            recommendations.append("Security configuration appears to be properly configured")
        else:
            recommendations.extend([v["recommendation"] for v in vulnerabilities])
            
        return {
            "scan_timestamp": datetime.now().isoformat(),
            "vulnerabilities_found": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "recommendations": recommendations,
            "risk_level": self._calculate_overall_risk_level(vulnerabilities)
        }
        
    def _calculate_overall_risk_level(self, vulnerabilities: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level from vulnerabilities."""
        if not vulnerabilities:
            return "low"
            
        severity_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        max_severity_score = max(severity_scores.get(v["severity"], 0) for v in vulnerabilities)
        
        if max_severity_score >= 4:
            return "critical"
        elif max_severity_score >= 3:
            return "high"
        elif max_severity_score >= 2:
            return "medium"
        else:
            return "low"
            
    def get_comprehensive_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report with all components."""
        security_summary = self.auditor.get_security_summary()
        
        active_sessions_count = len(self.active_sessions)
        total_users = len(self.users)
        locked_users = sum(1 for u in self.users.values() if u.locked_until and u.locked_until > datetime.utcnow())
        
        return {
            'timestamp': datetime.now().isoformat(),
            'audit_summary': security_summary,
            'system_stats': {
                'total_users': total_users,
                'active_sessions': active_sessions_count,
                'locked_accounts': locked_users,
                'security_events_24h': security_summary['total_events']
            },
            'security_metrics': dict(self.security_metrics),
            'ip_security_stats': self.ip_security_manager.get_ip_statistics(),
            'vulnerability_scan': self.perform_vulnerability_scan(),
            'security_config': {
                'jwt_expiration_hours': self.config.jwt_expiration_hours,
                'max_login_attempts': self.config.max_login_attempts,
                'rate_limit_per_minute': self.config.rate_limit_requests_per_minute,
                '2fa_enabled': self.config.enable_2fa,
                'input_sanitization_enabled': self.config.enable_input_sanitization,
                'audit_logging_enabled': self.config.enable_audit_logging,
                'geolocation_blocking_enabled': self.config.enable_geolocation_blocking
            },
            'active_threat_indicators': self._get_active_threat_indicators(),
            'security_recommendations': self._generate_security_recommendations()
        }
        
    def _get_active_threat_indicators(self) -> List[str]:
        """Get currently active threat indicators."""
        recent_events = [
            event for event in self.auditor.security_events
            if (datetime.now() - event.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        threat_indicators = set()
        for event in recent_events:
            threat_indicators.update(event.threat_indicators)
            
        return list(threat_indicators)
        
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current state."""
        recommendations = []
        
        # Check recent security events
        recent_high_risk_events = [
            event for event in self.auditor.security_events
            if (datetime.now() - event.timestamp).total_seconds() < 86400  # Last 24 hours
            and event.risk_score > 0.7
        ]
        
        if len(recent_high_risk_events) > 10:
            recommendations.append("High number of security events detected. Consider investigating potential threats.")
            
        # Check failed login attempts
        total_failed_attempts = sum(len(attempts) for attempts in self.failed_login_attempts.values())
        if total_failed_attempts > 50:
            recommendations.append("High number of failed login attempts. Consider implementing additional protection.")
            
        # Check locked accounts
        locked_accounts = sum(1 for u in self.users.values() if u.locked_until and u.locked_until > datetime.utcnow())
        if locked_accounts > len(self.users) * 0.1:  # More than 10% of accounts locked
            recommendations.append("High percentage of locked accounts. Investigate potential attack patterns.")
            
        return recommendations
        
    def _get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role."""
        role_permissions = {
            'user': ['read:sensors', 'read:detections'],
            'operator': ['read:sensors', 'read:detections', 'write:detections', 'read:system'],
            'admin': ['admin', 'read:*', 'write:*', 'delete:*'],
            'system': ['system', 'read:*', 'write:*']
        }
        return role_permissions.get(role, ['read:sensors'])


# Legacy compatibility - keep old name but use enhanced version
SecurityManager = EnhancedSecurityManager


# Global enhanced security manager instance with thread safety
_security_manager = None
_security_manager_lock = threading.Lock()


def get_security_manager() -> EnhancedSecurityManager:
    """Get thread-safe global security manager instance."""
    global _security_manager
    
    if _security_manager is None:
        with _security_manager_lock:
            if _security_manager is None:
                security_config = SecurityConfig(
                    jwt_secret_key=os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(64))
                )
                _security_manager = EnhancedSecurityManager(security_config)
                
    return _security_manager


def configure_security_manager(config: SecurityConfig) -> EnhancedSecurityManager:
    """Configure global security manager with specific settings."""
    global _security_manager
    
    with _security_manager_lock:
        _security_manager = EnhancedSecurityManager(config)
        
    return _security_manager


# Security decorator for protecting endpoints
def security_required(
    permissions: Optional[List[str]] = None,
    sanitize_input: bool = True,
    check_ip: bool = True,
    log_access: bool = True
):
    """Decorator for endpoint security with comprehensive protection."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_manager = get_security_manager()
            
            # Extract request information (this would be framework-specific)
            # For now, using placeholder values
            ip_address = "127.0.0.1"  # Would extract from request
            user_agent = "unknown"    # Would extract from request
            user_id = None            # Would extract from session/token
            
            try:
                # Check IP security if enabled
                if check_ip:
                    ip_allowed, ip_reason = security_manager.ip_security_manager.is_ip_allowed(ip_address)
                    if not ip_allowed:
                        if log_access:
                            security_manager.auditor.log_security_event(SecurityEvent(
                                event_id=str(uuid.uuid4()),
                                event_type="access_denied",
                                user_id=user_id,
                                ip_address=ip_address,
                                user_agent=user_agent,
                                resource=func.__name__,
                                action="access_attempt",
                                result="blocked_ip",
                                timestamp=datetime.now(),
                                details={"reason": ip_reason},
                                risk_score=0.8
                            ))
                        raise SecurityError(f"Access denied: {ip_reason}")
                        
                # Sanitize inputs if enabled
                if sanitize_input and args:
                    sanitized_args = []
                    for i, arg in enumerate(args):
                        sanitized_arg, is_safe = security_manager.sanitize_and_validate_input(
                            arg, f"{func.__name__}_arg_{i}", user_id, ip_address
                        )
                        if not is_safe:
                            raise SecurityError("Input validation failed")
                        sanitized_args.append(sanitized_arg)
                    args = tuple(sanitized_args)
                    
                # Execute function
                result = func(*args, **kwargs)
                
                # Log successful access
                if log_access:
                    security_manager.auditor.log_security_event(SecurityEvent(
                        event_id=str(uuid.uuid4()),
                        event_type="resource_accessed",
                        user_id=user_id,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        resource=func.__name__,
                        action="access",
                        result="success",
                        timestamp=datetime.now(),
                        details={"function": func.__name__},
                        risk_score=0.1
                    ))
                    
                return result
                
            except Exception as e:
                # Log security exception
                if log_access:
                    security_manager.auditor.log_security_event(SecurityEvent(
                        event_id=str(uuid.uuid4()),
                        event_type="access_error",
                        user_id=user_id,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        resource=func.__name__,
                        action="access_attempt",
                        result="error",
                        timestamp=datetime.now(),
                        details={"error": str(e)},
                        risk_score=0.5
                    ))
                raise
                
        return wrapper
    return decorator


# Initialize on import
get_security_manager()