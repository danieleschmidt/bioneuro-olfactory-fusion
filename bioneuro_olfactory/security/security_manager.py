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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from enum import Enum
import time
from collections import defaultdict, deque

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
    """Security configuration parameters."""
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
    """Security event for audit logging."""
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


class SecurityAuditor:
    """Security event auditing and monitoring."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.security_events: deque = deque(maxlen=10000)
        self.threat_patterns = self._load_threat_patterns()
        
    def log_security_event(self, event: SecurityEvent):
        """Log a security event."""
        self.security_events.append(event)
        
        # Calculate risk score
        event.risk_score = self._calculate_risk_score(event)
        
        # Check for threats
        if event.risk_score > 0.7:
            self._handle_high_risk_event(event)
            
        logger.info(f"Security event: {event.event_type} - Risk: {event.risk_score:.2f}")
        
    def _calculate_risk_score(self, event: SecurityEvent) -> float:
        """Calculate risk score for security event."""
        risk_score = 0.0
        
        # Failed authentication attempts
        if event.event_type == "authentication_failed":
            risk_score += 0.3
            
        # Multiple failed attempts from same IP
        recent_failures = [
            e for e in self.security_events
            if e.ip_address == event.ip_address
            and e.event_type == "authentication_failed"
            and (event.timestamp - e.timestamp).total_seconds() < 300  # 5 minutes
        ]
        
        if len(recent_failures) >= 3:
            risk_score += 0.4
            
        # Suspicious user agent
        if self._is_suspicious_user_agent(event.user_agent):
            risk_score += 0.2
            
        # Access to sensitive resources
        sensitive_resources = ["admin", "config", "users", "keys"]
        if any(resource in event.resource.lower() for resource in sensitive_resources):
            risk_score += 0.2
            
        # Unusual access patterns
        if self._is_unusual_access_pattern(event):
            risk_score += 0.3
            
        return min(risk_score, 1.0)
        
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
        """Handle high-risk security events."""
        logger.warning(f"High-risk security event detected: {event.event_type}")
        
        # Could trigger additional security measures:
        # - Temporary IP blocking
        # - User account lockout
        # - Admin notifications
        # - Enhanced monitoring
        
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
        """Get security summary for specified time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_events = [
            e for e in self.security_events
            if e.timestamp > cutoff_time
        ]
        
        event_counts = defaultdict(int)
        high_risk_events = []
        unique_ips = set()
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            unique_ips.add(event.ip_address)
            
            if event.risk_score > 0.7:
                high_risk_events.append(event)
                
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_types': dict(event_counts),
            'unique_ips': len(unique_ips),
            'high_risk_events': len(high_risk_events),
            'average_risk_score': sum(e.risk_score for e in recent_events) / max(len(recent_events), 1),
            'most_common_event': max(event_counts, key=event_counts.get) if event_counts else None
        }


class SecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_manager = PasswordManager(config)
        self.encryption_manager = EncryptionManager()
        self.jwt_manager = JWTManager(config)
        self.rate_limiter = RateLimiter(config.rate_limit_requests_per_minute)
        self.auditor = SecurityAuditor(config)
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict] = {}
        
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
        
    def authenticate_user(self, username: str, password: str,
                         ip_address: str, user_agent: str) -> Tuple[Optional[str], str]:
        """Authenticate user and return JWT token."""
        user = self.users.get(username)
        
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
            
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            event.result = "failed"
            event.details["reason"] = "account_locked"
            self.auditor.log_security_event(event)
            return None, f"Account locked until {user.locked_until}"
            
        # Check password
        if not self.password_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after max attempts
            if user.failed_login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.utcnow() + timedelta(
                    minutes=self.config.lockout_duration_minutes
                )
                
            event.result = "failed"
            event.details["reason"] = "invalid_password"
            event.details["failed_attempts"] = user.failed_login_attempts
            self.auditor.log_security_event(event)
            return None, "Invalid username or password"
            
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Generate JWT token
        token = self.jwt_manager.generate_token(user)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        self.active_sessions[session_id] = {
            'user_id': user.user_id,
            'username': username,
            'created_at': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'ip_address': ip_address,
            'user_agent': user_agent
        }
        
        event.result = "success"
        self.auditor.log_security_event(event)
        
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
        
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report."""
        security_summary = self.auditor.get_security_summary()
        
        active_sessions_count = len(self.active_sessions)
        total_users = len(self.users)
        locked_users = sum(1 for u in self.users.values() if u.locked_until and u.locked_until > datetime.utcnow())
        
        return {
            'audit_summary': security_summary,
            'system_stats': {
                'total_users': total_users,
                'active_sessions': active_sessions_count,
                'locked_accounts': locked_users,
                'security_events_24h': security_summary['total_events']
            },
            'security_config': {
                'jwt_expiration_hours': self.config.jwt_expiration_hours,
                'max_login_attempts': self.config.max_login_attempts,
                'rate_limit_per_minute': self.config.rate_limit_requests_per_minute,
                '2fa_enabled': self.config.enable_2fa
            }
        }
        
    def _get_role_permissions(self, role: str) -> List[str]:
        """Get permissions for a role."""
        role_permissions = {
            'user': ['read:sensors', 'read:detections'],
            'operator': ['read:sensors', 'read:detections', 'write:detections', 'read:system'],
            'admin': ['admin', 'read:*', 'write:*', 'delete:*'],
            'system': ['system', 'read:*', 'write:*']
        }
        return role_permissions.get(role, ['read:sensors'])


# Global security manager instance
security_config = SecurityConfig(
    jwt_secret_key=os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(64))
)
security_manager = SecurityManager(security_config)