"""API dependencies for the BioNeuro-Olfactory-Fusion system."""

from functools import lru_cache
from typing import Optional
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models.fusion.multimodal_fusion import create_standard_fusion_network, OlfactoryFusionSNN
from ..sensors.enose.sensor_array import create_standard_enose, ENoseArray
from ..data.database.connection import get_database_manager, DatabaseManager

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


@lru_cache()
def get_detection_model() -> OlfactoryFusionSNN:
    """Get the gas detection fusion network model.
    
    This function creates and caches the fusion network model
    for gas detection. Uses LRU cache to ensure single instance.
    
    Returns:
        OlfactoryFusionSNN: The fusion network model
    """
    try:
        logger.info("Loading gas detection fusion network...")
        model = create_standard_fusion_network()
        logger.info("Gas detection model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load detection model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize detection model: {str(e)}"
        )


@lru_cache()
def get_sensor_array() -> ENoseArray:
    """Get the electronic nose sensor array.
    
    This function creates and caches the sensor array
    for reading gas concentrations. Uses LRU cache to ensure single instance.
    
    Returns:
        ENoseArray: The sensor array interface
    """
    try:
        logger.info("Initializing sensor array...")
        sensor_array = create_standard_enose()
        logger.info("Sensor array initialized successfully")
        return sensor_array
    except Exception as e:
        logger.error(f"Failed to initialize sensor array: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize sensor array: {str(e)}"
        )


@lru_cache()
def get_db_manager() -> DatabaseManager:
    """Get the database manager instance.
    
    This function creates and caches the database manager
    for storing and retrieving experiment data.
    
    Returns:
        DatabaseManager: The database manager instance
    """
    try:
        logger.info("Connecting to database...")
        db_manager = get_database_manager()
        logger.info("Database connection established")
        return db_manager
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection failed: {str(e)}"
        )


def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify API key authentication.
    
    This is a simplified authentication mechanism. In production,
    this should be replaced with proper JWT or OAuth2 authentication.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        bool: True if authentication is valid
        
    Raises:
        HTTPException: If authentication fails
    """
    # For development/testing, authentication is optional
    # In production, uncomment the following lines:
    
    # if not credentials:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="API key required",
    #         headers={"WWW-Authenticate": "Bearer"}
    #     )
    
    # # Verify API key (replace with actual key validation)
    # if credentials.credentials != "your-secret-api-key":
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid API key",
    #         headers={"WWW-Authenticate": "Bearer"}
    #     )
    
    return True


def get_current_user(authenticated: bool = Depends(verify_api_key)) -> dict:
    """Get current authenticated user information.
    
    Args:
        authenticated: Authentication status from verify_api_key
        
    Returns:
        dict: User information
    """
    # For development, return a default user
    # In production, extract user info from JWT token
    return {
        "user_id": "default_user",
        "permissions": ["read", "write"],
        "role": "operator"
    }


def check_system_health() -> dict:
    """Check overall system health status.
    
    Returns:
        dict: System health information
        
    Raises:
        HTTPException: If system is unhealthy
    """
    health_status = {
        "status": "healthy",
        "checks": {}
    }
    
    try:
        # Check model availability
        model = get_detection_model()
        health_status["checks"]["model"] = "healthy"
    except Exception as e:
        health_status["checks"]["model"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Check sensor array
        sensors = get_sensor_array()
        health_status["checks"]["sensors"] = "healthy"
    except Exception as e:
        health_status["checks"]["sensors"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Check database
        db = get_db_manager()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # If critical components are down, raise error
    if health_status["status"] == "degraded":
        unhealthy_components = [
            component for component, status in health_status["checks"].items()
            if "unhealthy" in status
        ]
        
        if len(unhealthy_components) > 1:  # More than one component down
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"System unhealthy: {', '.join(unhealthy_components)} down"
            )
    
    return health_status


def validate_request_size(content_length: Optional[int] = None) -> bool:
    """Validate request size to prevent DoS attacks.
    
    Args:
        content_length: Content length from request headers
        
    Returns:
        bool: True if request size is acceptable
        
    Raises:
        HTTPException: If request is too large
    """
    max_size = 10 * 1024 * 1024  # 10MB limit
    
    if content_length and content_length > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Request too large. Maximum size: {max_size} bytes"
        )
    
    return True


def get_rate_limit_key(user: dict = Depends(get_current_user)) -> str:
    """Get rate limiting key for the current user.
    
    Args:
        user: Current user information
        
    Returns:
        str: Rate limiting key
    """
    return f"user:{user['user_id']}"


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if request is allowed under rate limit.
        
        Args:
            key: Rate limiting key
            limit: Request limit per window
            window: Time window in seconds
            
        Returns:
            bool: True if request is allowed
        """
        import time
        
        now = time.time()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window
        ]
        
        # Check if under limit
        if len(self.requests[key]) >= limit:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


def check_rate_limit(
    rate_limit_key: str = Depends(get_rate_limit_key),
    limit: int = 100,
    window: int = 3600
) -> bool:
    """Check and enforce rate limiting.
    
    Args:
        rate_limit_key: Rate limiting key
        limit: Request limit per window
        window: Time window in seconds
        
    Returns:
        bool: True if request is allowed
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    if not rate_limiter.is_allowed(rate_limit_key, limit, window):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {limit} requests per {window} seconds"
        )
    
    return True


def get_request_id() -> str:
    """Generate unique request ID for tracking.
    
    Returns:
        str: Unique request identifier
    """
    import uuid
    return str(uuid.uuid4())


def log_request_start(request_id: str = Depends(get_request_id)) -> str:
    """Log request start for monitoring.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        str: Request ID for downstream use
    """
    logger.info(f"Request started: {request_id}")
    return request_id


def get_system_config() -> dict:
    """Get system configuration parameters.
    
    Returns:
        dict: System configuration
    """
    return {
        "max_batch_size": 1000,
        "max_simulation_duration": 1000,  # milliseconds
        "supported_gas_types": ["methane", "carbon_monoxide", "ammonia", "propane"],
        "default_confidence_threshold": 0.8,
        "max_sensor_array_size": 20,
        "max_audio_features": 512,
        "api_version": "1.0.0"
    }


def validate_gas_type(gas_type: str, config: dict = Depends(get_system_config)) -> bool:
    """Validate gas type against supported types.
    
    Args:
        gas_type: Gas type to validate
        config: System configuration
        
    Returns:
        bool: True if gas type is supported
        
    Raises:
        HTTPException: If gas type is not supported
    """
    if gas_type not in config["supported_gas_types"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported gas type: {gas_type}. "
                   f"Supported types: {config['supported_gas_types']}"
        )
    
    return True


def get_emergency_contacts() -> dict:
    """Get emergency contact information for critical alerts.
    
    Returns:
        dict: Emergency contact configuration
    """
    return {
        "email": ["safety@company.com", "emergency@company.com"],
        "phone": ["+1-555-0123", "+1-555-0124"],
        "webhook": "https://alerts.company.com/webhook"
    }


def validate_experiment_access(
    experiment_id: int,
    user: dict = Depends(get_current_user),
    db: DatabaseManager = Depends(get_db_manager)
) -> bool:
    """Validate user access to specific experiment.
    
    Args:
        experiment_id: Experiment ID to validate
        user: Current user information
        db: Database manager
        
    Returns:
        bool: True if access is allowed
        
    Raises:
        HTTPException: If access is denied or experiment not found
    """
    try:
        # Check if experiment exists
        exp_data = db.get_experiment_data(experiment_id)
        
        # For now, allow all authenticated users access
        # In production, implement proper access control
        return True
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found"
        )
    except Exception as e:
        logger.error(f"Error validating experiment access: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating experiment access"
        )