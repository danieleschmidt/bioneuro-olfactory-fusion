"""FastAPI server for neuromorphic gas detection API."""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator

from .routes.detection import detection_bp
from .routes.experiments import experiments_bp
from .routes.sensors import sensors_bp
from .routes.health import health_bp
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.logging import LoggingMiddleware
from ..data.database.connection import get_database_manager

logger = logging.getLogger(__name__)


# Global state
app_state = {
    "db_manager": None,
    "detection_model": None,
    "sensor_array": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting BioNeuro-Olfactory-Fusion API server...")
    
    try:
        # Initialize database
        app_state["db_manager"] = get_database_manager()
        logger.info("Database connection established")
        
        # Initialize detection model
        from .. import create_efficient_network
        app_state["detection_model"] = create_efficient_network(num_sensors=6)
        logger.info("Detection model loaded")
        
        # Initialize sensor array (in simulation mode for API)
        from ..sensors.enose.sensor_array import create_standard_enose
        app_state["sensor_array"] = create_standard_enose()
        logger.info("Sensor array initialized")
        
        logger.info("API server startup complete")
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise
        
    yield
    
    # Shutdown
    logger.info("Shutting down BioNeuro-Olfactory-Fusion API server...")
    
    if app_state["db_manager"]:
        app_state["db_manager"].close()
        logger.info("Database connection closed")
        
    logger.info("API server shutdown complete")


def create_app(
    debug: bool = False,
    enable_cors: bool = True,
    enable_auth: bool = True,
    enable_rate_limiting: bool = True,
    enable_metrics: bool = True
) -> FastAPI:
    """Create and configure FastAPI application.
    
    Args:
        debug: Enable debug mode
        enable_cors: Enable CORS middleware
        enable_auth: Enable authentication middleware
        enable_rate_limiting: Enable rate limiting middleware
        enable_metrics: Enable Prometheus metrics
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="BioNeuro-Olfactory-Fusion API",
        description="Neuromorphic multi-modal gas detection system",
        version="0.1.0",
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        lifespan=lifespan
    )
    
    # Add middleware (order matters!)
    
    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if debug else ["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Custom middleware
    if enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware)
        
    if enable_auth:
        app.add_middleware(AuthMiddleware)
        
    app.add_middleware(LoggingMiddleware)
    
    # Metrics
    if enable_metrics:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app)
    
    # Exception handlers
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": "The requested resource was not found",
                "path": str(request.url.path)
            }
        )
        
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception):
        logger.error(f"Internal server error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred"
            }
        )
    
    # Register blueprints/routers
    app.include_router(health_bp, prefix="/health", tags=["Health"])
    app.include_router(detection_bp, prefix="/api/v1/detection", tags=["Detection"])
    app.include_router(experiments_bp, prefix="/api/v1/experiments", tags=["Experiments"])
    app.include_router(sensors_bp, prefix="/api/v1/sensors", tags=["Sensors"])
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "BioNeuro-Olfactory-Fusion API",
            "version": "0.1.0",
            "description": "Neuromorphic multi-modal gas detection system",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "detection": "/api/v1/detection",
                "experiments": "/api/v1/experiments",
                "sensors": "/api/v1/sensors",
                "docs": "/docs" if debug else "disabled",
                "metrics": "/metrics" if enable_metrics else "disabled"
            }
        }
    
    return app


def get_app_state():
    """Get global application state."""
    return app_state


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info"
):
    """Run the API server with uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        debug: Enable debug mode
        reload: Enable auto-reload on code changes
        workers: Number of worker processes
        log_level: Logging level
    """
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration from environment
    host = os.getenv("API_HOST", host)
    port = int(os.getenv("API_PORT", port))
    debug = os.getenv("API_DEBUG", str(debug)).lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Workers: {workers}")
    
    # Create app with configuration
    app = create_app(
        debug=debug,
        enable_cors=True,
        enable_auth=not debug,  # Disable auth in debug mode
        enable_rate_limiting=not debug,  # Disable rate limiting in debug mode
        enable_metrics=True
    )
    
    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload and debug,
        workers=workers if not reload else 1,
        access_log=debug
    )


if __name__ == "__main__":
    # Load configuration from environment
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    
    run_server(
        debug=debug,
        reload=debug,
        log_level="debug" if debug else "info"
    )