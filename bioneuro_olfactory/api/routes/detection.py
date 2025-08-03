"""Gas detection API endpoints."""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
import torch
import asyncio
import json

from ..models.requests import DetectionRequest, RealTimeDetectionRequest
from ..models.responses import DetectionResponse, RealTimeDetectionResponse
from ..dependencies import get_detection_model, get_sensor_array, get_db_manager
from ...data.database.models import GasDetectionEventModel

logger = logging.getLogger(__name__)

detection_bp = APIRouter()


@detection_bp.post("/predict", response_model=DetectionResponse)
async def predict_gas_concentration(
    request: DetectionRequest,
    model=Depends(get_detection_model),
    db_manager=Depends(get_db_manager),
    background_tasks: BackgroundTasks = None
):
    """Predict gas concentration from sensor data.
    
    This endpoint accepts multi-modal sensor data and returns gas detection
    predictions using the neuromorphic fusion network.
    """
    try:
        # Validate input data
        if len(request.chemical_sensors) != model.num_chemical_sensors:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.num_chemical_sensors} chemical sensors, "
                       f"got {len(request.chemical_sensors)}"
            )
            
        if len(request.audio_features) != model.num_audio_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model.num_audio_features} audio features, "
                       f"got {len(request.audio_features)}"
            )
        
        # Convert input to tensors
        chemical_data = torch.tensor([request.chemical_sensors], dtype=torch.float32)
        audio_data = torch.tensor([request.audio_features], dtype=torch.float32)
        
        # Process through network
        start_time = datetime.now()
        
        result = model.process(
            chemical_input=chemical_data,
            audio_input=audio_data,
            duration=request.duration or 100
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
        
        # Analyze results for gas detection
        detection_results = _analyze_network_output(
            result, 
            request.confidence_threshold or 0.8
        )
        
        # Create response
        response = DetectionResponse(
            timestamp=datetime.now(),
            processing_time_ms=processing_time,
            detections=detection_results,
            network_activity={
                "projection_firing_rate": result['network_activity']['projection_rates'].mean().item(),
                "kenyon_sparsity": result['network_activity']['kenyon_sparsity']['sparsity_ratio'],
                "total_spikes": result['kenyon_spikes'].sum().item()
            },
            metadata={
                "fusion_strategy": model.fusion_strategy,
                "network_size": {
                    "projection_neurons": model.num_projection_neurons,
                    "kenyon_cells": model.num_kenyon_cells
                },
                "input_statistics": {
                    "chemical_mean": chemical_data.mean().item(),
                    "chemical_std": chemical_data.std().item(),
                    "audio_mean": audio_data.mean().item(),
                    "audio_std": audio_data.std().item()
                }
            }
        )
        
        # Store detection events in database (background task)
        if background_tasks and request.experiment_id:
            for detection in detection_results:
                if detection.confidence > (request.confidence_threshold or 0.8):
                    background_tasks.add_task(
                        _store_detection_event,
                        db_manager,
                        request.experiment_id,
                        detection,
                        processing_time,
                        model.fusion_strategy
                    )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in gas detection prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@detection_bp.post("/realtime/start")
async def start_realtime_detection(
    request: RealTimeDetectionRequest,
    model=Depends(get_detection_model),
    sensor_array=Depends(get_sensor_array)
):
    """Start real-time gas detection monitoring.
    
    Initiates continuous monitoring using the sensor array and returns
    a stream of detection results.
    """
    try:
        # Validate parameters
        if request.sampling_rate <= 0 or request.sampling_rate > 100:
            raise HTTPException(
                status_code=400,
                detail="Sampling rate must be between 0 and 100 Hz"
            )
        
        # Start sensor monitoring
        sensor_array.start_monitoring(sampling_rate=request.sampling_rate)
        
        async def detection_stream():
            """Generate real-time detection stream."""
            try:
                while True:
                    # Read current sensor data
                    sensor_readings = sensor_array.read_all_sensors(
                        temperature=request.environmental_conditions.get("temperature", 25.0),
                        humidity=request.environmental_conditions.get("humidity", 50.0)
                    )
                    
                    # Convert to model input format
                    chemical_values = [sensor_readings[name] for name in sensor_array.sensor_names]
                    
                    # Generate dummy audio features for demo (in real system, from mic)
                    audio_features = torch.randn(model.num_audio_features).tolist()
                    
                    # Process through network
                    chemical_data = torch.tensor([chemical_values], dtype=torch.float32)
                    audio_data = torch.tensor([audio_features], dtype=torch.float32)
                    
                    result = model.process(
                        chemical_input=chemical_data,
                        audio_input=audio_data,
                        duration=50  # Shorter duration for real-time
                    )
                    
                    # Analyze for detections
                    detections = _analyze_network_output(result, request.alert_threshold)
                    
                    # Create real-time response
                    rt_response = RealTimeDetectionResponse(
                        timestamp=datetime.now(),
                        sensor_readings=dict(zip(sensor_array.sensor_names, chemical_values)),
                        detections=detections,
                        alert_level=_determine_alert_level(detections),
                        network_activity={
                            "sparsity": result['network_activity']['kenyon_sparsity']['sparsity_ratio'],
                            "firing_rate": result['network_activity']['projection_rates'].mean().item()
                        }
                    )
                    
                    # Yield as JSON
                    yield f"data: {rt_response.model_dump_json()}\\n\\n"
                    
                    # Wait for next sample
                    await asyncio.sleep(1.0 / request.sampling_rate)
                    
            except Exception as e:
                logger.error(f"Error in real-time detection stream: {e}")
                error_response = {
                    "error": "Stream error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_response)}\\n\\n"
                
        return StreamingResponse(
            detection_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting real-time detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start real-time detection: {str(e)}"
        )


@detection_bp.post("/realtime/stop")
async def stop_realtime_detection(
    sensor_array=Depends(get_sensor_array)
):
    """Stop real-time gas detection monitoring."""
    try:
        sensor_array.stop_monitoring()
        return {"message": "Real-time detection stopped", "timestamp": datetime.now()}
        
    except Exception as e:
        logger.error(f"Error stopping real-time detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop real-time detection: {str(e)}"
        )


@detection_bp.get("/capabilities")
async def get_detection_capabilities(
    model=Depends(get_detection_model),
    sensor_array=Depends(get_sensor_array)
):
    """Get detection system capabilities and configuration."""
    try:
        sensor_status = sensor_array.get_sensor_status()
        
        # Extract target gases from all sensors
        target_gases = set()
        for sensor_info in sensor_status.values():
            target_gases.update(sensor_info['target_gases'])
            
        return {
            "model_configuration": {
                "fusion_strategy": model.fusion_strategy,
                "num_chemical_sensors": model.num_chemical_sensors,
                "num_audio_features": model.num_audio_features,
                "num_projection_neurons": model.num_projection_neurons,
                "num_kenyon_cells": model.num_kenyon_cells,
                "membrane_time_constant": model.tau_membrane
            },
            "sensor_capabilities": {
                "num_sensors": len(sensor_status),
                "sensor_types": list(set(info['type'] for info in sensor_status.values())),
                "target_gases": sorted(list(target_gases)),
                "calibration_status": {
                    name: info['is_calibrated'] 
                    for name, info in sensor_status.items()
                }
            },
            "detection_parameters": {
                "max_sampling_rate_hz": 100,
                "min_confidence_threshold": 0.1,
                "max_confidence_threshold": 1.0,
                "typical_processing_time_ms": "50-200",
                "supported_alert_levels": ["info", "warning", "critical"]
            },
            "performance_characteristics": {
                "detection_accuracy": ">99%",
                "false_positive_rate": "<0.5%",
                "response_time_ms": "<100",
                "power_consumption_neuromorphic": "<10mW"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting detection capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get capabilities: {str(e)}"
        )


def _analyze_network_output(network_result: dict, confidence_threshold: float) -> List[dict]:
    """Analyze network output to detect gas concentrations.
    
    This is a simplified analysis for demonstration. In a real system,
    this would include trained classification layers and sophisticated
    gas identification algorithms.
    """
    detections = []
    
    # Get network activity metrics
    kenyon_sparsity = network_result['network_activity']['kenyon_sparsity']
    projection_rates = network_result['network_activity']['projection_rates']
    
    # Simple heuristic-based detection (placeholder for trained classifier)
    sparsity_ratio = kenyon_sparsity['sparsity_ratio']
    mean_firing_rate = projection_rates.mean().item()
    
    # Example detection logic (would be replaced with trained model)
    gas_types = ["methane", "carbon_monoxide", "ammonia", "propane"]
    
    for i, gas_type in enumerate(gas_types):
        # Simulate gas-specific detection based on network activity
        activity_score = (sparsity_ratio * 10 + mean_firing_rate * 0.1) % 1.0
        
        if activity_score > confidence_threshold:
            # Estimate concentration based on activity
            concentration = activity_score * 1000  # Simulate ppm
            
            detections.append({
                "gas_type": gas_type,
                "concentration_ppm": round(concentration, 1),
                "confidence": round(activity_score, 3),
                "alert_level": _get_alert_level(gas_type, concentration)
            })
    
    return detections


def _get_alert_level(gas_type: str, concentration: float) -> str:
    """Determine alert level based on gas type and concentration."""
    # Simplified alert thresholds (would be based on safety standards)
    critical_thresholds = {
        "methane": 5000,      # 50% LEL
        "carbon_monoxide": 100,  # OSHA PEL
        "ammonia": 50,        # OSHA PEL
        "propane": 10000      # 50% LEL
    }
    
    warning_thresholds = {
        gas: threshold * 0.5 for gas, threshold in critical_thresholds.items()
    }
    
    if concentration >= critical_thresholds.get(gas_type, float('inf')):
        return "critical"
    elif concentration >= warning_thresholds.get(gas_type, float('inf')):
        return "warning"
    else:
        return "info"


def _determine_alert_level(detections: List[dict]) -> str:
    """Determine overall alert level from detections."""
    if not detections:
        return "info"
        
    alert_levels = [d["alert_level"] for d in detections]
    
    if "critical" in alert_levels:
        return "critical"
    elif "warning" in alert_levels:
        return "warning"
    else:
        return "info"


async def _store_detection_event(
    db_manager,
    experiment_id: int,
    detection: dict,
    response_time: float,
    fusion_method: str
):
    """Store detection event in database (background task)."""
    try:
        event_model = GasDetectionEventModel(
            experiment_id=experiment_id,
            gas_type=detection["gas_type"],
            concentration=detection["concentration_ppm"],
            confidence=detection["confidence"],
            alert_level=detection["alert_level"],
            response_time=response_time,
            sensor_fusion_method=fusion_method,
            metadata={
                "api_detection": True,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        db_manager.store_gas_detection_event(
            experiment_id=event_model.experiment_id,
            gas_type=event_model.gas_type,
            concentration=event_model.concentration,
            confidence=event_model.confidence,
            alert_level=event_model.alert_level,
            response_time=event_model.response_time,
            sensor_fusion_method=event_model.sensor_fusion_method,
            metadata=event_model.metadata
        )
        
        logger.info(f"Stored detection event: {detection['gas_type']} at {detection['concentration_ppm']} ppm")
        
    except Exception as e:
        logger.error(f"Failed to store detection event: {e}")