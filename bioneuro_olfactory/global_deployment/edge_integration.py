"""Edge Computing Integration for Neuromorphic Gas Detection.

This module implements comprehensive edge computing capabilities for deploying
neuromorphic gas detection systems on edge devices with cloud synchronization,
mobile optimization, and real-time processing capabilities.

Key Features:
- Edge device management and orchestration
- Mobile edge computing optimization
- Real-time edge-cloud synchronization
- Lightweight neuromorphic model deployment
- Edge-specific security and compliance
- Offline operation capabilities
"""

import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class EdgeDeviceType(Enum):
    """Types of edge devices supported."""
    INDUSTRIAL_GATEWAY = "industrial_gateway"
    IOT_SENSOR_HUB = "iot_sensor_hub" 
    MOBILE_DEVICE = "mobile_device"
    EMBEDDED_SYSTEM = "embedded_system"
    NEUROMORPHIC_CHIP = "neuromorphic_chip"
    RASPBERRY_PI = "raspberry_pi"
    NVIDIA_JETSON = "nvidia_jetson"


class EdgeComputeCapability(Enum):
    """Edge computing capability levels."""
    MINIMAL = "minimal"        # Basic sensor data processing
    STANDARD = "standard"      # Full inference capability
    ENHANCED = "enhanced"      # Advanced AI processing
    NEUROMORPHIC = "neuromorphic"  # Hardware-accelerated spiking networks


@dataclass
class EdgeDeviceConfig:
    """Configuration for edge computing devices."""
    device_id: str
    device_type: EdgeDeviceType
    location: str
    
    # Hardware specifications
    cpu_cores: int = 4
    memory_mb: int = 1024
    storage_gb: int = 16
    has_gpu: bool = False
    has_neuromorphic_chip: bool = False
    
    # Network capabilities
    network_bandwidth_mbps: float = 10.0
    cellular_enabled: bool = False
    wifi_enabled: bool = True
    ethernet_enabled: bool = False
    
    # Power and environmental
    battery_powered: bool = False
    battery_life_hours: float = 24.0
    operating_temp_range: Tuple[int, int] = (-20, 70)
    ip_rating: str = "IP54"
    
    # Computing capabilities
    compute_capability: EdgeComputeCapability = EdgeComputeCapability.STANDARD
    max_concurrent_inferences: int = 10
    inference_latency_ms: float = 50.0
    
    # Security and compliance
    secure_boot_enabled: bool = True
    encryption_enabled: bool = True
    compliance_certifications: List[str] = field(default_factory=list)


class EdgeComputingManager:
    """Manager for edge computing infrastructure and deployment.
    
    Handles deployment, monitoring, and management of neuromorphic
    gas detection systems across diverse edge computing devices.
    """
    
    def __init__(self):
        self.edge_devices: Dict[str, EdgeDeviceConfig] = {}
        self.device_status: Dict[str, Dict[str, Any]] = {}
        self.deployment_registry: Dict[str, Dict[str, Any]] = {}
        self.sync_manager = EdgeCloudSynchronizer()
        
    def register_edge_device(self, device_config: EdgeDeviceConfig):
        """Register new edge device in the system."""
        device_id = device_config.device_id
        
        self.edge_devices[device_id] = device_config
        self.device_status[device_id] = {
            'status': 'registered',
            'last_heartbeat': time.time(),
            'health_score': 1.0,
            'active_deployments': 0,
            'resource_utilization': {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'storage_percent': 0.0
            }
        }
        
    def deploy_to_edge_device(
        self,
        device_id: str,
        neuromorphic_model: Dict[str, Any],
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Deploy neuromorphic model to specific edge device."""
        
        if device_id not in self.edge_devices:
            return {'success': False, 'error': 'Device not registered'}
            
        device = self.edge_devices[device_id]
        
        # Check device compatibility
        compatibility_check = self._check_device_compatibility(
            device, neuromorphic_model
        )
        
        if not compatibility_check['compatible']:
            return {
                'success': False,
                'error': 'Device incompatible',
                'details': compatibility_check['issues']
            }
            
        # Optimize model for edge device
        optimized_model = self._optimize_model_for_edge(
            neuromorphic_model, device
        )
        
        # Execute deployment
        deployment_result = self._execute_edge_deployment(
            device_id, optimized_model, deployment_config or {}
        )
        
        # Update deployment registry
        if deployment_result['success']:
            deployment_id = f"edge_deploy_{device_id}_{int(time.time())}"
            self.deployment_registry[deployment_id] = {
                'device_id': device_id,
                'model_config': optimized_model,
                'deployment_time': time.time(),
                'status': 'active'
            }
            deployment_result['deployment_id'] = deployment_id
            
        return deployment_result
    
    def _check_device_compatibility(
        self,
        device: EdgeDeviceConfig,
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if device is compatible with model requirements."""
        
        issues = []
        
        # Memory requirements
        required_memory = model.get('memory_requirements_mb', 512)
        if device.memory_mb < required_memory:
            issues.append(f"Insufficient memory: {device.memory_mb}MB < {required_memory}MB")
            
        # Storage requirements
        required_storage = model.get('storage_requirements_gb', 2)
        if device.storage_gb < required_storage:
            issues.append(f"Insufficient storage: {device.storage_gb}GB < {required_storage}GB")
            
        # Compute capability
        required_capability = model.get('compute_capability', 'standard')
        if device.compute_capability.value != required_capability and required_capability != 'minimal':
            issues.append(f"Incompatible compute capability: {device.compute_capability.value}")
            
        # Neuromorphic chip requirement
        if model.get('requires_neuromorphic_chip', False) and not device.has_neuromorphic_chip:
            issues.append("Neuromorphic chip required but not available")
            
        return {
            'compatible': len(issues) == 0,
            'issues': issues
        }
    
    def _optimize_model_for_edge(
        self,
        model: Dict[str, Any],
        device: EdgeDeviceConfig
    ) -> Dict[str, Any]:
        """Optimize neuromorphic model for edge device constraints."""
        
        optimized_model = model.copy()
        
        # Model compression based on device capabilities
        if device.compute_capability == EdgeComputeCapability.MINIMAL:
            # Aggressive compression for minimal devices
            optimized_model['compression_ratio'] = 0.1
            optimized_model['precision'] = 'int8'
            optimized_model['max_neurons'] = 1000
            
        elif device.compute_capability == EdgeComputeCapability.STANDARD:
            # Moderate compression for standard devices
            optimized_model['compression_ratio'] = 0.3
            optimized_model['precision'] = 'fp16'
            optimized_model['max_neurons'] = 5000
            
        elif device.compute_capability == EdgeComputeCapability.ENHANCED:
            # Light compression for enhanced devices
            optimized_model['compression_ratio'] = 0.7
            optimized_model['precision'] = 'fp32'
            optimized_model['max_neurons'] = 20000
            
        elif device.compute_capability == EdgeComputeCapability.NEUROMORPHIC:
            # No compression for neuromorphic hardware
            optimized_model['compression_ratio'] = 1.0
            optimized_model['precision'] = 'spike'
            optimized_model['max_neurons'] = 100000
            
        # Memory optimization
        memory_budget = int(device.memory_mb * 0.7)  # Use 70% of available memory
        optimized_model['memory_budget_mb'] = memory_budget
        
        # Latency optimization
        optimized_model['target_latency_ms'] = device.inference_latency_ms
        
        return optimized_model
    
    def _execute_edge_deployment(
        self,
        device_id: str,
        model: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute deployment to edge device."""
        
        # Simulate deployment process
        deployment_steps = [
            'model_transfer',
            'dependency_installation',
            'configuration_setup',
            'model_loading',
            'health_verification'
        ]
        
        # Update device status
        self.device_status[device_id].update({
            'status': 'deploying',
            'last_deployment': time.time()
        })
        
        # Simulate deployment success
        result = {
            'success': True,
            'device_id': device_id,
            'deployment_time': time.time(),
            'model_size_mb': model.get('size_mb', 50),
            'deployment_duration_seconds': 30.0,
            'steps_completed': deployment_steps,
            'optimizations_applied': [
                f"compression_{model.get('compression_ratio', 1.0)}",
                f"precision_{model.get('precision', 'fp32')}",
                f"memory_budget_{model.get('memory_budget_mb', 512)}MB"
            ]
        }
        
        # Update device status after successful deployment
        if result['success']:
            self.device_status[device_id].update({
                'status': 'active',
                'active_deployments': self.device_status[device_id]['active_deployments'] + 1
            })
            
        return result
    
    def monitor_edge_devices(self) -> Dict[str, Any]:
        """Monitor health and performance of all edge devices."""
        
        monitoring_summary = {
            'total_devices': len(self.edge_devices),
            'active_devices': 0,
            'offline_devices': 0,
            'average_health_score': 0.0,
            'total_deployments': len(self.deployment_registry),
            'device_details': {}
        }
        
        health_scores = []
        
        for device_id, status in self.device_status.items():
            device_config = self.edge_devices[device_id]
            
            # Update health based on last heartbeat
            time_since_heartbeat = time.time() - status['last_heartbeat']
            
            if time_since_heartbeat < 300:  # 5 minutes
                status['health_score'] = 1.0
                monitoring_summary['active_devices'] += 1
            else:
                status['health_score'] = max(0.0, 1.0 - (time_since_heartbeat / 3600))
                if status['health_score'] < 0.1:
                    monitoring_summary['offline_devices'] += 1
                    
            health_scores.append(status['health_score'])
            
            # Device-specific monitoring
            monitoring_summary['device_details'][device_id] = {
                'device_type': device_config.device_type.value,
                'location': device_config.location,
                'status': status['status'],
                'health_score': status['health_score'],
                'active_deployments': status['active_deployments'],
                'resource_utilization': status['resource_utilization']
            }
            
        if health_scores:
            monitoring_summary['average_health_score'] = sum(health_scores) / len(health_scores)
            
        return monitoring_summary
    
    def scale_edge_deployment(
        self,
        scaling_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Scale deployment across edge devices based on criteria."""
        
        # Identify devices that meet scaling criteria
        candidate_devices = []
        
        for device_id, device_config in self.edge_devices.items():
            device_status = self.device_status[device_id]
            
            # Check availability and health
            if (device_status['status'] == 'registered' and 
                device_status['health_score'] > 0.8 and
                device_status['active_deployments'] < device_config.max_concurrent_inferences):
                
                candidate_devices.append(device_id)
                
        # Plan scaling deployment
        scaling_plan = {
            'candidate_devices': candidate_devices,
            'target_deployments': min(len(candidate_devices), scaling_criteria.get('max_devices', 10)),
            'scaling_strategy': scaling_criteria.get('strategy', 'capacity_based')
        }
        
        return scaling_plan


class EdgeCloudSynchronizer:
    """Synchronizes data and models between edge devices and cloud."""
    
    def __init__(self):
        self.sync_policies: Dict[str, Dict[str, Any]] = {}
        self.sync_queues: Dict[str, List[Dict[str, Any]]] = {}
        self.offline_buffers: Dict[str, List[Dict[str, Any]]] = {}
        
    def configure_sync_policy(
        self,
        device_id: str,
        policy: Dict[str, Any]
    ):
        """Configure synchronization policy for device."""
        self.sync_policies[device_id] = policy
        self.sync_queues[device_id] = []
        self.offline_buffers[device_id] = []
        
    def sync_sensor_data(
        self,
        device_id: str,
        sensor_data: List[Dict[str, Any]],
        priority: str = 'normal'
    ) -> Dict[str, Any]:
        """Synchronize sensor data from edge to cloud."""
        
        sync_entry = {
            'device_id': device_id,
            'data_type': 'sensor_data',
            'data': sensor_data,
            'timestamp': time.time(),
            'priority': priority,
            'sync_status': 'pending'
        }
        
        # Check if device is online
        if self._is_device_online(device_id):
            # Direct sync
            sync_result = self._execute_cloud_sync(sync_entry)
        else:
            # Buffer for offline sync
            self.offline_buffers[device_id].append(sync_entry)
            sync_result = {
                'success': True,
                'sync_method': 'offline_buffered',
                'buffer_size': len(self.offline_buffers[device_id])
            }
            
        return sync_result
    
    def sync_model_updates(
        self,
        device_ids: List[str],
        model_update: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Push model updates to edge devices."""
        
        sync_results = {}
        
        for device_id in device_ids:
            update_entry = {
                'device_id': device_id,
                'data_type': 'model_update',
                'data': model_update,
                'timestamp': time.time(),
                'priority': 'high'
            }
            
            if self._is_device_online(device_id):
                result = self._execute_device_sync(device_id, update_entry)
            else:
                # Queue for when device comes online
                self.sync_queues[device_id].append(update_entry)
                result = {'success': True, 'sync_method': 'queued'}
                
            sync_results[device_id] = result
            
        return {
            'total_devices': len(device_ids),
            'successful_syncs': sum(1 for r in sync_results.values() if r['success']),
            'sync_results': sync_results
        }
    
    def _is_device_online(self, device_id: str) -> bool:
        """Check if device is currently online."""
        # Simulate online/offline status
        return hash(device_id) % 3 != 0  # 2/3 devices online
    
    def _execute_cloud_sync(self, sync_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Execute synchronization to cloud."""
        # Simulate cloud sync
        return {
            'success': True,
            'sync_method': 'direct',
            'sync_duration_ms': 150.0,
            'bytes_transferred': len(str(sync_entry['data'])) * 8
        }
    
    def _execute_device_sync(
        self,
        device_id: str,
        sync_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute synchronization to device."""
        # Simulate device sync
        return {
            'success': True,
            'sync_method': 'direct',
            'sync_duration_ms': 200.0,
            'bytes_transferred': len(str(sync_entry['data'])) * 8
        }


class MobileEdgeOptimizer:
    """Optimizer for mobile edge computing scenarios."""
    
    def __init__(self):
        self.mobile_profiles: Dict[str, Dict[str, Any]] = {}
        self.battery_optimization: Dict[str, Dict[str, Any]] = {}
        self.network_adaptation: Dict[str, Dict[str, Any]] = {}
        
    def create_mobile_profile(
        self,
        device_id: str,
        battery_capacity_mah: int,
        screen_size_inches: float,
        cpu_cores: int,
        ram_gb: int
    ) -> Dict[str, Any]:
        """Create optimization profile for mobile device."""
        
        mobile_profile = {
            'device_id': device_id,
            'battery_capacity_mah': battery_capacity_mah,
            'screen_size_inches': screen_size_inches,
            'cpu_cores': cpu_cores,
            'ram_gb': ram_gb,
            
            # Computed optimization parameters
            'power_budget_mw': battery_capacity_mah * 3.7 * 0.1,  # 10% of battery per hour
            'processing_budget_ms': 100.0,  # Max processing time per inference
            'memory_budget_mb': ram_gb * 1024 * 0.3,  # 30% of RAM
            
            # Adaptive parameters
            'cpu_throttling_enabled': True,
            'background_processing_enabled': False,
            'network_compression_enabled': True
        }
        
        self.mobile_profiles[device_id] = mobile_profile
        return mobile_profile
    
    def optimize_for_battery_life(
        self,
        device_id: str,
        current_battery_percent: float,
        target_runtime_hours: float
    ) -> Dict[str, Any]:
        """Optimize processing for battery life."""
        
        if device_id not in self.mobile_profiles:
            return {'error': 'Mobile profile not found'}
            
        profile = self.mobile_profiles[device_id]
        
        # Calculate power budget based on remaining battery
        remaining_capacity = profile['battery_capacity_mah'] * (current_battery_percent / 100)
        power_budget_per_hour = remaining_capacity * 3.7 / target_runtime_hours
        
        # Optimization strategies
        optimizations = {
            'power_budget_mw': power_budget_per_hour,
            'cpu_frequency_percent': min(100, max(20, power_budget_per_hour / 10)),
            'inference_interval_seconds': 5.0 if power_budget_per_hour > 500 else 10.0,
            'model_precision': 'int8' if power_budget_per_hour < 300 else 'fp16',
            'background_sync_enabled': power_budget_per_hour > 800
        }
        
        self.battery_optimization[device_id] = optimizations
        return optimizations
    
    def adapt_to_network_conditions(
        self,
        device_id: str,
        network_type: str,
        bandwidth_mbps: float,
        latency_ms: float
    ) -> Dict[str, Any]:
        """Adapt processing based on network conditions."""
        
        network_adaptations = {
            'compression_level': 'high' if bandwidth_mbps < 1.0 else 'medium',
            'batch_size': 1 if latency_ms > 200 else 5,
            'local_processing_ratio': 0.9 if bandwidth_mbps < 0.5 else 0.7,
            'sync_frequency_minutes': 10 if bandwidth_mbps < 1.0 else 5
        }
        
        # Network-specific optimizations
        if network_type == '5G':
            network_adaptations['edge_cloud_enabled'] = True
            network_adaptations['real_time_sync'] = True
        elif network_type == '4G':
            network_adaptations['edge_cloud_enabled'] = True
            network_adaptations['real_time_sync'] = bandwidth_mbps > 5.0
        elif network_type == 'WiFi':
            network_adaptations['edge_cloud_enabled'] = True
            network_adaptations['real_time_sync'] = True
        else:  # 3G or slower
            network_adaptations['edge_cloud_enabled'] = False
            network_adaptations['real_time_sync'] = False
            
        self.network_adaptation[device_id] = network_adaptations
        return network_adaptations


def create_edge_deployment_fleet() -> List[EdgeDeviceConfig]:
    """Create a fleet of diverse edge devices for testing."""
    
    fleet = [
        EdgeDeviceConfig(
            device_id='industrial_hub_001',
            device_type=EdgeDeviceType.INDUSTRIAL_GATEWAY,
            location='Manufacturing Plant A',
            cpu_cores=8,
            memory_mb=4096,
            storage_gb=64,
            has_gpu=True,
            network_bandwidth_mbps=100.0,
            ethernet_enabled=True,
            compute_capability=EdgeComputeCapability.ENHANCED,
            compliance_certifications=['IEC 61508', 'ATEX']
        ),
        
        EdgeDeviceConfig(
            device_id='iot_sensor_hub_002',
            device_type=EdgeDeviceType.IOT_SENSOR_HUB,
            location='Warehouse B',
            cpu_cores=4,
            memory_mb=2048,
            storage_gb=32,
            network_bandwidth_mbps=10.0,
            wifi_enabled=True,
            cellular_enabled=True,
            compute_capability=EdgeComputeCapability.STANDARD,
            battery_powered=True,
            battery_life_hours=72.0
        ),
        
        EdgeDeviceConfig(
            device_id='mobile_inspector_003',
            device_type=EdgeDeviceType.MOBILE_DEVICE,
            location='Field Inspection Unit',
            cpu_cores=6,
            memory_mb=6144,
            storage_gb=128,
            network_bandwidth_mbps=50.0,
            cellular_enabled=True,
            wifi_enabled=True,
            compute_capability=EdgeComputeCapability.ENHANCED,
            battery_powered=True,
            battery_life_hours=12.0
        ),
        
        EdgeDeviceConfig(
            device_id='neuromorphic_edge_004',
            device_type=EdgeDeviceType.NEUROMORPHIC_CHIP,
            location='Research Lab C',
            cpu_cores=2,
            memory_mb=1024,
            storage_gb=16,
            has_neuromorphic_chip=True,
            network_bandwidth_mbps=10.0,
            wifi_enabled=True,
            compute_capability=EdgeComputeCapability.NEUROMORPHIC,
            inference_latency_ms=5.0,
            max_concurrent_inferences=100
        )
    ]
    
    return fleet


if __name__ == "__main__":
    # Demonstrate edge computing integration
    print("📱 Edge Computing Integration for Neuromorphic Gas Detection")
    print("=" * 70)
    
    # Create edge manager
    edge_manager = EdgeComputingManager()
    
    # Register edge device fleet
    fleet = create_edge_deployment_fleet()
    for device in fleet:
        edge_manager.register_edge_device(device)
        
    print(f"✅ Registered {len(fleet)} edge devices")
    
    # Deploy to edge devices
    neuromorphic_model = {
        'model_version': '2.0.0',
        'memory_requirements_mb': 512,
        'storage_requirements_gb': 2,
        'compute_capability': 'standard',
        'requires_neuromorphic_chip': False
    }
    
    deployment_results = []
    for device in fleet[:3]:  # Deploy to first 3 devices
        result = edge_manager.deploy_to_edge_device(
            device.device_id, neuromorphic_model
        )
        deployment_results.append(result)
        
    successful_deployments = sum(1 for r in deployment_results if r['success'])
    print(f"✅ Deployed to {successful_deployments}/{len(deployment_results)} edge devices")
    
    # Monitor edge devices
    monitoring_summary = edge_manager.monitor_edge_devices()
    print(f"✅ Edge monitoring: {monitoring_summary['active_devices']} active devices")
    print(f"   Average health score: {monitoring_summary['average_health_score']:.3f}")
    
    # Demonstrate mobile optimization
    mobile_optimizer = MobileEdgeOptimizer()
    mobile_profile = mobile_optimizer.create_mobile_profile(
        'mobile_inspector_003', 5000, 6.5, 6, 8
    )
    
    battery_optimization = mobile_optimizer.optimize_for_battery_life(
        'mobile_inspector_003', 45.0, 8.0
    )
    
    print(f"✅ Mobile optimization configured")
    print(f"   Power budget: {battery_optimization['power_budget_mw']:.1f} mW")
    print(f"   CPU frequency: {battery_optimization['cpu_frequency_percent']:.1f}%")
    
    # Test edge-cloud synchronization
    sync_manager = EdgeCloudSynchronizer()
    sync_manager.configure_sync_policy(
        'industrial_hub_001',
        {'sync_interval': 60, 'compression': True}
    )
    
    sync_result = sync_manager.sync_sensor_data(
        'industrial_hub_001',
        [{'sensor_id': 'MQ2', 'value': 250, 'timestamp': time.time()}]
    )
    
    print(f"✅ Edge-cloud sync: {sync_result['sync_method']}")
    
    print("\n✅ Edge computing integration demonstration complete!")