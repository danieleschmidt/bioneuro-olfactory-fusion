"""
Advanced connection pooling for sensor data streams with adaptive buffering.

This module provides high-performance connection pooling specifically optimized
for neuromorphic sensor data streams with adaptive buffering, backpressure control,
and intelligent data routing.
"""

import asyncio
import time
import threading
import queue
import logging
import socket
import struct
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import aiohttp
import websockets
import serial
import serial.tools.list_ports
from contextlib import asynccontextmanager
import weakref
import psutil

logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """Types of sensor connections."""
    SERIAL = "serial"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    HTTP_STREAM = "http_stream"
    I2C = "i2c"
    SPI = "spi"
    USB = "usb"


class SensorDataType(Enum):
    """Types of sensor data."""
    GAS_CONCENTRATION = "gas_concentration"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity" 
    PRESSURE = "pressure"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    MICROPHONE = "microphone"
    CAMERA = "camera"
    LIDAR = "lidar"


@dataclass
class ConnectionConfig:
    """Configuration for sensor connection."""
    connection_type: ConnectionType
    address: str
    port: Optional[int] = None
    baudrate: Optional[int] = 115200
    timeout_seconds: float = 5.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    buffer_size: int = 8192
    max_connections: int = 5
    keepalive_enabled: bool = True
    keepalive_interval_seconds: float = 30.0
    compression_enabled: bool = False
    encryption_enabled: bool = False
    
    # Adaptive buffering
    adaptive_buffer_enabled: bool = True
    min_buffer_size: int = 1024
    max_buffer_size: int = 65536
    buffer_growth_factor: float = 1.5
    
    # Quality of Service
    priority: int = 0  # Higher values = higher priority
    max_latency_ms: float = 100.0
    max_jitter_ms: float = 10.0


@dataclass
class SensorReading:
    """Individual sensor reading."""
    sensor_id: str
    timestamp: float
    data_type: SensorDataType
    value: Any
    unit: str
    quality: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sensor_id': self.sensor_id,
            'timestamp': self.timestamp,
            'data_type': self.data_type.value,
            'value': self.value,
            'unit': self.unit,
            'quality': self.quality,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create from dictionary."""
        return cls(
            sensor_id=data['sensor_id'],
            timestamp=data['timestamp'],
            data_type=SensorDataType(data['data_type']),
            value=data['value'],
            unit=data['unit'],
            quality=data.get('quality', 1.0),
            metadata=data.get('metadata', {})
        )


@dataclass
class ConnectionMetrics:
    """Connection performance metrics."""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    timeouts: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    avg_throughput_bps: float = 0.0
    buffer_overflows: int = 0
    buffer_underruns: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.connection_attempts == 0:
            return 0.0
        return self.successful_connections / self.connection_attempts
    
    @property
    def error_rate(self) -> float:
        total_operations = self.messages_sent + self.messages_received
        if total_operations == 0:
            return 0.0
        return self.errors / total_operations


class AdaptiveBuffer:
    """Adaptive buffer that grows and shrinks based on usage patterns."""
    
    def __init__(self, initial_size: int, min_size: int, max_size: int, 
                 growth_factor: float = 1.5):
        self.min_size = min_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        self.current_size = initial_size
        
        self._buffer = bytearray(initial_size)
        self._write_pos = 0
        self._read_pos = 0
        self._data_length = 0
        
        # Performance tracking
        self._overflow_count = 0
        self._underrun_count = 0
        self._resize_count = 0
        self._last_resize_time = time.time()
        self._usage_history = deque(maxlen=100)
        
    def write(self, data: bytes) -> bool:
        """Write data to buffer, growing if necessary."""
        data_len = len(data)
        
        if self._data_length + data_len > self.current_size:
            if not self._try_grow(data_len):
                self._overflow_count += 1
                return False
        
        # Handle circular buffer wraparound
        if self._write_pos + data_len > self.current_size:
            # Split write across buffer boundary
            first_part = self.current_size - self._write_pos
            self._buffer[self._write_pos:] = data[:first_part]
            self._buffer[:data_len - first_part] = data[first_part:]
            self._write_pos = data_len - first_part
        else:
            self._buffer[self._write_pos:self._write_pos + data_len] = data
            self._write_pos = (self._write_pos + data_len) % self.current_size
        
        self._data_length += data_len
        self._record_usage()
        return True
    
    def read(self, size: int) -> bytes:
        """Read data from buffer."""
        if size > self._data_length:
            self._underrun_count += 1
            size = self._data_length
        
        if size == 0:
            return b''
        
        # Handle circular buffer wraparound
        if self._read_pos + size > self.current_size:
            # Split read across buffer boundary
            first_part = self.current_size - self._read_pos
            data = bytes(self._buffer[self._read_pos:]) + bytes(self._buffer[:size - first_part])
            self._read_pos = size - first_part
        else:
            data = bytes(self._buffer[self._read_pos:self._read_pos + size])
            self._read_pos = (self._read_pos + size) % self.current_size
        
        self._data_length -= size
        self._consider_shrinking()
        return data
    
    def peek(self, size: int = 0) -> bytes:
        """Peek at data without consuming it."""
        if size == 0:
            size = self._data_length
        
        size = min(size, self._data_length)
        if size == 0:
            return b''
        
        # Handle circular buffer wraparound
        if self._read_pos + size > self.current_size:
            first_part = self.current_size - self._read_pos
            return bytes(self._buffer[self._read_pos:]) + bytes(self._buffer[:size - first_part])
        else:
            return bytes(self._buffer[self._read_pos:self._read_pos + size])
    
    def available(self) -> int:
        """Get amount of data available to read."""
        return self._data_length
    
    def free_space(self) -> int:
        """Get amount of free space in buffer."""
        return self.current_size - self._data_length
    
    def clear(self):
        """Clear all data from buffer."""
        self._write_pos = 0
        self._read_pos = 0
        self._data_length = 0
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get buffer usage statistics."""
        if self._usage_history:
            avg_usage = sum(self._usage_history) / len(self._usage_history)
            max_usage = max(self._usage_history)
        else:
            avg_usage = max_usage = 0.0
        
        return {
            'current_size': self.current_size,
            'data_length': self._data_length,
            'usage_percent': (self._data_length / self.current_size) * 100,
            'avg_usage_percent': avg_usage,
            'max_usage_percent': max_usage,
            'overflow_count': self._overflow_count,
            'underrun_count': self._underrun_count,
            'resize_count': self._resize_count
        }
    
    def _try_grow(self, required_space: int) -> bool:
        """Try to grow buffer to accommodate required space."""
        if self.current_size >= self.max_size:
            return False
        
        # Calculate new size
        required_total = self._data_length + required_space
        new_size = self.current_size
        
        while new_size < required_total and new_size < self.max_size:
            new_size = min(int(new_size * self.growth_factor), self.max_size)
        
        if new_size <= self.current_size:
            return False
        
        # Resize buffer
        self._resize_buffer(new_size)
        return True
    
    def _consider_shrinking(self):
        """Consider shrinking buffer if underutilized."""
        current_time = time.time()
        
        # Only consider shrinking every 30 seconds
        if current_time - self._last_resize_time < 30.0:
            return
        
        # Check if buffer is consistently underutilized
        if len(self._usage_history) >= 10:
            avg_usage = sum(self._usage_history[-10:]) / 10
            
            # Shrink if usage is consistently below 25%
            if avg_usage < 25.0 and self.current_size > self.min_size:
                new_size = max(
                    self.min_size,
                    int(self.current_size / self.growth_factor)
                )
                self._resize_buffer(new_size)
    
    def _resize_buffer(self, new_size: int):
        """Resize the buffer, preserving existing data."""
        if new_size == self.current_size:
            return
        
        # Create new buffer
        new_buffer = bytearray(new_size)
        
        # Copy existing data
        if self._data_length > 0:
            data = self.peek(self._data_length)
            new_buffer[:len(data)] = data
        
        # Update buffer and positions
        self._buffer = new_buffer
        self.current_size = new_size
        self._read_pos = 0
        self._write_pos = self._data_length
        
        self._resize_count += 1
        self._last_resize_time = time.time()
        
        logger.debug(f"Resized buffer from {self.current_size} to {new_size} bytes")
    
    def _record_usage(self):
        """Record current buffer usage for statistics."""
        usage_percent = (self._data_length / self.current_size) * 100
        self._usage_history.append(usage_percent)


class SensorConnection(ABC):
    """Abstract base class for sensor connections."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection_id = f"{config.connection_type.value}_{id(self)}"
        self.is_connected = False
        self.last_activity = time.time()
        self.metrics = ConnectionMetrics()
        
        # Adaptive buffering
        if config.adaptive_buffer_enabled:
            self.buffer = AdaptiveBuffer(
                config.buffer_size,
                config.min_buffer_size,
                config.max_buffer_size,
                config.buffer_growth_factor
            )
        else:
            self.buffer = None
        
        # Data processing
        self._data_callbacks: List[Callable[[SensorReading], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []
        
        # Quality monitoring
        self._latency_history = deque(maxlen=100)
        self._throughput_history = deque(maxlen=100)
        self._quality_history = deque(maxlen=100)
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to sensor."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to sensor."""
        pass
    
    @abstractmethod
    async def send_data(self, data: bytes) -> bool:
        """Send data to sensor."""
        pass
    
    @abstractmethod
    async def receive_data(self) -> Optional[bytes]:
        """Receive data from sensor."""
        pass
    
    def add_data_callback(self, callback: Callable[[SensorReading], None]):
        """Add callback for received sensor data."""
        self._data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """Add callback for connection errors."""
        self._error_callbacks.append(callback)
    
    def _notify_data(self, reading: SensorReading):
        """Notify all data callbacks."""
        for callback in self._data_callbacks:
            try:
                callback(reading)
            except Exception as e:
                logger.error(f"Error in data callback: {e}")
    
    def _notify_error(self, error: Exception):
        """Notify all error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def is_idle(self, idle_timeout_seconds: float = 300.0) -> bool:
        """Check if connection has been idle."""
        return (time.time() - self.last_activity) > idle_timeout_seconds
    
    def record_latency(self, latency_ms: float):
        """Record latency measurement."""
        self._latency_history.append(latency_ms)
        if self._latency_history:
            self.metrics.avg_latency_ms = sum(self._latency_history) / len(self._latency_history)
    
    def record_throughput(self, bytes_per_second: float):
        """Record throughput measurement."""
        self._throughput_history.append(bytes_per_second)
        if self._throughput_history:
            self.metrics.avg_throughput_bps = sum(self._throughput_history) / len(self._throughput_history)
    
    def get_quality_score(self) -> float:
        """Calculate connection quality score (0.0 to 1.0)."""
        # Base score
        quality_score = 1.0
        
        # Penalize high error rate
        quality_score *= (1.0 - min(0.5, self.metrics.error_rate))
        
        # Penalize timeouts
        if self.metrics.connection_attempts > 0:
            timeout_rate = self.metrics.timeouts / self.metrics.connection_attempts
            quality_score *= (1.0 - min(0.3, timeout_rate))
        
        # Penalize buffer issues
        if self.buffer:
            buffer_stats = self.buffer.get_usage_stats()
            overflow_penalty = min(0.2, buffer_stats['overflow_count'] / 100.0)
            underrun_penalty = min(0.1, buffer_stats['underrun_count'] / 100.0)
            quality_score *= (1.0 - overflow_penalty - underrun_penalty)
        
        return max(0.0, quality_score)


class SerialSensorConnection(SensorConnection):
    """Serial connection for sensors."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.serial_conn: Optional[serial.Serial] = None
    
    async def connect(self) -> bool:
        """Connect to serial sensor."""
        try:
            self.metrics.connection_attempts += 1
            
            self.serial_conn = serial.Serial(
                port=self.config.address,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout_seconds,
                write_timeout=self.config.timeout_seconds
            )
            
            if self.serial_conn.is_open:
                self.is_connected = True
                self.metrics.successful_connections += 1
                self.update_activity()
                logger.info(f"Serial connection established: {self.config.address}")
                return True
            
        except Exception as e:
            self.metrics.failed_connections += 1
            self._notify_error(e)
            logger.error(f"Failed to connect to serial sensor {self.config.address}: {e}")
        
        return False
    
    async def disconnect(self):
        """Disconnect from serial sensor."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
    
    async def send_data(self, data: bytes) -> bool:
        """Send data to serial sensor."""
        if not self.is_connected or not self.serial_conn:
            return False
        
        try:
            bytes_sent = self.serial_conn.write(data)
            self.metrics.bytes_sent += bytes_sent
            self.metrics.messages_sent += 1
            self.update_activity()
            return bytes_sent == len(data)
        
        except Exception as e:
            self.metrics.errors += 1
            self._notify_error(e)
            return False
    
    async def receive_data(self) -> Optional[bytes]:
        """Receive data from serial sensor."""
        if not self.is_connected or not self.serial_conn:
            return None
        
        try:
            if self.serial_conn.in_waiting > 0:
                data = self.serial_conn.read(self.serial_conn.in_waiting)
                if data:
                    self.metrics.bytes_received += len(data)
                    self.metrics.messages_received += 1
                    self.update_activity()
                    
                    if self.buffer:
                        self.buffer.write(data)
                    
                    return data
        
        except Exception as e:
            self.metrics.errors += 1
            self._notify_error(e)
        
        return None


class TCPSensorConnection(SensorConnection):
    """TCP connection for sensors."""
    
    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
    
    async def connect(self) -> bool:
        """Connect to TCP sensor."""
        try:
            self.metrics.connection_attempts += 1
            
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.address, self.config.port),
                timeout=self.config.timeout_seconds
            )
            
            self.is_connected = True
            self.metrics.successful_connections += 1
            self.update_activity()
            logger.info(f"TCP connection established: {self.config.address}:{self.config.port}")
            return True
        
        except Exception as e:
            self.metrics.failed_connections += 1
            self._notify_error(e)
            logger.error(f"Failed to connect to TCP sensor {self.config.address}:{self.config.port}: {e}")
        
        return False
    
    async def disconnect(self):
        """Disconnect from TCP sensor."""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.is_connected = False
    
    async def send_data(self, data: bytes) -> bool:
        """Send data to TCP sensor."""
        if not self.is_connected or not self.writer:
            return False
        
        try:
            self.writer.write(data)
            await self.writer.drain()
            self.metrics.bytes_sent += len(data)
            self.metrics.messages_sent += 1
            self.update_activity()
            return True
        
        except Exception as e:
            self.metrics.errors += 1
            self._notify_error(e)
            return False
    
    async def receive_data(self) -> Optional[bytes]:
        """Receive data from TCP sensor."""
        if not self.is_connected or not self.reader:
            return None
        
        try:
            data = await asyncio.wait_for(
                self.reader.read(self.config.buffer_size),
                timeout=0.1  # Short timeout for non-blocking read
            )
            
            if data:
                self.metrics.bytes_received += len(data)
                self.metrics.messages_received += 1
                self.update_activity()
                
                if self.buffer:
                    self.buffer.write(data)
                
                return data
        
        except asyncio.TimeoutError:
            # Normal timeout, no data available
            pass
        except Exception as e:
            self.metrics.errors += 1
            self._notify_error(e)
        
        return None


class SensorConnectionPool:
    """Connection pool manager for sensor data streams."""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.connections: Dict[str, SensorConnection] = {}
        self.connection_configs: Dict[str, ConnectionConfig] = {}
        
        # Connection management
        self._connection_lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._data_processing_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Load balancing and routing
        self.connection_weights: Dict[str, float] = {}  # Quality-based weights
        self.round_robin_index = 0
        
        # Statistics
        self.total_connections_created = 0
        self.total_connections_destroyed = 0
        self.total_data_processed = 0
        
        # Data routing
        self._data_handlers: Dict[SensorDataType, List[Callable]] = defaultdict(list)
        self._global_data_queue = asyncio.Queue(maxsize=10000)
    
    async def start(self):
        """Start the connection pool."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._data_processing_task = asyncio.create_task(self._data_processing_loop())
        
        logger.info("Sensor connection pool started")
    
    async def stop(self):
        """Stop the connection pool."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._data_processing_task:
            self._data_processing_task.cancel()
        
        # Close all connections
        for connection in self.connections.values():
            await connection.disconnect()
        
        self.connections.clear()
        logger.info("Sensor connection pool stopped")
    
    async def add_sensor(self, sensor_id: str, config: ConnectionConfig) -> bool:
        """Add a sensor to the connection pool."""
        async with self._connection_lock:
            if len(self.connections) >= self.max_connections:
                logger.warning("Connection pool at maximum capacity")
                return False
            
            if sensor_id in self.connections:
                logger.warning(f"Sensor {sensor_id} already exists in pool")
                return False
            
            # Create appropriate connection type
            if config.connection_type == ConnectionType.SERIAL:
                connection = SerialSensorConnection(config)
            elif config.connection_type == ConnectionType.TCP:
                connection = TCPSensorConnection(config)
            else:
                logger.error(f"Unsupported connection type: {config.connection_type}")
                return False
            
            # Add data callback
            connection.add_data_callback(self._handle_sensor_data)
            connection.add_error_callback(self._handle_connection_error)
            
            # Attempt connection
            if await connection.connect():
                self.connections[sensor_id] = connection
                self.connection_configs[sensor_id] = config
                self.connection_weights[sensor_id] = 1.0
                self.total_connections_created += 1
                
                logger.info(f"Added sensor {sensor_id} to connection pool")
                return True
            else:
                logger.error(f"Failed to connect to sensor {sensor_id}")
                return False
    
    async def remove_sensor(self, sensor_id: str) -> bool:
        """Remove a sensor from the connection pool."""
        async with self._connection_lock:
            if sensor_id not in self.connections:
                return False
            
            connection = self.connections[sensor_id]
            await connection.disconnect()
            
            del self.connections[sensor_id]
            del self.connection_configs[sensor_id]
            del self.connection_weights[sensor_id]
            
            self.total_connections_destroyed += 1
            logger.info(f"Removed sensor {sensor_id} from connection pool")
            return True
    
    def get_connection(self, sensor_id: str) -> Optional[SensorConnection]:
        """Get specific connection by sensor ID."""
        return self.connections.get(sensor_id)
    
    def get_best_connection(self, data_type: Optional[SensorDataType] = None) -> Optional[SensorConnection]:
        """Get the best available connection based on quality scores."""
        if not self.connections:
            return None
        
        # Filter by data type if specified
        candidates = list(self.connections.values())
        if data_type:
            # This would be implemented with sensor capability matching
            pass
        
        # Sort by quality score
        candidates.sort(key=lambda c: c.get_quality_score(), reverse=True)
        return candidates[0] if candidates else None
    
    async def send_to_sensor(self, sensor_id: str, data: bytes) -> bool:
        """Send data to specific sensor."""
        connection = self.get_connection(sensor_id)
        if connection and connection.is_connected:
            return await connection.send_data(data)
        return False
    
    async def broadcast_data(self, data: bytes) -> int:
        """Broadcast data to all connected sensors."""
        success_count = 0
        
        for connection in self.connections.values():
            if connection.is_connected:
                if await connection.send_data(data):
                    success_count += 1
        
        return success_count
    
    def add_data_handler(self, data_type: SensorDataType, handler: Callable[[SensorReading], None]):
        """Add handler for specific sensor data type."""
        self._data_handlers[data_type].append(handler)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        total_metrics = ConnectionMetrics()
        quality_scores = []
        connection_stats = {}
        
        for sensor_id, connection in self.connections.items():
            metrics = connection.metrics
            total_metrics.bytes_sent += metrics.bytes_sent
            total_metrics.bytes_received += metrics.bytes_received
            total_metrics.messages_sent += metrics.messages_sent
            total_metrics.messages_received += metrics.messages_received
            total_metrics.errors += metrics.errors
            total_metrics.timeouts += metrics.timeouts
            
            quality_scores.append(connection.get_quality_score())
            
            connection_stats[sensor_id] = {
                'connected': connection.is_connected,
                'quality_score': connection.get_quality_score(),
                'last_activity': connection.last_activity,
                'metrics': {
                    'bytes_sent': metrics.bytes_sent,
                    'bytes_received': metrics.bytes_received,
                    'error_rate': metrics.error_rate,
                    'avg_latency_ms': metrics.avg_latency_ms,
                    'avg_throughput_bps': metrics.avg_throughput_bps
                }
            }
            
            if connection.buffer:
                connection_stats[sensor_id]['buffer'] = connection.buffer.get_usage_stats()
        
        return {
            'total_connections': len(self.connections),
            'connected_count': sum(1 for c in self.connections.values() if c.is_connected),
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'total_connections_created': self.total_connections_created,
            'total_connections_destroyed': self.total_connections_destroyed,
            'total_data_processed': self.total_data_processed,
            'aggregate_metrics': total_metrics.__dict__,
            'connection_details': connection_stats
        }
    
    async def _health_check_loop(self):
        """Background health check for all connections."""
        while self._running:
            try:
                # Check connection health
                for sensor_id, connection in list(self.connections.items()):
                    if not connection.is_connected:
                        # Attempt reconnection
                        config = self.connection_configs.get(sensor_id)
                        if config and config.retry_attempts > 0:
                            logger.info(f"Attempting to reconnect sensor {sensor_id}")
                            if await connection.connect():
                                logger.info(f"Successfully reconnected sensor {sensor_id}")
                            else:
                                logger.warning(f"Failed to reconnect sensor {sensor_id}")
                    
                    # Update connection weights based on quality
                    quality_score = connection.get_quality_score()
                    self.connection_weights[sensor_id] = quality_score
                    
                    # Remove idle connections
                    if connection.is_idle():
                        logger.info(f"Removing idle sensor {sensor_id}")
                        await self.remove_sensor(sensor_id)
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)
    
    async def _data_processing_loop(self):
        """Background data processing loop."""
        while self._running:
            try:
                # Process queued sensor data
                reading = await asyncio.wait_for(
                    self._global_data_queue.get(),
                    timeout=1.0
                )
                
                self.total_data_processed += 1
                
                # Route data to appropriate handlers
                handlers = self._data_handlers.get(reading.data_type, [])
                for handler in handlers:
                    try:
                        handler(reading)
                    except Exception as e:
                        logger.error(f"Error in data handler: {e}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")
                await asyncio.sleep(1)
    
    def _handle_sensor_data(self, reading: SensorReading):
        """Handle received sensor data."""
        try:
            # Add to processing queue
            if not self._global_data_queue.full():
                asyncio.create_task(self._global_data_queue.put(reading))
            else:
                logger.warning("Global data queue full, dropping sensor reading")
        
        except Exception as e:
            logger.error(f"Error handling sensor data: {e}")
    
    def _handle_connection_error(self, error: Exception):
        """Handle connection errors."""
        logger.error(f"Connection error: {error}")


# Global connection pool instance
_sensor_pool: Optional[SensorConnectionPool] = None


async def get_sensor_pool(max_connections: int = 50) -> SensorConnectionPool:
    """Get global sensor connection pool instance."""
    global _sensor_pool
    
    if _sensor_pool is None:
        _sensor_pool = SensorConnectionPool(max_connections)
        await _sensor_pool.start()
    
    return _sensor_pool


# Context manager for sensor connection pool
@asynccontextmanager
async def sensor_pool_context(max_connections: int = 50):
    """Context manager for sensor connection pool lifecycle."""
    pool = SensorConnectionPool(max_connections)
    
    try:
        await pool.start()
        yield pool
    finally:
        await pool.stop()