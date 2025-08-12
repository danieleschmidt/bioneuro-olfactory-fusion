"""
Memory-mapped files for large spike datasets with efficient I/O patterns.

This module provides high-performance I/O operations for neuromorphic spike datasets
using memory mapping, compression, and optimized access patterns for temporal data.
"""

import os
import mmap
import struct
import time
import threading
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import deque
from abc import ABC, abstractmethod
import pickle
import gzip
import lz4.frame
import zstd
import h5py
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiofiles

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Types of compression for spike data."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    CUSTOM_DELTA = "custom_delta"  # Delta compression for spike times


class SpikeDataFormat(Enum):
    """Formats for spike data storage."""
    BINARY_DENSE = "binary_dense"        # Dense binary matrix
    BINARY_SPARSE = "binary_sparse"      # Sparse binary format
    TIME_NEURON_PAIRS = "time_neuron"    # (time, neuron_id) pairs
    COMPRESSED_SPARSE = "compressed_sparse"  # Compressed sparse format
    HDF5 = "hdf5"                        # HDF5 hierarchical format


@dataclass
class SpikeDataHeader:
    """Header for spike data files."""
    version: int = 1
    format_type: SpikeDataFormat = SpikeDataFormat.TIME_NEURON_PAIRS
    compression: CompressionType = CompressionType.NONE
    num_neurons: int = 0
    duration_ms: float = 0.0
    timestep_ms: float = 1.0
    num_spikes: int = 0
    data_offset: int = 64  # Offset to actual data
    metadata_size: int = 0
    checksum: int = 0
    
    def to_bytes(self) -> bytes:
        """Convert header to bytes."""
        return struct.pack(
            '<IIIIQDDQII',
            self.version,
            self.format_type.value.__hash__() & 0xFFFFFFFF,
            self.compression.value.__hash__() & 0xFFFFFFFF,
            self.num_neurons,
            self.num_spikes,
            self.duration_ms,
            self.timestep_ms,
            self.data_offset,
            self.metadata_size,
            self.checksum
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SpikeDataHeader':
        """Create header from bytes."""
        unpacked = struct.unpack('<IIIIQDDQII', data[:48])
        
        header = cls()
        header.version = unpacked[0]
        # Note: Enum reconstruction would need a lookup table in practice
        header.num_neurons = unpacked[3]
        header.num_spikes = unpacked[4]
        header.duration_ms = unpacked[5]
        header.timestep_ms = unpacked[6]
        header.data_offset = unpacked[7]
        header.metadata_size = unpacked[8]
        header.checksum = unpacked[9]
        
        return header


@dataclass 
class SpikeDataStats:
    """Statistics for spike dataset."""
    total_spikes: int = 0
    spike_rate_hz: float = 0.0
    active_neurons: int = 0
    max_spike_rate_per_neuron: float = 0.0
    avg_spike_rate_per_neuron: float = 0.0
    spike_time_range_ms: Tuple[float, float] = (0.0, 0.0)
    temporal_density_profile: Optional[np.ndarray] = None
    spatial_activity_profile: Optional[np.ndarray] = None
    file_size_bytes: int = 0
    compression_ratio: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_spikes': self.total_spikes,
            'spike_rate_hz': self.spike_rate_hz,
            'active_neurons': self.active_neurons,
            'max_spike_rate_per_neuron': self.max_spike_rate_per_neuron,
            'avg_spike_rate_per_neuron': self.avg_spike_rate_per_neuron,
            'spike_time_range_ms': self.spike_time_range_ms,
            'file_size_bytes': self.file_size_bytes,
            'compression_ratio': self.compression_ratio
        }


class SpikeDataCompressor:
    """Handles compression of spike data."""
    
    @staticmethod
    def compress_data(data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression_type == CompressionType.LZ4:
            return lz4.frame.compress(data)
        elif compression_type == CompressionType.ZSTD:
            return zstd.compress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    @staticmethod
    def decompress_data(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression_type == CompressionType.LZ4:
            return lz4.frame.decompress(data)
        elif compression_type == CompressionType.ZSTD:
            return zstd.decompress(data)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")
    
    @staticmethod
    def compress_spike_times_delta(spike_times: np.ndarray) -> bytes:
        """Delta compress spike times for temporal locality."""
        if len(spike_times) == 0:
            return b''
        
        # Sort spike times
        sorted_times = np.sort(spike_times)
        
        # Calculate deltas
        deltas = np.diff(sorted_times, prepend=sorted_times[0])
        
        # Compress deltas (they should be small integers)
        # Use variable-length encoding for small deltas
        compressed = bytearray()
        
        for delta in deltas:
            delta_int = int(delta * 1000)  # Convert to microseconds
            
            if delta_int < 128:
                # Single byte for small deltas
                compressed.append(delta_int)
            elif delta_int < 16384:
                # Two bytes for medium deltas
                compressed.append(0x80 | (delta_int >> 8))
                compressed.append(delta_int & 0xFF)
            else:
                # Four bytes for large deltas
                compressed.append(0xC0)
                compressed.extend(struct.pack('<I', delta_int))
        
        return bytes(compressed)
    
    @staticmethod
    def decompress_spike_times_delta(compressed_data: bytes) -> np.ndarray:
        """Decompress delta-compressed spike times."""
        if not compressed_data:
            return np.array([])
        
        deltas = []
        i = 0
        
        while i < len(compressed_data):
            first_byte = compressed_data[i]
            
            if first_byte < 0x80:
                # Single byte delta
                deltas.append(first_byte)
                i += 1
            elif first_byte < 0xC0:
                # Two byte delta
                if i + 1 < len(compressed_data):
                    delta = ((first_byte & 0x3F) << 8) | compressed_data[i + 1]
                    deltas.append(delta)
                    i += 2
                else:
                    break
            else:
                # Four byte delta
                if i + 4 < len(compressed_data):
                    delta = struct.unpack('<I', compressed_data[i+1:i+5])[0]
                    deltas.append(delta)
                    i += 5
                else:
                    break
        
        # Convert back to spike times
        delta_array = np.array(deltas, dtype=np.float32) / 1000.0  # Back to milliseconds
        spike_times = np.cumsum(delta_array)
        
        return spike_times


class MappedSpikeFile:
    """Memory-mapped spike data file for efficient access."""
    
    def __init__(self, filepath: Union[str, Path], mode: str = 'r'):
        self.filepath = Path(filepath)
        self.mode = mode
        self.file_handle: Optional[BinaryIO] = None
        self.memory_map: Optional[mmap.mmap] = None
        self.header: Optional[SpikeDataHeader] = None
        self.metadata: Dict[str, Any] = {}
        
        # Access statistics
        self.access_count = 0
        self.last_access_time = 0.0
        self.bytes_read = 0
        self.bytes_written = 0
        
        # Caching
        self._cache: Dict[str, Any] = {}
        self._cache_size_limit = 50 * 1024 * 1024  # 50MB cache
        self._cache_size = 0
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def open(self):
        """Open the memory-mapped file."""
        if self.memory_map is not None:
            return
        
        # Determine access mode
        if self.mode == 'r':
            file_mode = 'rb'
            map_access = mmap.ACCESS_READ
        elif self.mode == 'w':
            file_mode = 'r+b'
            map_access = mmap.ACCESS_WRITE
        elif self.mode == 'c':
            file_mode = 'r+b'  
            map_access = mmap.ACCESS_COPY
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        # Open file
        self.file_handle = open(self.filepath, file_mode)
        
        # Create memory map
        if self.file_handle.seekable():
            self.memory_map = mmap.mmap(
                self.file_handle.fileno(),
                0,
                access=map_access
            )
        
        # Read header if file exists and has data
        if self.filepath.exists() and self.filepath.stat().st_size > 64:
            self._read_header()
            self._read_metadata()
    
    def close(self):
        """Close the memory-mapped file."""
        if self.memory_map:
            self.memory_map.close()
            self.memory_map = None
        
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        
        self._cache.clear()
        self._cache_size = 0
    
    def create_new(self, num_neurons: int, duration_ms: float, 
                   format_type: SpikeDataFormat = SpikeDataFormat.TIME_NEURON_PAIRS,
                   compression: CompressionType = CompressionType.LZ4,
                   initial_size_mb: int = 100):
        """Create a new spike data file."""
        if self.mode == 'r':
            raise ValueError("Cannot create file in read-only mode")
        
        # Create header
        self.header = SpikeDataHeader(
            format_type=format_type,
            compression=compression,
            num_neurons=num_neurons,
            duration_ms=duration_ms
        )
        
        # Create file with initial size
        initial_size = initial_size_mb * 1024 * 1024
        with open(self.filepath, 'wb') as f:
            f.write(b'\x00' * initial_size)
        
        # Reopen with memory mapping
        if self.file_handle:
            self.file_handle.close()
        
        self.file_handle = open(self.filepath, 'r+b')
        self.memory_map = mmap.mmap(self.file_handle.fileno(), 0)
        
        # Write header
        self._write_header()
    
    def write_spikes(self, spike_times: np.ndarray, neuron_indices: np.ndarray,
                     chunk_id: Optional[str] = None) -> bool:
        """Write spike data to file."""
        if not self.header:
            raise RuntimeError("File not initialized")
        
        if self.mode == 'r':
            raise ValueError("Cannot write to read-only file")
        
        try:
            # Prepare data based on format
            if self.header.format_type == SpikeDataFormat.TIME_NEURON_PAIRS:
                data = self._encode_time_neuron_pairs(spike_times, neuron_indices)
            elif self.header.format_type == SpikeDataFormat.BINARY_SPARSE:
                data = self._encode_binary_sparse(spike_times, neuron_indices)
            else:
                raise ValueError(f"Unsupported format: {self.header.format_type}")
            
            # Compress if needed
            if self.header.compression != CompressionType.NONE:
                data = SpikeDataCompressor.compress_data(data, self.header.compression)
            
            # Find write position
            write_pos = self.header.data_offset
            if hasattr(self, '_current_write_pos'):
                write_pos = self._current_write_pos
            
            # Check if we need to extend file
            required_size = write_pos + len(data)
            if required_size > len(self.memory_map):
                self._extend_file(required_size)
            
            # Write data
            self.memory_map[write_pos:write_pos + len(data)] = data
            self.bytes_written += len(data)
            
            # Update header
            self.header.num_spikes += len(spike_times)
            self._current_write_pos = write_pos + len(data)
            
            # Cache chunk if specified
            if chunk_id:
                self._cache_chunk(chunk_id, spike_times, neuron_indices)
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing spikes: {e}")
            return False
    
    def read_spikes(self, start_time_ms: float = 0, end_time_ms: Optional[float] = None,
                    neuron_filter: Optional[List[int]] = None,
                    chunk_id: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Read spike data from file."""
        if not self.header:
            raise RuntimeError("File not initialized")
        
        # Check cache first
        cache_key = f"{start_time_ms}_{end_time_ms}_{neuron_filter}_{chunk_id}"
        if cache_key in self._cache:
            self.access_count += 1
            self.last_access_time = time.time()
            cached_data = self._cache[cache_key]
            return cached_data['spike_times'], cached_data['neuron_indices']
        
        try:
            # Read data based on format
            if self.header.format_type == SpikeDataFormat.TIME_NEURON_PAIRS:
                spike_times, neuron_indices = self._decode_time_neuron_pairs()
            elif self.header.format_type == SpikeDataFormat.BINARY_SPARSE:
                spike_times, neuron_indices = self._decode_binary_sparse()
            else:
                raise ValueError(f"Unsupported format: {self.header.format_type}")
            
            # Apply temporal filtering
            if start_time_ms > 0 or end_time_ms is not None:
                mask = spike_times >= start_time_ms
                if end_time_ms is not None:
                    mask &= spike_times <= end_time_ms
                
                spike_times = spike_times[mask]
                neuron_indices = neuron_indices[mask]
            
            # Apply neuron filtering
            if neuron_filter:
                mask = np.isin(neuron_indices, neuron_filter)
                spike_times = spike_times[mask]
                neuron_indices = neuron_indices[mask]
            
            # Update access stats
            self.access_count += 1
            self.last_access_time = time.time()
            self.bytes_read += len(spike_times) * 8 + len(neuron_indices) * 4
            
            # Cache result if it's not too large
            data_size = len(spike_times) * 8 + len(neuron_indices) * 4
            if data_size < 1024 * 1024:  # Cache if less than 1MB
                self._cache_result(cache_key, spike_times, neuron_indices, data_size)
            
            return spike_times, neuron_indices
            
        except Exception as e:
            logger.error(f"Error reading spikes: {e}")
            return np.array([]), np.array([])
    
    def get_stats(self) -> SpikeDataStats:
        """Get comprehensive statistics for the spike dataset."""
        if not self.header:
            return SpikeDataStats()
        
        # Read all spike data for statistics
        spike_times, neuron_indices = self.read_spikes()
        
        if len(spike_times) == 0:
            return SpikeDataStats()
        
        # Calculate statistics
        stats = SpikeDataStats()
        stats.total_spikes = len(spike_times)
        stats.spike_rate_hz = len(spike_times) / (self.header.duration_ms / 1000.0)
        stats.active_neurons = len(np.unique(neuron_indices))
        
        # Per-neuron statistics
        neuron_spike_counts = np.bincount(neuron_indices)
        nonzero_counts = neuron_spike_counts[neuron_spike_counts > 0]
        
        if len(nonzero_counts) > 0:
            max_spikes = np.max(nonzero_counts)
            avg_spikes = np.mean(nonzero_counts)
            
            stats.max_spike_rate_per_neuron = max_spikes / (self.header.duration_ms / 1000.0)
            stats.avg_spike_rate_per_neuron = avg_spikes / (self.header.duration_ms / 1000.0)
        
        # Temporal statistics
        stats.spike_time_range_ms = (float(np.min(spike_times)), float(np.max(spike_times)))
        
        # File statistics
        stats.file_size_bytes = self.filepath.stat().st_size if self.filepath.exists() else 0
        
        return stats
    
    def _read_header(self):
        """Read file header."""
        if not self.memory_map:
            return
        
        header_data = self.memory_map[:48]
        self.header = SpikeDataHeader.from_bytes(header_data)
    
    def _write_header(self):
        """Write file header."""
        if not self.memory_map or not self.header:
            return
        
        header_data = self.header.to_bytes()
        self.memory_map[:len(header_data)] = header_data
        self.memory_map.flush()
    
    def _read_metadata(self):
        """Read metadata from file."""
        if not self.header or self.header.metadata_size == 0:
            return
        
        metadata_start = 64  # After header
        metadata_end = metadata_start + self.header.metadata_size
        
        try:
            metadata_bytes = self.memory_map[metadata_start:metadata_end]
            if self.header.compression != CompressionType.NONE:
                metadata_bytes = SpikeDataCompressor.decompress_data(
                    metadata_bytes, self.header.compression
                )
            
            self.metadata = json.loads(metadata_bytes.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            self.metadata = {}
    
    def _write_metadata(self):
        """Write metadata to file."""
        if not self.metadata:
            return
        
        try:
            metadata_str = json.dumps(self.metadata)
            metadata_bytes = metadata_str.encode('utf-8')
            
            if self.header.compression != CompressionType.NONE:
                metadata_bytes = SpikeDataCompressor.compress_data(
                    metadata_bytes, self.header.compression
                )
            
            self.header.metadata_size = len(metadata_bytes)
            self.header.data_offset = 64 + len(metadata_bytes)
            
            # Write metadata after header
            metadata_start = 64
            metadata_end = metadata_start + len(metadata_bytes)
            self.memory_map[metadata_start:metadata_end] = metadata_bytes
            
            # Update header
            self._write_header()
            
        except Exception as e:
            logger.error(f"Error writing metadata: {e}")
    
    def _encode_time_neuron_pairs(self, spike_times: np.ndarray, 
                                 neuron_indices: np.ndarray) -> bytes:
        """Encode spike data as time-neuron pairs."""
        # Pack as alternating float64 (time) and uint32 (neuron_id)
        data = bytearray()
        
        for time_val, neuron_id in zip(spike_times, neuron_indices):
            data.extend(struct.pack('<dI', float(time_val), int(neuron_id)))
        
        return bytes(data)
    
    def _decode_time_neuron_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Decode time-neuron pairs format."""
        data_start = self.header.data_offset
        
        # Read compressed data if needed
        if self.header.compression != CompressionType.NONE:
            # For simplicity, read entire compressed block
            # In practice, you'd want streaming decompression
            compressed_size = len(self.memory_map) - data_start
            compressed_data = self.memory_map[data_start:data_start + compressed_size]
            data = SpikeDataCompressor.decompress_data(
                compressed_data, self.header.compression
            )
        else:
            data = self.memory_map[data_start:]
        
        # Parse time-neuron pairs
        num_pairs = len(data) // 12  # 8 bytes for double + 4 bytes for uint32
        
        spike_times = np.empty(num_pairs, dtype=np.float64)
        neuron_indices = np.empty(num_pairs, dtype=np.uint32)
        
        for i in range(num_pairs):
            offset = i * 12
            time_val, neuron_id = struct.unpack('<dI', data[offset:offset + 12])
            spike_times[i] = time_val
            neuron_indices[i] = neuron_id
        
        return spike_times, neuron_indices
    
    def _encode_binary_sparse(self, spike_times: np.ndarray, 
                             neuron_indices: np.ndarray) -> bytes:
        """Encode as sparse binary format."""
        # Create sparse representation
        # This is a simplified implementation
        data = bytearray()
        
        # Sort by time for better compression
        sort_indices = np.argsort(spike_times)
        sorted_times = spike_times[sort_indices]
        sorted_neurons = neuron_indices[sort_indices]
        
        # Group by time bins
        time_bin_size = self.header.timestep_ms
        time_bins = (sorted_times / time_bin_size).astype(int)
        
        current_bin = -1
        for i, (time_bin, neuron_id) in enumerate(zip(time_bins, sorted_neurons)):
            if time_bin != current_bin:
                # New time bin
                data.extend(struct.pack('<I', time_bin))
                current_bin = time_bin
            
            # Add neuron ID
            data.extend(struct.pack('<I', neuron_id))
        
        return bytes(data)
    
    def _decode_binary_sparse(self) -> Tuple[np.ndarray, np.ndarray]:
        """Decode sparse binary format."""
        # This would implement the reverse of _encode_binary_sparse
        # For now, return empty arrays
        return np.array([]), np.array([])
    
    def _extend_file(self, new_size: int):
        """Extend file size and remap memory."""
        # Close current mapping
        if self.memory_map:
            self.memory_map.close()
        
        # Extend file
        self.file_handle.seek(new_size - 1)
        self.file_handle.write(b'\x00')
        self.file_handle.flush()
        
        # Recreate memory map
        self.memory_map = mmap.mmap(self.file_handle.fileno(), 0)
    
    def _cache_chunk(self, chunk_id: str, spike_times: np.ndarray, 
                     neuron_indices: np.ndarray):
        """Cache a chunk of spike data."""
        data_size = len(spike_times) * 8 + len(neuron_indices) * 4
        
        # Check cache size limit
        if self._cache_size + data_size > self._cache_size_limit:
            self._evict_cache()
        
        self._cache[chunk_id] = {
            'spike_times': spike_times.copy(),
            'neuron_indices': neuron_indices.copy(),
            'access_time': time.time(),
            'size': data_size
        }
        self._cache_size += data_size
    
    def _cache_result(self, cache_key: str, spike_times: np.ndarray,
                      neuron_indices: np.ndarray, data_size: int):
        """Cache a query result."""
        if self._cache_size + data_size > self._cache_size_limit:
            self._evict_cache()
        
        self._cache[cache_key] = {
            'spike_times': spike_times.copy(),
            'neuron_indices': neuron_indices.copy(),
            'access_time': time.time(),
            'size': data_size
        }
        self._cache_size += data_size
    
    def _evict_cache(self):
        """Evict least recently used cache entries."""
        # Sort by access time
        items = sorted(
            self._cache.items(),
            key=lambda x: x[1]['access_time']
        )
        
        # Remove oldest 25%
        num_to_remove = max(1, len(items) // 4)
        for i in range(num_to_remove):
            key, value = items[i]
            self._cache_size -= value['size']
            del self._cache[key]


class SpikeDatasetManager:
    """Manager for multiple spike datasets with intelligent caching."""
    
    def __init__(self, base_path: Union[str, Path], max_open_files: int = 10):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_open_files = max_open_files
        
        # File management
        self.open_files: Dict[str, MappedSpikeFile] = {}
        self.file_access_times: Dict[str, float] = {}
        
        # Background processing
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._async_executor = ProcessPoolExecutor(max_workers=2)
        
        # Statistics
        self.total_files_opened = 0
        self.total_bytes_read = 0
        self.total_bytes_written = 0
    
    def create_dataset(self, dataset_id: str, num_neurons: int, duration_ms: float,
                       format_type: SpikeDataFormat = SpikeDataFormat.TIME_NEURON_PAIRS,
                       compression: CompressionType = CompressionType.LZ4) -> bool:
        """Create a new spike dataset."""
        filepath = self.base_path / f"{dataset_id}.spk"
        
        try:
            spike_file = MappedSpikeFile(filepath, 'w')
            spike_file.create_new(num_neurons, duration_ms, format_type, compression)
            
            # Add to open files
            self.open_files[dataset_id] = spike_file
            self.file_access_times[dataset_id] = time.time()
            self.total_files_opened += 1
            
            # Check if we need to close old files
            self._manage_open_files()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating dataset {dataset_id}: {e}")
            return False
    
    def get_dataset(self, dataset_id: str, mode: str = 'r') -> Optional[MappedSpikeFile]:
        """Get dataset file handle."""
        # Check if already open
        if dataset_id in self.open_files:
            self.file_access_times[dataset_id] = time.time()
            return self.open_files[dataset_id]
        
        # Open file
        filepath = self.base_path / f"{dataset_id}.spk"
        if not filepath.exists():
            return None
        
        try:
            spike_file = MappedSpikeFile(filepath, mode)
            spike_file.open()
            
            self.open_files[dataset_id] = spike_file
            self.file_access_times[dataset_id] = time.time()
            self.total_files_opened += 1
            
            self._manage_open_files()
            return spike_file
            
        except Exception as e:
            logger.error(f"Error opening dataset {dataset_id}: {e}")
            return None
    
    def close_dataset(self, dataset_id: str):
        """Close specific dataset."""
        if dataset_id in self.open_files:
            self.open_files[dataset_id].close()
            del self.open_files[dataset_id]
            del self.file_access_times[dataset_id]
    
    def close_all(self):
        """Close all open datasets."""
        for spike_file in self.open_files.values():
            spike_file.close()
        
        self.open_files.clear()
        self.file_access_times.clear()
        
        self._executor.shutdown(wait=True)
        self._async_executor.shutdown(wait=True)
    
    async def write_spikes_async(self, dataset_id: str, spike_times: np.ndarray,
                                neuron_indices: np.ndarray) -> bool:
        """Asynchronously write spike data."""
        spike_file = self.get_dataset(dataset_id, 'w')
        if not spike_file:
            return False
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            spike_file.write_spikes,
            spike_times,
            neuron_indices
        )
    
    async def read_spikes_async(self, dataset_id: str, start_time_ms: float = 0,
                               end_time_ms: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Asynchronously read spike data."""
        spike_file = self.get_dataset(dataset_id, 'r')
        if not spike_file:
            return np.array([]), np.array([])
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            spike_file.read_spikes,
            start_time_ms,
            end_time_ms
        )
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        total_cache_size = sum(
            spike_file._cache_size for spike_file in self.open_files.values()
        )
        
        total_access_count = sum(
            spike_file.access_count for spike_file in self.open_files.values()
        )
        
        dataset_stats = {}
        for dataset_id, spike_file in self.open_files.items():
            dataset_stats[dataset_id] = {
                'access_count': spike_file.access_count,
                'bytes_read': spike_file.bytes_read,
                'bytes_written': spike_file.bytes_written,
                'cache_size': spike_file._cache_size,
                'last_access': spike_file.last_access_time
            }
        
        return {
            'open_files_count': len(self.open_files),
            'total_files_opened': self.total_files_opened,
            'total_cache_size_bytes': total_cache_size,
            'total_access_count': total_access_count,
            'dataset_details': dataset_stats
        }
    
    def _manage_open_files(self):
        """Manage number of open files, closing least recently used."""
        while len(self.open_files) > self.max_open_files:
            # Find least recently used file
            lru_dataset = min(
                self.file_access_times.keys(),
                key=lambda k: self.file_access_times[k]
            )
            
            self.close_dataset(lru_dataset)


# Global dataset manager
_spike_dataset_manager: Optional[SpikeDatasetManager] = None


def get_spike_dataset_manager(base_path: Union[str, Path] = "./spike_datasets",
                             max_open_files: int = 10) -> SpikeDatasetManager:
    """Get global spike dataset manager."""
    global _spike_dataset_manager
    
    if _spike_dataset_manager is None:
        _spike_dataset_manager = SpikeDatasetManager(base_path, max_open_files)
    
    return _spike_dataset_manager