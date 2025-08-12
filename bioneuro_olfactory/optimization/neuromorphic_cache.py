"""
Neuromorphic-specific caching system with spike train patterns and weight matrices optimization.

This module implements advanced caching strategies optimized for neuromorphic computing
workloads, including spike train pattern recognition, weight matrix caching, and 
temporal locality exploitation.
"""

import numpy as np
import torch
import time
import hashlib
import threading
import pickle
import gzip
import mmap
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, OrderedDict
from abc import ABC, abstractmethod
import psutil
import weakref
import struct
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CacheType(Enum):
    """Types of neuromorphic cache entries."""
    SPIKE_TRAIN = "spike_train"
    WEIGHT_MATRIX = "weight_matrix" 
    NETWORK_STATE = "network_state"
    INFERENCE_RESULT = "inference_result"
    PATTERN_TEMPLATE = "pattern_template"
    ENCODED_INPUT = "encoded_input"


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    access_time_total_ms: float = 0.0
    compression_ratio: float = 1.0
    pattern_matches: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(total, 1)
    
    @property
    def avg_access_time_ms(self) -> float:
        total_accesses = self.hits + self.misses
        return self.access_time_total_ms / max(total_accesses, 1)


@dataclass
class SpikePattern:
    """Spike pattern for pattern matching and compression."""
    pattern_id: str
    spike_times: np.ndarray
    neuron_indices: np.ndarray
    duration_ms: float
    frequency: float
    similarity_threshold: float = 0.95
    usage_count: int = 0
    last_used: float = field(default_factory=time.time)
    
    def matches(self, spike_data: 'SpikeData', tolerance_ms: float = 2.0) -> bool:
        """Check if spike data matches this pattern."""
        if abs(self.duration_ms - spike_data.duration_ms) > tolerance_ms:
            return False
            
        # Check temporal similarity
        temporal_similarity = self._compute_temporal_similarity(spike_data)
        spatial_similarity = self._compute_spatial_similarity(spike_data)
        
        combined_similarity = 0.7 * temporal_similarity + 0.3 * spatial_similarity
        return combined_similarity >= self.similarity_threshold
    
    def _compute_temporal_similarity(self, spike_data: 'SpikeData') -> float:
        """Compute temporal similarity between patterns."""
        if len(self.spike_times) == 0 or len(spike_data.spike_times) == 0:
            return 0.0
            
        # Use correlation-based similarity
        pattern_hist, _ = np.histogram(
            self.spike_times, bins=50, range=(0, self.duration_ms)
        )
        data_hist, _ = np.histogram(
            spike_data.spike_times, bins=50, range=(0, spike_data.duration_ms)
        )
        
        correlation = np.corrcoef(pattern_hist, data_hist)[0, 1]
        return max(0, correlation)
    
    def _compute_spatial_similarity(self, spike_data: 'SpikeData') -> float:
        """Compute spatial similarity between patterns."""
        pattern_neurons = set(self.neuron_indices)
        data_neurons = set(spike_data.neuron_indices)
        
        intersection = len(pattern_neurons.intersection(data_neurons))
        union = len(pattern_neurons.union(data_neurons))
        
        return intersection / max(union, 1)
    
    def update_usage(self):
        """Update usage statistics."""
        self.usage_count += 1
        self.last_used = time.time()


class CacheEntry:
    """Generic cache entry with metadata."""
    
    def __init__(self, key: str, data: Any, cache_type: CacheType, 
                 ttl_seconds: float = 3600.0, compressed: bool = False):
        self.key = key
        self.data = data
        self.cache_type = cache_type
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
        self.ttl_seconds = ttl_seconds
        self.compressed = compressed
        self.size_bytes = self._calculate_size()
        
    def _calculate_size(self) -> int:
        """Calculate memory size of cached data."""
        try:
            if isinstance(self.data, np.ndarray):
                return self.data.nbytes
            elif isinstance(self.data, torch.Tensor):
                return self.data.element_size() * self.data.nelement()
            else:
                return len(pickle.dumps(self.data))
        except Exception:
            return 1024  # Default estimate
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def access(self) -> Any:
        """Access cached data and update statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
        
        if self.compressed:
            return pickle.loads(gzip.decompress(self.data))
        return self.data
    
    def get_priority_score(self) -> float:
        """Calculate priority score for LRU/LFU eviction."""
        age_factor = (time.time() - self.last_accessed) / 3600.0  # Hours
        frequency_factor = self.access_count
        size_factor = self.size_bytes / (1024 * 1024)  # MB
        
        # Higher score = higher priority (less likely to evict)
        return frequency_factor * 2 - age_factor - size_factor * 0.1


class NeuromorphicCache:
    """Advanced neuromorphic-specific caching system."""
    
    def __init__(self, 
                 max_memory_mb: int = 512,
                 enable_compression: bool = True,
                 enable_pattern_matching: bool = True,
                 pattern_similarity_threshold: float = 0.95,
                 max_patterns: int = 1000):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.enable_compression = enable_compression
        self.enable_pattern_matching = enable_pattern_matching
        self.pattern_similarity_threshold = pattern_similarity_threshold
        self.max_patterns = max_patterns
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.type_indices: Dict[CacheType, set] = {ct: set() for ct in CacheType}
        
        # Pattern matching
        self.spike_patterns: Dict[str, SpikePattern] = {}
        self.pattern_cache: Dict[str, str] = {}  # spike_data_hash -> pattern_id
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Metrics
        self.metrics = CacheMetrics()
        
        # Background cleanup
        self._cleanup_thread = None
        self._running = False
        
    def start(self):
        """Start cache background processes."""
        with self._lock:
            if not self._running:
                self._running = True
                self._cleanup_thread = threading.Thread(
                    target=self._cleanup_loop, daemon=True
                )
                self._cleanup_thread.start()
                logger.info("Neuromorphic cache started")
    
    def stop(self):
        """Stop cache background processes."""
        with self._lock:
            self._running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=5.0)
            logger.info("Neuromorphic cache stopped")
    
    def put(self, key: str, data: Any, cache_type: CacheType, 
            ttl_seconds: float = 3600.0) -> bool:
        """Store data in cache with neuromorphic optimizations."""
        start_time = time.perf_counter()
        
        with self._lock:
            # Check if we need to evict entries
            if self._get_current_memory_usage() > self.max_memory_bytes * 0.9:
                self._evict_entries()
            
            # Handle spike train pattern matching
            if cache_type == CacheType.SPIKE_TRAIN and self.enable_pattern_matching:
                pattern_id = self._find_or_create_pattern(data)
                if pattern_id:
                    # Use pattern-based compression
                    data = self._compress_with_pattern(data, pattern_id)
                    key = f"{key}_pattern_{pattern_id}"
            
            # Apply compression if enabled
            compressed = False
            if self.enable_compression and self._should_compress(data, cache_type):
                data = gzip.compress(pickle.dumps(data))
                compressed = True
            
            # Create cache entry
            entry = CacheEntry(key, data, cache_type, ttl_seconds, compressed)
            
            # Store in cache
            self.cache[key] = entry
            self.type_indices[cache_type].add(key)
            
            # Update metrics
            self.metrics.memory_usage_bytes = self._get_current_memory_usage()
            access_time = (time.perf_counter() - start_time) * 1000
            self.metrics.access_time_total_ms += access_time
            
            return True
    
    def get(self, key: str, cache_type: Optional[CacheType] = None) -> Optional[Any]:
        """Retrieve data from cache with pattern matching."""
        start_time = time.perf_counter()
        
        with self._lock:
            # Direct lookup
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    self._remove_entry(key)
                    self.metrics.misses += 1
                    return None
                
                self.metrics.hits += 1
                access_time = (time.perf_counter() - start_time) * 1000
                self.metrics.access_time_total_ms += access_time
                
                return entry.access()
            
            # Pattern-based lookup for spike trains
            if cache_type == CacheType.SPIKE_TRAIN and self.enable_pattern_matching:
                pattern_key = self._find_pattern_match(key)
                if pattern_key and pattern_key in self.cache:
                    entry = self.cache[pattern_key]
                    if not entry.is_expired():
                        self.metrics.hits += 1
                        self.metrics.pattern_matches += 1
                        access_time = (time.perf_counter() - start_time) * 1000
                        self.metrics.access_time_total_ms += access_time
                        
                        return entry.access()
            
            # Cache miss
            self.metrics.misses += 1
            access_time = (time.perf_counter() - start_time) * 1000
            self.metrics.access_time_total_ms += access_time
            
            return None
    
    def get_by_type(self, cache_type: CacheType, limit: int = 100) -> List[Tuple[str, Any]]:
        """Get all cached entries of specified type."""
        with self._lock:
            results = []
            keys = list(self.type_indices[cache_type])[:limit]
            
            for key in keys:
                if key in self.cache:
                    entry = self.cache[key]
                    if not entry.is_expired():
                        results.append((key, entry.access()))
                    else:
                        self._remove_entry(key)
            
            return results
    
    def invalidate(self, key: str) -> bool:
        """Remove specific entry from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def invalidate_by_type(self, cache_type: CacheType) -> int:
        """Remove all entries of specified type."""
        with self._lock:
            keys_to_remove = list(self.type_indices[cache_type])
            for key in keys_to_remove:
                self._remove_entry(key)
            return len(keys_to_remove)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            for type_index in self.type_indices.values():
                type_index.clear()
            self.spike_patterns.clear()
            self.pattern_cache.clear()
            self.metrics = CacheMetrics()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            memory_usage_mb = self.metrics.memory_usage_bytes / (1024 * 1024)
            memory_usage_percent = (
                self.metrics.memory_usage_bytes / self.max_memory_bytes * 100
            )
            
            type_counts = {
                cache_type.value: len(keys) 
                for cache_type, keys in self.type_indices.items()
            }
            
            return {
                'hit_rate': self.metrics.hit_rate,
                'hits': self.metrics.hits,
                'misses': self.metrics.misses,
                'evictions': self.metrics.evictions,
                'total_entries': len(self.cache),
                'memory_usage_mb': memory_usage_mb,
                'memory_usage_percent': memory_usage_percent,
                'avg_access_time_ms': self.metrics.avg_access_time_ms,
                'compression_ratio': self.metrics.compression_ratio,
                'pattern_matches': self.metrics.pattern_matches,
                'spike_patterns_count': len(self.spike_patterns),
                'type_distribution': type_counts
            }
    
    def _find_or_create_pattern(self, spike_data: Any) -> Optional[str]:
        """Find existing pattern or create new one for spike data."""
        if not hasattr(spike_data, 'spike_times') or not hasattr(spike_data, 'neuron_indices'):
            return None
        
        # Check existing patterns
        for pattern_id, pattern in self.spike_patterns.items():
            if pattern.matches(spike_data):
                pattern.update_usage()
                return pattern_id
        
        # Create new pattern if under limit
        if len(self.spike_patterns) < self.max_patterns:
            pattern_id = f"pattern_{len(self.spike_patterns):06d}"
            pattern = SpikePattern(
                pattern_id=pattern_id,
                spike_times=spike_data.spike_times.copy(),
                neuron_indices=spike_data.neuron_indices.copy(),
                duration_ms=spike_data.duration_ms,
                frequency=len(spike_data.spike_times) / (spike_data.duration_ms / 1000.0),
                similarity_threshold=self.pattern_similarity_threshold
            )
            self.spike_patterns[pattern_id] = pattern
            return pattern_id
        
        return None
    
    def _find_pattern_match(self, key: str) -> Optional[str]:
        """Find pattern-based cache key for spike data."""
        # This would be implemented with actual spike data comparison
        # For now, return None as pattern matching requires the actual data
        return None
    
    def _compress_with_pattern(self, spike_data: Any, pattern_id: str) -> Any:
        """Compress spike data using pattern-based encoding."""
        # Implementation would encode differences from pattern
        # For now, return original data
        return spike_data
    
    def _should_compress(self, data: Any, cache_type: CacheType) -> bool:
        """Determine if data should be compressed."""
        # Compress large arrays and network states
        if cache_type in [CacheType.WEIGHT_MATRIX, CacheType.NETWORK_STATE]:
            return True
        
        # Compress if data is large enough
        try:
            size_bytes = 0
            if isinstance(data, np.ndarray):
                size_bytes = data.nbytes
            elif isinstance(data, torch.Tensor):
                size_bytes = data.element_size() * data.nelement()
            
            return size_bytes > 1024  # Compress if > 1KB
        except Exception:
            return False
    
    def _get_current_memory_usage(self) -> int:
        """Calculate current memory usage of cache."""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    def _evict_entries(self):
        """Evict least valuable entries to free memory."""
        if not self.cache:
            return
        
        # Calculate target eviction amount (20% of max memory)
        target_free_bytes = int(self.max_memory_bytes * 0.2)
        freed_bytes = 0
        
        # Sort entries by priority (lower score = evict first)
        entries_by_priority = sorted(
            self.cache.items(),
            key=lambda x: x[1].get_priority_score()
        )
        
        for key, entry in entries_by_priority:
            if freed_bytes >= target_free_bytes:
                break
            
            freed_bytes += entry.size_bytes
            self._remove_entry(key)
            self.metrics.evictions += 1
        
        logger.debug(f"Evicted {self.metrics.evictions} entries, freed {freed_bytes} bytes")
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and indices."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            
            # Remove from type index
            self.type_indices[entry.cache_type].discard(key)
            
            # Update memory usage
            self.metrics.memory_usage_bytes = self._get_current_memory_usage()
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while self._running:
            try:
                with self._lock:
                    expired_keys = [
                        key for key, entry in self.cache.items() 
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
                
                # Sleep for 60 seconds before next cleanup
                for _ in range(600):  # 60 seconds in 0.1s intervals
                    if not self._running:
                        break
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
                time.sleep(5.0)


class WeightMatrixCache:
    """Specialized cache for neural network weight matrices."""
    
    def __init__(self, max_memory_mb: int = 256):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
    def store_weights(self, model_id: str, layer_weights: Dict[str, torch.Tensor],
                      metadata: Dict[str, Any] = None) -> bool:
        """Store weight matrices for a model layer."""
        with self._lock:
            # Calculate total size
            total_size = sum(
                w.element_size() * w.nelement() 
                for w in layer_weights.values()
            )
            
            if total_size > self.max_memory_bytes:
                return False
            
            # Check if eviction needed
            if self._get_current_usage() + total_size > self.max_memory_bytes:
                self._evict_lru()
            
            # Store weights
            entry = {
                'weights': {name: w.clone() for name, w in layer_weights.items()},
                'metadata': metadata or {},
                'created_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 0,
                'size_bytes': total_size
            }
            
            self.cache[model_id] = entry
            return True
    
    def load_weights(self, model_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load cached weight matrices."""
        with self._lock:
            if model_id in self.cache:
                entry = self.cache[model_id]
                entry['last_accessed'] = time.time()
                entry['access_count'] += 1
                return entry['weights']
            return None
    
    def _get_current_usage(self) -> int:
        """Get current memory usage."""
        return sum(entry['size_bytes'] for entry in self.cache.values())
    
    def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.cache:
            return
            
        # Sort by last accessed time
        lru_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest 25%
        num_to_remove = max(1, len(lru_entries) // 4)
        for i in range(num_to_remove):
            key, _ = lru_entries[i]
            del self.cache[key]


# Global cache instances
_neuromorphic_cache: Optional[NeuromorphicCache] = None
_weight_cache: Optional[WeightMatrixCache] = None


def get_neuromorphic_cache(max_memory_mb: int = 512) -> NeuromorphicCache:
    """Get global neuromorphic cache instance."""
    global _neuromorphic_cache
    
    if _neuromorphic_cache is None:
        _neuromorphic_cache = NeuromorphicCache(max_memory_mb=max_memory_mb)
        _neuromorphic_cache.start()
    
    return _neuromorphic_cache


def get_weight_cache(max_memory_mb: int = 256) -> WeightMatrixCache:
    """Get global weight matrix cache instance."""
    global _weight_cache
    
    if _weight_cache is None:
        _weight_cache = WeightMatrixCache(max_memory_mb=max_memory_mb)
    
    return _weight_cache


# Decorator for automatic caching of neuromorphic computations
def cache_neuromorphic_computation(cache_type: CacheType, ttl_seconds: float = 3600.0):
    """Decorator to automatically cache neuromorphic computation results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__name__}_{str(args)}_{str(sorted(kwargs.items()))}"
            key = hashlib.md5(key_data.encode()).hexdigest()
            
            cache = get_neuromorphic_cache()
            
            # Try to get from cache
            result = cache.get(key, cache_type)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result, cache_type, ttl_seconds)
            
            return result
        return wrapper
    return decorator