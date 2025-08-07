"""High-performance caching system for neuromorphic processing.

This module implements adaptive caching mechanisms optimized for
spiking neural networks and real-time gas detection scenarios.
"""

import torch
import numpy as np
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod
import hashlib
import pickle
import logging
import weakref
from enum import Enum
import psutil

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    TTL = "ttl"           # Time To Live


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    priority: float = 1.0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl
        
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_usage = 0
        self.start_time = time.time()
        
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / max(total, 1)
        
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        return 1.0 - self.hit_rate
        
    def reset(self):
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.start_time = time.time()


class BaseCache(ABC):
    """Abstract base class for cache implementations."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 512,
        policy: CachePolicy = CachePolicy.LRU
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.stats = CacheStats()
        self._lock = threading.RLock()
        self._entries: Dict[str, CacheEntry] = {}
        
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
        
    @abstractmethod
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in cache."""
        pass
        
    @abstractmethod
    def evict(self, key: str) -> bool:
        """Evict specific key from cache."""
        pass
        
    @abstractmethod
    def clear(self):
        """Clear all cache entries."""
        pass
        
    def _calculate_size(self, value: Any) -> int:
        """Calculate memory size of value."""
        try:
            if isinstance(value, torch.Tensor):
                return value.nelement() * value.element_size()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default size estimate
            
    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.sha256(key_data.encode()).hexdigest()
        
    @property
    def size(self) -> int:
        """Get number of entries in cache."""
        return len(self._entries)
        
    @property
    def memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        return sum(entry.size_bytes for entry in self._entries.values())
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': self.size,
            'max_size': self.max_size,
            'memory_usage_mb': self.memory_usage / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'hit_rate': self.stats.hit_rate,
            'miss_rate': self.stats.miss_rate,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'evictions': self.stats.evictions,
            'uptime_seconds': time.time() - self.stats.start_time
        }


class LRUCache(BaseCache):
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        super().__init__(max_size, max_memory_mb, CachePolicy.LRU)
        self._access_order = OrderedDict()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from LRU cache."""
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                
                # Check if expired
                if entry.is_expired():
                    self.evict(key)
                    self.stats.misses += 1
                    return None
                    
                # Update access and move to end
                entry.update_access()
                self._access_order.move_to_end(key)
                self.stats.hits += 1
                return entry.value
            else:
                self.stats.misses += 1
                return None
                
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in LRU cache."""
        with self._lock:
            entry_size = self._calculate_size(value)
            
            # Check if we need to evict entries
            while (len(self._entries) >= self.max_size or 
                   self.memory_usage + entry_size > self.max_memory_bytes):
                if not self._evict_lru():
                    return False  # Cannot make space
                    
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=entry_size,
                ttl=ttl
            )
            
            # Add to cache
            self._entries[key] = entry
            self._access_order[key] = True
            
            return True
            
    def evict(self, key: str) -> bool:
        """Evict specific key from cache."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                del self._access_order[key]
                self.stats.evictions += 1
                return True
            return False
            
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._access_order:
            return False
            
        lru_key = next(iter(self._access_order))
        return self.evict(lru_key)
        
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()


class AdaptiveCache(BaseCache):
    """Adaptive cache with intelligent eviction strategy."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 512):
        super().__init__(max_size, max_memory_mb, CachePolicy.ADAPTIVE)
        self._access_patterns = defaultdict(list)
        self._frequency_counter = defaultdict(int)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from adaptive cache."""
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                
                if entry.is_expired():
                    self.evict(key)
                    self.stats.misses += 1
                    return None
                    
                # Update access patterns
                now = time.time()
                self._access_patterns[key].append(now)
                
                # Keep only recent accesses (last hour)
                cutoff = now - 3600
                self._access_patterns[key] = [
                    t for t in self._access_patterns[key] if t > cutoff
                ]
                
                self._frequency_counter[key] += 1
                entry.update_access()
                self.stats.hits += 1
                return entry.value
            else:
                self.stats.misses += 1
                return None
                
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Put value in adaptive cache."""
        with self._lock:
            entry_size = self._calculate_size(value)
            
            # Evict based on adaptive strategy
            while (len(self._entries) >= self.max_size or 
                   self.memory_usage + entry_size > self.max_memory_bytes):
                if not self._evict_adaptive():
                    return False
                    
            # Calculate priority based on predicted future access
            priority = self._calculate_priority(key, value)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=entry_size,
                ttl=ttl,
                priority=priority
            )
            
            self._entries[key] = entry
            return True
            
    def evict(self, key: str) -> bool:
        """Evict specific key from cache."""
        with self._lock:
            if key in self._entries:
                del self._entries[key]
                if key in self._access_patterns:
                    del self._access_patterns[key]
                if key in self._frequency_counter:
                    del self._frequency_counter[key]
                self.stats.evictions += 1
                return True
            return False
            
    def _calculate_priority(self, key: str, value: Any) -> float:
        """Calculate entry priority for adaptive eviction."""
        priority = 1.0
        
        # Frequency-based component
        frequency = self._frequency_counter.get(key, 0)
        priority += np.log1p(frequency) * 0.3
        
        # Recency component
        if key in self._access_patterns and self._access_patterns[key]:
            last_access = max(self._access_patterns[key])
            recency = 1.0 / (time.time() - last_access + 1.0)
            priority += recency * 0.4
            
        # Size penalty (prefer smaller entries when memory is tight)
        size_factor = self._calculate_size(value) / (1024 * 1024)  # MB
        priority -= size_factor * 0.1
        
        # Type-specific bonuses
        if isinstance(value, torch.Tensor):
            # Neural network computations are expensive to recompute
            priority += 0.2
            
        return max(priority, 0.1)  # Minimum priority
        
    def _evict_adaptive(self) -> bool:
        """Evict entry using adaptive strategy."""
        if not self._entries:
            return False
            
        # Find entry with lowest priority
        min_priority = float('inf')
        evict_key = None
        
        for key, entry in self._entries.items():
            current_priority = entry.priority
            
            # Adjust priority based on age
            age_hours = (time.time() - entry.created_at) / 3600
            if age_hours > 1:
                current_priority *= (1.0 / age_hours)
                
            if current_priority < min_priority:
                min_priority = current_priority
                evict_key = key
                
        if evict_key:
            return self.evict(evict_key)
            
        return False
        
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
            self._access_patterns.clear()
            self._frequency_counter.clear()


class TensorCache(AdaptiveCache):
    """Specialized cache for PyTorch tensors with compression."""
    
    def __init__(self, max_size: int = 500, max_memory_mb: int = 1024):
        super().__init__(max_size, max_memory_mb)
        self._compression_enabled = True
        self._compression_threshold = 1024 * 1024  # 1MB
        
    def put(self, key: str, value: torch.Tensor, ttl: Optional[float] = None) -> bool:
        """Put tensor in cache with optional compression."""
        if not isinstance(value, torch.Tensor):
            return super().put(key, value, ttl)
            
        with self._lock:
            original_size = value.numel() * value.element_size()
            
            # Compress large tensors
            if (self._compression_enabled and 
                original_size > self._compression_threshold):
                # Move to CPU and compress
                cpu_tensor = value.detach().cpu()
                compressed_value = {
                    'data': cpu_tensor,
                    'device': value.device,
                    'requires_grad': value.requires_grad,
                    'compressed': True
                }
            else:
                compressed_value = {
                    'data': value.detach(),
                    'device': value.device,
                    'requires_grad': value.requires_grad,
                    'compressed': False
                }
                
            return super().put(key, compressed_value, ttl)
            
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache with decompression."""
        cached_data = super().get(key)
        
        if cached_data is None or not isinstance(cached_data, dict):
            return cached_data
            
        # Reconstruct tensor
        tensor = cached_data['data']
        
        # Restore device if needed
        if cached_data.get('compressed', False):
            tensor = tensor.to(cached_data['device'])
            
        # Restore grad requirements
        if cached_data.get('requires_grad', False):
            tensor.requires_grad_(True)
            
        return tensor


class SpikeTrainCache(TensorCache):
    """Specialized cache for spike trains with pattern recognition."""
    
    def __init__(self, max_size: int = 200, max_memory_mb: int = 512):
        super().__init__(max_size, max_memory_mb)
        self._pattern_cache = {}
        self._similarity_threshold = 0.95
        
    def get_similar(self, spike_pattern: torch.Tensor, threshold: float = None) -> Optional[torch.Tensor]:
        """Get cached spike train similar to input pattern."""
        if threshold is None:
            threshold = self._similarity_threshold
            
        with self._lock:
            pattern_hash = self._hash_spike_pattern(spike_pattern)
            
            # Check for exact match first
            exact_match = self.get(pattern_hash)
            if exact_match is not None:
                return exact_match
                
            # Search for similar patterns
            for key, entry in self._entries.items():
                cached_data = entry.value
                if isinstance(cached_data, dict) and 'data' in cached_data:
                    cached_tensor = cached_data['data']
                    
                    if self._are_patterns_similar(spike_pattern, cached_tensor, threshold):
                        entry.update_access()
                        return self.get(key)  # Use regular get for proper statistics
                        
            return None
            
    def put_pattern(self, spike_pattern: torch.Tensor, ttl: Optional[float] = None) -> str:
        """Put spike pattern in cache and return key."""
        pattern_hash = self._hash_spike_pattern(spike_pattern)
        self.put(pattern_hash, spike_pattern, ttl)
        return pattern_hash
        
    def _hash_spike_pattern(self, spike_pattern: torch.Tensor) -> str:
        """Create hash for spike pattern."""
        # Use statistical features for hashing to group similar patterns
        if spike_pattern.numel() == 0:
            return "empty"
            
        features = torch.tensor([
            spike_pattern.mean().item(),
            spike_pattern.std().item(),
            spike_pattern.sum().item(),
            float(spike_pattern.shape[-1]) if spike_pattern.dim() > 0 else 0.0
        ])
        
        # Quantize features for grouping
        quantized = torch.round(features * 1000) / 1000
        return hashlib.sha256(str(quantized.tolist()).encode()).hexdigest()[:16]
        
    def _are_patterns_similar(
        self, 
        pattern1: torch.Tensor, 
        pattern2: torch.Tensor, 
        threshold: float
    ) -> bool:
        """Check if two spike patterns are similar."""
        if pattern1.shape != pattern2.shape:
            return False
            
        # Calculate correlation coefficient
        if pattern1.numel() == 0:
            return pattern2.numel() == 0
            
        flat1 = pattern1.flatten().float()
        flat2 = pattern2.flatten().float()
        
        # Pearson correlation
        mean1, mean2 = flat1.mean(), flat2.mean()
        
        numerator = ((flat1 - mean1) * (flat2 - mean2)).sum()
        denominator = torch.sqrt(((flat1 - mean1) ** 2).sum() * ((flat2 - mean2) ** 2).sum())
        
        if denominator == 0:
            correlation = 1.0 if torch.allclose(flat1, flat2) else 0.0
        else:
            correlation = (numerator / denominator).item()
            
        return abs(correlation) >= threshold


class CacheManager:
    """Manages multiple cache instances with automatic optimization."""
    
    def __init__(self):
        self.caches: Dict[str, BaseCache] = {}
        self._optimization_thread = None
        self._running = False
        self._optimization_interval = 300  # 5 minutes
        
    def create_cache(
        self,
        name: str,
        cache_type: str = "adaptive",
        max_size: int = 1000,
        max_memory_mb: int = 512,
        **kwargs
    ) -> BaseCache:
        """Create and register a new cache."""
        if cache_type == "lru":
            cache = LRUCache(max_size, max_memory_mb)
        elif cache_type == "adaptive":
            cache = AdaptiveCache(max_size, max_memory_mb)
        elif cache_type == "tensor":
            cache = TensorCache(max_size, max_memory_mb)
        elif cache_type == "spike":
            cache = SpikeTrainCache(max_size, max_memory_mb)
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        self.caches[name] = cache
        return cache
        
    def get_cache(self, name: str) -> Optional[BaseCache]:
        """Get cache by name."""
        return self.caches.get(name)
        
    def remove_cache(self, name: str) -> bool:
        """Remove cache by name."""
        if name in self.caches:
            self.caches[name].clear()
            del self.caches[name]
            return True
        return False
        
    def start_optimization(self):
        """Start background cache optimization."""
        if not self._running:
            self._running = True
            self._optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self._optimization_thread.start()
            logger.info("Cache optimization started")
            
    def stop_optimization(self):
        """Stop background cache optimization."""
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5.0)
        logger.info("Cache optimization stopped")
        
    def _optimization_loop(self):
        """Background optimization loop."""
        while self._running:
            try:
                self._optimize_caches()
                time.sleep(self._optimization_interval)
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
                time.sleep(30)  # Back off on error
                
    def _optimize_caches(self):
        """Optimize all caches based on usage patterns."""
        system_memory = psutil.virtual_memory()
        memory_pressure = system_memory.percent > 80
        
        for name, cache in self.caches.items():
            stats = cache.get_stats()
            
            # Adjust cache size based on hit rate and memory pressure
            if stats['hit_rate'] < 0.3 and stats['size'] > 10:
                # Poor hit rate, reduce cache size
                new_max_size = max(int(cache.max_size * 0.8), 10)
                cache.max_size = new_max_size
                logger.info(f"Reduced {name} cache size to {new_max_size} due to poor hit rate")
                
            elif stats['hit_rate'] > 0.8 and not memory_pressure:
                # High hit rate and memory available, increase cache size
                new_max_size = min(int(cache.max_size * 1.2), cache.max_size + 200)
                cache.max_size = new_max_size
                logger.info(f"Increased {name} cache size to {new_max_size} due to high hit rate")
                
            # Clear expired entries
            self._cleanup_expired(cache)
            
    def _cleanup_expired(self, cache: BaseCache):
        """Clean up expired entries from cache."""
        with cache._lock:
            expired_keys = [
                key for key, entry in cache._entries.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                cache.evict(key)
                
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        total_memory = sum(cache.memory_usage for cache in self.caches.values())
        total_entries = sum(cache.size for cache in self.caches.values())
        
        cache_stats = {
            name: cache.get_stats() 
            for name, cache in self.caches.items()
        }
        
        # Calculate global hit rate
        total_hits = sum(stats['hits'] for stats in cache_stats.values())
        total_requests = sum(stats['hits'] + stats['misses'] for stats in cache_stats.values())
        global_hit_rate = total_hits / max(total_requests, 1)
        
        return {
            'total_caches': len(self.caches),
            'total_entries': total_entries,
            'total_memory_mb': total_memory / (1024 * 1024),
            'global_hit_rate': global_hit_rate,
            'individual_caches': cache_stats
        }


# Global cache manager instance
cache_manager = CacheManager()


def cached_computation(
    cache_name: str = "default",
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None
):
    """Decorator for caching expensive computations."""
    def decorator(func: Callable):
        # Ensure cache exists
        if cache_name not in cache_manager.caches:
            cache_manager.create_cache(cache_name, "adaptive")
            
        def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_name)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._make_key(func.__name__, *args, **kwargs)
                
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl)
            
            return result
            
        return wrapper
    return decorator