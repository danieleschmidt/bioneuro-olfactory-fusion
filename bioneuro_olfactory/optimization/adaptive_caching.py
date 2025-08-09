"""Adaptive caching system with intelligent cache management."""

import time
import threading
import weakref
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
import json
from pathlib import Path
from collections import OrderedDict, defaultdict

from ..core.error_handling import get_error_handler, BioNeuroError, ErrorSeverity
from .performance_profiler import get_profiler


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CacheLevel(Enum):
    """Cache storage levels."""
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"


@dataclass
class CacheEntry:
    """Individual cache entry."""
    key: str
    value: Any
    created_time: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_time > self.ttl
        
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def calculate_priority(self, policy: CachePolicy) -> float:
        """Calculate priority score based on policy."""
        current_time = time.time()
        
        if policy == CachePolicy.LRU:
            return current_time - self.last_accessed  # Higher = older
            
        elif policy == CachePolicy.LFU:
            return -self.access_count  # Higher = less frequent
            
        elif policy == CachePolicy.TTL:
            if self.ttl:
                remaining_ttl = self.ttl - (current_time - self.created_time)
                return -remaining_ttl  # Higher = expires sooner
            return 0
            
        elif policy == CachePolicy.ADAPTIVE:
            # Combine multiple factors
            age_factor = (current_time - self.last_accessed) / 3600  # Hours
            frequency_factor = 1.0 / (self.access_count + 1)
            size_factor = self.size_bytes / (1024 * 1024)  # MB
            
            return age_factor + frequency_factor + size_factor * 0.1
            
        return self.priority


class SmartCache:
    """Intelligent adaptive cache with multiple policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        enable_persistence: bool = False,
        cache_dir: str = "/tmp/bioneuro_cache"
    ):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.policy = policy
        self.enable_persistence = enable_persistence
        self.cache_dir = Path(cache_dir)
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_memory = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background maintenance
        self.maintenance_active = False
        self.maintenance_thread: Optional[threading.Thread] = None
        
        # Error handling and profiling
        self.error_handler = get_error_handler()
        self.profiler = get_profiler()
        
        # Initialize persistence if enabled
        if self.enable_persistence:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
            
        # Start background maintenance
        self.start_maintenance()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self.lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(key)
                    self.misses += 1
                    self._record_access_pattern(key, False)
                    return default
                    
                # Update access statistics
                entry.update_access()
                self.hits += 1
                self._record_access_pattern(key, True)
                
                # Move to end (LRU behavior)
                self._cache.move_to_end(key)
                
                return entry.value
            else:
                self.misses += 1
                self._record_access_pattern(key, False)
                return default
                
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        priority: float = 1.0
    ) -> bool:
        """Set value in cache."""
        with self.lock:
            try:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Check if single item is too large
                if size_bytes > self.max_memory_bytes:
                    self.error_handler.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                    return False
                    
                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_time=time.time(),
                    last_accessed=time.time(),
                    size_bytes=size_bytes,
                    ttl=ttl,
                    priority=priority
                )
                
                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)
                    
                # Ensure we have space
                while (len(self._cache) >= self.max_size or 
                       self.current_memory + size_bytes > self.max_memory_bytes):
                    if not self._evict_one():
                        break  # Can't evict any more
                        
                # Add to cache
                self._cache[key] = entry
                self.current_memory += size_bytes
                
                # Persist if enabled
                if self.enable_persistence:
                    self._persist_entry(key, entry)
                    
                return True
                
            except Exception as e:
                self.error_handler.handle_error(
                    BioNeuroError(
                        f"Cache set error: {str(e)}",
                        error_code="CACHE_SET_ERROR",
                        severity=ErrorSeverity.MEDIUM,
                        context={"key": key}
                    )
                )
                return False
                
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
            
    def clear(self):
        """Clear entire cache."""
        with self.lock:
            self._cache.clear()
            self.current_memory = 0
            
            # Clear persistence
            if self.enable_persistence:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                    
    def _remove_entry(self, key: str):
        """Remove entry and update memory tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.current_memory -= entry.size_bytes
            
            # Remove persistence
            if self.enable_persistence:
                cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self._cache:
            return False
            
        # Find entry to evict
        if self.policy == CachePolicy.ADAPTIVE:
            # Use access patterns to make intelligent decisions
            evict_key = self._adaptive_eviction()
        else:
            # Use standard policy
            evict_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].calculate_priority(self.policy)
            )
            
        if evict_key:
            self._remove_entry(evict_key)
            self.evictions += 1
            return True
            
        return False
        
    def _adaptive_eviction(self) -> Optional[str]:
        """Intelligent eviction based on access patterns."""
        candidates = []
        
        for key, entry in self._cache.items():
            # Calculate adaptive score
            access_pattern = self.access_patterns.get(key, [])
            
            # Recent access frequency
            recent_accesses = len([
                t for t in access_pattern 
                if time.time() - t < 3600  # Last hour
            ])
            
            # Access regularity (lower variance = more regular)
            if len(access_pattern) > 1:
                intervals = [access_pattern[i] - access_pattern[i-1] 
                           for i in range(1, len(access_pattern))]
                regularity = 1.0 / (1.0 + (max(intervals) - min(intervals)) / 3600) if intervals else 0
            else:
                regularity = 0
                
            # Size penalty
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB
            
            # Combine factors (higher score = more likely to evict)
            eviction_score = (
                entry.calculate_priority(CachePolicy.LRU) +
                (10 - recent_accesses) +
                (1 - regularity) * 5 +
                size_penalty * 2
            )
            
            candidates.append((key, eviction_score))
            
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        return None
        
    def _record_access_pattern(self, key: str, hit: bool):
        """Record access pattern for adaptive caching."""
        if hit:
            self.access_patterns[key].append(time.time())
            # Keep only recent history
            cutoff = time.time() - 86400  # 24 hours
            self.access_patterns[key] = [
                t for t in self.access_patterns[key] if t > cutoff
            ]
            
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value."""
        try:
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            # Fallback size estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            else:
                return 64  # Default estimate
                
    def _hash_key(self, key: str) -> str:
        """Generate hash for key."""
        return hashlib.md5(key.encode()).hexdigest()
        
    def _persist_entry(self, key: str, entry: CacheEntry):
        """Persist entry to disk."""
        try:
            cache_file = self.cache_dir / f"{self._hash_key(key)}.cache"
            cache_data = {
                "key": key,
                "value": entry.value,
                "created_time": entry.created_time,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "ttl": entry.ttl,
                "priority": entry.priority,
                "metadata": entry.metadata
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            self.error_handler.logger.warning(f"Failed to persist cache entry {key}: {e}")
            
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        if not self.cache_dir.exists():
            return
            
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Reconstruct entry
                entry = CacheEntry(
                    key=cache_data["key"],
                    value=cache_data["value"],
                    created_time=cache_data["created_time"],
                    last_accessed=cache_data["last_accessed"],
                    access_count=cache_data["access_count"],
                    size_bytes=self._calculate_size(cache_data["value"]),
                    ttl=cache_data.get("ttl"),
                    priority=cache_data.get("priority", 1.0),
                    metadata=cache_data.get("metadata", {})
                )
                
                # Check if expired
                if not entry.is_expired():
                    self._cache[entry.key] = entry
                    self.current_memory += entry.size_bytes
                else:
                    cache_file.unlink()  # Remove expired entry
                    
            except Exception as e:
                self.error_handler.logger.warning(f"Failed to load cache file {cache_file}: {e}")
                
    def start_maintenance(self):
        """Start background maintenance thread."""
        if self.maintenance_active:
            return
            
        self.maintenance_active = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
        
    def stop_maintenance(self):
        """Stop background maintenance thread."""
        self.maintenance_active = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5.0)
            
    def _maintenance_loop(self):
        """Background maintenance loop."""
        while self.maintenance_active:
            try:
                with self.lock:
                    # Remove expired entries
                    expired_keys = [
                        key for key, entry in self._cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                        
                    # Periodic optimization based on access patterns
                    if len(self._cache) > self.max_size * 0.8:  # 80% full
                        self._optimize_cache()
                        
                # Sleep for maintenance interval
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.error_handler.handle_error(e)
                
    def _optimize_cache(self):
        """Optimize cache based on access patterns."""
        # Analyze access patterns
        hot_keys = []
        cold_keys = []
        
        current_time = time.time()
        
        for key, entry in self._cache.items():
            # Recent activity
            recent_activity = entry.access_count / max(1, (current_time - entry.created_time) / 3600)
            
            if recent_activity > 1.0:  # More than 1 access per hour
                hot_keys.append(key)
            elif recent_activity < 0.1:  # Less than 0.1 access per hour
                cold_keys.append(key)
                
        # Increase priority for hot keys
        for key in hot_keys:
            if key in self._cache:
                self._cache[key].priority = min(5.0, self._cache[key].priority * 1.1)
                
        # Consider evicting cold keys if we're near capacity
        if len(self._cache) > self.max_size * 0.9:
            for key in cold_keys[:len(cold_keys)//4]:  # Evict 25% of cold keys
                if key in self._cache:
                    self._remove_entry(key)
                    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "current_memory_mb": self.current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": (self.current_memory / self.max_memory_bytes * 100),
                "policy": self.policy.value,
                "top_accessed": self._get_top_accessed(10)
            }
            
    def _get_top_accessed(self, n: int) -> List[Dict[str, Any]]:
        """Get top N accessed entries."""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )
        
        return [
            {
                "key": key,
                "access_count": entry.access_count,
                "size_mb": entry.size_bytes / (1024 * 1024),
                "age_hours": (time.time() - entry.created_time) / 3600
            }
            for key, entry in sorted_entries[:n]
        ]


class MultiLevelCache:
    """Multi-level cache system (Memory -> Disk -> Distributed)."""
    
    def __init__(
        self,
        l1_memory_mb: float = 50.0,
        l2_disk_mb: float = 500.0,
        l3_distributed: bool = False
    ):
        # Level 1: Memory cache
        self.l1_cache = SmartCache(
            max_size=1000,
            max_memory_mb=l1_memory_mb,
            policy=CachePolicy.LRU,
            enable_persistence=False
        )
        
        # Level 2: Disk cache
        self.l2_cache = SmartCache(
            max_size=10000,
            max_memory_mb=l2_disk_mb,
            policy=CachePolicy.ADAPTIVE,
            enable_persistence=True,
            cache_dir="/tmp/bioneuro_l2_cache"
        )
        
        # Level 3: Distributed cache (placeholder)
        self.l3_enabled = l3_distributed
        self.l3_cache = None  # Would be Redis/Memcached in production
        
        self.error_handler = get_error_handler()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get from multi-level cache."""
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
            
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
            
        # Try L3 if enabled
        if self.l3_enabled and self.l3_cache:
            value = self._get_from_l3(key)
            if value is not None:
                # Promote to L2 and L1
                self.l2_cache.set(key, value)
                self.l1_cache.set(key, value)
                return value
                
        return default
        
    def set(self, key: str, value: Any, **kwargs) -> bool:
        """Set in multi-level cache."""
        # Always set in L1
        l1_success = self.l1_cache.set(key, value, **kwargs)
        
        # Set in L2 for persistence
        l2_success = self.l2_cache.set(key, value, **kwargs)
        
        # Set in L3 if enabled
        l3_success = True
        if self.l3_enabled and self.l3_cache:
            l3_success = self._set_to_l3(key, value, **kwargs)
            
        return l1_success or l2_success or l3_success
        
    def _get_from_l3(self, key: str) -> Any:
        """Get from L3 distributed cache."""
        # Placeholder for distributed cache implementation
        return None
        
    def _set_to_l3(self, key: str, value: Any, **kwargs) -> bool:
        """Set to L3 distributed cache."""
        # Placeholder for distributed cache implementation
        return True
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all cache levels."""
        return {
            "l1_memory": self.l1_cache.get_statistics(),
            "l2_disk": self.l2_cache.get_statistics(),
            "l3_distributed": {"enabled": self.l3_enabled}
        }


# Function memoization decorator with cache
class CachedFunction:
    """Decorator for caching function results."""
    
    def __init__(
        self,
        cache: Optional[SmartCache] = None,
        ttl: Optional[float] = None,
        key_func: Optional[Callable] = None
    ):
        self.cache = cache or SmartCache(max_size=1000, max_memory_mb=50.0)
        self.ttl = ttl
        self.key_func = key_func or self._default_key_func
        
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{self.key_func(args, kwargs)}"
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Execute function
            with get_profiler().profile_context(f"cached_func.{func.__name__}"):
                result = func(*args, **kwargs)
                
            # Cache result
            self.cache.set(cache_key, result, ttl=self.ttl)
            
            return result
            
        wrapper._cache = self.cache  # Allow access to cache for testing
        return wrapper
        
    def _default_key_func(self, args: Tuple, kwargs: Dict) -> str:
        """Default cache key generation."""
        key_data = {
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()


# Global cache instances
_global_cache = None
_multilevel_cache = None


def get_cache() -> SmartCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache(
            max_size=2000,
            max_memory_mb=100.0,
            policy=CachePolicy.ADAPTIVE,
            enable_persistence=True
        )
    return _global_cache


def get_multilevel_cache() -> MultiLevelCache:
    """Get global multi-level cache instance."""
    global _multilevel_cache
    if _multilevel_cache is None:
        _multilevel_cache = MultiLevelCache()
    return _multilevel_cache


# Convenience decorator
def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    return CachedFunction(cache=get_cache(), ttl=ttl, key_func=key_func)