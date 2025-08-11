"""
Advanced multi-level caching system with distributed support.

This module implements a sophisticated caching framework optimized for
neuromorphic gas detection workloads with support for local, distributed,
and intelligent cache management.
"""

import asyncio
import hashlib
import pickle
import time
import threading
import weakref
import zlib
import json
import redis
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import asynccontextmanager, contextmanager
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"      # In-process memory cache
    L2_SHARED = "l2_shared"      # Shared memory cache
    L3_DISK = "l3_disk"          # Local disk cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache (Redis)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive"  # ML-based adaptive eviction
    FIFO = "fifo"        # First In First Out


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    compression_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
        
    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
        
    def touch(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStats:
    """Cache performance statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        self.total_size = 0
        self.avg_access_time = 0.0
        self.access_times = []
        
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def record_hit(self, access_time: float = 0.0):
        """Record cache hit."""
        self.hits += 1
        if access_time > 0:
            self.access_times.append(access_time)
            
    def record_miss(self):
        """Record cache miss."""
        self.misses += 1
        
    def record_eviction(self):
        """Record cache eviction."""
        self.evictions += 1
        
    def record_error(self):
        """Record cache error."""
        self.errors += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        if self.access_times:
            self.avg_access_time = np.mean(self.access_times[-100:])  # Last 100 accesses
            
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'evictions': self.evictions,
            'errors': self.errors,
            'total_size': self.total_size,
            'avg_access_time_ms': self.avg_access_time * 1000,
            'total_requests': self.hits + self.misses
        }


class CacheBackend(ABC):
    """Abstract cache backend interface."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache."""
        pass
        
    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in cache."""
        pass
        
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
        
    @abstractmethod
    async def clear(self) -> bool:
        """Clear entire cache."""
        pass
        
    @abstractmethod
    async def size(self) -> int:
        """Get cache size."""
        pass
        
    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all keys."""
        pass


class MemoryCache(CacheBackend):
    """High-performance in-memory cache backend."""
    
    def __init__(self, max_size: int = 1000, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._frequency_counter: Dict[str, int] = defaultdict(int)
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from memory cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                if entry.is_expired:
                    del self._cache[key]
                    return None
                    
                # Update access statistics
                entry.touch()
                self._frequency_counter[key] += 1
                
                # Move to end for LRU
                if self.eviction_policy == EvictionPolicy.LRU:
                    self._cache.move_to_end(key)
                    
                return entry
                
        return None
        
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Check if eviction is needed
            if len(self._cache) >= self.max_size and key not in self._cache:
                await self._evict()
                
            self._cache[key] = entry
            
            # Move to end for LRU
            if self.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
                
            return True
            
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._frequency_counter:
                    del self._frequency_counter[key]
                return True
        return False
        
    async def clear(self) -> bool:
        """Clear memory cache."""
        with self._lock:
            self._cache.clear()
            self._frequency_counter.clear()
        return True
        
    async def size(self) -> int:
        """Get cache size."""
        return len(self._cache)
        
    async def keys(self) -> List[str]:
        """Get all keys."""
        with self._lock:
            return list(self._cache.keys())
            
    async def _evict(self):
        """Evict entries based on policy."""
        if not self._cache:
            return
            
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first item)
            self._cache.popitem(last=False)
            
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_freq_key = min(self._frequency_counter, key=self._frequency_counter.get)
            del self._cache[min_freq_key]
            del self._frequency_counter[min_freq_key]
            
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.ttl and (current_time - entry.created_at) > entry.ttl
            ]
            
            if expired_keys:
                for key in expired_keys:
                    del self._cache[key]
                    if key in self._frequency_counter:
                        del self._frequency_counter[key]
            else:
                # Fall back to LRU
                self._cache.popitem(last=False)
                
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove first inserted
            self._cache.popitem(last=False)


class DiskCache(CacheBackend):
    """Persistent disk-based cache backend."""
    
    def __init__(self, cache_dir: str = "/tmp/bioneuro_cache", max_size_mb: int = 1000):
        import os
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from disk cache."""
        file_path = self._get_file_path(key)
        
        try:
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
                
            # Decompress and deserialize
            data = zlib.decompress(compressed_data)
            entry = pickle.loads(data)
            
            if entry.is_expired:
                await self.delete(key)
                return None
                
            entry.touch()
            return entry
            
        except (FileNotFoundError, pickle.PickleError, zlib.error):
            return None
            
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in disk cache."""
        file_path = self._get_file_path(key)
        
        try:
            # Serialize and compress
            data = pickle.dumps(entry)
            compressed_data = zlib.compress(data)
            entry.size_bytes = len(compressed_data)
            
            with self._lock:
                # Check disk space
                await self._cleanup_if_needed()
                
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
                    
            return True
            
        except (pickle.PickleError, IOError) as e:
            logger.error(f"Failed to write to disk cache: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        file_path = self._get_file_path(key)
        
        try:
            import os
            os.remove(file_path)
            return True
        except FileNotFoundError:
            return False
            
    async def clear(self) -> bool:
        """Clear disk cache."""
        import os
        import shutil
        
        try:
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            return True
        except OSError:
            return False
            
    async def size(self) -> int:
        """Get cache size."""
        import os
        
        if not os.path.exists(self.cache_dir):
            return 0
            
        return len([f for f in os.listdir(self.cache_dir) if f.endswith('.cache')])
        
    async def keys(self) -> List[str]:
        """Get all keys."""
        import os
        
        if not os.path.exists(self.cache_dir):
            return []
            
        files = [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
        return [f[:-6] for f in files]  # Remove .cache extension
        
    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return f"{self.cache_dir}/{key_hash}.cache"
        
    async def _cleanup_if_needed(self):
        """Cleanup cache if size limit exceeded."""
        import os
        
        # Calculate current size
        total_size = 0
        files = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                filepath = os.path.join(self.cache_dir, filename)
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                files.append((filepath, size, mtime))
                total_size += size
                
        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size_mb > self.max_size_mb:
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[2])
            
            # Remove oldest files until under limit
            for filepath, size, _ in files:
                os.remove(filepath)
                total_size_mb -= size / (1024 * 1024)
                if total_size_mb <= self.max_size_mb * 0.8:  # Leave some headroom
                    break


class RedisCache(CacheBackend):
    """Distributed Redis-based cache backend."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "bioneuro"):
        self.redis_url = redis_url
        self.prefix = prefix
        self._redis = None
        
    async def _get_redis(self):
        """Get Redis connection."""
        if self._redis is None:
            import aioredis
            self._redis = aioredis.from_url(self.redis_url)
        return self._redis
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            prefixed_key = f"{self.prefix}:{key}"
            
            data = await redis_client.get(prefixed_key)
            if data is None:
                return None
                
            # Deserialize
            entry = pickle.loads(data)
            
            if entry.is_expired:
                await self.delete(key)
                return None
                
            entry.touch()
            
            # Update TTL in Redis if specified
            if entry.ttl:
                await redis_client.expire(prefixed_key, int(entry.ttl))
                
            return entry
            
        except (redis.RedisError, pickle.PickleError) as e:
            logger.error(f"Redis cache get error: {e}")
            return None
            
    async def set(self, key: str, entry: CacheEntry) -> bool:
        """Set value in Redis cache."""
        try:
            redis_client = await self._get_redis()
            prefixed_key = f"{self.prefix}:{key}"
            
            # Serialize
            data = pickle.dumps(entry)
            
            # Set with TTL if specified
            if entry.ttl:
                await redis_client.setex(prefixed_key, int(entry.ttl), data)
            else:
                await redis_client.set(prefixed_key, data)
                
            return True
            
        except (redis.RedisError, pickle.PickleError) as e:
            logger.error(f"Redis cache set error: {e}")
            return False
            
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            redis_client = await self._get_redis()
            prefixed_key = f"{self.prefix}:{key}"
            result = await redis_client.delete(prefixed_key)
            return result > 0
            
        except redis.RedisError as e:
            logger.error(f"Redis cache delete error: {e}")
            return False
            
    async def clear(self) -> bool:
        """Clear Redis cache."""
        try:
            redis_client = await self._get_redis()
            pattern = f"{self.prefix}:*"
            
            # Use scan for better performance
            keys = []
            async for key in redis_client.scan_iter(match=pattern):
                keys.append(key)
                
            if keys:
                await redis_client.delete(*keys)
                
            return True
            
        except redis.RedisError as e:
            logger.error(f"Redis cache clear error: {e}")
            return False
            
    async def size(self) -> int:
        """Get Redis cache size."""
        try:
            redis_client = await self._get_redis()
            pattern = f"{self.prefix}:*"
            
            count = 0
            async for _ in redis_client.scan_iter(match=pattern):
                count += 1
                
            return count
            
        except redis.RedisError as e:
            logger.error(f"Redis cache size error: {e}")
            return 0
            
    async def keys(self) -> List[str]:
        """Get all keys from Redis cache."""
        try:
            redis_client = await self._get_redis()
            pattern = f"{self.prefix}:*"
            
            keys = []
            async for key in redis_client.scan_iter(match=pattern):
                # Remove prefix
                clean_key = key.decode() if isinstance(key, bytes) else key
                clean_key = clean_key[len(self.prefix) + 1:]
                keys.append(clean_key)
                
            return keys
            
        except redis.RedisError as e:
            logger.error(f"Redis cache keys error: {e}")
            return []


class MultiLevelCache:
    """Multi-level hierarchical cache with intelligent promotion/demotion."""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Initialize cache levels
        self.levels: Dict[CacheLevel, CacheBackend] = {}
        
        # L1: Memory cache (fastest)
        if config.get('enable_l1', True):
            self.levels[CacheLevel.L1_MEMORY] = MemoryCache(
                max_size=config.get('l1_size', 1000),
                eviction_policy=EvictionPolicy(config.get('l1_eviction', 'lru'))
            )
            
        # L2: Shared memory (medium speed)
        # Note: In production, this would use shared memory between processes
        if config.get('enable_l2', True):
            self.levels[CacheLevel.L2_SHARED] = MemoryCache(
                max_size=config.get('l2_size', 5000),
                eviction_policy=EvictionPolicy(config.get('l2_eviction', 'lru'))
            )
            
        # L3: Disk cache (slower, persistent)
        if config.get('enable_l3', True):
            self.levels[CacheLevel.L3_DISK] = DiskCache(
                cache_dir=config.get('l3_dir', '/tmp/bioneuro_cache'),
                max_size_mb=config.get('l3_size_mb', 1000)
            )
            
        # L4: Distributed cache (slowest, shared across nodes)
        if config.get('enable_l4', False):
            self.levels[CacheLevel.L4_DISTRIBUTED] = RedisCache(
                redis_url=config.get('redis_url', 'redis://localhost:6379'),
                prefix=config.get('redis_prefix', 'bioneuro')
            )
            
        # Statistics
        self.stats: Dict[CacheLevel, CacheStats] = {
            level: CacheStats() for level in self.levels.keys()
        }
        
        # Configuration
        self.promotion_threshold = config.get('promotion_threshold', 3)
        self.enable_promotion = config.get('enable_promotion', True)
        self.enable_prefetching = config.get('enable_prefetching', True)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._prefetch_task: Optional[asyncio.Task] = None
        
    async def get(self, key: str, promote: bool = True) -> Optional[Any]:
        """Get value from multi-level cache with promotion."""
        start_time = time.perf_counter()
        
        # Try each level from fastest to slowest
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_SHARED, 
                     CacheLevel.L3_DISK, CacheLevel.L4_DISTRIBUTED]:
            
            if level not in self.levels:
                continue
                
            backend = self.levels[level]
            entry = await backend.get(key)
            
            if entry is not None:
                access_time = time.perf_counter() - start_time
                self.stats[level].record_hit(access_time)
                
                # Promote to higher levels if enabled
                if promote and self.enable_promotion and level != CacheLevel.L1_MEMORY:
                    await self._promote_entry(key, entry, level)
                    
                return entry.value
                
            # Record miss for this level
            self.stats[level].record_miss()
            
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 level: Optional[CacheLevel] = None) -> bool:
        """Set value in multi-level cache."""
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl,
            size_bytes=len(pickle.dumps(value)) if value is not None else 0
        )
        
        # If specific level requested, use only that level
        if level and level in self.levels:
            return await self.levels[level].set(key, entry)
            
        # Otherwise, set in all available levels
        results = []
        for cache_level, backend in self.levels.items():
            try:
                result = await backend.set(key, entry)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to set in {cache_level}: {e}")
                results.append(False)
                
        return any(results)
        
    async def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        results = []
        for backend in self.levels.values():
            try:
                result = await backend.delete(key)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to delete from cache: {e}")
                results.append(False)
                
        return any(results)
        
    async def clear(self, level: Optional[CacheLevel] = None) -> bool:
        """Clear cache(s)."""
        if level and level in self.levels:
            return await self.levels[level].clear()
            
        results = []
        for backend in self.levels.values():
            try:
                result = await backend.clear()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                results.append(False)
                
        return all(results)
        
    async def _promote_entry(self, key: str, entry: CacheEntry, current_level: CacheLevel):
        """Promote entry to higher cache levels."""
        # Promote to levels higher than current
        target_levels = []
        
        if current_level == CacheLevel.L4_DISTRIBUTED:
            target_levels = [CacheLevel.L3_DISK, CacheLevel.L2_SHARED, CacheLevel.L1_MEMORY]
        elif current_level == CacheLevel.L3_DISK:
            target_levels = [CacheLevel.L2_SHARED, CacheLevel.L1_MEMORY]
        elif current_level == CacheLevel.L2_SHARED:
            target_levels = [CacheLevel.L1_MEMORY]
            
        for target_level in target_levels:
            if target_level in self.levels:
                try:
                    await self.levels[target_level].set(key, entry)
                except Exception as e:
                    logger.error(f"Failed to promote to {target_level}: {e}")
                    
    async def start_background_tasks(self):
        """Start background maintenance tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.enable_prefetching:
            self._prefetch_task = asyncio.create_task(self._prefetch_loop())
            
    async def stop_background_tasks(self):
        """Stop background maintenance tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        if self._prefetch_task:
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass
                
    async def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                
    async def _cleanup_expired(self):
        """Remove expired entries from all levels."""
        for level, backend in self.levels.items():
            try:
                if isinstance(backend, (MemoryCache, DiskCache)):
                    # For local caches, we can access entries directly
                    keys_to_delete = []
                    
                    for key in await backend.keys():
                        entry = await backend.get(key)
                        if entry and entry.is_expired:
                            keys_to_delete.append(key)
                            
                    for key in keys_to_delete:
                        await backend.delete(key)
                        self.stats[level].record_eviction()
                        
                    if keys_to_delete:
                        logger.debug(f"Cleaned {len(keys_to_delete)} expired entries from {level}")
                        
            except Exception as e:
                logger.error(f"Failed to cleanup {level}: {e}")
                
    async def _prefetch_loop(self):
        """Background prefetching of popular content."""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                await self._prefetch_popular()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Prefetch loop error: {e}")
                
    async def _prefetch_popular(self):
        """Prefetch popular content to faster cache levels."""
        # This is a simplified version - in practice, you'd use ML models
        # to predict what should be prefetched
        pass
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        level_stats = {}
        
        for level, stats in self.stats.items():
            level_stats[level.value] = stats.get_summary()
            
        # Overall statistics
        total_hits = sum(stats.hits for stats in self.stats.values())
        total_misses = sum(stats.misses for stats in self.stats.values())
        total_requests = total_hits + total_misses
        
        overall_stats = {
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_requests': total_requests,
            'overall_hit_rate': total_hits / total_requests if total_requests > 0 else 0,
            'levels': level_stats
        }
        
        return overall_stats


# Decorator for automatic caching
def cached(
    cache: MultiLevelCache,
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    level: Optional[CacheLevel] = None
):
    """Decorator for automatic function result caching."""
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Cache result
            await cache.set(cache_key, result, ttl=ttl, level=level)
            
            return result
            
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, use asyncio to run cache operations
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in event loop, create new task
                return asyncio.create_task(async_wrapper(*args, **kwargs))
            else:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Global cache instance
_global_cache: Optional[MultiLevelCache] = None


async def get_cache() -> MultiLevelCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MultiLevelCache()
        await _global_cache.start_background_tasks()
    return _global_cache


@contextmanager
def cache_context(cache_config: Dict[str, Any] = None):
    """Context manager for cache lifecycle."""
    cache = MultiLevelCache(cache_config)
    
    try:
        # Start background tasks
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cache.start_background_tasks())
        
        yield cache
        
    finally:
        # Clean up
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cache.stop_background_tasks())