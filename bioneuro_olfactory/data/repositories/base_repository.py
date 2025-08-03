"""Base repository pattern for data access operations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generic, TypeVar
from datetime import datetime
import logging

from ..database.connection import DatabaseManager, get_database_manager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository providing common data access patterns."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or get_database_manager()
        
    @abstractmethod
    def create(self, entity: T) -> int:
        """Create new entity and return ID."""
        pass
        
    @abstractmethod
    def get_by_id(self, entity_id: int) -> Optional[T]:
        """Get entity by ID."""
        pass
        
    @abstractmethod
    def update(self, entity_id: int, updates: Dict[str, Any]) -> bool:
        """Update entity and return success status."""
        pass
        
    @abstractmethod
    def delete(self, entity_id: int) -> bool:
        """Delete entity and return success status."""
        pass
        
    @abstractmethod
    def list_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List all entities with pagination."""
        pass
        
    def exists(self, entity_id: int) -> bool:
        """Check if entity exists."""
        entity = self.get_by_id(entity_id)
        return entity is not None
        
    def count(self) -> int:
        """Count total entities."""
        table_name = self._get_table_name()
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.db.execute_query(query)
        return result[0]['count'] if result else 0
        
    @abstractmethod
    def _get_table_name(self) -> str:
        """Get table name for this repository."""
        pass
        
    def _build_where_clause(self, filters: Dict[str, Any]) -> tuple:
        """Build WHERE clause from filters."""
        if not filters:
            return "", ()
            
        conditions = []
        params = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                placeholders = ",".join(["?" for _ in value])
                conditions.append(f"{key} IN ({placeholders})")
                params.extend(value)
            elif isinstance(value, dict):
                # Handle range queries like {'>=': 10, '<=': 100}
                for op, val in value.items():
                    conditions.append(f"{key} {op} ?")
                    params.append(val)
            else:
                conditions.append(f"{key} = ?")
                params.append(value)
                
        where_clause = " WHERE " + " AND ".join(conditions)
        return where_clause, tuple(params)
        
    def find_by(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[Dict]:
        """Find entities by filters."""
        table_name = self._get_table_name()
        where_clause, params = self._build_where_clause(filters)
        
        query = f"""
        SELECT * FROM {table_name}
        {where_clause}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """
        
        return self.db.execute_query(query, params + (limit, offset))
        
    def find_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime,
        date_column: str = "created_at",
        limit: int = 100
    ) -> List[Dict]:
        """Find entities within date range."""
        table_name = self._get_table_name()
        
        query = f"""
        SELECT * FROM {table_name}
        WHERE {date_column} BETWEEN ? AND ?
        ORDER BY {date_column} DESC
        LIMIT ?
        """
        
        return self.db.execute_query(query, (start_date, end_date, limit))
        
    def get_latest(self, count: int = 10) -> List[Dict]:
        """Get latest entities."""
        table_name = self._get_table_name()
        
        query = f"""
        SELECT * FROM {table_name}
        ORDER BY id DESC
        LIMIT ?
        """
        
        return self.db.execute_query(query, (count,))
        
    def batch_create(self, entities: List[T]) -> List[int]:
        """Create multiple entities in batch."""
        ids = []
        for entity in entities:
            entity_id = self.create(entity)
            ids.append(entity_id)
        return ids
        
    def batch_update(self, updates: List[Dict[str, Any]]) -> int:
        """Update multiple entities in batch."""
        updated_count = 0
        for update in updates:
            entity_id = update.pop('id')
            if self.update(entity_id, update):
                updated_count += 1
        return updated_count
        
    def batch_delete(self, entity_ids: List[int]) -> int:
        """Delete multiple entities in batch."""
        deleted_count = 0
        for entity_id in entity_ids:
            if self.delete(entity_id):
                deleted_count += 1
        return deleted_count
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        table_name = self._get_table_name()
        
        stats_query = f"""
        SELECT 
            COUNT(*) as total_count,
            MIN(created_at) as oldest_record,
            MAX(created_at) as newest_record
        FROM {table_name}
        """
        
        try:
            result = self.db.execute_query(stats_query)
            if result:
                return result[0]
        except Exception as e:
            logger.warning(f"Could not get statistics for {table_name}: {e}")
            
        return {
            'total_count': 0,
            'oldest_record': None,
            'newest_record': None
        }
        
    def search(self, search_term: str, columns: List[str], limit: int = 50) -> List[Dict]:
        """Search entities by text in specified columns."""
        table_name = self._get_table_name()
        
        # Build search conditions
        search_conditions = []
        params = []
        
        for column in columns:
            search_conditions.append(f"{column} LIKE ?")
            params.append(f"%{search_term}%")
            
        where_clause = " WHERE " + " OR ".join(search_conditions)
        
        query = f"""
        SELECT * FROM {table_name}
        {where_clause}
        ORDER BY id DESC
        LIMIT ?
        """
        
        params.append(limit)
        return self.db.execute_query(query, tuple(params))
        
    def execute_custom_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute custom query with parameters."""
        return self.db.execute_query(query, params)
        
    def get_aggregated_data(
        self, 
        group_by: str, 
        aggregations: Dict[str, str],
        filters: Dict[str, Any] = None
    ) -> List[Dict]:
        """Get aggregated data grouped by specified column.
        
        Args:
            group_by: Column to group by
            aggregations: Dict of {alias: 'function(column)'} e.g. {'avg_temp': 'AVG(temperature)'}
            filters: Optional filters to apply
        """
        table_name = self._get_table_name()
        where_clause, params = self._build_where_clause(filters or {})
        
        # Build aggregation clause
        agg_clauses = [f"{func} as {alias}" for alias, func in aggregations.items()]
        agg_clause = ", ".join(agg_clauses)
        
        query = f"""
        SELECT {group_by}, {agg_clause}
        FROM {table_name}
        {where_clause}
        GROUP BY {group_by}
        ORDER BY {group_by}
        """
        
        return self.db.execute_query(query, params)


class TimestampMixin:
    """Mixin for repositories with timestamp columns."""
    
    def get_records_since(self, since: datetime, limit: int = 100) -> List[Dict]:
        """Get records created since specified datetime."""
        return self.find_by_date_range(
            since, 
            datetime.now(),
            limit=limit
        )
        
    def get_records_between(
        self, 
        start: datetime, 
        end: datetime, 
        limit: int = 100
    ) -> List[Dict]:
        """Get records between two datetimes."""
        return self.find_by_date_range(start, end, limit=limit)
        
    def get_hourly_counts(self, days: int = 7) -> List[Dict]:
        """Get hourly record counts for the last N days."""
        table_name = self._get_table_name()
        
        query = f"""
        SELECT 
            strftime('%Y-%m-%d %H:00:00', created_at) as hour,
            COUNT(*) as count
        FROM {table_name}
        WHERE created_at >= datetime('now', '-{days} days')
        GROUP BY strftime('%Y-%m-%d %H:00:00', created_at)
        ORDER BY hour
        """
        
        return self.db.execute_query(query)
        
    def get_daily_counts(self, days: int = 30) -> List[Dict]:
        """Get daily record counts for the last N days."""
        table_name = self._get_table_name()
        
        query = f"""
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as count
        FROM {table_name}
        WHERE created_at >= datetime('now', '-{days} days')
        GROUP BY DATE(created_at)
        ORDER BY date
        """
        
        return self.db.execute_query(query)


class CacheableMixin:
    """Mixin for repositories with caching capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes default
        self._cache_timestamps = {}
        
    def _get_from_cache(self, key: str):
        """Get item from cache if not expired."""
        import time
        
        if key not in self._cache:
            return None
            
        if key in self._cache_timestamps:
            age = time.time() - self._cache_timestamps[key]
            if age > self._cache_ttl:
                del self._cache[key]
                del self._cache_timestamps[key]
                return None
                
        return self._cache[key]
        
    def _set_cache(self, key: str, value):
        """Set item in cache."""
        import time
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()
        
    def clear_cache(self):
        """Clear all cached items."""
        self._cache.clear()
        self._cache_timestamps.clear()
        
    def set_cache_ttl(self, ttl: int):
        """Set cache TTL in seconds."""
        self._cache_ttl = ttl