"""Database connection management for neuromorphic gas detection system."""

import os
import sqlite3
import logging
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import numpy as np

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pymongo
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    db_type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "bioneuro_olfactory"
    username: str = ""
    password: str = ""
    sqlite_path: str = "data/bioneuro.db"
    connection_pool_size: int = 10
    timeout: int = 30


class DatabaseManager:
    """Unified database manager supporting multiple database backends."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or self._load_config_from_env()
        self.connection = None
        self._connection_pool = []
        self._setup_database()
        
    def _load_config_from_env(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            db_type=os.getenv("DATABASE_TYPE", "sqlite"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "bioneuro_olfactory"),
            username=os.getenv("POSTGRES_USER", ""),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            sqlite_path=os.getenv("SQLITE_PATH", "data/bioneuro.db")
        )
        
    def _setup_database(self):
        """Initialize database and create tables if needed."""
        if self.config.db_type == "sqlite":
            self._setup_sqlite()
        elif self.config.db_type == "postgresql":
            self._setup_postgresql()
        elif self.config.db_type == "mongodb":
            self._setup_mongodb()
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
            
    def _setup_sqlite(self):
        """Setup SQLite database."""
        # Ensure data directory exists
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = sqlite3.connect(
            self.config.sqlite_path,
            timeout=self.config.timeout,
            check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        
        # Enable foreign keys
        self.connection.execute("PRAGMA foreign_keys = ON")
        
        # Create tables
        self._create_sqlite_tables()
        
        logger.info(f"SQLite database initialized at {self.config.sqlite_path}")
        
    def _setup_postgresql(self):
        """Setup PostgreSQL database."""
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")
            
        connection_string = (
            f"host={self.config.host} "
            f"port={self.config.port} "
            f"dbname={self.config.database} "
            f"user={self.config.username} "
            f"password={self.config.password}"
        )
        
        self.connection = psycopg2.connect(
            connection_string,
            cursor_factory=RealDictCursor
        )
        
        # Create tables
        self._create_postgresql_tables()
        
        logger.info(f"PostgreSQL database connected at {self.config.host}:{self.config.port}")
        
    def _setup_mongodb(self):
        """Setup MongoDB database."""
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo not available. Install with: pip install pymongo")
            
        connection_string = f"mongodb://{self.config.host}:{self.config.port}/"
        if self.config.username and self.config.password:
            connection_string = (
                f"mongodb://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}/"
            )
            
        self.client = MongoClient(connection_string)
        self.connection = self.client[self.config.database]
        
        # Create indexes
        self._create_mongodb_indexes()
        
        logger.info(f"MongoDB database connected at {self.config.host}:{self.config.port}")
        
    def _create_sqlite_tables(self):
        """Create SQLite tables."""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                config TEXT,  -- JSON configuration
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'created'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sensor_type TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                raw_value REAL,
                calibrated_value REAL,
                temperature REAL,
                humidity REAL,
                metadata TEXT,  -- JSON metadata
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS network_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                network_type TEXT NOT NULL,
                layer_name TEXT NOT NULL,
                state_data BLOB,  -- Pickled numpy arrays
                sparsity_level REAL,
                firing_rate REAL,
                metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS gas_detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                gas_type TEXT NOT NULL,
                concentration REAL,
                confidence REAL,
                alert_level TEXT,
                response_time REAL,
                sensor_fusion_method TEXT,
                metadata TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                checkpoint_name TEXT NOT NULL,
                model_data BLOB,
                performance_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            """
        ]
        
        for table_sql in tables:
            self.connection.execute(table_sql)
            
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_type ON sensor_data(sensor_type)",
            "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)",
            "CREATE INDEX IF NOT EXISTS idx_gas_events_type ON gas_detection_events(gas_type)",
            "CREATE INDEX IF NOT EXISTS idx_network_states_layer ON network_states(layer_name)"
        ]
        
        for index_sql in indexes:
            self.connection.execute(index_sql)
            
        self.connection.commit()
        
    def _create_postgresql_tables(self):
        """Create PostgreSQL tables."""
        # Similar to SQLite but with PostgreSQL-specific syntax
        # Implementation would include proper data types and constraints
        pass
        
    def _create_mongodb_indexes(self):
        """Create MongoDB indexes."""
        collections = ['experiments', 'sensor_data', 'network_states', 'gas_detection_events']
        
        for collection_name in collections:
            collection = self.connection[collection_name]
            
            if collection_name == 'sensor_data':
                collection.create_index([("timestamp", 1), ("sensor_type", 1)])
            elif collection_name == 'gas_detection_events':
                collection.create_index([("timestamp", 1), ("gas_type", 1)])
            elif collection_name == 'network_states':
                collection.create_index([("timestamp", 1), ("layer_name", 1)])
                
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        if self.config.db_type == "sqlite":
            yield self.connection
        elif self.config.db_type == "postgresql":
            yield self.connection
        elif self.config.db_type == "mongodb":
            yield self.connection
            
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute query and return results."""
        if self.config.db_type == "sqlite":
            cursor = self.connection.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        elif self.config.db_type == "postgresql":
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        else:
            raise NotImplementedError(f"Query execution not implemented for {self.config.db_type}")
            
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute insert and return ID."""
        if self.config.db_type == "sqlite":
            cursor = self.connection.execute(query, params)
            self.connection.commit()
            return cursor.lastrowid
        elif self.config.db_type == "postgresql":
            with self.connection.cursor() as cursor:
                cursor.execute(query + " RETURNING id", params)
                self.connection.commit()
                return cursor.fetchone()['id']
        else:
            raise NotImplementedError(f"Insert execution not implemented for {self.config.db_type}")
            
    def store_numpy_array(self, array: np.ndarray) -> bytes:
        """Serialize numpy array for database storage."""
        return pickle.dumps(array)
        
    def load_numpy_array(self, data: bytes) -> np.ndarray:
        """Deserialize numpy array from database."""
        return pickle.loads(data)
        
    def store_json(self, data: Dict[str, Any]) -> str:
        """Serialize dictionary to JSON string."""
        return json.dumps(data, default=str)
        
    def load_json(self, data: str) -> Dict[str, Any]:
        """Deserialize JSON string to dictionary."""
        if not data:
            return {}
        return json.loads(data)
        
    def create_experiment(self, name: str, description: str = "", config: Dict = None) -> int:
        """Create new experiment record."""
        config_json = self.store_json(config or {})
        
        query = """
        INSERT INTO experiments (name, description, config)
        VALUES (?, ?, ?)
        """
        
        experiment_id = self.execute_insert(query, (name, description, config_json))
        logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment_id
        
    def store_sensor_reading(
        self,
        experiment_id: int,
        sensor_type: str,
        sensor_id: str,
        raw_value: float,
        calibrated_value: float,
        temperature: float = None,
        humidity: float = None,
        metadata: Dict = None
    ) -> int:
        """Store sensor reading."""
        metadata_json = self.store_json(metadata or {})
        
        query = """
        INSERT INTO sensor_data 
        (experiment_id, sensor_type, sensor_id, raw_value, calibrated_value, 
         temperature, humidity, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(query, (
            experiment_id, sensor_type, sensor_id, raw_value, calibrated_value,
            temperature, humidity, metadata_json
        ))
        
    def store_network_state(
        self,
        experiment_id: int,
        network_type: str,
        layer_name: str,
        state_data: np.ndarray,
        sparsity_level: float = None,
        firing_rate: float = None,
        metadata: Dict = None
    ) -> int:
        """Store network state snapshot."""
        state_blob = self.store_numpy_array(state_data)
        metadata_json = self.store_json(metadata or {})
        
        query = """
        INSERT INTO network_states
        (experiment_id, network_type, layer_name, state_data, sparsity_level, 
         firing_rate, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(query, (
            experiment_id, network_type, layer_name, state_blob,
            sparsity_level, firing_rate, metadata_json
        ))
        
    def store_gas_detection_event(
        self,
        experiment_id: int,
        gas_type: str,
        concentration: float,
        confidence: float,
        alert_level: str = "info",
        response_time: float = None,
        sensor_fusion_method: str = None,
        metadata: Dict = None
    ) -> int:
        """Store gas detection event."""
        metadata_json = self.store_json(metadata or {})
        
        query = """
        INSERT INTO gas_detection_events
        (experiment_id, gas_type, concentration, confidence, alert_level,
         response_time, sensor_fusion_method, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.execute_insert(query, (
            experiment_id, gas_type, concentration, confidence, alert_level,
            response_time, sensor_fusion_method, metadata_json
        ))
        
    def get_experiment_data(self, experiment_id: int) -> Dict:
        """Get complete experiment data."""
        # Get experiment info
        exp_query = "SELECT * FROM experiments WHERE id = ?"
        experiment = self.execute_query(exp_query, (experiment_id,))
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        experiment = experiment[0]
        experiment['config'] = self.load_json(experiment['config'])
        
        # Get sensor data
        sensor_query = """
        SELECT * FROM sensor_data 
        WHERE experiment_id = ? 
        ORDER BY timestamp
        """
        sensor_data = self.execute_query(sensor_query, (experiment_id,))
        
        # Get network states
        state_query = """
        SELECT * FROM network_states 
        WHERE experiment_id = ? 
        ORDER BY timestamp
        """
        network_states = self.execute_query(state_query, (experiment_id,))
        
        # Get detection events
        event_query = """
        SELECT * FROM gas_detection_events 
        WHERE experiment_id = ? 
        ORDER BY timestamp
        """
        detection_events = self.execute_query(event_query, (experiment_id,))
        
        return {
            'experiment': experiment,
            'sensor_data': sensor_data,
            'network_states': network_states,
            'detection_events': detection_events
        }
        
    def close(self):
        """Close database connection."""
        if self.connection:
            if self.config.db_type in ["sqlite", "postgresql"]:
                self.connection.close()
            elif self.config.db_type == "mongodb":
                self.client.close()
            logger.info("Database connection closed")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Singleton instance for easy access
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get singleton database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager