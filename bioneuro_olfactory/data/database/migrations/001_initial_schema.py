"""Initial database schema migration for BioNeuro-Olfactory-Fusion system."""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def upgrade_postgresql(connection) -> None:
    """Apply PostgreSQL schema upgrade."""
    
    # Create experiments table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            config JSONB,
            status VARCHAR(50) DEFAULT 'created',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT valid_status CHECK (status IN ('created', 'running', 'completed', 'failed', 'paused'))
        );
    """)
    
    # Create sensor_data table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id BIGSERIAL PRIMARY KEY,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            sensor_type VARCHAR(100) NOT NULL,
            sensor_id VARCHAR(100) NOT NULL,
            raw_value DOUBLE PRECISION,
            calibrated_value DOUBLE PRECISION,
            temperature DOUBLE PRECISION,
            humidity DOUBLE PRECISION,
            pressure DOUBLE PRECISION,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create network_states table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS network_states (
            id BIGSERIAL PRIMARY KEY,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            network_type VARCHAR(100) NOT NULL,
            layer_name VARCHAR(100) NOT NULL,
            sparsity_level DOUBLE PRECISION,
            firing_rate DOUBLE PRECISION,
            energy_consumption DOUBLE PRECISION,
            processing_time_ms DOUBLE PRECISION,
            state_shape INTEGER[],
            state_dtype VARCHAR(50),
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create network_state_data table for large binary data
    connection.execute("""
        CREATE TABLE IF NOT EXISTS network_state_data (
            id BIGSERIAL PRIMARY KEY,
            network_state_id BIGINT NOT NULL REFERENCES network_states(id) ON DELETE CASCADE,
            state_data BYTEA,
            compression_type VARCHAR(50) DEFAULT 'none',
            original_size INTEGER,
            compressed_size INTEGER
        );
    """)
    
    # Create gas_detection_events table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS gas_detection_events (
            id BIGSERIAL PRIMARY KEY,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            gas_type VARCHAR(100) NOT NULL,
            concentration DOUBLE PRECISION NOT NULL,
            concentration_unit VARCHAR(20) DEFAULT 'ppm',
            confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            alert_level VARCHAR(20) NOT NULL DEFAULT 'info',
            response_time_ms DOUBLE PRECISION,
            sensor_fusion_method VARCHAR(100),
            detection_method VARCHAR(100),
            false_positive_probability DOUBLE PRECISION,
            environmental_conditions JSONB,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT valid_alert_level CHECK (alert_level IN ('info', 'warning', 'critical', 'emergency')),
            CONSTRAINT valid_concentration CHECK (concentration >= 0)
        );
    """)
    
    # Create model_checkpoints table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS model_checkpoints (
            id SERIAL PRIMARY KEY,
            experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
            checkpoint_name VARCHAR(255) NOT NULL,
            model_version VARCHAR(50),
            model_architecture VARCHAR(100),
            performance_metrics JSONB,
            training_metadata JSONB,
            file_path VARCHAR(500),
            file_size_bytes BIGINT,
            checksum VARCHAR(128),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(experiment_id, checkpoint_name)
        );
    """)
    
    # Create sensor_calibration table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS sensor_calibration (
            id SERIAL PRIMARY KEY,
            sensor_type VARCHAR(100) NOT NULL,
            sensor_id VARCHAR(100) NOT NULL,
            reference_gas VARCHAR(100) NOT NULL,
            reference_concentrations DOUBLE PRECISION[] NOT NULL,
            sensor_readings DOUBLE PRECISION[] NOT NULL,
            calibration_coefficients DOUBLE PRECISION[] NOT NULL,
            r_squared DOUBLE PRECISION,
            temperature DOUBLE PRECISION,
            humidity DOUBLE PRECISION,
            pressure DOUBLE PRECISION,
            calibration_date TIMESTAMP WITH TIME ZONE NOT NULL,
            expiry_date TIMESTAMP WITH TIME ZONE,
            calibration_method VARCHAR(100),
            notes TEXT,
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sensor_type, sensor_id, calibration_date)
        );
    """)
    
    # Create dataset_metadata table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS dataset_metadata (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            version VARCHAR(50) DEFAULT '1.0.0',
            source VARCHAR(255),
            gases VARCHAR(100)[],
            sensors VARCHAR(100)[],
            sample_count INTEGER DEFAULT 0,
            duration_seconds DOUBLE PRECISION DEFAULT 0,
            sampling_rate_hz DOUBLE PRECISION DEFAULT 1.0,
            features JSONB,
            preprocessing_steps TEXT[],
            quality_metrics JSONB,
            file_path VARCHAR(500),
            file_size_bytes BIGINT,
            checksum VARCHAR(128),
            is_public BOOLEAN DEFAULT false,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create alert_rules table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS alert_rules (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            gas_type VARCHAR(100) NOT NULL,
            warning_threshold DOUBLE PRECISION,
            critical_threshold DOUBLE PRECISION,
            emergency_threshold DOUBLE PRECISION,
            threshold_unit VARCHAR(20) DEFAULT 'ppm',
            confidence_threshold DOUBLE PRECISION DEFAULT 0.8,
            time_window_seconds INTEGER DEFAULT 60,
            notification_methods TEXT[],
            is_active BOOLEAN DEFAULT true,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create system_metrics table
    connection.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metric_name VARCHAR(100) NOT NULL,
            metric_value DOUBLE PRECISION,
            metric_unit VARCHAR(50),
            component VARCHAR(100),
            instance_id VARCHAR(100),
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    connection.commit()
    logger.info("PostgreSQL tables created successfully")


def create_indexes_postgresql(connection) -> None:
    """Create PostgreSQL indexes for performance."""
    
    indexes = [
        # Experiments
        "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)",
        "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at)",
        
        # Sensor data
        "CREATE INDEX IF NOT EXISTS idx_sensor_data_experiment_timestamp ON sensor_data(experiment_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_sensor_data_sensor_type ON sensor_data(sensor_type)",
        "CREATE INDEX IF NOT EXISTS idx_sensor_data_sensor_id ON sensor_data(sensor_id)",
        "CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp)",
        
        # Network states
        "CREATE INDEX IF NOT EXISTS idx_network_states_experiment_timestamp ON network_states(experiment_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_network_states_layer ON network_states(layer_name)",
        "CREATE INDEX IF NOT EXISTS idx_network_states_network_type ON network_states(network_type)",
        
        # Gas detection events
        "CREATE INDEX IF NOT EXISTS idx_gas_events_experiment_timestamp ON gas_detection_events(experiment_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_gas_events_gas_type ON gas_detection_events(gas_type)",
        "CREATE INDEX IF NOT EXISTS idx_gas_events_alert_level ON gas_detection_events(alert_level)",
        "CREATE INDEX IF NOT EXISTS idx_gas_events_confidence ON gas_detection_events(confidence)",
        "CREATE INDEX IF NOT EXISTS idx_gas_events_timestamp ON gas_detection_events(timestamp)",
        
        # Model checkpoints
        "CREATE INDEX IF NOT EXISTS idx_model_checkpoints_experiment ON model_checkpoints(experiment_id)",
        "CREATE INDEX IF NOT EXISTS idx_model_checkpoints_name ON model_checkpoints(checkpoint_name)",
        
        # Sensor calibration
        "CREATE INDEX IF NOT EXISTS idx_sensor_calibration_sensor ON sensor_calibration(sensor_type, sensor_id)",
        "CREATE INDEX IF NOT EXISTS idx_sensor_calibration_active ON sensor_calibration(is_active)",
        "CREATE INDEX IF NOT EXISTS idx_sensor_calibration_expiry ON sensor_calibration(expiry_date)",
        
        # System metrics
        "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name)",
        "CREATE INDEX IF NOT EXISTS idx_system_metrics_component ON system_metrics(component)",
        
        # JSONB indexes for metadata queries
        "CREATE INDEX IF NOT EXISTS idx_experiments_config_gin ON experiments USING GIN (config)",
        "CREATE INDEX IF NOT EXISTS idx_sensor_data_metadata_gin ON sensor_data USING GIN (metadata)",
        "CREATE INDEX IF NOT EXISTS idx_gas_events_metadata_gin ON gas_detection_events USING GIN (metadata)",
        "CREATE INDEX IF NOT EXISTS idx_gas_events_env_conditions_gin ON gas_detection_events USING GIN (environmental_conditions)",
    ]
    
    for index_sql in indexes:
        try:
            connection.execute(index_sql)
        except Exception as e:
            logger.warning(f"Failed to create index: {e}")
    
    connection.commit()
    logger.info("PostgreSQL indexes created successfully")


def create_triggers_postgresql(connection) -> None:
    """Create PostgreSQL triggers for automatic updates."""
    
    # Update timestamp trigger function
    connection.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Triggers for updated_at
    triggers = [
        "CREATE TRIGGER update_experiments_updated_at BEFORE UPDATE ON experiments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()",
        "CREATE TRIGGER update_dataset_metadata_updated_at BEFORE UPDATE ON dataset_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()",
        "CREATE TRIGGER update_alert_rules_updated_at BEFORE UPDATE ON alert_rules FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()",
    ]
    
    for trigger_sql in triggers:
        try:
            connection.execute(trigger_sql)
        except Exception as e:
            logger.warning(f"Failed to create trigger: {e}")
    
    connection.commit()
    logger.info("PostgreSQL triggers created successfully")


def create_views_postgresql(connection) -> None:
    """Create PostgreSQL views for common queries."""
    
    # Latest sensor readings view
    connection.execute("""
        CREATE OR REPLACE VIEW latest_sensor_readings AS
        SELECT DISTINCT ON (experiment_id, sensor_type, sensor_id)
            experiment_id,
            sensor_type,
            sensor_id,
            raw_value,
            calibrated_value,
            temperature,
            humidity,
            timestamp
        FROM sensor_data
        ORDER BY experiment_id, sensor_type, sensor_id, timestamp DESC;
    """)
    
    # Gas detection summary view
    connection.execute("""
        CREATE OR REPLACE VIEW gas_detection_summary AS
        SELECT 
            experiment_id,
            gas_type,
            COUNT(*) as detection_count,
            AVG(concentration) as avg_concentration,
            MAX(concentration) as max_concentration,
            AVG(confidence) as avg_confidence,
            COUNT(CASE WHEN alert_level = 'critical' THEN 1 END) as critical_alerts,
            COUNT(CASE WHEN alert_level = 'warning' THEN 1 END) as warning_alerts,
            MIN(timestamp) as first_detection,
            MAX(timestamp) as last_detection
        FROM gas_detection_events
        GROUP BY experiment_id, gas_type;
    """)
    
    # Active calibrations view
    connection.execute("""
        CREATE OR REPLACE VIEW active_sensor_calibrations AS
        SELECT *
        FROM sensor_calibration
        WHERE is_active = true
        AND (expiry_date IS NULL OR expiry_date > CURRENT_TIMESTAMP);
    """)
    
    connection.commit()
    logger.info("PostgreSQL views created successfully")


def upgrade_sqlite(connection) -> None:
    """Apply SQLite schema upgrade (for development/testing)."""
    
    # SQLite schema (simplified version of PostgreSQL)
    tables = [
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            config TEXT,  -- JSON as TEXT
            status TEXT DEFAULT 'created',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sensor_type TEXT NOT NULL,
            sensor_id TEXT NOT NULL,
            raw_value REAL,
            calibrated_value REAL,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            metadata TEXT,  -- JSON as TEXT
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS gas_detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            gas_type TEXT NOT NULL,
            concentration REAL NOT NULL,
            concentration_unit TEXT DEFAULT 'ppm',
            confidence REAL NOT NULL,
            alert_level TEXT NOT NULL DEFAULT 'info',
            response_time_ms REAL,
            sensor_fusion_method TEXT,
            detection_method TEXT,
            metadata TEXT,  -- JSON as TEXT
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        )
        """
    ]
    
    for table_sql in tables:
        connection.execute(table_sql)
    
    connection.commit()
    logger.info("SQLite tables created successfully")


def downgrade_postgresql(connection) -> None:
    """Rollback PostgreSQL schema changes."""
    
    tables = [
        'system_metrics',
        'alert_rules', 
        'dataset_metadata',
        'sensor_calibration',
        'model_checkpoints',
        'network_state_data',
        'gas_detection_events',
        'network_states',
        'sensor_data',
        'experiments'
    ]
    
    for table in tables:
        connection.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    
    # Drop views
    views = [
        'latest_sensor_readings',
        'gas_detection_summary', 
        'active_sensor_calibrations'
    ]
    
    for view in views:
        connection.execute(f"DROP VIEW IF EXISTS {view}")
    
    # Drop functions
    connection.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    connection.commit()
    logger.info("PostgreSQL schema downgraded successfully")


def downgrade_sqlite(connection) -> None:
    """Rollback SQLite schema changes."""
    
    tables = [
        'gas_detection_events',
        'sensor_data', 
        'experiments'
    ]
    
    for table in tables:
        connection.execute(f"DROP TABLE IF EXISTS {table}")
    
    connection.commit()
    logger.info("SQLite schema downgraded successfully")