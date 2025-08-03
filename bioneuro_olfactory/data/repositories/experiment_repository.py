"""Repository for experiment data management."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .base_repository import BaseRepository, TimestampMixin, CacheableMixin
from ..database.models import ExperimentModel

logger = logging.getLogger(__name__)


class ExperimentRepository(BaseRepository[ExperimentModel], TimestampMixin, CacheableMixin):
    """Repository for managing experiment data."""
    
    def _get_table_name(self) -> str:
        return "experiments"
        
    def create(self, experiment: ExperimentModel) -> int:
        """Create new experiment."""
        config_json = self.db.store_json(experiment.config)
        
        query = """
        INSERT INTO experiments (name, description, config, status)
        VALUES (?, ?, ?, ?)
        """
        
        experiment_id = self.db.execute_insert(
            query, 
            (experiment.name, experiment.description, config_json, experiment.status)
        )
        
        logger.info(f"Created experiment {experiment_id}: {experiment.name}")
        return experiment_id
        
    def get_by_id(self, experiment_id: int) -> Optional[ExperimentModel]:
        """Get experiment by ID."""
        cache_key = f"experiment_{experiment_id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
            
        query = "SELECT * FROM experiments WHERE id = ?"\n        result = self.db.execute_query(query, (experiment_id,))
        
        if not result:
            return None
            
        row = result[0]
        experiment = ExperimentModel(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            config=self.db.load_json(row['config']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
            status=row['status']
        )
        
        self._set_cache(cache_key, experiment)
        return experiment
        
    def update(self, experiment_id: int, updates: Dict[str, Any]) -> bool:
        """Update experiment."""
        if not updates:
            return False
            
        # Handle config updates specially
        if 'config' in updates:
            updates['config'] = self.db.store_json(updates['config'])
            
        # Add updated timestamp
        updates['updated_at'] = datetime.now().isoformat()
        
        # Build UPDATE query
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
            
        params.append(experiment_id)
        
        query = f"""
        UPDATE experiments 
        SET {', '.join(set_clauses)}
        WHERE id = ?
        """
        
        try:
            self.db.execute_query(query, tuple(params))
            self.clear_cache()  # Clear cache after update
            logger.info(f"Updated experiment {experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update experiment {experiment_id}: {e}")
            return False
            
    def delete(self, experiment_id: int) -> bool:
        """Delete experiment and related data."""
        try:
            # Delete related data first (cascade delete)
            related_tables = [
                "model_checkpoints",
                "gas_detection_events", 
                "network_states",
                "sensor_data"
            ]
            
            for table in related_tables:
                delete_query = f"DELETE FROM {table} WHERE experiment_id = ?"
                self.db.execute_query(delete_query, (experiment_id,))
                
            # Delete experiment
            query = "DELETE FROM experiments WHERE id = ?"
            self.db.execute_query(query, (experiment_id,))
            
            self.clear_cache()
            logger.info(f"Deleted experiment {experiment_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
            
    def list_all(self, limit: int = 100, offset: int = 0) -> List[ExperimentModel]:
        """List all experiments."""
        query = """
        SELECT * FROM experiments 
        ORDER BY created_at DESC 
        LIMIT ? OFFSET ?
        """
        
        results = self.db.execute_query(query, (limit, offset))
        experiments = []
        
        for row in results:
            experiment = ExperimentModel(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                config=self.db.load_json(row['config']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                status=row['status']
            )
            experiments.append(experiment)
            
        return experiments
        
    def get_by_status(self, status: str) -> List[ExperimentModel]:
        """Get experiments by status."""
        query = """
        SELECT * FROM experiments 
        WHERE status = ? 
        ORDER BY created_at DESC
        """
        
        results = self.db.execute_query(query, (status,))
        experiments = []
        
        for row in results:
            experiment = ExperimentModel(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                config=self.db.load_json(row['config']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                status=row['status']
            )
            experiments.append(experiment)
            
        return experiments
        
    def search_by_name(self, search_term: str) -> List[ExperimentModel]:
        """Search experiments by name."""
        results = self.search(search_term, ['name', 'description'])
        experiments = []
        
        for row in results:
            experiment = ExperimentModel(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                config=self.db.load_json(row['config']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                status=row['status']
            )
            experiments.append(experiment)
            
        return experiments
        
    def update_status(self, experiment_id: int, status: str) -> bool:
        """Update experiment status."""
        return self.update(experiment_id, {'status': status})
        
    def get_experiment_summary(self, experiment_id: int) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        experiment = self.get_by_id(experiment_id)
        if not experiment:
            return None
            
        # Get sensor data count
        sensor_count_query = """
        SELECT COUNT(*) as count 
        FROM sensor_data 
        WHERE experiment_id = ?
        """
        sensor_count = self.db.execute_query(sensor_count_query, (experiment_id,))
        
        # Get detection events count
        events_count_query = """
        SELECT COUNT(*) as count 
        FROM gas_detection_events 
        WHERE experiment_id = ?
        """
        events_count = self.db.execute_query(events_count_query, (experiment_id,))
        
        # Get network states count
        states_count_query = """
        SELECT COUNT(*) as count 
        FROM network_states 
        WHERE experiment_id = ?
        """
        states_count = self.db.execute_query(states_count_query, (experiment_id,))
        
        # Get experiment duration
        duration_query = """
        SELECT 
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time
        FROM sensor_data 
        WHERE experiment_id = ?
        """
        duration_result = self.db.execute_query(duration_query, (experiment_id,))
        
        duration = None
        if duration_result and duration_result[0]['start_time']:
            start = datetime.fromisoformat(duration_result[0]['start_time'])
            end = datetime.fromisoformat(duration_result[0]['end_time'])
            duration = (end - start).total_seconds()
            
        return {
            'experiment': experiment.to_dict(),
            'sensor_readings_count': sensor_count[0]['count'] if sensor_count else 0,
            'detection_events_count': events_count[0]['count'] if events_count else 0,
            'network_states_count': states_count[0]['count'] if states_count else 0,
            'duration_seconds': duration,
            'data_collection_period': {
                'start': duration_result[0]['start_time'] if duration_result else None,
                'end': duration_result[0]['end_time'] if duration_result else None
            }
        }
        
    def get_recent_experiments(self, days: int = 7, limit: int = 20) -> List[ExperimentModel]:
        """Get recent experiments."""
        query = """
        SELECT * FROM experiments 
        WHERE created_at >= datetime('now', '-{} days')
        ORDER BY created_at DESC 
        LIMIT ?
        """.format(days)
        
        results = self.db.execute_query(query, (limit,))
        experiments = []
        
        for row in results:
            experiment = ExperimentModel(
                id=row['id'],
                name=row['name'],
                description=row['description'],
                config=self.db.load_json(row['config']),
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                status=row['status']
            )
            experiments.append(experiment)
            
        return experiments
        
    def get_experiment_metrics(self, experiment_id: int) -> Dict[str, Any]:
        """Get detailed experiment metrics."""
        # Gas detection accuracy
        accuracy_query = """
        SELECT 
            gas_type,
            AVG(confidence) as avg_confidence,
            COUNT(*) as total_detections,
            AVG(response_time) as avg_response_time
        FROM gas_detection_events 
        WHERE experiment_id = ?
        GROUP BY gas_type
        """
        
        accuracy_results = self.db.execute_query(accuracy_query, (experiment_id,))
        
        # Sensor performance
        sensor_query = """
        SELECT 
            sensor_type,
            COUNT(*) as readings_count,
            AVG(calibrated_value) as avg_value,
            MIN(calibrated_value) as min_value,
            MAX(calibrated_value) as max_value
        FROM sensor_data 
        WHERE experiment_id = ?
        GROUP BY sensor_type
        """
        
        sensor_results = self.db.execute_query(sensor_query, (experiment_id,))
        
        # Network activity
        network_query = """
        SELECT 
            layer_name,
            AVG(sparsity_level) as avg_sparsity,
            AVG(firing_rate) as avg_firing_rate,
            COUNT(*) as state_snapshots
        FROM network_states 
        WHERE experiment_id = ?
        GROUP BY layer_name
        """
        
        network_results = self.db.execute_query(network_query, (experiment_id,))
        
        return {
            'gas_detection_metrics': accuracy_results,
            'sensor_performance': sensor_results,
            'network_activity': network_results
        }
        
    def clone_experiment(self, experiment_id: int, new_name: str) -> int:
        """Clone an existing experiment with new name."""
        original = self.get_by_id(experiment_id)
        if not original:
            raise ValueError(f"Experiment {experiment_id} not found")
            
        cloned = ExperimentModel(
            name=new_name,
            description=f"Cloned from: {original.description}",
            config=original.config.copy(),
            status="created"
        )
        
        return self.create(cloned)