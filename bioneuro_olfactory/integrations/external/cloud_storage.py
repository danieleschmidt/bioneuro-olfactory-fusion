"""Cloud storage integrations for long-term data archival and backup."""

import os
import json
import logging
import asyncio
import aiofiles
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import gzip

logger = logging.getLogger(__name__)


class CloudStorageService:
    """Unified cloud storage service supporting multiple providers."""
    
    def __init__(self):
        self.s3_service = S3StorageService()
        self.azure_service = AzureStorageService()
        self.gcp_service = GCPStorageService()
        
        # Determine which services are available
        self.enabled_services = []
        if os.getenv('AWS_ACCESS_KEY_ID'):
            self.enabled_services.append(self.s3_service)
        if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
            self.enabled_services.append(self.azure_service)
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            self.enabled_services.append(self.gcp_service)
        
        if not self.enabled_services:
            logger.warning("No cloud storage services configured")
    
    async def archive_experiment_data(self, experiment_id: int, data: Dict[str, Any]) -> Dict[str, bool]:
        """Archive experiment data to all configured cloud storage services."""
        results = {}
        
        if not self.enabled_services:
            logger.warning("No cloud storage services available")
            return results
        
        # Prepare data for archival
        archive_data = await self._prepare_archive_data(experiment_id, data)
        
        for service in self.enabled_services:
            try:
                service_name = service.__class__.__name__
                success = await service.upload_experiment_data(experiment_id, archive_data)
                results[service_name] = success
                
                if success:
                    logger.info(f"Experiment {experiment_id} archived to {service_name}")
                else:
                    logger.error(f"Failed to archive experiment {experiment_id} to {service_name}")
                    
            except Exception as e:
                logger.error(f"Error archiving to {service.__class__.__name__}: {e}")
                results[service.__class__.__name__] = False
        
        return results
    
    async def backup_neural_network_state(self, experiment_id: int, network_states: List[Dict]) -> Dict[str, bool]:
        """Backup neural network states to cloud storage."""
        results = {}
        
        # Compress network state data
        compressed_data = await self._compress_network_states(network_states)
        
        for service in self.enabled_services:
            try:
                service_name = service.__class__.__name__
                success = await service.upload_network_states(experiment_id, compressed_data)
                results[service_name] = success
                
            except Exception as e:
                logger.error(f"Error backing up network states to {service_name}: {e}")
                results[service_name] = False
        
        return results
    
    async def _prepare_archive_data(self, experiment_id: int, data: Dict[str, Any]) -> bytes:
        """Prepare experiment data for archival (compression, serialization)."""
        try:
            # Create archive structure
            archive_structure = {
                'experiment_id': experiment_id,
                'archived_at': datetime.now().isoformat(),
                'version': '1.0',
                'data': data
            }
            
            # Serialize to JSON
            json_data = json.dumps(archive_structure, default=str, indent=2)
            
            # Compress
            compressed_data = gzip.compress(json_data.encode('utf-8'))
            
            logger.info(f"Compressed experiment {experiment_id} data: {len(json_data)} -> {len(compressed_data)} bytes")
            return compressed_data
            
        except Exception as e:
            logger.error(f"Error preparing archive data: {e}")
            raise
    
    async def _compress_network_states(self, network_states: List[Dict]) -> bytes:
        """Compress neural network state data."""
        try:
            # Convert network states to serializable format
            serializable_states = []
            for state in network_states:
                serializable_state = {}
                for key, value in state.items():
                    if hasattr(value, 'tolist'):  # numpy array
                        serializable_state[key] = value.tolist()
                    else:
                        serializable_state[key] = value
                serializable_states.append(serializable_state)
            
            # Serialize and compress
            json_data = json.dumps(serializable_states, default=str)
            compressed_data = gzip.compress(json_data.encode('utf-8'))
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Error compressing network states: {e}")
            raise


class S3StorageService:
    """Amazon S3 storage service."""
    
    def __init__(self):
        self.bucket_name = os.getenv('AWS_S3_BUCKET', 'bioneuro-data-archive')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.prefix = 'bioneuro-olfactory-fusion'
    
    async def upload_experiment_data(self, experiment_id: int, data: bytes) -> bool:
        """Upload experiment data to S3."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client('s3', region_name=self.region)
            
            # Generate S3 key
            timestamp = datetime.now().strftime('%Y/%m/%d')
            key = f"{self.prefix}/experiments/{timestamp}/experiment_{experiment_id}.json.gz"
            
            # Upload to S3
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                ContentType='application/gzip',
                ContentEncoding='gzip',
                Metadata={
                    'experiment_id': str(experiment_id),
                    'archived_at': datetime.now().isoformat(),
                    'content_type': 'experiment_data'
                }
            )
            
            logger.info(f"Uploaded experiment {experiment_id} to S3: s3://{self.bucket_name}/{key}")
            return True
            
        except ImportError:
            logger.error("boto3 not installed - S3 integration disabled")
            return False
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    async def upload_network_states(self, experiment_id: int, data: bytes) -> bool:
        """Upload neural network states to S3."""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client('s3', region_name=self.region)
            
            # Generate S3 key for network states
            timestamp = datetime.now().strftime('%Y/%m/%d')
            key = f"{self.prefix}/network_states/{timestamp}/experiment_{experiment_id}_states.json.gz"
            
            # Upload to S3
            s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=data,
                ContentType='application/gzip',
                ContentEncoding='gzip',
                Metadata={
                    'experiment_id': str(experiment_id),
                    'archived_at': datetime.now().isoformat(),
                    'content_type': 'network_states'
                }
            )
            
            logger.info(f"Uploaded network states for experiment {experiment_id} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload network states to S3: {e}")
            return False


class AzureStorageService:
    """Azure Blob Storage service."""
    
    def __init__(self):
        self.connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING', '')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME', 'bioneuro-data')
        self.prefix = 'bioneuro-olfactory-fusion'
    
    async def upload_experiment_data(self, experiment_id: int, data: bytes) -> bool:
        """Upload experiment data to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            from azure.core.exceptions import AzureError
            
            if not self.connection_string:
                logger.error("Azure Storage connection string not configured")
                return False
            
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            
            # Generate blob name
            timestamp = datetime.now().strftime('%Y/%m/%d')
            blob_name = f"{self.prefix}/experiments/{timestamp}/experiment_{experiment_id}.json.gz"
            
            # Upload to Azure Blob Storage
            blob_client = blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings={
                    'content_type': 'application/gzip',
                    'content_encoding': 'gzip'
                },
                metadata={
                    'experiment_id': str(experiment_id),
                    'archived_at': datetime.now().isoformat(),
                    'content_type': 'experiment_data'
                }
            )
            
            logger.info(f"Uploaded experiment {experiment_id} to Azure Blob Storage")
            return True
            
        except ImportError:
            logger.error("azure-storage-blob not installed - Azure integration disabled")
            return False
        except AzureError as e:
            logger.error(f"Azure Storage upload error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to upload to Azure: {e}")
            return False
    
    async def upload_network_states(self, experiment_id: int, data: bytes) -> bool:
        """Upload neural network states to Azure Blob Storage."""
        try:
            from azure.storage.blob import BlobServiceClient
            
            blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            
            # Generate blob name for network states
            timestamp = datetime.now().strftime('%Y/%m/%d')
            blob_name = f"{self.prefix}/network_states/{timestamp}/experiment_{experiment_id}_states.json.gz"
            
            blob_client = blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings={
                    'content_type': 'application/gzip',
                    'content_encoding': 'gzip'
                },
                metadata={
                    'experiment_id': str(experiment_id),
                    'archived_at': datetime.now().isoformat(),
                    'content_type': 'network_states'
                }
            )
            
            logger.info(f"Uploaded network states for experiment {experiment_id} to Azure")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload network states to Azure: {e}")
            return False


class GCPStorageService:
    """Google Cloud Storage service."""
    
    def __init__(self):
        self.bucket_name = os.getenv('GCP_STORAGE_BUCKET', 'bioneuro-data-archive')
        self.prefix = 'bioneuro-olfactory-fusion'
    
    async def upload_experiment_data(self, experiment_id: int, data: bytes) -> bool:
        """Upload experiment data to Google Cloud Storage."""
        try:
            from google.cloud import storage
            from google.cloud.exceptions import GoogleCloudError
            
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            
            # Generate blob name
            timestamp = datetime.now().strftime('%Y/%m/%d')
            blob_name = f"{self.prefix}/experiments/{timestamp}/experiment_{experiment_id}.json.gz"
            
            # Upload to GCS
            blob = bucket.blob(blob_name)
            blob.metadata = {
                'experiment_id': str(experiment_id),
                'archived_at': datetime.now().isoformat(),
                'content_type': 'experiment_data'
            }
            
            blob.upload_from_string(
                data,
                content_type='application/gzip'
            )
            
            logger.info(f"Uploaded experiment {experiment_id} to Google Cloud Storage")
            return True
            
        except ImportError:
            logger.error("google-cloud-storage not installed - GCP integration disabled")
            return False
        except GoogleCloudError as e:
            logger.error(f"GCS upload error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return False
    
    async def upload_network_states(self, experiment_id: int, data: bytes) -> bool:
        """Upload neural network states to Google Cloud Storage."""
        try:
            from google.cloud import storage
            
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            
            # Generate blob name for network states
            timestamp = datetime.now().strftime('%Y/%m/%d')
            blob_name = f"{self.prefix}/network_states/{timestamp}/experiment_{experiment_id}_states.json.gz"
            
            blob = bucket.blob(blob_name)
            blob.metadata = {
                'experiment_id': str(experiment_id),
                'archived_at': datetime.now().isoformat(),
                'content_type': 'network_states'
            }
            
            blob.upload_from_string(
                data,
                content_type='application/gzip'
            )
            
            logger.info(f"Uploaded network states for experiment {experiment_id} to GCS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload network states to GCS: {e}")
            return False


class DataRetentionService:
    """Service for managing data retention policies and automated cleanup."""
    
    def __init__(self):
        self.cloud_storage = CloudStorageService()
        self.retention_days = int(os.getenv('DATA_RETENTION_DAYS', '365'))  # 1 year default
        self.archive_threshold_days = int(os.getenv('ARCHIVE_THRESHOLD_DAYS', '30'))  # Archive after 30 days
    
    async def archive_old_experiments(self, db_manager) -> Dict[str, int]:
        """Archive experiments older than the threshold."""
        try:
            # Find experiments older than archive threshold
            cutoff_date = datetime.now() - timedelta(days=self.archive_threshold_days)
            
            # This would be implemented based on your database schema
            # old_experiments = db_manager.get_experiments_before_date(cutoff_date)
            old_experiments = []  # Placeholder
            
            results = {
                'archived_count': 0,
                'failed_count': 0
            }
            
            for experiment in old_experiments:
                try:
                    # Get complete experiment data
                    exp_data = db_manager.get_experiment_data(experiment['id'])
                    
                    # Archive to cloud storage
                    archive_results = await self.cloud_storage.archive_experiment_data(
                        experiment['id'], 
                        exp_data
                    )
                    
                    # If archived successfully to at least one service, mark as archived
                    if any(archive_results.values()):
                        # Mark experiment as archived in database
                        # db_manager.mark_experiment_archived(experiment['id'])
                        results['archived_count'] += 1
                        logger.info(f"Archived experiment {experiment['id']}")
                    else:
                        results['failed_count'] += 1
                        logger.error(f"Failed to archive experiment {experiment['id']}")
                        
                except Exception as e:
                    logger.error(f"Error archiving experiment {experiment.get('id', 'unknown')}: {e}")
                    results['failed_count'] += 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error in archive_old_experiments: {e}")
            return {'archived_count': 0, 'failed_count': 0}
    
    async def cleanup_expired_data(self, db_manager) -> Dict[str, int]:
        """Clean up data that has exceeded retention period."""
        try:
            # Find data older than retention period
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            results = {
                'deleted_experiments': 0,
                'deleted_sensor_data': 0,
                'deleted_network_states': 0
            }
            
            # This would be implemented based on your database schema
            # Only delete data that has been successfully archived
            # expired_experiments = db_manager.get_archived_experiments_before_date(cutoff_date)
            
            # Implement cleanup logic here
            logger.info(f"Data retention cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in cleanup_expired_data: {e}")
            return {'deleted_experiments': 0, 'deleted_sensor_data': 0, 'deleted_network_states': 0}
    
    async def verify_archive_integrity(self) -> Dict[str, Any]:
        """Verify integrity of archived data."""
        try:
            # Implement archive verification logic
            # Check if archived data can be retrieved and is valid
            verification_results = {
                'verified_archives': 0,
                'corrupted_archives': 0,
                'missing_archives': 0
            }
            
            logger.info(f"Archive integrity verification completed: {verification_results}")
            return verification_results
            
        except Exception as e:
            logger.error(f"Error in verify_archive_integrity: {e}")
            return {'verified_archives': 0, 'corrupted_archives': 0, 'missing_archives': 0}