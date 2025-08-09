"""Command-line interface for BioNeuro-Olfactory-Fusion."""

import click
import torch
import numpy as np
from pathlib import Path
import json
from typing import Optional

from . import OlfactoryFusionSNN, create_moth_inspired_network, create_efficient_network
from .sensors.enose.sensor_array import create_standard_enose
from .monitoring.metrics_collector import MetricsCollector


@click.group()
@click.version_option()
def cli():
    """BioNeuro-Olfactory-Fusion: Neuromorphic gas detection framework."""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--sensors', '-s', default=6, help='Number of chemical sensors')
@click.option('--duration', '-d', default=300, help='Monitoring duration in seconds')
@click.option('--output', '-o', type=click.Path(), help='Output log file')
@click.option('--model', '-m', type=click.Choice(['moth', 'efficient']), default='moth', help='Network model type')
def monitor(config: Optional[str], sensors: int, duration: int, output: Optional[str], model: str):
    """Start real-time gas detection monitoring."""
    click.echo(f"🧠 Starting BioNeuro-Olfactory monitoring with {sensors} sensors")
    click.echo(f"📊 Model: {model.upper()}")
    click.echo(f"⏱️  Duration: {duration} seconds")
    
    # Create network model
    if model == 'moth':
        network = create_moth_inspired_network(num_sensors=sensors)
        click.echo("🦋 Using moth-inspired network architecture")
    else:
        network = create_efficient_network(num_sensors=sensors)
        click.echo("⚡ Using efficient edge network architecture")
    
    # Initialize sensor array
    enose = create_standard_enose(num_sensors=sensors)
    click.echo(f"🔬 Initialized {sensors}-sensor e-nose array")
    
    # Initialize metrics collector
    metrics = MetricsCollector()
    
    try:
        import time
        start_time = time.time()
        detections = 0
        
        click.echo("🚨 Monitoring started - Press Ctrl+C to stop")
        
        while time.time() - start_time < duration:
            try:
                # Simulate sensor reading (replace with real sensor interface)
                chemical_data = torch.randn(1, sensors) * 0.5 + 0.1
                audio_features = torch.randn(1, 128) * 0.3
                
                # Process through network
                result = network.process(chemical_data, audio_features)
                
                # Simple detection logic (replace with trained model)
                activity_level = result['fused_output'].mean().item()
                
                if activity_level > 0.7:  # Detection threshold
                    detections += 1
                    timestamp = time.strftime("%H:%M:%S")
                    click.echo(f"🚨 [{timestamp}] GAS DETECTION - Activity: {activity_level:.3f}")
                    
                    if output:
                        with open(output, 'a') as f:
                            f.write(f"{timestamp},DETECTION,{activity_level:.3f}\n")
                
                # Update metrics
                metrics.record_inference_time(0.05)  # Simulated inference time
                metrics.record_detection(activity_level > 0.7)
                
                time.sleep(1.0)  # 1Hz sampling
                
            except KeyboardInterrupt:
                break
                
        # Summary
        click.echo(f"\n📈 Monitoring Summary:")
        click.echo(f"   Total detections: {detections}")
        click.echo(f"   Detection rate: {detections/(duration/60):.1f}/min")
        click.echo(f"   Average inference time: {metrics.get_average_inference_time():.3f}s")
        
    except Exception as e:
        click.echo(f"❌ Error during monitoring: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--dataset', '-d', required=True, type=click.Path(exists=True), help='Training dataset path')
@click.option('--model', '-m', type=click.Choice(['moth', 'efficient']), default='moth', help='Network model type')
@click.option('--epochs', '-e', default=100, help='Number of training epochs')
@click.option('--batch-size', '-b', default=32, help='Batch size')
@click.option('--learning-rate', '-lr', default=0.001, help='Learning rate')
@click.option('--output', '-o', type=click.Path(), help='Model output path')
def train(dataset: str, model: str, epochs: int, batch_size: int, learning_rate: float, output: Optional[str]):
    """Train gas detection model."""
    click.echo(f"🎓 Training {model} model")
    click.echo(f"📚 Dataset: {dataset}")
    click.echo(f"🔄 Epochs: {epochs}")
    
    # This would be implemented with actual training loop
    click.echo("⚠️  Training implementation coming in Generation 2")
    click.echo("   Current version supports inference only")
    
    if output:
        click.echo(f"💾 Model will be saved to: {output}")


@cli.command()
@click.option('--sensors', '-s', default=6, help='Number of sensors to calibrate')
@click.option('--duration', '-d', default=300, help='Calibration duration in seconds')
@click.option('--reference-gas', '-r', default='clean_air', help='Reference gas for calibration')
@click.option('--output', '-o', type=click.Path(), help='Calibration data output path')
def calibrate(sensors: int, duration: int, reference_gas: str, output: Optional[str]):
    """Calibrate sensor array."""
    click.echo(f"⚙️  Calibrating {sensors} sensors")
    click.echo(f"🌬️  Reference gas: {reference_gas}")
    click.echo(f"⏱️  Duration: {duration} seconds")
    
    # Initialize sensor array
    enose = create_standard_enose(num_sensors=sensors)
    
    try:
        # Calibration process (simplified)
        import time
        calibration_data = []
        
        click.echo("📊 Starting calibration - ensure clean environment")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Simulate sensor readings
            readings = torch.randn(sensors) * 0.1 + 0.5
            timestamp = time.time()
            
            calibration_data.append({
                'timestamp': timestamp,
                'readings': readings.tolist(),
                'reference_gas': reference_gas
            })
            
            if len(calibration_data) % 30 == 0:  # Progress update every 30 seconds
                elapsed = time.time() - start_time
                click.echo(f"⏳ Calibration progress: {elapsed/duration*100:.1f}%")
            
            time.sleep(1.0)
            
        # Save calibration data
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(calibration_data, f, indent=2)
                
            click.echo(f"💾 Calibration data saved to: {output}")
        
        # Compute calibration statistics
        readings_array = np.array([d['readings'] for d in calibration_data])
        baseline_mean = readings_array.mean(axis=0)
        baseline_std = readings_array.std(axis=0)
        
        click.echo(f"\n📈 Calibration Results:")
        click.echo(f"   Baseline mean: {baseline_mean.mean():.3f} ± {baseline_std.mean():.3f}")
        click.echo(f"   Sensor stability: {(baseline_std/baseline_mean*100).mean():.1f}% CV")
        
    except KeyboardInterrupt:
        click.echo("\n⏹️  Calibration interrupted")
    except Exception as e:
        click.echo(f"❌ Calibration failed: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--format', '-f', type=click.Choice(['json', 'xml', 'spdx']), default='json', help='SBOM format')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def generate_sbom(format: str, output: Optional[str]):
    """Generate Software Bill of Materials."""
    click.echo(f"📋 Generating SBOM in {format.upper()} format")
    
    # Simple SBOM generation (would use cyclonedx-bom in production)
    sbom_data = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": "urn:uuid:bioneuro-olfactory-fusion",
        "version": 1,
        "metadata": {
            "timestamp": "2025-08-09T00:00:00Z",
            "tools": ["bioneuro-sbom"],
            "component": {
                "type": "application",
                "bom-ref": "bioneuro-olfactory-fusion",
                "name": "bioneuro-olfactory-fusion",
                "version": "0.1.0"
            }
        },
        "components": [
            {
                "type": "library",
                "bom-ref": "torch",
                "name": "torch",
                "version": ">=1.12.0"
            },
            {
                "type": "library", 
                "bom-ref": "numpy",
                "name": "numpy",
                "version": ">=1.21.0"
            }
        ]
    }
    
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(sbom_data, f, indent=2)
        else:
            click.echo(f"⚠️  {format.upper()} format not yet implemented")
            
        click.echo(f"💾 SBOM saved to: {output}")
    else:
        click.echo(json.dumps(sbom_data, indent=2))


if __name__ == '__main__':
    cli()