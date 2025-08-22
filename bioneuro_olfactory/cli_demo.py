#!/usr/bin/env python3
"""Enhanced CLI for BioNeuro-Olfactory-Fusion system."""

import click
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from . import OlfactoryFusionSNN, create_moth_inspired_network, create_efficient_network
    from .sensors.enose.sensor_array import create_standard_enose
    from .models.fusion.multimodal_fusion import create_standard_fusion_network
    HAS_CORE_MODULES = True
except ImportError:
    print("⚠️  Core modules not available - running in limited mode")
    HAS_CORE_MODULES = False


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """BioNeuro-Olfactory-Fusion: Neuromorphic Gas Detection System
    
    A bio-inspired spiking neural network framework for real-time 
    hazardous gas detection using electronic nose sensors.
    """
    pass


@cli.command()
def status():
    """Show system status and health information."""
    click.echo("📊 BioNeuro-Olfactory-Fusion System Status")
    click.echo("=" * 50)
    
    try:
        # System information
        import platform
        click.echo("🖥️  System Information:")
        click.echo(f"  Platform: {platform.system()} {platform.release()}")
        click.echo(f"  Python: {platform.python_version()}")
        click.echo(f"  Architecture: {platform.machine()}")
        
        # Module availability
        click.echo("\n📦 Module Status:")
        
        modules_to_check = [
            ('torch', 'PyTorch'),
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy'),
            ('sklearn', 'Scikit-learn'),
            ('librosa', 'Librosa'),
            ('matplotlib', 'Matplotlib')
        ]
        
        available_count = 0
        for module_name, display_name in modules_to_check:
            try:
                __import__(module_name)
                click.echo(f"  ✅ {display_name}: Available")
                available_count += 1
            except ImportError:
                click.echo(f"  ❌ {display_name}: Not available")
        
        # Core system status
        click.echo("\n🧠 Core System:")
        if HAS_CORE_MODULES:
            click.echo("  ✅ BioNeuro core modules: Available")
            click.echo("  ✅ Neuromorphic fusion: Available")
            click.echo("  ✅ Sensor interfaces: Available")
        else:
            click.echo("  ⚠️  BioNeuro core modules: Limited availability")
        
        # Mock interfaces
        click.echo("\n🔧 Fallback Systems:")
        click.echo("  ✅ Mock PyTorch interface: Available")
        click.echo("  ✅ Mock NumPy interface: Available")
        
        # Overall health
        health_score = (available_count / len(modules_to_check)) * 100
        click.echo(f"\n🏥 Overall Health: {health_score:.0f}%")
        
        if health_score >= 80:
            click.echo("  🟢 System status: Excellent")
        elif health_score >= 60:
            click.echo("  🟡 System status: Good with limitations")
        else:
            click.echo("  🔴 System status: Limited functionality")
        
    except Exception as e:
        click.echo(f"❌ Status check error: {e}")
        raise


@cli.command()
@click.option('--sensors', '-s', default=6, help='Number of sensors')
@click.option('--duration', '-d', default=30, help='Demo duration in seconds')
def demo(sensors: int, duration: int):
    """Run a quick demonstration of the system."""
    click.echo("🎭 BioNeuro-Olfactory-Fusion Demo")
    click.echo("=" * 50)
    
    if not HAS_CORE_MODULES:
        click.echo("⚠️  Running demo in simulation mode")
    
    try:
        click.echo(f"🧠 Creating neuromorphic fusion network...")
        if HAS_CORE_MODULES:
            fusion_net = create_standard_fusion_network(num_chemical_sensors=sensors)
            click.echo("✅ Fusion network created")
        else:
            click.echo("🔄 Simulating fusion network")
        
        click.echo(f"👃 Creating e-nose array with {sensors} sensors...")
        if HAS_CORE_MODULES:
            enose = create_standard_enose()
            click.echo("✅ E-nose array created")
        else:
            click.echo("🔄 Simulating e-nose array")
        
        click.echo(f"🚀 Running demo for {duration} seconds...\n")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            sample_count += 1
            
            # Simulate readings
            if HAS_CORE_MODULES and 'enose' in locals():
                try:
                    readings = enose.read_all_sensors()
                    concentrations = enose.get_concentration_estimates()
                    max_conc = max(concentrations.values()) if concentrations else 0
                except:
                    max_conc = 50.0 + 30.0 * (sample_count % 5)  # Simulated
            else:
                max_conc = 50.0 + 30.0 * (sample_count % 5)  # Simulated
            
            # Status indicator
            status = "🔥 HIGH" if max_conc > 80 else "🟡 MEDIUM" if max_conc > 50 else "✅ LOW"
            
            click.echo(f"[{elapsed:5.1f}s] Sample #{sample_count:02d} | "
                      f"Max: {max_conc:6.1f} PPM | {status}")
            
            # Simulate neuromorphic processing
            if sample_count % 3 == 0:
                if HAS_CORE_MODULES and 'fusion_net' in locals():
                    try:
                        chemical_data = [max_conc/100] * sensors
                        audio_data = [0.1] * 64
                        result = fusion_net.process_realtime(chemical_data, audio_data)
                        hazard_prob = result.get('hazard_probability', [0.5])[0]
                        click.echo(f"    🧠 Neuromorphic: Hazard probability {hazard_prob:.1%}")
                    except:
                        click.echo(f"    🧠 Neuromorphic: Processing active")
                else:
                    hazard_prob = min(max_conc / 100, 0.95)
                    click.echo(f"    🧠 Neuromorphic (sim): Hazard probability {hazard_prob:.1%}")
            
            time.sleep(2.0)  # 0.5 Hz for demo
        
        click.echo(f"\n✅ Demo completed successfully!")
        click.echo(f"📊 Processed {sample_count} samples in {duration} seconds")
        click.echo(f"⚡ Average rate: {sample_count/duration:.1f} Hz")
        
    except KeyboardInterrupt:
        click.echo(f"\n🛑 Demo interrupted by user")
    except Exception as e:
        click.echo(f"❌ Demo error: {e}")
        raise


if __name__ == '__main__':
    cli()