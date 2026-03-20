# bioneuro-olfactory-fusion

Bio-inspired multi-modal gas detection combining Spiking Neural Network (SNN) temporal dynamics with CNN pattern recognition.

## Architecture

- **ChemicalSensorArray**: 8-sensor array with gas-specific selectivity profiles
- **SNNTemporalEncoder**: LIF neurons encode sensor readings as spike trains
- **CNNCrossSection**: 1D convolution over sensor channel patterns
- **FusionClassifier**: concatenated SNN+CNN features → gas classification

## Usage

```python
from bioneuro.pipeline import BioNeuroOlfactoryPipeline

pipeline = BioNeuroOlfactoryPipeline()
result = pipeline.detect({"ethanol": 0.8, "acetone": 0.2})
print(result["gas"])  # "ethanol"
```

## Detectable Gases

ethanol, acetone, ammonia, benzene, methane, CO2, mixture, clean_air
