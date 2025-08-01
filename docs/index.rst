BioNeuro-Olfactory-Fusion Documentation
=======================================

.. image:: https://img.shields.io/pypi/v/bioneuro-olfactory-fusion.svg
   :target: https://pypi.org/project/bioneuro-olfactory-fusion/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/bioneuro-olfactory-fusion.svg
   :target: https://pypi.org/project/bioneuro-olfactory-fusion/
   :alt: Python versions

.. image:: https://github.com/terragonlabs/bioneuro-olfactory-fusion/workflows/CI/badge.svg
   :target: https://github.com/terragonlabs/bioneuro-olfactory-fusion/actions
   :alt: CI status

.. image:: https://codecov.io/gh/terragonlabs/bioneuro-olfactory-fusion/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/terragonlabs/bioneuro-olfactory-fusion
   :alt: Coverage

Welcome to BioNeuro-Olfactory-Fusion
------------------------------------

BioNeuro-Olfactory-Fusion is a neuromorphic spiking neural network framework that fuses electronic nose (e-nose) sensor arrays with CNN audio features for real-time hazardous gas detection. Inspired by biological olfactory processing, this system mimics moth antenna research for ultra-efficient chemical sensing.

Key Features
^^^^^^^^^^^

* **Spiking Neural Networks**: Energy-efficient temporal processing mimicking biological neurons
* **Multi-Modal Fusion**: Combines e-nose chemical sensors with acoustic signatures  
* **Real-Time Processing**: Sub-millisecond response times for critical safety applications
* **Neuromorphic Hardware Support**: Compatible with Intel Loihi, SpiNNaker, and BrainScaleS
* **Bio-Inspired Architecture**: Models moth olfactory system's projection neurons and Kenyon cells
* **Adaptive Learning**: Online STDP (Spike-Timing-Dependent Plasticity) for environmental adaptation

Quick Start
^^^^^^^^^^^

Install the package:

.. code-block:: bash

   pip install bioneuro-olfactory-fusion

Basic usage:

.. code-block:: python

   from bioneuro_olfactory import OlfactoryFusionSNN
   from bioneuro_olfactory.sensors import ENoseArray

   # Initialize sensor array
   enose = ENoseArray(
       sensor_types=["MQ2", "MQ3", "MQ7"],
       sampling_rate=100
   )

   # Create fusion network
   model = OlfactoryFusionSNN(
       num_chemical_sensors=3,
       num_audio_features=128,
       num_projection_neurons=1000,
       num_kenyon_cells=5000
   )

   # Process sensor data
   chemical_data = enose.read()
   spikes, prediction = model.process(chemical_data)

Documentation Contents
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/models
   api/sensors
   api/neuromorphic
   api/applications

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   architecture
   biological_inspiration
   neuromorphic_deployment
   performance_optimization

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`