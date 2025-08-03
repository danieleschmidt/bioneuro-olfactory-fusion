#!/bin/bash

# BioNeuro-Olfactory-Fusion Development Environment Setup
echo "🧠 Setting up BioNeuro-Olfactory-Fusion development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    tree \
    htop \
    vim \
    nano \
    jq \
    graphviz \
    libgraphviz-dev \
    pkg-config

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
pip install -e ".[dev]"

# Install additional development tools
pip install \
    jupyterlab \
    tensorboard \
    plotly \
    graphviz \
    networkx \
    matplotlib-backend-kitty

# Setup pre-commit hooks
echo "🔨 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p {logs,data,models,experiments,notebooks,scripts}

# Setup Jupyter Lab extensions
echo "🔬 Setting up Jupyter Lab..."
jupyter lab --generate-config

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || echo "# BioNeuro-Olfactory-Fusion Environment Variables" > .env
fi

# Install neuromorphic simulation dependencies if available
echo "🧠 Installing neuromorphic dependencies..."
pip install brian2 || echo "⚠️  Brian2 not available - will use CPU simulation"

# Setup git configuration helpers
git config --global --add safe.directory /workspaces/bioneuro-olfactory-fusion

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 Quick Start Commands:"
echo "  - Run tests: pytest"
echo "  - Format code: black ."
echo "  - Lint code: ruff check ."
echo "  - Type check: mypy bioneuro_olfactory"
echo "  - Start Jupyter: jupyter lab --allow-root --ip=0.0.0.0"
echo ""
echo "📖 See README.md for detailed usage instructions"