# Makefile for BioNeuro-Olfactory-Fusion

.PHONY: help install install-dev test test-cov lint format type-check security clean build docs docker run-agent

# Default target
help: ## Show this help message
	@echo "BioNeuro-Olfactory-Fusion Development Commands"
	@echo "=============================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation
install: ## Install package for production
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev,test,sensors]"
	pre-commit install

install-neuromorphic: ## Install with neuromorphic hardware support
	pip install -e ".[neuromorphic,sensors]"

# Testing
test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=bioneuro_olfactory --cov-report=html --cov-report=term-missing

test-fast: ## Run tests excluding slow ones
	pytest tests/ -v -m "not slow"

test-integration: ## Run integration tests only
	pytest tests/integration/ -v

test-e2e: ## Run end-to-end tests
	pytest tests/e2e/ -v -m "not neuromorphic"

test-neuromorphic: ## Run neuromorphic hardware tests (requires hardware)
	pytest tests/ -v -m "neuromorphic"

# Code Quality
lint: ## Run linting checks
	ruff check bioneuro_olfactory/ tests/
	black --check bioneuro_olfactory/ tests/

format: ## Format code
	black bioneuro_olfactory/ tests/
	ruff check --fix bioneuro_olfactory/ tests/

type-check: ## Run type checking
	mypy bioneuro_olfactory/

# Security
security: ## Run security checks
	bandit -r bioneuro_olfactory/
	safety check

security-full: ## Run comprehensive security analysis
	bandit -r bioneuro_olfactory/ -f json -o security-report.json
	safety check --json --output safety-report.json
	semgrep --config=auto bioneuro_olfactory/ || true

# Quality gates (CI/CD)
quality-gate: lint type-check security test ## Run all quality checks

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# Build and Release
clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf htmlcov/ .coverage .pytest_cache/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

build-wheel: ## Build wheel only
	python -m build --wheel

upload-test: build ## Upload to test PyPI
	twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	twine upload dist/*

# Documentation
docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && make clean

# Docker
docker-build: ## Build Docker image
	docker build -t bioneuro-olfactory-fusion:latest .

docker-run: ## Run Docker container
	docker run -it --rm bioneuro-olfactory-fusion:latest

docker-compose-up: ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

# Development
dev-setup: install-dev ## Set up development environment
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

dev-clean: clean ## Clean development environment
	pip uninstall -y bioneuro-olfactory-fusion
	rm -rf .venv/

# Autonomous SDLC & Value Discovery
run-agent: ## Run autonomous SDLC agent
	python .terragon/autonomous_agent.py

discover-value: ## Run autonomous value discovery
	python .terragon/value-discovery.py

update-backlog: discover-value ## Update value backlog with latest opportunities
	@echo "Backlog updated with latest value opportunities"

agent-discover: ## Run value discovery only (legacy)
	python -c "from .terragon.autonomous_agent import AutonomousSDLCAgent; from pathlib import Path; agent = AutonomousSDLCAgent(Path('.')); opps = agent.discover_value_opportunities(); print(f'Found {len(opps)} opportunities')"

# Performance
profile: ## Run performance profiling
	python -m cProfile -o profile.stats examples/profile_model.py
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

benchmark: ## Run benchmarks
	python -m pytest benchmarks/ -v --benchmark-only

# Hardware Testing (requires hardware)
test-loihi: ## Test Loihi deployment
	pytest tests/ -v -m "neuromorphic" -k "loihi"

test-spinnaker: ## Test SpiNNaker deployment  
	pytest tests/ -v -m "neuromorphic" -k "spinnaker"

calibrate-sensors: ## Run sensor calibration
	bioneuro-calibrate --sensors all --reference clean_air --duration 300

# Monitoring
monitor: ## Start monitoring system
	bioneuro-monitor --config config/monitor.yaml

simulate-sensors: ## Run sensor simulation for testing
	python scripts/simulate_sensors.py --count 6 --rate 100

# Data Management
clean-data: ## Clean temporary data files
	rm -rf sensor_data/temp/ calibration_data/temp/
	rm -rf experiments/temp/ analysis_temp/

backup-data: ## Backup important data
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz sensor_data/ calibration_data/ config/

# Continuous Integration helpers
ci-install: ## Install for CI environment
	pip install -e ".[dev,test]"

ci-test: ## Run CI tests
	pytest tests/ --cov=bioneuro_olfactory --cov-report=xml --junit-xml=test-results.xml

ci-security: ## Run CI security checks
	bandit -r bioneuro_olfactory/ -f xml -o bandit-report.xml
	safety check --json --output safety-report.json

# Version management
version: ## Show current version
	python -c "import bioneuro_olfactory; print(bioneuro_olfactory.__version__)"

version-bump-patch: ## Bump patch version
	bumpversion patch

version-bump-minor: ## Bump minor version  
	bumpversion minor

version-bump-major: ## Bump major version
	bumpversion major

# Advanced features
mutation-test: ## Run mutation testing
	mutmut run --paths-to-mutate bioneuro_olfactory/

complexity: ## Analyze code complexity
	radon cc bioneuro_olfactory/ -a
	radon mi bioneuro_olfactory/

dependency-graph: ## Generate dependency graph
	pydeps bioneuro_olfactory --show-deps --max-bacon 3 -o dependency-graph.svg

# Maintenance
update-deps: ## Update dependencies
	pip-compile --upgrade pyproject.toml
	pip-sync

check-deps: ## Check for dependency issues
	pip check
	safety check