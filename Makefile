# BioNeuro-Olfactory-Fusion Makefile
# Comprehensive build and development automation

# Configuration
PROJECT_NAME := bioneuro-olfactory-fusion
PYTHON_VERSION := 3.11
DOCKER_REGISTRY := your-registry.com
IMAGE_TAG := latest
COMPOSE_FILE := docker-compose.yml
DEV_COMPOSE_FILE := docker-compose.dev.yml

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
WHITE := \033[37m
RESET := \033[0m

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)BioNeuro-Olfactory-Fusion Build System$(RESET)"
	@echo "$(YELLOW)Available commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# Environment Setup
.PHONY: install
install: ## Install project dependencies
	@echo "$(BLUE)Installing project dependencies...$(RESET)"
	pip install -e .
	pip install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed successfully!$(RESET)"

.PHONY: install-dev
install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)Development environment ready!$(RESET)"

.PHONY: setup-env
setup-env: ## Setup environment variables and configuration
	@echo "$(BLUE)Setting up environment...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)Created .env file from template. Please configure it.$(RESET)"; \
	fi
	@echo "$(GREEN)Environment setup complete!$(RESET)"

# Code Quality
.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	black bioneuro_olfactory/ tests/
	isort bioneuro_olfactory/ tests/
	@echo "$(GREEN)Code formatted successfully!$(RESET)"

.PHONY: lint
lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(RESET)"
	flake8 bioneuro_olfactory/ tests/
	pylint bioneuro_olfactory/
	mypy bioneuro_olfactory/
	@echo "$(GREEN)Linting completed!$(RESET)"

.PHONY: security-check
security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r bioneuro_olfactory/ -ll
	safety check
	@echo "$(GREEN)Security checks passed!$(RESET)"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)Pre-commit checks completed!$(RESET)"

# Testing
.PHONY: test
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(RESET)"
	python -m pytest tests/ -v --cov=bioneuro_olfactory --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Tests completed!$(RESET)"

.PHONY: test-unit
test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(RESET)"
	python -m pytest tests/unit/ -v
	@echo "$(GREEN)Unit tests completed!$(RESET)"

.PHONY: test-integration
test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	python -m pytest tests/integration/ -v
	@echo "$(GREEN)Integration tests completed!$(RESET)"

.PHONY: test-performance
test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(RESET)"
	python -m pytest tests/performance/ -v --benchmark-only
	@echo "$(GREEN)Performance tests completed!$(RESET)"

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Starting test watcher...$(RESET)"
	python -m pytest tests/ --watch

# Database Operations
.PHONY: db-migrate
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	python -m bioneuro_olfactory.data.database.migrate
	@echo "$(GREEN)Database migrations completed!$(RESET)"

.PHONY: db-seed
db-seed: ## Seed database with test data
	@echo "$(BLUE)Seeding database...$(RESET)"
	python -m bioneuro_olfactory.data.database.seed
	@echo "$(GREEN)Database seeded!$(RESET)"

.PHONY: db-reset
db-reset: ## Reset database (WARNING: Destroys all data)
	@echo "$(RED)WARNING: This will destroy all database data!$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	python -m bioneuro_olfactory.data.database.reset
	@echo "$(GREEN)Database reset completed!$(RESET)"

# Docker Operations
.PHONY: docker-build
docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(RESET)"
	docker build -t $(PROJECT_NAME):$(IMAGE_TAG) .
	docker build -t $(PROJECT_NAME):dev --target development .
	@echo "$(GREEN)Docker images built successfully!$(RESET)"

.PHONY: docker-push
docker-push: docker-build ## Push Docker images to registry
	@echo "$(BLUE)Pushing Docker images...$(RESET)"
	docker tag $(PROJECT_NAME):$(IMAGE_TAG) $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(IMAGE_TAG)
	docker push $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(IMAGE_TAG)
	@echo "$(GREEN)Docker images pushed!$(RESET)"

.PHONY: up
up: ## Start production services
	@echo "$(BLUE)Starting production services...$(RESET)"
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)Production services started!$(RESET)"

.PHONY: up-dev
up-dev: ## Start development services
	@echo "$(BLUE)Starting development services...$(RESET)"
	docker-compose -f $(DEV_COMPOSE_FILE) up -d
	@echo "$(GREEN)Development services started!$(RESET)"

.PHONY: down
down: ## Stop all services
	@echo "$(BLUE)Stopping services...$(RESET)"
	docker-compose -f $(COMPOSE_FILE) down
	docker-compose -f $(DEV_COMPOSE_FILE) down
	@echo "$(GREEN)Services stopped!$(RESET)"

.PHONY: logs
logs: ## Show service logs
	docker-compose -f $(COMPOSE_FILE) logs -f

.PHONY: logs-dev
logs-dev: ## Show development service logs
	docker-compose -f $(DEV_COMPOSE_FILE) logs -f

# Development Tools
.PHONY: jupyter
jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	docker-compose -f $(DEV_COMPOSE_FILE) up -d jupyter
	@echo "$(GREEN)Jupyter Lab available at http://localhost:8888$(RESET)"

.PHONY: docs
docs: ## Build and serve documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	mkdocs build
	mkdocs serve --dev-addr=127.0.0.1:8080
	@echo "$(GREEN)Documentation available at http://localhost:8080$(RESET)"

.PHONY: docs-build
docs-build: ## Build documentation for deployment
	@echo "$(BLUE)Building documentation for deployment...$(RESET)"
	mkdocs build --clean
	@echo "$(GREEN)Documentation built in site/ directory!$(RESET)"

# Monitoring and Debugging
.PHONY: monitor
monitor: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring services...$(RESET)"
	docker-compose -f $(COMPOSE_FILE) up -d prometheus grafana
	@echo "$(GREEN)Monitoring available:$(RESET)"
	@echo "  $(CYAN)Prometheus:$(RESET) http://localhost:9090"
	@echo "  $(CYAN)Grafana:$(RESET) http://localhost:3000 (admin/admin)"

.PHONY: profile
profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	python -m cProfile -o profile.stats -m bioneuro_olfactory.models.fusion.multimodal_fusion
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)Profiling completed!$(RESET)"

# Deployment
.PHONY: deploy-staging
deploy-staging: docker-build docker-push ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(RESET)"
	# Add your staging deployment commands here
	@echo "$(GREEN)Deployed to staging!$(RESET)"

.PHONY: deploy-prod
deploy-prod: test docker-build docker-push ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(RESET)"
	@echo "$(RED)WARNING: Deploying to production!$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	# Add your production deployment commands here
	@echo "$(GREEN)Deployed to production!$(RESET)"

# Maintenance
.PHONY: clean
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "$(GREEN)Cleanup completed!$(RESET)"

.PHONY: clean-docker
clean-docker: ## Clean up Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(RESET)"
	docker system prune -f
	docker volume prune -f
	@echo "$(GREEN)Docker cleanup completed!$(RESET)"

.PHONY: reset-all
reset-all: clean clean-docker ## Reset everything (clean + Docker cleanup)
	@echo "$(BLUE)Resetting all environments...$(RESET)"
	docker-compose -f $(COMPOSE_FILE) down -v
	docker-compose -f $(DEV_COMPOSE_FILE) down -v
	@echo "$(GREEN)Full reset completed!$(RESET)"

# Release Management
.PHONY: version
version: ## Show current version
	@python -c "import bioneuro_olfactory; print(f'Version: {bioneuro_olfactory.__version__}')"

.PHONY: release-patch
release-patch: ## Create patch release
	@echo "$(BLUE)Creating patch release...$(RESET)"
	semantic-release version --patch
	@echo "$(GREEN)Patch release created!$(RESET)"

.PHONY: release-minor
release-minor: ## Create minor release
	@echo "$(BLUE)Creating minor release...$(RESET)"
	semantic-release version --minor
	@echo "$(GREEN)Minor release created!$(RESET)"

.PHONY: release-major
release-major: ## Create major release
	@echo "$(BLUE)Creating major release...$(RESET)"
	semantic-release version --major
	@echo "$(GREEN)Major release created!$(RESET)"

# Comprehensive Quality Check
.PHONY: check-all
check-all: format lint security-check test ## Run all quality checks
	@echo "$(GREEN)All quality checks passed!$(RESET)"

# Development Workflow
.PHONY: dev-setup
dev-setup: setup-env install-dev ## Complete development setup
	@echo "$(GREEN)Development environment fully configured!$(RESET)"

.PHONY: dev-start
dev-start: up-dev ## Start development environment
	@echo "$(GREEN)Development environment started!$(RESET)"
	@echo "$(CYAN)Available services:$(RESET)"
	@echo "  $(WHITE)API:$(RESET) http://localhost:8000"
	@echo "  $(WHITE)Jupyter:$(RESET) http://localhost:8888"
	@echo "  $(WHITE)pgAdmin:$(RESET) http://localhost:5050"
	@echo "  $(WHITE)Redis Commander:$(RESET) http://localhost:8081"

.PHONY: dev-stop
dev-stop: ## Stop development environment
	docker-compose -f $(DEV_COMPOSE_FILE) down
	@echo "$(GREEN)Development environment stopped!$(RESET)"

# Default target
.DEFAULT_GOAL := help