.PHONY: help build up down restart logs shell test clean backup

# Default target
help:
	@echo "Kaggle Multi-Agent System - Docker Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initial setup (copy .env, create dirs)"
	@echo "  make build          - Build Docker images"
	@echo ""
	@echo "Service Management:"
	@echo "  make up             - Start all services"
	@echo "  make up-monitoring  - Start with monitoring (Prometheus + Grafana)"
	@echo "  make down           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make ps             - Show running containers"
	@echo "  make stats          - Show resource usage"
	@echo ""
	@echo "Development:"
	@echo "  make logs           - Follow all logs"
	@echo "  make logs-app       - Follow app logs only"
	@echo "  make shell          - Open bash in main container"
	@echo "  make shell-db       - Open PostgreSQL shell"
	@echo "  make shell-redis    - Open Redis CLI"
	@echo "  make jupyter        - Show Jupyter Lab URL"
	@echo ""
	@echo "Application:"
	@echo "  make run            - Attach to interactive app"
	@echo "  make test           - Run tests"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Lint code"
	@echo ""
	@echo "Maintenance:"
	@echo "  make backup         - Backup database"
	@echo "  make restore        - Restore database from backup"
	@echo "  make clean          - Stop and remove containers"
	@echo "  make clean-all      - Remove containers, volumes, and images"
	@echo "  make prune          - Clean up Docker system"

# Setup commands
setup:
	@echo "Setting up environment..."
	@cp -n .env.example .env || true
	@mkdir -p data models submissions logs notebooks monitoring
	@echo "Setup complete! Edit .env with your Kaggle credentials"

build:
	@echo "Building Docker images..."
	docker-compose build --parallel

# Service management
up:
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started! Run 'make logs' to view logs"

up-monitoring:
	@echo "Starting services with monitoring..."
	docker-compose --profile monitoring up -d
	@echo "Services started with monitoring!"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"

down:
	@echo "Stopping services..."
	docker-compose down

restart:
	@echo "Restarting services..."
	docker-compose restart

ps:
	docker-compose ps

stats:
	docker stats

# Logs
logs:
	docker-compose logs -f

logs-app:
	docker-compose logs -f kaggle-agent

logs-ollama:
	docker-compose logs -f ollama

logs-error:
	docker-compose logs | grep -i error

# Shell access
shell:
	docker exec -it kaggle-agent-app bash

shell-db:
	docker exec -it kaggle-postgres psql -U kaggle -d kaggle_agents

shell-redis:
	docker exec -it kaggle-redis redis-cli

jupyter:
	@echo "Jupyter Lab: http://localhost:8888"
	@echo "No token required (development mode)"

# Application commands
run:
	@echo "Attaching to application (Ctrl+P then Ctrl+Q to detach)..."
	docker attach kaggle-agent-app

test:
	@echo "Running tests..."
	docker exec -it kaggle-agent-app pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	docker exec -it kaggle-agent-app pytest --cov=src tests/

format:
	@echo "Formatting code..."
	docker exec -it kaggle-agent-app black src/
	docker exec -it kaggle-agent-app isort src/

lint:
	@echo "Linting code..."
	docker exec -it kaggle-agent-app flake8 src/
	docker exec -it kaggle-agent-app mypy src/

# Database operations
backup:
	@echo "Backing up database..."
	@mkdir -p backups
	docker exec kaggle-postgres pg_dump -U kaggle kaggle_agents > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup created in backups/"

restore:
	@read -p "Enter backup file path: " backup_file; \
	docker exec -i kaggle-postgres psql -U kaggle kaggle_agents < $$backup_file
	@echo "Database restored"

db-reset:
	@echo "Resetting database..."
	docker-compose down postgres
	docker volume rm finetuning_llms_postgres-data || true
	docker-compose up -d postgres
	@echo "Database reset complete"

# Cleanup
clean:
	@echo "Stopping and removing containers..."
	docker-compose down
	@echo "Cleanup complete"

clean-all:
	@echo "WARNING: This will remove all containers, volumes, and images"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		docker-compose down -v --rmi all; \
		echo "Full cleanup complete"; \
	else \
		echo "Cancelled"; \
	fi

prune:
	@echo "Cleaning up Docker system..."
	docker system prune -f
	@echo "System cleanup complete"

# Quick rebuild after code changes
rebuild:
	@echo "Rebuilding and restarting app..."
	docker-compose up -d --build kaggle-agent
	@echo "App rebuilt and restarted"

# Check service health
health:
	@echo "Checking service health..."
	@echo "\nOllama:"
	@curl -s http://localhost:11434/api/tags | python -m json.tool || echo "Ollama not responding"
	@echo "\nRedis:"
	@docker exec kaggle-redis redis-cli ping || echo "Redis not responding"
	@echo "\nPostgreSQL:"
	@docker exec kaggle-postgres pg_isready -U kaggle || echo "PostgreSQL not responding"
	@echo "\nContainer Status:"
	@docker-compose ps

# Initialize database
init-db:
	@echo "Initializing database..."
	docker exec -i kaggle-postgres psql -U kaggle kaggle_agents < init-db.sql
	@echo "Database initialized"

# Pull latest models
pull-models:
	@echo "Pulling Ollama models..."
	docker exec kaggle-ollama ollama pull gpt-oss
	@echo "Models pulled"