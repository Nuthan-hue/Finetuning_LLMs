# Docker Setup Guide

This guide explains how to use the Docker-based setup for the Kaggle Multi-Agent System.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose 1.29+
- At least 16GB RAM (32GB recommended)
- 50GB+ free disk space
- Kaggle API credentials

## Quick Start

### 1. Setup Environment

```bash
# Copy and configure environment variables
cp .env.example .env

# Edit .env with your Kaggle credentials
nano .env  # or use your preferred editor
```

**Important:** Set your Kaggle credentials in `.env`:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 2. Start All Services

```bash
# Start core services (ollama, redis, postgres, jupyter, app)
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f kaggle-agent
```

### 3. Access Services

Once running, you can access:

- **Main Application**: Interactive terminal via `docker attach kaggle-agent-app`
- **Jupyter Lab**: http://localhost:8888
- **Redis**: localhost:6379
- **PostgreSQL**: localhost:5432
- **Ollama API**: http://localhost:11434

## Service Descriptions

### Core Services

#### `kaggle-agent` (Main Application)
- Runs the multi-agent Kaggle competition system
- Interactive menu-based interface
- Mounts local directories for data, models, and submissions
- Access: `docker attach kaggle-agent-app` or `docker exec -it kaggle-agent-app bash`

#### `ollama` (Local LLM)
- Provides local LLM inference via Ollama
- Auto-pulls the `gpt-oss` model on startup
- Health checks ensure service readiness
- Memory limit: 8GB

#### `redis` (Caching & State)
- Caches API responses and intermediate results
- Stores agent state for distributed workflows
- Persistent storage with AOF
- 2GB memory limit with LRU eviction

#### `postgres` (Data Storage)
- Stores agent results, workflow history, and metadata
- Persistent storage for long-term tracking
- Database: `kaggle_agents`
- Default credentials in `.env` (change for production!)

#### `jupyter` (Experimentation)
- Jupyter Lab for data exploration and visualization
- Access all agents programmatically
- Shared volumes with main app
- No authentication (development only!)

### Optional Services (Monitoring)

Start with monitoring profile:
```bash
docker-compose --profile monitoring up -d
```

#### `prometheus` (Metrics Collection)
- Port: 9090
- Scrapes metrics from services
- Requires `monitoring/prometheus.yml` configuration

#### `grafana` (Visualization)
- Port: 3000
- Username: `admin`, Password: `admin`
- Visualize agent performance and system metrics

## Usage Examples

### Run Full Competition Workflow

```bash
# Start services
docker-compose up -d

# Attach to the main application
docker attach kaggle-agent-app

# Follow the interactive menu
# Option 1: Run Full Competition Workflow
# Enter competition name: titanic
```

### Use Jupyter for Experimentation

```bash
# Access Jupyter Lab
open http://localhost:8888

# In a notebook:
from agents import OrchestratorAgent
import asyncio

orchestrator = OrchestratorAgent(
    competition_name="titanic",
    target_percentile=0.20
)

results = await orchestrator.run({})
```

### Direct Agent Usage

```bash
# Execute Python in container
docker exec -it kaggle-agent-app python

>>> from agents import DataCollectorAgent
>>> import asyncio
>>> collector = DataCollectorAgent()
>>> data = asyncio.run(collector.run({"competition_name": "titanic"}))
```

### View Real-time Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f kaggle-agent

# Last 100 lines
docker-compose logs --tail=100 redis
```

## Data Persistence

Data is persisted in Docker volumes and local directories:

### Local Directories (Mounted)
- `./data` - Competition datasets
- `./models` - Trained models
- `./submissions` - Submission files
- `./logs` - Application logs
- `./notebooks` - Jupyter notebooks

### Docker Volumes (Managed)
- `ollama-data` - LLM models
- `redis-data` - Redis persistence
- `postgres-data` - Database files
- `prometheus-data` - Metrics
- `grafana-data` - Dashboards

## Common Commands

### Service Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart a service
docker-compose restart kaggle-agent

# Rebuild after code changes
docker-compose up -d --build

# View running containers
docker-compose ps

# Remove everything (including volumes)
docker-compose down -v
```

### Debugging

```bash
# Shell access to main app
docker exec -it kaggle-agent-app bash

# Check service health
docker-compose ps
docker inspect kaggle-agent-app

# View resource usage
docker stats

# Check logs for errors
docker-compose logs | grep -i error
```

### Database Access

```bash
# PostgreSQL shell
docker exec -it kaggle-postgres psql -U kaggle -d kaggle_agents

# Redis CLI
docker exec -it kaggle-redis redis-cli

# Backup database
docker exec kaggle-postgres pg_dump -U kaggle kaggle_agents > backup.sql
```

## Development Workflow

### 1. Code Changes

The `./src` directory is mounted, so code changes are reflected immediately:

```bash
# Edit code locally
nano src/agents/orchestrator.py

# Restart to apply changes
docker-compose restart kaggle-agent
```

### 2. Install New Dependencies

```bash
# Add to requirements.txt
echo "new-package==1.0.0" >> requirements.txt

# Rebuild
docker-compose up -d --build kaggle-agent
```

### 3. Run Tests

```bash
# Execute tests in container
docker exec -it kaggle-agent-app pytest tests/

# With coverage
docker exec -it kaggle-agent-app pytest --cov=src tests/
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs kaggle-agent

# Check health
docker-compose ps

# Restart specific service
docker-compose restart kaggle-agent
```

### Out of Memory

```bash
# Check resource usage
docker stats

# Adjust memory limits in docker-compose.yml
# Under deploy.resources.limits.memory
```

### Kaggle API Not Working

```bash
# Verify credentials mounted
docker exec -it kaggle-agent-app cat ~/.kaggle/kaggle.json

# Or check environment variables
docker exec -it kaggle-agent-app env | grep KAGGLE
```

### Ollama Model Not Loading

```bash
# Check Ollama logs
docker-compose logs ollama

# Manually pull model
docker exec -it kaggle-ollama ollama pull gpt-oss

# Test Ollama
curl http://localhost:11434/api/tags
```

## Production Deployment

For production use:

1. **Change default passwords** in `.env`:
   - `POSTGRES_PASSWORD`
   - `GF_SECURITY_ADMIN_PASSWORD` (Grafana)

2. **Use secrets management**:
   - Docker secrets or external secret managers
   - Don't commit `.env` to version control

3. **Enable authentication**:
   - Add Jupyter password/token
   - Configure Redis password
   - Use proper Grafana credentials

4. **Resource limits**:
   - Adjust memory/CPU limits based on workload
   - Monitor with Prometheus/Grafana

5. **Networking**:
   - Use reverse proxy (nginx/traefik)
   - Enable HTTPS/TLS
   - Restrict port exposure

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (deletes all data!)
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Clean up system
docker system prune -a
```

## Support

For issues or questions:
- Check logs: `docker-compose logs`
- Inspect containers: `docker inspect <container>`
- Review documentation: `README.md`, `CLAUDE.md`