# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Manager is a FastAPI-based service for intelligent GPU orchestration and auto-scaling for LLM inference workloads on SURF Research Cloud. It manages multiple GPU nodes running Ollama, providing automatic GPU discovery, load balancing, lifecycle management (pause/resume), and intelligent request routing.

**Architecture**: Centralized manager node running the GPU Manager API + OpenWebUI, connected to multiple GPU nodes running Ollama in Docker containers.

## System-Specific Configuration

**Important**: This system has custom configurations:

1. **File Removal**: `rm` is remapped to `rip` (trash/bin instead of permanent deletion)
   - `rip` does NOT use the `-r` flag for directories
   - Simply use `rm <path>` for both files and directories
   - Do NOT use `rm -r` - it won't work as expected

2. **Package Management**: This project uses `uv` exclusively
   - Use `uv add <package>` to add dependencies (NOT `pip install`)
   - Use `uv run <command>` to run commands in the venv
   - Use `uv sync` to install dependencies (NOT `pip install -r requirements.txt`)
   - Never use pip commands in this project

## Common Commands

### Development

```bash
# Install dependencies
uv sync

# Run the manager server
uv run gpumanager server
# or simply
uv run gpumanager

# Run tests
pytest
pytest tests/test_integration.py -v -s  # Specific test with verbose output
```

### Deployment

```bash
# Deploy GPU nodes (auto-discover from cloud)
uv run gpumanager deploy <username>

# Deploy GPU nodes (manual IP list)
uv run gpumanager deploy <username> --ips ips.txt

# Deploy manager node with WebUI and API
uv run gpumanager deploy --manager <manager-ip> --with-api

# Sync models from one node to all others
uv run gpumanager sync-models --source <source-node-ip>

# Open firewall ports
uv run gpumanager open-port --ip <node-ip> --ports 8000,8080

# Generate API key
uv run gpumanager generate-key --name "User Name" --email "user@email.com"
```

## Architecture Overview

### Core Components

**1. GPU Manager (`src/gpumanager/gpu/manager.py`)**
- Central orchestration engine managing GPU lifecycle and request routing
- Maintains in-memory state of all GPUs via `GPUInfo` objects
- Implements intelligent GPU selection algorithm with 4-tier priority system:
  1. GPU with model already loaded + available slots
  2. Available idle GPU (no model loaded)
  3. STARTING GPU (prevents race conditions)
  4. PAUSED GPU (wake up on demand)
- Runs 3 background tasks:
  - `_idle_monitor_loop`: Auto-pauses GPUs idle > reservation_minutes
  - `_reservation_cleanup_loop`: Cleans expired reservations every 30s
  - `_status_sync_loop`: Syncs GPU status with Cloud API every 30s

**2. Ollama Proxy (`src/gpumanager/api/ollama_proxy.py`)**
- Intelligent proxy for Ollama API requests
- Routes requests to optimal GPU based on model availability and load
- Handles streaming responses (chat/generate)
- Auto-wakes paused GPUs when needed
- Aggregates model lists from all active GPUs
- Prevents concurrent requests from same user via per-user locks

**3. Cloud API Integration (`src/gpumanager/cloud/api.py`)**
- Wrapper around SURF Research Cloud API (via surf-controller package)
- Discovers GPU workspaces based on `machine_name_filter` in config
- Controls workspace lifecycle: resume, pause, wait for status
- Manages Network Security Group (NSG) firewall rules
- All operations are async using httpx

**4. Request Handlers (`src/gpumanager/api/handlers.py`)**
- FastAPI routes for GPU management + Ollama proxy endpoints
- Authentication via API key middleware (Bearer tokens)
- GPU management endpoints: `/gpu/discover`, `/gpu/{id}/status`, `/gpu/{id}/resume`, `/gpu/{id}/pause`
- Ollama proxy endpoints: `/api/generate`, `/api/chat`, `/api/pull`, `/api/tags`, `/v1/chat/completions`
- Optional authentication support via `get_optional_user` for flexibility

**5. Deployment Manager (`src/gpumanager/deployment.py`)**
- Handles automated deployment to GPU nodes and manager node
- Uses SSH + rsync to deploy docker-compose setups
- Supports both auto-discovery (via Cloud API) and manual IP lists
- Idempotent deployment with progress markers to resume interrupted deployments
- Manages firewall rules, Docker installation, and service startup

### State Management

**GPU States (`GPUModelStatus` enum)**:
- `PAUSED`: VM paused (no cost)
- `STARTING`: VM resuming
- `IDLE`: VM running, no model loaded
- `LOADING_MODEL`: Model being pulled/loaded
- `MODEL_READY`: Model loaded, ready for requests
- `BUSY`: Processing requests
- `PAUSING`: VM being paused
- `ERROR`: Error state

**Key State Tracking**:
- `GPUInfo`: Complete GPU state including status, loaded_model, reservation, active_requests, max_slots
- `ModelInfo`: Loaded model metadata (name, size, context_length, loaded_at, last_used)
- `GPUReservation`: Temporary exclusive lock for user requests (expires in reservation_minutes)

### Request Flow

1. **Request arrives** → Ollama Proxy receives `/api/chat` or `/api/generate`
2. **GPU Selection** → GPU Manager selects best GPU via `select_gpu()`
3. **Reservation** → GPU reserved for user (prevents concurrent access)
4. **Startup (if needed)** → Wake paused GPU, wait for Ollama ready
5. **Model Load (if needed)** → Load model if not already present
6. **Proxy Request** → Forward to GPU's Ollama instance
7. **Stream Response** → Stream tokens back to client
8. **Cleanup** → Mark request complete, clear reservation

## Configuration

**config.toml**:
- `cloud_api.machine_name_filter`: Pattern to filter GPU workspaces (e.g., "LMSTUDIO")
- `timing.reservation_minutes`: How long to reserve GPU for user (default: 10)
- `timing.startup_timeout_seconds`: Max time to wait for GPU startup (default: 120)
- `paths.api_keys_file`: Location of API keys JSON file

**Environment (.env)**:
- `CLOUD_API_TOKEN`: SURF Cloud API token (required for cloud operations)
- `CLOUD_CSRF_TOKEN`: CSRF token (optional)
- `SSH_USER`: SSH username for deployment

## GPU Node vs Manager Node

**GPU Nodes** (`gpu-node/`):
- Run Ollama in Docker with GPU access via `--gpus all`
- Minimal setup: docker-compose.yml, entrypoint.sh (pulls models on startup)
- Models stored in Docker volume (`ollama_data`) at `/root/.ollama` (inside container)
- **Redeployment preserves models** - Ollama models persist across redeployments
- Environment: `OLLAMA_MODELS` (comma-separated list of models to pre-load, default: `llama3.1:8b-instruct-q8_0`)
- Port 11434 must be open in firewall

**Manager Node** (`manager-node/`):
- Runs GPU Manager API, OpenWebUI, and Caddy (reverse proxy with basic auth)
- Centralized access point for users (single URL: `http://<manager-ip>:8080`)
- Auto-discovers and load-balances across GPU nodes
- **Redeployment wipes WebUI data** - Creates fresh admin user with new credentials
- Credentials auto-generated on deployment (check deployment logs or `/srv/shared/.env`)
- Ports 8000 (API) and 8080 (WebUI) must be open in firewall
- Deployed files location: `/srv/shared/` (docker-compose.yml, Caddyfile, .env, config.toml, src/)

**Authentication Layers**:
1. **Caddy Basic Auth** (Gatekeeper): Username `gatekeeper`, password in `/srv/shared/.env`
2. **WebUI Login**: Email `admin@gpumanager.local`, password in `/srv/shared/.env`
3. **API Authentication**: Bearer token API keys (from `api_keys.json`)

## Important Patterns

**1. Async/Await Everywhere**
- All GPU operations, Cloud API calls, and HTTP requests are async
- Use `asyncio.create_task()` for background operations (don't block)
- Background tasks tracked in `_background_tasks` set with done callbacks

**2. Reservation System**
- Prevents race conditions when multiple users request GPUs simultaneously
- Temporary (10 min default) exclusive lock per user
- Auto-expires via background cleanup loop

**3. Model Loading Strategy**
- Models loaded on-demand, not pre-loaded
- Once loaded, model stays in GPU memory until GPU is paused
- Multiple requests can share same loaded model (up to `max_slots`, default 3)

**4. Error Handling**
- Cloud API errors wrapped in `CloudAPIError`
- GPUs marked as `ERROR` state on failures
- Status sync loop attempts recovery by reconciling with cloud state

**5. Logging**
- Uses `loguru` with structured logging
- Console: INFO level, color-coded
- File (`logs/app.log`): DEBUG level, rotated at 10MB, 7-day retention
- Log GPU names in messages for clarity (not just IDs)

## Testing Strategy

- **Integration tests** (`tests/test_integration.py`): Real Cloud API + GPU interactions
- **Unit tests** (`tests/test_gpu_manager.py`, `tests/test_ollama_proxy.py`): Mock Cloud API
- Use `pytest-asyncio` for async test support
- Test configuration: `asyncio_mode = "strict"` in pyproject.toml

## Development Notes

- **Cloud API Discovery**: Workspace discovery is automatic if `machine_name_filter` is set. Manual IP lists bypass discovery but lose smart features (auto-resume, reverse lookup).
- **Firewall Management**: NSG rules must be opened for ports 8000 (API), 8080 (WebUI), 11434 (Ollama). Use `gpumanager open-port` command.
- **Model Synchronization**: Models are node-local. Use `sync-models` command to distribute across nodes via rsync over SSH.
- **Deployment Idempotency**: Deployment uses progress markers (`/srv/shared/.setup_progress`) to safely resume interrupted deployments.
- **Docker-in-Docker**: GPU nodes run Ollama in containers with `--gpus all` flag for NVIDIA GPU access.
- **Deployment Location**: All nodes deploy to `/srv/shared/` directory (shared volume accessible by collaborators group).
- **Model Storage**: Ollama models stored in Docker volume `ollama_data` at `/root/.ollama` inside container. Persists across container restarts.
- **Credentials**: Manager node auto-generates `WEBUI_ADMIN_PASSWORD`, `GATEKEEPER_PASSWORD`, and `GATEKEEPER_HASH` on first deployment. Check logs or `/srv/shared/.env`.

## File Structure

```
src/gpumanager/
├── main.py                 # Entry point, CLI commands, app factory
├── config/
│   ├── models.py          # Pydantic config models
│   └── loader.py          # TOML + env loading
├── cloud/
│   ├── api.py             # CloudAPI wrapper (SURF API client)
│   └── models.py          # Workspace models
├── gpu/
│   ├── manager.py         # GPUManager (core orchestration)
│   ├── state.py           # GPUInfo, GPUModelStatus, ModelInfo
│   └── models.py          # Request/response models
├── auth/
│   ├── manager.py         # APIKeyManager (JSON-based auth)
│   └── models.py          # User models
├── api/
│   ├── handlers.py        # FastAPI routes
│   ├── middleware.py      # Auth middleware
│   ├── ollama_proxy.py    # OllamaProxy (intelligent routing)
│   └── ollama_models.py   # Request/response models
├── deployment.py          # DeploymentManager (SSH deployment)
└── sync.py                # ModelSynchronizer (rsync models)

gpu-node/                   # GPU node deployment files
manager-node/               # Manager node deployment files
tests/                      # Test suite
```

## Common Workflows

### Deployment Workflows

**Adding a new GPU node**:
```bash
# Deploy to new node
uv run gpumanager deploy --ips new_node_ip.txt

# Sync models from existing node
uv run gpumanager sync-models --source <existing-node-ip>

# Open firewall port
uv run gpumanager open-port --ip <new-node-ip> --port 11434

# GPU Manager API automatically discovers the new node - no manual config needed!
```

**Updating models across all nodes**:
```bash
# Download model to one node via WebUI or directly
ssh <gpu-ip> 'docker exec -it ollama ollama pull mistral:7b'

# Then sync to all others
uv run gpumanager sync-models --source <node-with-new-model>
```

**Redeploying**:
```bash
# Redeploy manager (WARNING: wipes WebUI data, creates fresh admin)
uv run gpumanager deploy --manager <manager-ip> --with-api

# Redeploy GPU nodes (preserves Ollama models in Docker volumes)
uv run gpumanager deploy
```

**Pre-configuring models before deployment**:
Edit `gpu-node/.env` before deploying:
```bash
OLLAMA_MODELS="llama3.1:8b-instruct-q8_0,mistral:7b,codellama:13b"
```
Then deploy - all nodes will pull these models on first startup.

### Code Workflows

**Adding a new GPU endpoint**:
1. Add route in `RequestHandler._create_app()` (handlers.py)
2. Implement handler method in `RequestHandler` class
3. Add authentication dependency: `dependencies=[Depends(self.get_current_user)]`
4. Add response model if needed

**Modifying GPU selection logic**:
- Edit `GPUManager.select_gpu()` (gpu/manager.py)
- Priority system is: model loaded → idle → starting → paused
- Test with `tests/test_gpu_manager.py`

**Adding cloud provider support**:
- Extend `CloudAPI` class (cloud/api.py)
- Implement: `list_workspaces()`, `resume_workspace()`, `pause_workspace()`, `wait_for_workspace_status()`
- Update `WorkspaceStatus` enum if needed

## Troubleshooting

### Manager Node Issues

**Can't access WebUI**:
```bash
# Check if ports are open in firewall
uv run gpumanager open-port --ip <manager-ip> --ports 8000,8080

# Check if services are running
ssh <manager-ip>
cd /srv/shared
docker compose logs -f
```

**Admin login fails**:
- Check deployment logs for auto-generated credentials
- Or check `/srv/shared/.env` on manager node for `WEBUI_ADMIN_PASSWORD` and `GATEKEEPER_PASSWORD`
- Or redeploy (WARNING: wipes WebUI data): `uv run gpumanager deploy --manager <manager-ip>`

**"No GPUs available"**:
- Check `machine_name_filter` in config.toml matches your workspace names in SURF Cloud
- Verify Cloud API credentials in `.env` file

### GPU Node Issues

**Ollama not responding**:
```bash
# Test locally on the GPU node
ssh <gpu-node-ip>
curl localhost:11434

# Check Ollama logs
cd /srv/shared
docker compose logs ollama
```

**Models not loading**:
```bash
# Check if models are pulling
ssh <gpu-node-ip>
docker compose logs ollama | grep -i pull

# Manually pull a model
docker exec -it ollama ollama pull llama3.1:8b-instruct-q8_0
```

**Can't connect from WebUI to GPU node**:
```bash
# Ensure port 11434 is open
uv run gpumanager open-port --ip <gpu-node-ip> --port 11434

# Test connectivity from manager node
ssh <manager-ip>
curl http://<gpu-node-ip>:11434
```

**GPU stuck in STARTING**:
- Check `startup_timeout_seconds` in config.toml
- Verify SSH access to GPU node
- Check if Ollama service started: `ssh <gpu-ip> 'cd /srv/shared && docker compose logs ollama'`

**Models not syncing between nodes**:
- Verify SSH key-based auth is configured (no password prompts)
- Check `/srv/shared` directory exists on all nodes
- Verify source node has models: `ssh <source-ip> 'docker exec ollama ollama list'`

**Authentication failures**:
- Verify API key exists in `api_keys.json`
- Check Authorization header format: `Bearer sk-...`
- For WebUI: verify Caddy basic auth (username: `gatekeeper`) and WebUI login (email: `admin@gpumanager.local`)
