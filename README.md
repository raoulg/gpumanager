# LLM GPU Controller

A FastAPI-based service for managing GPU workspaces on SURF Research Cloud, designed for auto-scaling LLM inference workloads.

## Features

- **Intelligent GPU Routing**: Automatically selects the best GPU for each request based on model availability, load, and reservation status.
- **Slot Management**: Supports multiple concurrent requests per GPU (configurable slots).
- **Auto-Scaling**: Wakes up paused GPUs when needed and pauses them when idle.
- **Ollama Integration**: Seamless proxy for Ollama API, handling model loading and context management.
- **Docker Support**: Easy deployment of Ollama on GPU nodes using Docker Compose.

## Setup

### GPU Nodes

1. Install Docker and, if necessary, NVIDIA Container Toolkit on your GPU machines.
2. Copy `gpu-node/docker-compose.yml` and `gpu-node/entrypoint.sh` to your GPU machine.
3. (Optional) Edit `docker-compose.yml` to change the `OLLAMA_MODELS` environment variable. You can list multiple models separated by commas (e.g., `llama3.1:8b,qwen2.5:7b`).
4. Run `docker-compose up -d` to start Ollama with GPU support.

### Manager Node

1. Install dependencies:
   ```bash
   pip install -e .
   ```
2. Configure `config.toml` with your cloud API credentials.
   > **Note**: You do NOT need to manually enter GPU IP addresses. The manager automatically discovers them from the cloud provider based on the `machine_name_filter` in your config.

3. Generate an API key for yourself:
   ```bash
   gpumanager generate-key --name "Your Name" --email "your@email.com"
   ```

4. Run the manager:
   ```bash
   gpumanager server
   # or simply
   gpumanager
   ```

## Testing

Run the test suite with pytest:

```bash
pytest
```

## Project Structure

```
llm-gpu-controller/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── config.toml.example
├── api_keys.json.example
├── requirements.txt
├── src/
│   └── gpumanager/
│       ├── __init__.py
│       ├── main.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── models.py          # Pydantic config models
│       │   └── loader.py          # Config loading logic
│       ├── cloud/
│       │   ├── __init__.py
│       │   ├── api.py             # CloudAPI class
│       │   └── models.py          # Cloud response models
│       ├── gpu/
│       │   ├── __init__.py
│       │   ├── manager.py         # GPUManager class (planned)
│       │   ├── state.py           # GPUState class + enums (planned)
│       │   └── models.py          # GPU-related models (planned)
│       ├── auth/
│       │   ├── __init__.py
│       │   ├── manager.py         # APIKeyManager class (planned)
│       │   └── models.py          # User models (planned)
│       └── api/
│           ├── __init__.py
│           ├── handlers.py        # FastAPI route handlers
│           └── middleware.py      # Auth middleware (planned)
└── tests/
    └── ...
```

## Classes Overview

### **Configuration (src/gpumanager/config/)**
- `AppConfig` - Main Pydantic config model with server, cloud API, timing, and paths settings
- `ConfigLoader` - Loads config from TOML files and environment variables

### **Cloud Integration (src/gpumanager/cloud/)**
- `CloudAPI` - SURF Cloud API client with async HTTP operations
- `Workspace` - Workspace model matching SURF cloud API response
- `WorkspaceStatus` - Enum for cloud statuses (running, paused, resuming, etc.)
- `ActionResponse` - Response model for workspace actions

### **API Layer (src/gpumanager/api/)**
- `RequestHandler` - FastAPI application with dynamic GPU management routes

### **GPU Management (src/gpumanager/gpu/)** *(Planned)*
- `GPUState` - Enum for GPU states (PAUSED, STARTING, ACTIVE, BUSY, IDLE)
- `GPUInfo` - Dataclass for GPU metadata (id, ip, status, etc.)
- `GPUManager` - Core GPU orchestration logic

### **Authentication (src/gpumanager/auth/)** *(Planned)*
- `APIKeyManager` - User API key validation
- `UserInfo` - User metadata model

## Setup Instructions

### 1. Clone and Setup Environment

```bash
git clone <repository>
cd llm-gpu-controller

# Copy example files
cp config.toml.example config.toml
cp .env.example .env
cp api_keys.json.example api_keys.json
```
create api keys with

```bash
openssl rand -hex 32
```

### 1.1 set up a host
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b-instruct-q8_0
sudo systemctl edit ollama.service
```
then, add
```toml
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

save and exit. then

```bash 
sudo systemctl daemon-reload
sudo systemctl restart ollama
ss -tulnp | grep 11434
```


### 2. Configure Environment

**Edit `.env`** with your SURF Cloud API credentials:
```bash
CLOUD_API_TOKEN=your_cloud_api_token_here
CLOUD_CSRF_TOKEN=your_csrf_token_here
```

**Edit `config.toml`** if needed (defaults should work):
```toml
[server]
host = "0.0.0.0"
port = 8000

[cloud_api]
base_url = "https://gw.live.surfresearchcloud.nl/v1"
machine_name_filter = "GroulsR"

[timing]
reservation_minutes = 10
fallback_reservation_minutes = 3
startup_timeout_seconds = 120
ollama_readiness_wait_seconds = 10
```

### 3. Install and Run

#### **Option A: Local with UV (Recommended)**

```bash
# Install dependencies
uv sync

# Run the application
uv run python -m gpumanager.main
```

#### **Option B: Local with pip**

```bash
# Install dependencies
pip install -r requirements.txt

# Add src to Python path and run
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m gpumanager.main
```

#### **Option C: Docker**

```bash
# Build and run
docker build -t llm-gpu-controller .
docker run -p 8000:8000 \
  -v $(pwd)/config.toml:/app/config.toml \
  -v $(pwd)/.env:/app/.env \
  llm-gpu-controller
```

## API Usage

### Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/gpu/discover` | Discover all available GPU workspaces |
| `GET` | `/gpu/{gpu_id}/status` | Get specific GPU status |
| `POST` | `/gpu/{gpu_id}/resume` | Resume a GPU workspace |
| `POST` | `/gpu/{gpu_id}/pause` | Pause a GPU workspace |

### Testing with cURL

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Discover Available GPUs
```bash
curl http://localhost:8000/gpu/discover
```

Expected response:
```json
{
  "discovered_gpus": 2,
  "gpus": [
    {
      "id": "451226aa-5ca5-460a-a6d1-5718fcc10e3c",
      "name": "GroulsR-LLM-host",
      "status": "paused",
      "ip_address": "145.38.194.153",
      "can_resume": true,
      "can_pause": false,
      "flavor": "gpu-a10-11core-88gb-50gb-2tb"
    }
  ]
}
```

#### 3. Check GPU Status
```bash
# Use a GPU ID from the discover response
curl http://localhost:8000/gpu/451226aa-5ca5-460a-a6d1-5718fcc10e3c/status
```

#### 4. Resume a GPU
```bash
curl -X POST http://localhost:8000/gpu/451226aa-5ca5-460a-a6d1-5718fcc10e3c/resume
```

#### 5. Pause a GPU
```bash
curl -X POST http://localhost:8000/gpu/451226aa-5ca5-460a-a6d1-5718fcc10e3c/pause
```

### Typical Workflow

1. **Discover GPUs**: `GET /gpu/discover` to get available GPU IDs
2. **Check Status**: `GET /gpu/{gpu_id}/status` to see current state
3. **Control GPU**: `POST /gpu/{gpu_id}/resume` to start, `POST /gpu/{gpu_id}/pause` to stop
4. **Monitor**: Check status periodically during state transitions

## Development Status

### ✅ Phase 1 Complete: Basic GPU Control
- [x] Configuration management with TOML + environment variables
- [x] SURF Cloud API integration
- [x] Dynamic GPU discovery
- [x] Individual GPU control (resume/pause)
- [x] Docker containerization
- [x] Full type safety with Pydantic models

### 🚧 Phase 2: Authentication (In Progress)
- [ ] API key management from JSON file
- [ ] Request authentication middleware
- [ ] User context tracking

### 📋 Phase 3: GPU Management (Planned)
- [ ] Smart GPU state management
- [ ] Request routing and load balancing
- [ ] Automatic GPU lifecycle management
- [ ] Usage-based auto-scaling

## Configuration Reference

### Environment Variables
- `CLOUD_API_TOKEN` - SURF Cloud API authentication token (required)
- `CLOUD_CSRF_TOKEN` - CSRF token for API requests (optional)

### Configuration File (`config.toml`)
- `server.host/port` - Server binding configuration
- `cloud_api.base_url` - SURF Cloud API endpoint
- `cloud_api.machine_name_filter` - Filter for discovering GPU machines
- `timing.*` - Various timeout and timing configurations
- `paths.api_keys_file` - Path to API keys file (for Phase 2)

## Troubleshooting

### Common Issues

1. **"Required environment variable CLOUD_API_TOKEN is not set"**
   - Ensure `.env` file exists and contains valid tokens
   - Check that `.env` is in the project root directory

2. **"Failed to discover GPUs"**
   - Verify your cloud API tokens are valid
   - Check that your `machine_name_filter` matches your GPU workspace names
   - Ensure you have permission to access the SURF Cloud API

3. **"GPU cannot be resumed/paused in current state"**
   - Check the GPU status first with `/gpu/{gpu_id}/status`
   - Wait for current transitions to complete before new actions

### Logs

The application logs to both console and `logs/app.log` with detailed information about:
- Configuration loading
- Cloud API requests and responses  
- GPU state changes
- Error details

## Contributing

1. Follow the existing code structure and typing
2. Add tests for new functionality
3. Update documentation for API changes
4. Use proper logging with the `loguru` logger
