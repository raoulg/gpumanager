## Folder Structure
```
llm-gpu-controller/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── config.toml
├── api_keys.json.example
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── models.py          # Pydantic config models
│   │   └── loader.py          # Config loading logic
│   ├── cloud/
│   │   ├── __init__.py
│   │   ├── api.py             # CloudAPI class
│   │   └── models.py          # Cloud response models
│   ├── gpu/
│   │   ├── __init__.py
│   │   ├── manager.py         # GPUManager class
│   │   ├── state.py           # GPUState class + enums
│   │   └── models.py          # GPU-related models
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── manager.py         # APIKeyManager class
│   │   └── models.py          # User models
│   └── api/
│       ├── __init__.py
│       ├── handlers.py        # FastAPI route handlers
│       └── middleware.py      # Auth middleware
└── tests/
    └── ...
```

## Main Classes Overview

**Configuration (src/config/)**
- `AppConfig` - Main pydantic config model
- `ConfigLoader` - Loads config from TOML/env/JSON

**GPU Management (src/gpu/)**
- `GPUState` - Enum for GPU states (PAUSED, STARTING, ACTIVE, BUSY, IDLE)
- `GPUInfo` - Dataclass for GPU metadata (id, ip, status, etc.)
- `GPUManager` - Core GPU orchestration logic

**Cloud Integration (src/cloud/)**
- `CloudAPI` - SURF cloud API client
- `WorkspaceStatus` - Enum for cloud statuses

**Authentication (src/auth/)**
- `APIKeyManager` - User API key validation
- `UserInfo` - User metadata model

**API Layer (src/api/)**
- `RequestHandler` - FastAPI application and routes
- `AuthMiddleware` - API key validation middleware

