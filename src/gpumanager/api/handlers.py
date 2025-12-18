"""FastAPI request handlers."""

from typing import Any, Dict, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Path,  # Add Request
    Request,
    status,
)
from fastapi.responses import (
    JSONResponse,  # Add JSONResponse
    StreamingResponse,
)
from loguru import logger
from pydantic import BaseModel

from gpumanager.api.middleware import (
    create_auth_dependency,
    create_optional_auth_dependency,
)
from gpumanager.api.ollama_models import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OpenAIChatRequest,
)
from gpumanager.api.ollama_proxy import OllamaProxy
from gpumanager.auth.manager import APIKeyManager
from gpumanager.auth.models import AuthenticatedUser
from gpumanager.cloud.api import CloudAPI
from gpumanager.gpu.manager import GPUManager
from gpumanager.gpu.models import GPUManagerStats


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str


class GPUStatusResponse(BaseModel):
    """GPU status response."""

    gpu_id: str
    status: str
    ip_address: str
    can_resume: bool
    can_pause: bool


class ActionResponse(BaseModel):
    """Action response."""

    success: bool
    message: str
    action_id: Optional[str] = None


class RequestHandler:
    """FastAPI request handlers."""

    def __init__(
        self,
        cloud_api: CloudAPI,
        api_key_manager: APIKeyManager,
        gpu_manager: GPUManager,
        lifespan=None,
    ):
        """Initialize request handler."""
        self.cloud_api = cloud_api
        self.api_key_manager = api_key_manager
        self.gpu_manager = gpu_manager
        self.lifespan = lifespan

        # Initialize Ollama proxy
        self.ollama_proxy = OllamaProxy(gpu_manager)

        # Create auth dependencies
        self.get_current_user = create_auth_dependency(api_key_manager)
        self.get_optional_user = create_optional_auth_dependency(api_key_manager)

        self.app = self._create_app()

        logger.info(
            "Initialized RequestHandler with GPU management, authentication, and Ollama proxy"
        )

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="LLM GPU Controller",
            description="GPU management API for LLM inference",
            version="0.1.0",
            lifespan=self.lifespan,
        )

        # Register routes
        app.get("/health", response_model=HealthResponse)(self.health_check)

        # GPU management routes (require authentication)
        app.get("/gpu/discover", dependencies=[Depends(self.get_current_user)])(
            self.discover_gpus
        )
        app.get(
            "/gpu/stats",
            response_model=GPUManagerStats,
            dependencies=[Depends(self.get_current_user)],
        )(self.get_gpu_stats)
        app.get(
            "/gpu/{gpu_id}/status",
            response_model=GPUStatusResponse,
            dependencies=[Depends(self.get_current_user)],
        )(self.get_gpu_status)
        app.post(
            "/gpu/{gpu_id}/resume",
            response_model=ActionResponse,
            dependencies=[Depends(self.get_current_user)],
        )(self.resume_gpu)
        app.post(
            "/gpu/{gpu_id}/pause",
            response_model=ActionResponse,
            dependencies=[Depends(self.get_current_user)],
        )(self.pause_gpu)

        # Ollama proxy routes (require authentication)
        app.post("/api/generate")(self._create_ollama_generate_handler())
        app.post("/api/chat")(self._create_ollama_chat_handler())
        app.post("/v1/chat/completions")(self._create_openai_chat_handler())
        # Add this line in _create_app():
        app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(
            self.ollama_passthrough
        )

        return app

    async def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", service="llm-gpu-controller")

    async def discover_gpus(self) -> Dict[str, Any]:
        """Discover available GPU workspaces with enhanced information."""
        try:
            gpu_info = []
            for gpu in self.gpu_manager.gpus.values():
                gpu_data = {
                    "id": gpu.gpu_id,
                    "name": gpu.name,
                    "status": gpu.status.value,
                    "ip_address": gpu.ip_address,
                    "flavor": gpu.flavor,
                    "total_requests": gpu.total_requests,
                    "requests_today": gpu.requests_today,
                    "loaded_model": gpu.loaded_model.name if gpu.loaded_model else None,
                    "model_size": gpu.loaded_model.size if gpu.loaded_model else None,
                    "idle_since": gpu.idle_since.isoformat()
                    if gpu.idle_since
                    else None,
                    "is_available": gpu.is_available(),
                    "reservation": {
                        "user_id": gpu.reservation.user_id,
                        "expires_at": gpu.reservation.expires_at.isoformat(),
                        "model_name": gpu.reservation.model_name,
                    }
                    if gpu.reservation
                    else None,
                }
                gpu_info.append(gpu_data)

            return {"discovered_gpus": len(gpu_info), "gpus": gpu_info}

        except Exception as e:
            logger.error(f"Failed to discover GPUs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to discover GPUs: {str(e)}",
            )

    async def get_gpu_stats(self) -> GPUManagerStats:
        """Get GPU manager statistics."""
        try:
            return self.gpu_manager.get_gpu_stats()
        except Exception as e:
            logger.error(f"Failed to get GPU stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get GPU stats: {str(e)}",
            )

    async def get_gpu_status(
        self, gpu_id: str = Path(..., description="GPU workspace ID")
    ) -> GPUStatusResponse:
        """Get current GPU status with enhanced information."""
        try:
            if gpu_id not in self.gpu_manager.gpus:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"GPU {gpu_id} not found",
                )

            gpu = self.gpu_manager.gpus[gpu_id]

            return GPUStatusResponse(
                gpu_id=gpu.gpu_id,
                status=gpu.status.value,
                ip_address=gpu.ip_address,
                can_resume=gpu.status.value == "paused",
                can_pause=gpu.status.value in ["idle", "model_ready"],
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get GPU status for {gpu_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get GPU status: {str(e)}",
            )

    async def resume_gpu(
        self, gpu_id: str = Path(..., description="GPU workspace ID")
    ) -> ActionResponse:
        """Resume the GPU workspace using GPU manager."""
        try:
            if gpu_id not in self.gpu_manager.gpus:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"GPU {gpu_id} not found",
                )

            gpu = self.gpu_manager.gpus[gpu_id]

            if gpu.status.value != "paused":
                return ActionResponse(
                    success=True, message=f"GPU is already in {gpu.status.value} state"
                )

            # Use GPU manager to start the GPU
            success = await self.gpu_manager.start_gpu(gpu_id)

            if success:
                return ActionResponse(
                    success=True, message="GPU started successfully", action_id=gpu_id
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to start GPU",
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to resume GPU {gpu_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to resume GPU: {str(e)}",
            )

    async def pause_gpu(
        self, gpu_id: str = Path(..., description="GPU workspace ID")
    ) -> ActionResponse:
        """Pause the GPU workspace using GPU manager."""
        try:
            if gpu_id not in self.gpu_manager.gpus:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"GPU {gpu_id} not found",
                )

            gpu = self.gpu_manager.gpus[gpu_id]

            if gpu.status.value == "paused":
                return ActionResponse(success=True, message="GPU is already paused")

            if gpu.status.value not in ["idle", "model_ready"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"GPU cannot be paused in {gpu.status.value} state",
                )

            # Use GPU manager to pause the GPU
            success = await self.gpu_manager.pause_gpu(gpu_id)

            if success:
                return ActionResponse(
                    success=True, message="GPU paused successfully", action_id=gpu_id
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to pause GPU",
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to pause GPU {gpu_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to pause GPU: {str(e)}",
            )

    def _create_ollama_generate_handler(self):
        """Create Ollama generate handler with proper dependency injection."""

        async def ollama_generate(
            request: OllamaGenerateRequest,
            current_user: AuthenticatedUser = Depends(self.get_current_user),
        ) -> StreamingResponse:
            """Ollama generate endpoint with intelligent GPU routing."""
            return await self.ollama_proxy.generate(request, current_user)

        return ollama_generate

    def _create_ollama_chat_handler(self):
        """Create Ollama chat handler with proper dependency injection."""

        async def ollama_chat(
            request: OllamaChatRequest,
            current_user: AuthenticatedUser = Depends(self.get_current_user),
        ) -> StreamingResponse:
            """Ollama chat endpoint with intelligent GPU routing."""
            return await self.ollama_proxy.chat(request, current_user)

        return ollama_chat

    def _create_openai_chat_handler(self):
        """Create OpenAI chat handler with proper dependency injection."""

        async def openai_chat_completions(
            request: OpenAIChatRequest,
            current_user: AuthenticatedUser = Depends(self.get_current_user),
        ) -> StreamingResponse:
            """OpenAI-compatible chat completions endpoint."""
            return await self.ollama_proxy.openai_chat_completions(
                request, current_user
            )

        return openai_chat_completions

    async def ollama_passthrough(self, request: Request, path: str = Path(...)):
        """Pass-through proxy for any Ollama API endpoint."""
        try:
            # Get any available GPU
            # We use select_gpu with a dummy model request to find an available GPU
            # This ensures we respect slot limits
            from gpumanager.gpu.models import GPUSelectionRequest
            
            # Use a dummy user ID for passthrough requests if not authenticated
            # Ideally this endpoint should be authenticated too, but for now we'll use a placeholder
            user_id = "passthrough_user"
            
            selection_request = GPUSelectionRequest(
                user_id=user_id, 
                model_name="unknown", # We don't know the model here easily without parsing body
                context_length=None
            )
            
            # 1. Select GPU
            gpu_result = await self.gpu_manager.select_gpu(selection_request)
            
            if not gpu_result.gpu_info:
                raise HTTPException(status_code=503, detail="No GPUs available")
                
            gpu = gpu_result.gpu_info
            
            # 2. Reserve GPU (to claim a slot)
            if not await self.gpu_manager.reserve_gpu(gpu.gpu_id, user_id):
                 raise HTTPException(status_code=503, detail="Failed to reserve GPU slot")

            # 2.1 Start GPU if needed
            if gpu_result.requires_gpu_startup:
                logger.info(f"Passthrough selected paused GPU, starting {gpu.gpu_id}...")
                try:
                    if not await self.gpu_manager.start_gpu(gpu.gpu_id):
                        # Use 500 explicitly
                        raise HTTPException(status_code=500, detail="Failed to start GPU")
                except Exception:
                    # Clear reservation if startup fails
                    logger.warning(f"Startup failed for {gpu.gpu_id}, clearing reservation")
                    gpu.clear_reservation()
                    raise

            # Get request body if present
            body = None
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.json()
                except:
                    pass

            # 3. Mark as busy
            logger.debug(f"Starting passthrough request on GPU {gpu.gpu_id}")
            gpu.start_request(user_id)
            
            try:
                # Proxy the request
                import httpx
    
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.request(
                        method=request.method,
                        url=f"http://{gpu.ip_address}:11434/api/{path}",
                        json=body,
                        headers={"Content-Type": "application/json"} if body else None,
                    )
    
                    return JSONResponse(response.json() if response.content else {})
            finally:
                # 4. Release slot
                logger.debug(f"Finishing passthrough request on GPU {gpu.gpu_id}")
                gpu.finish_request()

        except Exception as e:
            logger.error(f"Error in passthrough for /api/{path}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
