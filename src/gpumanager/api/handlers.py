"""FastAPI request handlers."""

from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status, Path, Depends
from pydantic import BaseModel

from loguru import logger

from gpumanager.cloud.api import CloudAPI, CloudAPIError
from gpumanager.cloud.models import WorkspaceStatus
from gpumanager.auth.manager import APIKeyManager
from gpumanager.api.middleware import (
    create_auth_dependency,
    create_optional_auth_dependency,
)


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

    def __init__(self, cloud_api: CloudAPI, api_key_manager: APIKeyManager):
        """Initialize request handler."""
        self.cloud_api = cloud_api
        self.api_key_manager = api_key_manager

        # Create auth dependencies
        self.get_current_user = create_auth_dependency(api_key_manager)
        self.get_optional_user = create_optional_auth_dependency(api_key_manager)

        self.app = self._create_app()

        logger.info("Initialized RequestHandler with authentication")

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="LLM GPU Controller",
            description="GPU management API for LLM inference",
            version="0.1.0",
        )

        # Register routes with authentication
        app.get("/health", response_model=HealthResponse)(self.health_check)

        # Protected routes (require authentication)
        app.get("/gpu/discover", dependencies=[Depends(self.get_current_user)])(
            self.discover_gpus
        )
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

        return app

    async def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy", service="llm-gpu-controller")

    async def discover_gpus(self) -> Dict[str, Any]:
        """Discover available GPU workspaces."""
        try:
            workspaces = await self.cloud_api.discover_gpu_workspaces()

            gpu_info = []
            for workspace in workspaces:
                gpu_info.append(
                    {
                        "id": workspace.id,
                        "name": workspace.name,
                        "status": workspace.status.value,
                        "ip_address": workspace.ip_address,
                        "can_resume": workspace.can_resume,
                        "can_pause": workspace.can_pause,
                        "flavor": workspace.resource_meta.flavor_name,
                    }
                )

            return {"discovered_gpus": len(gpu_info), "gpus": gpu_info}

        except CloudAPIError as e:
            logger.error(f"Failed to discover GPUs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to discover GPUs: {str(e)}",
            )

    async def get_gpu_status(
        self, gpu_id: str = Path(..., description="GPU workspace ID")
    ) -> GPUStatusResponse:
        """Get current GPU status."""
        try:
            workspace = await self.cloud_api.get_workspace(gpu_id)

            return GPUStatusResponse(
                gpu_id=workspace.id,
                status=workspace.status.value,
                ip_address=workspace.ip_address,
                can_resume=workspace.can_resume,
                can_pause=workspace.can_pause,
            )

        except CloudAPIError as e:
            logger.error(f"Failed to get GPU status for {gpu_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get GPU status: {str(e)}",
            )

    async def resume_gpu(
        self, gpu_id: str = Path(..., description="GPU workspace ID")
    ) -> ActionResponse:
        """Resume the GPU workspace."""
        try:
            # Check current status
            workspace = await self.cloud_api.get_workspace(gpu_id)

            if workspace.status == WorkspaceStatus.RUNNING:
                return ActionResponse(success=True, message="GPU is already running")

            if not workspace.can_resume:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="GPU cannot be resumed in current state",
                )

            # Initiate resume
            action_response = await self.cloud_api.resume_workspace(gpu_id)

            return ActionResponse(
                success=True,
                message="GPU resume initiated",
                action_id=action_response.id,
            )

        except CloudAPIError as e:
            logger.error(f"Failed to resume GPU {gpu_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to resume GPU: {str(e)}",
            )

    async def pause_gpu(
        self, gpu_id: str = Path(..., description="GPU workspace ID")
    ) -> ActionResponse:
        """Pause the GPU workspace."""
        try:
            # Check current status
            workspace = await self.cloud_api.get_workspace(gpu_id)

            if workspace.status == WorkspaceStatus.PAUSED:
                return ActionResponse(success=True, message="GPU is already paused")

            if not workspace.can_pause:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="GPU cannot be paused in current state",
                )

            # Initiate pause
            action_response = await self.cloud_api.pause_workspace(gpu_id)

            return ActionResponse(
                success=True,
                message="GPU pause initiated",
                action_id=action_response.id,
            )

        except CloudAPIError as e:
            logger.error(f"Failed to pause GPU {gpu_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to pause GPU: {str(e)}",
            )
