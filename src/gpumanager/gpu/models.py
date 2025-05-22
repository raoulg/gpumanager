"""GPU management data models."""

from typing import Dict, Optional
from pydantic import BaseModel, Field

from .state import GPUInfo, ModelInfo


class GPUSelectionRequest(BaseModel):
    """Request for GPU selection."""

    user_id: str = Field(description="User making the request")
    model_name: str = Field(description="Requested model name")
    context_length: Optional[int] = Field(
        default=None, description="Required context length"
    )
    priority: int = Field(default=1, description="Request priority (1=normal, 2=high)")


class GPUSelectionResult(BaseModel):
    """Result of GPU selection."""

    gpu_info: Optional[GPUInfo] = Field(
        description="Selected GPU, None if none available"
    )
    estimated_wait_seconds: int = Field(default=0, description="Estimated wait time")
    requires_model_load: bool = Field(
        default=False, description="Whether model needs to be loaded"
    )
    requires_gpu_startup: bool = Field(
        default=False, description="Whether GPU needs to be started"
    )
    message: str = Field(description="Human-readable status message")


class GPUManagerStats(BaseModel):
    """Statistics from the GPU manager."""

    total_gpus: int = Field(description="Total number of GPUs")
    active_gpus: int = Field(description="Number of active GPUs")
    busy_gpus: int = Field(description="Number of busy GPUs")
    paused_gpus: int = Field(description="Number of paused GPUs")
    models_loaded: Dict[str, int] = Field(
        default_factory=dict, description="Models loaded across GPUs"
    )
    total_requests_today: int = Field(default=0, description="Total requests today")
    average_response_time: Optional[float] = Field(
        default=None, description="Average response time in seconds"
    )


class ModelLoadRequest(BaseModel):
    """Request to load a model on a GPU."""

    gpu_id: str = Field(description="Target GPU ID")
    model_name: str = Field(description="Model to load")
    context_length: Optional[int] = Field(
        default=None, description="Context length to set"
    )
    force_reload: bool = Field(
        default=False, description="Force reload even if already loaded"
    )


class ModelLoadResult(BaseModel):
    """Result of model loading operation."""

    success: bool = Field(description="Whether model loading succeeded")
    message: str = Field(description="Result message")
    model_info: Optional[ModelInfo] = Field(
        default=None, description="Loaded model information"
    )
    load_time_seconds: Optional[float] = Field(
        default=None, description="Time taken to load model"
    )
