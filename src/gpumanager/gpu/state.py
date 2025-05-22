"""GPU state management enums and models."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GPUModelStatus(str, Enum):
    """GPU and model status enumeration."""

    PAUSED = "paused"  # GPU is paused (no models loaded)
    STARTING = "starting"  # GPU is being resumed
    IDLE = "idle"  # GPU is active but no model loaded
    LOADING_MODEL = "loading_model"  # Model is being pulled/loaded
    MODEL_READY = "model_ready"  # Model loaded and ready for requests
    BUSY = "busy"  # Processing a request
    PAUSING = "pausing"  # GPU is being paused
    ERROR = "error"  # Error state (GPU or model issue)


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str = Field(description="Model name (e.g., 'llama3:70b')")
    size: Optional[str] = Field(default=None, description="Model size (e.g., '42 GB')")
    loaded_at: datetime = Field(
        default_factory=datetime.now, description="When model was loaded"
    )
    last_used: datetime = Field(
        default_factory=datetime.now, description="Last time model was used"
    )
    context_length: Optional[int] = Field(
        default=None, description="Model's context length (num_ctx)"
    )

    def update_last_used(self) -> None:
        """Update the last used timestamp."""
        self.last_used = datetime.now()


class GPUReservation(BaseModel):
    """GPU reservation for pending requests."""

    user_id: str = Field(description="User who reserved the GPU")
    reserved_at: datetime = Field(
        default_factory=datetime.now, description="When reservation was made"
    )
    expires_at: datetime = Field(description="When reservation expires")
    model_name: Optional[str] = Field(default=None, description="Requested model name")

    def is_expired(self) -> bool:
        """Check if reservation has expired."""
        return datetime.now() > self.expires_at


class GPUInfo(BaseModel):
    """Complete GPU information and state."""

    gpu_id: str = Field(description="GPU workspace ID")
    name: str = Field(description="GPU workspace name")
    ip_address: str = Field(description="GPU IP address")
    flavor: str = Field(description="GPU flavor (e.g., 'gpu-a10-11core-88gb-50gb-2tb')")

    # State management
    status: GPUModelStatus = Field(description="Current GPU and model status")
    loaded_model: Optional[ModelInfo] = Field(
        default=None, description="Currently loaded model"
    )
    reservation: Optional[GPUReservation] = Field(
        default=None, description="Current reservation"
    )

    # Timestamps
    last_state_change: datetime = Field(
        default_factory=datetime.now, description="Last status change"
    )
    last_request: Optional[datetime] = Field(
        default=None, description="Last request timestamp"
    )
    idle_since: Optional[datetime] = Field(
        default=None, description="When GPU became idle"
    )

    # Statistics
    total_requests: int = Field(default=0, description="Total requests processed")
    requests_today: int = Field(default=0, description="Requests processed today")

    def update_status(self, new_status: GPUModelStatus) -> None:
        """Update GPU status and timestamp."""
        if new_status != self.status:
            self.status = new_status
            self.last_state_change = datetime.now()

            # Update idle timestamp
            if new_status == GPUModelStatus.MODEL_READY:
                self.idle_since = datetime.now()
            elif new_status == GPUModelStatus.BUSY:
                self.idle_since = None

    def update_model(self, model_info: Optional[ModelInfo]) -> None:
        """Update loaded model information."""
        self.loaded_model = model_info
        if model_info:
            model_info.update_last_used()

    def start_request(self, user_id: str) -> None:
        """Mark GPU as busy with a new request."""
        self.update_status(GPUModelStatus.BUSY)
        self.last_request = datetime.now()
        self.total_requests += 1
        self.requests_today += 1

        if self.loaded_model:
            self.loaded_model.update_last_used()

    def finish_request(self) -> None:
        """Mark request as finished, return to ready state."""
        if self.loaded_model:
            self.update_status(GPUModelStatus.MODEL_READY)
        else:
            self.update_status(GPUModelStatus.IDLE)

        self.clear_reservation()

    def set_reservation(
        self, user_id: str, duration_minutes: int, model_name: Optional[str] = None
    ) -> None:
        """Set a reservation for this GPU."""
        expires_at = datetime.now().replace(microsecond=0) + timedelta(
            minutes=duration_minutes
        )

        self.reservation = GPUReservation(
            user_id=user_id, expires_at=expires_at, model_name=model_name
        )

    def clear_reservation(self) -> None:
        """Clear the current reservation."""
        self.reservation = None

    def is_available(self) -> bool:
        """Check if GPU is available for new requests."""
        # Clear expired reservations
        if self.reservation and self.reservation.is_expired():
            self.clear_reservation()

        # Available if ready and no reservation
        return (
            self.status in [GPUModelStatus.MODEL_READY, GPUModelStatus.IDLE]
            and self.reservation is None
        )

    def is_idle_too_long(self, idle_timeout_minutes: int) -> bool:
        """Check if GPU has been idle too long and should be paused."""
        if not self.idle_since or self.status != GPUModelStatus.MODEL_READY:
            return False

        idle_duration = datetime.now() - self.idle_since
        return idle_duration.total_seconds() > (idle_timeout_minutes * 60)

    def can_handle_model(self, model_name: str) -> bool:
        """Check if GPU can handle the requested model."""
        # For now, assume all GPUs can handle any model
        # TODO: Add model-specific requirements (memory, etc.)
        return True

    def has_model_loaded(self, model_name: str) -> bool:
        """Check if specific model is already loaded."""
        return (
            self.loaded_model is not None
            and self.loaded_model.name == model_name
            and self.status == GPUModelStatus.MODEL_READY
        )
