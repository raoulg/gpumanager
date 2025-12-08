"""Cloud API response models."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class WorkspaceStatus(str, Enum):
    """Workspace status enumeration."""

    RUNNING = "running"
    PAUSED = "paused"
    RESUMING = "resuming"
    PAUSING = "pausing"
    UPDATING = "updating"
    UNKNOWN = "unknown"


class WorkspaceAction(str, Enum):
    """Available workspace actions."""

    RESUME = "resume"
    PAUSE = "pause"
    REBOOT = "reboot"
    UPDATE_NSGS = "update_nsgs"
    UPDATE_STORAGES = "update_storages"


class ResourceMeta(BaseModel):
    """Resource metadata."""

    id: str
    ip: str
    vm_id: str
    workspace_fqdn: str
    flavor_name: str
    instance_user: str = "ubuntu"


class WorkspaceActionHistory(BaseModel):
    """Workspace action history item."""

    id: str
    type: str
    status: str
    reason: str
    time_created: datetime
    time_updated: datetime
    issuer_display_name: Optional[str] = Field(default=None)  # Allow None values

    model_config = ConfigDict(extra="allow")  # Allow extra fields from API response


class Workspace(BaseModel):
    """Workspace model matching SURF cloud API response."""

    id: str
    name: str
    description: str
    status: WorkspaceStatus
    active: bool
    actions: List[WorkspaceAction]
    allowed_actions: List[WorkspaceAction]
    resource_meta: ResourceMeta
    workspace_actions: List[WorkspaceActionHistory] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")  # Allow extra fields from API response

    @property
    def ip_address(self) -> str:
        """Get IP address."""
        return self.resource_meta.ip

    @property
    def can_resume(self) -> bool:
        """Check if workspace can be resumed."""
        return WorkspaceAction.RESUME in self.actions

    @property
    def can_pause(self) -> bool:
        """Check if workspace can be paused."""
        return WorkspaceAction.PAUSE in self.actions


class WorkspaceListResponse(BaseModel):
    """Response model for workspace list API."""

    count: int
    next: Optional[str]
    previous: Optional[str]
    results: List[Workspace]

    model_config = ConfigDict(extra="allow")  # Allow extra fields from API response


class ActionResponse(BaseModel):
    """Response model for workspace actions - returns the updated workspace object."""

    id: str
    name: str
    description: str
    status: WorkspaceStatus
    active: bool
    actions: List[WorkspaceAction]
    allowed_actions: List[WorkspaceAction]
    resource_meta: ResourceMeta
    workspace_actions: List[WorkspaceActionHistory] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")  # Allow extra fields from API response

    @property
    def ip_address(self) -> str:
        """Get IP address."""
        return self.resource_meta.ip

    @property
    def can_resume(self) -> bool:
        """Check if workspace can be resumed."""
        return WorkspaceAction.RESUME in self.actions

    @property
    def can_pause(self) -> bool:
        """Check if workspace can be paused."""
        return WorkspaceAction.PAUSE in self.actions
