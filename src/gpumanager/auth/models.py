"""Authentication models."""

from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field, ConfigDict


class UserInfo(BaseModel):
    """User information model."""

    name: str = Field(description="User's display name")
    email: str = Field(description="User's email address")
    created: str = Field(description="Creation date (YYYY-MM-DD format)")
    requests_today: int = Field(default=0, description="Number of requests made today")
    total_requests: int = Field(default=0, description="Total number of requests made")
    last_request: Optional[datetime] = Field(
        default=None, description="Timestamp of last request"
    )

    model_config = ConfigDict(extra="allow")  # Allow additional fields for future extensions


class APIKeysFile(BaseModel):
    """API keys file structure."""

    api_keys: Dict[str, UserInfo] = Field(
        default_factory=dict, description="Mapping of API keys to user information"
    )

    model_config = ConfigDict(extra="forbid")  # Don't allow extra fields at root level


class AuthenticatedUser(BaseModel):
    """Authenticated user context."""

    api_key: str = Field(description="The API key used for authentication")
    user_info: UserInfo = Field(description="User information")

    @property
    def name(self) -> str:
        """Get user's display name."""
        return self.user_info.name

    @property
    def email(self) -> str:
        """Get user's email."""
        return self.user_info.email
