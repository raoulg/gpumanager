"""Configuration models using Pydantic."""

from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from typing import Optional


class ServerConfig(BaseModel):
    """Server configuration."""

    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")


class CloudAPIConfig(BaseModel):
    """Cloud API configuration."""

    base_url: str = Field(description="SURF cloud API base URL")
    machine_name_filter: str = Field(description="Filter for machine names")
    auth_token: str = Field(description="Cloud API authentication token")
    csrf_token: Optional[str] = Field(
        default=None, description="CSRF token if required"
    )


class TimingConfig(BaseModel):
    """Timing configuration for GPU management."""

    reservation_minutes: int = Field(
        default=10, description="GPU reservation time in minutes"
    )
    fallback_reservation_minutes: int = Field(
        default=3, description="Fallback reservation time when all GPUs busy"
    )
    startup_timeout_seconds: int = Field(
        default=120, description="Timeout for GPU startup"
    )
    ollama_readiness_wait_seconds: int = Field(
        default=10, description="Wait time for Ollama to be ready"
    )


class PathsConfig(BaseModel):
    """File paths configuration."""

    api_keys_file: Path = Field(
        default=Path("api_keys.json"), description="Path to API keys file"
    )


class AppConfig(BaseModel):
    """Main application configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    cloud_api: CloudAPIConfig
    timing: TimingConfig = Field(default_factory=TimingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
