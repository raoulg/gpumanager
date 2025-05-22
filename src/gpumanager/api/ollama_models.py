"""Ollama API request/response models."""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class OllamaGenerateRequest(BaseModel):
    """Ollama /api/generate request model."""

    model: str = Field(description="Model name to use")
    prompt: str = Field(description="Text prompt to generate from")
    suffix: Optional[str] = Field(
        default=None, description="Text after the generated text"
    )
    images: Optional[List[str]] = Field(
        default=None, description="Base64 encoded images"
    )
    format: Optional[str] = Field(
        default=None, description="Response format (json, etc.)"
    )
    options: Optional[Dict[str, Any]] = Field(default=None, description="Model options")
    system: Optional[str] = Field(default=None, description="System prompt")
    template: Optional[str] = Field(default=None, description="Prompt template")
    context: Optional[List[int]] = Field(
        default=None, description="Context from previous conversation"
    )
    stream: bool = Field(default=True, description="Whether to stream the response")
    raw: bool = Field(default=False, description="Return raw response")
    keep_alive: Optional[Union[str, int]] = Field(
        default=None, description="How long to keep model loaded"
    )


class OllamaMessage(BaseModel):
    """Message in Ollama chat format."""

    role: str = Field(description="Message role (system, user, assistant)")
    content: str = Field(description="Message content")
    images: Optional[List[str]] = Field(
        default=None, description="Base64 encoded images"
    )


class OllamaChatRequest(BaseModel):
    """Ollama /api/chat request model."""

    model: str = Field(description="Model name to use")
    messages: List[OllamaMessage] = Field(description="Chat messages")
    format: Optional[str] = Field(default=None, description="Response format")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Model options")
    stream: bool = Field(default=True, description="Whether to stream the response")
    keep_alive: Optional[Union[str, int]] = Field(
        default=None, description="How long to keep model loaded"
    )


class OpenAIMessage(BaseModel):
    """OpenAI-compatible message format."""

    role: str = Field(description="Message role")
    content: str = Field(description="Message content")


class OpenAIChatRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request."""

    model: str = Field(description="Model name")
    messages: List[OpenAIMessage] = Field(description="Chat messages")
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(default=None, description="Top-p sampling")
    n: Optional[int] = Field(default=1, description="Number of responses")
    stream: bool = Field(default=False, description="Whether to stream response")
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Stop sequences"
    )
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens")
    presence_penalty: Optional[float] = Field(
        default=None, description="Presence penalty"
    )
    frequency_penalty: Optional[float] = Field(
        default=None, description="Frequency penalty"
    )
    user: Optional[str] = Field(default=None, description="User identifier")


class ModelOptions(BaseModel):
    """Model configuration options."""

    # Core options
    num_ctx: Optional[int] = Field(default=None, description="Context length")
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature"
    )
    top_k: Optional[int] = Field(default=None, description="Top-k sampling")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling")
    repeat_penalty: Optional[float] = Field(default=None, description="Repeat penalty")
    seed: Optional[int] = Field(default=None, description="Random seed")

    # Advanced options
    mirostat: Optional[int] = Field(default=None, description="Mirostat sampling")
    mirostat_eta: Optional[float] = Field(default=None, description="Mirostat eta")
    mirostat_tau: Optional[float] = Field(default=None, description="Mirostat tau")
    num_gqa: Optional[int] = Field(default=None, description="Number of GQA groups")
    num_gpu: Optional[int] = Field(default=None, description="Number of GPU layers")
    num_thread: Optional[int] = Field(default=None, description="Number of threads")
    repeat_last_n: Optional[int] = Field(
        default=None, description="Repeat last N tokens"
    )
    tfs_z: Optional[float] = Field(default=None, description="TFS Z parameter")

    class Config:
        extra = "allow"  # Allow additional options


class OllamaErrorResponse(BaseModel):
    """Ollama error response format."""

    error: str = Field(description="Error message")


class GPURoutingInfo(BaseModel):
    """Information about GPU routing decision."""

    selected_gpu_id: str = Field(description="ID of selected GPU")
    gpu_ip: str = Field(description="IP address of selected GPU")
    model_already_loaded: bool = Field(description="Whether model was already loaded")
    estimated_load_time: int = Field(
        description="Estimated time to load model (seconds)"
    )
    reasoning: str = Field(description="Why this GPU was selected")
