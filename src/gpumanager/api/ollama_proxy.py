"""Ollama proxy for intelligent request routing."""

from typing import Optional, Dict, Any
import httpx
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from loguru import logger

from gpumanager.gpu.manager import GPUManager
from gpumanager.gpu.models import GPUSelectionRequest
from gpumanager.gpu.state import GPUModelStatus, ModelInfo
from gpumanager.auth.models import AuthenticatedUser
from .ollama_models import (
    OllamaGenerateRequest,
    OllamaChatRequest,
    OpenAIChatRequest,
    OllamaMessage,
)


class OllamaProxy:
    """Intelligent Ollama proxy with GPU management."""

    def __init__(self, gpu_manager: GPUManager):
        """Initialize Ollama proxy."""
        self.gpu_manager = gpu_manager
        logger.info("Initialized OllamaProxy")

    async def generate(
        self, request: OllamaGenerateRequest, user: AuthenticatedUser
    ) -> StreamingResponse:
        """Handle /api/generate requests with intelligent GPU routing."""
        try:
            # Select the best GPU for this request
            gpu_result = await self._select_and_prepare_gpu(
                model_name=request.model,
                user_id=user.name,
                context_length=self._extract_context_length(request.options),
            )

            if not gpu_result.gpu_info:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No GPUs available. Please try again later.",
                )

            # Mark GPU as busy
            gpu_result.gpu_info.start_request(user.name)

            try:
                # Proxy the request to the selected GPU
                response = await self._proxy_generate_request(
                    gpu_ip=gpu_result.gpu_info.ip_address, request=request
                )

                if request.stream:
                    # Return streaming response
                    return StreamingResponse(
                        self._stream_response(response, gpu_result.gpu_info),
                        media_type="application/json",
                    )
                else:
                    # Return complete response
                    full_response = await self._get_complete_response(response)
                    gpu_result.gpu_info.finish_request()
                    return full_response

            except Exception:
                # Mark GPU as available again on error
                gpu_result.gpu_info.finish_request()
                raise

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in generate request: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}",
            )

    async def chat(
        self, request: OllamaChatRequest, user: AuthenticatedUser
    ) -> StreamingResponse:
        """Handle /api/chat requests with intelligent GPU routing."""
        try:
            # Select the best GPU for this request
            gpu_result = await self._select_and_prepare_gpu(
                model_name=request.model,
                user_id=user.name,
                context_length=self._extract_context_length(request.options),
            )

            if not gpu_result.gpu_info:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No GPUs available. Please try again later.",
                )

            # Mark GPU as busy
            gpu_result.gpu_info.start_request(user.name)

            try:
                # Proxy the request to the selected GPU
                response = await self._proxy_chat_request(
                    gpu_ip=gpu_result.gpu_info.ip_address, request=request
                )

                if request.stream:
                    return StreamingResponse(
                        self._stream_response(response, gpu_result.gpu_info),
                        media_type="application/json",
                    )
                else:
                    full_response = await self._get_complete_response(response)
                    gpu_result.gpu_info.finish_request()
                    return full_response

            except Exception:
                gpu_result.gpu_info.finish_request()
                raise

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in chat request: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}",
            )

    async def openai_chat_completions(
        self, request: OpenAIChatRequest, user: AuthenticatedUser
    ) -> StreamingResponse:
        """Handle OpenAI-compatible /v1/chat/completions requests."""
        try:
            # Convert OpenAI request to Ollama format
            ollama_request = self._convert_openai_to_ollama_chat(request)

            # Use the existing chat handler
            return await self.chat(ollama_request, user)

        except Exception as e:
            logger.error(f"Error in OpenAI chat completions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}",
            )

    async def _select_and_prepare_gpu(
        self, model_name: str, user_id: str, context_length: Optional[int] = None
    ) -> Any:
        """Select and prepare the best GPU for the request."""
        selection_request = GPUSelectionRequest(
            user_id=user_id, model_name=model_name, context_length=context_length
        )

        gpu_result = await self.gpu_manager.select_gpu(selection_request)

        if not gpu_result.gpu_info:
            return gpu_result

        gpu = gpu_result.gpu_info

        # Start GPU if needed
        if gpu_result.requires_gpu_startup:
            logger.info(f"Starting GPU {gpu.gpu_id} for user {user_id}")
            success = await self.gpu_manager.start_gpu(gpu.gpu_id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to start GPU",
                )

        # Reserve the GPU
        await self.gpu_manager.reserve_gpu(gpu.gpu_id, user_id, model_name)

        # Load model if needed
        if gpu_result.requires_model_load:
            logger.info(f"Loading model {model_name} on GPU {gpu.gpu_id}")
            await self._ensure_model_loaded(gpu.ip_address, model_name, context_length)

            # Update GPU state
            model_info = ModelInfo(name=model_name, context_length=context_length)
            gpu.update_model(model_info)
            gpu.update_status(GPUModelStatus.MODEL_READY)

        return gpu_result

    async def _ensure_model_loaded(
        self, gpu_ip: str, model_name: str, context_length: Optional[int] = None
    ) -> None:
        """Ensure model is loaded on the GPU."""
        # Make a simple generation request to trigger model loading
        load_request = {
            "model": model_name,
            "prompt": "test",
            "stream": False,
            "options": {},
        }

        if context_length:
            load_request["options"]["num_ctx"] = context_length

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"http://{gpu_ip}:11434/api/generate", json=load_request
                )
                if response.status_code != 200:
                    logger.warning(
                        f"Model loading returned status {response.status_code}"
                    )
                else:
                    logger.success(
                        f"Model {model_name} loaded successfully on {gpu_ip}"
                    )
            except Exception as e:
                logger.error(f"Failed to load model {model_name} on {gpu_ip}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to load model: {e}",
                )

    async def _proxy_generate_request(
        self, gpu_ip: str, request: OllamaGenerateRequest
    ) -> httpx.Response:
        """Proxy generate request to GPU."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://{gpu_ip}:11434/api/generate",
                json=request.model_dump(),
                headers={"Content-Type": "application/json"},
            )
            return response

    async def _proxy_chat_request(
        self, gpu_ip: str, request: OllamaChatRequest
    ) -> httpx.Response:
        """Proxy chat request to GPU."""
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"http://{gpu_ip}:11434/api/chat",
                json=request.model_dump(),
                headers={"Content-Type": "application/json"},
            )
            return response

    async def _stream_response(self, response: httpx.Response, gpu_info) -> Any:
        """Stream response from GPU and mark as finished when done."""
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            # Mark GPU as available when streaming is complete
            gpu_info.finish_request()

    async def _get_complete_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Get complete non-streaming response."""
        return response.json()

    def _extract_context_length(
        self, options: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        """Extract context length from options."""
        if not options:
            return None
        return options.get("num_ctx")

    def _convert_openai_to_ollama_chat(
        self, openai_request: OpenAIChatRequest
    ) -> OllamaChatRequest:
        """Convert OpenAI chat request to Ollama format."""
        # Convert messages
        ollama_messages = []
        for msg in openai_request.messages:
            ollama_messages.append(OllamaMessage(role=msg.role, content=msg.content))

        # Convert options
        options = {}
        if openai_request.temperature is not None:
            options["temperature"] = openai_request.temperature
        if openai_request.top_p is not None:
            options["top_p"] = openai_request.top_p
        if openai_request.max_tokens is not None:
            options["num_ctx"] = openai_request.max_tokens

        return OllamaChatRequest(
            model=openai_request.model,
            messages=ollama_messages,
            options=options if options else None,
            stream=openai_request.stream,
        )
