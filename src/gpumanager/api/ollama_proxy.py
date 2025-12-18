"""Ollama proxy for intelligent request routing."""

from typing import Optional, Dict, Any
import asyncio
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
        # Track active requests per user to prevent concurrent requests from same user
        self.active_user_requests: Dict[str, asyncio.Lock] = {}
        self.user_request_timeout = 120  # 2 minutes timeout for queued requests
        logger.info("Initialized OllamaProxy")

    async def _acquire_user_lock(self, user_id: str) -> asyncio.Lock:
        """Acquire a lock for a user to prevent concurrent requests.

        If the user already has an active request, this will wait (queue) with a timeout.
        If the wait exceeds the timeout, raises an HTTPException.
        """
        # Get or create lock for this user
        if user_id not in self.active_user_requests:
            self.active_user_requests[user_id] = asyncio.Lock()

        user_lock = self.active_user_requests[user_id]

        # Try to acquire lock with timeout
        try:
            logger.debug(f"User {user_id} attempting to acquire request lock...")
            acquired = await asyncio.wait_for(
                user_lock.acquire(),
                timeout=self.user_request_timeout
            )
            if acquired or user_lock.locked():
                logger.info(f"User {user_id} acquired request lock")
                return user_lock
        except asyncio.TimeoutError:
            logger.warning(f"User {user_id} request timed out waiting for previous request (waited {self.user_request_timeout}s)")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Your previous request is still processing. Please wait for it to complete before sending a new request. (Timeout: {self.user_request_timeout}s)"
            )

    async def generate(
        self, request: OllamaGenerateRequest, user: AuthenticatedUser
    ) -> StreamingResponse:
        """Handle /api/generate requests with intelligent GPU routing."""
        # Acquire per-user lock to prevent concurrent requests from same user
        user_lock = await self._acquire_user_lock(user.name)

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
        finally:
            # Release user lock
            if user_lock.locked():
                user_lock.release()
                logger.debug(f"User {user.name} released request lock")

    async def chat(
        self, request: OllamaChatRequest, user: AuthenticatedUser
    ) -> StreamingResponse:
        """Handle /api/chat requests with intelligent GPU routing."""
        # Acquire per-user lock to prevent concurrent requests from same user
        user_lock = await self._acquire_user_lock(user.name)

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
        finally:
            # Release user lock
            if user_lock.locked():
                user_lock.release()
                logger.debug(f"User {user.name} released request lock")

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

        # Retry loop for race conditions
        max_retries = 3
        for attempt in range(max_retries):
            selection_request = GPUSelectionRequest(
                user_id=user_id, model_name=model_name, context_length=context_length
            )

            gpu_result = await self.gpu_manager.select_gpu(selection_request)

            if not gpu_result.gpu_info:
                return gpu_result

            gpu = gpu_result.gpu_info

            # Start GPU if needed
            if gpu_result.requires_gpu_startup:
                logger.info(f"Starting GPU {gpu.name} for user {user_id}")
                success = await self.gpu_manager.start_gpu(gpu.gpu_id)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Failed to start GPU",
                    )

            # If GPU is STARTING, wait for it to become ready before reserving
            # This handles the case where another request just started the GPU
            if gpu.status == GPUModelStatus.STARTING:
                logger.info(f"GPU {gpu.name} is starting, waiting for it to become ready...")
                timeout = self.gpu_manager.timing_config.startup_timeout_seconds
                start_time = asyncio.get_event_loop().time()

                while (asyncio.get_event_loop().time() - start_time) < timeout:
                    if gpu.status in [GPUModelStatus.IDLE, GPUModelStatus.MODEL_READY, GPUModelStatus.BUSY]:
                        logger.info(f"GPU {gpu.name} is now ready (status: {gpu.status})")
                        break
                    await asyncio.sleep(2)  # Check every 2 seconds
                else:
                    logger.error(f"GPU {gpu.name} did not become ready within {timeout}s")
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"GPU startup timeout after {timeout}s",
                    )

            # Reserve the GPU (this might fail if someone else took the slot)
            reservation_success = await self.gpu_manager.reserve_gpu(gpu.gpu_id, user_id, model_name)

            if reservation_success:
                # Load model if needed
                if gpu_result.requires_model_load:
                    logger.info(f"Loading model {model_name} on GPU {gpu.name}")
                    try:
                        await self._ensure_model_loaded(gpu.ip_address, gpu.name, model_name, context_length)

                        # Update GPU state
                        model_info = ModelInfo(name=model_name, context_length=context_length)
                        gpu.update_model(model_info)
                        gpu.update_status(GPUModelStatus.MODEL_READY)
                    except Exception as e:
                        logger.error(f"Failed to load model on {gpu.name}. Marking node as ERROR state.")
                        gpu.update_status(GPUModelStatus.ERROR)
                        # We should also clear the reservation so it doesn't expire naturally,
                        # but status=ERROR already prevents selection.
                        # Re-raise to stop the request
                        raise

                return gpu_result

            # If reservation failed, retry
            logger.warning(f"Reservation failed for GPU {gpu.name}, retrying selection (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(0.5)  # Small backoff

        # If we get here, we failed to get a reservation after retries
        return gpu_result  # Return the last result (which might be failure or success but reservation failed)

    async def _ensure_model_loaded(
        self, gpu_ip: str, gpu_name: str, model_name: str, context_length: Optional[int] = None
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

        logger.info(f"Sending model load request to {gpu_name} ({gpu_ip}): {load_request}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"http://{gpu_ip}:11434/api/generate", json=load_request
                )
                if response.status_code != 200:
                    error_text = response.text
                    logger.warning(
                        f"Model loading on {gpu_name} ({gpu_ip}) returned status {response.status_code}: {error_text}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to load model on node {gpu_name} ({gpu_ip}): Status {response.status_code} - {error_text}",
                    )
                else:
                    logger.success(
                        f"Model {model_name} loaded successfully on {gpu_name} ({gpu_ip})"
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to load model {model_name} on {gpu_name} ({gpu_ip}): {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to load model on {gpu_name}: {e}",
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
