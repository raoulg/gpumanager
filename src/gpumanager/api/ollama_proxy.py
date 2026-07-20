"""Ollama proxy for intelligent request routing."""

from typing import Optional, Dict, Any, List
import asyncio
import json
from datetime import datetime
import httpx
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from loguru import logger

from gpumanager.gpu.manager import GPUManager
from gpumanager.gpu.models import GPUSelectionRequest
from gpumanager.gpu.state import GPUModelStatus, ModelInfo, GPUInfo
from gpumanager.auth.models import AuthenticatedUser
from .ollama_models import (
    OllamaGenerateRequest,
    OllamaChatRequest,
    OllamaMessage,
    OpenAIChatRequest,
    OllamaListResponse,
    OllamaModelResponse,
    OllamaPullRequest,
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

    async def list_models(self) -> OllamaListResponse:
        """List models aggregated from all available GPUs."""
        all_models: Dict[str, OllamaModelResponse] = {}

        gpu_manager = self.gpu_manager
        active_gpus = []
        paused_gpus = []

        for gpu in gpu_manager.gpus.values():
            if gpu.status not in [GPUModelStatus.ERROR, GPUModelStatus.PAUSED, GPUModelStatus.STARTING]:
                if gpu.ip_address:
                    active_gpus.append(gpu)
            elif gpu.status == GPUModelStatus.PAUSED:
                paused_gpus.append(gpu)

        # Auto-wake logic: If no active GPUs, wake one up and WAIT for it
        if not active_gpus and paused_gpus:
            # Pick the most recently used one if possible
            target_gpu = sorted(paused_gpus, key=lambda g: g.last_request or datetime.min, reverse=True)[0]
            logger.info(f"No active GPUs found for list_models. Auto-waking {target_gpu.name} and waiting...")

            # Start GPU and wait for it to become ready
            success = await gpu_manager.start_gpu(target_gpu.gpu_id)

            if success:
                # Add to active GPUs so we can fetch its models
                active_gpus.append(target_gpu)
                logger.info(f"GPU {target_gpu.name} is now ready, will fetch models from it")
            else:
                logger.error(f"Failed to wake up {target_gpu.name}, returning empty model list")
                return OllamaListResponse(models=[])

        logger.info(f"Aggregating models from {len(active_gpus)} active GPUs...")
        async def fetch_models(gpu_ip: str, gpu_name: str) -> List[OllamaModelResponse]:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"http://{gpu_ip}:11434/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = []
                        for m in data.get("models", []):
                            try:
                                models.append(OllamaModelResponse(**m))
                            except Exception as e:
                                logger.warning(f"Failed to parse model from {gpu_name}: {e}")
                        return models
            except Exception as e:
                logger.warning(f"Failed to fetch models from {gpu_name}: {e}")
            return []

        # Fetch from all active GPUs concurrently
        tasks = [fetch_models(gpu.ip_address, gpu.name) for gpu in active_gpus]
        results = await asyncio.gather(*tasks)

        # Aggregate results (deduplicate by name)
        for gpu_models in results:
            for model in gpu_models:
                if model.name not in all_models:
                    all_models[model.name] = model

        logger.info(f"Found {len(all_models)} unique models across {len(active_gpus)} GPUs")
        return OllamaListResponse(models=list(all_models.values()))

    async def pull_model(
        self, request: OllamaPullRequest, current_user: Optional[Any] = None
    ) -> StreamingResponse:
        """
        Pull a model on all available GPUs.

        Strategically:
        1. Access checks (optional)
        2. Identify ALL reachable GPUs (IDLE, READY, BUSY, etc.) using /api/tags check
        3. Pick ONE "primary" GPU to stream the response from (to show progress to user)
        4. Trigger background pulls on ALL OTHERS (fire and forget / independent tasks)
        5. Return stream from Primary
        """
        model_name = request.name
        user_id = getattr(current_user, "name", "anonymous")
        logger.info(f"User {user_id} requested pull for model: {model_name}")

        # 1. Identify all reachable GPUs (including PAUSED/STARTING)
        reachable_gpus: List[GPUInfo] = []
        for gpu in self.gpu_manager.gpus.values():
            if gpu.status != GPUModelStatus.ERROR:
                reachable_gpus.append(gpu)

        if not reachable_gpus:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No GPUs available to pull model",
            )

        logger.info(f"Broadcasting pull to {len(reachable_gpus)} GPUs: {[g.name for g in reachable_gpus]}")

        # 2. Select Primary GPU
        # Prefer IDLE/READY/BUSY over PAUSED/STARTING
        reachable_gpus.sort(key=lambda g: 1 if g.status in [GPUModelStatus.PAUSED, GPUModelStatus.STARTING] else 0)
        
        primary_gpu = reachable_gpus[0]
        secondary_gpus = reachable_gpus[1:]

        # 3. Trigger background pulls on secondary GPUs
        for gpu in secondary_gpus:
            asyncio.create_task(self._trigger_background_pull(gpu, request, user_id))

        # 4. Stream from Primary GPU
        return await self._stream_pull_from_gpu(primary_gpu, request, user_id)

    async def _trigger_background_pull(
        self, gpu: GPUInfo, request: OllamaPullRequest, user_id: str
    ):
        """Execute a pull on a secondary GPU without waiting for stream."""
        try:
            # Wake up GPU if needed
            if gpu.status == GPUModelStatus.PAUSED:
                logger.info(f"Background pull: Waking up {gpu.name}...")
                await self.gpu_manager.start_gpu(gpu.gpu_id)
            
            # Wait if starting
            if gpu.status == GPUModelStatus.STARTING:
                logger.info(f"Background pull: Waiting for {gpu.name} startup...")
                # Simple wait loop
                for _ in range(60): # 2 minutes max
                    if gpu.status not in [GPUModelStatus.PAUSED, GPUModelStatus.STARTING]:
                        break
                    await asyncio.sleep(2)
            
            if gpu.status not in [GPUModelStatus.IDLE, GPUModelStatus.MODEL_READY, GPUModelStatus.BUSY]:
                logger.warning(f"Background pull aborted for {gpu.name}: Status is {gpu.status}")
                return

            logger.info(f"Starting background pull of {request.name} on {gpu.name}")
            async with httpx.AsyncClient(timeout=3600.0) as client:
                async with client.stream(
                    "POST",
                    f"http://{gpu.ip_address}:11434/api/pull",
                    json=request.model_dump(),
                ) as response:
                    async for _ in response.aiter_bytes():
                        pass 
            
            logger.info(f"Background pull of {request.name} on {gpu.name} COMPLETED")
            
        except Exception as e:
            logger.error(f"Background pull of {request.name} on {gpu.name} FAILED: {e}")

    async def _stream_pull_from_gpu(
        self, gpu: GPUInfo, request: OllamaPullRequest, user_id: str
    ) -> StreamingResponse:
        """Stream the pull response from the primary GPU to the client."""
        logger.info(f"Streaming pull of {request.name} from PRIMARY {gpu.name}")
        
        async def stream_generator():
            try:
                # Wake up logic with status updates
                if gpu.status == GPUModelStatus.PAUSED:
                    yield json.dumps({"status": f"Waking up GPU node {gpu.name}..."}).encode("utf-8") + b"\n"
                    # Start GPU (this blocks until ready in current implementation)
                    await self.gpu_manager.start_gpu(gpu.gpu_id)

                # If still starting (or if start_gpu returned early due to race), wait
                while gpu.status == GPUModelStatus.STARTING:
                     yield json.dumps({"status": f"Waiting for GPU node {gpu.name} startup..."}).encode("utf-8") + b"\n"
                     await asyncio.sleep(2)
                     if gpu.status == GPUModelStatus.PAUSED: # Failed to start?
                         yield json.dumps({"status": f"GPU node {gpu.name} failed to start. Aborting."}).encode("utf-8") + b"\n"
                         return

                if gpu.status not in [GPUModelStatus.IDLE, GPUModelStatus.MODEL_READY, GPUModelStatus.BUSY]:
                     yield json.dumps({"status": f"GPU node {gpu.name} unavailable (Status: {gpu.status}). Aborting."}).encode("utf-8") + b"\n"
                     return

                # Proceed with pull
                async with httpx.AsyncClient(timeout=3600.0) as client:
                    async with client.stream(
                        "POST",
                        f"http://{gpu.ip_address}:11434/api/pull",
                        json=request.model_dump(),
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                            
                logger.info(f"Primary pull stream of {request.name} COMPLETED")
                
            except Exception as e:
                logger.error(f"Primary pull stream of {request.name} FAILED: {e}")
                raise

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"
        )

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
                    except HTTPException as e:
                        # Check if it's a 404 (model not found) - don't mark GPU as ERROR
                        if e.status_code == status.HTTP_404_NOT_FOUND:
                            logger.warning(f"Model '{model_name}' not found on {gpu.name}. GPU is healthy, model needs to be pulled.")
                            # Clear reservation and re-raise for user
                            gpu.clear_reservation()
                            raise
                        else:
                            # Other errors indicate GPU problems
                            logger.error(f"Failed to load model on {gpu.name}. Marking node as ERROR state.")
                            gpu.update_status(GPUModelStatus.ERROR)
                            raise
                    except Exception as e:
                        # Unexpected errors also indicate GPU problems
                        logger.error(f"Unexpected error loading model on {gpu.name}. Marking node as ERROR state: {e}")
                        gpu.update_status(GPUModelStatus.ERROR)
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
                if response.status_code == 404:
                    # Model not found - this is a user error, not a GPU error
                    error_text = response.text
                    logger.warning(
                        f"Model '{model_name}' not found on {gpu_name}. User should pull it first."
                    )
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model '{model_name}' not found on GPU. Please pull the model first using /api/pull",
                    )
                elif response.status_code != 200:
                    # Other errors (500, 503, etc.) indicate GPU problems
                    error_text = response.text
                    logger.error(
                        f"Model loading on {gpu_name} ({gpu_ip}) failed with status {response.status_code}: {error_text}"
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
                # Network/connection errors indicate GPU problems
                logger.error(f"Failed to connect to {gpu_name} ({gpu_ip}): {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to connect to GPU {gpu_name}: {e}",
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
