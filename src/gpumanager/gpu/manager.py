"""GPU Manager for intelligent GPU and model management."""

import asyncio
from typing import Dict, Optional, Set
from collections import defaultdict

from loguru import logger

from gpumanager.cloud.api import CloudAPI, CloudAPIError
from gpumanager.cloud.models import WorkspaceStatus
from gpumanager.config.models import TimingConfig
from .state import GPUInfo, GPUModelStatus
from .models import (
    GPUSelectionRequest,
    GPUSelectionResult,
    GPUManagerStats,
)


class GPUManager:
    """Manages GPU lifecycle, model loading, and intelligent request routing."""

    def __init__(self, cloud_api: CloudAPI, timing_config: TimingConfig):
        """Initialize GPU manager."""
        self.cloud_api = cloud_api
        self.timing_config = timing_config

        # GPU state tracking
        self.gpus: Dict[str, GPUInfo] = {}

        # Background task management
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown = False

        logger.info("Initialized GPUManager")

    async def initialize(self) -> None:
        """Initialize GPU manager by discovering available GPUs."""
        try:
            # Discover all GPU workspaces
            workspaces = await self.cloud_api.discover_gpu_workspaces()

            for workspace in workspaces:
                gpu_info = GPUInfo(
                    gpu_id=workspace.id,
                    name=workspace.name,
                    ip_address=workspace.ip_address,
                    flavor=workspace.resource_meta.flavor_name,
                    status=self._map_workspace_status(workspace.status),
                )

                self.gpus[workspace.id] = gpu_info
                logger.info(
                    f"Discovered GPU: {gpu_info.name} ({gpu_info.gpu_id}) - {gpu_info.status}"
                )

            # Start background tasks
            await self._start_background_tasks()

            logger.success(f"GPU Manager initialized with {len(self.gpus)} GPUs")

        except Exception as e:
            logger.error(f"Failed to initialize GPU manager: {e}")
            raise

    def _map_workspace_status(
        self, workspace_status: WorkspaceStatus
    ) -> GPUModelStatus:
        """Map cloud workspace status to our GPU model status."""
        mapping = {
            WorkspaceStatus.RUNNING: GPUModelStatus.IDLE,
            WorkspaceStatus.PAUSED: GPUModelStatus.PAUSED,
            WorkspaceStatus.RESUMING: GPUModelStatus.STARTING,
            WorkspaceStatus.PAUSING: GPUModelStatus.PAUSING,
        }
        return mapping.get(workspace_status, GPUModelStatus.ERROR)

    async def select_gpu(self, request: GPUSelectionRequest) -> GPUSelectionResult:
        """Select the best GPU for a request."""
        logger.debug(
            f"Selecting GPU for user {request.user_id}, model {request.model_name}"
        )

        # 1. Check for GPU with model already loaded AND available slots
        gpu_with_model = self._find_gpu_with_model(request.model_name)
        if gpu_with_model and gpu_with_model.is_available():
            logger.debug(f"Found GPU with model loaded: {gpu_with_model.gpu_id}")
            return GPUSelectionResult(
                gpu_info=gpu_with_model,
                estimated_wait_seconds=0,
                requires_model_load=False,
                requires_gpu_startup=False,
                message=f"GPU ready with {request.model_name} loaded",
            )

        # 2. Check for available idle GPU (no model loaded)
        idle_gpu = self._find_available_gpu()
        if idle_gpu:
            logger.debug(f"Found available GPU: {idle_gpu.gpu_id}")
            return GPUSelectionResult(
                gpu_info=idle_gpu,
                estimated_wait_seconds=30,  # Estimate for model loading
                requires_model_load=True,
                requires_gpu_startup=False,
                message=f"GPU available, will load {request.model_name}",
            )

        # 3. Check for paused GPU that can be started
        paused_gpu = self._find_paused_gpu()
        if paused_gpu:
            logger.debug(f"Found paused GPU to wake up: {paused_gpu.gpu_id}")
            return GPUSelectionResult(
                gpu_info=paused_gpu,
                estimated_wait_seconds=self.timing_config.startup_timeout_seconds + 30,
                requires_model_load=True,
                requires_gpu_startup=True,
                message=f"Will start GPU and load {request.model_name}",
            )

        # 4. No GPUs available
        logger.warning("No GPUs available for request")
        return GPUSelectionResult(
            gpu_info=None,
            estimated_wait_seconds=-1,
            requires_model_load=False,
            requires_gpu_startup=False,
            message="All GPUs are busy, please try again later",
        )

    def _find_gpu_with_model(self, model_name: str) -> Optional[GPUInfo]:
        """Find a GPU that already has the model loaded."""
        # Sort by active requests (least busy first)
        candidates = []
        for gpu in self.gpus.values():
            if gpu.has_model_loaded(model_name) and gpu.is_available():
                candidates.append(gpu)
            elif gpu.has_model_loaded(model_name):
                logger.debug(
                    f"GPU {gpu.gpu_id} has model {model_name} but is not available. "
                    f"Status: {gpu.status}, Active: {gpu.active_requests}/{gpu.max_slots}, "
                    f"Reserved: {gpu.reservation is not None}"
                )
        
        if not candidates:
            logger.debug(f"No GPUs found with model {model_name} loaded")
            return None
            
        # Return the one with fewest active requests
        # Log all candidates
        logger.debug(f"Found {len(candidates)} GPUs with model {model_name}: {[g.gpu_id for g in candidates]}")
        selected = sorted(candidates, key=lambda g: g.active_requests)[0]
        logger.debug(f"Selected GPU with model: {selected.gpu_id}")
        return selected

    def _find_available_gpu(self) -> Optional[GPUInfo]:
        """Find an available GPU (prioritizing idle ones)."""
        # Pass 1: Check for IDLE GPUs (Preferred)
        idle_candidates = []
        for gpu in self.gpus.values():
            if gpu.status == GPUModelStatus.IDLE:
                if gpu.is_available():
                    idle_candidates.append(gpu)
                else:
                    logger.debug(
                        f"Skipping IDLE GPU {gpu.gpu_id}: "
                        f"Active: {gpu.active_requests}/{gpu.max_slots}, "
                        f"Reserved: {gpu.reservation if gpu.reservation else 'No'}"
                    )
        
        if idle_candidates:
            logger.debug(f"Found {len(idle_candidates)} IDLE and available GPUs: {[g.gpu_id for g in idle_candidates]}")
            # For now, just pick the first one, but logging gives us visibility
            # Use sort to ensure deterministic behavior? Or just rely on list order.
            # Iteration order of dict values is insertion order.
            selected = idle_candidates[0]
            logger.debug(f"Selected IDLE GPU: {selected.gpu_id}")
            return selected
                
        # Pass 2: Check for MODEL_READY GPUs (Fallback)
        # We allow reusing these for generic requests to avoid waking up a paused GPU
        ready_candidates = []
        for gpu in self.gpus.values():
            if gpu.status == GPUModelStatus.MODEL_READY:
                if gpu.is_available():
                    ready_candidates.append(gpu)
                else:
                    logger.debug(
                        f"Skipping MODEL_READY GPU {gpu.gpu_id}: consumed by {gpu.loaded_model.name if gpu.loaded_model else 'Unknown'}. "
                        f"Active: {gpu.active_requests}/{gpu.max_slots}, "
                        f"Reserved: {gpu.reservation if gpu.reservation else 'No'}"
                    )
        
        if ready_candidates:
             logger.debug(f"Found {len(ready_candidates)} MODEL_READY and available GPUs (for reuse): {[g.gpu_id for g in ready_candidates]}")
             selected = ready_candidates[0]
             logger.debug(f"Selected MODEL_READY GPU: {selected.gpu_id}")
             return selected
             
        logger.debug("No available IDLE or MODEL_READY GPUs found in _find_available_gpu")
        return None

    def _find_paused_gpu(self) -> Optional[GPUInfo]:
        """Find a paused GPU that can be started."""
        for gpu in self.gpus.values():
            if gpu.status == GPUModelStatus.PAUSED:
                return gpu
        return None

    async def start_gpu(self, gpu_id: str) -> bool:
        """Start a paused GPU."""
        if gpu_id not in self.gpus:
            logger.error(f"GPU not found: {gpu_id}")
            return False

        gpu = self.gpus[gpu_id]
        if gpu.status != GPUModelStatus.PAUSED:
            logger.warning(f"GPU {gpu_id} is not paused, current status: {gpu.status}")
            return False

        try:
            # Update status to starting
            gpu.update_status(GPUModelStatus.STARTING)
            logger.info(f"Starting GPU: {gpu_id}")

            # Resume the workspace
            await self.cloud_api.resume_workspace(gpu_id)

            # Wait for GPU to be ready
            success = await self.cloud_api.wait_for_workspace_status(
                gpu_id,
                WorkspaceStatus.RUNNING,
                timeout_seconds=self.timing_config.startup_timeout_seconds,
            )

            if success:
                gpu.update_status(GPUModelStatus.IDLE)
                logger.success(f"GPU {gpu_id} started successfully")
                return True
            else:
                gpu.update_status(GPUModelStatus.ERROR)
                logger.error(f"GPU {gpu_id} failed to start within timeout")
                return False

        except CloudAPIError as e:
            gpu.update_status(GPUModelStatus.ERROR)
            logger.error(f"Failed to start GPU {gpu_id}: {e}")
            return False

    async def pause_gpu(self, gpu_id: str) -> bool:
        """Pause an idle GPU."""
        if gpu_id not in self.gpus:
            logger.error(f"GPU not found: {gpu_id}")
            return False

        gpu = self.gpus[gpu_id]
        
        # Don't pause if there are active requests
        if gpu.active_requests > 0:
            logger.warning(f"GPU {gpu_id} has active requests, cannot pause")
            return False

        if gpu.status not in [GPUModelStatus.IDLE, GPUModelStatus.MODEL_READY]:
            logger.warning(
                f"GPU {gpu_id} cannot be paused, current status: {gpu.status}"
            )
            return False

        try:
            # Update status to pausing
            gpu.update_status(GPUModelStatus.PAUSING)
            gpu.update_model(None)  # Clear loaded model
            logger.info(f"Pausing GPU: {gpu_id}")

            # Pause the workspace
            await self.cloud_api.pause_workspace(gpu_id)

            # Update status immediately (don't wait for confirmation)
            gpu.update_status(GPUModelStatus.PAUSED)
            logger.success(f"GPU {gpu_id} paused successfully")
            return True

        except CloudAPIError as e:
            gpu.update_status(GPUModelStatus.ERROR)
            logger.error(f"Failed to pause GPU {gpu_id}: {e}")
            return False

    async def reserve_gpu(
        self, gpu_id: str, user_id: str, model_name: Optional[str] = None
    ) -> bool:
        """Reserve a GPU for a user."""
        gpu = self.gpus[gpu_id]
        
        # Check if safe to reserve
        # We allow reserving if:
        # 1. No existing reservation
        # 2. Status is OK (Available OR Paused/Starting/Pausing)
        # Note: We bypass strict is_available() because we want to allow reserving Paused GPUs to wake them up
        if gpu.reservation is not None:
             # Already reserved
             return False
             
        # If it's active, check slots
        if gpu.status in [GPUModelStatus.IDLE, GPUModelStatus.MODEL_READY, GPUModelStatus.BUSY]:
            if not gpu.is_available(): # Checks slots + reservation
                return False
                
        # If Paused/Starting, we can reserve (exclusive lock effectively due to reservation check above)
        # Proceed to set reservation

        # Set reservation
        gpu.set_reservation(
            user_id=user_id,
            duration_minutes=self.timing_config.reservation_minutes,
            model_name=model_name,
        )

        logger.debug(f"Reserved GPU {gpu_id} for user {user_id}")
        return True

    def get_gpu_stats(self) -> GPUManagerStats:
        """Get current GPU manager statistics."""
        total_gpus = len(self.gpus)
        active_gpus = sum(
            1
            for gpu in self.gpus.values()
            if gpu.status not in [GPUModelStatus.PAUSED, GPUModelStatus.ERROR]
        )
        busy_gpus = sum(
            1 for gpu in self.gpus.values() if gpu.status == GPUModelStatus.BUSY
        )
        paused_gpus = sum(
            1 for gpu in self.gpus.values() if gpu.status == GPUModelStatus.PAUSED
        )

        # Count loaded models
        models_loaded = defaultdict(int)
        for gpu in self.gpus.values():
            if gpu.loaded_model:
                models_loaded[gpu.loaded_model.name] += 1

        total_requests_today = sum(gpu.requests_today for gpu in self.gpus.values())

        return GPUManagerStats(
            total_gpus=total_gpus,
            active_gpus=active_gpus,
            busy_gpus=busy_gpus,
            paused_gpus=paused_gpus,
            models_loaded=dict(models_loaded),
            total_requests_today=total_requests_today,
        )

    async def _start_background_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Task to monitor idle GPUs and pause them
        idle_monitor_task = asyncio.create_task(self._idle_monitor_loop())
        self._background_tasks.add(idle_monitor_task)
        idle_monitor_task.add_done_callback(self._background_tasks.discard)

        # Task to clean up expired reservations
        reservation_cleanup_task = asyncio.create_task(self._reservation_cleanup_loop())
        self._background_tasks.add(reservation_cleanup_task)
        reservation_cleanup_task.add_done_callback(self._background_tasks.discard)

        logger.info("Started background monitoring tasks")

    async def _idle_monitor_loop(self) -> None:
        """Monitor GPUs and pause idle ones."""
        while not self._shutdown:
            try:
                for gpu in self.gpus.values():
                    if gpu.is_idle_too_long(self.timing_config.reservation_minutes):
                        logger.info(f"GPU {gpu.gpu_id} idle too long, pausing...")
                        await self.pause_gpu(gpu.gpu_id)

                # Check every minute
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Error in idle monitor loop: {e}")
                await asyncio.sleep(60)

    async def _reservation_cleanup_loop(self) -> None:
        """Clean up expired reservations."""
        while not self._shutdown:
            try:
                for gpu in self.gpus.values():
                    if gpu.reservation and gpu.reservation.is_expired():
                        logger.debug(
                            f"Clearing expired reservation on GPU {gpu.gpu_id}"
                        )
                        gpu.clear_reservation()

                # Check every 30 seconds
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in reservation cleanup loop: {e}")
                await asyncio.sleep(30)

    async def shutdown(self) -> None:
        """Shutdown GPU manager and background tasks."""
        logger.info("Shutting down GPU manager...")
        self._shutdown = True

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        logger.info("GPU manager shutdown complete")
