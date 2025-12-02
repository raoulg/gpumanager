import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from gpumanager.gpu.manager import GPUManager
from gpumanager.gpu.state import GPUInfo, GPUModelStatus, ModelInfo
from gpumanager.gpu.models import GPUSelectionRequest
from gpumanager.config.models import TimingConfig

@pytest.fixture
def mock_cloud_api():
    return AsyncMock()

@pytest.fixture
def timing_config():
    return TimingConfig()

@pytest.fixture
def gpu_manager(mock_cloud_api, timing_config):
    manager = GPUManager(mock_cloud_api, timing_config)
    
    # Setup some dummy GPUs
    manager.gpus = {
        "gpu1": GPUInfo(
            gpu_id="gpu1",
            name="GPU 1",
            ip_address="10.0.0.1",
            flavor="gpu-small",
            status=GPUModelStatus.IDLE,
            max_slots=2
        ),
        "gpu2": GPUInfo(
            gpu_id="gpu2",
            name="GPU 2",
            ip_address="10.0.0.2",
            flavor="gpu-large",
            status=GPUModelStatus.PAUSED,
            max_slots=1
        )
    }
    return manager

@pytest.mark.asyncio
async def test_select_gpu_idle(gpu_manager):
    request = GPUSelectionRequest(user_id="user1", model_name="llama3")
    result = await gpu_manager.select_gpu(request)
    
    assert result.gpu_info.gpu_id == "gpu1"
    assert result.requires_model_load is True
    assert result.requires_gpu_startup is False

@pytest.mark.asyncio
async def test_select_gpu_with_model(gpu_manager):
    # Setup GPU1 with model loaded
    gpu1 = gpu_manager.gpus["gpu1"]
    gpu1.update_status(GPUModelStatus.MODEL_READY)
    gpu1.update_model(ModelInfo(name="llama3"))
    
    request = GPUSelectionRequest(user_id="user1", model_name="llama3")
    result = await gpu_manager.select_gpu(request)
    
    assert result.gpu_info.gpu_id == "gpu1"
    assert result.requires_model_load is False

@pytest.mark.asyncio
async def test_slot_management(gpu_manager):
    gpu1 = gpu_manager.gpus["gpu1"]
    gpu1.update_status(GPUModelStatus.MODEL_READY)
    gpu1.update_model(ModelInfo(name="llama3"))
    
    # Fill first slot
    gpu1.start_request("user1")
    assert gpu1.active_requests == 1
    assert gpu1.is_available() is True # Max slots is 2
    
    # Select again - should still pick gpu1
    request = GPUSelectionRequest(user_id="user2", model_name="llama3")
    result = await gpu_manager.select_gpu(request)
    assert result.gpu_info.gpu_id == "gpu1"
    
    # Fill second slot
    gpu1.start_request("user2")
    assert gpu1.active_requests == 2
    assert gpu1.is_available() is False
    
    # Select again - should NOT pick gpu1 (slots full)
    # Should pick gpu2 (which is paused) or none if we didn't have gpu2
    result = await gpu_manager.select_gpu(request)
    assert result.gpu_info.gpu_id == "gpu2"
    assert result.requires_gpu_startup is True

@pytest.mark.asyncio
async def test_reserve_gpu(gpu_manager):
    success = await gpu_manager.reserve_gpu("gpu1", "user1", "llama3")
    assert success is True
    assert gpu_manager.gpus["gpu1"].reservation.user_id == "user1"
    
    # Try to reserve again - should fail (already reserved, even if slots available, reservation locks it for that user momentarily)
    # Wait, reservation logic:
    # is_available() checks reservation. If reserved, it returns False.
    # So reserve_gpu calls is_available(), which returns False.
    success = await gpu_manager.reserve_gpu("gpu1", "user2", "llama3")
    assert success is False
