import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from gpumanager.api.ollama_proxy import OllamaProxy
from gpumanager.gpu.models import GPUSelectionResult
from gpumanager.gpu.state import GPUInfo, GPUModelStatus

@pytest.fixture
def mock_gpu_manager():
    manager = AsyncMock()
    return manager

@pytest.fixture
def proxy(mock_gpu_manager):
    return OllamaProxy(mock_gpu_manager)

@pytest.mark.asyncio
async def test_select_and_prepare_gpu_success(proxy, mock_gpu_manager):
    # Setup successful selection
    gpu_info = GPUInfo(
        gpu_id="gpu1", name="GPU 1", ip_address="1.2.3.4", flavor="test", status=GPUModelStatus.IDLE
    )
    result = GPUSelectionResult(
        gpu_info=gpu_info,
        estimated_wait_seconds=0,
        requires_model_load=True,
        requires_gpu_startup=False,
        message="Ready"
    )
    mock_gpu_manager.select_gpu.return_value = result
    mock_gpu_manager.reserve_gpu.return_value = True
    
    # Mock ensure_model_loaded
    with patch.object(proxy, '_ensure_model_loaded', new_callable=AsyncMock) as mock_load:
        final_result = await proxy._select_and_prepare_gpu("llama3", "user1")
        
        assert final_result == result
        mock_gpu_manager.select_gpu.assert_called()
        mock_gpu_manager.reserve_gpu.assert_called_with("gpu1", "user1", "llama3")
        mock_load.assert_called()

@pytest.mark.asyncio
async def test_select_and_prepare_gpu_retry(proxy, mock_gpu_manager):
    # Setup selection
    gpu_info = GPUInfo(
        gpu_id="gpu1", name="GPU 1", ip_address="1.2.3.4", flavor="test", status=GPUModelStatus.IDLE
    )
    result = GPUSelectionResult(
        gpu_info=gpu_info,
        estimated_wait_seconds=0,
        requires_model_load=True,
        requires_gpu_startup=False,
        message="Ready"
    )
    mock_gpu_manager.select_gpu.return_value = result
    
    # Fail reservation twice, then succeed
    mock_gpu_manager.reserve_gpu.side_effect = [False, False, True]
    
    with patch.object(proxy, '_ensure_model_loaded', new_callable=AsyncMock):
        final_result = await proxy._select_and_prepare_gpu("llama3", "user1")
        
        assert final_result == result
        assert mock_gpu_manager.select_gpu.call_count == 3
        assert mock_gpu_manager.reserve_gpu.call_count == 3

@pytest.mark.asyncio
async def test_select_and_prepare_gpu_fail_all_retries(proxy, mock_gpu_manager):
    # Setup selection
    gpu_info = GPUInfo(
        gpu_id="gpu1", name="GPU 1", ip_address="1.2.3.4", flavor="test", status=GPUModelStatus.IDLE
    )
    result = GPUSelectionResult(
        gpu_info=gpu_info,
        estimated_wait_seconds=0,
        requires_model_load=True,
        requires_gpu_startup=False,
        message="Ready"
    )
    mock_gpu_manager.select_gpu.return_value = result
    
    # Fail reservation always
    mock_gpu_manager.reserve_gpu.return_value = False
    
    final_result = await proxy._select_and_prepare_gpu("llama3", "user1")
    
    # Should return the result but without having reserved/loaded
    assert final_result == result
    assert mock_gpu_manager.select_gpu.call_count == 3
