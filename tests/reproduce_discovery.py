import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from gpumanager.api.ollama_proxy import OllamaProxy
from gpumanager.gpu.manager import GPUManager
from gpumanager.gpu.state import GPUInfo, GPUModelStatus

class TestModelDiscovery(unittest.IsolatedAsyncioTestCase):
    async def test_list_models_aggregation(self):
        # Mock GPU Manager
        mock_gpu_manager = MagicMock(spec=GPUManager)
        
        # Create 2 fake GPUs
        gpu1 = GPUInfo(
            gpu_id="gpu1", name="GPU-1", ip_address="10.0.0.1", 
            flavor="gpu-small", status=GPUModelStatus.IDLE
        )
        gpu2 = GPUInfo(
            gpu_id="gpu2", name="GPU-2", ip_address="10.0.0.2", 
            flavor="gpu-small", status=GPUModelStatus.IDLE
        )
        
        mock_gpu_manager.gpus = {
            "gpu1": gpu1,
            "gpu2": gpu2
        }
        
        # Initialize Proxy
        proxy = OllamaProxy(mock_gpu_manager)
        
        # Mock httpx response
        async def mock_get(url, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            
            if "10.0.0.1" in url:
                mock_response.json.return_value = {
                    "models": [
                        {"name": "llama3:8b", "model": "llama3:8b", "modified_at": "2024-01-01T00:00:00Z", "size": 100, "digest": "sha1", "details": {}},
                        {"name": "common-model", "model": "common", "modified_at": "2024-01-01T00:00:00Z", "size": 100, "digest": "sha2", "details": {}}
                    ]
                }
            elif "10.0.0.2" in url:
                mock_response.json.return_value = {
                    "models": [
                        {"name": "mistral:7b", "model": "mistral:7b", "modified_at": "2024-01-01T00:00:00Z", "size": 100, "digest": "sha3", "details": {}},
                        {"name": "common-model", "model": "common", "modified_at": "2024-01-01T00:00:00Z", "size": 100, "digest": "sha2", "details": {}}
                    ]
                }
            return mock_response

        # Patch httpx
        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = await proxy.list_models()
            
            print(f"Found {len(response.models)} models")
            model_names = [m.name for m in response.models]
            print(f"Models: {model_names}")
            
            # Verify aggregation
            self.assertIn("llama3:8b", model_names)
            self.assertIn("mistral:7b", model_names)
            self.assertIn("common-model", model_names)
            
            # Verify deduplication (common-model should appear once)
            self.assertEqual(len(model_names), 3)

if __name__ == "__main__":
    unittest.main()
