
import asyncio
import os
import httpx
import pytest
from loguru import logger

from gpumanager.config.loader import ConfigLoader
from gpumanager.cloud.api import CloudAPI, CloudAPIError
from gpumanager.cloud.models import WorkspaceStatus

# Skip if credentials not present (e.g. CI)
if not os.path.exists(".env") and not os.environ.get("CLOUD_API_TOKEN"):
    pytest.skip("Skipping integration tests: No credentials found", allow_module_level=True)


class SSHTunnel:
    """Context manager for SSH Tunneling."""
    def __init__(self, remote_host: str, local_port: int = 11434, remote_port: int = 11434):
        self.remote_host = remote_host
        # The test currently doesn't strictly define the user. 
        # We'll use a pragmatic approach: try 'ubuntu' (default cloud) or strict via env.
        self.username = os.environ.get("SSH_USER", "ubuntu") 
        self.local_port = local_port
        self.remote_port = remote_port
        self.process = None

    async def __aenter__(self):
        logger.info(f"Establishing SSH tunnel: localhost:{self.local_port} -> {self.remote_host}:{self.remote_port}")
        # Command: ssh -N -L local:localhost:remote <target>
        # Note: 'localhost' in the middle refers to the target's view of itself
        
        # We assume the user running the test has SSH access.
        cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-N",
            "-L", f"{self.local_port}:localhost:{self.remote_port}",
            f"{self.username}@{self.remote_host}"
        ]
        
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Give it a moment to establish
        await asyncio.sleep(2)
        
        # Check if it died immediately
        if self.process.returncode is not None:
            stdout, stderr = await self.process.communicate()
            raise RuntimeError(f"SSH tunnel failed to start: {stderr.decode()}")
            
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            logger.info("Tearing down SSH tunnel")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()

from gpumanager.deployment import DeploymentManager

@pytest.mark.asyncio
async def test_full_lifecycle():
    """
    Test the full lifecycle of a GPU node:
    1. Discover
    2. Resume & Deploy (Ensure Service is Up)
    3. Query Ollama (Resource Check) via SSH Tunnel
    4. Pause
    """
    # 1. Setup
    config = ConfigLoader.load_config()
    api = CloudAPI(config.cloud_api)
    manager = DeploymentManager(api)
    
    # Define test user - hardcoded for this environment or from env
    test_user = os.environ.get("SSH_USER")
    
    logger.info("Step 1: Discovering Workspaces")
    workspaces = await api.discover_gpu_workspaces()
    assert len(workspaces) > 0, "No GPU workspaces found to test with"
    
    # Pick a target. Configurable via env, default to 'LMSTUDIO-ollama4'
    target_name = os.environ.get("TEST_TARGET_WORKSPACE", "LMSTUDIO-ollama4")
    logger.info(f"Targeting workspace: {target_name}")
    target = next((ws for ws in workspaces if ws.name == target_name), None)
            
    if not target:
        # If specific target not found, maybe fallback or fail?
        # User wants flexibility, so better to warn and pick first GPU one, or fail if strict.
        # Let's fail if the specific target is requested but missing, unless it's the default.
        if target_name != "LMSTUDIO-ollama4":
             pytest.fail(f"Requested target {target_name} not found.")
        else:
             logger.warning(f"{target_name} not found. Using first available.")
             target = workspaces[0]

    logger.info(f"Selected target: {target.name} ({target.id}) - Status: {target.status}")

    # 2. Resume & Deploy
    # This ensures the VM is running AND the service is installed/started
    logger.info("Step 2: Processing Workspace (Resume + Deploy)")
    await manager.process_workspace(target, test_user)
    
    # Refresh target details to get IP
    target = await api.get_workspace(target.id)

    # Ensure we have an IP
    ip = target.resource_meta.ip
    assert ip, "Workspace has no IP address even after being RUNNING"
    logger.info(f"Target IP: {ip}")

    # 3. Query Ollama via Direct Connection (Port 11434 - User Opened)
    logger.info("Step 3: Querying Ollama Endpoint (Port 11434)")
    ollama_url = f"http://{ip}:11434"
    
    max_retries = 10
    retry_delay = 5
    
    async with httpx.AsyncClient() as client:
        # 3a. Check Version/Root
        connected = False
        for i in range(max_retries):
            try:
                resp = await client.get(ollama_url, timeout=5.0)
                if resp.status_code == 200:
                    logger.info("Ollama service is reachable!")
                    connected = True
                    break
            except Exception as e:
                logger.debug(f"Connection attempt {i+1} failed: {e}")
                await asyncio.sleep(retry_delay)
        
        assert connected, "Could not connect to Ollama service after multiple retries"

        # 3b. List Models
        resp = await client.get(f"{ollama_url}/api/tags")
        assert resp.status_code == 200
        models = resp.json()
        logger.info(f"Available models: {[m['name'] for m in models.get('models', [])]}")
        
        # 3c. Generate (Optional - simple test)
        # Only if models exist
        if models.get('models'):
            model_name = models['models'][0]['name']
            logger.info(f"Step 3c: Generating text with {model_name}")
            generate_payload = {
                "model": model_name,
                "prompt": "Say hello!",
                "stream": False
            }
            resp = await client.post(f"{ollama_url}/api/generate", json=generate_payload, timeout=60.0)
            assert resp.status_code == 200
            result = resp.json()
            logger.info(f"Response: {result.get('response')}")
            assert "response" in result

    # 4. Pause
    logger.info("Step 4: Pausing Workspace")
    await api.pause_workspace(target.id)
    
    # Wait for PAUSED (optional, or just verify action triggered)
    # Testing wait logic again
    success = await api.wait_for_workspace_status(
        target.id, 
        WorkspaceStatus.PAUSED, 
        timeout_seconds=60
    )
    assert success, "Failed to pause workspace"
    logger.success("Lifecycle test completed successfully!")

if __name__ == "__main__":
    # Allow running directly
    asyncio.run(test_full_lifecycle())
