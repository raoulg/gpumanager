
import asyncio
import re
from pathlib import Path
from typing import List, Optional

from loguru import logger

from gpumanager.deployment import DeploymentManager
from gpumanager.cloud.api import CloudAPI
from gpumanager.config.loader import ConfigLoader

class ModelSynchronizer:
    """Synchronizes models from a source node to the local configuration and other nodes."""

    def __init__(self, deployment_manager: DeploymentManager):
        self.deployment_manager = deployment_manager

    async def get_models_from_node(self, ip: str) -> List[str]:
        """Fetch list of installed models from a node."""
        logger.info(f"Fetching models from source node {ip}...")
        
        # Run ollama list
        # output format:
        # NAME                            ID              SIZE      MODIFIED
        # llama3.1:8b-instruct-q8_0      8874ee2b8a3e    8.0 GB    2 hours ago
        
        # We need a robust way to get just the names.
        # We can use `ollama list` and parse the first column.
        # But `ollama list` creates a table.
        # Better: run `ollama list` and parse.
        
        # We must use the deployment manager's safe remote execution
        cmd = "ollama list"
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", ip, cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            logger.error(f"Failed to list models on {ip}: {stderr.decode()}")
            raise Exception("Failed to list models")

        lines = stdout.decode().splitlines()
        models = []
        # Skip header
        if len(lines) > 0 and "NAME" in lines[0]:
            lines = lines[1:]
            
        for line in lines:
            parts = line.split()
            if parts:
                model_name = parts[0]
                # Filter out 'latest' tag if redundant? No, keep exact names.
                # However, ollama sometimes shows 'latest' implicitly.
                models.append(model_name)
                
        return models

    def update_local_env(self, models: List[str]):
        """Update the local .env file with the model list."""
        env_path = Path("gpu-node/.env")
        model_str = ",".join(models)
        
        logger.info(f"Updating {env_path} with models: {model_str}")
        
        content = ""
        if env_path.exists():
            content = env_path.read_text()
            
        # Replace or append OLLAMA_MODELS
        if "OLLAMA_MODELS=" in content:
            content = re.sub(r"OLLAMA_MODELS=.*", f"OLLAMA_MODELS=\"{model_str}\"", content)
        else:
            content += f"\nOLLAMA_MODELS=\"{model_str}\"\n"
            
        env_path.write_text(content)

    async def sync_and_deploy(self, source_ip: str, username: str, ips_file: Optional[str] = None):
        """Main sync workflow."""
        try:
            # 1. Fetch models
            models = await self.get_models_from_node(source_ip)
            if not models:
                logger.warning("No models found on source node. Aborting sync.")
                return

            logger.info(f"Found {len(models)} models: {models}")

            # 2. Update local config
            self.update_local_env(models)
            
            # 3. Deploy to all
            logger.info("Triggering deployment to propagate changes...")
            await self.deployment_manager.deploy_all(username, ips_file)
            
            logger.success("Sync and deployment complete!")
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            raise
