"""SURF Cloud API client."""

import asyncio
from typing import List, Optional, Dict, Any

import httpx
from loguru import logger

from gpumanager.config.models import CloudAPIConfig
from gpumanager.cloud.models import (
    Workspace,
    WorkspaceListResponse,
    ActionResponse,
    WorkspaceStatus,
)


class CloudAPIError(Exception):
    """Cloud API related errors."""

    pass


class CloudAPI:
    """SURF Cloud API client."""

    def __init__(self, config: CloudAPIConfig):
        """Initialize cloud API client."""
        self.config = config
        self.base_url = config.base_url.rstrip("/")

        # Setup HTTP headers
        self.headers = {
            "accept": "application/json;Compute",
            "authorization": config.auth_token,
            "Content-Type": "application/json",
        }

        if config.csrf_token:
            self.headers["X-CSRFTOKEN"] = config.csrf_token

        logger.info(f"Initialized CloudAPI with base URL: {self.base_url}")

    async def _make_request(
        self, method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to cloud API."""
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=json_data,
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"API request failed: {method} {url} -> {response.status_code}: {response.text}"
                    )
                    raise CloudAPIError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                return response.json()

        except httpx.RequestError as e:
            logger.error(f"Network error during API request: {e}")
            raise CloudAPIError(f"Network error: {e}")

    async def list_workspaces(self) -> List[Workspace]:
        """List all workspaces matching the name filter."""
        endpoint = "/workspace/workspaces/"
        params = {
            "application_type": "Compute",
            "deleted": "false",
            "name": self.config.machine_name_filter,
        }

        # Build query string manually to match the curl format
        query_params = "&".join([f"{k}={v}" for k, v in params.items()])
        full_endpoint = f"{endpoint}?{query_params}"

        logger.info(
            f"Listing workspaces with filter: {self.config.machine_name_filter}"
        )

        response_data = await self._make_request("GET", full_endpoint)
        workspace_list = WorkspaceListResponse(**response_data)

        logger.info(f"Found {len(workspace_list.results)} workspaces")
        return workspace_list.results

    async def get_workspace(self, workspace_id: str, name: Optional[str] = None) -> Workspace:
        """Get specific workspace details."""
        endpoint = f"/workspace/workspaces/{workspace_id}/"

        log_name = f"{name} ({workspace_id})" if name else workspace_id
        logger.info(f"Getting workspace details: {log_name}")

        response_data = await self._make_request("GET", endpoint)
        workspace = Workspace(**response_data)

        logger.info(f"Workspace {log_name} status: {workspace.status}")
        return workspace

    async def resume_workspace(self, workspace_id: str, name: Optional[str] = None) -> ActionResponse:
        """Resume a paused workspace."""
        endpoint = f"/workspace/workspaces/{workspace_id}/actions/resume/"

        log_name = f"{name} ({workspace_id})" if name else workspace_id
        logger.info(f"Resuming workspace: {log_name}")

        response_data = await self._make_request("POST", endpoint, json_data={})
        action_response = ActionResponse(**response_data)

        logger.info(f"Resume action initiated for {log_name}: {action_response.id}")
        return action_response

    async def pause_workspace(self, workspace_id: str, name: Optional[str] = None) -> ActionResponse:
        """Pause an active workspace."""
        endpoint = f"/workspace/workspaces/{workspace_id}/actions/pause/"

        log_name = f"{name} ({workspace_id})" if name else workspace_id
        logger.info(f"Pausing workspace: {log_name}")

        response_data = await self._make_request("POST", endpoint, json_data={})
        action_response = ActionResponse(**response_data)

        logger.info(f"Pause action initiated for {log_name}: {action_response.id}")
        return action_response

    async def wait_for_workspace_status(
        self,
        workspace_id: str,
        target_status: WorkspaceStatus,
        timeout_seconds: int = 120,
        poll_interval: int = 10,
        name: Optional[str] = None,
    ) -> bool:
        """Wait for workspace to reach target status."""
        log_name = f"{name} ({workspace_id})" if name else workspace_id
        logger.info(
            f"Waiting for workspace {log_name} to reach status {target_status}"
        )

        elapsed = 0
        while elapsed < timeout_seconds:
            workspace = await self.get_workspace(workspace_id, name)

            if workspace.status == target_status:
                logger.success(
                    f"Workspace {log_name} reached status {target_status}"
                )
                return True

            if workspace.status == WorkspaceStatus.UNKNOWN:
                logger.warning(f"Workspace {log_name} in unknown status")

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            logger.debug(
                f"Workspace {log_name} status: {workspace.status} (elapsed: {elapsed}s)"
            )

        logger.error(
            f"Timeout waiting for workspace {log_name} to reach status {target_status}"
        )
        return False

    async def update_nsgs(self, workspace_id: str, custom_rules: List[str]) -> ActionResponse:
        """Update Network Security Groups for a workspace.
        
        Note: SURF Research Cloud has mandatory immutable rules. This method handles
        appending the 'immutable' suffix to mandatory rules and 'mutable' to custom rules.
        """
        endpoint = f"/workspace/workspaces/{workspace_id}/actions/update_nsgs/"
        
        # Mandatory rules provided by SURF documentation/support
        mandatory_rules = [
            "in tcp 443 443 0.0.0.0/0 immutable",
            "in tcp 80 80 0.0.0.0/0 immutable",
            "in tcp 1 65535 10.10.10.0/24 immutable",
            "in tcp 3389 3389 0.0.0.0/0 immutable",
            "out tcp 1 65535 10.10.10.0/24 immutable",
            "out tcp 1 65535 0.0.0.0/0 immutable",
            "in tcp 22 22 0.0.0.0/0 immutable",
            "in udp 1 65535 10.10.10.0/24 immutable",
            "out udp 1 65535 10.10.10.0/24 immutable",
            "out udp 1 65535 0.0.0.0/0 immutable",
        ]

        # Format custom rules with 'mutable' suffix
        formatted_custom_rules = [f"{rule} mutable" for rule in custom_rules]
        
        full_rules = mandatory_rules + formatted_custom_rules
        
        payload = {
            "network_security_group_rules": full_rules
        }

        logger.info(f"Updating NSGs for workspace {workspace_id} with {len(formatted_custom_rules)} custom rules")
        
        response_data = await self._make_request("POST", endpoint, json_data=payload)
        action_response = ActionResponse(**response_data)
        
        return action_response

    async def discover_gpu_workspaces(self) -> List[Workspace]:
        """Discover all GPU workspaces for management."""
        workspaces = await self.list_workspaces()

        # Filter for GPU workspaces (those with GPU flavors)
        gpu_workspaces = []
        for workspace in workspaces:
            if "gpu" in workspace.resource_meta.flavor_name.lower():
                gpu_workspaces.append(workspace)

        logger.info(f"Discovered {len(gpu_workspaces)} GPU workspaces")
        return gpu_workspaces
