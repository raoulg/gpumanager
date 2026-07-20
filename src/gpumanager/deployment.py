
import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional

from loguru import logger

from gpumanager.config.loader import ConfigLoader
from gpumanager.cloud.api import CloudAPI
from gpumanager.cloud.models import Workspace, WorkspaceStatus


class DeploymentManager:
    """Manages the deployment of GPU node software to remote machines."""

    def __init__(self, cloud_api: Optional[CloudAPI] = None):
        """Initialize deployment manager."""
        self.cloud_api = cloud_api

    async def wait_for_ssh(self, ip: str, timeout: int = 60, interval: int = 5) -> bool:
        """Wait for SSH to be available."""
        elapsed = 0
        while elapsed < timeout:
            try:
                # Netcat check for port 22
                result = subprocess.run(
                    ["nc", "-z", "-w", str(interval), ip, "22"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                if result.returncode == 0:
                    return True
            except Exception:
                pass
            
            await asyncio.sleep(interval)
            elapsed += interval
        return False

    async def run_remote_command(self, ip: str, command: str, description: str = "", log_error: bool = True, log_name: Optional[str] = None) -> bool:
        """Run a remote command via SSH."""
        display = f"[{log_name}]" if log_name else f"[{ip}]"
        
        if description:
            logger.info(f"{display} {description}")
        
        ssh_cmd = [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            ip, command
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            if log_error:
                logger.error(f"{display} Command failed: {command}")
                logger.error(f"{display} Error: {stderr.decode().strip()}")
            return False
        
        return True

    async def check_remote_progress(self, ip: str, step_marker: str) -> bool:
        """Check if a specific step is marked as done in the remote progress file."""
        marker_file = "/srv/shared/.setup_progress"
        cmd = f"test -f {marker_file} && grep -q '{step_marker}' {marker_file}"
        return await self.run_remote_command(ip, cmd, log_error=False)

    async def mark_remote_progress(self, ip: str, step_marker: str):
        """Mark a step as done in the remote progress file."""
        cmd = f"echo '{step_marker}' | sudo tee -a /srv/shared/.setup_progress"
        await self.run_remote_command(ip, cmd, f"Marking step {step_marker} as complete")

    async def deploy_gpu_node(self, ip: str, workspace_name: str, username: str):
        """Deploy Ollama to a GPU node."""
        logger.info(f"Starting GPU node deployment for {workspace_name} ({ip})")

        # 1. Wait for SSH availability
        if not await self.wait_for_ssh(ip):
            logger.error(f"[{workspace_name}] SSH not available after timeout. Skipping.")
            return

        # 2. Setup Remote Environment
        REMOTE_DIR = "~/gpu-node-install"
        SETUP_CMD = f"sudo bash setup.sh --shared --user {username}"

        if not await self.run_remote_command(ip, f"mkdir -p {REMOTE_DIR}", "Creating remote directory", log_name=workspace_name):
            return

        # 3. Copy Files
        logger.info(f"[{workspace_name}] Copying installation files...")
        project_root = Path.cwd()
        gpu_node_dir = project_root / "gpu-node"

        if not gpu_node_dir.exists():
            logger.error(f"Could not find gpu-node directory at {gpu_node_dir}")
            return

        # GPU nodes only need Ollama files (no Caddy, no .env for passwords)
        files_to_copy = [
            "docker-compose.yml", "entrypoint.sh", "setup.sh", "install-docker.sh", ".env"
        ]

        scp_args = []
        for f in files_to_copy:
            file_path = gpu_node_dir / f
            if not file_path.exists():
                logger.error(f"Missing required file: {file_path}")
                return
            scp_args.append(str(file_path))

        scp_cmd = [
            "scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            *scp_args,
            f"{ip}:{REMOTE_DIR}/"
        ]

        proc = await asyncio.create_subprocess_exec(*scp_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
             logger.error(f"[{workspace_name}] SCP failed: {stderr.decode()}")
             return

        # 4. Run Setup
        logger.info(f"[{workspace_name}] Running setup script...")
        if await self.run_remote_command(ip, f"cd {REMOTE_DIR} && {SETUP_CMD}", log_name=workspace_name):
             await self.mark_remote_progress(ip, "SETUP_COMPLETED")
             logger.success(f"[{workspace_name}] Setup script completed successfully.")
        else:
             logger.error(f"[{workspace_name}] Setup script failed.")
             return

        # 5. Verify Ollama Service
        if await self.run_remote_command(ip, "curl -s localhost:11434 > /dev/null", "Verifying Ollama health", log_name=workspace_name):
            logger.success(f"[{workspace_name}] Ollama is UP and responding!")
        else:
            logger.warning(f"[{workspace_name}] Ollama does not seem to be responding on port 11434.")

        # Cleanup
        REMOTE_DIR = "~/gpu-node-install"
        await self.run_remote_command(ip, f"rm -rf {REMOTE_DIR}", "Cleaning up remote directory", log_name=workspace_name)

    async def deploy_manager_node(self, ip: str, manager_name: str, username: str, workspace_id: Optional[str] = None, with_api: bool = False):
        """Deploy OpenWebUI to the manager node."""
        logger.info(f"Starting manager node deployment for {manager_name} ({ip})")

        # 1. Wait for SSH availability
        if not await self.wait_for_ssh(ip):
            logger.error(f"[{manager_name}] SSH not available after timeout. Skipping.")
            return

        # 2. Open ports for WebUI (8080) and optionally API (8000)
        if workspace_id and self.cloud_api:
            try:
                ports_to_open = ["in tcp 8080 8080 0.0.0.0/0"]
                if with_api:
                    ports_to_open.append("in tcp 8000 8000 0.0.0.0/0")

                logger.info(f"[{manager_name}] Opening ports in firewall...")
                await self.cloud_api.add_nsg_rules(workspace_id, ports_to_open, name=manager_name)
            except Exception as e:
                logger.warning(f"[{manager_name}] Failed to open ports: {e}")

        # 3. Setup Remote Environment
        REMOTE_DIR = "~/manager-node-install"
        SETUP_CMD = f"sudo bash setup.sh --shared --user {username}"

        if not await self.run_remote_command(ip, f"mkdir -p {REMOTE_DIR}", "Creating remote directory", log_name=manager_name):
            return

        # 3. Copy Files
        logger.info(f"[{manager_name}] Copying installation files...")
        project_root = Path.cwd()
        manager_node_dir = project_root / "manager-node"

        if not manager_node_dir.exists():
            logger.error(f"Could not find manager-node directory at {manager_node_dir}")
            return

        files_to_copy = [
            "docker-compose.yml", "setup.sh", ".env", "Caddyfile"
        ]

        # If deploying with API, also copy source code and config
        if with_api:
            logger.info(f"[{manager_name}] Preparing GPU Manager API files...")

            # Additional files for API deployment
            files_to_copy.extend(["config.toml", "pyproject.toml", "README.md"])

            # Create a tarball of the source code
            import tarfile
            import tempfile

            src_dir = project_root / "src"
            if not src_dir.exists():
                logger.error(f"Could not find src directory at {src_dir}")
                return

            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                tar_path = tmp.name
                with tarfile.open(tar_path, 'w:gz') as tar:
                    tar.add(src_dir, arcname='src')

                files_to_copy.append(tar_path)

        scp_args = []
        for f in files_to_copy:
            if f.endswith('.tar.gz'):
                file_path = Path(f)
            else:
                file_path = manager_node_dir / f if f in ["docker-compose.yml", "setup.sh", ".env", "Caddyfile"] else project_root / f

            if not file_path.exists():
                logger.warning(f"Optional file not found: {file_path}, skipping")
                continue
            scp_args.append(str(file_path))

        if not scp_args:
            logger.error(f"[{manager_name}] No files to copy")
            return

        scp_cmd = [
            "scp", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            *scp_args,
            f"{ip}:{REMOTE_DIR}/"
        ]

        proc = await asyncio.create_subprocess_exec(*scp_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
             logger.error(f"[{manager_name}] SCP failed: {stderr.decode()}")
             return

        # Extract source code and prepare files if with_api
        if with_api:
            # Extract source code
            extract_cmd = f"cd {REMOTE_DIR} && tar -xzf *.tar.gz && rm *.tar.gz"
            await self.run_remote_command(ip, extract_cmd, "Extracting source code", log_name=manager_name)

            # Copy files that docker-compose will need to /srv/shared BEFORE running docker compose
            # This prevents Docker from creating directories instead of mounting files
            # Also copy src directory recursively
            copy_cmd = (
                f"cd {REMOTE_DIR} && "
                f"sudo cp config.toml pyproject.toml README.md .env /srv/shared/ 2>/dev/null || true && "
                f"sudo cp -r src /srv/shared/ 2>/dev/null || true"
            )
            await self.run_remote_command(ip, copy_cmd, "Copying API configuration and source files", log_name=manager_name)

            # Cleanup local tar file
            import os
            for f in files_to_copy:
                if f.endswith('.tar.gz'):
                    os.remove(f)

        # 4. Wipe OpenWebUI data (for clean deployment)
        wipe_cmd = (
            "cd /srv/shared && "
            "docker compose stop webui && "
            "docker compose rm -f webui && "
            "docker volume rm shared_open-webui_data || true"
        )
        logger.info(f"[{manager_name}] Wiping OpenWebUI data for fresh install...")
        await self.run_remote_command(ip, wipe_cmd, "Wiping OpenWebUI data", log_name=manager_name)

        # 5. Run Setup
        logger.info(f"[{manager_name}] Running setup script...")
        if await self.run_remote_command(ip, f"cd {REMOTE_DIR} && {SETUP_CMD}", log_name=manager_name):
             await self.mark_remote_progress(ip, "SETUP_COMPLETED")
             logger.success(f"[{manager_name}] Setup script completed successfully.")
        else:
             logger.error(f"[{manager_name}] Setup script failed.")
             return

        # 6. Provision Admin User
        await self.provision_admin_user(ip, "http", log_name=manager_name)

        # Cleanup
        await self.run_remote_command(ip, f"rm -rf {REMOTE_DIR}", "Cleaning up remote directory", log_name=manager_name)
        
    async def provision_admin_user(self, ip: str, scheme: str = "http", log_name: Optional[str] = None):
        """Provision the admin user if not exists."""
        from gpumanager.config.loader import ConfigLoader
        import httpx
        import os
        
        display = f"[{log_name}]" if log_name else f"[{ip}]"
        logger.info(f"{display} Checking Admin user status...")
        
        # Load credentials (we assume they are in local .env since we just deployed them)
        # Better: pass them in args, but lazy loading from file is okay for this CLI tool
        env_path = Path("gpu-node/.env")
        if not env_path.exists():
            logger.warning("No local .env found, skipping admin provision.")
            return

        import dotenv
        config = dotenv.dotenv_values(env_path)
        admin_pass = config.get("WEBUI_ADMIN_PASSWORD")
        gatekeeper_pass = config.get("GATEKEEPER_PASSWORD")
        
        if not admin_pass or not gatekeeper_pass:
            logger.warning("Missing admin or gatekeeper password in .env, skipping admin provision.")
            return

        # URL for signup
        url = f"{scheme}://{ip}:8080/api/v1/auths/signup"
        
        # Payload
        payload = {
            "name": "Admin",
            "email": "admin@gpumanager.local",
            "password": admin_pass,
            "profile_image_url": ""
        }
        
        import asyncio
        for attempt in range(24): # Try for 120 seconds (2 minutes)
            try:
               # Optional: log docker status for debugging if requested
               # if attempt == 0:
               #     await self.run_remote_command(ip, "docker ps | grep webui", "Checking container status", log_name=log_name)

               async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
                   response = await client.post(
                       url, 
                       json=payload,
                       auth=("gatekeeper", gatekeeper_pass) # Basic Auth for Caddy
                   )
                   
                   if response.status_code == 200:
                       logger.success(f"{display} Admin user created successfully.")
                       logger.info(f"{display} Credentials: admin@gpumanager.local / {admin_pass}")
                       break
                   elif response.status_code == 400 and "Email already registered" in response.text:
                       logger.info(f"{display} Admin user already exists.")
                       break
                   elif response.status_code == 502:
                       logger.warning(f"{display} OpenWebUI starting up (502), retrying in 5s... ({attempt+1}/24)")
                       await asyncio.sleep(5)
                       continue
                   else:
                       logger.warning(f"{display} Failed to create admin user: {response.status_code} {response.text}")
                       # If 403 or other perm error, it means app IS up but rejecting us. Breaking is correct.
                       # But for robustness we might want to log it clearly.
                       break
                       
            except Exception as e:
                logger.error(f"{display} Error provisioning admin: {e}")
                # Don't break on connection error immediately, might be transient
                await asyncio.sleep(5)

        # Cleanup
        REMOTE_DIR = "~/gpu-node-install"
        await self.run_remote_command(ip, f"rm -rf {REMOTE_DIR}", "Cleaning up remote directory", log_name=log_name)

    async def process_workspace(self, ws: Workspace, username: str):
        """Process a single workspace for deployment."""
        try:
            if not self.cloud_api:
                logger.error("Cloud API not initialized")
                return

            logger.info(f"Processing {ws.name} ({ws.status})")
            
            # Initial IP might be missing if paused, so use getattr or handle carefully
            # But ResourceMeta definition implies it is required.
            target_ip = ws.resource_meta.ip
            
            # Check status and resume if needed
            if ws.status != WorkspaceStatus.RUNNING:
                if ws.can_resume:
                     logger.info(f"Resuming {ws.name}...")
                     await self.cloud_api.resume_workspace(ws.id, name=ws.name)
                     if not await self.cloud_api.wait_for_workspace_status(ws.id, WorkspaceStatus.RUNNING, name=ws.name):
                         logger.error(f"Failed to resume {ws.name}. Skipping.")
                         return
                     
                     
                     # Fetch fresh details
                     ws = await self.cloud_api.get_workspace(ws.id, name=ws.name)
                     target_ip = ws.resource_meta.ip
                else:
                    logger.warning(f"Workspace {ws.name} is in state {ws.status} and cannot be resumed. Skipping.")
                    return

            if not target_ip:
                logger.error(f"Workspace {ws.name} has no IP address. Skipping.")
                return

            # Ensure Network Security Groups are configured (Open Port 11434)
            # Using 0.0.0.0/0 as verified by user curl.
            try:
                logger.info(f"Configuring NSGs for {ws.name}...")
                await self.cloud_api.add_nsg_rules(ws.id, [
                    "in tcp 11434 11434 0.0.0.0/0",
                ], name=ws.name)
            except Exception as e:
                logger.error(f"Failed to update NSGs for {ws.name}: {e}")
                # We don't return here because deployment might still work if ports were already open manually

            await self.deploy_gpu_node(target_ip, ws.name, username)
        
        except Exception as e:
            logger.error(f"Error processing workspace {ws.name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    async def deploy_all(self, username: str, ips_file: Optional[str] = None):
        """Run deployment for all discovered or manual nodes."""
        
        # 1. Try to fetch all workspaces to enable Smart Manual Mode
        ip_to_workspace = {}
        all_workspaces = []
        
        if self.cloud_api:
            try:
                # We need ALL workspaces, not just running ones, so we can find paused ones by IP(from metadata? NO, paused VMs don't always have IPs in list... 
                # actually list_workspaces returns everything.
                # BUT: Paused VMs often lose their public IP in some clouds. 
                # SURF Research Cloud: "The IP address is released when the workspace is paused/stopped."
                # User says: "145.38.188.41 is paused." -> Implies they KNOW the IP.
                # If the API returns the LAST KNOWN IP or if the IP is persistent, this works.
                # If the IP is gone, we can't match by IP.
                # HOWEVER: The user provided an IP. If the VM is paused and has no IP, we can't connect anyway.
                # If it HAS an IP, it's either static or still assigned.
                # Let's assume we can map them.
                
                logger.info("Fetching workspace list for smart lookup...")
                all_workspaces = await self.cloud_api.discover_gpu_workspaces()
                for ws in all_workspaces:
                    if ws.resource_meta and ws.resource_meta.ip:
                         ip_to_workspace[ws.resource_meta.ip] = ws
            except Exception as e:
                logger.warning(f"Failed to fetch workspaces for smart lookup: {e}")

        # 2. Manual Mode
        if ips_file:
             logger.info(f"Using manual IP list from {ips_file}")
             tasks = []
             with open(ips_file, 'r') as f:
                 for line in f:
                     ip = line.strip()
                     if not ip:
                         continue
                         
                     # Smart Lookup
                     if ip in ip_to_workspace:
                         ws = ip_to_workspace[ip]
                         logger.info(f"Smart Match: IP {ip} corresponds to workspace {ws.name} ({ws.status})")
                         tasks.append(self.process_workspace(ws, username))
                     else:
                         logger.info(f"Added manual target: {ip} (No Cloud Workspace match found)")
                         # Increase timeout for manual/unknown nodes as they might differ
                         tasks.append(self.deploy_gpu_node(ip, f"Manual-{ip}", username))
             
             if tasks:
                 await asyncio.gather(*tasks)
             return

        # 3. Auto-Discovery Mode (Default)
        if not self.cloud_api:
             logger.error("Cloud API is required for auto-discovery but is not initialized.")
             return

        if not all_workspaces:
             logger.warning("No GPU workspaces found.")
             return

        logger.info(f"Found {len(all_workspaces)} GPU workspaces.")
        
        tasks = []
        for ws in all_workspaces:
            tasks.append(self.process_workspace(ws, username))
            
        await asyncio.gather(*tasks)
