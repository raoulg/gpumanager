"""Main application entry point."""

import sys
import os
from pathlib import Path

import uvicorn
from loguru import logger

from gpumanager.config.loader import ConfigLoader
from gpumanager.cloud.api import CloudAPI
from gpumanager.api.handlers import RequestHandler
from gpumanager.auth.manager import APIKeyManager
from gpumanager.gpu.manager import GPUManager


def setup_logging():
    """Setup logging configuration."""
    logger.remove()  # Remove default handler

    # Add console handler with nice formatting
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file handler for detailed logs
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )


def create_app_sync():
    """Synchronous app factory for uvicorn."""
    from contextlib import asynccontextmanager

    # Setup logging
    setup_logging()

    logger.info("Starting LLM GPU Controller...")

    try:
        # Load configuration
        config = ConfigLoader.load_config()
        logger.info(
            f"Configuration loaded for {len(config.timing.__dict__)} timing settings"
        )

        # Initialize cloud API
        cloud_api = CloudAPI(config.cloud_api)

        # Initialize API key manager
        api_key_manager = APIKeyManager(config.paths.api_keys_file)

        # Initialize GPU manager (without async initialization for now)
        gpu_manager = GPUManager(cloud_api, config.timing)

        @asynccontextmanager
        async def lifespan(app):
            # Startup
            logger.info("Starting GPU manager initialization...")
            await gpu_manager.initialize()
            logger.success("GPU manager initialized successfully")
            yield
            # Shutdown
            logger.info("Shutting down GPU manager...")
            await gpu_manager.shutdown()
            logger.success("GPU manager shutdown complete")

        # Create request handler with lifespan
        request_handler = RequestHandler(
            cloud_api, api_key_manager, gpu_manager, lifespan
        )

        logger.success("Application initialized successfully")
        return request_handler.app

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


def generate_key(name: str, email: str):
    """Generate a new API key."""
    import secrets
    from gpumanager.config.loader import ConfigLoader
    from gpumanager.auth.manager import APIKeyManager

    try:
        config = ConfigLoader.load_config()
        api_key_manager = APIKeyManager(config.paths.api_keys_file)
        
        # Generate secure key
        api_key = f"sk-{secrets.token_urlsafe(32)}"
        
        if api_key_manager.add_user(api_key, name, email):
            print(f"\nAPI Key generated successfully for {name} ({email}):")
            print(f"\n{api_key}\n")
            print("Keep this key safe! It has been saved to api_keys.json")
        else:
            print("Failed to add user (key might already exist, try again)")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Failed to generate key: {e}")
        sys.exit(1)

def ensure_credentials():
    """Ensure security credentials exist in gpu-node/.env."""
    import secrets
    import bcrypt
    import base64
    
    env_path = Path("gpu-node/.env")
    if not env_path.exists():
        logger.error(f"Environment file not found: {env_path}")
        sys.exit(1)
        
    content = env_path.read_text()
    updates = []
    
    # 1. WebUI Admin Password
    if "WEBUI_ADMIN_PASSWORD=" not in content:
        password = secrets.token_urlsafe(16)
        updates.append(f"WEBUI_ADMIN_PASSWORD={password}")
        logger.info("Generated new WEBUI_ADMIN_PASSWORD")
        
    # 2. Gatekeeper Password
    gatekeeper_password = None
    if "GATEKEEPER_PASSWORD=" not in content:
        gatekeeper_password = secrets.token_urlsafe(16)
        updates.append(f"GATEKEEPER_PASSWORD={gatekeeper_password}")
        logger.info("Generated new GATEKEEPER_PASSWORD")
    else:
        # Extract existing if needed
        import dotenv
        config = dotenv.dotenv_values(env_path)
        gatekeeper_password = config.get("GATEKEEPER_PASSWORD")

    # 3. Gatekeeper Hash
    if "GATEKEEPER_HASH=" not in content and gatekeeper_password:
        logger.info("Generating bcrypt hash for Gatekeeper...")
        try:
            # Generate salt and hash
            # Caddy expects standard bcrypt hash.
            # Python bcrypt generates b'$2b$...', we need string.
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(gatekeeper_password.encode('utf-8'), salt)
            hash_str = hashed.decode('utf-8')
            
            # Escape $ signs for docker-compose/shell if needed? 
            # Usually .env handles it, but docker-compose might interpolate. 
            # Single quotes usually safe.
            updates.append(f"GATEKEEPER_HASH='{hash_str}'")
            logger.info("Generated GATEKEEPER_HASH")
        except Exception as e:
            logger.error(f"Failed to generate hash: {e}")
            sys.exit(1)

    if updates:
        with open(env_path, "a") as f:
            f.write("\n" + "\n".join(updates) + "\n")
        logger.success(f"Updated {env_path} with new credentials")




def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM GPU Controller")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the GPU manager server")
    
    # Generate key command
    key_parser = subparsers.add_parser("generate-key", help="Generate a new API key")
    key_parser.add_argument("--name", required=True, help="User name")
    key_parser.add_argument("--email", required=True, help="User email")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy GPU nodes")
    deploy_parser.add_argument("username", nargs="?", help="Username for remote setup (defaults to SSH_USER env var)")
    deploy_parser.add_argument("--ips", help="Path to file containing IP addresses (disables discovery)")
    deploy_parser.add_argument("--manager", help="IP address for the manager node (deploys centralized WebUI)")
    deploy_parser.add_argument("--with-api", action="store_true", help="Deploy GPU Manager API on manager node (enables auto load-balancing)")

    # Sync Models Command
    sync_parser = subparsers.add_parser("sync-models", help="Sync models from a source node to all others")
    sync_parser.add_argument("username", nargs="?", help="Username for remote setup (defaults to SSH_USER env var)")
    sync_parser.add_argument("--source", required=True, help="IP address of the source node")
    sync_parser.add_argument("--ips", help="Path to file containing IP addresses (disables discovery)")
    
    # Configure WebUI Command
    configure_parser = subparsers.add_parser("configure-webui", help="Configure WebUI connections to all GPU nodes")
    configure_parser.add_argument("manager_ip", help="IP address of the manager node")
    configure_parser.add_argument("--email", default="admin@gpumanager.local", help="Admin email (default: admin@gpumanager.local)")
    configure_parser.add_argument("--password", help="Admin password (optional, will read from .env if not provided)")
    configure_parser.add_argument("--ips", help="Path to file containing GPU node IP addresses (optional, will auto-discover if not provided)")

    # Open Port Command
    open_port_parser = subparsers.add_parser("open-port", help="Open port(s) in the cloud firewall")
    group = open_port_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ip", help="IP address of the target machine")
    group.add_argument("--name", help="Name of the target machine")
    open_port_parser.add_argument("--ports", help="Comma-separated list of ports to open (e.g., '8000,8080')")
    open_port_parser.add_argument("--port", type=int, help="Single port to open (legacy, use --ports for multiple)")
    
    args = parser.parse_args()
    
    # Default to server if no command provided
    if not args.command or args.command == "server":
        try:
            # Create logs directory
            Path("logs").mkdir(exist_ok=True)

            # Setup logging first
            setup_logging()

            # Load config synchronously for server settings
            config = ConfigLoader.load_config()

            logger.info(f"Starting server on {config.server.host}:{config.server.port}")

            # Run with uvicorn, passing the factory function
            uvicorn.run(
                "gpumanager.main:create_app_sync",
                host=config.server.host,
                port=config.server.port,
                reload=False,  # Set to True for development
                log_level="info",
                factory=True,
            )

        except Exception as e:
            logger.error(f"Failed to start application: {e}")
            sys.exit(1)
            
    elif args.command == "generate-key":
        generate_key(args.name, args.email)

    elif args.command == "deploy":
        import asyncio
        from gpumanager.deployment import DeploymentManager

        setup_logging()
        ensure_credentials()

        # Load env vars to get SSH_USER if needed
        ConfigLoader.load_env_file()

        username = args.username or os.getenv("SSH_USER")
        if not username:
             logger.error("Username not provided and SSH_USER environment variable not set.")
             sys.exit(1)

        cloud_api = None
        # Always try to initialize Cloud API for smart features (auto-resume/reverse lookup)
        try:
            config = ConfigLoader.load_config()
            cloud_api = CloudAPI(config.cloud_api)
        except Exception as e:
            if not args.ips and not args.manager:
                 # If no manual IPs/manager and API fails, we can't do anything
                 logger.error(f"Failed to initialize cloud API for discovery: {e}")
                 sys.exit(1)
            else:
                 # If manual IPs or manager provided, we can fallback to dumb mode
                 logger.warning(f"Cloud API not available ({e}). Smart features disabled.")

        deployment_manager = DeploymentManager(cloud_api)
        try:
            # If manager IP is specified, deploy manager node first
            if args.manager:
                logger.info(f"Deploying manager node to {args.manager}")

                # Try to find workspace ID for the manager IP
                workspace_id = None
                if cloud_api:
                    async def find_workspace():
                        workspaces = await cloud_api.list_workspaces()
                        for ws in workspaces:
                            if ws.resource_meta and ws.resource_meta.ip == args.manager:
                                return ws.id, ws.name
                        return None, None

                    workspace_id, workspace_name = asyncio.run(find_workspace())
                    if workspace_id:
                        logger.info(f"Found workspace: {workspace_name} ({workspace_id})")
                        manager_name = workspace_name
                    else:
                        logger.warning(f"No workspace found for IP {args.manager}, firewall rules may need manual configuration")
                        manager_name = "Manager"
                else:
                    manager_name = "Manager"

                asyncio.run(deployment_manager.deploy_manager_node(args.manager, manager_name, username, workspace_id, with_api=args.with_api))
                logger.success("Manager node deployment complete!")
                if args.with_api:
                    logger.info(f"GPU Manager API: http://{args.manager}:8000")
                logger.info(f"Access WebUI at: http://{args.manager}:8080")
            else:
                # Deploy GPU nodes as before
                asyncio.run(deployment_manager.deploy_all(username, args.ips))
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            sys.exit(1)

    elif args.command == "sync-models":
        import asyncio
        from gpumanager.deployment import DeploymentManager
        from gpumanager.sync import ModelSynchronizer

        setup_logging()

        # Load env vars to get SSH_USER if needed
        ConfigLoader.load_env_file()
        
        username = args.username or os.getenv("SSH_USER")
        if not username:
             logger.error("Username not provided and SSH_USER environment variable not set.")
             sys.exit(1)

        # Initialize API same as deploy
        cloud_api = None
        try:
            config = ConfigLoader.load_config()
            cloud_api = CloudAPI(config.cloud_api)
        except Exception:
            if not args.ips:
                 logger.error("Cloud API init failed and no IPs file provided.")
                 sys.exit(1)

        manager = DeploymentManager(cloud_api)
        synchronizer = ModelSynchronizer(manager)

        try:
            asyncio.run(synchronizer.sync_and_deploy(args.source, username, args.ips))
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            sys.exit(1)

    elif args.command == "configure-webui":
        import asyncio
        import httpx
        from gpumanager.deployment import DeploymentManager

        setup_logging()

        async def configure_webui():
            # Load credentials
            admin_email = args.email
            admin_pass = args.password

            # If password not provided, try to load from .env
            if not admin_pass:
                env_path = Path("manager-node/.env")
                if env_path.exists():
                    import dotenv
                    config = dotenv.dotenv_values(env_path)
                    admin_pass = config.get("WEBUI_ADMIN_PASSWORD")

            if not admin_pass:
                logger.error("Password not provided and not found in manager-node/.env")
                logger.error("Use --password option or ensure manager-node/.env exists")
                sys.exit(1)

            # Load gatekeeper password
            env_path = Path("manager-node/.env")
            if not env_path.exists():
                logger.error(f"Manager .env not found at {env_path}")
                sys.exit(1)

            import dotenv
            config = dotenv.dotenv_values(env_path)
            gatekeeper_pass = config.get("GATEKEEPER_PASSWORD")

            if not gatekeeper_pass:
                logger.error("Gatekeeper password not found in manager-node/.env")
                sys.exit(1)

            # Discover or load GPU nodes
            gpu_nodes = []
            if args.ips:
                logger.info(f"Loading GPU nodes from {args.ips}")
                with open(args.ips, 'r') as f:
                    gpu_nodes = [{"ip": line.strip(), "name": f"GPU-{line.strip()}"} for line in f if line.strip()]
            else:
                logger.info("Auto-discovering GPU nodes...")
                try:
                    app_config = ConfigLoader.load_config()
                    cloud_api = CloudAPI(app_config.cloud_api)
                    workspaces = await cloud_api.discover_gpu_workspaces()
                    gpu_nodes = [{"ip": ws.resource_meta.ip, "name": ws.name} for ws in workspaces if ws.resource_meta.ip]
                    logger.info(f"Discovered {len(gpu_nodes)} GPU nodes")
                except Exception as e:
                    logger.error(f"Failed to discover GPU nodes: {e}")
                    logger.error("Please use --ips to manually specify GPU node IPs")
                    sys.exit(1)

            if not gpu_nodes:
                logger.error("No GPU nodes found")
                sys.exit(1)

            # Login to WebUI
            webui_url = f"http://{args.manager_ip}:8080"
            logger.info(f"Connecting to WebUI at {webui_url}")

            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                # Login
                login_response = await client.post(
                    f"{webui_url}/api/v1/auths/signin",
                    json={"email": admin_email, "password": admin_pass},
                    auth=("gatekeeper", gatekeeper_pass)
                )

                if login_response.status_code != 200:
                    logger.error(f"Login failed: {login_response.status_code} {login_response.text}")
                    sys.exit(1)

                token = login_response.json().get("token")
                logger.success("Logged in to WebUI")

                # Build Ollama URLs list
                headers = {"Authorization": f"Bearer {token}"}
                ollama_urls = [{"url": f"http://{node['ip']}:11434", "name": node['name']} for node in gpu_nodes]

                logger.info(f"Configuring {len(ollama_urls)} Ollama connections...")

                # Try different API endpoints based on OpenWebUI version
                endpoints_to_try = [
                    ("/api/v1/configs/ollama/urls", "POST"),
                    ("/api/config/update", "POST"),
                    ("/api/configs", "POST"),
                ]

                config_updated = False
                for endpoint, method in endpoints_to_try:
                    try:
                        logger.debug(f"Trying {method} {endpoint}")

                        # Build payload - try different formats
                        payloads_to_try = [
                            {"OLLAMA_BASE_URLS": [u["url"] for u in ollama_urls]},
                            {"ollama_base_urls": [u["url"] for u in ollama_urls]},
                            {"urls": ollama_urls},
                        ]

                        for payload in payloads_to_try:
                            response = await client.request(
                                method,
                                f"{webui_url}{endpoint}",
                                json=payload,
                                headers=headers,
                                auth=("gatekeeper", gatekeeper_pass)
                            )

                            if response.status_code in [200, 201]:
                                logger.success(f"✓ Successfully configured Ollama connections via {endpoint}")
                                for node in gpu_nodes:
                                    logger.info(f"  - {node['name']}: http://{node['ip']}:11434")
                                config_updated = True
                                break

                        if config_updated:
                            break

                    except Exception as e:
                        logger.debug(f"Failed {endpoint}: {e}")
                        continue

                if not config_updated:
                    logger.info("API auto-config didn't work. Using docker-compose approach...")

                    # Build comma-separated list of Ollama URLs
                    ollama_base_urls = ";".join([f"http://{node['ip']}:11434" for node in gpu_nodes])

                    # Update manager docker-compose.yml via SSH
                    logger.info("Updating manager node configuration...")

                    # Create updated docker-compose content
                    update_script = f"""
cd /srv/shared && \
docker compose down && \
sed -i '/OLLAMA_BASE_URL=/d' docker-compose.yml && \
sed -i '/environment:/a\\      - OLLAMA_BASE_URL={ollama_base_urls}' docker-compose.yml && \
docker compose up -d
"""

                    # Execute via SSH
                    ssh_cmd = [
                        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                        args.manager_ip, update_script
                    ]

                    import subprocess
                    result = subprocess.run(ssh_cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        logger.success(f"✓ Successfully configured {len(gpu_nodes)} Ollama connections!")
                        logger.info("WebUI is restarting with new configuration...")
                        logger.info(f"Access WebUI at: http://{args.manager_ip}:8080")
                        for node in gpu_nodes:
                            logger.info(f"  - {node['name']}: http://{node['ip']}:11434")
                    else:
                        logger.error(f"Failed to update configuration: {result.stderr}")
                        logger.warning("Please add connections manually:")
                        logger.info("Go to Admin Panel → Settings → Connections in the WebUI")
                        for node in gpu_nodes:
                            logger.info(f"  - {node['name']}: http://{node['ip']}:11434")
                else:
                    logger.success(f"Configuration complete! Added {len(gpu_nodes)} GPU nodes to WebUI")

        asyncio.run(configure_webui())

    elif args.command == "open-port":
        import asyncio

        async def run_open_port():
            config = ConfigLoader.load_config()
            # Clear filter to find ANY workspace (e.g. manager)
            config.cloud_api.machine_name_filter = ""
            api = CloudAPI(config.cloud_api)

            target = None
            if args.ip:
                logger.info(f"Searching for workspace with IP {args.ip}...")
                workspaces = await api.list_workspaces()
                target = next((w for w in workspaces if w.resource_meta and w.resource_meta.ip == args.ip), None)
            elif args.name:
                logger.info(f"Searching for workspace with name {args.name}...")
                workspaces = await api.list_workspaces()
                target = next((w for w in workspaces if w.name == args.name), None)

            if not target:
                logger.error("Target workspace not found.")
                sys.exit(1)

            logger.info(f"Found workspace: {target.name} ({target.id})")

            # Parse ports
            ports = []
            if args.ports:
                # Multiple ports via comma-separated list
                ports = [int(p.strip()) for p in args.ports.split(',')]
            elif args.port:
                # Single port (legacy)
                ports = [args.port]
            else:
                logger.error("Must specify either --port or --ports")
                sys.exit(1)

            # Create rules for all ports
            rules = [f"in tcp {port} {port} 0.0.0.0/0" for port in ports]
            logger.info(f"Opening {len(ports)} port(s): {', '.join(map(str, ports))}")

            try:
                # Use update_nsgs directly with all ports at once
                await api.update_nsgs(target.id, rules, name=target.name)
                logger.success(f"Ports {', '.join(map(str, ports))} opened successfully on {target.name}")
            except Exception as e:
                logger.error(f"Failed to open ports: {e}")
                sys.exit(1)

        setup_logging()
        asyncio.run(run_open_port())

if __name__ == "__main__":
    main()
