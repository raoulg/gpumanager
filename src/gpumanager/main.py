"""Main application entry point."""

import sys
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

if __name__ == "__main__":
    main()
