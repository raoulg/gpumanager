"""Main application entry point."""

import sys
from pathlib import Path

import uvicorn
from loguru import logger

from gpumanager.config.loader import ConfigLoader
from gpumanager.cloud.api import CloudAPI
from gpumanager.api.handlers import RequestHandler
from gpumanager.auth.manager import APIKeyManager


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

        # Create request handler
        request_handler = RequestHandler(cloud_api, api_key_manager)

        logger.success("Application initialized successfully")
        return request_handler.app

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise


def main():
    """Main entry point."""
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


if __name__ == "__main__":
    main()
