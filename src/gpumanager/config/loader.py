"""Configuration loader."""

import os
import tomllib
from pathlib import Path
from typing import Dict, Any

from loguru import logger
from pydantic import ValidationError
from dotenv import load_dotenv

from .models import AppConfig


class ConfigLoader:
    """Loads and validates configuration from TOML files and environment variables."""

    @staticmethod
    def load_env_file(env_path: Path = Path(".env")) -> None:
        """Load environment variables from .env file."""
        if not env_path.exists():
            logger.warning(f"Environment file not found: {env_path}")
            return

        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")

    @staticmethod
    def load_toml(config_path: Path) -> Dict[str, Any]:
        """Load TOML configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config_data

    @staticmethod
    def load_env_secrets() -> Dict[str, str]:
        """Load secrets from environment variables."""
        secrets = {}

        # Required environment variables
        required_env_vars = [
            "CLOUD_API_TOKEN",
        ]

        # Optional environment variables
        optional_env_vars = [
            "CLOUD_CSRF_TOKEN",
        ]

        # Load required variables
        for var in required_env_vars:
            value = os.getenv(var)
            if not value:
                raise ValueError(f"Required environment variable {var} is not set")
            secrets[var.lower()] = value

        # Load optional variables
        for var in optional_env_vars:
            value = os.getenv(var)
            if value:
                secrets[var.lower()] = value

        logger.info(f"Loaded {len(secrets)} environment variables")
        return secrets

    @staticmethod
    def merge_config_with_secrets(
        config_data: Dict[str, Any], secrets: Dict[str, str]
    ) -> Dict[str, Any]:
        """Merge configuration with environment secrets."""
        # Add secrets to cloud_api section
        if "cloud_api" not in config_data:
            config_data["cloud_api"] = {}

        config_data["cloud_api"]["auth_token"] = secrets["cloud_api_token"]

        if "cloud_csrf_token" in secrets:
            config_data["cloud_api"]["csrf_token"] = secrets["cloud_csrf_token"]

        return config_data

    @classmethod
    def load_config(cls, config_path: Path = Path("config.toml")) -> AppConfig:
        """Load complete application configuration."""
        try:
            # First, load environment variables from .env file
            cls.load_env_file()

            # Load TOML configuration
            config_data = cls.load_toml(config_path)

            # Load environment secrets
            secrets = cls.load_env_secrets()

            # Merge configuration with secrets
            merged_config = cls.merge_config_with_secrets(config_data, secrets)

            # Validate and create AppConfig
            app_config = AppConfig(**merged_config)

            logger.success("Configuration loaded and validated successfully")
            return app_config

        except (FileNotFoundError, ValueError, ValidationError) as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
