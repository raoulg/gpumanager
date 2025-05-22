"""API key management."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

from loguru import logger
from pydantic import ValidationError

from .models import APIKeysFile, UserInfo, AuthenticatedUser


class APIKeyManager:
    """Manages API key validation and user information."""

    def __init__(self, api_keys_file: Path):
        """Initialize API key manager."""
        self.api_keys_file = api_keys_file
        self._api_keys_data: Optional[APIKeysFile] = None
        self._last_loaded: Optional[datetime] = None

        logger.info(f"Initialized APIKeyManager with file: {api_keys_file}")

    def _load_api_keys(self) -> APIKeysFile:
        """Load API keys from file with caching."""
        # Check if file has been modified since last load
        if self.api_keys_file.exists():
            file_mtime = datetime.fromtimestamp(self.api_keys_file.stat().st_mtime)
            if (
                self._api_keys_data
                and self._last_loaded
                and file_mtime <= self._last_loaded
            ):
                return self._api_keys_data

        try:
            if not self.api_keys_file.exists():
                logger.warning(f"API keys file not found: {self.api_keys_file}")
                return APIKeysFile()

            with open(self.api_keys_file, "r") as f:
                data = json.load(f)

            api_keys_data = APIKeysFile(**data)
            self._api_keys_data = api_keys_data
            self._last_loaded = datetime.now()

            logger.info(f"Loaded {len(api_keys_data.api_keys)} API keys from file")
            return api_keys_data

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to load API keys file: {e}")
            raise ValueError(f"Invalid API keys file format: {e}")
        except Exception as e:
            logger.error(f"Error loading API keys file: {e}")
            raise

    def _save_api_keys(self, api_keys_data: APIKeysFile) -> None:
        """Save API keys data to file."""
        try:
            # Create directory if it doesn't exist
            self.api_keys_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save with nice formatting
            data = api_keys_data.model_dump()
            with open(self.api_keys_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            # Update cache
            self._api_keys_data = api_keys_data
            self._last_loaded = datetime.now()

            logger.debug(f"Saved API keys to {self.api_keys_file}")

        except Exception as e:
            logger.error(f"Failed to save API keys file: {e}")
            raise

    def validate_api_key(self, api_key: str) -> Optional[AuthenticatedUser]:
        """Validate an API key and return user information."""
        if not api_key or not api_key.strip():
            return None

        try:
            api_keys_data = self._load_api_keys()

            if api_key not in api_keys_data.api_keys:
                logger.debug(f"Invalid API key attempted: {api_key[:8]}...")
                return None

            user_info = api_keys_data.api_keys[api_key]
            logger.debug(f"Valid API key for user: {user_info.name}")

            return AuthenticatedUser(api_key=api_key, user_info=user_info)

        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None

    def update_user_stats(self, api_key: str) -> None:
        """Update user request statistics."""
        try:
            api_keys_data = self._load_api_keys()

            if api_key not in api_keys_data.api_keys:
                logger.warning(
                    f"Attempted to update stats for invalid API key: {api_key[:8]}..."
                )
                return

            user_info = api_keys_data.api_keys[api_key]

            # Update statistics
            user_info.total_requests += 1
            user_info.requests_today += 1  # TODO: Reset daily counter at midnight
            user_info.last_request = datetime.now()

            # Save updated data
            self._save_api_keys(api_keys_data)

            logger.debug(
                f"Updated stats for user {user_info.name}: {user_info.total_requests} total requests"
            )

        except Exception as e:
            logger.error(f"Failed to update user stats: {e}")

    def get_all_users(self) -> Dict[str, UserInfo]:
        """Get all users and their information (for admin purposes)."""
        try:
            api_keys_data = self._load_api_keys()
            return api_keys_data.api_keys.copy()
        except Exception as e:
            logger.error(f"Failed to get all users: {e}")
            return {}

    def add_user(self, api_key: str, name: str, email: str) -> bool:
        """Add a new user (for admin purposes)."""
        try:
            api_keys_data = self._load_api_keys()

            if api_key in api_keys_data.api_keys:
                logger.warning(f"API key already exists: {api_key[:8]}...")
                return False

            user_info = UserInfo(
                name=name, email=email, created=datetime.now().strftime("%Y-%m-%d")
            )

            api_keys_data.api_keys[api_key] = user_info
            self._save_api_keys(api_keys_data)

            logger.info(f"Added new user: {name} ({email})")
            return True

        except Exception as e:
            logger.error(f"Failed to add user: {e}")
            return False

    def remove_user(self, api_key: str) -> bool:
        """Remove a user (for admin purposes)."""
        try:
            api_keys_data = self._load_api_keys()

            if api_key not in api_keys_data.api_keys:
                logger.warning(f"API key not found: {api_key[:8]}...")
                return False

            user_info = api_keys_data.api_keys[api_key]
            del api_keys_data.api_keys[api_key]
            self._save_api_keys(api_keys_data)

            logger.info(f"Removed user: {user_info.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove user: {e}")
            return False
