"""Authentication middleware."""

from typing import Optional, Callable
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from loguru import logger

from gpumanager.auth.manager import APIKeyManager
from gpumanager.auth.models import AuthenticatedUser


# HTTP Bearer token scheme for extracting API keys from Authorization header
security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """Authentication middleware for validating API keys."""

    def __init__(self, api_key_manager: APIKeyManager):
        """Initialize auth middleware."""
        self.api_key_manager = api_key_manager
        logger.info("Initialized AuthMiddleware")

    def get_current_user(
        self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> AuthenticatedUser:
        """
        Dependency to get current authenticated user.

        Expects Authorization header: "Bearer <api_key>"
        """
        if not credentials:
            logger.debug("No authorization credentials provided")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        api_key = credentials.credentials
        user = self.api_key_manager.validate_api_key(api_key)

        if not user:
            logger.debug(f"Invalid API key attempted: {api_key[:8]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Update user statistics
        try:
            self.api_key_manager.update_user_stats(api_key)
        except Exception as e:
            logger.warning(f"Failed to update user stats: {e}")
            # Don't fail the request if stats update fails

        logger.debug(f"Authenticated user: {user.name}")
        return user

    def get_optional_user(
        self, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[AuthenticatedUser]:
        """
        Optional dependency to get current user (doesn't require authentication).

        Returns None if no valid credentials provided.
        Useful for endpoints that have optional authentication.
        """
        if not credentials:
            return None

        api_key = credentials.credentials
        user = self.api_key_manager.validate_api_key(api_key)

        if user:
            # Update user statistics
            try:
                self.api_key_manager.update_user_stats(api_key)
            except Exception as e:
                logger.warning(f"Failed to update user stats: {e}")

        return user


def create_auth_dependency(api_key_manager: APIKeyManager) -> Callable:
    """
    Create an authentication dependency function.

    This is a factory function that creates the dependency with the API key manager.
    """
    auth_middleware = AuthMiddleware(api_key_manager)
    return auth_middleware.get_current_user


def create_optional_auth_dependency(api_key_manager: APIKeyManager) -> Callable:
    """
    Create an optional authentication dependency function.
    """
    auth_middleware = AuthMiddleware(api_key_manager)
    return auth_middleware.get_optional_user
