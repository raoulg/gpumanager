#!/bin/bash
set -e

# Capture current directory where script and files are located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Setting up GPU Manager Node..."

# Default values
SHARED_SETUP=false
SetupUser=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --shared)
            SHARED_SETUP=true
            shift
            ;;
        --user)
            SetupUser="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

if [ "$SHARED_SETUP" = true ]; then
    echo "Setting up shared directory /srv/shared..."

    # Determine user
    if [ -n "$SetupUser" ]; then
        CURRENT_USER="$SetupUser"
    elif [ -n "$SUDO_USER" ]; then
        CURRENT_USER="$SUDO_USER"
    else
        CURRENT_USER=$(whoami)
    fi

    echo "Adding user $CURRENT_USER to collaborators group..."

    # Create group if not exists
    if ! getent group collaborators > /dev/null; then
        sudo groupadd collaborators
    fi

    # Create directory
    sudo mkdir -p /srv/shared

    # Set permissions
    sudo chown "$CURRENT_USER":collaborators /srv/shared
    sudo chmod g+rws,o-w /srv/shared

    # Add user to group
    sudo usermod -aG collaborators "$CURRENT_USER"

    echo "Shared directory setup complete."
    echo "Switching to /srv/shared..."
    cd /srv/shared
fi

# Copy configuration files
if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    echo "Using local docker-compose.yml..."
    if [ "$SHARED_SETUP" = true ]; then
        if [ "$SCRIPT_DIR/docker-compose.yml" != "/srv/shared/docker-compose.yml" ]; then
            cp "$SCRIPT_DIR/docker-compose.yml" /srv/shared/
        fi
    fi
fi

# Handle Caddyfile
if [ -f "$SCRIPT_DIR/Caddyfile" ]; then
    echo "Using local Caddyfile..."
    if [ "$SHARED_SETUP" = true ]; then
        if [ "$SCRIPT_DIR/Caddyfile" != "/srv/shared/Caddyfile" ]; then
            if [ -d "/srv/shared/Caddyfile" ]; then
                echo "Removing stuck Caddyfile directory..."
                rm -rf "/srv/shared/Caddyfile"
            fi
            cp "$SCRIPT_DIR/Caddyfile" /srv/shared/
        fi
    fi
fi

# Handle .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Using local .env..."
    if [ "$SHARED_SETUP" = true ]; then
        if [ "$SCRIPT_DIR/.env" != "/srv/shared/.env" ]; then
            cp "$SCRIPT_DIR/.env" /srv/shared/
        fi
    fi
fi

echo "Starting services in $(pwd)..."

# Ensure clean state
docker compose down --remove-orphans || true

docker compose up -d

echo "GPU Manager Node is running!"
if [ "$SHARED_SETUP" = true ]; then
    echo "Files are located in /srv/shared"
    echo "Note: You may need to logout and login again for group permissions to take effect."
fi
echo "To check logs: docker compose logs -f"
echo "Access WebUI at: http://<manager-ip>:8080"
