#!/bin/bash
set -e

# Base URL for raw files
BASE_URL="https://raw.githubusercontent.com/raoulg/gpumanager/refs/heads/main/gpu-node"

echo "Setting up GPU Node..."

# Download docker-compose.yml if it doesn't exist
if [ ! -f "docker-compose.yml" ]; then
    echo "Downloading docker-compose.yml..."
    curl -fsSL "$BASE_URL/docker-compose.yml" -o docker-compose.yml
fi

# Download entrypoint.sh if it doesn't exist
if [ ! -f "entrypoint.sh" ]; then
    echo "Downloading entrypoint.sh..."
    curl -fsSL "$BASE_URL/entrypoint.sh" -o entrypoint.sh
fi

# Make entrypoint executable (required for mounting)
chmod +x entrypoint.sh

# Check for --shared flag
SHARED_SETUP=false
for arg in "$@"; do
    if [ "$arg" == "--shared" ]; then
        SHARED_SETUP=true
        break
    fi
done

if [ "$SHARED_SETUP" = true ]; then
    echo "Setting up shared directory /srv/shared..."
    
    # Determine user
    if [ -n "$SUDO_USER" ]; then
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
    sudo chgrp collaborators /srv/shared
    sudo chmod g+rws,o-w /srv/shared
    
    # Add user to group
    sudo usermod -aG collaborators "$CURRENT_USER"
    
    echo "Shared directory setup complete. You may need to logout and login again for group changes to take effect."
fi

echo "Starting services..."
docker compose up -d

echo "GPU Node is running!"
echo "To check logs: docker compose logs -f"
echo "To change models: edit docker-compose.yml and run 'docker compose up -d'"
