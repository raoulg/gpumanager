#!/bin/bash
set -e

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker first using:"
    echo "curl -fsSL https://raw.githubusercontent.com/raoulg/mlflow-serversetup/refs/heads/main/install-docker.sh | sudo bash"
    exit 1
fi
# Base URL for raw files
BASE_URL="https://raw.githubusercontent.com/raoulg/gpumanager/refs/heads/main/gpu-node"

echo "Setting up GPU Node..."

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
    sudo chown "$CURRENT_USER":collaborators /srv/shared
    sudo chmod g+rws,o-w /srv/shared
    
    # Add user to group
    sudo usermod -aG collaborators "$CURRENT_USER"
    
    echo "Shared directory setup complete."
    echo "Switching to /srv/shared..."
    cd /srv/shared
fi

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

echo "Starting services in $(pwd)..."
docker compose up -d

echo "GPU Node is running!"
if [ "$SHARED_SETUP" = true ]; then
    echo "Files are located in /srv/shared"
    echo "Note: You may need to logout and login again for group permissions to take effect."
fi
echo "To check logs: docker compose logs -f"
echo "To change models: edit docker-compose.yml and run 'docker compose up -d'"
