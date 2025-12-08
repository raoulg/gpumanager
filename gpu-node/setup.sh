#!/bin/bash
set -e

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed."
    run_installer=true
else
    # Check for NVIDIA runtime AND functional driver
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo "Docker installed but NVIDIA runtime missing."
        run_installer=true
    elif ! nvidia-smi &> /dev/null; then
         echo "Docker installed but nvidia-smi failed (drivers missing/broken)."
         run_installer=true
    else
        run_installer=false
    fi
fi

if [ "$run_installer" = true ]; then
    if [ -f "install-docker.sh" ]; then
        echo "Found install-docker.sh, running it..."
        chmod +x install-docker.sh
        sudo ./install-docker.sh
    else
        echo "Error: Docker/NVIDIA setup required but install-docker.sh not found."
        exit 1
    fi
fi
# Base URL for raw files
BASE_URL="https://raw.githubusercontent.com/raoulg/gpumanager/refs/heads/main/gpu-node"

# Capture current directory where script and files are located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Setting up GPU Node..."

# Default values
SHARED_SETUP=false
SetupUser=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --shared)
            SHARED_SETUP=true
            shift # past argument
            ;;
        --user)
            SetupUser="$2"
            shift # past argument
            shift # past value
            ;;
        *)
            shift # past argument
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed."
    run_installer=true
else
    # Check for NVIDIA runtime AND functional driver
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo "Docker installed but NVIDIA runtime missing."
        run_installer=true
    elif ! nvidia-smi &> /dev/null; then
         echo "Docker installed but nvidia-smi failed (drivers missing/broken)."
         run_installer=true
    else
        run_installer=false
    fi
fi

if [ "$run_installer" = true ]; then
    if [ -f "$SCRIPT_DIR/install-docker.sh" ]; then
        echo "Found install-docker.sh, running it..."
        chmod +x "$SCRIPT_DIR/install-docker.sh"
        # Pass user if defined
        if [ -n "$SetupUser" ]; then
            sudo "$SCRIPT_DIR/install-docker.sh" "$SetupUser"
        else
            sudo "$SCRIPT_DIR/install-docker.sh"
        fi
    else
        echo "Error: Docker/NVIDIA setup required but install-docker.sh not found in $SCRIPT_DIR."
        exit 1
    fi
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

# Handle docker-compose.yml
if [ -f "$SCRIPT_DIR/docker-compose.yml" ]; then
    echo "Using local docker-compose.yml..."
    if [ "$SHARED_SETUP" = true ]; then
        # prevent copying detailed same file
        if [ "$SCRIPT_DIR/docker-compose.yml" != "/srv/shared/docker-compose.yml" ]; then
            cp "$SCRIPT_DIR/docker-compose.yml" /srv/shared/
        fi
    fi
else
    echo "Downloading docker-compose.yml..."
    curl -fsSL "$BASE_URL/docker-compose.yml" -o docker-compose.yml
    if [ "$SHARED_SETUP" = true ]; then
        # If we are in shared dir (cd /srv/shared), curl saved it here.
        # Logic matches original "mv" if not shared, but here we are IN shared.
        # If downloaded while IN shared, it is already there. nothing to do.
        :
    fi
fi

# Handle entrypoint.sh
if [ -f "$SCRIPT_DIR/entrypoint.sh" ]; then
    echo "Using local entrypoint.sh..."
    if [ "$SHARED_SETUP" = true ]; then
        if [ "$SCRIPT_DIR/entrypoint.sh" != "/srv/shared/entrypoint.sh" ]; then
            cp "$SCRIPT_DIR/entrypoint.sh" /srv/shared/
        fi
    fi
else
    echo "Downloading entrypoint.sh..."
    curl -fsSL "$BASE_URL/entrypoint.sh" -o entrypoint.sh
fi

# Make entrypoint executable (required for mounting)
chmod +x entrypoint.sh

echo "Starting services in $(pwd)..."
# Check connectivity to nvidia
if ! nvidia-smi &> /dev/null; then
  echo "Warning: nvidia-smi failed. GPU might not be available."
fi

docker compose up -d

echo "GPU Node is running!"
if [ "$SHARED_SETUP" = true ]; then
    echo "Files are located in /srv/shared"
    echo "Note: You may need to logout and login again for group permissions to take effect."
fi
echo "To check logs: docker compose logs -f"
echo "To change models: edit docker-compose.yml and run 'docker compose up -d'"
