#!/bin/bash

# Function to print messages with colors
print_status() {
    local color=$1
    local message=$2
    case $color in
        "green") echo -e "\e[32m$message\e[0m" ;;
        "red") echo -e "\e[31m$message\e[0m" ;;
        "yellow") echo -e "\e[33m$message\e[0m" ;;
    esac
}

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        print_status "green" "✔ $1 successful"
    else
        print_status "red" "✘ $1 failed"
        exit 1
    fi
}

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
    print_status "red" "Please run this script as root or with sudo"
    exit 1
fi

print_status "yellow" "Starting Docker installation process..."

# Remove old versions
print_status "yellow" "Removing old Docker packages if they exist..."
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
    apt-get remove -y $pkg &>/dev/null
done
check_status "Removal of old packages"

# Update package index
print_status "yellow" "Updating package index..."
apt-get update
check_status "Package index update"

# Install prerequisites
print_status "yellow" "Installing prerequisites..."
apt-get install -y ca-certificates curl
check_status "Prerequisites installation"

# Setup Docker repository
print_status "yellow" "Setting up Docker repository..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
check_status "Docker GPG key setup"

# Add Docker repository
print_status "yellow" "Adding Docker repository..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
tee /etc/apt/sources.list.d/docker.list > /dev/null
check_status "Docker repository addition"

# Update package index again
print_status "yellow" "Updating package index with Docker repository..."
apt-get update
check_status "Package index update with Docker repository"

# Install Docker
print_status "yellow" "Installing Docker packages..."
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
check_status "Docker installation"

# Install NVIDIA Drivers if missing or broken (check for valid GPU output)
if ! nvidia-smi -L | grep -q "GPU"; then
    print_status "yellow" "NVIDIA drivers not detected. Installing drivers..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    # User confirmed nvidia-driver-535 works reliably
    apt-get install -y nvidia-driver-535
    check_status "NVIDIA driver installation"
    print_status "yellow" "Drivers installed. A reboot will be required."
fi

# Install NVIDIA Container Toolkit
print_status "yellow" "Setting up NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

check_status "NVIDIA repository setup"

print_status "yellow" "Installing NVIDIA Container Toolkit..."
apt-get update
apt-get install -y nvidia-container-toolkit
check_status "NVIDIA Container Toolkit installation"

print_status "yellow" "Configuring Docker to use Nvidia driver..."
nvidia-ctk runtime configure --runtime=docker
check_status "NVIDIA runtime configuration"

print_status "yellow" "Restarting Docker..."
systemctl restart docker
check_status "Docker restart"


# Determine the actual user to add to the docker group
if [ -n "$1" ]; then
    ACTUAL_USER="$1"
elif [ -n "$SUDO_USER" ]; then
    ACTUAL_USER="$SUDO_USER"
else
    ACTUAL_USER=$(logname || whoami)
fi

# Add user to docker group
print_status "yellow" "Adding user $ACTUAL_USER to docker group..."
usermod -aG docker $ACTUAL_USER
check_status "User addition to docker group"
print_status "green" "User $ACTUAL_USER added to the docker group. Please log out and back in." 

# Verify installation
print_status "yellow" "Verifying Docker installation..."
if docker run --rm hello-world &>/dev/null; then
    print_status "green" "✔ Docker installation verified successfully!"
else
    print_status "red" "✘ Docker verification failed. Please check the system logs for more details."
    exit 1
fi

print_status "yellow" "Verifying NVIDIA Container Toolkit..."
# We can't easily verify nvidia-smi inside a container without a GPU present and accessible, 
# but we can check if the runtime is registered.
if docker info | grep -q "Runtimes.*nvidia"; then
     print_status "green" "✔ NVIDIA runtime detected in Docker info!"
else
     print_status "yellow" "⚠ NVIDIA runtime NOT detected in 'docker info'. You might need to check your configuration."
fi

print_status "green" "Setup complete! Please log out and log back in for group changes to take effect."
