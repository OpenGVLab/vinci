#!/bin/bash

# Function to install Git LFS
install_git_lfs() {
    echo "Installing Git LFS..."
    # Check the operating system and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # For Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install git-lfs -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # For macOS
        brew install git-lfs
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # For Windows (using Chocolatey)
        choco install git-lfs
    else
        echo "Unsupported OS. Please install Git LFS manually."
        exit 1
    fi
    git lfs install
}

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS is not installed."
    install_git_lfs
else
    echo "Git LFS is already installed."
fi
REPO_URL="https://huggingface.co/hyf015/Vinci-8B-base"
echo "Cloning the repository: $REPO_URL"
git clone "$REPO_URL"
REPO_URL2="https://huggingface.co/hyf015/Vinci-8B-ckpt"
echo "Cloning the repository: $REPO_URL2"
git clone "$REPO_URL2"
REPO_URL3="https://huggingface.co/hyf015/seine_weights"
echo "Cloning the repository: $REPO_URL3"
git clone "$REPO_URL3"
