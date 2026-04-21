#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}--- Performing Setup ---${NC}"

# ==========================================
# PYTHON SETUP
# ==========================================
echo -e "${GREEN}--- Setting up Environment ---${NC}"

# Check/Install python3-venv
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "The 'python3-venv' package is missing. Installing it now..."
    sudo apt-get update
    sudo apt-get install -y python3-venv
fi

# Create virtual environment 'coraldev'
if [ ! -d "coraldev" ]; then
    echo "Creating virtual environment 'coraldev'..."
    python3 -m venv coraldev
    echo "Environment created."
else
    echo "Virtual environment 'coraldev' already exists. Skipping creation."
fi

# Activate venv
source coraldev/bin/activate
echo "Virtual environment activated."

# ==========================================
# SUBMODULES & HARDWARE LIBS
# ==========================================
echo -e "${GREEN}--- Checking Submodules ---${NC}"
git submodule update --init --recursive

echo -e "${GREEN}--- Setting up Coralmicro ---${NC}"
# Use parentheses to run in a subshell so we don't mess up directories
(
    cd coralmicro || { echo -e "${RED}Failed to find coralmicro dir${NC}"; exit 1; }
    
    # Fix broken coralmicro requirements
    cat > scripts/requirements.txt <<EOF
hexformat==0.2
hidapi
progress==1.5
pyserial==3.5
pyusb==1.2.0
EOF

    # Setup and build coralmicro
    # Note: setup.sh might ask for sudo password
    bash setup.sh
    bash build.sh
)

# Install Python Requirements
echo -e "${GREEN}--- Installing Python Dependencies ---${NC}"
# Install library requirements
pip install -r requirements.txt


# ==========================================
# DONE
# ==========================================
echo -e "${BLUE}--- Setup Complete ---${NC}"

# Launch shell with venv active
exec bash --rcfile "coraldev/bin/activate"