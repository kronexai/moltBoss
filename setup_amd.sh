#!/bin/bash
# VGPU AMD MI300X Setup Script
# Run this on your AMD Developer Cloud droplet

set -e

echo "========================================="
echo "VGPU Sustainment Agent - AMD MI300X Setup"
echo "========================================="

# Check ROCm installation
echo "[1/5] Checking ROCm..."
if ! command -v rocm-smi &> /dev/null; then
    echo "ERROR: ROCm not installed. Please use ROCm-enabled image."
    exit 1
fi

rocm-smi

# Install Python dependencies
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install numpy pyyaml

# Try to install HIP Python bindings (optional)
pip install hip-python 2>/dev/null || echo "[Note] HIP bindings not available via pip - will use CPU fallback"

# Copy kernel source
echo "[3/5] Setting up kernel sources..."
mkdir -p /opt/vgpu/kernels
cp vgpu_kernels.hip /opt/vgpu/kernels/

# Compile HIP kernel (optional - requires hipcc)
echo "[4/5] Compiling HIP kernel..."
if command -v hipcc &> /dev/null; then
    hipcc -O3 --fast-math -ffast-math \
        /opt/vgpu/kernels/vgpu_kernels.hip \
        -o /opt/vgpu/kernels/vgpu_sustainment.kernel \
        -fkernel-open-only || echo "[Note] Kernel compilation skipped"
else
    echo "[Note] hipcc not found - using CPU fallback mode"
fi

# Copy agent code
echo "[5/5] Installing agent..."
cp vgpu_agent.py /opt/vgpu/
chmod +x /opt/vgpu/vgpu_agent.py

echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run the agent:"
echo "  cd /opt/vgpu"
echo "  python3 vgpu_agent.py"
echo ""
echo "To monitor GPU:"
echo "  rocm-smi"
echo ""
echo "The agent listens on UDP port 5555 for RQA updates"
