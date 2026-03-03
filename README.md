# VGPU Sustainment Agent - AMD MI300X Edition

Reverse Quantum Annealing (RQA) for Network Traffic Arbitration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AMD MI300X (ROCm)                        │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │ HIP Compute     │    │ Network Arbitration         │  │
│  │ - RQA Kernel    │◄───│ - Flow Classification        │  │
│  │ - Quantum Phase │    │ - Priority Scoring           │  │
│  └─────────────────┘    └─────────────────────────────┘  │
│           │                          │                     │
│           ▼                          ▼                     │
│  ┌──────────────────────────────────────────────┐         │
│  │         UDP Port 5555 (RQA Updates)         │         │
│  │    Recurrence Rate, Determinism, Entropy,    │         │
│  │    Laminarity                                │         │
│  └──────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `vgpu-config.yaml` | Configuration for MI300X deployment |
| `vgpu_kernels.hip` | HIP kernels (AMD GPU code) |
| `vgpu_agent.py` | Main agent with RQA dynamics |
| `setup_amd.sh` | Setup script for droplet |

## RQA Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Recurrence Rate | RR | Rate of solution space exploration |
| Determinism | DET | How focused vs exploratory |
| Entropy | ENTR | Randomness in transitions |
| Laminarity | LAM | Memory of good solutions |

## Usage

1. **On AMD Droplet:**
   ```bash
   chmod +x setup_amd.sh
   ./setup_amd.sh
   cd /opt/vgpu
   python3 vgpu_agent.py
   ```

2. **Send RQA Updates (from any machine):**
   ```python
   import socket, struct
   
   sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
   
   # Pack 4 floats: RR, DET, ENTR, LAM
   data = struct.pack("ffff", 0.5, 0.8, 0.1, 0.5)
   sock.sendto(data, ("DROPLET_IP", 5555))
   ```

## iFind Integration

For Apple's iFind network arbitration, send RQA signatures based on:
- Network congestion levels → Entropy
- Packet priority → Determinism  
- Flow stability → Laminarity
- Traffic rate → Recurrence Rate

## Notes

- **Metal → ROCm**: Original Apple Metal code ported to HIP
- **Quantum-like**: Uses phase-based dynamics inspired by quantum annealing
- **Network**: UDP for low-latency parameter updates
- **Memory**: <90MB footprint
