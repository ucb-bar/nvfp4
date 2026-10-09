#!/usr/bin/env bash
# Check that the workspace can build, simulate, and reach the board and the GPU host.
# Exit status 0: no fatal problem (warnings allowed). Exit status 1: a fatal problem.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
status=0
ok()   { printf '  ok    %s\n' "$1"; }
warn() { printf '  warn  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; status=1; }

if [ -n "${PIXI_PROJECT_ROOT:-}" ]; then ok "pixi environment is active"
else fail "pixi environment is not active. Run: pixi run doctor"; fi

if command -v verilator >/dev/null; then ok "$(verilator --version)"
else fail "verilator not found. Run: pixi install"; fi

if python -c "import numpy, serial, torch, pyuartsi" 2>/dev/null; then ok "Python packages: numpy, pyserial, torch, pyuartsi"
else fail "Python packages missing. Run: pixi install"; fi

python - <<'EOF' || status=1
import subprocess, sys, yaml
repos = yaml.safe_load(open("third_party.repos"))["repositories"]
bad = False
for name, r in repos.items():
    head = subprocess.run(["git", "-C", f"third_party/{name}", "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode:
        print(f"  FAIL  third_party/{name} is missing. Run: pixi run setup"); bad = True
    elif head.stdout.strip() != r["version"]:
        print(f"  warn  third_party/{name} is at {head.stdout.strip()[:12]}, not at the pinned {r['version'][:12]}")
    else:
        print(f"  ok    third_party/{name} at {r['version'][:12]}")
sys.exit(1 if bad else 0)
EOF

settings="${VIVADO_SETTINGS:-$HOME/Documents/Xilinx/Vivado/2024.2/settings64.sh}"
if [ -f "$settings" ]; then ok "Vivado settings at $settings"
else warn "Vivado settings not found at $settings. build, program and the bus test need Vivado 2024.2: set VIVADO_SETTINGS"; fi

bit="build/nvfp4_tsi_50mhz/nvfp4_tsi_50mhz.bit"
if [ -f "$bit" ]; then ok "bitstream $bit"
else warn "no tensor-core bitstream. Run: pixi run build"; fi

port="$(ls /dev/serial/by-id/usb-Xilinx_SophiaLake_Bringup_Platform_*-if01-port0 2>/dev/null | head -1)"
if [ -z "$port" ]; then warn "SophiaLake USB UART not found. Connect the board for program, real and transformer"
elif [ -r "$port" ] && [ -w "$port" ]; then ok "board UART $port"
else warn "no read/write permission on $port. Add your user to the dialout group"; fi

gpu_host="${GPU_HOST:-cell}"
if ssh -o BatchMode=yes -o ConnectTimeout=5 "$gpu_host" test -x /usr/local/cuda-13.2/bin/nvcc 2>/dev/null; then
    ok "GPU host $gpu_host with CUDA 13.2"
else
    warn "cannot reach $gpu_host with CUDA 13.2 over ssh. Only the --gpu checks need it"
fi

exit $status
