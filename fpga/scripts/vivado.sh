#!/usr/bin/env bash
# Run a Vivado batch Tcl script in build/logs with the Vivado environment.
#   fpga/scripts/vivado.sh <script.tcl> [tclargs...]
# VIVADO_SETTINGS overrides the path of the Vivado 2024.2 settings64.sh.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
settings="${VIVADO_SETTINGS:-$HOME/Documents/Xilinx/Vivado/2024.2/settings64.sh}"
if [ ! -f "$settings" ]; then
    echo "error: Vivado settings not found at $settings. Set VIVADO_SETTINGS to the settings64.sh of Vivado 2024.2." >&2
    exit 1
fi

script="$(realpath "$1")"
shift
log="$(basename "$script" .tcl)_$(basename "${1:-run}" .bit).log"

mkdir -p "$root/build/logs"
cd "$root/build/logs"
set +u
# shellcheck disable=SC1090
source "$settings" >/dev/null
set -u
exec vivado -mode batch -nojournal -log "$log" -source "$script" -tclargs "$@"
