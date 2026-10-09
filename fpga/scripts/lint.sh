#!/usr/bin/env bash
# Lint nvfp4_tsi_soc (fpga/designs/nvfp4_tsi without the board wrapper) with Verilator -Wall.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
mapfile -t files < <(grep -v -E '^(#|\+|$)' fpga/designs/nvfp4_tsi/sources.f | grep -v -E 'board_clk_rst|sophialake_tsi_top')
mapfile -t incs < <(sed -n 's/^+incdir+/-I/p' fpga/designs/nvfp4_tsi/sources.f)
verilator --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY \
    "${incs[@]}" --top-module nvfp4_tsi_soc fpga/scripts/lint.vlt "${files[@]}"
echo "lint: clean"
