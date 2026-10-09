"""Run vector ready/valid, capture, empty-mask and reset tests with Verilator."""
import subprocess

from harness import ROOT


def main():
    work = ROOT / 'build/tb_vector'
    work.mkdir(parents=True, exist_ok=True)
    sources = ['fpga/compute/fp32_pkg.sv', 'fpga/compute/vector_pkg.sv', 'fpga/compute/fp32_alu.sv',
               'fpga/compute/vector_lane.sv', 'fpga/compute/flash_vector_unit.sv', 'fpga/sim/tb_vector.sv']
    command = ['verilator', '--binary', '--timing', '--top-module', 'tb_vector', '-Wno-fatal',
               f'-I{ROOT / "fpga/compute"}', '--Mdir', str(work), '-o', 'tb_vector',
               *[str(ROOT / p) for p in sources]]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stdout[-2000:] + process.stderr[-2000:])
    subprocess.run([str(work / 'tb_vector')], check=True)


if __name__ == '__main__':
    main()
