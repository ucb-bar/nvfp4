"""Verilator builds: one design's dut_wrap as a bit-exact RTL reference, and the shared build step."""
import fcntl
import subprocess

from harness import ROOT


def read_filelist(path):
    files, incs, defs = [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("+incdir+"):
            incs.append(ROOT / line[8:])
        elif line.startswith("+define+"):
            defs.append(line[8:])
        elif line.startswith("+top+"):
            continue
        else:
            files.append(ROOT / line)
    return files, incs, defs


def verilate(work, top, filelist, files, cpp, options=()):
    """Return the executable work/<cpp stem> of a Verilator C++ model of top.

    files are the RTL sources; include directories and defines come from filelist. The build runs
    only when filelist, a source, an include-directory header or cpp is newer than the executable.
    A lock file serializes concurrent builds of the same work directory.
    """
    _, incs, defs = read_filelist(filelist)
    work.mkdir(parents=True, exist_ok=True)
    exe = work / cpp.stem
    with (work / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        headers = [h for d in incs for h in d.glob("*.svh")]
        if exe.exists() and exe.stat().st_mtime > max(p.stat().st_mtime for p in [filelist, *files, *headers, cpp]):
            return exe
        cmd = ["verilator", "--cc", "--exe", "--build", "-O3", "-j", "0", "--top-module", top,
               "-Wno-fatal", "-Wno-lint", "-Wno-style", "-Wno-UNOPTFLAT", "--Mdir", str(work), "-o", exe.name,
               *options, *(f"-I{i}" for i in incs), *(f"-D{d}" for d in defs), *map(str, files), str(cpp)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"verilator build failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return exe


def run(design, cases):
    """Return [(out_bytes, latency)] from the RTL for each case."""
    dsgn = ROOT / "fpga/designs" / design.name
    filelist = dsgn / "sources.f"
    files = [dsgn / "dut_cfg_pkg.sv", *read_filelist(filelist)[0], dsgn / "dut_wrap.sv"]
    exe = verilate(ROOT / "build/vl_ref" / design.name, "dut_wrap", filelist, files, ROOT / "fpga/sim/vl_ref.cpp",
                   ["-CFLAGS", f"-O2 -DIN_BYTES={design.in_bytes} -DOUT_BYTES={design.out_bytes}"])
    frames = "".join(design.pack(c).hex() + "\n" for c in cases)
    p = subprocess.run([str(exe)], input=frames, capture_output=True, text=True, check=True)
    rows = [l.split() for l in p.stdout.splitlines()]
    return [(bytes.fromhex(h), int(lat)) for h, lat in rows]
