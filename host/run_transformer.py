"""Upload and execute the complete transformer kernel over direct MMIO or UART TSI."""
import argparse
import json

import numpy as np

from harness import ROOT, open_tsi
from kernel import KernelEngine, SCRATCH_BASE
from transformer import TransformerKernel
from transformer_model import transformer


def fixture(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .4, (16, 32)).astype(np.float32)
    weights = {name: rng.normal(0, .08, (64 if name == 'down' else 32,
                                       64 if name in ('gate', 'up') else 32)).astype(np.float32)
               for name in ('q', 'k', 'v', 'o', 'gate', 'up', 'down')}
    gamma1 = rng.uniform(.8, 1.2, 32).astype(np.float32)
    gamma2 = rng.uniform(.8, 1.2, 32).astype(np.float32)
    return x, weights, gamma1, gamma2


def execute(tsi, seed, causal):
    x, weights, gamma1, gamma2 = fixture(seed)
    kernel = TransformerKernel(x, weights, gamma1, gamma2, causal)
    expected = transformer(x, weights, gamma1, gamma2, causal)
    print(f'Compiled {len(kernel.program.instructions)} instructions; {len(kernel.image)} scratch bytes', flush=True)
    for offset in range(0, len(kernel.image), 1024):
        tsi.write_bytes(SCRATCH_BASE + offset, bytes(kernel.image[offset:offset + 1024]))
    engine = KernelEngine(tsi)
    cycles = engine.run(kernel.program, timeout=120)
    errors = {}
    for name, want in expected.items():
        raw = b''.join(tsi.read_bytes(kernel.addresses[name] + offset, min(1024, want.nbytes - offset))
                       for offset in range(0, want.nbytes, 1024))
        actual = np.frombuffer(raw, dtype='<f4').reshape(want.shape)
        difference = np.abs(actual.astype(np.float64) - want.astype(np.float64))
        errors[name] = float(difference.max())
        if not np.isfinite(actual).all() or errors[name] > 3e-5:
            index = np.unravel_index(np.argmax(difference), want.shape)
            raise AssertionError(f'{name}: max error {errors[name]} at {index}; FPGA {actual[index]}, reference {want[index]}')
    print(f'Transformer seed={seed} causal={causal}: PASS, {cycles} cycles, output max abs error {errors["output"]:.3g}', flush=True)
    return dict(seed=seed, causal=causal, instructions=len(kernel.program.instructions), scratch_bytes=len(kernel.image),
                cycles=cycles, stage_max_abs_error=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', choices=('direct', 'tsi-sim', 'fpga'), default='direct')
    parser.add_argument('--port')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--noncausal', action='store_true')
    args = parser.parse_args()
    tsi = open_tsi(args.transport, args.port)
    try:
        report = execute(tsi, args.seed, not args.noncausal)
        report['transport'] = args.transport
        path = ROOT / f'results/transformer_{args.transport}_{args.seed}_{"dense" if args.noncausal else "causal"}.json'
        path.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        tsi.close()


if __name__ == '__main__':
    main()
