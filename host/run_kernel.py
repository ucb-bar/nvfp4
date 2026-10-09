"""Execute kernel programs against direct RTL, UART TSI RTL, or the FPGA."""
import argparse
import struct

import flash_vector
from harness import open_tsi
from kernel import (ADD, AND, COPY, FILL, JMP, KERNEL_BASE, SCRATCH_BASE, SHL, SHL_REG, SHR, SLTU, KernelEngine,
                    Program)

VECTOR = 0x10010000
TC = 0x10000000


def check_execution(tsi):
    engine = KernelEngine(tsi)
    source, output = SCRATCH_BASE, SCRATCH_BASE + 0x100
    x = [float(i - 3) for i in range(8)]
    y = [float(i * 2 + 1) for i in range(8)]
    command = source + 0x80
    tsi.write_bytes(source, struct.pack('<16f', *x, *y))
    beat = flash_vector.command_beat(flash_vector.ADD, source, source + 32, output, 1)
    tsi.write_bytes(command, struct.pack('<8I', *beat))
    program = Program()
    program.copy(command, VECTOR + 0x20, 8)  # one beat, start word last
    program.wait(VECTOR + 8)
    program.movi(0, 5)
    program.movi(1, 0)
    program.label('loop')
    program.addi(1, 1, 3)
    program.addi(0, 0, -1)
    program.jnz(0, 'loop')
    program.movi(2, output + 32)
    program.store(2, 1)
    program.load(3, 2)
    program.movi(4, output + 36)
    program.emit(FILL, rs1=4, rs2=3, immediate=3)
    program.emit(COPY, rs1=4, rs2=4, immediate=0)
    # Unaligned copies: 21 words from word 1 of a beat to word 6, and back to word 0.
    words = list(range(1000, 1021))
    tsi.write_bytes(source + 0x404, struct.pack('<21I', *words))
    program.copy(source + 0x404, output + 0x118, 21)
    program.copy(output + 0x118, output + 0x200, 21)
    program.halt()
    cycles = engine.run(program)
    actual = struct.unpack('<8f4I', tsi.read_bytes(output, 48))
    assert actual == tuple(a + b for a, b in zip(x, y)) + (15,) * 4, actual
    assert tsi.read_word(KERNEL_BASE + 0x10c) == 15
    assert struct.unpack('<21I', tsi.read_bytes(output + 0x118, 84)) == tuple(words)
    assert struct.unpack('<21I', tsi.read_bytes(output + 0x200, 84)) == tuple(words)
    print(f'Kernel vector ADD, DMA, unaligned COPY, loop, LOAD/STORE and FILL: PASS ({cycles} cycles)')

    # Register arithmetic. Bit 31 of r1 tells a logical SHR from an arithmetic one and SLTU from a
    # signed compare. r2[4:0] = 4 differs from r2 = 36, so SHL_REG must take its shift from rs2[4:0].
    r1, r2 = 0x8000_0013, 36
    program = Program()
    program.movi(1, r1)
    program.movi(2, r2)
    program.emit(ADD, rd=3, rs1=1, rs2=2)
    program.emit(SHL, rd=4, rs1=1, immediate=5)
    program.emit(SHR, rd=5, rs1=1, immediate=3)
    program.emit(AND, rd=6, rs1=1, immediate=0x8000_00f1)
    program.emit(SHL_REG, rd=7, rs1=1, rs2=2)
    program.emit(SLTU, rd=8, rs1=2, rs2=1)
    program.emit(SLTU, rd=9, rs1=1, rs2=2)
    program.movi(0, output + 0x380)
    for register in range(3, 10):
        program.store(0, register, 4 * (register - 3))
    program.halt()
    engine.run(program)
    want = (r1 + r2, r1 << 5 & 0xffff_ffff, r1 >> 3, r1 & 0x8000_00f1, r1 << 4 & 0xffff_ffff, 1, 0)
    actual = struct.unpack('<7I', tsi.read_bytes(output + 0x380, 28))
    assert actual == want, [f'{v:08x}' for v in actual]
    print('Kernel ADD, SHL, SHR, AND, SHL_REG and SLTU: PASS')

    # Constant NVFP4 values 1 * 1: the kernel transfers, launches and returns a full 2x8 tile.
    a, b, scales = source + 0x200, source + 0x220, source + 0x240
    tsi.write_bytes(a, bytes([0x22]) * 16)
    tsi.write_bytes(b, bytes([0x22]) * 16)
    tsi.write_word(scales, 0x3838)
    program = Program()
    for row in range(2):
        program.copy(a, TC + 0x2000 + row * 128, 4)
        program.copy(scales, TC + 0x4000 + row * 16, 1)
    for column in range(8):
        program.copy(b, TC + 0x6000 + column * 128, 4)
        program.copy(scales, TC + 0x8000 + column * 16, 1)
    program.write(TC + 0x14, 0)
    program.write(TC + 0x18, 2 << 24 | 8 << 16 | 32)
    program.write(TC + 0x0c, 3)
    program.wait(TC + 0x10)
    for row in range(2):
        program.copy(TC + 0xc000 + row * 256, output + row * 32, 8)
    program.halt()
    cycles = engine.run(program)
    actual = struct.unpack('<16f', tsi.read_bytes(output, 64))
    assert actual == (32.,) * 16, actual
    print(f'Kernel NVFP4 matmul and result DMA: PASS ({cycles} cycles)')

    # Faults must terminate and leave the engine reusable for the next upload.
    for kind in ('illegal', 'pc', 'bus', 'wait'):
        program = Program()
        if kind == 'illegal': program.emit(255)
        elif kind == 'pc': program.emit(JMP, immediate=4096)
        elif kind == 'bus': program.copy(0x30000000, output, 1)
        else:
            tsi.write_word(source, 2)
            program.wait(source)
        program.halt()
        engine.upload(program)
        engine.start()
        try:
            engine.wait()
        except RuntimeError as error:
            expected = {'illegal': 1, 'pc': 2, 'bus': 3, 'wait': 4}[kind]
            assert f'kernel fault {expected} ' in str(error), error
        else:
            raise AssertionError(f'{kind} did not fault')
    program = Program()
    program.halt()
    engine.run(program)
    print('Kernel fault reporting and recovery: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', choices=('direct', 'tsi-sim', 'fpga'), default='direct')
    parser.add_argument('--port')
    args = parser.parse_args()
    tsi = open_tsi(args.transport, args.port)
    try:
        check_execution(tsi)
    finally:
        tsi.close()


if __name__ == '__main__':
    main()
