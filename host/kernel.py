"""Assembler and TSI loader for the on-device MMIO kernel engine."""
import struct
import time

KERNEL_BASE = 0x10020000
SCRATCH_BASE = 0x20000000
SCRATCH_BYTES = 128 * 1024
PROGRAM = 0x1000
CAPACITY = 8192
HALT, MOVI, ADDI, ADD, LOAD, STORE, JMP, JNZ, COPY, FILL, WAIT, SHL, SHR, AND, SHL_REG, SLTU = range(16)


class Program:
    def __init__(self):
        self.instructions = []
        self.labels = {}

    def emit(self, opcode, rd=0, rs1=0, rs2=0, immediate=0):
        if not (0 <= opcode < 256 and all(0 <= r < 16 for r in (rd, rs1, rs2))):
            raise ValueError('instruction opcode or register out of range')
        if len(self.instructions) >= CAPACITY:
            raise ValueError('program exceeds instruction memory')
        self.instructions.append((opcode, rd, rs1, rs2, immediate))

    def label(self, name):
        if name in self.labels:
            raise ValueError(f'duplicate label {name}')
        self.labels[name] = len(self.instructions)

    def movi(self, register, value):
        self.emit(MOVI, rd=register, immediate=value)

    def addi(self, destination, source, value):
        self.emit(ADDI, rd=destination, rs1=source, immediate=value)

    def load(self, destination, address_register, offset=0):
        self.emit(LOAD, rd=destination, rs1=address_register, immediate=offset)

    def store(self, address_register, value_register, offset=0):
        self.emit(STORE, rs1=address_register, rs2=value_register, immediate=offset)

    def jnz(self, register, label):
        self.emit(JNZ, rs1=register, immediate=label)

    def write(self, address, value):
        """Emit a constant MMIO write; reserves registers 12 and 13."""
        self.movi(12, address)
        self.movi(13, value)
        self.store(12, 13)

    def copy(self, source, destination, words):
        """Emit a word transfer; reserves registers 10 and 11. Overlap is unsupported."""
        if words < 0 or source % 4 or destination % 4:
            raise ValueError('copy requires aligned addresses and a nonnegative word count')
        if words and max(source, destination) + 4 * words > 1 << 32:
            raise ValueError('copy wraps the address space')
        self.movi(10, source)
        self.movi(11, destination)
        self.emit(COPY, rs1=10, rs2=11, immediate=words)

    def wait(self, address, busy_mask=1, error_mask=2):
        if address % 4 or not (0 <= busy_mask <= 0xffff and 0 <= error_mask <= 0xffff):
            raise ValueError('WAIT requires an aligned address and 16-bit masks')
        self.movi(12, address)
        self.emit(WAIT, rs1=12, immediate=error_mask << 16 | busy_mask)

    def halt(self):
        self.emit(HALT)

    def encode(self):
        words = []
        for opcode, rd, rs1, rs2, immediate in self.instructions:
            if isinstance(immediate, str):
                if immediate not in self.labels:
                    raise ValueError(f'undefined label {immediate}')
                immediate = self.labels[immediate]
            if not -(1 << 31) <= immediate < 1 << 32:
                raise ValueError('immediate does not fit 32 bits')
            words.append(opcode << 56 | rd << 52 | rs1 << 48 | rs2 << 44 | (immediate & 0xffffffff))
        return struct.pack(f'<{len(words)}Q', *words)


class KernelEngine:
    def __init__(self, tsi):
        self.tsi = tsi
        if tsi.read_word(KERNEL_BASE) != 0x4b45524e:
            raise RuntimeError('no kernel engine in this bitstream')

    def upload(self, program):
        """Load program and set the entry point to its first instruction."""
        if self.tsi.read_word(KERNEL_BASE + 8) & 1:
            raise RuntimeError('kernel engine is busy')
        raw = program.encode()
        if not raw:
            raise ValueError('the program is empty')
        for offset in range(0, len(raw), 1024):
            self.tsi.write_bytes(KERNEL_BASE + PROGRAM + offset, raw[offset:offset + 1024])
        self.tsi.write_word(KERNEL_BASE + 0x10, 0)
        self.tsi.write_word(KERNEL_BASE + 0x14, len(raw) // 8)

    def start(self):
        self.tsi.write_word(KERNEL_BASE + 0x0c, 1)

    def wait(self, timeout=30):
        deadline = time.monotonic() + timeout
        while (status := self.tsi.read_word(KERNEL_BASE + 8)) & 1:
            if time.monotonic() > deadline:
                raise TimeoutError('kernel execution timed out')
        if status & 2:
            fault = self.tsi.read_word(KERNEL_BASE + 0x20)
            pc = self.tsi.read_word(KERNEL_BASE + 0x1c)
            raise RuntimeError(f'kernel fault {fault} at PC {pc}')
        if not status & 4:
            raise RuntimeError('kernel has not halted')
        return self.tsi.read_word(KERNEL_BASE + 0x18)

    def run(self, program, timeout=30):
        self.upload(program)
        self.start()
        return self.wait(timeout)
