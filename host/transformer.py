"""Compile a pre-norm RMSNorm, FlashAttention, SwiGLU transformer layer into a kernel program.

The host quantizes static weights and uploads FP32 inputs. All activation quantization,
online attention state, normalization and FFN arithmetic execute in the FPGA kernel.
The shape is fixed: one head, D=32, FFN=64 and S=16 (two eight-key attention tiles).
"""
import struct

import numpy as np

from flash_vector import ADD, MUL, SCALE, AXPY, SUM, SOFTMAX, NORMALIZE, RSQRT, SILU, QUANT, command_beat
from kernel import Program, COPY, FILL, SCRATCH_BASE, SCRATCH_BYTES
from quantization import quantize_rows
from nvfp4_tc import pack_codes

VECTOR, TC = 0x10010000, 0x10000000
ACT_INVERSE = 64.
P_INVERSE = 2688.
EPSILON = 1e-5


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


class TransformerKernel:
    def __init__(self, x, weights, gamma1, gamma2, causal=True):
        self.x = np.asarray(x, dtype=np.float32)
        if self.x.shape != (16, 32):
            raise ValueError('the transformer kernel supports S=16, D=32 and one attention head')
        shapes = {'q': (32, 32), 'k': (32, 32), 'v': (32, 32), 'o': (32, 32),
                  'gate': (32, 64), 'up': (32, 64), 'down': (64, 32)}
        self.weights = {name: np.asarray(weights[name], dtype=np.float32) for name in shapes}
        if any(self.weights[name].shape != shape for name, shape in shapes.items()):
            raise ValueError(f'weight shapes must be {shapes}')
        self.gamma1 = np.asarray(gamma1, dtype=np.float32)
        self.gamma2 = np.asarray(gamma2, dtype=np.float32)
        if self.gamma1.shape != (32,) or self.gamma2.shape != (32,):
            raise ValueError('RMSNorm gamma vectors must have 32 elements')
        if not all(np.isfinite(a).all() for a in [self.x, self.gamma1, self.gamma2, *self.weights.values()]):
            raise ValueError('transformer inputs and parameters must be finite')
        self.causal = causal
        self.image = bytearray()
        self.addresses = {}
        self.program = Program()
        self.label_number = 0
        self.input = self.allocate('input', self.x.nbytes, self.x.tobytes())
        self.gamma1_address = self.allocate('gamma1', 128, self.gamma1.tobytes())
        self.gamma2_address = self.allocate('gamma2', 128, self.gamma2.tobytes())
        self.epsilon = self.allocate('epsilon', 32, struct.pack('<8f', *([EPSILON] * 8)))
        self.temp = self.allocate('scalar_temp', 32)
        self.weight_packed = {}
        for name, weight in self.weights.items():
            codes, scales = quantize_rows(weight.T, ACT_INVERSE)
            packed = b''.join(pack_codes(row.tolist()).ljust(128, b'\0') for row in codes)
            scale_bytes = b''.join(bytes(row).ljust(16, b'\0') for row in scales)
            self.weight_packed[name] = (self.allocate(name + '_codes', len(packed), packed),
                                        self.allocate(name + '_scales', len(scale_bytes), scale_bytes))
        self.compile()

    def allocate(self, name, size, initial=None):
        offset = (len(self.image) + 31) & ~31
        if offset + size > SCRATCH_BYTES:
            raise ValueError('transformer buffers exceed scratch memory')
        self.image.extend(bytes(offset + size - len(self.image)))
        if initial is not None:
            if len(initial) != size:
                raise ValueError('initial buffer size mismatch')
            self.image[offset:offset + size] = initial
        address = SCRATCH_BASE + offset
        self.addresses[name] = address
        return address

    def loop_label(self, prefix):
        self.label_number += 1
        return f'{prefix}_{self.label_number}'

    def vector(self, operation, x, destination, groups=1, y=0, scalar=1., mask=255, patch=()):
        """Run one vector command on groups of 8 FP32 elements and wait for it. The command beat
        is a scratch buffer; each (address, word) in patch first copies a scratch word into it."""
        beat = command_beat(operation, x, y, destination, groups, mask, bits(scalar))
        command = self.allocate(f'vector_command_{len(self.addresses)}', 32, struct.pack('<8I', *beat))
        for source, word in patch:
            self.program.copy(source, command + 4 * word, 1)
        self.program.copy(command, VECTOR + 0x20, 8)
        self.program.wait(VECTOR + 8)

    def elementwise(self, operation, x, destination, elements, y=0, scalar=1., scalar_address=None):
        self.vector(operation, x, destination, elements // 8, y=y, scalar=scalar,
                    patch=[(scalar_address, 4)] if scalar_address else [])

    def rmsnorm(self, x, destination, gamma):
        squares = self.allocate(f'norm_squares_{x:x}', 128)
        for row in range(16):
            source, output = x + 128 * row, destination + 128 * row
            self.vector(MUL, source, squares, 4, y=source)
            self.vector(SUM, squares, squares, 4)  # REDUCTION: the 4 group sums, left to right
            self.program.copy(VECTOR + 0x10, self.temp, 1)
            self.vector(SCALE, self.temp, self.temp, scalar=1 / 32, mask=1)
            self.vector(ADD, self.temp, self.temp, y=self.epsilon, mask=1)
            self.vector(RSQRT, self.temp, self.temp, mask=1)
            self.vector(MUL, source, output, 4, y=gamma)
            self.vector(SCALE, output, output, 4, patch=[(self.temp, 4)])

    def quant_buffers(self, name, rows):
        return self.allocate(name + '_codes', rows * 128), self.allocate(name + '_scales', rows * 16)

    def quantize(self, x, packed, rows, width, inverse=ACT_INVERSE):
        """NVFP4 rows of width elements: codes to packed[0], 128 B per row; scales to packed[1], 16 B per row."""
        self.vector(QUANT, x, packed[0], rows, y=packed[1], scalar=inverse, mask=width // 16)

    def matmul(self, a, b, destination, m, n, k, factor=1 / ACT_INVERSE**2):
        p = self.program
        p.copy(a[0], TC + 0x2000, m * 32)
        p.copy(a[1], TC + 0x4000, m * 4)
        p.copy(b[0], TC + 0x6000, n * 32)
        p.copy(b[1], TC + 0x8000, n * 4)
        p.write(TC + 0x14, 0)
        p.write(TC + 0x18, m << 24 | n << 16 | k)
        p.write(TC + 0xc, 3)
        p.wait(TC + 0x10)
        p.movi(0, TC + 0xc000)
        p.movi(2, destination)
        p.movi(3, m)
        label = self.loop_label('matrix_result')
        p.label(label)
        p.emit(COPY, rs1=0, rs2=2, immediate=n)
        p.addi(0, 0, 256)
        p.addi(2, 2, n * 4)
        p.addi(3, 3, -1)
        p.jnz(3, label)
        self.elementwise(SCALE, destination, destination, m * n, scalar=factor)

    def causal_mask(self, query, key_base):
        """Lanes of the eight keys from key_base that query may attend to."""
        if not self.causal:
            return 255
        return (1 << min(8, max(0, query + 1 - key_base))) - 1

    def attention(self, q, k, v, destination):
        p = self.program
        qp, kp = self.quant_buffers('q', 16), self.quant_buffers('k', 16)
        self.quantize(q, qp, 16, 32)
        self.quantize(k, kp, 16, 32)
        vt = self.allocate('v_transpose', 32 * 32 * 4)
        vb = []
        for block in range(2):
            packed = self.quant_buffers(f'v_tile{block}', 32)
            vb.append(packed)
            p.movi(0, v + block * 8 * 128)
            p.movi(1, vt)
            p.movi(3, 32)
            label = self.loop_label('v_transpose')
            p.label(label)
            for key in range(8):
                p.load(4, 0, key * 128)
                p.store(1, 4, key * 4)
            p.addi(0, 0, 4)
            p.addi(1, 1, 128)
            p.addi(3, 3, -1)
            p.jnz(3, label)
            self.quantize(vt, packed, 32, 32)
        scores = self.allocate('score_tile', 2 * 8 * 4)
        probabilities = self.allocate('probability_tile', 2 * 32 * 4)
        pp = self.quant_buffers('probability', 2)
        product = self.allocate('pv_tile', 2 * 32 * 4)
        numerator = self.allocate('online_numerator', 2 * 32 * 4)
        maximum = self.allocate('online_max', 32)
        denominator = self.allocate('online_sum', 32)
        alpha = self.allocate('online_alpha', 32)
        scale = np.log2(np.e) / np.sqrt(32)
        for pair in range(8):  # query rows 2 pair and 2 pair + 1
            for row in range(2):
                p.write(maximum + row * 4, bits(-np.inf))
                p.write(denominator + row * 4, 0)
            p.movi(4, numerator)
            p.movi(5, 0)
            p.emit(FILL, rs1=4, rs2=5, immediate=64)
            for block in range(2):
                self.matmul((qp[0] + 256 * pair, qp[1] + 32 * pair),
                            (kp[0] + block * 8 * 128, kp[1] + block * 8 * 16), scores, 2, 8, 32)
                for row in range(2):
                    self.vector(SOFTMAX, scores + row * 32, probabilities + row * 128, scalar=scale,
                                mask=self.causal_mask(2 * pair + row, block * 8),
                                patch=[(maximum + row * 4, 5), (denominator + row * 4, 6)])
                    p.copy(VECTOR + 0x14, maximum + row * 4, 1)
                    p.copy(VECTOR + 0x18, denominator + row * 4, 1)
                    p.copy(VECTOR + 0x1c, alpha + row * 4, 1)
                self.quantize(probabilities, pp, 2, 32, P_INVERSE)
                self.matmul(pp, vb[block], product, 2, 32, 32, factor=1 / (P_INVERSE * ACT_INVERSE))
                for row in range(2):
                    self.elementwise(AXPY, numerator + row * 128, numerator + row * 128, 32,
                                     y=product + row * 128, scalar_address=alpha + row * 4)
            for row in range(2):
                self.elementwise(NORMALIZE, numerator + row * 128, numerator + row * 128,
                                 32, scalar_address=denominator + row * 4)
            p.copy(numerator, destination + 256 * pair, 64)

    def compile(self):
        norm1 = self.allocate('norm1', 16 * 32 * 4)
        self.rmsnorm(self.input, norm1, self.gamma1_address)
        norm_packed = self.quant_buffers('norm1', 16)
        self.quantize(norm1, norm_packed, 16, 32)
        qkv = {}
        for name in ('q', 'k', 'v'):
            qkv[name] = self.allocate(name, 16 * 32 * 4)
            self.matmul(norm_packed, self.weight_packed[name], qkv[name], 16, 32, 32)
        attention = self.allocate('attention', 16 * 32 * 4)
        self.attention(qkv['q'], qkv['k'], qkv['v'], attention)
        attention_packed = self.quant_buffers('attention', 16)
        self.quantize(attention, attention_packed, 16, 32)
        residual = self.allocate('residual', 16 * 32 * 4)
        self.matmul(attention_packed, self.weight_packed['o'], residual, 16, 32, 32)
        self.elementwise(ADD, residual, residual, 512, y=self.input)
        norm2 = self.allocate('norm2', 16 * 32 * 4)
        self.rmsnorm(residual, norm2, self.gamma2_address)
        norm_packed = self.quant_buffers('norm2', 16)
        self.quantize(norm2, norm_packed, 16, 32)
        gate = self.allocate('gate', 16 * 64 * 4)
        up = self.allocate('up', 16 * 64 * 4)
        self.matmul(norm_packed, self.weight_packed['gate'], gate, 16, 64, 32)
        self.matmul(norm_packed, self.weight_packed['up'], up, 16, 64, 32)
        self.elementwise(SILU, gate, gate, 1024)
        self.elementwise(MUL, gate, gate, 1024, y=up)
        hidden_packed = self.quant_buffers('hidden', 16)
        self.quantize(gate, hidden_packed, 16, 64)
        self.output = self.allocate('output', 16 * 32 * 4)
        self.matmul(hidden_packed, self.weight_packed['down'], self.output, 16, 32, 64)
        self.elementwise(ADD, self.output, self.output, 512, y=residual)
        self.program.halt()
