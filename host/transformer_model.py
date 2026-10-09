"""Independent quantized transformer reference with explicit FP32 reduction order."""
import numpy as np

from nvfp4_model import tile
from quantization import quantize_rows
from transformer import ACT_INVERSE, P_INVERSE, EPSILON


def qmatmul(a, b, inverse_a=ACT_INVERSE, inverse_b=ACT_INVERSE):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    ac, sa = quantize_rows(a, inverse_a)
    bc, sb = quantize_rows(b.T, inverse_b)
    initial = [[0] * b.shape[1] for _ in range(len(a))]
    result = tile(ac.tolist(), bc.tolist(), sa.tolist(), sb.tolist(), initial, overwrite=True)
    result = np.asarray(result, dtype=np.uint32).view(np.float32)
    return np.float32(result * np.float32(1 / (inverse_a * inverse_b)))


def rmsnorm(x, gamma):
    result = np.empty_like(x)
    for row, values in enumerate(x):
        total = np.float32(0.)
        for start in range(0, len(values), 8):
            subtotal = np.float32(0.)
            for v in np.float32(values[start:start + 8] * values[start:start + 8]):
                subtotal = np.float32(subtotal + v)
            total = np.float32(total + subtotal)
        variance = np.float32(np.float32(total * np.float32(1 / len(values))) + np.float32(EPSILON))
        factor = np.float32(1 / np.sqrt(float(variance)))
        result[row] = np.float32(np.float32(values * gamma) * factor)
    return result


def flash_attention(q, k, v, causal):
    output = np.zeros_like(q)
    factor = np.float32(np.log2(np.e) / np.sqrt(q.shape[1]))
    for query_base in range(0, len(q), 2):
        maximum = np.full(2, -np.inf, dtype=np.float32)
        denominator = np.zeros(2, dtype=np.float32)
        numerator = np.zeros((2, q.shape[1]), dtype=np.float32)
        for key_base in range(0, len(k), 8):
            scores = qmatmul(q[query_base:query_base + 2], k[key_base:key_base + 8].T)
            probabilities = np.zeros((2, 32), dtype=np.float32)
            alpha = np.zeros(2, dtype=np.float32)
            for row in range(2):
                active = [lane for lane in range(8) if not causal or key_base + lane <= query_base + row]
                m = max([maximum[row], *(scores[row, lane] for lane in active)])
                alpha[row] = np.float32(0.) if denominator[row] == 0 else np.float32(
                    np.exp2(np.float32(np.float32(maximum[row] - m) * factor)))
                subtotal = np.float32(0.)
                for lane in active:
                    p = np.float32(np.exp2(np.float32(np.float32(scores[row, lane] - m) * factor)))
                    probabilities[row, lane] = p
                    subtotal = np.float32(subtotal + p)
                denominator[row] = np.float32(np.float32(alpha[row] * denominator[row]) + subtotal)
                maximum[row] = m
            padded_v = np.zeros((32, v.shape[1]), dtype=np.float32)
            padded_v[:8] = v[key_base:key_base + 8]
            product = qmatmul(probabilities, padded_v, P_INVERSE, ACT_INVERSE)
            numerator = np.float32(np.float32(numerator * alpha[:, None]) + product)
        output[query_base:query_base + 2] = np.float32(numerator * np.float32(1 / denominator)[:, None])
    return output


def transformer(x, weights, gamma1, gamma2, causal=True):
    stages = {}
    stages['norm1'] = rmsnorm(x, gamma1)
    for name in ('q', 'k', 'v'):
        stages[name] = qmatmul(stages['norm1'], weights[name])
    stages['attention'] = flash_attention(stages['q'], stages['k'], stages['v'], causal)
    stages['residual'] = np.float32(qmatmul(stages['attention'], weights['o']) + x)
    stages['norm2'] = rmsnorm(stages['residual'], gamma2)
    gate = qmatmul(stages['norm2'], weights['gate'])
    stages['up'] = qmatmul(stages['norm2'], weights['up'])
    exponent = np.float32(gate * np.float32(-np.log2(np.e)))
    denominator = np.float32(np.float32(np.exp2(exponent)) + np.float32(1.))
    silu = np.float32(gate * np.float32(1 / denominator))
    stages['gate'] = np.float32(silu * stages['up'])
    stages['output'] = np.float32(qmatmul(stages['gate'], weights['down']) + stages['residual'])
    return stages
