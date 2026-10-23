"""Regenerate synthetic codec goldens with pinned gguf-py, never with the new decoder.

Requires numpy and a checkout of llama.cpp at third_party/ggml/VERSION.txt. Only test
maintainers need those dependencies; normal C++ tests consume the checked-in vectors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np


REVISION = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
QUANTS_SHA256 = '2c927a1b3d9f0920dcf4007fb686e1b0999333e9f65ce43dcc689900c0beae8b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf-py', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('tests/hpu/fixtures'))
    args = parser.parse_args()
    module = args.gguf_py / 'gguf' / 'quants.py'
    if hashlib.sha256(module.read_bytes()).hexdigest() != QUANTS_SHA256:
        raise ValueError(f'oracle must match gguf-py at {REVISION}')
    sys.path.insert(0, str(args.gguf_py))
    from gguf import quants
    from gguf.constants import GGMLQuantizationType

    rng = np.random.default_rng(0x6A2D1)
    cases = []
    # Zero, signed zero, subnormals, min normal, normal, negative and max finite half scales.
    scales = [0, 0x8000, 1, 0x8001, 0x3ff, 0x400, 0x2401, 0x3c00, 0xbc00, 0x7bff]
    for kind, block_bytes in [(10, 84), (11, 110), (21, 110), (22, 82), (23, 136)]:
        for rows, columns in [(3, 512), (2, 4096)]:
            blocks = rows * columns // 256
            packed = rng.integers(0, 256, (blocks, block_bytes), dtype=np.uint8)
            for i, block in enumerate(packed):
                offset = 80 if kind == 10 else 108 if kind == 11 else 0
                block[offset:offset + 2] = np.frombuffer(struct.pack('<H', scales[i % len(scales)]), dtype=np.uint8)
                if kind == 10:
                    block[82:84] = np.frombuffer(struct.pack('<H', scales[(i + 3) % len(scales)]), dtype=np.uint8)
            fp32 = quants.dequantize(packed.reshape(rows, -1), GGMLQuantizationType(kind)).astype('<f4')
            bits = fp32.view('<u4')
            bf16 = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype('<u2')
            cases.append((kind, rows, columns, packed.tobytes(), fp32.tobytes(), bf16.tobytes()))

    data = bytearray(b'STHPUDQ1' + struct.pack('<I', len(cases)))
    for kind, rows, columns, packed, fp32, bf16 in cases:
        data.extend(struct.pack('<IIII', kind, rows, columns, len(packed)))
        data.extend(packed)
        data.extend(fp32)
        data.extend(bf16)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'packed_decode_vectors.bin').write_bytes(data)
    manifest = {
        'scope': 'Deterministic synthetic codes, no model weights; FP32 and BF16 RNE output.',
        'oracle_repository': 'ggml-org/llama.cpp', 'oracle_revision': REVISION,
        'oracle_path': 'gguf-py/gguf/quants.py', 'oracle_sha256': QUANTS_SHA256,
        'oracle_url': f'https://github.com/ggml-org/llama.cpp/blob/{REVISION}/gguf-py/gguf/quants.py',
        'generator': 'tools/hpu/generate_decode_vectors.py', 'seed': 0x6A2D1,
        'cases': len(cases), 'file_sha256': hashlib.sha256(data).hexdigest(), 'file_bytes': len(data),
        'formats': ['Q2_K', 'Q3_K', 'IQ3_S', 'IQ2_S', 'IQ4_XS'],
        'numpy_version': np.__version__,
    }
    (args.out_dir / 'packed_decode_vectors.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
