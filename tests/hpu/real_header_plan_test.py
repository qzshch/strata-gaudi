"""Plan the public GLM IQ3 tensor directory using sparse, weight-free files."""
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


EXE = Path(sys.argv.pop(1)).resolve()
FIXTURE = Path(__file__).parent / 'fixtures/glm53_iq3_inventory.json'
GIB = 1 << 30


def string(value):
    encoded = value.encode('utf-8')
    return struct.pack('<Q', len(encoded)) + encoded


def sparse_file(path, header, length):
    with path.open('wb') as stream:
        if os.name == 'nt':
            # Require successful sparse marking before extending a 40+ GiB file.
            # No filesystem command, driver installation or administrator rights.
            import msvcrt
            from ctypes import wintypes
            control = ctypes.WinDLL('kernel32', use_last_error=True).DeviceIoControl
            control.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            control.restype = wintypes.BOOL
            returned = wintypes.DWORD()
            if not control(msvcrt.get_osfhandle(stream.fileno()), 0x900C4,
                           None, 0, None, 0, ctypes.byref(returned), None):
                raise OSError(ctypes.get_last_error(), 'cannot create sparse GGUF test fixture')
        stream.write(header)
        stream.flush()
        if os.name == 'nt':
            # The Windows CRT's _chsize_s (used by Python truncate) writes zeroes
            # when growing files, even after sparse marking. Extend via Win32.
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            seek = kernel.SetFilePointerEx
            seek.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                             ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD]
            seek.restype = wintypes.BOOL
            end = kernel.SetEndOfFile
            end.argtypes = [wintypes.HANDLE]
            end.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(stream.fileno())
            if not seek(handle, length, None, 0) or not end(handle):
                raise OSError(ctypes.get_last_error(), 'cannot extend sparse GGUF test fixture')
        else:
            os.ftruncate(stream.fileno(), length)
    if os.name == 'nt':
        allocated = ctypes.WinDLL('kernel32', use_last_error=True).GetCompressedFileSizeW
        allocated.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
        allocated.restype = ctypes.c_uint32
        high = ctypes.c_uint32()
        low = allocated(str(path), ctypes.byref(high))
        physical = (high.value << 32) | low
    else:
        physical = path.stat().st_blocks * 512
    if physical > max(1024 * 1024, len(header) + 65536):
        raise AssertionError(f'fixture was not sparse: {physical} physical bytes')


def write_shard(directory, source):
    metadata = source['metadata']
    tensors = source['tensors']
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += string(key)
        if isinstance(value, str):
            header += struct.pack('<I', 8) + string(value)
        elif key in ('split.no', 'split.count'):
            header += struct.pack('<IH', 2, value)
        elif key == 'split.tensors.count':
            header += struct.pack('<Ii', 5, value)
        else:
            header += struct.pack('<II', 4, value)
    for tensor in tensors:
        shape = tensor['shape']
        header += string(tensor['name']) + struct.pack('<I', len(shape))
        header += struct.pack('<' + 'Q' * len(shape), *shape)
        header += struct.pack('<IQ', tensor['ggml_type'], tensor['offset'])
    header += bytes(-len(header) % 32)
    # Tensor-relative offsets retain the real checkpoint's gaps and large extents;
    # our smaller metadata directory has a different aligned data start.
    data_bytes = source['source_file_bytes'] - source['source_data_offset']
    path = directory / source['file']
    sparse_file(path, header, len(header) + data_bytes)
    return path


class PublicGeometryTest(unittest.TestCase):
    def test_all_real_experts_fit_disjoint_default_banks(self):
        fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
        # TemporaryDirectory owns only files created by this test. All native
        # consumers are reaped before cleanup, including timeout/failure paths.
        with tempfile.TemporaryDirectory(prefix='strata-real-geometry-', dir=EXE.parent) as temp:
            directory = Path(temp)
            shards = [write_shard(directory, source) for source in fixture['files']]
            placements = directory / 'placements.csv'
            command = [str(EXE), '--placements', str(placements), *map(str, shards)]
            started = datetime.now(timezone.utc).isoformat()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(json.dumps({'purpose': 'real GLM header geometry planning', 'pid': process.pid,
                              'started_utc': started, 'command': str(EXE)}), flush=True)
            try:
                stdout, stderr = process.communicate(timeout=20)
                self.assertEqual(process.returncode, 0, stderr.decode(errors='replace'))
                plan = json.loads(stdout)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            self.assertFalse(plan['hpu_inference_available'])
            self.assertEqual(plan['trunk_layers'], 45)
            self.assertEqual(plan['experts_per_layer'], 288)
            self.assertEqual(plan['device_experts'] + plan['host_experts'] + plan['file_experts'],
                             fixture['expected_trunk_experts'])
            with placements.open(newline='', encoding='utf-8') as stream:
                entries = list(csv.DictReader(stream))
            ids = {(int(row['layer']), int(row['expert'])) for row in entries}
            self.assertEqual(ids, {(layer, expert) for layer in range(3, 45) for expert in range(288)})
            self.assertEqual(len(entries), len(ids))
            self.assertEqual(sum(int(row['packed_bytes']) for row in entries),
                             fixture['expected_trunk_expert_bytes'])
            self.assertEqual(plan['file_experts'], 0)
            self.assertLessEqual(plan['device_allocated_bytes'] + plan['device_staging_bytes'] + 27 * GIB,
                                 96 * GIB)
            self.assertLessEqual(plan['host_allocated_bytes'] + plan['host_staging_bytes'] + 24 * GIB,
                                 64 * GIB)
            self.assertGreater(plan['skipped_mtp_bytes'], 0)
            print(json.dumps({'source_revision': fixture['source_revision'], 'geometry_plan': plan}), flush=True)


if __name__ == '__main__':
    unittest.main()
