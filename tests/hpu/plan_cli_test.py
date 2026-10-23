"""Independent, tiny GGUF writers exercise real split directories; no model download/numpy."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

EXE = sys.argv.pop(1)


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def write(path, tensors, *, number=0, count=3, total=7, trunk=2, blocks=3):
    metadata = {
        "general.architecture": "glm5-next",
        "glm5-next.block_count": blocks,
        "glm5-next.trunk_block_count": trunk,
        "glm5-next.nextn_predict_layers": 1,
        "glm5-next.leading_dense_block_count": 1,
        "glm5-next.expert_count": 3,
        "glm5-next.embedding_length": 256,
        "split.no": number,
        "split.count": count,
        "split.tensors.count": total,
    }
    if number != 0:
        # The pinned gguf writer keeps model metadata in shard zero; later shards carry
        # only split metadata. A metadata-only first shard must still be supplied.
        metadata = {key: value for key, value in metadata.items() if key.startswith("split.")}
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += string(key)
        if isinstance(value, str):
            header += struct.pack("<I", 8) + string(value)
        elif key == "split.tensors.count":
            # llama.cpp's writer uses INT32 here (split.no/count are UINT16).
            header += struct.pack("<Ii", 5, value)
        elif key in ("split.no", "split.count"):
            header += struct.pack("<IH", 2, value)
        else:
            header += struct.pack("<IQ", 10, value)  # metadata U64
    directory, body = b"", b""
    # Independently specified GGML block byte sizes (not read from the C++ implementation).
    geometry = {0: (1, 4), 1: (1, 2), 10: (256, 84), 11: (256, 110),
                21: (256, 110), 22: (256, 82), 23: (256, 136)}
    for name, kind in tensors:
        shape = (256, 256, 3) if "exps.weight" in name else (256,)
        elements = 1
        for dimension in shape:
            elements *= dimension
        block, size = geometry[kind]
        directory += string(name) + struct.pack("<I", len(shape))
        directory += b"".join(struct.pack("<Q", d) for d in shape)
        directory += struct.pack("<IQ", kind, len(body))
        body += bytes([kind]) * (elements // block * size)
        body += b"\0" * (-len(body) % 32)
    header += directory
    path.write_bytes(header + b"\0" * (-len(header) % 32) + body)


class PlannerCli(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name) / f"shard{i}.gguf" for i in range(3)]

    def fixture(self, types=(21, 22, 23)):
        write(self.paths[0], [], number=0)
        write(self.paths[1], [("blk.1.ffn_gate_exps.weight", types[0]),
                              ("output_norm.weight", 0)], number=1)
        write(self.paths[2], [("blk.1.ffn_up_exps.weight", types[1]),
                              ("blk.1.ffn_down_exps.weight", types[2]),
                              ("blk.2.ffn_gate_exps.weight", 1),
                              ("blk.2.ffn_up_exps.weight", 1),
                              ("blk.2.ffn_down_exps.weight", 1)], number=2)

    def run_plan(self, paths=None, *flags):
        return subprocess.run([EXE, *flags, *map(str, self.paths if paths is None else paths)],
                              text=True, capture_output=True, timeout=20)

    def test_mixed_iq_and_metadata_only_shard(self):
        self.fixture()
        result = self.run_plan()
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["device_experts"], 3)
        self.assertEqual(plan["device_allocated_bytes"], 3 * 256 * (110 + 82 + 136))
        self.assertEqual(plan["skipped_mtp_bytes"], 3 * 3 * 256 * 256 * 2)
        self.assertEqual(plan["nonexpert_packed_bytes"], 1024)
        self.assertFalse(plan["hpu_inference_available"])
        # Directory order and shard argument order cannot alter ownership or byte totals.
        reordered = self.run_plan(list(reversed(self.paths)))
        self.assertEqual(json.loads(reordered.stdout), plan)

    def test_q2k_q3k_geometry(self):
        self.fixture((10, 11, 21))
        result = self.run_plan()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["device_allocated_bytes"],
                         3 * 256 * (84 + 110 + 110))

    def test_missing_metadata_shard(self):
        self.fixture()
        result = self.run_plan(self.paths[1:])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("all shards", result.stderr)

    def test_repeated_shard_number(self):
        self.fixture()
        write(self.paths[0], [], number=1)
        result = self.run_plan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("split.no", result.stderr)

    def test_declared_tensor_count(self):
        self.fixture()
        for path in self.paths:
            data = path.read_bytes()
            # Replace the total INT32 independently of the directory, retaining consistent metadata.
            marker = string("split.tensors.count") + struct.pack("<I", 5)
            offset = data.index(marker) + len(marker)
            path.write_bytes(data[:offset] + struct.pack("<i", 8) + data[offset + 4:])
        result = self.run_plan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tensor count", result.stderr)

    def test_inconsistent_trunk(self):
        self.fixture()
        for path in self.paths[:1]:
            data = path.read_bytes()
            marker = string("glm5-next.trunk_block_count") + struct.pack("<I", 10)
            offset = data.index(marker) + len(marker)
            path.write_bytes(data[:offset] + struct.pack("<Q", 1) + data[offset + 8:])
        result = self.run_plan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inconsistent trunk", result.stderr)

    def test_truncated_payload(self):
        self.fixture()
        data = self.paths[2].read_bytes()
        self.paths[2].write_bytes(data[:-128])
        self.assertNotEqual(self.run_plan().returncode, 0)

    def test_negative_budget_rejected(self):
        self.fixture()
        self.assertNotEqual(self.run_plan(None, "--host-gib", "-1").returncode, 0)

    def test_negative_split_tensor_count_rejected(self):
        self.fixture()
        write(self.paths[0], [], number=0, total=-1)
        result = self.run_plan()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonnegative integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
