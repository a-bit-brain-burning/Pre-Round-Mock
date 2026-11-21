"""P2 boundary regressions: byte guards, ABI values, dependency checks and buckets."""

from pathlib import Path
import tempfile
import unittest

from pipeline import GUARD, Program, Scalar, read_snapshot
from run_suite import bucket_check, check_case
from run_swiglu import torch, raw_tensor
from workloads import Case


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="sim002-tests-"))
        (cls.root / "AGENTS.md").write_text("# Retained test fixtures\n\nFlat binary fixtures; no automatic cleanup.\n")

    def test_scalar_signed_integer_and_float_bit_encoding(self):
        self.assertEqual(Scalar(-1, "i32").bits(), 0xffffffff)
        self.assertEqual(Scalar(1.0, "fp32").bits(), 0x3f800000)
        self.assertEqual(Scalar(-0.0, "fp64").bits(), 0x8000000000000000)
        with self.assertRaises(Exception):
            Scalar(1 << 40, "i32").bits()

    def test_buffer_names_are_unique_and_cannot_escape(self):
        p = Program("test")
        p.buffer("valid", torch.ones(1))
        for name in ("valid", "../escape", "has space"):
            with self.assertRaises(ValueError):
                p.buffer(name, torch.ones(1))

    def test_output_storage_is_poisoned(self):
        p = Program("poison")
        p.output("x", (3,), torch.float32)
        self.assertTrue(torch.isnan(p.buffers["x"]["tensor"]).all())

    def test_both_guards_and_exact_size_are_required(self):
        tensor = torch.tensor([1., 2.])
        data = GUARD + raw_tensor(tensor) + GUARD
        for name, damaged in (("before", b"\0" + data[1:]), ("after", data[:-1] + b"\0"), ("short", data[:-1])):
            path = self.root / f"{name}.bin"
            path.write_bytes(damaged)
            with self.assertRaises((AssertionError, ValueError)):
                read_snapshot(path, tensor)
        path = self.root / "valid.bin"
        path.write_bytes(data)
        self.assertTrue(torch.equal(read_snapshot(path, tensor), tensor))

    def test_actual_batch_bit_difference_is_rejected(self):
        p = Program("invariance")
        expected = torch.ones((2, 1, 1), dtype=torch.bfloat16)
        c = Case("batch", "gqa", p, {"output": expected}, .005,
                 {"invariance": {"group": "group", "role": "compare", "output_slice": [0, 2], "base_slice": [0, 2]}})
        base = expected.clone()
        base.view(torch.int16).flatten()[0] += 1
        result = check_case(c, {"output": expected}, {"group": base})
        self.assertTrue(result["checks"]["output"]["passed"])
        self.assertFalse(result["checks"]["invariance"]["passed"])
        self.assertEqual(result["status"], "FUNCTIONAL_FAIL")

    def test_bucket_membership_rejects_duplicates_without_assuming_order(self):
        p = Program("bucket")
        p.buffer("ids", torch.tensor([[0, 1], [1, 0]], dtype=torch.int32), readonly=True)
        out = {"count": torch.tensor([2, 2], dtype=torch.int32),
               "routes": torch.tensor([[3, 0], [2, 1]], dtype=torch.int32)}
        self.assertTrue(bucket_check(p, out)["passed"])
        out["routes"][1, 1] = 2
        with self.assertRaises(AssertionError):
            bucket_check(p, out)

    def test_unwritten_chain_output_fails_original_numerical_gate(self):
        expected = torch.ones((2, 3))
        case = Case("missing", "moe", Program("missing"), {"output": expected}, 1e-4)
        result = check_case(case, {"output": torch.full_like(expected, float("nan"))}, {})
        self.assertEqual(result["status"], "FUNCTIONAL_FAIL")
        self.assertTrue(result["negative_control_rejected"])


if __name__ == "__main__":
    unittest.main()
