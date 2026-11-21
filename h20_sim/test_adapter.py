"""Failure-oriented checks for the CPU replay boundary; no GPU is initialized."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from bootstrap import build_environment
from operator_source import ENGINE, evidence_key, evidence_path, operator_root, report_directory, source_key
from run_swiglu import (checked_command, instruction_counts, numerical_check,
                        raw_tensor, read_tensor, write_json)
import subprocess
import torch


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Retained deliberately: this workspace does not permit automatic deletion.
        cls.root = Path(tempfile.mkdtemp(prefix="sim001-adapter-tests-"))
        (cls.root / "AGENTS.md").write_text(
            "# Test artifacts\n\nFlat binary/JSON/log fixtures; retain, no automatic cleanup.\n")

    def fixture(self, name, contents):
        path = self.root / name
        path.write_bytes(contents)
        return path

    def test_operators_and_candidate_results_use_external_checkout(self):
        root = operator_root()
        self.assertNotEqual(root, ENGINE.parent)
        self.assertEqual(source_key(root / "SwiGLU_backward/solution.py"), "SwiGLU_backward/solution.py")
        self.assertEqual(source_key(ENGINE / "run_candidates.py"), "h20_sim/run_candidates.py")
        for phase, folder in (("sim005", ENGINE), ("sim006", root / "h20_sim")):
            key = f"h20_sim/results-{phase}-example.json"
            self.assertEqual(report_directory(phase), folder)
            self.assertEqual(evidence_path(key), folder / Path(key).name)
            self.assertEqual(evidence_key(evidence_path(key)), key)

    def test_engine_cannot_be_used_as_operator_checkout(self):
        with patch.dict(os.environ, {"PREROUND_OPERATOR_ROOT": str(ENGINE.parent)}):
            with self.assertRaisesRegex(ValueError, "separate checkout of csxsl"):
                operator_root()

    def test_output_bytes_preserve_dtype_and_shape(self):
        for dtype in (torch.float32, torch.bfloat16):
            expected = torch.tensor([[0, -1.25, 3.5]], dtype=dtype)
            data = raw_tensor(expected) + b"\xa5" * (32 * expected.element_size())
            path = self.fixture(f"roundtrip-{dtype}.bin", data)
            actual = read_tensor(path, dtype, (1, 3))
            self.assertEqual(actual.dtype, dtype)
            self.assertTrue(torch.equal(actual, expected))

    def test_wrong_output_length_is_rejected(self):
        path = self.fixture("truncated.bin", b"\0" * 8)
        with self.assertRaisesRegex(ValueError, "Unexpected output size"):
            read_tensor(path, torch.float32, (1, 3))

    def test_guard_overwrite_is_rejected(self):
        path = self.fixture("bad-guard.bin", b"\0" * ((3 + 32) * 4))
        with self.assertRaisesRegex(AssertionError, "guard overwritten"):
            read_tensor(path, torch.float32, (1, 3))

    def test_wrong_and_unwritten_outputs_are_rejected(self):
        expected = torch.tensor([[1.0, -2.0]])
        for actual in (expected + 1, torch.full_like(expected, float("nan"))):
            self.assertFalse(numerical_check(actual, expected, 2e-5, "wrong")["passed"])

    def test_json_does_not_overwrite_or_emit_nonstandard_numbers(self):
        path = self.root / "report.json"
        write_json(path, {"error": float("inf"), "nested": [float("nan")]})
        self.assertEqual(json.loads(path.read_text()), {"error": "inf", "nested": ["nan"]})
        with self.assertRaises(FileExistsError):
            write_json(path, {"status": "incorrect overwrite"})

    def test_process_failure_and_timeout_are_not_passes(self):
        env = os.environ.copy()
        with self.assertRaisesRegex(RuntimeError, "Command failed \\(7\\)"):
            checked_command([sys.executable, "-c", "raise SystemExit(7)"],
                            self.root, env, "nonzero.log")
        with self.assertRaises(subprocess.TimeoutExpired):
            checked_command([sys.executable, "-c", "import time; time.sleep(10)"],
                            self.root, env, "timeout.log", timeout=0.1)

    def test_physical_devices_are_disabled_in_child_environment(self):
        env = build_environment(self.root)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertNotIn("/usr/lib/wsl", env["LD_LIBRARY_PATH"])
        self.assertFalse(torch.cuda.is_initialized())

    def test_static_instruction_inventory_includes_predicated_operations(self):
        ptx = """.version 8.7
        .entry kernel() {
        .reg .b32 %r<2>;
        mov.u32 %r0, 1; // add.f32 is not an instruction here
        @%p0 ld.global.b32 %r1, [%rd0];
        @!%p0 st.global.b32 [%rd1], %r1;
        ret;
        }
        """
        self.assertEqual(instruction_counts(ptx),
                         {"mov.u32": 1, "ld.global.b32": 1, "st.global.b32": 1, "ret": 1})


if __name__ == "__main__":
    unittest.main()
