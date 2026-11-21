"""Independent accounting, units, scheduling and fail-closed ranking checks."""

import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from performance_model import (audit_calls, compare, kernel_time, load_profile, predict,
                               residency, scenarios, spill_bounds, timeline)
from performance_work import WIDTH, account
from run_performance import checked_hash, source_gate


def buf(name, shape, dtype="torch.bfloat16", readonly=True):
    return {"name": name, "shape": shape, "dtype": dtype,
            "bytes": math.prod(shape) * WIDTH[dtype], "readonly": readonly}


def launch(entry, constants, grid, *, reg=32, shared=0):
    return {"entry": entry, "constants": constants, "grid": grid, "threads": 128,
            "options": {"num_warps": 4}, "shared_bytes": shared,
            "resources": {"reg": reg, "shared": 0, "stack": 0, "local": 0}}


def swiglu(n=257, block=256):
    return {"buffers": [buf(k, [n], readonly=k in ("a", "b", "g")) for k in ("a", "b", "g", "da", "db")],
            "launches": [launch("_backward", {"N": n, "BLOCK": block}, [(n + block - 1) // block, 1, 1])]}


class PerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile()
        cls.p = {k: v["value"] for k, v in cls.profile["parameters"].items()}

    def test_swiglu_valid_bytes_and_padded_math_are_distinct(self):
        w = account("swiglu", swiglu(), {})[0]
        self.assertEqual(w["logical_bytes"], 257 * 10)
        self.assertEqual(w["operations"]["fp32"], 512 * 11)
        self.assertEqual(w["operations"]["exp2"], 512)

    def test_scope_fails_closed(self):
        with self.assertRaises(ValueError):
            account("generic_gemm", swiglu(), {})
        bad = swiglu()
        bad["launches"][0]["entry"] = "replacement"
        with self.assertRaises(ValueError):
            account("swiglu", bad, {})

    def test_gqa_position31_32_branches_and_fp64(self):
        buffers = [buf("q", [2, 1, 1]), buf("k", [33, 1, 1]), buf("v", [33, 1, 1]),
                   buf("qoff", [2], "torch.int32"), buf("koff", [2], "torch.int32"),
                   buf("output", [2, 1, 1], readonly=False)]
        p = {"buffers": buffers, "launches": [launch("_gqa", dict(HQ=1, HK=1, D=1, BD=1, BN=128, WINDOW=0), [2, 1, 1])]}
        w = account("gqa", p, {"q_offsets": [0, 2], "kv_offsets": [0, 33]})[0]
        self.assertEqual(w["traffic"]["k"]["read_bytes"], (32 + 33) * 2)
        self.assertEqual(w["traffic"]["k"]["unique_read_bytes"], 33 * 2)
        self.assertEqual(w["operations"]["fp64"], 98 + 386)
        self.assertEqual(w["operations"]["div64"], 2)
        self.assertEqual(w["operations"]["tensor"], 0)

    def test_sink_tiled_loads_block_boundary_but_masks_math_semantics(self):
        p = {"buffers": [buf("q", [1, 4, 4, 128]), buf("k", [1, 17, 1, 128]), buf("v", [1, 17, 1, 128]),
                         buf("sink", [1, 128]), buf("output", [1, 4, 4, 128], readonly=False)],
             "launches": [launch("_sink_tiled", dict(HQ=4, HK=1, D=128, BD=128, BN=32, BM=16, P_PARTS=3, WINDOW=0, LQ=4, LK=17), [1, 1, 1])]}
        w = account("sink", p, {})[0]
        self.assertEqual(w["traffic"]["k"]["read_bytes"], 17 * 128 * 2)
        self.assertEqual(w["operations"]["tensor"], 8 * 16 * 32 * 128)
        self.assertEqual(w["traffic"]["output"]["write_bytes"], 1 * 4 * 4 * 128 * 2)

    def test_grouped_moe_empty_experts_and_three_terms(self):
        p = {"buffers": [buf("x", [1, 2]), buf("wgu", [2, 4, 2], "torch.float8_e4m3fn"),
                         buf("wdown", [2, 2, 2], "torch.float8_e4m3fn"), buf("ids", [1, 1], "torch.int32"),
                         buf("weights", [1, 1], "torch.float32"), buf("scale_gu", [2], "torch.float32"),
                         buf("scale_down", [2], "torch.float32"), buf("count", [2], "torch.int32", False),
                         buf("routes", [2, 1], "torch.int32", False), buf("hidden", [1, 1, 2], "torch.float32", False),
                         buf("output", [1, 2], readonly=False)],
             "launches": [launch("_grouped_gate_up", dict(BM=16, BN=64, BK=64), [2, 1, 1]),
                          launch("_grouped_down", dict(BM=16, BN=64, BK=64, TERMS=3, FUSE_SINGLE=True), [2, 1, 1])]}
        gu, down = account("moe", p, {"expert_counts": [1, 0]})
        self.assertEqual(gu["active_ctas"], 1)
        self.assertEqual(gu["empty_ctas"], 1)
        self.assertEqual(gu["logical_bytes"], 36)
        self.assertEqual(gu["traffic"]["count"]["read_bytes"], 8)
        self.assertEqual(down["traffic"]["wdown"]["read_bytes"], 4)
        self.assertEqual(down["operations"]["tensor"], 6 * 16 * 64 * 64)
        self.assertEqual(down["traffic"]["output"]["write_bytes"], 4)

    def test_router_pack_is_per_call_full_weight_pass(self):
        p = {"buffers": [buf("x", [2048, 3072]), buf("w", [256, 3072], "torch.float32"),
                         buf("high", [256, 3072], readonly=False), buf("low", [256, 3072], "torch.float16", False)],
             "launches": [launch("_pack", {"BLOCK": 1024}, [768, 1, 1])]}
        w = account("router", p, {})[0]
        self.assertEqual(w["logical_bytes"], 6291456)
        self.assertEqual(w["operations"]["convert"], 2359296)

    def test_tensor_units_from_paper_clock(self):
        self.assertAlmostEqual(self.p["sm_count"] * self.p["clock_GHz"] * self.p["tensor_ops_per_cycle_sm"] / 1000, 146.16576)

    def test_integer_architecture_prior_tracks_clock(self):
        p = swiglu()
        w = account("swiglu", p, {})[0]
        a = kernel_time(p["launches"][0], w, self.p)
        b = kernel_time(p["launches"][0], w, {**self.p, "clock_GHz": self.p["clock_GHz"] * 2})
        self.assertAlmostEqual(a["pipeline_us"]["integer"], 2 * b["pipeline_us"]["integer"])

    def test_requested_stages_do_not_invent_compiled_overlap(self):
        p = swiglu()
        w = account("swiglu", p, {})[0]
        p["launches"][0]["options"]["num_stages"] = 1
        a = kernel_time(p["launches"][0], w, self.p)
        p["launches"][0]["options"]["num_stages"] = 3
        self.assertEqual(a, kernel_time(p["launches"][0], w, self.p))

    def test_dynamic_bytes_ignore_old_fraction_and_misses_cost_latency(self):
        program = swiglu()
        launch = program["launches"][0]
        launch["resources"]["stack"] = 4
        # Isolated service fixture; the CLI separately verifies its trace source.
        launch["dynamic_spill"] = {"status": "COMPLETE_DYNAMIC_LOCAL_ACCESSES", "trace_sha256": "fixture",
            "load_bytes": 512, "store_bytes": 512, "static_sites": [{}, {}],
            "classes": [{"warps": 4, "cta_multiplicity": 1, "events": [[0, "STL", 0, 4, 1], [16, "LDL", 0, 4, 1]]}]}
        work = account("swiglu", program, {})[0]
        cached = {**self.p, "spill_l1_budget_bytes": 131072}
        a = kernel_time(launch, work, cached)
        b = kernel_time(launch, work, {**cached, "spill_l1_budget_bytes": 0})
        c = kernel_time(launch, work, {**cached, "spill_executed_fraction": 0, "spill_l1_hit": 1})
        self.assertEqual(a["modeled_spill_bytes"], 1024)
        self.assertEqual(a, c)
        self.assertGreater(b["pipeline_us"]["local_dependency_proxy"], a["pipeline_us"]["local_dependency_proxy"])
        self.assertGreater(b["spill_service"]["l2_bytes"], a["spill_service"]["l2_bytes"])

    def test_profile_rejects_unknown_spill_policy(self):
        p = copy.deepcopy(self.profile)
        p["parameters"]["spill_warp_interleave"]["value"] = 0.5
        with patch.object(Path, "read_text", return_value=json.dumps(p)), self.assertRaises(ValueError):
            load_profile()

    def test_register_allocation_and_shared_reservation(self):
        l = launch("_backward", {}, [128, 1, 1], reg=255, shared=24576)
        l["resources"]["shared"] = 1024
        r = residency(l, self.p)
        self.assertEqual(r["allocated_registers_per_cta"], 32768)
        self.assertEqual(r["ctas_per_sm"], 2)
        self.assertEqual(r["allocated_shared_per_cta"], 26624)
        self.assertIn("registers", r["limiting_resources"])

    def test_impossible_resource_allocation_rejected(self):
        l = launch("_backward", {}, [1, 1, 1], shared=240000)
        with self.assertRaises(ValueError):
            residency(l, self.p)

    def test_async_host_submission_overlaps_gpu_work(self):
        p = {**self.p, "host_wrapper_us": 2, "host_allocation_us": 1, "host_submit_us": 3,
             "host_sync_us": 4, "graph_submit_us": 2}
        t = timeline([20, 5], 2, p)
        self.assertEqual(t["events"][0]["start_us"], 7)
        self.assertEqual(t["events"][1]["start_us"], 27)
        self.assertEqual(t["e2e_synced_us"], 36)
        self.assertEqual(t["graph_replay_us"], 27)

    def test_empty_chain_retains_allocation_and_wrapper_only(self):
        t = timeline([], 1, self.p)
        self.assertEqual(t["device_chain_us"], 0)
        self.assertEqual(t["graph_replay_us"], 0)
        self.assertEqual(t["e2e_synced_us"], self.p["host_wrapper_us"] + self.p["host_allocation_us"])

    def test_bandwidth_units_and_monotonicity(self):
        program = swiglu(32768, 256)
        works = account("swiglu", program, {})
        a = predict(program, works, self.p)
        b = predict(program, works, {**self.p, "hbm_GBps": self.p["hbm_GBps"] * 2})
        self.assertAlmostEqual(a["kernels"][0]["pipeline_us"]["hbm"], 2 * b["kernels"][0]["pipeline_us"]["hbm"])
        self.assertLessEqual(b["timeline"]["e2e_synced_us"], a["timeline"]["e2e_synced_us"])
        expected = 32768 * 10 * self.p["transaction_amplification"]
        self.assertEqual(a["kernels"][0]["modeled_hbm_bytes"], expected)

    def test_warm_cache_does_not_skip_output_stores(self):
        p = swiglu()
        w = account("swiglu", p, {})
        cold = predict(p, w, self.p)
        warm = predict(p, w, self.p, cache="warm")
        hbm = warm["kernels"][0]["modeled_hbm_bytes"]
        self.assertLess(hbm, cold["kernels"][0]["modeled_hbm_bytes"])
        self.assertGreaterEqual(hbm, 2 * 257 * 2)

    def test_spill_widths_and_dynamic_cap(self):
        l = launch("_packed_gemm", {}, [4, 1, 1])
        l["resources"]["stack"] = 64
        l["sass_stack_or_call_instructions"] = ["LDL.128 R4, [R1];", "STL.64 [R1], R4;"]
        s = spill_bounds(l, {"launched_ctas": 4, "max_loop_visits": 3})
        self.assertEqual(s["dynamic_bytes_cap"], (16 + 8) * 128 * 4 * 4)
        self.assertEqual(s["dynamic_bytes_min"], 0)
        self.assertFalse(s["ranking_eligible"])

    def test_unknown_stack_or_call_cannot_be_silently_free(self):
        l = launch("_gqa", {}, [1, 1, 1])
        l["resources"]["stack"] = 8
        with self.assertRaises(ValueError):
            spill_bounds(l, {"launched_ctas": 1, "max_loop_visits": 1})
        l["sass_stack_or_call_instructions"] = ["CALL.REL.NOINC 0x10 ;"]
        with self.assertRaises(ValueError):
            spill_bounds(l, {"launched_ctas": 1, "max_loop_visits": 1})

    def test_division_helper_audit_rejects_backward_edges(self):
        l = launch("_gqa", {}, [1, 1, 1])
        sass = "/*0000*/ CALL.REL.NOINC 0x20;\n/*0020*/ @P0 BRA 0x40;\n/*0030*/ MOV R0, R1;\n/*0040*/ RET.REL.NODEC R2 0x0;"
        self.assertTrue(audit_calls(l, sass)["acyclic_helpers"])
        with self.assertRaises(ValueError):
            audit_calls(l, sass.replace("BRA 0x40", "BRA 0x20"))

    def test_profile_rejects_nan_and_invalid_fraction(self):
        p = copy.deepcopy(self.profile)
        p["parameters"]["tensor_efficiency"]["range"][1] = 2
        with patch.object(Path, "read_text", return_value=json.dumps(p)), self.assertRaises(ValueError):
            load_profile()
        p["parameters"]["tensor_efficiency"]["value"] = float("nan")
        with patch.object(Path, "read_text", return_value=json.dumps(p)), self.assertRaises(ValueError):
            load_profile()

    def test_joint_scenarios_are_reproducible_and_include_endpoints(self):
        a, b = scenarios(self.profile, 16), scenarios(self.profile, 16)
        self.assertEqual(a, b)
        self.assertEqual(a[0][0], "nominal")
        self.assertIn("fp64_TFps:low", [n for n, _ in a])

    def test_stale_source_and_corrupt_file_rejected(self):
        with self.assertRaises(ValueError):
            source_gate({"source_sha256": {}}, "router")
        with patch("run_performance.sha", return_value="bad"), self.assertRaises(ValueError):
            checked_hash("ignored", "correct")

    def test_ranking_gates_and_reversal(self):
        def row(times):
            return {"task": "swiglu", "functional_status": "FUNCTIONAL_PASS", "input_identity": ["same"], "semantic_identity": {},
                    "model": {"nominal": {"ranking_eligible": True}, "_paired_times": {"e2e_synced_us": times}}}
        a, b = row([2, 4]), row([1, 2])
        self.assertEqual(compare(a, b)["status"], "RIGHT_FASTER_IN_SAMPLED_SCENARIOS")
        self.assertEqual(compare(row([2, 1]), b)["status"], "INCONCLUSIVE_WITHIN_MODEL_ASSUMPTIONS")
        for change in ({"functional_status": "FUNCTIONAL_FAIL"}, {"known_numerical_risk": True}, {"input_identity": ["other"]}):
            self.assertEqual(compare({**a, **change}, b)["status"], "RANKING_BLOCKED")
        a["model"]["nominal"]["ranking_eligible"] = False
        self.assertEqual(compare(a, b)["status"], "RANKING_BLOCKED")
        self.assertEqual(compare({"functional_status": "FUNCTIONAL_FAIL"}, b)["status"], "RANKING_BLOCKED")


if __name__ == "__main__":
    unittest.main()
