import unittest
from pathlib import Path

from contest_scoring import (contest_profile, forecast_card, load_probe, read_json, replay_call_time,
                             score_times, traffic_time)
from performance_model import load_profile


HERE = Path(__file__).resolve().parent


class ContestScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = load_probe()
        cls.profile = contest_profile(load_profile(), cls.probe)
        cls.card = read_json(HERE / "swiglu_scorecard.json")
        cls.rows = read_json(HERE / "swiglu_replay_case.json")["cases"]

    def test_official_published_examples(self):
        rule = {"mode": "preliminary_per_case", "base_points": 1500}
        cases = [{"benchmark_us": 1000}] * 3
        for times, expected in [([500]*3, 2250), ([1000]*3, 1500),
                                ([1000,1500,1500], 1166.6667), ([1000,2000,2000], 1000)]:
            self.assertEqual(score_times(rule, cases, times), expected)

    def test_recorded_memory_states(self):
        params = {k: v["value"] for k, v in self.profile["parameters"].items()}
        f = {"matrix_bytes": 16777216, "layout": self.probe["layout"], "launches": 1, "operation": "copy"}
        rotated = traffic_time(f, params, self.probe)
        repeated = traffic_time({**f, "operation": "warmcopy"}, params, self.probe)
        self.assertAlmostEqual(rotated - repeated, 15.872 - 11.168)
        with self.assertRaises(ValueError):
            traffic_time({**f, "matrix_bytes": 256}, params, self.probe)
        self.assertEqual(params["l2_bytes"], 62914560)
        self.assertAlmostEqual(params["device_launch_us"], 2.256)

    def test_verified_replay_needs_benchmark(self):
        result = forecast_card(self.card, self.rows, self.profile, self.probe)
        self.assertEqual(result["decision"], "AWAITING_BENCHMARKS")
        self.assertIsNone(result["score"])
        self.assertGreater(result["cases"][0]["estimate_us"], 0)
        case = self.card["cases"][0]
        with self.assertRaises(ValueError):
            forecast_card({**self.card, "cases": [{**case, "input_identity": []}]}, self.rows, self.profile, self.probe)
        with self.assertRaises(ValueError):
            forecast_card({**self.card, "cases": []}, self.rows, self.profile, self.probe)

    def test_full_call_preserves_recorded_kernel_order(self):
        case = self.card["cases"][0]
        row, = (r for r in self.rows if r["id"] == case["model_id"])
        nominal = row["model"]["nominal"]
        timing = {"receipt_id": row["id"], "protocol": "archived-model", "device_steps": [{"kernel": 0}]}
        elapsed = replay_call_time(row, timing, load_profile(), "archived-model")
        self.assertAlmostEqual(elapsed, nominal["timeline"]["e2e_synced_us"])
        with self.assertRaises(ValueError):
            replay_call_time(row, {**timing, "device_steps": []}, self.profile, "archived-model")
        stage = {"kernel": 0, "device_us": nominal["kernels"][0]["device_us"],
                 "cubin_sha256": row["launches"][0]["cubin_sha256"]}
        recorded = {**timing, "input_identity": row["input_identity"], "device_steps": [stage]}
        self.assertAlmostEqual(replay_call_time(row, recorded, load_profile(), "archived-model"), elapsed)
        with self.assertRaisesRegex(ValueError, "inputs"):
            replay_call_time(row, {**timing, "device_steps": [stage]}, self.profile, "archived-model")


if __name__ == "__main__":
    unittest.main()
