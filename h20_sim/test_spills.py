"""Independent integer, slicing, rejection and cache regressions (CPU only)."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_execute import Lane, STACK_TOP, abi_parameters, signed
from spill_sass import Projection, UnsupportedSpill, parse
from spill_service import cached_service, sectors, service
from spill_trace import binding, check_sink_path, closed_form


def assembly(*lines):
    return "\n".join(f"/*{16*i:04x}*/ {line};" for i, line in enumerate(lines))


def execute(*lines, frame=8, cta=(0, 0, 0), tid=0):
    text = assembly("LDC R1, c[0x0][0x28]", f"VIADD R1, R1, 0x{(1<<32)-frame:x}", *lines, "EXIT")
    q = Projection(text)
    c = {0x28+i: b for i, b in enumerate(STACK_TOP.to_bytes(4, "little"))}
    return Lane(q, {"resources": {"stack": frame}}, c, {}, cta, tid).run()


class SliceTests(unittest.TestCase):
    def test_loop_spilled_counter(self):
        s = execute("IMAD.MOV.U32 R0, RZ, RZ, RZ", "STL [R1], R0", "LDL R2, [R1]",
                    "VIADD R2, R2, 0x1", "STL [R1], R2", "ISETP.LT.AND P0, PT, R2, 0x5, PT", "@P0 BRA 0x40")
        self.assertEqual(len(s.events), 11)
        self.assertEqual(s.reg["R2"], 5)
        self.assertEqual(s.branches, [[128, 1]]*4+[[128, 0]])

    def test_irrelevant_float_branch_skipped(self):
        s = execute("FSETP.GT.AND P0, PT, R100, R101, PT", "@P0 BRA 0x50", "FADD R8, R8, R9",
                    "STL [R1], R8", "LDL R3, [R1]")
        self.assertEqual(len(s.events), 2)
        self.assertNotIn(0x30, s.q.selected)

    def test_live_float_branch_rejected(self):
        with self.assertRaises(UnsupportedSpill):
            execute("IMAD.MOV.U32 R8, RZ, RZ, RZ", "FADD R8, R8, R8",
                    "FSETP.GT.AND P0, PT, R8, R8, PT", "@P0 BRA 0x70", "STL [R1], R8")

    def test_uninitialized_local_rejected(self):
        with self.assertRaisesRegex(UnsupportedSpill, "uninitialized"):
            execute("LDL R0, [R1]")

    def test_partial_initialization_rejected(self):
        with self.assertRaisesRegex(UnsupportedSpill, "uninitialized"):
            execute("STL [R1], RZ", "LDL.64 R0, [R1]")

    def test_dead_s16_padding_remains_unknown_and_counted(self):
        s = execute("STL.S16 [R1], RZ", "LDL R0, [R1]")
        self.assertEqual(s.events[1][3], 4)
        self.assertIsNone(s.reg["R0"])
        self.assertEqual(s.padding_reads[0x30], 1)

    def test_live_s16_padding_rejected(self):
        with self.assertRaises(UnsupportedSpill):
            execute("STL.S16 [R1], RZ", "LDL R0, [R1]", "ISETP.EQ.AND P0, PT, R0, RZ, PT", "@P0 STL [R1], RZ")

    def test_frame_bounds_rejected(self):
        with self.assertRaisesRegex(UnsupportedSpill, "frame"):
            execute("STL.64 [R1+0x4], RZ")

    def test_dynamic_address_rejected(self):
        with self.assertRaisesRegex(UnsupportedSpill, "Non-fixed"):
            execute("STL [R2], RZ")

    def test_unknown_dead_side_effect_rejected(self):
        with self.assertRaisesRegex(UnsupportedSpill, "Unknown register"):
            execute("MAGIC R0, RZ", "STL [R1], RZ")

    def test_unknown_modifier_rejected_even_on_dead_value(self):
        with self.assertRaisesRegex(UnsupportedSpill, "Unknown register"):
            execute("FADD.UNKNOWN R0, RZ, RZ", "STL [R1], RZ")

    def test_doubleword_value_dependency(self):
        s = execute("CS2R R2, SRZ", "VIADD R3, R3, 0x7", "STL.64 [R1], R2", "LDL.64 R4, [R1]",
                    "ISETP.EQ.AND P0, PT, R5, 0x7, PT", "@!P0 BRA 0x90", "STL [R1], RZ")
        self.assertEqual(s.reg["R5"], 7)
        self.assertEqual(len(s.events), 3)

    def test_thread_dependence_retained(self):
        s = execute("S2R R2, SR_TID.X", "ISETP.LT.AND P0, PT, R2, 0x10, PT", "@P0 STL [R1], RZ", tid=17)
        self.assertEqual(s.events[-1][-1], 0)
        self.assertIn("SR_TID.X", s.q.live_specials)

    def test_local_control_predicate_retained(self):
        for tid in (0, 1):
            s = execute("S2R R2, SR_TID.X", "ISETP.EQ.AND P0, PT, R2, RZ, PT", "@P0 BRA 0x60", "STL [R1], RZ", tid=tid)
            self.assertEqual(len(s.events), tid)

    def test_unknown_helper_spill_rejected(self):
        with self.assertRaisesRegex(UnsupportedSpill, "Helper"):
            Projection(assembly("CALL.REL.NOINC 0x20", "EXIT", "STL [R1], R0", "RET.REL.NODEC R0"))


class SinkSpillTests(unittest.TestCase):
    def fixture(self, threshold="-0xe8"):
        text = assembly(
            "LDC R1, c[0x0][0x28]", "VIADD R1, R1, 0xffffffe8", "S2UR UR4, SR_CTAID.X",
            "USHF.L.U32 UR17, UR4, 0x4, URZ", "UIADD3 UR5, UR17, 0xf, URZ",
            f"UISETP.GT.AND UP4, UPT, UR5, {threshold}, UPT",
            "PLOP3.LUT P0, PT, PT, PT, UP4, 0x80, 0x0",
            "STL [R1+0x10], RZ", "STL [R1+0xc], RZ", "STL [R1+0x8], RZ", "STL [R1], RZ",
            "@P0 BRA 0xe0", "STL [R1+0x4], RZ", "BRA 0xf0", "STL [R1+0x4], RZ",
            "LDL.LU R2, [R1+0x4]", "LDL.LU R2, [R1+0x10]", "LDL.LU R2, [R1]",
            "LDL.LU R2, [R1+0xc]", "LDL.LU R2, [R1+0x8]", "EXIT")
        launch = {"entry": "_sink_tiled", "grid": [2, 1, 1], "resources": {"stack": 24},
                  "constants": dict(LQ=8, LK=65, HQ=4, HK=1, D=128, BD=128, BM=16, BN=32, WINDOW=0, P_PARTS=3)}
        return text, launch

    def test_uniform_spill_path_for_both_source_tiles(self):
        text, launch = self.fixture()
        q = Projection(text)
        bank = {0x28+i: b for i, b in enumerate(STACK_TOP.to_bytes(4, "little"))}
        for tile, visits in enumerate((2, 3)):
            cta = (tile, 0, 0)
            oracle = closed_form(q, launch, {}, cta)
            lane = Lane(q, launch, bank, {}, cta, 0).run()
            check_sink_path(oracle, lane.events, lane.branches)
            self.assertEqual(oracle["source_kv_visits"], visits)
            self.assertEqual([e[2] for e in lane.events], [16, 12, 8, 0, 4, 4, 16, 0, 12, 8])
            self.assertEqual(lane.reg["UR5"], tile * 16 + 15)
            self.assertEqual(q.live_specials, ["SR_CTAID.X"])

    def test_wrong_arm_rejected_even_with_identical_byte_totals(self):
        text, launch = self.fixture(threshold="0x10")
        q = Projection(text)
        cta = (0, 0, 0)
        bank = {0x28+i: b for i, b in enumerate(STACK_TOP.to_bytes(4, "little"))}
        lane = Lane(q, launch, bank, {}, cta, 0).run()
        self.assertEqual(sum(e[3] for e in lane.events), 40)
        with self.assertRaisesRegex(UnsupportedSpill, "path or site order"):
            check_sink_path(closed_form(q, launch, {}, cta), lane.events, lane.branches)

    def test_unknown_config_and_unexpected_slot_rejected(self):
        text, launch = self.fixture()
        with self.assertRaisesRegex(UnsupportedSpill, "frame offsets"):
            closed_form(Projection(text.replace("R1+0xc", "R1+0x14")), launch, {}, (0,0,0))
        launch["constants"]["WINDOW"] = 33
        with self.assertRaisesRegex(UnsupportedSpill, "configuration"):
            closed_form(Projection(text), launch, {}, (0,0,0))

    def test_vector_and_tensor_register_footprints(self):
        instructions = parse(assembly("LDS.128 R4, [R2]", "LDSM.16.MT88.4 R4, [R2]",
                                      "HMMA.16816.F32.BF16 R4, R16, R32, R40", "STSM.16.M88.2 [R6], R8"))
        for pc in (0, 16, 32):
            self.assertEqual(instructions[pc].definitions(), {"R4", "R5", "R6", "R7"})
        self.assertEqual(instructions[32].dependencies(), {"R16", "R17", "R18", "R19", "R32", "R33",
                                                            "R40", "R41", "R42", "R43"})
        self.assertEqual(instructions[48].definitions(), set())

    def test_unknown_live_predicate_lut_rejected(self):
        text, launch = self.fixture()
        q = Projection(text.replace("0x80, 0x0", "0x40, 0x0"))
        bank = {0x28+i: b for i, b in enumerate(STACK_TOP.to_bytes(4, "little"))}
        with self.assertRaisesRegex(UnsupportedSpill, "Unsupported live instruction"):
            Lane(q, launch, bank, {}, (0,0,0), 0).run()


class IntegerTests(unittest.TestCase):
    def lane(self):
        return Lane(None, {}, {}, {}, (0, 0, 0), 0)

    def instruction(self, s, text):
        s.execute(parse(assembly(text))[0])

    def test_signed_wrap(self):
        self.assertEqual(signed(0xffffffff), -1)
        self.assertEqual(signed(0x80000000), -(1<<31))

    def test_address_carry(self):
        s = self.lane(); s.reg.update(R4=3, R8=0xfffffffc, R9=2)
        self.instruction(s, "LEA R2, P0, R4, R8, 0x2")
        self.instruction(s, "LEA.HI.X.SX32 R3, R4, R9, 0x2, P0")
        self.assertEqual(s.reg["R2"] + (s.reg["R3"]<<32), 0x2fffffffc + 12)

    def test_midpoint_and_negative_truncation(self):
        for value in (-5, -1, 0, 1, 3, 30, 101, 0x7ffffffe):
            s = self.lane(); s.reg["R2"] = value & 0xffffffff
            self.instruction(s, "LEA.HI R2, R2, R2, RZ, 0x1")
            self.instruction(s, "SHF.R.S32.HI R4, RZ, 0x1, R2")
            self.assertEqual(signed(s.reg["R4"]), int(value/2))

    def test_shift_pair(self):
        s = self.lane(); s.reg.update(R2=0xc0000001, R3=7)
        self.instruction(s, "SHF.L.U64.HI R0, R2, 0x2, R3")
        self.assertEqual(s.reg["R0"], 31)

    def test_window_clamp_and_alignment(self):
        for position in (0, 15, 16, 31, 32, 127, 128, 129):
            s = self.lane(); s.reg["R5"] = (position-16) & 0xffffffff
            self.instruction(s, "VIMNMX R8, RZ, R5, !PT")
            self.instruction(s, "LOP3.LUT R8, R8, 0xffffff80, RZ, 0xc0, !PT")
            self.assertEqual(s.reg["R8"], max(0, position-16)//128*128)

    def test_add_carry(self):
        s = self.lane(); s.reg.update(R2=0xffffffff, R3=9)
        self.instruction(s, "IADD3 R2, P0, R2, 0x1, RZ")
        self.instruction(s, "IADD3.X R3, R3, RZ, RZ, P0, !PT")
        self.assertEqual((s.reg["R2"], s.reg["R3"]), (0, 10))

    def test_abi_padding_and_width(self):
        elf = "EIATTR_PARAM_CBANK\nFormat: EIFMT_SVAL\nValue: 0xf 0x100210\nOrdinal: 0x0 Offset: 0x0 Size: 0x4\nOrdinal: 0x1 Offset: 0x8 Size: 0x8"
        launch = {"arguments": [{"kind": "scalar", "dtype": "i32", "bits": 3}, {"kind": "buffer", "name": "qoff", "offset": 4}]}
        c, abi = abi_parameters(elf, launch, {"qoff": (0x300000000, b"", True)})
        self.assertEqual(c[0x210], 3)
        self.assertNotIn(0x214, c)
        self.assertEqual(c[0x218], 4)
        launch["arguments"][0]["dtype"] = "u64"
        with self.assertRaises(UnsupportedSpill):
            abi_parameters(elf, launch, {"qoff": (0x300000000, b"", True)})


class ServiceTests(unittest.TestCase):
    def test_coalesced_widths(self):
        self.assertEqual(sectors(0, 4, 0, 8), (0, 1, 2, 3))
        self.assertEqual(len(sectors(0, 8, 0, 8)), 8)
        self.assertFalse(set(sectors(0, 8, 0, 8)) & set(sectors(0, 8, 1, 8)))

    def test_hit_and_eviction_traffic(self):
        events = ((0, "STL", 0, 4, 1), (16, "LDL", 0, 4, 1), (32, "STL", 4, 4, 1), (48, "LDL", 0, 4, 1))
        stats = cached_service(events, 1, 8, 128, False, False)
        self.assertEqual(stats["load_hit_sectors"], 4)
        self.assertEqual(stats["load_miss_sectors"], 4)
        self.assertEqual(stats["l2_write_sectors"], 8)
        self.assertEqual(stats["dirty_final_sectors"], 0)

    def test_schedule_matters(self):
        events = ((0, "STL", 0, 4, 1), (16, "LDL", 0, 4, 1))
        serial = cached_service(events, 2, 4, 128, False, False)
        interleaved = cached_service(events, 2, 4, 128, True, False)
        self.assertEqual(serial["load_hit_sectors"], 8)
        self.assertEqual(interleaved["load_hit_sectors"], 0)

    def test_store_policy_and_disabled_cache(self):
        events = ((0, "STL", 0, 4, 1), (16, "LDL", 0, 4, 1))
        wt = cached_service(events, 1, 4, 128, False, True)
        off = cached_service(events, 1, 4, 0, False, False)
        self.assertEqual(wt["l2_write_sectors"], 4)
        self.assertEqual(wt["dirty_final_sectors"], 0)
        self.assertEqual(off["l2_read_sectors"], 4)
        self.assertEqual(off["l2_write_sectors"], 4)

    def test_partial_store_requires_refill(self):
        stats = cached_service(((0, "STL", 0, 2, 1),), 1, 4, 128, False, False)
        self.assertEqual(stats["write_sectors"], 4)
        self.assertEqual(stats["l2_read_sectors"], 4)

    def test_input_and_launch_binding(self):
        p = {"buffers": [{"input_sha256": "a"}], "protocol": "SIM002_V1"}
        launch = {"grid": [1, 1, 1], "cubin_sha256": "a", "call_audit": {}}
        key = binding(p, launch)
        launch["call_audit"] = {"changed": True}
        self.assertEqual(key, binding(p, launch))
        launch["grid"][0] = 2
        self.assertNotEqual(key, binding(p, launch))
        launch["grid"][0] = 1
        p["buffers"][0]["input_sha256"] = "b"
        self.assertNotEqual(key, binding(p, launch))


if __name__ == "__main__":
    unittest.main()
