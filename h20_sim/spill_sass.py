"""Five-task-only SASS control/address projection for local-memory accesses.

This is not a numerical SASS emulator. A backward slice preserves every LDL/
STL event, its predicate/address, and data/control dependencies. Irrelevant
floating-point regions may be skipped only at their proved postdominator.
Unknown live instruction semantics are rejected by the executor.
"""

from collections import defaultdict
from dataclasses import dataclass
import re


class UnsupportedSpill(ValueError):
    pass


REG = re.compile(r"(?<![A-Za-z0-9_])(?:UR\d+|R\d+|UP\d+|P\d+)(?![A-Za-z0-9_])")

# Register footprints audited in these five-task cubins, including the GQA
# leaf helper. Recognition here does NOT imply executable numerical semantics.
# A new modifier must not silently change a destination's width/side effects.
FOOTPRINT_OPS = set("""
BAR.SYNC.DEFER_BLOCKING BRA BREAK BSSY BSYNC CALL.REL.NOINC CS2R
DADD DFMA DMUL DMUL.RP DSETP.NAN.AND EXIT F2F.BF16.F64 F2F.F64.BF16 F2F.F64.F32
F2FP.F16.F32.PACK_AB FADD FENCE.VIEW.ASYNC.S FFMA FMNMX FMUL FSEL
FSETP.GEU.AND FSETP.GT.AND FSETP.GTU.AND FSETP.NEU.AND HADD2.F32
HGMMA.64x64x16.F16 HGMMA.64x64x16.F32.BF16 IADD3 IADD3.X
IMAD IMAD.HI IMAD.HI.U32 IMAD.IADD IMAD.MOV IMAD.MOV.U32 IMAD.SHL.U32
IMAD.U32 IMAD.WIDE IMAD.WIDE.U32 IMAD.X ISETP.EQ.AND ISETP.EQ.OR
ISETP.GE.AND ISETP.GE.AND.EX ISETP.GE.U32.AND ISETP.GT.AND
ISETP.GT.U32.AND ISETP.GT.U32.AND.EX ISETP.GT.U32.OR ISETP.LE.AND
ISETP.LT.AND ISETP.LT.U32.AND ISETP.NE.AND ISETP.NE.U32.AND
LDC LDC.64 LDG.E LDG.E.U16 LDL LDL.64 LDL.LU LDL.LU.64 LDS LDS.64 LDSM.16.M88.4
LEA LEA.HI LEA.HI.X LEA.HI.X.SX32 LOP3.LUT MEMBAR.ALL.CTA MOV MUFU.EX2
MUFU.RCP64H NOP P2R PRMT R2UR RET.REL.NODEC S2R S2UR SEL SGXT SGXT.U32
SHF.L.U32 SHF.L.U64.HI SHF.R.S32.HI SHF.R.U32.HI SHFL.BFLY STG.E STG.E.U16
STL STL.64 STL.S16 STS STS.64 STS.U16 UIADD3 UIADD3.64 UIADD3.X UIMAD ULDC.64
ULEA ULEA.HI.X ULEA.HI.X.SX32 ULOP3.LUT UMOV USGXT.U32 USHF.L.U32
USHF.R.S32.HI USHF.R.U32.HI VIADD VIADDMNMX VIMNMX
WARPGROUP.ARRIVE WARPGROUP.DEPBAR.LE WARPSYNC.ALL
F2F.BF16.F32 F2FP.BF16.F32.PACK_AB HMMA.16816.F32.BF16 ISETP.GE.U32.AND.EX
ISETP.GT.AND.EX LDS.128 LDSM.16.MT88.4 LEA.HI.SX32 MUFU.RCP PLOP3.LUT
STS.128 STSM.16.M88.2 STSM.16.MT88.4 UIMAD.WIDE.U32 UISETP.GE.AND
UISETP.GT.AND UISETP.LT.AND ULEA.HI USEL USGXT
""".split())


def refs(text):
    result = set(REG.findall(text))
    for m in re.finditer(r"\b(U?R)(\d+)\.64\b", text):
        result.add(f"{m[1]}{int(m[2])+1}")
    return result


def register_words(name, count=1):
    if name in ("RZ", "URZ", "PT", "UPT"):
        return set()
    m = re.fullmatch(r"(U?R|U?P)(\d+)", name)
    if not m:
        raise UnsupportedSpill(f"Unknown destination: {name}")
    return {m[1] + str(int(m[2]) + i) for i in range(count)}


def width(op):
    match = re.search(r"\.(U|S)?(8|16|32|64|128)(?:\.|$)", op)
    return int(match[2]) // 8 if match else 4


@dataclass(frozen=True)
class Instruction:
    pc: int
    op: str
    args: tuple
    predicate: str | None
    text: str

    @property
    def base(self):
        return self.op.split(".")[0]

    @property
    def local(self):
        return self.base in ("LDL", "STL")

    def local_offset(self):
        arg = self.args[0 if self.base == "STL" else 1]
        m = re.fullmatch(r"\[R1(?:\+(0x[0-9a-f]+))?\]", arg)
        if not m:
            raise UnsupportedSpill(f"Non-fixed frame addressing: {self.text}")
        return int(m[1] or "0", 0)

    def memory_tokens(self):
        if not self.local:
            return set()
        start = self.local_offset()
        return {f"L:{i}" for i in range(start, start + width(self.op))}

    def definitions(self):
        base, a = self.base, self.args
        known = {"STL", "LDL", "LDC", "ULDC", "LDG", "LDS", "LDSM", "HGMMA", "CS2R", "S2R", "S2UR",
                 "BRA", "EXIT", "BSSY", "BSYNC", "BREAK", "RET", "CALL", "NOP", "BAR", "MEMBAR", "FENCE",
                 "WARPGROUP", "WARPSYNC", "STG", "STS", "ISETP", "UISETP", "FSETP", "DSETP", "PSETP",
                 "SHFL", "R2P", "P2R", "R2UR", "DMUL", "DADD", "DFMA", "F2F", "F2FP", "FADD", "FFMA",
                 "FMNMX", "FMUL", "FSEL", "MUFU", "HADD2", "IMAD", "UIMAD", "IADD3", "UIADD3", "LEA",
                 "ULEA", "LOP3", "ULOP3", "MOV", "UMOV", "SEL", "SGXT", "USGXT", "SHF", "USHF", "VIADD", "PRMT", "VIADDMNMX", "VIMNMX",
                 "HMMA", "PLOP3", "STSM", "USEL"}
        if base not in known or self.op not in FOOTPRINT_OPS:
            raise UnsupportedSpill(f"Unknown register/side-effect footprint: {self.text}")
        if base == "STL":
            return self.memory_tokens()
        if base in ("BRA", "EXIT", "BSSY", "BSYNC", "BREAK", "RET", "CALL", "NOP", "BAR", "MEMBAR", "FENCE", "WARPGROUP", "WARPSYNC", "STG", "STS", "STSM"):
            return set()
        if base in ("ISETP", "UISETP", "FSETP", "DSETP", "PSETP", "PLOP3"):
            return register_words(a[0]) | register_words(a[1])
        if base == "SHFL":
            return register_words(a[0]) | register_words(a[1])
        if base == "R2P":
            return {f"P{i}" for i in range(7)}
        count = 1
        if base in ("LDL", "LDC", "ULDC", "LDG", "LDS"):
            count = max(1, width(self.op) // 4)
        elif base == "CS2R" or base in ("DMUL", "DADD", "DFMA") or self.op.startswith("F2F.F64."):
            count = 2
        elif base in ("IMAD", "UIMAD") and ".WIDE" in self.op:
            count = 2
        elif base == "UIADD3" and ".64" in self.op:
            count = 2
        elif base == "LDSM":
            count = int(self.op.split(".")[-1])
        elif self.op == "HMMA.16816.F32.BF16":
            count = 4
        elif base == "HGMMA":
            shape = re.search(r"\.(\d+)x(\d+)x\d+\.(F32|F16)", self.op)
            if not shape:
                raise UnsupportedSpill(f"Unknown accumulator shape: {self.text}")
            count = int(shape[1]) * int(shape[2]) // 128 // (2 if shape[3] == "F16" else 1)
        result = register_words(a[0], count)
        if base in ("LOP3", "ULOP3") and re.fullmatch(r"U?P(?:\d+|T)", a[0]):
            result |= register_words(a[1])
        if base in ("IADD3", "UIADD3", "LEA", "ULEA", "IMAD", "UIMAD", "LOP3", "ULOP3"):
            for arg in a[1:]:
                if not re.fullmatch(r"U?P(?:\d+|T)", arg):
                    break
                result |= register_words(arg)
        return result

    def dependencies(self, *, value=True):
        result = refs(self.predicate or "")
        if self.local:
            result.add("R1")
            if value:
                if self.base == "LDL":
                    result |= self.memory_tokens()
                else:
                    result |= refs(self.args[1])
                    if width(self.op) > 4 and self.args[1] not in ("RZ", "URZ"):
                        result |= register_words(self.args[1], width(self.op) // 4)
            return result
        if self.base == "LDG":
            # Hopper's descriptor is not a data address. Only the final bracket
            # supplies the flat pointer; readonly input bounds are checked.
            bracket = re.findall(r"\[([^]]+)\]", self.args[-1])
            if not bracket:
                raise UnsupportedSpill("Unknown global address")
            return result | refs(bracket[-1])
        if self.base == "P2R":
            mask = int(self.args[-1], 0)
            return result | refs(self.args[-2]) | {f"P{i}" for i in range(7) if mask & (1 << i)}
        start = 0 if self.base in ("BRA", "EXIT", "BREAK") else 1
        if self.base in ("ISETP", "UISETP", "FSETP", "DSETP", "PSETP", "PLOP3", "SHFL"):
            start = 2
        elif self.base in ("IADD3", "UIADD3", "LEA", "ULEA", "IMAD", "UIMAD", "LOP3", "ULOP3"):
            if self.base in ("LOP3", "ULOP3") and re.fullmatch(r"U?P(?:\d+|T)", self.args[0]):
                start = 2
            while start < len(self.args) and re.fullmatch(r"U?P(?:\d+|T)", self.args[start]):
                start += 1
        for arg in self.args[start:]:
            result |= refs(arg)
        if self.op == "HMMA.16816.F32.BF16":
            # A and B are packed BF16 fragments, C is a four-register FP32 accumulator.
            for arg, count in zip(self.args[1:], (4, 2, 4), strict=True):
                result |= register_words(arg.removesuffix(".reuse"), count)
        if self.base in ("DFMA", "DMUL", "DADD", "DSETP"):
            for arg in self.args[start:]:
                m = re.fullmatch(r"-?(R\d+)(?:\.reuse)?", arg)
                if m:
                    result |= register_words(m[1], 2)
        return result


def parse(sass):
    instructions = {}
    for line in sass.splitlines():
        m = re.search(r"/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*;", line)
        if not m:
            continue
        pc, text = int(m[1], 16), m[2]
        if pc in instructions:
            raise UnsupportedSpill("Multiple SASS functions are outside this projection")
        match = re.fullmatch(r"(?:@(!?U?P\d+)\s+)?([A-Za-z0-9_.]+)(?:\s+(.*))?", text)
        if not match:
            raise UnsupportedSpill(f"Unknown disassembly: {text}")
        args = tuple(a.strip() for a in (match[3] or "").split(",") if a.strip())
        instructions[pc] = Instruction(pc, match[2], args, match[1], text)
    if not instructions or min(instructions) != 0:
        raise UnsupportedSpill("Missing function entry")
    return instructions


def target(insn):
    try:
        return int(insn.args[-1], 0)
    except (ValueError, IndexError):
        raise UnsupportedSpill(f"Indirect control flow: {insn.text}") from None


class Projection:
    def __init__(self, sass):
        self.insns = parse(sass)
        ordered = sorted(self.insns)
        self.follow = {pc: (ordered[i+1] if i+1 < len(ordered) else None) for i, pc in enumerate(ordered)}
        self.successors = {}
        sync = {}
        self.call_clobbers = {}
        for pc in ordered:
            inst = self.insns[pc]
            base = inst.base
            if base.startswith(("BRX", "JMX", "JMP")):
                raise UnsupportedSpill(f"Indirect branch: {inst.text}")
            if base == "BSSY":
                sync[inst.args[0]] = target(inst)
            if base == "CALL":
                # Only forward, local, spill-free leaf helpers may be opaque.
                start = target(inst)
                returns = [p for p in ordered if p >= start and self.insns[p].base == "RET"]
                if not returns or start not in self.insns:
                    raise UnsupportedSpill("Unresolved device helper")
                end = min(returns)
                clobbers = set()
                for p in ordered:
                    if not start <= p <= end:
                        continue
                    body = self.insns[p]
                    if body.local or body.base in ("CALL", "STG", "STS", "EXIT", "BREAK"):
                        raise UnsupportedSpill("Helper has unsupported side effects")
                    if body.base == "BRA" and not p < target(body) <= end:
                        raise UnsupportedSpill("Helper has a backward/escaping edge")
                    clobbers |= body.definitions()
                self.call_clobbers[pc] = clobbers
            if base == "BRA":
                successors = {target(inst)}
                if inst.predicate or len(inst.args) != 1:
                    successors.add(self.follow[pc])
            elif base == "BREAK":
                if inst.args[-1] not in sync:
                    raise UnsupportedSpill("Unknown reconvergence target")
                successors = {sync[inst.args[-1]]}
                if inst.predicate:
                    successors.add(self.follow[pc])
            elif base in ("EXIT", "RET"):
                successors = {None}
                if inst.predicate:
                    successors.add(self.follow[pc])
            else:
                successors = {self.follow[pc]}
            if any(p is not None and p not in self.insns for p in successors):
                raise UnsupportedSpill("Missing branch target")
            self.successors[pc] = successors
        reachable, todo = set(), [0]
        while todo:
            pc = todo.pop()
            if pc is None or pc in reachable:
                continue
            reachable.add(pc)
            todo.extend(self.successors[pc])
        self.reachable = reachable
        self.locals = {pc for pc in reachable if self.insns[pc].local}
        self.blocks, self.block_of, self.block_next = self.basic_blocks()
        self.postdom, self.ipdom = self.postdominators()
        self.control = defaultdict(set)
        for block, successors in self.block_next.items():
            if block is None or len(successors) < 2:
                continue
            for runner in successors:
                seen = set()
                while runner != self.ipdom[block]:
                    if runner is None or runner in seen:
                        raise UnsupportedSpill("Unresolved control dependence")
                    seen.add(runner)
                    self.control[runner].add(self.blocks[block][-1])
                    runner = self.ipdom[runner]
        self.selected, self.live_after, self.value_needed = self.slice()
        self.live_specials = sorted({a for pc in self.selected for a in self.insns[pc].args if a.startswith("SR_")})

    def basic_blocks(self):
        leaders = {0}
        for pc in self.reachable:
            if self.insns[pc].base in ("BRA", "BREAK", "EXIT", "RET"):
                leaders |= {x for x in self.successors[pc] if x is not None}
                if self.follow[pc] in self.reachable:
                    leaders.add(self.follow[pc])
        blocks, owners = {}, {}
        for pc in sorted(self.reachable):
            if pc in leaders:
                leader = pc
                blocks[leader] = []
            blocks[leader].append(pc)
            owners[pc] = leader
        following = {b: {owners[p] if p is not None else None for p in self.successors[pcs[-1]]} for b, pcs in blocks.items()}
        following[None] = set()
        return blocks, owners, following

    def postdominators(self):
        universe = set(self.block_next)
        result = {b: ({None} if b is None else set(universe)) for b in universe}
        changed = True
        while changed:
            changed = False
            for b in sorted(self.blocks, reverse=True):
                following = self.block_next[b]
                value = {b} | set.intersection(*(result[s] for s in following))
                if value != result[b]:
                    result[b] = value
                    changed = True
        ipdom = {}
        for b in self.blocks:
            candidates = result[b] - {b}
            if not candidates:
                raise UnsupportedSpill("Control flow has no terminating postdominator")
            ipdom[b] = max(candidates, key=lambda p: len(result[p]))
        return result, ipdom

    def slice(self):
        roots = set(self.locals)
        definitions = {pc: self.call_clobbers.get(pc, self.insns[pc].definitions()) for pc in self.reachable}
        live = {b: set() for b in self.block_next}
        selected, after, needed_values = set(), {}, set()
        for outer in range(len(self.blocks) + 2):
            changed = True
            while changed:
                changed = False
                for block in sorted(self.blocks, reverse=True):
                    needed = set().union(*(live[s] for s in self.block_next[block]))
                    for pc in reversed(self.blocks[block]):
                        inst = self.insns[pc]
                        after[pc] = set(needed)
                        data_needed = bool(definitions[pc] & needed)
                        if pc in roots or data_needed:
                            selected.add(pc)
                            if data_needed:
                                needed_values.add(pc)
                            if not inst.predicate:
                                needed -= definitions[pc]
                            needed |= inst.dependencies(value=data_needed) if inst.local else inst.dependencies()
                    if needed != live[block]:
                        live[block] = needed
                        changed = True
            controls = {branch for pc in selected for branch in self.control[self.block_of[pc]]}
            if controls <= roots:
                break
            roots |= controls
        else:
            raise UnsupportedSpill("Slice did not converge")
        if live[0]:
            raise UnsupportedSpill(f"Slice needs undefined entry state: {sorted(live[0])}")
        if self.call_clobbers.keys() & selected:
            raise UnsupportedSpill("Opaque numerical helper influences local traffic")
        return selected, after, needed_values

    def description(self):
        return {"reachable_instructions": len(self.reachable), "selected_instructions": len(self.selected),
                "local_sites": len(self.locals), "basic_blocks": len(self.blocks), "live_special_registers": self.live_specials,
                "selected_opcodes": sorted({self.insns[p].op for p in self.selected}),
                "opaque_spill_free_helpers": sorted(self.call_clobbers),
                "method": "Backward data and control slice; skip unselected branches only at their CFG postdominator"}
