"""Execute the integer/control slice, never the numerical SASS kernel."""

from collections import Counter
import re

from spill_sass import UnsupportedSpill, target, width

MASK = 0xffffffff
STACK_TOP = 0x10000000


def signed(x):
    return (x & MASK) - (0x100000000 if x & 0x80000000 else 0)


def abi_parameters(elf, launch, buffers):
    """Read the actual cubin KPARAM offsets, rather than assume packed args."""
    bank = re.findall(r"EIATTR_PARAM_CBANK\s+Format:\s+EIFMT_SVAL\s+Value:\s+0x[0-9a-f]+\s+0x([0-9a-f]+)", elf)
    entries = re.findall(r"Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)", elf)
    args = launch["arguments"]
    if len(bank) != 1 or len(entries) != len(args):
        raise UnsupportedSpill("Unrecognized cubin parameter ABI")
    encoded = int(bank[0], 16)
    base, size = encoded & 0xffff, encoded >> 16
    result = {0x28 + i: b for i, b in enumerate(STACK_TOP.to_bytes(4, "little"))}
    seen = set()
    for ordinal, offset, count in entries:
        ordinal, offset, count = int(ordinal, 16), int(offset, 16), int(count, 16)
        if ordinal in seen or ordinal >= len(args) or offset + count > size:
            raise UnsupportedSpill("Invalid parameter layout")
        seen.add(ordinal)
        a = args[ordinal]
        if a["kind"] == "buffer":
            expected, bits = 8, buffers[a["name"]][0] + a["offset"]
        else:
            expected, bits = (8 if a["dtype"].endswith("64") else 4), a["bits"]
        if expected != count:
            raise UnsupportedSpill("Scalar/pointer ABI width mismatch")
        for i, b in enumerate(bits.to_bytes(count, "little")):
            if base + offset + i in result:
                raise UnsupportedSpill("Overlapping parameter layout")
            result[base + offset + i] = b
    return result, {"base": base, "size": size, "parameter_count": len(entries)}


class Lane:
    def __init__(self, projection, launch, constants, buffers, cta, tid):
        self.q, self.launch, self.constants, self.buffers = projection, launch, constants, buffers
        self.cta, self.tid = cta, tid
        self.reg = {}
        self.local = {}
        self.events, self.branches, self.global_reads = [], [], Counter()
        self.local_origins, self.padding_reads = {}, Counter()

    def get(self, operand):
        operand = operand.replace(".reuse", "")
        if operand.startswith("!"):
            return int(not self.get(operand[1:]))
        if operand.startswith("-"):
            return -self.get(operand[1:])
        if operand in ("RZ", "URZ"):
            return 0
        if operand in ("PT", "UPT"):
            return 1
        if re.fullmatch(r"U?[RP]\d+", operand):
            if operand not in self.reg or self.reg[operand] is None:
                raise UnsupportedSpill(f"Undefined live value {operand}")
            return self.reg[operand]
        try:
            return int(operand, 0)
        except ValueError:
            raise UnsupportedSpill(f"Unknown live operand {operand}") from None

    def put(self, name, value):
        if name not in ("RZ", "URZ", "PT", "UPT"):
            self.reg[name] = None if value is None else value & MASK

    def pair(self, name, count):
        if name in ("RZ", "URZ"):
            return [name] * count
        m = re.fullmatch(r"(U?R)(\d+)", name)
        if not m:
            raise UnsupportedSpill("Invalid register vector")
        return [m[1] + str(int(m[2]) + i) for i in range(count)]

    def execute(self, inst):
        a, op, b = inst.args, inst.op, inst.base
        v = self.get
        if inst.local:
            offset, count = inst.local_offset(), width(op)
            frame = self.launch["resources"]["stack"]
            if v("R1") != STACK_TOP - frame or offset < 0 or offset + count > frame:
                raise UnsupportedSpill("Local address escapes audited stack frame")
            active = not inst.predicate or bool(v(inst.predicate))
            self.events.append([inst.pc, b, offset, count, int(active)])
            if not active:
                return
            needed = inst.pc in self.q.value_needed
            if b == "STL":
                data = []
                for reg in self.pair(a[1], (count+3)//4):
                    data.extend(v(reg).to_bytes(4, "little") if needed else [None]*4)
                self.local.update({offset+i: data[i] for i in range(count)})
                self.local_origins.update({offset+i: (offset, count) for i in range(count)})
            else:
                missing = {i for i in range(count) if offset+i not in self.local}
                if missing:
                    if (needed or count != 4 or missing != {2, 3}
                            or any(self.local_origins.get(offset+i) != (offset, 2) for i in (0, 1))):
                        raise UnsupportedSpill("Load of uninitialized local bytes")
                    # Observed S16 -> U32 spill packing. The WHOLE result is
                    # dead for this projection, so retain unknown bits rather
                    # than fabricate a numerical value. Still count four bytes.
                    self.padding_reads[inst.pc] += 1
                data = [self.local.get(offset+i) for i in range(count)]
                if needed and None in data:
                    raise UnsupportedSpill("Unresolved spilled data influences control/address")
                for i, reg in enumerate(self.pair(a[0], (count+3)//4)):
                    value = int.from_bytes(bytes(data[i*4:(i+1)*4]), "little") if needed else None
                    self.put(reg, value)
            return
        if inst.predicate and not v(inst.predicate):
            return
        if b in ("LDC", "ULDC") and op in (b, b+".64"):
            m = re.fullmatch(r"c\[0x0\]\[0x([0-9a-f]+)\]", a[1])
            if not m:
                raise UnsupportedSpill("Unknown live constant bank")
            offset, n = int(m[1], 16), width(op)
            if any(offset+i not in self.constants for i in range(n)):
                raise UnsupportedSpill("Live constant absent from parameter ABI")
            for i, reg in enumerate(self.pair(a[0], n//4)):
                self.put(reg, int.from_bytes(bytes(self.constants[offset+4*i+j] for j in range(4)), "little"))
        elif op == "CS2R" and a[1] == "SRZ":
            for reg in self.pair(a[0], 2):
                self.put(reg, 0)
        elif b in ("S2R", "S2UR"):
            specials = {"SR_TID.X": self.tid, "SR_CTAID.X": self.cta[0], "SR_CTAID.Y": self.cta[1], "SR_CTAID.Z": self.cta[2]}
            if a[1] not in specials:
                raise UnsupportedSpill(f"Unsupported live special register {a[1]}")
            self.put(a[0], specials[a[1]])
        elif op == "LDG.E":
            m = re.search(r"\[(R\d+)\.64(?:\+0x([0-9a-f]+))?\]$", a[1])
            if not m:
                raise UnsupportedSpill("Unsupported live global address")
            lo, hi = self.pair(m[1], 2)
            address = v(lo) + (v(hi) << 32) + int(m[2] or "0", 16)
            matches = [(name, address-base, data) for name, (base, data, readonly) in self.buffers.items()
                       if readonly and base <= address and address+4 <= base+len(data)]
            if len(matches) != 1 or matches[0][0] not in ("qoff", "koff"):
                raise UnsupportedSpill("Live load is not a verified GQA offset")
            name, offset, data = matches[0]
            if offset % 4:
                raise UnsupportedSpill("Misaligned offset input")
            self.global_reads[(name, offset)] += 1
            self.put(a[0], int.from_bytes(data[offset:offset+4], "little"))
        elif op in ("IMAD.MOV.U32", "IMAD.IADD", "IMAD.SHL.U32"):
            self.put(a[0], v(a[1])*v(a[2])+v(a[3]))
        elif op == "VIADD":
            self.put(a[0], v(a[1])+v(a[2]))
        elif op == "UIADD3" and len(a) == 4:
            self.put(a[0], v(a[1]) + v(a[2]) + v(a[3]))
        elif op == "USHF.L.U32" and a[3] == "URZ" and 0 <= v(a[2]) < 32:
            self.put(a[0], v(a[1]) << v(a[2]))
        elif op == "PLOP3.LUT" and a[1:4] == ("PT", "PT", "PT") and a[5:] == ("0x80", "0x0"):
            # With the first two inputs true, LUT 0x80 copies its third predicate.
            self.put(a[0], v(a[4]))
        elif op in ("IADD3", "IADD3.X"):
            start = next(i for i in range(1, len(a)) if not re.fullmatch(r"P\d+|PT", a[i]))
            value = sum(v(x) for x in a[start:start+3])
            if op.endswith(".X"):
                value += sum(v(x) for x in a[start+3:])
            self.put(a[0], value)
            for pred in a[1:start]:
                self.put(pred, int(value > MASK))
        elif op == "UISETP.GT.AND" or re.fullmatch(r"ISETP\.(?:GE|GT|LE|LT|NE|EQ)(?:\.U32)?\.AND", op):
            left, right = v(a[2]) & MASK, v(a[3]) & MASK
            if ".U32" not in op:
                left, right = signed(left), signed(right)
            test = {"GE": left >= right, "GT": left > right, "LE": left <= right,
                    "LT": left < right, "NE": left != right, "EQ": left == right}[op.split(".")[1]]
            guard = bool(v(a[4]))
            self.put(a[0], int(test and guard))
            self.put(a[1], int(not test and guard))
        elif op == "SEL":
            self.put(a[0], v(a[1]) if v(a[3]) else v(a[2]))
        elif op == "VIMNMX":
            left, right = signed(v(a[1])), signed(v(a[2]))
            self.put(a[0], min(left, right) if v(a[3]) else max(left, right))
        elif op == "LOP3.LUT" and len(a) == 6 and a[-1] == "!PT" and a[0].startswith("R"):
            x, y, z, lut = (v(t) & MASK for t in a[1:5])
            result = 0
            for bit in range(32):
                index = ((x >> bit & 1) << 2) | ((y >> bit & 1) << 1) | (z >> bit & 1)
                result |= ((lut >> index) & 1) << bit
            self.put(a[0], result)
        elif op == "LEA":
            carry = a[1].startswith("P")
            start = 2 if carry else 1
            value = (v(a[start]) << v(a[start+2])) + v(a[start+1])
            self.put(a[0], value)
            if carry:
                self.put(a[1], int(value > MASK))
        elif op == "LEA.HI":
            value = (((v(a[3]) << 32) | v(a[1])) << v(a[4])) >> 32
            self.put(a[0], value + v(a[2]))
        elif op == "LEA.HI.X.SX32":
            self.put(a[0], ((signed(v(a[1])) << v(a[3])) >> 32) + v(a[2]) + v(a[4]))
        elif op == "SHF.R.S32.HI":
            if a[1] != "RZ" or not 0 <= v(a[2]) <= 31:
                raise UnsupportedSpill("Unsupported signed funnel shift")
            self.put(a[0], signed(v(a[3])) >> v(a[2]))
        elif op == "SHF.L.U64.HI":
            if not 0 <= v(a[2]) <= 31:
                raise UnsupportedSpill("Unsupported funnel shift count")
            self.put(a[0], (((v(a[3]) << 32) | v(a[1])) << v(a[2])) >> 32)
        else:
            raise UnsupportedSpill(f"Unsupported live instruction: {inst.text}")

    def run(self, limit=2000000):
        pc, steps = 0, 0
        while pc is not None:
            steps += 1
            if steps > limit:
                raise UnsupportedSpill("Projected execution exceeded step bound")
            inst = self.q.insns[pc]
            if inst.base in ("BRA", "BREAK", "EXIT", "RET"):
                if pc not in self.q.selected:
                    # All skipped paths are proven irrelevant to local traffic.
                    pc = self.q.ipdom[self.q.block_of[pc]]
                    continue
                if inst.base != "BRA" or len(inst.args) != 1:
                    raise UnsupportedSpill("Unsupported live branch encoding")
                taken = not inst.predicate or bool(self.get(inst.predicate))
                self.branches.append([pc, int(taken)])
                pc = target(inst) if taken else self.q.follow[pc]
                continue
            if pc in self.q.selected:
                try:
                    self.execute(inst)
                except UnsupportedSpill as error:
                    raise UnsupportedSpill(f"PC 0x{pc:x}: {error}") from error
            pc = self.q.follow[pc]
        return self
