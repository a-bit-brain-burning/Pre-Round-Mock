"""Explicit cache hypotheses applied to verified dynamic local-access streams."""

from collections import OrderedDict
from functools import lru_cache
import math


def sectors(offset, width, warp, frame):
    return tuple(s for s, _ in sector_masks(offset, width, warp, frame))


@lru_cache(maxsize=65536)
def sector_masks(offset, width, warp, frame):
    # Assumed local layout: consecutive lanes, then consecutive 32-bit words.
    # Only uniform 32-lane accesses are accepted by the trace generator.
    base = warp * math.ceil(frame/16)*16*32
    addresses = (base + ((offset+i)//4)*128 + lane*4 + (offset+i)%4
                 for lane in range(32) for i in range(width))
    masks = {}
    for a in addresses:
        masks[a//32] = masks.get(a//32, 0) | (1 << (a % 32))
    return tuple(sorted(masks.items()))


@lru_cache(maxsize=4096)
def cached_service(events, warps, frame, capacity, interleave, write_through):
    cache = OrderedDict()
    counters = {"read_sectors": 0, "write_sectors": 0, "load_hit_sectors": 0,
                "load_miss_sectors": 0, "l2_read_sectors": 0, "l2_write_sectors": 0,
                "dirty_final_sectors": 0, "warp_instructions": 0}
    capacity = max(0, int(capacity)//32)
    streams = [[(op, sector_masks(off, n, warp, frame)) for _, op, off, n, active in events if active]
               for warp in range(warps)]
    schedule = ((streams[w][i]) for i in range(len(streams[0])) for w in range(warps)) if interleave else (
                event for stream in streams for event in stream)
    for op, touched in schedule:
        counters["warp_instructions"] += 1
        for sector, mask in touched:
            read = op == "LDL"
            counters["read_sectors" if read else "write_sectors"] += 1
            present = sector in cache
            if read:
                counters["load_hit_sectors" if present else "load_miss_sectors"] += 1
                if not present:
                    counters["l2_read_sectors"] += 1
            elif write_through or not capacity:
                counters["l2_write_sectors"] += 1
            # Full-sector stores need no ownership read. Partial stores fetch
            # the sector on an allocating miss, preserving untouched bytes.
            if not read and not present and capacity and mask != 0xffffffff:
                counters["l2_read_sectors"] += 1
            dirty = (cache.pop(sector, False) or not read) and not write_through
            if capacity:
                cache[sector] = dirty
                if len(cache) > capacity:
                    _, evicted_dirty = cache.popitem(last=False)
                    counters["l2_write_sectors"] += int(evicted_dirty)
    counters["dirty_final_sectors"] = sum(cache.values())
    return counters


def service(trace, launch, occupancy, p):
    resident = occupancy["ctas_per_sm"]
    # Reserve a configurable local-cache share, capped by shared/L1 carveout.
    available = max(0, p["unified_l1_bytes"] - resident*occupancy["allocated_shared_per_cta"])
    budget = min(p["spill_l1_budget_bytes"], available) / resident
    counts = {k: 0 for k in ("read_sectors", "write_sectors", "load_hit_sectors", "load_miss_sectors",
                            "l2_read_sectors", "l2_write_sectors", "dirty_final_sectors", "warp_instructions")}
    for group in trace["classes"]:
        stats = cached_service(tuple(tuple(e) for e in group["events"]), group["warps"], launch["resources"]["stack"],
                               int(budget), bool(p["spill_warp_interleave"]), bool(p["spill_write_through"]))
        for k in counts:
            counts[k] += stats[k]*group["cta_multiplicity"]
    read = counts["l2_read_sectors"]*32
    write = (counts["l2_write_sectors"] + p["spill_final_writeback_fraction"]*counts["dirty_final_sectors"])*32
    # L2 residency is still a bounded hypothesis, not a fabricated hit count.
    footprint = launch["resources"]["stack"]*launch["threads"]*min(math.prod(launch["grid"]), p["sm_count"]*resident)
    l2_hit = p["spill_l2_hit"]*min(1, p["l2_bytes"]/max(1, footprint))
    return {**counts, "l1_budget_per_cta": int(budget), "l2_bytes": read+write,
            "hbm_bytes": (read+write)*(1-l2_hit), "l1_sector_bytes": 32*(counts["read_sectors"]+counts["write_sectors"]),
            "instruction_count": counts["warp_instructions"], "effective_l2_hit_assumption": l2_hit,
            "load_hit_fraction": counts["load_hit_sectors"]/max(1, counts["read_sectors"]),
            "assumptions": "32-byte sectors, word-interleaved lanes, fully associative LRU, cold per-CTA cache partition, write allocate with ownership read only for partial sectors; .LU treated as ordinary load. Warp order/store policy varied. L2/HBM hit fraction and final dirty drain uncalibrated."}
