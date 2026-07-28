#!/usr/bin/env python3
"""
cache_demo.py -- reproduce Assignment 7 (Memory Architecture) with cache_sim.

Problem 2(b): the given 10-bit read address sequence is run through a
Direct-Mapped, 2-way, 4-way, and Fully-Associative cache (16 blocks each,
1-word blocks, LRU) and the per-access Index / Tag / Hit-Miss table plus
hit and miss rates are printed for each.

Problem 1(b): the CPI-slowdown calculation.

Also serves as the automated test: the hand-verified hit counts are
asserted, so a wrong result fails loudly.
"""

from cache_sim import Cache, cpi_with_cache

# A7 Problem 2(b) read sequence (10-bit addresses)
TRACE = [0x058, 0x068, 0x058, 0x268, 0x068, 0x340,
         0x360, 0x368, 0x178, 0x068, 0x340, 0x368]

# Hand-verified expected hit counts (byte-addressed, 1-word block => 2 offset bits).
# Direct-mapped thrashes at one index (0x068/0x268/0x368 collide); the
# associative caches resolve those conflict misses.
EXPECTED_HITS = {"Direct-Mapped": 2, "2-way": 5, "4-way": 5, "Fully-Associative": 5}


def build(name, associativity):
    return Cache(num_blocks=16, block_size_words=1, associativity=associativity,
                 addr_bits=10, name=name)


def main():
    print("A7 Problem 2(b): 16 blocks, 1-word blocks, LRU, 10-bit addresses\n")
    configs = [
        ("Direct-Mapped", 1),
        ("2-way", 2),
        ("4-way", 4),
        ("Fully-Associative", 16),
    ]
    failures = 0
    for name, assoc in configs:
        c = build(name, assoc)
        for addr in TRACE:
            c.access(addr)
        c.report()
        exp = EXPECTED_HITS[name]
        ok = c.hits == exp
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: "
              f"hits={c.hits}, expected {exp}\n")

    print("-" * 60)
    print("A7 Problem 1(b): CPI slowdown from cache misses")
    r = cpi_with_cache(base_cpi=1.2, data_access_frac=0.20,
                       miss_rate=0.01, miss_penalty=200)
    print(f"  accesses/instruction   = {r['accesses_per_instr']:.2f}  "
          f"(1 instr-fetch + 0.20 data)")
    print(f"  stall cycles/instr     = {r['stall_cycles_per_instr']:.2f}")
    print(f"  CPI with misses        = {r['cpi_with_misses']:.2f}")
    print(f"  => processor is {r['slowdown_x']:.2f}x slower than the ideal case")
    # data-access-only interpretation, for comparison
    r2 = cpi_with_cache(count_instruction_fetch=False)
    print(f"  (if only data accesses counted: {r2['slowdown_x']:.3f}x slower)")
    print("-" * 60)

    # AMAT example tie-in
    print("\nExample AMAT (hit time = 1 cycle, miss penalty = 200 cycles):")
    for name, assoc in configs:
        c = build(name, assoc)
        for addr in TRACE:
            c.access(addr)
        print(f"  {name:>18}: miss rate {c.miss_rate:5.1%}  "
              f"AMAT = {c.amat(1, 200):6.2f} cycles")

    print()
    if failures:
        print(f"RESULT: {failures} configuration(s) failed.")
        raise SystemExit(1)
    print("RESULT: all cache configurations match hand-verified hit counts.")


if __name__ == "__main__":
    main()
