#!/usr/bin/env python3
"""
perf.py -- Phase 5 of the project: performance analysis across the models.

For each benchmark this runs:
  * the single-cycle datapath  -> instruction count (its CPI is 1 by design)
  * the pipeline               -> cycles, CPI, stalls, flushes
  * the pipeline + I/D caches  -> real hit/miss rates on the program's own
                                  access streams, then an effective CPI that
                                  adds memory-stall cycles (P&H style).

It then reports speedup and gives an Amdahl's-law reading.

Performance model (assumptions stated so the numbers are reproducible):
  * A pipeline stage takes 1 time unit, so the pipeline clock period = 1.
  * The single-cycle machine must do all 5 stages in one clock, so its clock
    period = 5 units (CPI 1, but a 5x longer cycle).
  * Execution time = instructions x CPI x clock_period.
  * A cache miss costs MISS_PENALTY extra cycles.
"""

from riscv_sim import Assembler
from datapath import SingleCycleCPU
from pipeline import PipelinedCPU
from cache_sim import Cache

STAGES = 5           # single-cycle clock period, in pipeline-stage units
MISS_PENALTY = 20    # extra cycles per cache miss


def make_caches():
    """Small caches so the tiny benchmarks show compulsory misses then hits."""
    icache = Cache(num_blocks=8, block_size_words=4, associativity=1,
                   addr_bits=32, name="I$")
    dcache = Cache(num_blocks=8, block_size_words=4, associativity=2,
                   addr_bits=32, name="D$")
    return icache, dcache


BENCHMARKS = {
    "vector-scan (A3 P1)": """
.data
buff:   .word 7, 7, 7, 7, 7, 7, 7, 7, 3, 9
.text
        la   s0, buff
        li   a0, 7
        li   s2, 0
loop:   slli t0, s2, 2
        add  t0, s0, t0
        lw   t1, 0(t0)
        bne  t1, a0, done
        addi s2, s2, 1
        j    loop
done:   ecall
""",
    "func-call (A3 P2)": """
.text
        li   a0, 6
        li   a1, 10
        call test
        ecall
test:   addi sp, sp, -8
        sw   ra, 4(sp)
        sw   s0, 0(sp)
        add  s0, a0, a1
        call abs
        sub  s0, s0, a0
        mv   a0, s0
        lw   s0, 0(sp)
        lw   ra, 4(sp)
        addi sp, sp, 8
        ret
abs:    bge  a0, a1, abs_xy
        sub  a0, a1, a0
        ret
abs_xy: sub  a0, a0, a1
        ret
""",
    "sum-loop": """
.text
        li   t0, 0          # sum
        li   t1, 0          # i
        li   t2, 20         # n
loop:   add  t0, t0, t1
        addi t1, t1, 1
        blt  t1, t2, loop
        ecall
""",
    "alu-chain": """
.text
        li   t0, 1
        addi t1, t0, 1
        addi t2, t1, 1
        add  t3, t2, t1
        sub  t4, t3, t2
        xor  t5, t4, t3
        ecall
""",
}


def run_benchmark(name, src):
    asm = Assembler(); words = asm.assemble(src)

    # single-cycle: instruction count
    sc = SingleCycleCPU(); sc.load_program(words, asm.data)
    n_instr = sc.run()

    # pipeline without cache
    p = PipelinedCPU(); p.load_program(words, asm.data); p.run()
    cpi = p.cpi()

    # pipeline with caches (fresh run, observing real streams)
    ic, dc = make_caches()
    pc = PipelinedCPU(icache=ic, dcache=dc)
    pc.load_program(words, asm.data); pc.run()
    total_misses = ic.misses + dc.misses
    mem_stall_per_instr = total_misses * MISS_PENALTY / n_instr
    cpi_eff = cpi + mem_stall_per_instr

    # times and speedups
    t_single = n_instr * 1.0 * STAGES
    t_pipe = n_instr * cpi * 1.0
    t_pipe_cache = n_instr * cpi_eff * 1.0

    return {
        "name": name, "instr": n_instr, "cpi": cpi,
        "stalls": p.stalls, "flushes": p.flushes,
        "i_acc": ic.accesses, "i_miss": ic.misses, "i_hr": ic.hit_rate,
        "d_acc": dc.accesses, "d_miss": dc.misses, "d_hr": dc.hit_rate,
        "cpi_eff": cpi_eff,
        "speedup_pipe": t_single / t_pipe,
        "speedup_cache": t_single / t_pipe_cache,
        "mem_stall_frac": mem_stall_per_instr / cpi_eff if cpi_eff else 0,
    }


def main():
    rows = [run_benchmark(n, s) for n, s in BENCHMARKS.items()]

    print("PIPELINE BEHAVIOR")
    print(f"  {'benchmark':<22}{'instr':>6}{'CPI':>7}{'stalls':>8}{'flushes':>9}")
    for r in rows:
        print(f"  {r['name']:<22}{r['instr']:>6}{r['cpi']:>7.3f}"
              f"{r['stalls']:>8}{r['flushes']:>9}")

    print("\nCACHE BEHAVIOR (I$: 8x4w direct, D$: 8x4w 2-way, LRU)")
    print(f"  {'benchmark':<22}{'I-hit':>8}{'D-acc':>7}{'D-hit':>8}")
    for r in rows:
        print(f"  {r['name']:<22}{r['i_hr']:>7.1%}{r['d_acc']:>7}{r['d_hr']:>8.1%}")

    print(f"\nSPEEDUP vs single-cycle (clock: single={STAGES}u, pipe=1u; "
          f"miss penalty={MISS_PENALTY})")
    print(f"  {'benchmark':<22}{'CPI':>7}{'CPI_eff':>9}{'pipe x':>8}{'+cache x':>9}")
    for r in rows:
        print(f"  {r['name']:<22}{r['cpi']:>7.3f}{r['cpi_eff']:>9.3f}"
              f"{r['speedup_pipe']:>8.2f}{r['speedup_cache']:>9.2f}")

    # ---- Amdahl's law reading on the worst memory-bound benchmark ----
    worst = max(rows, key=lambda r: r["mem_stall_frac"])
    f = worst["mem_stall_frac"]
    print("\nAMDAHL'S LAW READING")
    print(f"  On '{worst['name']}', memory stalls are {f:.1%} of execution time.")
    print(f"  Even with a perfect (0-miss) cache, the most you could speed up")
    print(f"  THAT run is 1/(1-{f:.3f}) = {1/(1-f):.2f}x -- the rest is compute.")


if __name__ == "__main__":
    main()
