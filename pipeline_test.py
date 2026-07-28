#!/usr/bin/env python3
"""
pipeline_test.py -- verify the Phase-3 pipeline and report CPI.

The pipeline must land on exactly the same architectural state as the
single-cycle datapath (Phase 2), which itself matches the Phase-0 simulator.
So for every program we require all 32 registers to agree. Then we print the
CPI each program achieved, which shows the effect of stalls and flushes.

These programs deliberately include back-to-back dependencies (forwarding),
a load immediately used (load-use stall), and branches/jumps (flushes).
"""

from riscv_sim import Assembler, to_s32
from datapath import SingleCycleCPU
from pipeline import PipelinedCPU

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += (not cond)
    if not cond:
        print(f"  [FAIL] {name}  {detail}")


def compare(name, src):
    a1 = Assembler(); w1 = a1.assemble(src)
    ref = SingleCycleCPU(); ref.load_program(w1, a1.data); ref.run()

    a2 = Assembler(); w2 = a2.assemble(src)
    dut = PipelinedCPU(); dut.load_program(w2, a2.data); dut.run()

    mism = [i for i in range(32) if ref.rf.read(i) != dut.rf.read(i)]
    check(f"{name}: registers match single-cycle", not mism,
          detail=", ".join(f"x{i} ref={to_s32(ref.rf.read(i))} "
                           f"pipe={to_s32(dut.rf.read(i))}" for i in mism))
    print(f"    {name:<16} cycles={dut.cycles:>4}  retired={dut.retired:>4}  "
          f"CPI={dut.cpi():.3f}  stalls={dut.stalls}  flushes={dut.flushes}")
    return dut


# dependent chain -> exercises forwarding heavily
FORWARD_CHAIN = """
.text
        li   t0, 1
        addi t1, t0, 1
        addi t2, t1, 1
        addi t3, t2, 1
        add  t4, t3, t2
        sub  t5, t4, t1
        ecall
"""

# load immediately used -> forces a load-use stall
LOAD_USE = """
.data
v:      .word 100, 200
.text
        la   t0, v
        lw   t1, 0(t0)
        addi t2, t1, 5
        lw   t3, 4(t0)
        add  t4, t3, t2
        ecall
"""

# branch + jump heavy -> exercises flushes
CONTROL_HEAVY = """
.text
        li   a0, 0
        li   a1, 4
loop:
        addi a0, a0, 1
        blt  a0, a1, loop
        jal  skip
        li   a2, 999
skip:
        li   a3, 7
        ecall
"""

A3P1 = """
.data
buff:   .word 7, 7, 7, 7, 3, 9
.text
        la   s0, buff
        li   a0, 7
        li   s2, 0
loop:
        slli t0, s2, 2
        add  t0, s0, t0
        lw   t1, 0(t0)
        bne  t1, a0, done
        addi s2, s2, 1
        j    loop
done:
        ecall
"""

A3P2 = """
.text
        li   a0, 6
        li   a1, 10
        call test
        ecall
test:
        addi sp, sp, -8
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
abs:
        bge  a0, a1, abs_xy
        sub  a0, a1, a0
        ret
abs_xy:
        sub  a0, a0, a1
        ret
"""

if __name__ == "__main__":
    print("Cross-check (pipeline vs single-cycle) + CPI:\n")
    compare("forward chain", FORWARD_CHAIN)
    compare("load-use", LOAD_USE)
    compare("control heavy", CONTROL_HEAVY)
    compare("A3 Problem 1", A3P1)
    d = compare("A3 Problem 2", A3P2)
    check("A3 P2 a0 == 12", to_s32(d.get_reg("a0")) == 12)

    total = PASS + FAIL
    print(f"\nPipeline tests: {PASS}/{total} checks passed"
          + (f"  ({FAIL} FAILED)" if FAIL else ""))
    raise SystemExit(1 if FAIL else 0)
