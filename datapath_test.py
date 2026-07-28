#!/usr/bin/env python3
"""
datapath_test.py -- verify the Phase-2 single-cycle datapath.

Strategy: run the same programs on BOTH the trusted Phase-0 functional
simulator (riscv_sim.CPU) and the new block-structured datapath
(datapath.SingleCycleCPU), and require that all 32 registers agree. Two
independent implementations landing on the same state is strong evidence the
datapath's control signals, ALU wiring, and muxes are correct.

Also checks the Assignment 3 programs land on their known answers
(Problem 1 -> s2 = 4, Problem 2 -> a0 = 12).
"""

from riscv_sim import Assembler, CPU, to_s32
from datapath import SingleCycleCPU

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    PASS += bool(cond)
    FAIL += (not cond)
    if not cond:
        print(f"  [FAIL] {name}  {detail}")


def run_both(src):
    asm1 = Assembler(); w1 = asm1.assemble(src)
    ref = CPU(); ref.load_program(w1, asm1.data); ref.run()

    asm2 = Assembler(); w2 = asm2.assemble(src)
    dut = SingleCycleCPU(); dut.load_program(w2, asm2.data); dut.run()
    return ref, dut


def cross_check(name, src):
    ref, dut = run_both(src)
    mism = [i for i in range(32) if ref.x[i] != dut.rf.read(i)]
    check(f"{name}: registers match", not mism,
          detail=(f"differ at {mism}: "
                  + ", ".join(f"x{i} ref={to_s32(ref.x[i])} dut={to_s32(dut.rf.read(i))}"
                              for i in mism)) if mism else "")


# ---- a broad ALU / instruction workout ----
ALU_WORKOUT = """
.text
        li   t0, 12
        li   t1, 5
        add  t2, t0, t1
        sub  t3, t0, t1
        and  t4, t0, t1
        or   t5, t0, t1
        xor  t6, t0, t1
        slt  s1, t1, t0
        sltu s3, t1, t0
        slli s4, t0, 2
        srli s5, t0, 1
        li   a2, -8
        srai s6, a2, 1
        slti a3, t1, 7
        xori a4, t0, -1
        lui  s7, 1
        auipc s8, 0
        ecall
"""

# ---- control-flow: count to 5 with a signed branch ----
BRANCH_LOOP = """
.text
        li   a0, 0
        li   a1, 5
loop:
        addi a0, a0, 1
        blt  a0, a1, loop
        ecall
"""

# ---- loads/stores through the data segment ----
MEM_TEST = """
.data
arr:    .word 11, 22, 33, 44
.text
        la   t0, arr
        lw   t1, 0(t0)
        lw   t2, 8(t0)
        add  t3, t1, t2
        sw   t3, 12(t0)
        lw   t4, 12(t0)
        ecall
"""


def a3_problem1():
    src = """
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
    ref, dut = run_both(src)
    check("A3 P1 datapath s2==4", dut.get_reg("s2") == 4,
          f"got {dut.get_reg('s2')}")
    cross_check("A3 P1", src)


def a3_problem2():
    src = """
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
    ref, dut = run_both(src)
    check("A3 P2 datapath a0==12", to_s32(dut.get_reg("a0")) == 12,
          f"got {to_s32(dut.get_reg('a0'))}")
    cross_check("A3 P2", src)


if __name__ == "__main__":
    cross_check("ALU workout", ALU_WORKOUT)
    cross_check("branch loop", BRANCH_LOOP)
    cross_check("mem test", MEM_TEST)
    a3_problem1()
    a3_problem2()
    total = PASS + FAIL
    print(f"Datapath tests: {PASS}/{total} checks passed"
          + (f"  ({FAIL} FAILED)" if FAIL else ""))
    raise SystemExit(1 if FAIL else 0)
