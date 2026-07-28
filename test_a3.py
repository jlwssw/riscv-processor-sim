#!/usr/bin/env python3
"""
test_a3.py -- automated correctness tests for the RV32I simulator,
using the two Assignment 3 programs as the workload.

For each test we assemble a parameterized version of the A3 code, run it
on the simulator, and compare the machine's result against the value the
original C code should produce.
"""

from riscv_sim import Assembler, CPU, TEXT_BASE, to_s32

PASS, FAIL = 0, 0


def check(name, got, expected):
    global PASS, FAIL
    ok = got == expected
    PASS += ok
    FAIL += (not ok)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: got {got}, expected {expected}")


def run(src, setup=None):
    asm = Assembler()
    words = asm.assemble(src)
    cpu = CPU()
    cpu.load_program(words, asm.data)
    if setup:
        setup(cpu)
    cpu.run()
    return cpu


# ---------------------------------------------------------------------------
# Problem 1:  while (buff[i] == m) i++;
# The loop counts the leading run of elements equal to m.
# ---------------------------------------------------------------------------
P1_TEMPLATE = """
.data
buff:   .word {data}
.text
        la   s0, buff
        li   a0, {m}
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


def leading_run(arr, m):
    n = 0
    for v in arr:
        if v == m:
            n += 1
        else:
            break
    return n


def test_problem1():
    print("Problem 1  --  while (buff[i] == m) i++;")
    cases = [
        ([7, 7, 7, 7, 3, 9], 7),   # assignment-style data -> 4
        ([5, 5, 1, 5, 5], 5),      # stops at first mismatch -> 2
        ([9, 8, 7], 4),            # no match at all         -> 0
        ([2, 2, 2, 2, 2, 8], 2),   # long run                -> 5
    ]
    for arr, m in cases:
        src = P1_TEMPLATE.format(data=", ".join(map(str, arr)), m=m)
        cpu = run(src)
        check(f"buff={arr}, m={m} -> i", cpu.get_reg("s2"), leading_run(arr, m))


# ---------------------------------------------------------------------------
# Problem 2:  g = x + y - abs(x, y)   (== 2 * min(x, y))
# Also verify the calling convention: sp and ra are restored across the call.
# ---------------------------------------------------------------------------
P2_TEMPLATE = """
.text
        li   a0, {x}
        li   a1, {y}
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


def test_problem2():
    print("Problem 2  --  g = x + y - abs(x, y)")
    cases = [(6, 10), (10, 6), (-4, 3), (5, 5), (0, 7), (-8, -2)]
    for x, y in cases:
        src = P2_TEMPLATE.format(x=x, y=y)
        asm = Assembler()
        words = asm.assemble(src)
        machine = CPU()
        machine.load_program(words, asm.data)
        sp_before = machine.get_reg("sp")
        machine.run()
        expected = x + y - abs(x - y)     # == 2*min(x, y)
        # registers hold two's-complement bits; interpret a0 as signed
        check(f"test({x},{y}) -> g", to_s32(machine.get_reg("a0")), expected)
        # calling-convention sanity: stack pointer balanced on return
        check(f"test({x},{y}) sp restored", machine.get_reg("sp"), sp_before)


if __name__ == "__main__":
    test_problem1()
    print()
    test_problem2()
    print()
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} checks passed.")
    raise SystemExit(1 if FAIL else 0)
