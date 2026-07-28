#!/usr/bin/env python3
"""
alu_test.py -- correctness tests for the Phase 1 ALU.

Every ALU operation is checked against an independent Python reference over
edge cases (0, 1, -1, INT_MIN, INT_MAX) and many random inputs, including the
flags (zero / carry-out / overflow / negative) for add and subtract.
"""

import random
from alu import ALU, ALUControl, WIDTH, MASK, SIGN_BIT

alu = ALU()
PASS = FAIL = 0


def signed(x):
    return x - (1 << WIDTH) if x & SIGN_BIT else x


def check(name, got, expected):
    global PASS, FAIL
    ok = got == expected
    PASS += ok
    FAIL += (not ok)
    if not ok:
        print(f"  [FAIL] {name}: got {got}, expected {expected}")


EDGE = [0, 1, 2, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x55555555, 0xAAAAAAAA]


def operands():
    pairs = [(a, b) for a in EDGE for b in EDGE]
    pairs += [(random.getrandbits(32), random.getrandbits(32)) for _ in range(2000)]
    return pairs


def test_add():
    for a, b in operands():
        res, fl = alu.compute(ALUControl.ADD, a, b)
        check(f"ADD {a:#x}+{b:#x}", res, (a + b) & MASK)
        check(f"ADD carry", fl.carry_out, (a + b) >> WIDTH & 1)
        ssum = signed(a) + signed(b)
        check(f"ADD overflow", fl.overflow, int(not (-(1 << 31) <= ssum < (1 << 31))))
        check(f"ADD zero", fl.zero, int(((a + b) & MASK) == 0))


def test_sub():
    for a, b in operands():
        res, fl = alu.compute(ALUControl.SUB, a, b)
        check(f"SUB {a:#x}-{b:#x}", res, (a - b) & MASK)
        # borrow: carry_out of a + ~b + 1
        raw = a + ((~b) & MASK) + 1
        check("SUB carry", fl.carry_out, (raw >> WIDTH) & 1)
        sdiff = signed(a) - signed(b)
        check("SUB overflow", fl.overflow, int(not (-(1 << 31) <= sdiff < (1 << 31))))


def test_logic():
    for a, b in operands():
        check("AND", alu.compute(ALUControl.AND, a, b)[0], a & b)
        check("OR", alu.compute(ALUControl.OR, a, b)[0], a | b)
        check("XOR", alu.compute(ALUControl.XOR, a, b)[0], a ^ b)
        check("NOR", alu.compute(ALUControl.NOR, a, b)[0], (~(a | b)) & MASK)


def test_compare():
    for a, b in operands():
        check("SLT", alu.compute(ALUControl.SLT, a, b)[0],
              int(signed(a) < signed(b)))
        check("SLTU", alu.compute(ALUControl.SLTU, a, b)[0], int(a < b))


def test_shifts():
    for a, _ in operands():
        for sh in (0, 1, 7, 15, 31):
            check("SLL", alu.compute(ALUControl.SLL, a, 0, sh)[0], (a << sh) & MASK)
            check("SRL", alu.compute(ALUControl.SRL, a, 0, sh)[0], a >> sh)
            expected = (signed(a) >> sh) & MASK          # arithmetic shift
            check("SRA", alu.compute(ALUControl.SRA, a, 0, sh)[0], expected)


if __name__ == "__main__":
    random.seed(5375)
    test_add()
    test_sub()
    test_logic()
    test_compare()
    test_shifts()
    total = PASS + FAIL
    print(f"ALU tests: {PASS}/{total} checks passed"
          + (f"  ({FAIL} FAILED)" if FAIL else ""))
    raise SystemExit(1 if FAIL else 0)
