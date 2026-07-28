#!/usr/bin/env python3
"""
alu.py -- Phase 1 of the project: a bit-level Arithmetic Logic Unit.

CS 5375: Computer Systems Organization and Architecture

The ALU is the calculator at the center of the processor.  Everything the
datapath does numerically -- add, subtract, compare, AND/OR, shift -- happens
here.  To show the hardware from the ground up, addition is NOT done with
Python's `+`; it is built from single-bit full adders wired into a ripple-carry
chain, exactly as gates would be on a chip.  Subtraction then reuses that same
adder via two's complement (invert the second operand, add 1).  A single
`ALUControl` code selects which operation runs -- this is the wire the control
unit will drive in Phase 2 (the single-cycle datapath).

Outputs of every operation:
    result   : 32-bit value
    Flags(zero, carry_out, overflow, negative)

These flags are what branch and set-less-than logic read.
"""

from collections import namedtuple

WIDTH = 32
MASK = (1 << WIDTH) - 1
SIGN_BIT = 1 << (WIDTH - 1)

Flags = namedtuple("Flags", ["zero", "carry_out", "overflow", "negative"])


# ---------------------------------------------------------------------------
# Gate-level addition
# ---------------------------------------------------------------------------
def full_adder(a, b, cin):
    """One bit of addition. sum = a XOR b XOR cin; carry = majority(a,b,cin)."""
    s = a ^ b ^ cin
    cout = (a & b) | (b & cin) | (a & cin)
    return s, cout


def ripple_add(a, b, cin=0):
    """
    Add two WIDTH-bit numbers one bit at a time, threading the carry.
    Returns (result, carry_out, overflow) where overflow is the signed
    overflow flag = carry_into_MSB XOR carry_out_of_MSB.
    """
    a &= MASK
    b &= MASK
    result = 0
    carry = cin
    carry_into_msb = 0
    for i in range(WIDTH):
        abit = (a >> i) & 1
        bbit = (b >> i) & 1
        s, carry_out = full_adder(abit, bbit, carry)
        result |= (s << i)
        if i == WIDTH - 1:
            carry_into_msb = carry          # carry entering the top bit
        carry = carry_out
    overflow = carry_into_msb ^ carry       # classic two's-complement overflow test
    return result & MASK, carry, overflow


# ---------------------------------------------------------------------------
# ALU control codes -- the select lines the control unit drives
# ---------------------------------------------------------------------------
class ALUControl:
    AND = 0b0000
    OR = 0b0001
    ADD = 0b0010
    SUB = 0b0110
    SLT = 0b0111    # set-less-than, signed
    NOR = 0b1100
    XOR = 0b1101
    SLL = 0b0011    # shift left logical
    SRL = 0b0100    # shift right logical
    SRA = 0b0101    # shift right arithmetic
    SLTU = 0b1000   # set-less-than, unsigned

    names = {AND: "AND", OR: "OR", ADD: "ADD", SUB: "SUB", SLT: "SLT",
             NOR: "NOR", XOR: "XOR", SLL: "SLL", SRL: "SRL", SRA: "SRA",
             SLTU: "SLTU"}


def _flags(result, carry_out, overflow):
    return Flags(zero=int(result == 0),
                 carry_out=int(bool(carry_out)),
                 overflow=int(bool(overflow)),
                 negative=int(bool(result & SIGN_BIT)))


class ALU:
    """A combinational 32-bit ALU. `compute` is one 'clock-less' evaluation."""

    def compute(self, ctrl, a, b, shamt=0):
        a &= MASK
        b &= MASK

        if ctrl == ALUControl.ADD:
            result, cout, ovf = ripple_add(a, b, 0)
            return result, _flags(result, cout, ovf)

        if ctrl == ALUControl.SUB:
            # a - b  ==  a + (~b) + 1
            result, cout, ovf = ripple_add(a, (~b) & MASK, 1)
            return result, _flags(result, cout, ovf)

        if ctrl == ALUControl.AND:
            result = a & b
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.OR:
            result = a | b
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.XOR:
            result = a ^ b
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.NOR:
            result = (~(a | b)) & MASK
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.SLT:
            # signed: compute a-b, then look at sign XOR overflow
            diff, cout, ovf = ripple_add(a, (~b) & MASK, 1)
            neg = 1 if (diff & SIGN_BIT) else 0
            result = neg ^ ovf
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.SLTU:
            # unsigned: a<b iff the subtract produced a borrow (carry_out==0)
            diff, cout, ovf = ripple_add(a, (~b) & MASK, 1)
            result = 0 if cout else 1
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.SLL:
            result = (a << (shamt & 0x1F)) & MASK
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.SRL:
            result = (a >> (shamt & 0x1F)) & MASK
            return result, _flags(result, 0, 0)

        if ctrl == ALUControl.SRA:
            sh = shamt & 0x1F
            if a & SIGN_BIT:                      # sign-extend during the shift
                result = ((a >> sh) | (MASK << (WIDTH - sh))) & MASK
            else:
                result = (a >> sh) & MASK
            return result, _flags(result, 0, 0)

        raise ValueError(f"unknown ALU control code: {ctrl:#06b}")
