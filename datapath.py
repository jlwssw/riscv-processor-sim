#!/usr/bin/env python3
"""
datapath.py -- Phase 2 of the project: a single-cycle RISC-V datapath.

CS 5375: Computer Systems Organization and Architecture

Phase 0 executed instructions with one big function. Here we re-express that
same execution as the explicit *hardware blocks* of a single-cycle processor
and the wires between them, so you can see how one instruction flows through
the machine in one clock:

     PC -> Instruction Memory -> Decode/Control
        -> Register File (read) -> [ImmGen / mux] -> ALU (Phase 1)
        -> Data Memory -> Writeback mux -> Register File (write)
        -> next-PC logic

The Control Unit reads the opcode and asserts control signals (RegWrite,
ALUSrc, MemRead, MemWrite, ResultSrc, Branch, Jump, ALUOp). The ALU Control
block turns ALUOp + funct fields into one of the Phase-1 ALU control codes.
Nothing here uses Python arithmetic for the datapath's math -- the ALU from
alu.py does it, gate-built adder and all.

Run:
    python datapath.py a3_problem1.s            # run a program
    python datapath.py a3_problem2.s --trace    # show control signals per instr
    python datapath.py --table                  # print the control-signal table
"""

import sys
from collections import namedtuple

from riscv_sim import Assembler, sign_extend, to_u32, to_s32, TEXT_BASE, DATA_BASE
from alu import ALU, ALUControl


# ---------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------
OP_RTYPE = 0x33
OP_ITYPE = 0x13
OP_LOAD = 0x03
OP_STORE = 0x23
OP_BRANCH = 0x63
OP_LUI = 0x37
OP_AUIPC = 0x17
OP_JAL = 0x6F
OP_JALR = 0x67
OP_ECALL = 0x73

# High-level ALUOp classes the control unit emits
ALUOP_ADD = "ADD"       # address / PC math -> ALU always adds
ALUOP_SUB = "SUB"       # branch comparison -> ALU subtracts
ALUOP_RTYPE = "RTYPE"   # decode from funct3/funct7
ALUOP_ITYPE = "ITYPE"   # decode from funct3 (+funct7 for shifts)

# ResultSrc: what the writeback mux selects
RES_ALU = "ALU"
RES_MEM = "MEM"
RES_PC4 = "PC+4"
RES_IMM = "IMM"

Control = namedtuple("Control", [
    "reg_write", "alu_src", "mem_read", "mem_write",
    "result_src", "branch", "jump", "jump_reg", "a_src", "alu_op",
])
# a_src: which value feeds ALU input A -> "rs1", "pc", or "zero"


def control_unit(opcode):
    """Decode the opcode into the datapath's control signals."""
    if opcode == OP_RTYPE:
        return Control(1, 0, 0, 0, RES_ALU, 0, 0, 0, "rs1", ALUOP_RTYPE)
    if opcode == OP_ITYPE:
        return Control(1, 1, 0, 0, RES_ALU, 0, 0, 0, "rs1", ALUOP_ITYPE)
    if opcode == OP_LOAD:
        return Control(1, 1, 1, 0, RES_MEM, 0, 0, 0, "rs1", ALUOP_ADD)
    if opcode == OP_STORE:
        return Control(0, 1, 0, 1, RES_ALU, 0, 0, 0, "rs1", ALUOP_ADD)
    if opcode == OP_BRANCH:
        return Control(0, 0, 0, 0, RES_ALU, 1, 0, 0, "rs1", ALUOP_SUB)
    if opcode == OP_LUI:
        return Control(1, 1, 0, 0, RES_IMM, 0, 0, 0, "zero", ALUOP_ADD)
    if opcode == OP_AUIPC:
        return Control(1, 1, 0, 0, RES_ALU, 0, 0, 0, "pc", ALUOP_ADD)
    if opcode == OP_JAL:
        return Control(1, 0, 0, 0, RES_PC4, 0, 1, 0, "pc", ALUOP_ADD)
    if opcode == OP_JALR:
        return Control(1, 1, 0, 0, RES_PC4, 0, 1, 1, "rs1", ALUOP_ADD)
    if opcode == OP_ECALL:
        return Control(0, 0, 0, 0, RES_ALU, 0, 0, 0, "rs1", ALUOP_ADD)
    raise ValueError(f"no control signals for opcode {opcode:#04x}")


def alu_control(alu_op, funct3, funct7):
    """ALU Control block: ALUOp + funct fields -> a Phase-1 ALU control code."""
    if alu_op == ALUOP_ADD:
        return ALUControl.ADD
    if alu_op == ALUOP_SUB:
        return ALUControl.SUB
    # R-type and I-type share the funct3 map; funct7 bit 5 distinguishes
    # sub vs add and sra vs srl.
    is_alt = (funct7 == 0x20)
    if funct3 == 0x0:
        # addi has no "sub-immediate"; only R-type uses funct7 for SUB
        return ALUControl.SUB if (alu_op == ALUOP_RTYPE and is_alt) else ALUControl.ADD
    if funct3 == 0x1:
        return ALUControl.SLL
    if funct3 == 0x2:
        return ALUControl.SLT
    if funct3 == 0x3:
        return ALUControl.SLTU
    if funct3 == 0x4:
        return ALUControl.XOR
    if funct3 == 0x5:
        return ALUControl.SRA if is_alt else ALUControl.SRL
    if funct3 == 0x6:
        return ALUControl.OR
    if funct3 == 0x7:
        return ALUControl.AND
    raise ValueError(f"bad funct3 {funct3}")


# ---------------------------------------------------------------------------
# Hardware blocks
# ---------------------------------------------------------------------------
class RegisterFile:
    def __init__(self):
        self.x = [0] * 32
        self.x[2] = 0x7FFFF000          # sp near top of memory

    def read(self, i):
        return self.x[i]

    def write(self, i, val):
        if i != 0:                       # x0 is hardwired to zero
            self.x[i] = to_u32(val)


class Memory:
    def __init__(self):
        self.m = {}

    def load(self, addr, size, signed):
        val = 0
        for i in range(size):
            val |= self.m.get(addr + i, 0) << (8 * i)
        return sign_extend(val, size * 8) if signed else val

    def store(self, addr, size, val):
        for i in range(size):
            self.m[addr + i] = (val >> (8 * i)) & 0xFF


def immediate_generator(instr, opcode):
    """Produce the sign-extended immediate for the instruction's format."""
    if opcode in (OP_ITYPE, OP_LOAD, OP_JALR):          # I-type
        return sign_extend(instr >> 20, 12)
    if opcode == OP_STORE:                              # S-type
        imm = ((instr >> 25) << 5) | ((instr >> 7) & 0x1F)
        return sign_extend(imm, 12)
    if opcode == OP_BRANCH:                             # B-type
        imm = (((instr >> 31) & 1) << 12) | (((instr >> 7) & 1) << 11)
        imm |= ((instr >> 25) & 0x3F) << 5
        imm |= ((instr >> 8) & 0xF) << 1
        return sign_extend(imm, 13)
    if opcode in (OP_LUI, OP_AUIPC):                    # U-type
        return to_u32((instr >> 12) << 12)
    if opcode == OP_JAL:                                # J-type
        imm = (((instr >> 31) & 1) << 20) | (((instr >> 12) & 0xFF) << 12)
        imm |= ((instr >> 20) & 1) << 11
        imm |= ((instr >> 21) & 0x3FF) << 1
        return sign_extend(imm, 21)
    return 0


class SingleCycleCPU:
    def __init__(self, trace=False):
        self.rf = RegisterFile()
        self.mem = Memory()
        self.alu = ALU()
        self.pc = TEXT_BASE
        self.halted = False
        self.trace = trace

    def load_program(self, words, data=b"", data_base=DATA_BASE):
        for addr, w in words:
            self.mem.store(addr, 4, w)
        for i, b in enumerate(data):
            self.mem.m[data_base + i] = b

    def branch_taken(self, funct3, a, b):
        """Compare a,b via the ALU's subtract flags, decode by funct3."""
        _, fl = self.alu.compute(ALUControl.SUB, a, b)
        lt_signed = fl.negative ^ fl.overflow          # a < b (signed)
        lt_unsigned = 0 if fl.carry_out else 1          # a < b (unsigned)
        return {
            0x0: fl.zero == 1,          # beq
            0x1: fl.zero == 0,          # bne
            0x4: lt_signed == 1,        # blt
            0x5: lt_signed == 0,        # bge
            0x6: lt_unsigned == 1,      # bltu
            0x7: lt_unsigned == 0,      # bgeu
        }[funct3]

    def step(self):
        # ---- FETCH ----
        instr = self.mem.load(self.pc, 4, False)
        opcode = instr & 0x7F
        if opcode == OP_ECALL:
            self.halted = True
            return

        # ---- DECODE + CONTROL ----
        rd = (instr >> 7) & 0x1F
        funct3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        funct7 = (instr >> 25) & 0x7F
        shamt = (instr >> 20) & 0x1F
        ctrl = control_unit(opcode)
        imm = immediate_generator(instr, opcode)

        # ---- REGISTER READ ----
        rv1 = self.rf.read(rs1)
        rv2 = self.rf.read(rs2)

        # ---- ALU input muxes ----
        # Control signals steer each operand. Input A is normally rs1 (but the PC
        # for auipc/jal, or 0 for lui); input B is the immediate when ALUSrc is set,
        # otherwise the second register rs2.
        a = {"rs1": rv1, "pc": self.pc, "zero": 0}[ctrl.a_src]
        b = imm if ctrl.alu_src else rv2
        actrl = alu_control(ctrl.alu_op, funct3, funct7)
        alu_result, _ = self.alu.compute(actrl, a, b, shamt)

        # ---- next-PC logic ----
        next_pc = to_u32(self.pc + 4)
        if ctrl.branch and self.branch_taken(funct3, rv1, rv2):
            next_pc = to_u32(self.pc + imm)
        elif ctrl.jump and ctrl.jump_reg:                 # jalr
            next_pc = to_u32((rv1 + imm) & ~1)
        elif ctrl.jump:                                   # jal
            next_pc = to_u32(self.pc + imm)

        # ---- MEMORY ----
        mem_data = 0
        if ctrl.mem_read:
            size = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}[funct3]
            signed = funct3 in (0, 1, 2)
            mem_data = to_u32(self.mem.load(alu_result, size, signed))
        if ctrl.mem_write:
            size = {0: 1, 1: 2, 2: 4}[funct3]
            self.mem.store(alu_result, size, rv2)

        # ---- WRITEBACK mux ----
        # ResultSrc chooses what value is written back to rd: the ALU output, data
        # read from memory (loads), PC+4 (the return address for jal/jalr), or the
        # immediate (lui).
        result = {
            RES_ALU: alu_result,
            RES_MEM: mem_data,
            RES_PC4: to_u32(self.pc + 4),
            RES_IMM: imm,
        }[ctrl.result_src]
        if ctrl.reg_write:
            self.rf.write(rd, result)

        if self.trace:
            print(f"  pc={self.pc:08x} {instr:08x} op={opcode:#04x} "
                  f"RegWrite={ctrl.reg_write} ALUSrc={ctrl.alu_src} "
                  f"Mem(r{ctrl.mem_read}/w{ctrl.mem_write}) "
                  f"Res={ctrl.result_src} Br={ctrl.branch} Jmp={ctrl.jump} "
                  f"-> rd x{rd}={to_s32(result)}")

        self.pc = next_pc

    def run(self, max_steps=1_000_000):
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        if steps >= max_steps:
            raise RuntimeError("step limit exceeded")
        return steps

    def get_reg(self, name):
        from riscv_sim import reg
        return self.rf.read(reg(name))

    def dump(self):
        names = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
                 "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                 "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
                 "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
        print("Register file:")
        for i in range(0, 32, 4):
            print("  " + "  ".join(f"{names[i+j]:>4}={to_s32(self.rf.read(i+j)):>11}"
                                   for j in range(4)))


CONTROL_TABLE_ROWS = [
    ("R-type",  OP_RTYPE), ("I-type",  OP_ITYPE), ("load",    OP_LOAD),
    ("store",   OP_STORE), ("branch",  OP_BRANCH), ("lui",     OP_LUI),
    ("auipc",   OP_AUIPC), ("jal",     OP_JAL),   ("jalr",    OP_JALR),
]


def print_control_table():
    hdr = ("instr", "RegWr", "ALUSrc", "MemRd", "MemWr", "Result", "Branch", "Jump", "ALUOp")
    print("{:<8}{:<7}{:<8}{:<7}{:<7}{:<8}{:<8}{:<6}{}".format(*hdr))
    for name, op in CONTROL_TABLE_ROWS:
        c = control_unit(op)
        print("{:<8}{:<7}{:<8}{:<7}{:<7}{:<8}{:<8}{:<6}{}".format(
            name, c.reg_write, c.alu_src, c.mem_read, c.mem_write,
            c.result_src, c.branch, c.jump, c.alu_op))


def run_file(path, trace=False):
    with open(path) as f:
        src = f.read()
    asm = Assembler()
    words = asm.assemble(src)
    cpu = SingleCycleCPU(trace=trace)
    cpu.load_program(words, asm.data)
    steps = cpu.run()
    print(f"Executed {steps} instructions.\n")
    cpu.dump()
    return cpu


if __name__ == "__main__":
    if "--table" in sys.argv:
        print_control_table()
    elif len(sys.argv) >= 2:
        run_file(sys.argv[1], trace="--trace" in sys.argv)
    else:
        print(__doc__)
