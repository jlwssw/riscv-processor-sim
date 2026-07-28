#!/usr/bin/env python3
"""
riscv_sim.py -- A small RISC-V RV32I assembler + instruction-set simulator.

CS 5375: Computer Systems Organization and Architecture
Make-up project: demonstrates assembly, loops, functions, the calling
convention, and how instructions are encoded and executed by hardware.

Pipeline of this program:
    source (.s text)
      -> Assembler:  two passes -> 32-bit machine-code words in memory
      -> CPU:        fetch / decode / execute loop with a real byte PC

Supported RV32I subset (enough to run Assignment 3):
    R-type : add sub sll slt sltu xor srl sra or and
    I-type : addi slti sltiu xori ori andi slli srli srai
             lb lh lw lbu lhu   jalr
    S-type : sb sh sw
    B-type : beq bne blt bge bltu bgeu
    U-type : lui auipc
    J-type : jal
    system : ecall            (halts the machine)
    pseudo : li mv j jal call ret nop la beqz bnez  (expanded by assembler)

Usage:
    python riscv_sim.py program.s          # assemble + run, dump registers
    python riscv_sim.py program.s --trace  # also print each executed instr
"""

import sys
import re

# ---------------------------------------------------------------------------
# Register name tables (numeric x0..x31 and the ABI names used in class)
# ---------------------------------------------------------------------------
ABI = {
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7,
    "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15, "a6": 16, "a7": 17,
    "s2": 18, "s3": 19, "s4": 20, "s5": 21, "s6": 22, "s7": 23,
    "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31,
}
for _i in range(32):
    ABI[f"x{_i}"] = _i

TEXT_BASE = 0x00000000   # where instructions are loaded
DATA_BASE = 0x00002000   # where .data lives (low 12 bits are 0 -> easy la)


def reg(name):
    name = name.strip().lower()
    if name not in ABI:
        raise ValueError(f"unknown register: {name}")
    return ABI[name]


def sign_extend(value, bits):
    """Interpret the low `bits` of value as a two's-complement signed number."""
    mask = (1 << bits) - 1
    value &= mask
    if value & (1 << (bits - 1)):
        value -= (1 << bits)
    return value


def to_u32(x):
    return x & 0xFFFFFFFF


def to_s32(x):
    return sign_extend(x, 32)


# ---------------------------------------------------------------------------
# Encoding tables
# ---------------------------------------------------------------------------
R_TYPE = {
    "add": (0x33, 0x0, 0x00), "sub": (0x33, 0x0, 0x20),
    "sll": (0x33, 0x1, 0x00), "slt": (0x33, 0x2, 0x00),
    "sltu": (0x33, 0x3, 0x00), "xor": (0x33, 0x4, 0x00),
    "srl": (0x33, 0x5, 0x00), "sra": (0x33, 0x5, 0x20),
    "or": (0x33, 0x6, 0x00), "and": (0x33, 0x7, 0x00),
}
I_ARITH = {
    "addi": (0x13, 0x0), "slti": (0x13, 0x2), "sltiu": (0x13, 0x3),
    "xori": (0x13, 0x4), "ori": (0x13, 0x6), "andi": (0x13, 0x7),
}
I_SHIFT = {"slli": (0x13, 0x1, 0x00), "srli": (0x13, 0x5, 0x00), "srai": (0x13, 0x5, 0x20)}
LOADS = {"lb": (0x03, 0x0), "lh": (0x03, 0x1), "lw": (0x03, 0x2),
         "lbu": (0x03, 0x4), "lhu": (0x03, 0x5)}
STORES = {"sb": (0x23, 0x0), "sh": (0x23, 0x1), "sw": (0x23, 0x2)}
BRANCHES = {"beq": (0x63, 0x0), "bne": (0x63, 0x1), "blt": (0x63, 0x4),
            "bge": (0x63, 0x5), "bltu": (0x63, 0x6), "bgeu": (0x63, 0x7)}


def enc_r(op, f3, f7, rd, rs1, rs2):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def enc_i(op, f3, rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def enc_i_shift(op, f3, f7, rd, rs1, shamt):
    return (f7 << 25) | ((shamt & 0x1F) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def enc_s(op, f3, rs1, rs2, imm):
    imm &= 0xFFF
    hi = (imm >> 5) & 0x7F
    lo = imm & 0x1F
    return (hi << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (lo << 7) | op


def enc_b(op, f3, rs1, rs2, imm):
    # B-type scatters the 13-bit branch offset across non-adjacent fields, and bit 0
    # is always 0 because branch targets are 2-byte aligned. The packing is awkward
    # for software but keeps the register fields in fixed positions for the hardware.
    imm &= 0x1FFF  # 13-bit, bit0 always 0
    b12 = (imm >> 12) & 1
    b11 = (imm >> 11) & 1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    return (b12 << 31) | (b10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
           (f3 << 12) | (b4_1 << 8) | (b11 << 7) | op


def enc_u(op, rd, imm):
    return ((imm & 0xFFFFF) << 12) | (rd << 7) | op


def enc_j(op, rd, imm):
    imm &= 0x1FFFFF  # 21-bit, bit0 always 0
    b20 = (imm >> 20) & 1
    b10_1 = (imm >> 1) & 0x3FF
    b11 = (imm >> 11) & 1
    b19_12 = (imm >> 12) & 0xFF
    return (b20 << 31) | (b10_1 << 21) | (b11 << 20) | (b19_12 << 12) | (rd << 7) | op


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------
class Assembler:
    def __init__(self):
        self.labels = {}
        self.data = bytearray()      # bytes for .data segment
        self.text = []               # list of (source_line, tokens) for .text

    def parse(self, src):
        section = "text"
        raw_text = []               # (addr, mnemonic, args, srcline)
        pc = TEXT_BASE
        dpc = DATA_BASE
        # ---- pass 1: collect labels, lay out data, expand pseudo counts ----
        for line in src.splitlines():
            line = line.split("#")[0].rstrip()      # strip comments
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped in (".text", ".data"):
                section = stripped[1:]
                continue
            # label?
            m = re.match(r"^([A-Za-z_.][\w.]*):\s*(.*)$", stripped)
            if m:
                label, rest = m.group(1), m.group(2)
                if section == "text":
                    self.labels[label] = pc
                else:
                    self.labels[label] = dpc
                stripped = rest.strip()
                if not stripped:
                    continue
            if section == "data":
                parts = stripped.split(None, 1)
                directive = parts[0]
                operand = parts[1] if len(parts) > 1 else ""
                if directive == ".word":
                    for tok in operand.split(","):
                        val = int(tok.strip(), 0)
                        self.data += (val & 0xFFFFFFFF).to_bytes(4, "little")
                        dpc += 4
                else:
                    raise ValueError(f"unsupported data directive: {directive}")
                continue
            # text: mnemonic + args
            parts = stripped.split(None, 1)
            mnem = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            expanded = self.expand_pseudo(mnem, args)
            for (em, ea) in expanded:
                raw_text.append((pc, em, ea, stripped))
                pc += 4
        self.raw_text = raw_text

    def expand_pseudo(self, mnem, args):
        """Return list of (real_mnemonic, args) after pseudo-instruction expansion."""
        a = [x.strip() for x in args.split(",")] if args else []
        if mnem == "nop":
            return [("addi", "x0, x0, 0")]
        if mnem == "mv":
            return [("addi", f"{a[0]}, {a[1]}, 0")]
        if mnem == "ret":
            return [("jalr", "x0, ra, 0")]
        if mnem == "j":
            return [("jal", f"x0, {a[0]}")]
        if mnem == "call":
            return [("jal", f"ra, {a[0]}")]
        if mnem == "jal" and len(a) == 1:          # jal label  == jal ra, label
            return [("jal", f"ra, {a[0]}")]
        if mnem == "beqz":
            return [("beq", f"{a[0]}, x0, {a[1]}")]
        if mnem == "bnez":
            return [("bne", f"{a[0]}, x0, {a[1]}")]
        if mnem == "li":
            imm = int(a[1], 0)
            if -2048 <= imm <= 2047:
                return [("addi", f"{a[0]}, x0, {imm}")]
            hi, lo = self._split_hi_lo(imm)
            return [("lui", f"{a[0]}, {hi}"), ("addi", f"{a[0]}, {a[0]}, {lo}")]
        if mnem == "la":
            # resolved to lui/addi in pass 2 (needs label address); mark specially
            return [("lui", f"{a[0]}, %hi({a[1]})"), ("addi", f"{a[0]}, {a[0]}, %lo({a[1]})")]
        return [(mnem, args)]

    @staticmethod
    def _split_hi_lo(addr):
        lo = addr & 0xFFF
        if lo & 0x800:                       # lo would be negative when sign-extended
            hi = ((addr + 0x1000) >> 12) & 0xFFFFF
            lo = lo - 0x1000
        else:
            hi = (addr >> 12) & 0xFFFFF
        return hi, lo

    def resolve_hilo(self, tok, kind, cur_pc):
        m = re.match(r"%(hi|lo)\(([\w.]+)\)", tok)
        if not m:
            return int(tok, 0)
        which, label = m.group(1), m.group(2)
        addr = self.labels[label]
        hi, lo = self._split_hi_lo(addr)
        return hi if which == "hi" else lo

    def assemble(self, src):
        self.parse(src)
        words = []       # (addr, 32-bit word)
        for (pc, mnem, args, srcline) in self.raw_text:
            a = [x.strip() for x in args.split(",")] if args else []
            try:
                w = self.encode(pc, mnem, a)
            except Exception as e:
                raise ValueError(f"error assembling '{srcline}': {e}")
            words.append((pc, w))
        return words

    def encode(self, pc, mnem, a):
        if mnem in R_TYPE:
            op, f3, f7 = R_TYPE[mnem]
            return enc_r(op, f3, f7, reg(a[0]), reg(a[1]), reg(a[2]))
        if mnem in I_ARITH:
            op, f3 = I_ARITH[mnem]
            imm = self.resolve_hilo(a[2], "lo", pc)
            return enc_i(op, f3, reg(a[0]), reg(a[1]), imm)
        if mnem in I_SHIFT:
            op, f3, f7 = I_SHIFT[mnem]
            return enc_i_shift(op, f3, f7, reg(a[0]), reg(a[1]), int(a[2], 0))
        if mnem in LOADS:
            op, f3 = LOADS[mnem]
            rd = reg(a[0])
            off, rs1 = self._mem_operand(a[1])
            return enc_i(op, f3, rd, rs1, off)
        if mnem in STORES:
            op, f3 = STORES[mnem]
            rs2 = reg(a[0])
            off, rs1 = self._mem_operand(a[1])
            return enc_s(op, f3, rs1, rs2, off)
        if mnem in BRANCHES:
            op, f3 = BRANCHES[mnem]
            target = self.labels[a[2]]
            return enc_b(op, f3, reg(a[0]), reg(a[1]), target - pc)
        if mnem == "lui":
            return enc_u(0x37, reg(a[0]), self.resolve_hilo(a[1], "hi", pc))
        if mnem == "auipc":
            return enc_u(0x17, reg(a[0]), int(a[1], 0))
        if mnem == "jal":
            target = self.labels[a[1]]
            return enc_j(0x6F, reg(a[0]), target - pc)
        if mnem == "jalr":
            off, rs1 = (int(a[2], 0), reg(a[1])) if len(a) == 3 else self._mem_operand(a[1])
            return enc_i(0x67, 0x0, reg(a[0]), rs1, off)
        if mnem == "ecall":
            return 0x00000073
        raise ValueError(f"unknown instruction: {mnem}")

    @staticmethod
    def _mem_operand(tok):
        """Parse 'offset(reg)'."""
        m = re.match(r"(-?\w+)?\((\w+)\)", tok.strip())
        if not m:
            raise ValueError(f"bad memory operand: {tok}")
        off = int(m.group(1), 0) if m.group(1) else 0
        return off, reg(m.group(2))


# ---------------------------------------------------------------------------
# CPU: fetch / decode / execute
# ---------------------------------------------------------------------------
class CPU:
    def __init__(self, trace=False):
        self.x = [0] * 32
        self.pc = TEXT_BASE
        self.mem = {}          # byte-addressable sparse memory
        self.trace = trace
        self.halted = False
        self.x[2] = 0x7FFFF000  # sp near top of memory

    # ---- memory helpers (little-endian) ----
    def load(self, addr, size, signed):
        val = 0
        for i in range(size):
            val |= self.mem.get(addr + i, 0) << (8 * i)
        return sign_extend(val, size * 8) if signed else val

    def store(self, addr, size, val):
        for i in range(size):
            self.mem[addr + i] = (val >> (8 * i)) & 0xFF

    def load_program(self, words, data=b"", data_base=DATA_BASE):
        for (addr, w) in words:
            self.store(addr, 4, w)
        for i, b in enumerate(data):
            self.mem[data_base + i] = b

    def set_reg(self, name, val):
        self.x[reg(name)] = to_u32(val)

    def get_reg(self, name):
        return self.x[reg(name)]

    def run(self, max_steps=1_000_000):
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        if steps >= max_steps:
            raise RuntimeError("step limit exceeded (possible infinite loop)")
        return steps

    def step(self):
        # Fetch the 32-bit word at the PC, then decode it by slicing out the fixed
        # bit-fields. The opcode (low 7 bits) identifies the format; each shift-and-
        # mask below pulls one field down to its low bits.
        instr = self.load(self.pc, 4, False)
        op = instr & 0x7F
        rd = (instr >> 7) & 0x1F
        f3 = (instr >> 12) & 0x7
        rs1 = (instr >> 15) & 0x1F
        rs2 = (instr >> 20) & 0x1F
        f7 = (instr >> 25) & 0x7F
        next_pc = self.pc + 4

        if self.trace:
            print(f"  pc={self.pc:08x}  instr={instr:08x}")

        if op == 0x33:      # R-type
            av, bv = to_s32(self.x[rs1]), to_s32(self.x[rs2])
            if f3 == 0x0:
                r = av - bv if f7 == 0x20 else av + bv
            elif f3 == 0x1:
                r = self.x[rs1] << (self.x[rs2] & 0x1F)
            elif f3 == 0x2:
                r = 1 if av < bv else 0
            elif f3 == 0x3:
                r = 1 if to_u32(self.x[rs1]) < to_u32(self.x[rs2]) else 0
            elif f3 == 0x4:
                r = self.x[rs1] ^ self.x[rs2]
            elif f3 == 0x5:
                sh = self.x[rs2] & 0x1F
                r = (av >> sh) if f7 == 0x20 else (to_u32(self.x[rs1]) >> sh)
            elif f3 == 0x6:
                r = self.x[rs1] | self.x[rs2]
            elif f3 == 0x7:
                r = self.x[rs1] & self.x[rs2]
            self.x[rd] = to_u32(r)

        elif op == 0x13:    # I-type arithmetic / shifts
            av = to_s32(self.x[rs1])
            imm = sign_extend(instr >> 20, 12)
            if f3 == 0x0:
                r = av + imm
            elif f3 == 0x2:
                r = 1 if av < imm else 0
            elif f3 == 0x3:
                r = 1 if to_u32(self.x[rs1]) < to_u32(imm) else 0
            elif f3 == 0x4:
                r = self.x[rs1] ^ (imm & 0xFFFFFFFF)
            elif f3 == 0x6:
                r = self.x[rs1] | (imm & 0xFFFFFFFF)
            elif f3 == 0x7:
                r = self.x[rs1] & (imm & 0xFFFFFFFF)
            elif f3 == 0x1:
                r = self.x[rs1] << (rs2 & 0x1F)
            elif f3 == 0x5:
                sh = rs2 & 0x1F
                r = (av >> sh) if f7 == 0x20 else (to_u32(self.x[rs1]) >> sh)
            self.x[rd] = to_u32(r)

        elif op == 0x03:    # loads
            imm = sign_extend(instr >> 20, 12)
            addr = to_u32(self.x[rs1] + imm)
            if f3 == 0x0:
                self.x[rd] = to_u32(self.load(addr, 1, True))
            elif f3 == 0x1:
                self.x[rd] = to_u32(self.load(addr, 2, True))
            elif f3 == 0x2:
                self.x[rd] = to_u32(self.load(addr, 4, True))
            elif f3 == 0x4:
                self.x[rd] = self.load(addr, 1, False)
            elif f3 == 0x5:
                self.x[rd] = self.load(addr, 2, False)

        elif op == 0x23:    # stores
            imm = sign_extend(((instr >> 25) << 5) | ((instr >> 7) & 0x1F), 12)
            addr = to_u32(self.x[rs1] + imm)
            size = {0: 1, 1: 2, 2: 4}[f3]
            self.store(addr, size, self.x[rs2])

        elif op == 0x63:    # branches
            imm = ((instr >> 31) & 1) << 12
            imm |= ((instr >> 7) & 1) << 11
            imm |= ((instr >> 25) & 0x3F) << 5
            imm |= ((instr >> 8) & 0xF) << 1
            imm = sign_extend(imm, 13)
            av, bv = to_s32(self.x[rs1]), to_s32(self.x[rs2])
            ua, ub = to_u32(self.x[rs1]), to_u32(self.x[rs2])
            take = {0x0: av == bv, 0x1: av != bv, 0x4: av < bv,
                    0x5: av >= bv, 0x6: ua < ub, 0x7: ua >= ub}[f3]
            if take:
                next_pc = to_u32(self.pc + imm)

        elif op == 0x37:    # lui
            self.x[rd] = to_u32((instr >> 12) << 12)

        elif op == 0x17:    # auipc
            self.x[rd] = to_u32(self.pc + ((instr >> 12) << 12))

        elif op == 0x6F:    # jal
            imm = ((instr >> 31) & 1) << 20
            imm |= ((instr >> 12) & 0xFF) << 12
            imm |= ((instr >> 20) & 1) << 11
            imm |= ((instr >> 21) & 0x3FF) << 1
            imm = sign_extend(imm, 21)
            self.x[rd] = to_u32(self.pc + 4)
            next_pc = to_u32(self.pc + imm)

        elif op == 0x67:    # jalr
            imm = sign_extend(instr >> 20, 12)
            target = to_u32((self.x[rs1] + imm) & ~1)
            self.x[rd] = to_u32(self.pc + 4)
            next_pc = target

        elif op == 0x73:    # ecall -> halt
            self.halted = True

        else:
            raise RuntimeError(f"illegal instruction {instr:08x} at pc={self.pc:08x}")

        self.x[0] = 0       # x0 hardwired to zero
        self.pc = next_pc

    def dump(self):
        names = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
                 "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                 "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
                 "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
        print("Register file:")
        for i in range(0, 32, 4):
            row = "  ".join(f"{names[i+j]:>4}={to_s32(self.x[i+j]):>11}"
                            for j in range(4))
            print("  " + row)


def run_file(path, trace=False):
    with open(path) as f:
        src = f.read()
    asm = Assembler()
    words = asm.assemble(src)
    cpu = CPU(trace=trace)
    cpu.load_program(words, asm.data)
    steps = cpu.run()
    print(f"Executed {steps} instructions.\n")
    cpu.dump()
    return cpu


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_file(sys.argv[1], trace="--trace" in sys.argv)
