#!/usr/bin/env python3
"""
pipeline.py -- Phase 3 of the project: a 5-stage pipelined RISC-V CPU.

CS 5375: Computer Systems Organization and Architecture

The single-cycle datapath (Phase 2) does one instruction per clock. Here the
datapath is sliced into five stages separated by pipeline registers, so up to
five instructions are in flight at once -- an assembly line:

    IF  -> ID  -> EX  -> MEM -> WB
       IF/ID  ID/EX  EX/MEM  MEM/WB     (the latches between stages)

Overlap creates HAZARDS, which this model handles the textbook way:

  * Data hazards      -> FORWARDING (bypass a result from EX/MEM or MEM/WB
                         straight back into the EX stage instead of waiting
                         for it to reach the register file).
  * Load-use hazard   -> a load's value isn't ready in time to forward, so we
                         STALL one cycle (insert a bubble) then forward.
  * Control hazards   -> branches/jumps resolve in EX; when taken we FLUSH the
                         two younger instructions and redirect the PC.

At the end it reports CPI (cycles per instruction) so you can compare against
the ideal of 1.0 and see the cost of stalls and flushes.

Correctness is checked in pipeline_test.py by requiring the final register
state to match the (already verified) single-cycle datapath exactly.
"""

import sys

from riscv_sim import Assembler, to_u32, to_s32, reg, TEXT_BASE, DATA_BASE
from alu import ALU, ALUControl
from datapath import (
    control_unit, immediate_generator, alu_control, RegisterFile, Memory,
    OP_RTYPE, OP_ITYPE, OP_LOAD, OP_STORE, OP_BRANCH, OP_JALR, OP_ECALL,
    RES_ALU, RES_MEM, RES_PC4, RES_IMM,
)


def _uses_rs1(op):
    return op in (OP_RTYPE, OP_ITYPE, OP_LOAD, OP_STORE, OP_BRANCH, OP_JALR)


def _uses_rs2(op):
    return op in (OP_RTYPE, OP_STORE, OP_BRANCH)


class PipelinedCPU:
    def __init__(self, trace=False, dcache=None, icache=None):
        self.rf = RegisterFile()
        self.mem = Memory()
        self.alu = ALU()
        self.pc = TEXT_BASE
        # pipeline latches (None = bubble/empty)
        self.IFID = self.IDEX = self.EXMEM = self.MEMWB = None
        self.halted = False
        self.stop_fetch = False
        self.trace = trace
        # Phase 4: optional caches observing the real access streams
        self.dcache = dcache      # observes data loads/stores (MEM stage)
        self.icache = icache      # observes instruction fetches (IF stage)
        # statistics
        self.cycles = 0
        self.retired = 0
        self.stalls = 0
        self.flushes = 0

    def load_program(self, words, data=b"", data_base=DATA_BASE):
        for addr, w in words:
            self.mem.store(addr, 4, w)
        for i, b in enumerate(data):
            self.mem.m[data_base + i] = b

    # ---- decode one fetched word into a flat record ----
    def decode(self, instr, pc):
        op = instr & 0x7F
        c = control_unit(op)
        rec = {
            "pc": pc, "instr": instr, "opcode": op,
            "rd": (instr >> 7) & 0x1F, "funct3": (instr >> 12) & 0x7,
            "rs1": (instr >> 15) & 0x1F, "rs2": (instr >> 20) & 0x1F,
            "funct7": (instr >> 25) & 0x7F, "shamt": (instr >> 20) & 0x1F,
            "imm": immediate_generator(instr, op) if op != OP_ECALL else 0,
            "is_ecall": op == OP_ECALL,
            "uses_rs1": _uses_rs1(op), "uses_rs2": _uses_rs2(op),
            # flattened control signals
            "reg_write": c.reg_write, "alu_src": c.alu_src,
            "mem_read": c.mem_read, "mem_write": c.mem_write,
            "result_src": c.result_src, "branch": c.branch,
            "jump": c.jump, "jump_reg": c.jump_reg,
            "a_src": c.a_src, "alu_op": c.alu_op,
        }
        return rec

    def branch_taken(self, funct3, a, b):
        _, fl = self.alu.compute(ALUControl.SUB, a, b)
        lt_s = fl.negative ^ fl.overflow
        lt_u = 0 if fl.carry_out else 1
        return {0x0: fl.zero == 1, 0x1: fl.zero == 0,
                0x4: lt_s == 1, 0x5: lt_s == 0,
                0x6: lt_u == 1, 0x7: lt_u == 0}[funct3]

    def _forward(self, src, uses, default, exmem, memwb):
        """Bypass network: prefer the most recent producer (EX/MEM), else MEM/WB."""
        # Instead of waiting for a result to reach the register file, grab it
        # straight from a later pipeline stage. EX/MEM is the more recent producer,
        # so it takes priority over MEM/WB when both hold the needed register.
        if not uses:
            return default
        if exmem and exmem["reg_write"] and exmem["rd"] != 0 \
                and exmem["rd"] == src and exmem["wb_ready"]:
            return exmem["wb_value"]
        if memwb and memwb["reg_write"] and memwb["rd"] != 0 and memwb["rd"] == src:
            return memwb["wb_value"]
        return default

    def cycle(self):
        oldIFID, oldIDEX = self.IFID, self.IDEX
        oldEXMEM, oldMEMWB = self.EXMEM, self.MEMWB

        # ---------- WB : write register file, retire instruction ----------
        if oldMEMWB:
            if oldMEMWB["reg_write"]:
                self.rf.write(oldMEMWB["rd"], oldMEMWB["wb_value"])
            self.retired += 1
            if oldMEMWB["is_ecall"]:
                self.halted = True

        # ---------- MEM : data memory, build MEM/WB ----------
        newMEMWB = None
        if oldEXMEM:
            r = oldEXMEM
            mem_data = 0
            if r["mem_read"]:
                size = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}[r["funct3"]]
                signed = r["funct3"] in (0, 1, 2)
                mem_data = to_u32(self.mem.load(r["alu_result"], size, signed))
            if r["mem_write"]:
                size = {0: 1, 1: 2, 2: 4}[r["funct3"]]
                self.mem.store(r["alu_result"], size, r["store_data"])
            if self.dcache and (r["mem_read"] or r["mem_write"]):
                self.dcache.access(r["alu_result"])     # Phase 4: observe data access
            wb = mem_data if r["result_src"] == RES_MEM else r["wb_value"]
            newMEMWB = {"reg_write": r["reg_write"], "rd": r["rd"],
                        "wb_value": wb, "is_ecall": r["is_ecall"]}

        # ---------- EX : ALU, forwarding, branch resolution ----------
        newEXMEM = None
        redirect = False
        target = None
        if oldIDEX:
            r = oldIDEX
            fa = self._forward(r["rs1"], r["uses_rs1"], r["rv1"], oldEXMEM, oldMEMWB)
            fb = self._forward(r["rs2"], r["uses_rs2"], r["rv2"], oldEXMEM, oldMEMWB)
            a = {"rs1": fa, "pc": r["pc"], "zero": 0}[r["a_src"]]
            b = r["imm"] if r["alu_src"] else fb
            actrl = alu_control(r["alu_op"], r["funct3"], r["funct7"])
            alu_result, _ = self.alu.compute(actrl, a, b, r["shamt"])

            if r["branch"] and self.branch_taken(r["funct3"], fa, fb):
                redirect, target = True, to_u32(r["pc"] + r["imm"])
            elif r["jump"] and r["jump_reg"]:
                redirect, target = True, to_u32((fa + r["imm"]) & ~1)
            elif r["jump"]:
                redirect, target = True, to_u32(r["pc"] + r["imm"])

            if r["result_src"] == RES_MEM:
                wb_ready, wb_value = False, None
            elif r["result_src"] == RES_PC4:
                wb_ready, wb_value = True, to_u32(r["pc"] + 4)
            elif r["result_src"] == RES_IMM:
                wb_ready, wb_value = True, r["imm"]
            else:
                wb_ready, wb_value = True, alu_result
            newEXMEM = {
                "reg_write": r["reg_write"], "rd": r["rd"],
                "alu_result": alu_result, "store_data": fb,
                "mem_read": r["mem_read"], "mem_write": r["mem_write"],
                "funct3": r["funct3"], "result_src": r["result_src"],
                "pc": r["pc"], "imm": r["imm"], "is_ecall": r["is_ecall"],
                "wb_ready": wb_ready, "wb_value": wb_value,
            }

        # ---------- load-use hazard detection (stall?) ----------
        # A load's value is not ready until the end of MEM, so it cannot forward to
        # the very next instruction. If the instruction in ID reads a register a
        # load in EX is about to write, stall one cycle; the value forwards from
        # MEM/WB on the following cycle.
        stall = False
        if oldIDEX and oldIDEX["mem_read"] and oldIDEX["rd"] != 0 and oldIFID:
            need = ((oldIFID["uses_rs1"] and oldIFID["rs1"] == oldIDEX["rd"]) or
                    (oldIFID["uses_rs2"] and oldIFID["rs2"] == oldIDEX["rd"]))
            stall = need and not redirect

        # ---------- ID and IF : advance, stall, or flush ----------
        if stall:
            self.stalls += 1
            newIDEX = None            # bubble into EX
            newIFID = oldIFID         # hold the instruction in ID
            # PC frozen (do not fetch)
        elif redirect:
            # A branch or jump was taken, so the two instructions already fetched
            # behind it are wrong. Squash them (turn them into bubbles) and send the
            # PC to the correct target.
            self.flushes += 1
            newIDEX = None            # squash instruction that was in ID
            newIFID = None            # squash instruction that was in IF
            self.pc = target
            self.stop_fetch = False   # resume fetching at the target
        else:
            newIDEX = self._id_stage(oldIFID)
            newIFID = self._if_stage()

        if self.trace:
            self._trace_line(oldIDEX, redirect, stall)

        self.IFID, self.IDEX = newIFID, newIDEX
        self.EXMEM, self.MEMWB = newEXMEM, newMEMWB

    def _id_stage(self, oldIFID):
        if oldIFID is None:
            return None
        r = dict(oldIFID)
        r["rv1"] = self.rf.read(r["rs1"])
        r["rv2"] = self.rf.read(r["rs2"])
        return r

    def _if_stage(self):
        if self.stop_fetch or self.halted:
            return None
        instr = self.mem.load(self.pc, 4, False)
        if self.icache:
            self.icache.access(self.pc)                 # Phase 4: observe instruction fetch
        rec = self.decode(instr, self.pc)
        if rec["opcode"] == OP_ECALL:
            self.stop_fetch = True
        self.pc = to_u32(self.pc + 4)
        return rec

    def _trace_line(self, ex, redirect, stall):
        tag = "STALL" if stall else ("FLUSH" if redirect else "     ")
        exi = f"{ex['instr']:08x}@{ex['pc']:04x}" if ex else "  bubble  "
        print(f"  cyc {self.cycles:>3}  {tag}  EX={exi}")

    def run(self, max_cycles=1_000_000):
        while not self.halted and self.cycles < max_cycles:
            self.cycle()
            self.cycles += 1
        if self.cycles >= max_cycles:
            raise RuntimeError("cycle limit exceeded")
        return self.cycles

    def get_reg(self, name):
        return self.rf.read(reg(name))

    def cpi(self):
        return self.cycles / self.retired if self.retired else 0.0

    def dump(self):
        names = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
                 "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                 "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
                 "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
        print("Register file:")
        for i in range(0, 32, 4):
            print("  " + "  ".join(f"{names[i+j]:>4}={to_s32(self.rf.read(i+j)):>11}"
                                   for j in range(4)))


def run_file(path, trace=False):
    with open(path) as f:
        src = f.read()
    asm = Assembler()
    words = asm.assemble(src)
    cpu = PipelinedCPU(trace=trace)
    cpu.load_program(words, asm.data)
    cpu.run()
    print(f"\nCycles={cpu.cycles}  Instructions retired={cpu.retired}  "
          f"CPI={cpu.cpi():.3f}  (stalls={cpu.stalls}, flushes={cpu.flushes})\n")
    cpu.dump()
    return cpu


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run_file(sys.argv[1], trace="--trace" in sys.argv)
    else:
        print(__doc__)
