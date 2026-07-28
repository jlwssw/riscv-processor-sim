# RISC-V Processor and Memory System: Technical Report

CS 5375, Computer Systems Organization and Architecture

## 1. System overview

A modular software model of a RISC-V processor and memory system. Each phase is
an independent module with its own automated test file, and each builds on the
previous one. This report documents the design and the verified results; the
accompanying reflection covers the reasoning and trade-offs.

| Phase | Module(s) | Function | Verification |
|-------|-----------|----------|--------------|
| 0 | `riscv_sim.py` | assembler + functional CPU | `test_a3.py`, 16/16 |
| 1 | `alu.py` | gate-level ALU | `alu_test.py`, 57,792/57,792 |
| 2 | `datapath.py` | single-cycle datapath | `datapath_test.py`, 7/7 |
| 3 | `pipeline.py` | 5-stage pipeline | `pipeline_test.py`, 6/6 |
| 4 | `cache_sim.py` | configurable cache | `cache_demo.py`, matches A7 |
| 5 | `perf.py` | performance analysis | benchmark suite |

## 2. Instruction set and assembler (Phase 0)

Target ISA is the RV32I base integer subset. The assembler is two-pass: pass one
records label addresses, lays out the `.data` segment, and expands pseudo-
instructions (`li`, `mv`, `j`, `call`, `ret`, `nop`, `la`, `beqz`, `bnez`); pass
two encodes each instruction into a 32-bit word.

Supported instructions:

- R-type: `add sub sll slt sltu xor srl sra or and`
- I-type: `addi slti sltiu xori ori andi slli srli srai`, loads `lb lh lw lbu lhu`, `jalr`
- S-type: `sb sh sw`
- B-type: `beq bne blt bge bltu bgeu`
- U-type: `lui auipc`
- J-type: `jal`
- System: `ecall` (halt)

Encoding covers all six instruction formats, including the split immediate fields
of the B and J formats and the `%hi`/`%lo` sign-correction used to materialize
addresses. Memory is byte-addressable and little-endian.

## 3. Arithmetic logic unit (Phase 1)

Addition is implemented at gate level: a full adder computes one bit
(`sum = a XOR b XOR cin`, `carry = majority(a, b, cin)`), and a 32-bit
ripple-carry chain threads the carry from bit 0 to bit 31. Subtraction reuses the
adder via two's complement (`a - b = a + ~b + 1`).

Operations: `ADD SUB AND OR XOR NOR SLL SRL SRA SLT SLTU`, selected by a 4-bit ALU
control code. Every operation returns four flags: zero, carry-out, overflow
(`carry_in_MSB XOR carry_out_MSB`), and negative. Signed set-less-than is derived
as `negative XOR overflow` so it remains correct when the subtraction overflows.

Verification: each operation is checked against independent reference arithmetic
over all edge cases (0, 1, -1, INT_MIN, INT_MAX) and 2,000 random operand pairs,
including flag values, for a total of 57,792 passing checks.

## 4. Single-cycle datapath (Phase 2)

Execution is expressed as discrete hardware blocks (program counter, instruction
memory, register file, immediate generator, ALU, data memory) plus the
multiplexers between them. A control unit decodes the opcode into control signals;
an ALU-control block maps `ALUOp` plus the funct fields to an ALU control code.

Control-signal table (`python datapath.py --table`):

```
instr   RegWr  ALUSrc  MemRd  MemWr  Result  Branch  Jump  ALUOp
R-type  1      0       0      0      ALU     0       0     RTYPE
I-type  1      1       0      0      ALU     0       0     ITYPE
load    1      1       1      0      MEM     0       0     ADD
store   0      1       0      1      ALU     0       0     ADD
branch  0      0       0      0      ALU     1       0     SUB
lui     1      1       0      0      IMM     0       0     ADD
auipc   1      1       0      0      ALU     0       0     ADD
jal     1      0       0      0      PC+4    0       1     ADD
jalr    1      1       0      0      PC+4    0       1     ADD
```

Verification: for every test program the datapath produces register-for-register
identical state to the Phase-0 functional simulator (7/7).

## 5. Five-stage pipeline (Phase 3)

Stages IF, ID, EX, MEM, WB are separated by pipeline registers, with up to five
instructions in flight. Hazard handling:

- Data hazards: forwarding from the EX/MEM and MEM/WB latches into the EX inputs,
  with EX/MEM taking priority over MEM/WB.
- Load-use hazard: detected when a load in EX targets a register the instruction
  in ID reads; resolved with a one-cycle stall, after which the value forwards
  from MEM/WB.
- Control hazards: branches and jumps resolve in EX; on a taken transfer the two
  younger instructions are flushed and the PC is redirected.

Measured CPI (cycles / instructions):

| Program | CPI | Dominant overhead |
|---------|-----|-------------------|
| ALU dependent chain | 1.57 | pipeline fill (forwarding absorbed all data hazards) |
| load-then-use | 1.86 | load-use stalls |
| branch/jump heavy | 1.92 | control-hazard flushes |
| A3 Problem 1 | 1.58 | stalls and flushes in the loop |
| A3 Problem 2 | 1.67 | flushes from function calls |

Verification: the pipeline produces the same final registers as the single-cycle
datapath on every test program (6/6).

## 6. Cache (Phase 4)

`cache_sim.py` is one parameterized structure: associativity of 1 is direct-
mapped, associativity equal to the block count is fully associative, and values
between are N-way set-associative. Replacement is LRU. For each reference it
computes the block index and tag and records hit or miss, then reports hit rate,
miss rate, and AMAT (`hit_time + miss_rate * miss_penalty`).

Reproduction of Assignment 7, Problem 2(b) (16 blocks, one-word blocks, LRU, the
given 10-bit read sequence):

| Cache | Hits / 12 | Hit rate |
|-------|-----------|----------|
| Direct-mapped | 2 | 16.7% |
| 2-way | 5 | 41.7% |
| 4-way | 5 | 41.7% |
| Fully associative | 5 | 41.7% |

Assignment 7, Problem 1(b): CPI 1.2, 20% data-access instructions, 1% miss rate,
200-cycle penalty gives `1.2 + (1 + 0.20)(0.01)(200) = 3.6`, a 3.0x slowdown.

In Phase 4 the cache is attached behind the pipeline and observes the real
instruction-fetch and data-access streams of the programs it runs.

## 7. Performance analysis (Phase 5)

Model: pipeline stage = 1 time unit (pipeline clock period 1); single-cycle clock
period = 5 (all five stages per cycle, CPI 1); cache miss penalty = 20 cycles;
execution time = instructions x CPI x clock period. Effective CPI adds
`(misses x penalty) / instructions`.

Benchmark output (`python perf.py`):

```
PIPELINE BEHAVIOR
  benchmark              instr    CPI  stalls  flushes
  vector-scan (A3 P1)       57  1.544       9        9
  func-call (A3 P2)         18  1.667       0        4
  sum-loop                  64  1.656       0       19
  alu-chain                  7  1.571       0        0

CACHE BEHAVIOR (I$: 8x4w direct, D$: 8x4w 2-way, LRU)
  benchmark                I-hit  D-acc   D-hit
  vector-scan (A3 P1)     95.5%      9   66.7%
  func-call (A3 P2)       77.3%      4   75.0%
  sum-loop                97.6%      0    0.0%
  alu-chain               71.4%      0    0.0%

SPEEDUP vs single-cycle (single=5u, pipe=1u; miss penalty=20)
  benchmark                 CPI  CPI_eff  pipe x +cache x
  vector-scan (A3 P1)     1.544    3.649    3.24     1.37
  func-call (A3 P2)       1.667    8.333    3.00     0.60
  sum-loop                1.656    2.281    3.02     2.19
  alu-chain               1.571    7.286    3.18     0.69
```

Pipelining yields roughly a 3x speedup across the suite (`5 / CPI`). Effective CPI
is governed by cache behavior: loop-dominated code (`sum-loop`, 97.6% instruction
hit rate) retains most of the speedup, while short straight-line code is dominated
by compulsory misses and gains little from the cache.

## 8. Verification summary

| Suite | Scope | Result |
|-------|-------|--------|
| `test_a3.py` | assembler + functional CPU on A3 programs | 16/16 |
| `alu_test.py` | ALU vs reference arithmetic, incl. flags | 57,792/57,792 |
| `datapath_test.py` | single-cycle vs functional CPU | 7/7 |
| `pipeline_test.py` | pipeline vs single-cycle, plus CPI | 6/6 |
| `cache_demo.py` | cache vs Assignment 7 hit counts | matches |

## 9. How to run

```
python test_a3.py                          # Phase 0 tests
python alu_test.py                         # Phase 1 tests
python datapath.py --table                 # Phase 2 control-signal table
python datapath_test.py                    # Phase 2 tests
python pipeline.py a3_problem1.s --trace    # Phase 3 stall/flush trace
python pipeline_test.py                    # Phase 3 tests + CPI
python cache_demo.py                       # Phase 4 Assignment 7 reproduction
python perf.py                             # Phase 5 benchmark + speedup
```

## References

[1] D. A. Patterson and J. L. Hennessy, Computer Organization and Design: The
Hardware/Software Interface, RISC-V Edition, 2nd ed. Morgan Kaufmann, 2020.

[2] A. Waterman and K. Asanovic, Eds., The RISC-V Instruction Set Manual,
Volume I: Unprivileged ISA, RISC-V International, 2019.
