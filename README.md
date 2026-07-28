# RISC-V Processor and Memory System (Simulator)

A software model of a RISC-V processor and its memory system, built in Python one
architectural layer at a time: instruction set and assembler, a gate-level ALU, a
single-cycle datapath, a five-stage pipeline with hazard handling, and a
configurable cache. Every layer has an automated test suite.

Built as an independent project for a graduate computer architecture course
(CS 5375, Computer Systems Organization and Architecture).

## What it does

| Layer | Module | Tests |
|-------|--------|-------|
| ISA, assembler, functional CPU | `riscv_sim.py` | `test_a3.py` (16/16) |
| Gate-level ALU | `alu.py` | `alu_test.py` (57,792/57,792) |
| Single-cycle datapath | `datapath.py` | `datapath_test.py` (7/7) |
| 5-stage pipeline (forwarding, stalls, flushes) | `pipeline.py` | `pipeline_test.py` (6/6) |
| Configurable cache (direct / set-assoc / fully, LRU) | `cache_sim.py` | `cache_demo.py` |
| Performance analysis (CPI, AMAT, speedup) | `perf.py` | benchmark suite |

## Highlights

- Assembler encodes RV32I to real 32-bit machine code across all six formats.
- ALU addition is built from full adders in a ripple-carry chain; subtraction
  reuses it via two's complement.
- The pipeline resolves data hazards by forwarding, load-use hazards by stalling,
  and control hazards by flushing, and it produces register-for-register identical
  results to the single-cycle datapath.
- The cache reproduces a standard associativity experiment and, when attached
  behind the pipeline, measures the real hit/miss behavior of running programs.

## Running it

Requires Python 3. From the project directory:

```
python test_a3.py                         # functional CPU tests
python alu_test.py                        # ALU tests
python datapath.py --table                # control-signal table
python datapath_test.py                   # single-cycle datapath tests
python pipeline.py a3_problem1.s --trace   # pipeline run with stall/flush trace
python pipeline_test.py                   # pipeline tests + CPI
python cache_demo.py                      # cache demonstration
python perf.py                            # performance analysis
```

See `PROJECT_REPORT.md` for the full technical write-up.
