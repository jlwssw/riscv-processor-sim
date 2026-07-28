# Cache Integration and Effective CPI

This note explains how the cache is connected to the pipeline so that it measures
the real memory behavior of a running program, and how those measurements are
turned into an effective cycles-per-instruction (CPI) figure. It covers the parts
of `pipeline.py`, `cache_sim.py`, and `perf.py` that work together for this.

## The idea in one picture

Think of the cache as a meter clipped onto the processor's memory port. Every time
the running program reaches for memory, the CPU also hands that address to the
cache, and the cache records a hit or a miss. Nothing here reads files or runs a
second program; it is all one simulation in which one software component (the
cache) observes another (the CPU) as it runs.

```
   CPU (pipeline)                         cache (observer)
   fetch instruction  --- address --->    record hit / miss
   load / store data  --- address --->    record hit / miss
```

## How the cache is wired in (`pipeline.py`)

The pipeline can be created with an optional instruction cache and data cache.
When they are present, the CPU notifies them on each memory access.

The caches are attached when the CPU is constructed:

```python
def __init__(self, trace=False, dcache=None, icache=None):
    ...
    self.dcache = dcache      # observes data loads/stores (MEM stage)
    self.icache = icache      # observes instruction fetches (IF stage)
```

Every instruction fetch is reported to the instruction cache, in the fetch step:

```python
instr = self.mem.load(self.pc, 4, False)
if self.icache:
    self.icache.access(self.pc)      # Phase 4: observe instruction fetch
```

Every load or store is reported to the data cache, in the memory stage:

```python
if self.dcache and (r["mem_read"] or r["mem_write"]):
    self.dcache.access(r["alu_result"])   # Phase 4: observe data access
```

Because these calls fire as the program executes, the caches see the program's
actual address stream, not a synthetic list of addresses. The lines are marked
with `# Phase 4:` in the source so they are easy to find.

## What the cache records (`cache_sim.py`)

Each call to `access(address)` computes the block index and tag for that address,
checks whether the block is present, updates the least-recently-used order, and
increments either the hit or the miss counter. The cache also exposes `hit_rate`,
`miss_rate`, and `amat` (average memory access time). See `cache_sim.py` for the
`Cache` class.

## An observer, not a stall (an honest note)

In this model the cache counts misses but does not pause the pipeline in real time
on each miss. The pipeline runs as if memory were instant, and the cost of the
misses is added afterward with arithmetic. This keeps the simulator simple while
still producing a realistic performance estimate. The step that adds that cost is
the effective-CPI calculation below.

## Turning misses into effective CPI (`perf.py`)

The plain pipeline CPI assumes memory is free. Effective CPI folds the miss cost
back in:

```
effective CPI = base CPI + (total misses * miss penalty) / instructions
```

In `run_benchmark`:

```python
ic, dc = make_caches()
pc = PipelinedCPU(icache=ic, dcache=dc)
pc.load_program(words, asm.data); pc.run()
total_misses = ic.misses + dc.misses
mem_stall_per_instr = total_misses * MISS_PENALTY / n_instr
cpi_eff = cpi + mem_stall_per_instr
```

## The timing model and speedup

To compare the pipeline against the single-cycle design fairly, the analysis uses
a simple, stated-up-front timing model:

- The single-cycle machine performs all five stages in one clock, so its clock
  period is 5 units.
- The pipeline performs one stage per clock, so its clock period is 1 unit.
- A cache miss costs 20 extra cycles.

Execution time is `instructions * CPI * clock period`, so the pipeline's speedup
over single-cycle is `5 / CPI`. With a pipeline CPI near 1.6 that is roughly a 3x
speedup from pipelining alone. Once realistic memory behavior is folded in through
effective CPI, the speedup varies with the program's locality: code that reuses
data and instructions keeps most of the gain, while code with poor locality loses
much of it to misses.

## Where to look

| File | What to read |
|------|--------------|
| `pipeline.py` | the `# Phase 4:` lines (cache attachment and the two `access` calls) |
| `cache_sim.py` | the `Cache` class: `access`, `hit_rate`, `miss_rate`, `amat` |
| `perf.py` | `run_benchmark`: effective CPI, speedup, and the benchmark table |
