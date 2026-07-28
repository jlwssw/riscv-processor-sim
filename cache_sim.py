#!/usr/bin/env python3
"""
cache_sim.py -- A configurable cache-memory simulator.

CS 5375: Computer Systems Organization and Architecture
Extension of the RISC-V project into the memory-hierarchy portion of the
course (cf. Assignment 7: Memory Architecture).

Given a stream of memory addresses, this models a cache and reports, for
every access, the block Index, the Tag, and whether it was a Hit or Miss.
It then reports the overall hit rate, miss rate, and AMAT.  The same class
covers all three mappings from A7 by varying one parameter:

    associativity = 1              -> Direct Mapped
    associativity = num_blocks     -> Fully Associative
    associativity = N  (1<N<blocks)-> N-way Set Associative

Replacement policy is LRU.

Addressing convention
---------------------
A block holds `block_size_words` words (4 bytes each), so the low
log2(block_bytes) address bits are the block offset and are ignored when
selecting a line.  With the A7 default (block size = 1 word) that is a
2-bit byte offset.  If your course instead treats the addresses as *word*
addresses (no byte offset), pass block_size_words=1 with byte_addressed=False.
"""

import math


class Cache:
    def __init__(self, num_blocks, block_size_words=1, associativity=1,
                 addr_bits=10, byte_addressed=True, name=""):
        if num_blocks % associativity != 0:
            raise ValueError("num_blocks must be a multiple of associativity")
        self.name = name or f"{associativity}-way"
        self.num_blocks = num_blocks
        self.assoc = associativity
        self.addr_bits = addr_bits
        block_bytes = block_size_words * (4 if byte_addressed else 1)
        self.offset_bits = int(math.log2(block_bytes)) if block_bytes > 1 else 0
        self.num_sets = num_blocks // associativity
        self.index_bits = int(math.log2(self.num_sets)) if self.num_sets > 1 else 0
        self.tag_bits = addr_bits - self.index_bits - self.offset_bits
        # each set is a list of tags, ordered front = LRU ... back = MRU
        self.sets = [[] for _ in range(self.num_sets)]
        self.hits = 0
        self.misses = 0
        self.trace = []          # list of (addr, index, tag, hit)

    # ---- address breakdown ----
    def fields(self, addr):
        block = addr >> self.offset_bits
        index = block & (self.num_sets - 1) if self.num_sets > 1 else 0
        tag = block >> self.index_bits
        return index, tag

    # ---- one memory reference ----
    def access(self, addr):
        index, tag = self.fields(addr)
        line = self.sets[index]
        if tag in line:
            line.remove(tag)         # touch -> becomes most-recently-used
            line.append(tag)
            hit = True
            self.hits += 1
        else:
            if len(line) >= self.assoc:
                line.pop(0)          # evict least-recently-used
            line.append(tag)
            hit = False
            self.misses += 1
        self.trace.append((addr, index, tag, hit))
        return index, tag, hit

    # ---- statistics ----
    @property
    def accesses(self):
        return self.hits + self.misses

    @property
    def hit_rate(self):
        return self.hits / self.accesses if self.accesses else 0.0

    @property
    def miss_rate(self):
        return 1.0 - self.hit_rate if self.accesses else 0.0

    def amat(self, hit_time, miss_penalty):
        """Average Memory Access Time = hit_time + miss_rate * miss_penalty."""
        return hit_time + self.miss_rate * miss_penalty

    # ---- pretty printing ----
    def report(self):
        print(f"=== {self.name}  "
              f"(sets={self.num_sets}, ways={self.assoc}, "
              f"index_bits={self.index_bits}, tag_bits={self.tag_bits}, "
              f"offset_bits={self.offset_bits}) ===")
        print(f"  {'#':>2}  {'address':>8}  {'index':>5}  {'tag':>5}  result")
        for i, (addr, index, tag, hit) in enumerate(self.trace, 1):
            print(f"  {i:>2}  0x{addr:0{(self.addr_bits+3)//4}X}  "
                  f"{index:>5}  0x{tag:>3X}  {'HIT ' if hit else 'miss'}")
        print(f"  -> hits={self.hits}, misses={self.misses}, "
              f"hit rate={self.hit_rate:.1%}, miss rate={self.miss_rate:.1%}")
        print()


def cpi_with_cache(base_cpi=1.2, data_access_frac=0.20, miss_rate=0.01,
                   miss_penalty=200, count_instruction_fetch=True):
    """
    A7 Problem 1(b): how much slower is the processor because of cache misses?

    Memory accesses per instruction = (1 instruction fetch, if counted)
                                       + data_access_frac (loads/stores).
    Stall cycles/instr = accesses/instr * miss_rate * miss_penalty.
    """
    accesses_per_instr = (1.0 if count_instruction_fetch else 0.0) + data_access_frac
    stall = accesses_per_instr * miss_rate * miss_penalty
    new_cpi = base_cpi + stall
    slowdown = new_cpi / base_cpi
    return {
        "accesses_per_instr": accesses_per_instr,
        "stall_cycles_per_instr": stall,
        "cpi_with_misses": new_cpi,
        "slowdown_x": slowdown,
    }
