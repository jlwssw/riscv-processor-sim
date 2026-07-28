# =====================================================================
# CS 5375 - Assignment 3, Problem 1
# Convert to RISC-V:   while (buff[i] == m) i++;
# Mapping (per the assignment):  i -> s2,  m -> a0,  base of buff -> s0
# =====================================================================

.data
# Test data: first four elements equal m (=7), buff[4] differs -> loop
# should run exactly four times and leave i = 4.
buff:   .word 7, 7, 7, 7, 3, 9

.text
        la   s0, buff        # s0 = base address of buff
        li   a0, 7           # m = 7
        li   s2, 0           # i = 0

# ---- direct translation of the while loop ----
loop:
        slli t0, s2, 2       # t0 = i * 4        (each int is 4 bytes)
        add  t0, s0, t0      # t0 = &buff[i]     (base + i*4)
        lw   t1, 0(t0)       # t1 = buff[i]
        bne  t1, a0, done    # while-condition: exit when buff[i] != m
        addi s2, s2, 1       # i++
        j    loop            # re-test the condition
done:
        ecall                # halt -- final value of i is in s2
