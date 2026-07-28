# =====================================================================
# CS 5375 - Assignment 3, Problem 2
# int test(int x, int y) { g = x + y - abs(x, y); return g; }
# int abs (int x, int y) { if (x >= y) return x - y; else return y - x; }
# Mapping:  x -> a0,  y -> a1,  g -> s0,  return value -> a0
#
# Demonstrates the calling convention: a nested call (test -> abs),
# saving/restoring ra and a callee-saved register (s0) on the stack.
# =====================================================================

.text
        li   a0, 6           # x = 6
        li   a1, 10          # y = 10
        call test            # g = test(6, 10)  -> expect 12  (= 2*min)
        ecall                # halt -- result g is in a0

# ---------------------------------------------------------------------
test:
        addi sp, sp, -8      # allocate an 8-byte stack frame
        sw   ra, 4(sp)       # save return address (abs call will clobber ra)
        sw   s0, 0(sp)       # save caller's s0 (s0 is callee-saved)
        add  s0, a0, a1      # s0 = x + y     (a0,a1 still hold x,y for abs)
        call abs             # a0 = abs(x, y)
        sub  s0, s0, a0      # g = (x + y) - abs(x, y)
        mv   a0, s0          # place return value g in a0
        lw   s0, 0(sp)       # restore s0
        lw   ra, 4(sp)       # restore ra
        addi sp, sp, 8       # free the stack frame
        ret                  # return to caller

# ---------------------------------------------------------------------
abs:
        bge  a0, a1, abs_xy  # if (x >= y) goto abs_xy
        sub  a0, a1, a0      #   else return y - x
        ret
abs_xy:
        sub  a0, a0, a1      #   return x - y
        ret
