
from sys import argv
from pysmt.shortcuts import *
from pympwmi.mpwmi import MPWMI
from pympwmi.mp2wmi import MP2WMI
from time import time


K = 5

NVARS = int(argv[1])
WMODE = int(argv[2])
CACHE = bool(int(argv[3]))
NPROC = int(argv[4])

VARS = [Symbol(f'x{i}', REAL) for i in range(NVARS)]

CLAUSES = [LE(Real(0), VARS[i]) for i in range(NVARS)]
CLAUSES.extend([LE(VARS[i], Real(1)) for i in range(NVARS)])
CLAUSES.extend([LE(Plus(VARS[i], VARS[i+1]), Real(1))
                for i in range(NVARS-1) if i % K != 0])


f = And(CLAUSES)
if WMODE == 0:
    w = Real(1)
elif WMODE == 1:
    w = Times([Ite(LE(VARS[i], Real(1/2)), Real(i+1), Real(1))
               for i in range(NVARS)])
elif WMODE == 2:
    w = Times([Ite(LE(VARS[i], VARS[i+1]), Real(i+1), Real(1))
               for i in range(NVARS-1) if i % 5 != 0])
elif WMODE == 3:
    w = Times([Ite(LE(VARS[i], VARS[i+1]), Plus(VARS[i], VARS[i+1]), Real(1))
               for i in range(NVARS-1) if i % 5 != 0])
    

queries = [LE(VARS[i], VARS[i+1]) for i in range(NVARS-1)
           if i % 5 != 0] #for j in range(i+1, NVARS)]

t0 = time()
mpwmi = MPWMI(f, w)
Z_mp, pq_mp = mpwmi.compute_volumes(queries=queries, cache=CACHE)
print("MPWMI")
print("time:", time()-t0, "\n")
print("Z:", Z_mp)
print("volumes:", pq_mp)

print("--------------------------------------------------")
t0 = time()
mpwmi = MP2WMI(f, w, n_processes=NPROC)
Z_mp, pq_mp = mpwmi.compute_volumes(queries=queries, cache=CACHE)
print(f"MPMPWMI-{NPROC}")
print("time:", time()-t0, "\n")
print("Z:", Z_mp)
print("volumes:", pq_mp)
