
from sys import argv
from pysmt.shortcuts import *
from pympwmi.mpwmi import MPWMI
from time import time

NVARS = int(argv[1])
WMODE = int(argv[2])
CACHE = bool(int(argv[3]))

VARS = [Symbol(f'x{i}', REAL) for i in range(NVARS)]

CLAUSES = [LE(Real(0), VARS[i]) for i in range(NVARS)]
CLAUSES.extend([LE(VARS[i], Real(1)) for i in range(NVARS)])
CLAUSES.extend([LE(Plus(VARS[i], VARS[i+1]), Real(1))
                for i in range(NVARS-1)])# for j in range(i+1, NVARS)])


f = And(CLAUSES)
if WMODE == 0:
    w = Real(1)
elif WMODE == 1:
    w = Times([Ite(LE(VARS[i], Real(1/2)), Real(i+1), Real(1)) for i in range(NVARS)])
elif WMODE == 2:
    w = Times([Ite(LE(VARS[i], VARS[i+1]), Real(i+1), Real(1)) for i in range(NVARS-1)])
elif WMODE == 3:
    w = Times([Ite(LE(VARS[i], VARS[i+1]), Plus(VARS[i], VARS[i+1]), Real(1)) for i in range(NVARS-1)])
    

queries = [LE(VARS[i], VARS[i+1]) for i in range(NVARS-1)] #for j in range(i+1, NVARS)]

t0 = time()
mpwmi = MPWMI(f, w)
Z_mp, pq_mp = mpwmi.compute_volumes(queries=queries, cache=CACHE)
print(time()-t0)


