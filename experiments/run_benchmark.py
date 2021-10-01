import sys
import os
import argparse
import time
from itertools import product

from exp_defaults import *
from multiprocessing import Process, Queue
from pympwmi import MPWMI
from oldpympwmi import oldMP2WMI, oldMPWMI
import psutil
from pysmt.shortcuts import Bool
from pywmi import Density, PyXaddAlgebra, XsddEngine
from wmipa import WMI

def kill_recursive(pid):
    proc = psutil.Process(pid)
    for subproc in proc.children(recursive=True):
        try:
            subproc.kill()
        except psutil.NoSuchProcess:
            continue
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass


def wrapped_solver(solver, density, queue):
    if solver.startswith("mpwmi-"): # should be "mpwmi-[numeric/symbolic]-NPROC"
        solverargs = solver.split("-")[1:]
        cache = ('cache' in solverargs)
        msgtype = solverargs[0]
        nproc = int(solverargs[1])
        #print(f"using {msgtype} mpwmi with {nproc} processes")
        t1 = time.perf_counter()
        #print("About to start mpwmi solver")
        mpmi = MPWMI(density.support, density.weight, smt_solver='msat',
                     n_processes=nproc, msgtype=msgtype)
        #print("Solver inited")
        Z, _ = mpmi.compute_volumes(cache=cache)
        #print("Volume computed")
        t2 = time.perf_counter()

    elif solver.startswith("oldmp2wmi-"): # should be "mpwmi-[numeric/symbolic]-NPROC"
        solverargs = solver.split("-")[1:]
        cache = ('cache' in solverargs)
        nproc = int(solverargs[0])
        #print(f"using (old) mp2wmi with {nproc} processes")
        t1 = time.perf_counter()
        #print("About to start mpwmi solver")
        mpmi = oldMP2WMI(density.support, density.weight, smt_solver='msat',
                         n_processes=nproc)
        #print("Solver inited")
        Z, _ = mpmi.compute_volumes(cache=cache)
        #print("Volume computed")
        t2 = time.perf_counter()

    elif solver.startswith("oldmpwmi"): # should be "mpwmi-[numeric/symbolic]-NPROC"
        #print(f"using oldmpwmi")
        # mpwmi
        t1 = time.perf_counter()
        #print("About to start mpwmi solver")
        mpmi = oldMPWMI(density.support, density.weight, smt_solver='msat')
        #print("Solver inited")
        Z, _ = mpmi.compute_volumes(cache=False)
        #print("Volume computed")
        t2 = time.perf_counter()

    elif solver == "pa":
        #print("using pa")

        # pa
        t1 = time.perf_counter()
        wmipa = WMI(density.support, density.weight)
        Z, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)
        t2 = time.perf_counter()

    elif solver == "xsdd":
        #print("using xsdd")

        # xsdd
        t1 = time.perf_counter()
        xsdd = XsddEngine(density.domain, density.support, density.weight,
                          factorized=False, algebra=PyXaddAlgebra(), ordered=False)
        Z = xsdd.compute_volume(add_bounds=False)
        t2 = time.perf_counter()

    else:
        print(f"Unrecognized solver: {solver}")
        exit()

    result = (t2 - t1, Z)
    queue.put(result)

        





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('solver', type=str,
                        help='Problem solver')

    parser.add_argument("--benchmark-dir", type=str, default=DEF_BENCHMARK_DIR,                        
                        help='Benchmark directory')

    parser.add_argument("--results-dir", type=str, default=DEF_RESULTS_DIR,
                        help='Results directory')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=DEF_SEED,
                        help='Seed for the random generator')

    parser.add_argument('--shape', type=str, nargs='+',
                        default=DEF_SHAPE,
                        help='Shape of the dependency graph')

    parser.add_argument('--rep', type=int, nargs='+',
                        default=list(range(DEF_REP)),
                        help='Indices of the densities')

    parser.add_argument('--vars', type=int, nargs='+',
                        default=DEF_N_VARS,
                        help='Number of random variables')

    parser.add_argument('--clauses', type=int, nargs='+',
                        default=DEF_N_CLAUSES,
                        help='Number of clauses')

    parser.add_argument('--lits', type=int, nargs='+',
                        default=DEF_N_LITS,
                        help='Number of literals per clause')

    parser.add_argument('--degree', type=int, nargs='+',
                        default=DEF_DEGREE,
                        help='Number of max polynomial degree')

    parser.add_argument('--timeout', type=int,
                        default=DEF_TIMEOUT,
                        help='Timeout (in seconds)')

    parser.add_argument('--overwrite', dest='overwrite',
                        action='store_true')
    parser.set_defaults(overwrite=False)

    #
    # parsing the args
    args = parser.parse_args()

    problem_configs = list(product(args.shape,
                                   args.vars,
                                   args.clauses,
                                   args.lits,
                                   args.degree,
                                   args.rep))

    for conf in problem_configs:

        shape, nv, nc, nl, d, i = conf
        print(f"\n#########################################\n"
              f"{shape}: {nv} VARS, {nc} CLAUSES, {nl} LITS, {d} degree, {i} index")

        density_fullpath = os.path.join(args.benchmark_dir, density_filename(conf))
        results_fullpath = os.path.join(args.results_dir, results_filename(conf, args.solver))

        if (all(os.path.exists(os.path.join(results_fullpath,r)) for r in ['Z', 'time'])
            and not args.overwrite):
            print(f"'{results_fullpath}' exists. Skipping.")
            continue
        
        os.makedirs(results_fullpath, exist_ok=True)

        print(f"Saving results to {results_fullpath}")
        print(f"Looking for density in: {density_fullpath}")

        density = Density.from_file(density_fullpath)

        queue = Queue()
        proc = Process(target=wrapped_solver,
                       args=(args.solver, density, queue,))

        try:
            print(f"****** STARTING PID: {proc.pid}")
            proc.start()
            print(f"****** WAITING FOR JOIN PID: {proc.pid}")
            proc.join(timeout=args.timeout)
            print(f"****** JOINED -> TERMINATING PID: {proc.pid}")
            proc.terminate() # proc.kill() python>3.7

            if proc.is_alive():
                pid = proc.pid
                print(f"Killing process {pid} and its children")
                kill_recursive(pid)

            if not queue.empty():
                t, Z = queue.get()
                print(f"done in {t} secs")
                print(f"Z: {Z}")

            else:
                t = args.timeout
                Z = 'timeout'
                print("TIMEOUT")

        except BrokenPipeError:
            print("catched BrokenPipeError, aborting")
            continue

        z_path = os.path.join(results_fullpath, 'Z')
        with open(z_path, 'w') as f:
            f.write(f"{Z}\n")
            print(f"Z saved to {z_path}")

        t_path = os.path.join(results_fullpath, 'time')
        with open(t_path, 'w') as f:
            f.write(f"{t}\n")
            print(f"time saved to {t_path}")
