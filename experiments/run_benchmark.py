import sys
import os
import argparse
import time
from itertools import product

from exp_defaults import *

from pywmi import Density

from wmipa import WMI
from pysmt.shortcuts import Bool
from pywmi import XsddEngine, PyXaddAlgebra
from pympwmi import MPWMI, oldMP2WMI, oldMPWMI




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

    parser.add_argument('--smt-solver', type=str,
                        default=DEF_SMT_SOLVER,
                        help='Default SMT solver in pySMT (for mpwmi)')

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

        t1, t2, Z = None, None, None

        if args.solver.startswith("mpwmi-"): # should be "mpwmi-[numeric/symbolic]-NPROC"
            solverargs = args.solver.split("-")[1:]
            msgtype = solverargs[0]
            nproc = int(solverargs[1])
            print(f"using {msgtype} mpwmi with {nproc} processes")
            # mpwmi
            t1 = time.perf_counter()
            print("About to start mpwmi solver")
            mpmi = MPWMI(density.support, density.weight, smt_solver=args.smt_solver,
                          n_processes=nproc, msgtype=msgtype)
            print("Solver inited")
            Z, _ = mpmi.compute_volumes(cache=False)
            print("Volume computed")
            t2 = time.perf_counter()

        elif args.solver.startswith("oldmp2wmi-"): # should be "mpwmi-[numeric/symbolic]-NPROC"
            solverargs = args.solver.split("-")[1:]
            nproc = int(solverargs[0])
            print(f"using (old) mp2wmi with {nproc} processes")
            # mpwmi
            t1 = time.perf_counter()
            print("About to start mpwmi solver")
            mpmi = oldMP2WMI(density.support, density.weight, smt_solver=args.smt_solver,
                          n_processes=nproc)
            print("Solver inited")
            Z, _ = mpmi.compute_volumes(cache=False)
            print("Volume computed")
            t2 = time.perf_counter()

        elif args.solver.startswith("oldmpwmi"): # should be "mpwmi-[numeric/symbolic]-NPROC"
            print(f"using oldmpwmi")
            # mpwmi
            t1 = time.perf_counter()
            print("About to start mpwmi solver")
            mpmi = oldMPWMI(density.support, density.weight, smt_solver=args.smt_solver)
            print("Solver inited")
            Z, _ = mpmi.compute_volumes(cache=False)
            print("Volume computed")
            t2 = time.perf_counter()


        elif args.solver == "pa":
            print("using pa")

            # pa
            t1 = time.perf_counter()
            wmipa = WMI(density.support, density.weight)
            Z, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)
            t2 = time.perf_counter()

        elif args.solver == "xsdd":
            print("using xsdd")

            # xsdd
            t1 = time.perf_counter()
            xsdd = XsddEngine(density.domain, density.support, density.weight,
                              factorized=False, algebra=PyXaddAlgebra(), ordered=False)
            Z = xsdd.compute_volume(add_bounds=False)
            t2 = time.perf_counter()

        else:
            print(f"Unrecognized solver: {args.solver}")

        print(f"done in {t2-t1} secs")
        print(f"Z: {Z}")

        z_path = os.path.join(results_fullpath, 'Z')
        with open(z_path, 'w') as f:
            f.write(f"{Z}\n")
            print(f"Z saved to {z_path}")

        t_path = os.path.join(results_fullpath, 'time')
        with open(t_path, 'w') as f:
            f.write(f"{t2-t1}\n")
            print(f"time saved to {t_path}")
