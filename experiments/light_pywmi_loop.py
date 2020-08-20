import sys
import os
import logging
import argparse
import time
import itertools

from pysmt.shortcuts import Bool
from pywmi import Density
from wmipa import WMI
from pywmi import XsddEngine, PyXaddAlgebra
from mpwmi import MPWMI, MP2WMI

#
# seed number
SEED = 666

#
# timeout for a solver on a single WMI problem
TIMEOUT = 1800

#
# benchmark problems
PROBLEMS = {'PATH', 'SNOW-3', 'STAR'}
#
#
# number of continuous variables
# N_VARS = [5, 10, 15, 20, 25, 30, 35, 40]
N_VARS = [2, 4, 6, 8, 10, 20, 30, 40]
#
# number of additional clauses (univar. bounds excluded)
# N_CLAUSES = [1, 5, 10, 15]
N_CLAUSES = [1, 2, 3]
#
# number of literals in each additional clause
# N_LITS = [1, 2, 3, 4, 5]
N_LITS = [2, 4, 6]
#
# degree of the polynomial weights associated with the atoms
DEGREE = [2, 4, 6, 8]


#
#
# python interpreter
# PY_CMD = "ipython3 -- "
PY_CMD = "python3 "

#
# output dir
EXP_DIR = "results"

#
# default SMT solver to be used in pySMT
SMT_SOLVER = "msat"
# SMT_SOLVER = "z3"
# SMT_SOLVER = "cvc4"


def get_res_path(problem_name):
    problem_name_list = problem_name.split('.')
    problem_name = f"{problem_name_list[0]}_{problem_name_list[1]}"
    return os.path.join(*(p for p in problem_name.split('_')))


def filename_from_config(n_vars, n_clauses, n_lits, degree, index, shape):
    # return f"density_{n_vars}_{n_clauses}_{n_lits}_{shape}_{degree}_{index}.json"
    return f"{shape}_{n_vars}_{n_clauses}_{n_lits}_{degree}.{index}.wmi"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('solver', type=str,
                        help='Problem solver')

    parser.add_argument("--dir", type=str, required=True,
                        help='Benchmark output directory')

    parser.add_argument("--expdir", type=str, default=EXP_DIR,
                        help='Command output directory')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=SEED,
                        help='Seed for the random generator')

    parser.add_argument('--problems', type=str, nargs='+',
                        default=PROBLEMS,
                        help='Problems in the benchmarks')

    parser.add_argument('--smt-solver', type=str,
                        default=SMT_SOLVER,
                        help='Default SMT solver in pySMT (for mpwmi)')

    parser.add_argument('--rep', type=int, nargs='+',
                        default=[0],
                        help='Number of repetitions, i.e., densities per configuration')

    parser.add_argument('--vars', type=int, nargs='+',
                        default=N_VARS,
                        help='Number of random variables')

    parser.add_argument('--id', type=int,
                        help='Fake id for debugging purposes')

    parser.add_argument('--clauses', type=int, nargs='+',
                        default=N_CLAUSES,
                        help='Number of clauses')

    parser.add_argument('--lits', type=int, nargs='+',
                        default=N_LITS,
                        help='Number of literals per clause')

    parser.add_argument('--degree', type=int, nargs='+',
                        default=DEGREE,
                        help='Number of max polynomial degree')

    #
    # parsing the args
    args = parser.parse_args()

    #
    # Logging
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(funcName)s:%(lineno)d]\t %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # file_handler = logging.FileHandler(os.path.join(res_path, f"log"))
    # file_handler.setFormatter(log_formatter)
    # root_logger.addHandler(file_handler)

    # and to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    problem_configs = list(itertools.product(args.vars,
                                             args.clauses,
                                             args.lits,
                                             args.degree,
                                             # [k for k in range(args.rep)],
                                             args.rep,
                                             args.problems))

    print(args.vars,
          args.clauses,
          args.lits,
          args.degree,
          # [k for k in range(args.rep)],
          args.rep,
          args.problems)
    print('PP', problem_configs)

    for p in problem_configs:

        v, c, l, d, r, ss = p
        print(f"\n#########################################\n"
              f"{ss}: {v} VARS, {c} CLAUSES, {l} LITS, {d} degree, {r} rep")

        wmi_problem_name = filename_from_config(v,
                                                c,
                                                l,
                                                d,
                                                r,
                                                ss)

        wmi_problem_path = os.path.join(args.dir, wmi_problem_name)
        res_path = os.path.join(args.expdir, get_res_path(wmi_problem_name))
        os.makedirs(res_path, exist_ok=True)

        logging.info(f"Saving results to {res_path}")
        logging.info(f"looking for wmi problem: {wmi_problem_path}")

        density = Density.from_file(wmi_problem_path)

        t1, t2, Z = None, None, None

        if args.solver == "mpwmi":
            logging.info("using mpwmi")
            # mpwmi
            t1 = time.perf_counter()
            logging.info("About to start mpwmi solver")
            mpmi = MPWMI(density.support, density.weight, smt_solver=args.smt_solver)
            logging.info("Solver inited")
            Z, _ = mpmi.compute_volumes(cache=True)
            logging.info("Volume computed")
            t2 = time.perf_counter()

        if args.solver.startswith("mp2wmi"): # should be "mp2wmi-NPROC"
            nproc = int(args.solver.partition("-")[-1])
            logging.info(f"using multiprocessing mpwmi with {nproc} processes")
            # mpwmi
            t1 = time.perf_counter()
            logging.info("About to start mpwmi solver")
            mpmi = MP2WMI(density.support, density.weight, smt_solver=args.smt_solver,
                         n_processes=nproc)
            logging.info("Solver inited")
            Z, _ = mpmi.compute_volumes(cache=True)
            logging.info("Volume computed")
            t2 = time.perf_counter()

        elif args.solver == "pa":
            logging.info("using pa")

            # pa
            t1 = time.perf_counter()
            wmipa = WMI(density.support, density.weight)
            Z, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)
            t2 = time.perf_counter()

        elif args.solver == "xsdd":
            logging.info("using xsdd")

            # xsdd
            t1 = time.perf_counter()
            xsdd = XsddEngine(density.domain, density.support, density.weight,
                              factorized=False, algebra=PyXaddAlgebra(), ordered=False)
            Z = xsdd.compute_volume(add_bounds=False)
            t2 = time.perf_counter()

        else:
            logging.info(f"Unrecognized solver: {args.solver}")

        logging.info(f"done in {t2-t1} secs")
        logging.info(f"Z: {Z}")

        z_path = os.path.join(res_path, 'Z')
        with open(z_path, 'w') as f:
            f.write(f"{Z}\n")
            logging.info(f"Z saved to {z_path}")

        t_path = os.path.join(res_path, 'time')
        with open(t_path, 'w') as f:
            f.write(f"{t2-t1}\n")
            logging.info(f"time saved to {t_path}")
