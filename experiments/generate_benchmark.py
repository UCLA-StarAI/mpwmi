import argparse
from time import perf_counter
from os import makedirs
from os.path import abspath, isfile, join

from numpy.random import RandomState

from generators.formulas import ForestGenerator
from generators.weights import RandomPolynomials
from exp_defaults import *
from itertools import product


from pysmt.shortcuts import Ite, Real, Times
from pywmi import Density




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, default=DEF_BENCHMARK_DIR,
                        help='Benchmark output directory')

    parser.add_argument('--w-ratio', type=float,
                        default=DEF_WEIGHT_RATIO,
                        help='Ratio of atoms in the support having a (polynomial) weight')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=DEF_SEED,
                        help='Seed for the random generator')

    parser.add_argument('--rep', type=int, nargs='+',
                        default=list(range(DEF_REP)),
                        help='Indices of the densities')

    parser.add_argument('--vars', type=int, nargs='+',
                        default=DEF_N_VARS,
                        help='Number of random variables')

    parser.add_argument('--clauses', type=int, nargs='+',
                        default=DEF_N_CLAUSES,
                        help='Number of clauses for each edge')

    parser.add_argument('--lits', type=int, nargs='+',
                        default=DEF_N_LITS,
                        help='Number of literals per clause')

    parser.add_argument('--shape', type=str, nargs='+',
                        default=DEF_SHAPE,
                        help='Problem shapes')

    parser.add_argument('--degree', type=int, nargs='+',
                        default=DEF_DEGREE,
                        help='Number of max polynomial degree')

    parser.add_argument('--overwrite', dest='overwrite',
                        action='store_true')
    parser.set_defaults(overwrite=False)


    # parsing the args
    args = parser.parse_args()
    makedirs(args.benchmark_dir, exist_ok=True)

    #
    # random generators
    rand_gen = RandomState(args.seed)
    fgen = ForestGenerator(rand_gen)
    pgen = RandomPolynomials(rand_gen)

    problem_configs = list(product(args.shape,
                                   args.vars,
                                   args.clauses,
                                   args.lits,
                                   args.degree,
                                   args.rep))

    bench_start_t = perf_counter()
    for conf in problem_configs:
        shape, nv, nc, nl, d, i = conf
        print(f"\n#########################################\n"
              f"{shape}: {nv} VARS, {nc} CLAUSES, {nl} LITS, {d} degree, {i} index")

        density_fullpath = join(args.benchmark_dir, density_filename(conf))
        if isfile(density_fullpath) and not args.overwrite:
            print(f"WARNING: {density_fullpath} exists. Skipping.")
            continue
        
        # generate support and domain
        f, dom = fgen.random_formula(nv, nc, nl, shape)
        f_atoms = list(f.get_atoms())
            
        random_atoms = rand_gen.choice(f_atoms, max([1, int(args.w_ratio * len(f_atoms))]),
                                       replace=False)
        # generate the weight function
        w = Times([Ite(atom, pgen.random_polynomial(list(atom.get_free_variables()), d, nonnegative=True),
                       Real(1)) for atom in random_atoms])

        # no queries
        queries = []

        # export the density
        print(f"dumping density at '{density_fullpath}'")
        density = Density(dom, f, w, queries)
        density.to_file(density_fullpath)

    bench_end_t = perf_counter()
    print(f"{len(problem_configs)} benchmarks generated in {bench_end_t - bench_start_t} secs")
