import argparse
from time import perf_counter
from os import makedirs
from os.path import abspath, isfile, join

from numpy.random import RandomState

from generators.formulas import ForestGenerator
from generators.weights import RandomPolynomials
from itertools import product


from pysmt.shortcuts import Ite, Real, Times
from pywmi import Density

PROBLEMS = {'p': 'PATH', 'sn-3': 'SNOW-3', 'st': 'STAR', 't': 'SNOW-2'}


def filename_from_config(config, index):
    n_vars, n_clauses, n_lits, shape, degree = config
    # return f"density_{n_vars}_{n_clauses}_{n_lits}_{shape}_{degree}_{index}.json"
    return f"{PROBLEMS[shape]}_{n_vars}_{n_clauses}_{n_lits}_{degree}.{index}.wmi"


# seed number used to generate the suite
SEED = 666

# ratio of atoms in the support having a (polynomial) weight
WEIGHT_RATIO = 0.1

# number of densities for each configuration
DENSITIES_PER_CONFIG = 1

# number of queries associated to each density
QUERIES_PER_DENSITY = 10

# CONFIGS are the cartesian product of:
# number of continuous variables
N_VARS = [2, 4, 6, 8, 10, 20, 30, 40]
# number of additional clauses (univar. bounds excluded)
N_CLAUSES = [1, 2, 3]
# number of literals in each additional clause
N_LITS = [2, 4, 6]
# shape of the resulting primal graph
# - path
# - snow-3
# - star
# - tree
# SHAPE = ['p', 'sn-3', 'st', 't']
SHAPE = ['p', 'sn-3', 'st']
# degree of the polynomial weights associated with the atoms
DEGREE = [2, 4, 6, 8]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str,
                        help='Benchmark output directory')

    parser.add_argument('--w-ratio', type=float,
                        default=WEIGHT_RATIO,
                        help='Ratio of atoms in the support having a (polynomial) weight')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=SEED,
                        help='Seed for the random generator')

    parser.add_argument('--rep', type=int,
                        default=DENSITIES_PER_CONFIG,
                        help='Number of repetitions, i.e., densities per configuration')

    parser.add_argument('--queries', type=int,
                        default=QUERIES_PER_DENSITY,
                        help='Number of queries per density')

    parser.add_argument('--vars', type=int, nargs='+',
                        default=N_VARS,
                        help='Number of random variables')

    parser.add_argument('--clauses', type=int, nargs='+',
                        default=N_CLAUSES,
                        help='Number of clauses for each edge')

    parser.add_argument('--lits', type=int, nargs='+',
                        default=N_LITS,
                        help='Number of literals per clause')

    parser.add_argument('--shape', type=str, nargs='+',
                        default=SHAPE,
                        help='Problem shapes')

    parser.add_argument('--degree', type=int, nargs='+',
                        default=DEGREE,
                        help='Number of max polynomial degree')

    #
    # parsing the args
    args = parser.parse_args()
    makedirs(args.dir, exist_ok=True)

    #
    # random generators
    rand_gen = RandomState(args.seed)
    fgen = ForestGenerator(rand_gen)
    pgen = RandomPolynomials(rand_gen)

    CONFIGS = product(args.vars, args.clauses, args.lits, args.shape)

    bench_start_t = perf_counter()
    for config in CONFIGS:
        n_vars, n_clauses, n_lits, shape = config
        print(
            f"generating the formula with n_vars : {n_vars}, n_clauses : {n_clauses}, n_lits : {n_lits}, shape : {shape}")

        for index in range(args.rep):

            # generate support and domain
            f, dom = fgen.random_formula(n_vars,
                                         n_clauses,
                                         n_lits,
                                         shape)

            f_atoms = list(f.get_atoms())
            
            for degree in args.degree:
                print(f"generating weight with degree: {degree}")
                path_density = join(args.dir, filename_from_config(config+(degree,), index))

                if isfile(path_density):
                    print(f"WARNING: {path_density} exists. Skipping.")
                    continue

                random_atoms = rand_gen.choice(f_atoms, max([1, int(args.w_ratio * len(f_atoms))]),
                                               replace=False)

                # generate the weight function
                w = Times([Ite(atom,
                               pgen.random_polynomial(list(atom.get_free_variables()), degree, nonnegative=True),
                               Real(1)) for atom in random_atoms])

                # generate the queries
                queries = []
                f_deps = list(filter(lambda x: len(x) == 2,
                                     [tuple(a.get_free_variables()) for a in f_atoms]))

                for j in range(args.queries):                
                    x, y = f_deps[rand_gen.choice(range(len(f_deps)))]
                    q = fgen.random_literal(x, y)
                    queries.append(q)


                # export the density
                print(f"dumping density at '{path_density}'")
                density = Density(dom, f, w, queries)
                density.to_file(path_density)

    bench_end_t = perf_counter()
    print(f"{len(list(CONFIGS))} benchmarks generated in {bench_end_t - bench_start_t} secs")
