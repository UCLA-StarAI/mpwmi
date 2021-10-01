
import os

def density_filename(config):
    shape, n_vars, n_clauses, n_lits, degree, index = config
    # return f"density_{n_vars}_{n_clauses}_{n_lits}_{shape}_{degree}_{index}.json"
    return f"{shape}_{n_vars}_{n_clauses}_{n_lits}_{degree}.{index}.wmi"


def results_filename(config, solver):
    subdirs = map(str, [solver]+list(config))
    return os.path.join(*subdirs)



########## DEFAULT BENCHMARK PARAMETERS ##########
DEF_SHAPE = ['PATH', 'SNOW-3', 'STAR', 'SNOW-2']
# number of continuous variables
DEF_N_VARS = [2, 4, 6, 8, 10, 20, 30, 40]
# number of additional clauses (univar. bounds excluded)
DEF_N_CLAUSES = [1, 2, 3]
# number of literals in each additional clause
DEF_N_LITS = [2, 4, 6]
# degree of the polynomial weights associated with the atoms
DEF_DEGREE = [2, 4, 6, 8]
# number of densities for each configuration
DEF_REP = 1

########## GENERATE ##########
# seed number used to generate the suite
DEF_SEED = 666

# ratio of atoms in the support having a (polynomial) weight
DEF_WEIGHT_RATIO = 0.1

# number of queries associated to each density
#DEF_QUERIES_PER_DENSITY = 10

########## RUN ##########
DEF_TIMEOUT = 1200 # seconds

########## FOLDERS ##########
DEF_BENCHMARK_DIR = "benchmark"
DEF_RESULTS_DIR = "results"
DEF_PLOTS_DIR = "plots"
