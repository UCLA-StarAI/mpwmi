


import argparse
from exp_defaults import *
from itertools import product
from math import fsum
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
import os


# settings for plotting
fontsize = 28
matplotlib.rcParams.update({'xtick.labelsize': fontsize - 4,
                            'ytick.labelsize': fontsize - 4,
                            'axes.labelsize': fontsize,
                            'axes.titlesize': fontsize,
                            'legend.fontsize': fontsize - 6,
                            'figure.autolayout': True,
                            'pdf.fonttype': 42})


def generate_colors(solvers):
    ssolvers = list(enumerate(sorted(solvers)))
    vmin = 0
    vmax = ssolvers[-1][0]
    scalarMap = cmx.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax),
                                   cmap=plt.get_cmap('cool'))
    colordict = {s : scalarMap.to_rgba(i) for i, s in ssolvers}
    return colordict
    
    
def generate_markers(solvers):
    MARKERS = ['^', 'x', 'v']
    return {s : MARKERS[i%len(MARKERS)]
            for i,s in enumerate(solvers)}
    


def get_results(results_dir, shape, solver,
                vars, clauses, lits, degree, rep):

    partial_configs = list(product(vars, clauses, lits, degree))

    Z = {}
    time = {}
    for pconf in partial_configs:
        Z_pconf = []
        time_pconf = []
        for i in rep:
            conf = [shape] + list(pconf) + [i]
            Z_path = os.path.join(results_dir, results_filename(conf, solver), "Z")
            time_path = os.path.join(results_dir, results_filename(conf, solver), "time")

            with open(Z_path, 'r') as f:
                z = float(f.read().strip())

            with open(time_path, 'r') as f:
                t = float(f.read().strip())

            Z_pconf.append(z)
            time_pconf.append(t)

        Z[pconf] = Z_pconf
        time[pconf] = time_pconf

    return Z, time


def is_consistent(Z):
    consistent = True
    solvers = list(Z.keys())
    reference_confs = set(Z[solvers[0]].keys())
    for i in range(1, len(solvers)):
        if not reference_confs == set(Z[solvers[i]].keys()):
            print("mismatch in configurations:")
            print(f"len({solvers[0]}) = {len(reference_confs)}")
            print("\n".join(map(str, reference_confs)))
            print()
            print(f"len({solvers[i]}) = {len(Z[solvers[i]].keys())}")
            print("\n".join(map(str, Z[solvers[i]].keys())))
            print()
            consistent = False

    for pc in reference_confs:
        reference_result = Z[solvers[0]][pc]
        for i in range(1, len(solvers)):
            if (not len(reference_result) == len(Z[solvers[i]][pc])
                or not all(np.isclose(reference_result[j], Z[solvers[i]][pc][j])
                       for j in range(len(reference_result)))):
                print(f"{pc} mismatch in volumes:")
                for i in range(len(solvers)):
                    print(f"Z[{solvers[i]}] = {Z[solvers[i]][pc]}")
                print()
                consistent = False

    return consistent


def plot_time(time, output=None, x_dim=0, fig_size=(10, 7), show=False):

    solver_color = generate_colors(time.keys())
    solver_marker = generate_markers(time.keys())
    

    fig, ax = plt.subplots(figsize=fig_size)    

    plt.xlabel(['# variables', '# clauses', '# lits', 'factor degree'][x_dim])
    plt.ylabel('Execution time (s)')

    for solver in time:
        xs = sorted({pconf[x_dim] for pconf in time[solver].keys()})
        xconfs = {x : [pconf for pconf in time[solver].keys()
                          if pconf[x_dim] == x] for x in xs}        
        ys = []
        for x in xs:
            aggr = []
            for pconf in xconfs[x]:
                for y in time[solver][pconf]:
                    aggr.append(y)
                    ax.scatter(x, y, s=20,
                               marker=solver_marker[solver],
                               color=solver_color[solver])

            ys.append(fsum(aggr)/len(aggr))

        ax.plot(xs, ys, markersize=20,
                label=solver,
                marker=solver_marker[solver],
                color=solver_color[solver])

    ax.legend()
    if show:
        plt.show()

    if output is not None:
        plt.savefig(output)
        print(f"Plotted to {output}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--solvers', type=str, required=True,
                        nargs='+',
                        help='Problem solvers')
    
    parser.add_argument("--results-dir", type=str, default=DEF_RESULTS_DIR,
                        help='Results directory')

    parser.add_argument('--plots-dir', type=str,
                        default=DEF_PLOTS_DIR,
                        help='Problem solvers')

    parser.add_argument('--shape', type=str, nargs='+',
                        default=DEF_SHAPE,
                        help='Problem shapes')

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

    parser.add_argument('--rep', type=int, nargs='+',
                        default=list(range(DEF_REP)),
                        help='Number of repetitions, i.e., densities per configuration')



    args = parser.parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    print("Plotting time:")

    # one plot for each problem shape
    for shape in args.shape:
        print("Processing", shape)
        Z = {}
        time = {}

        # one curve for each solver
        for solver in args.solvers:
            Z_solver, time_solver = get_results(args.results_dir, shape, solver,
                                                args.vars, args.clauses, args.lits,
                                                args.degree, args.rep)
            Z[solver] = Z_solver
            time[solver] = time_solver

        # check consistency on Z
        if not is_consistent(Z):
            print("ouch..")
            exit()

        # plot execution time
        plot_fullpath = os.path.join(args.plots_dir, f'{shape}-time.pdf')
        plot_time(time,
                  output=plot_fullpath,
                  x_dim=0,
                  fig_size=(10, 7),
                  show=False)
