

from fractions import Fraction
from multiprocessing import Pool, Manager
import networkx as nx
from networkx.algorithms.components import connected_components
import numpy as np
from pysmt.shortcuts import And, LE, LT, Not, Or, Real, is_sat
from pysmt.environment import get_env, push_env

from pympwmi.message import SympyMessage as Message
from pympwmi.primal import PrimalGraph
from pympwmi.utils import *




class MPWMI3:
    """A class that implements the Message Passing Weighted Model
    Integration exact inference algorithm. Works with SMT-LRA formula
    in CNF that have a forest-shaped primal graph, raises
    NotImplementedError otherwise.

    The weight expression has to be:
        Times(Ite(lit_1, w_1, Real(1)), ..., Ite(lit_n, w_n, Real(1)))

    otherwise, ValueError is raised.

    Attributes
    ----------
    cache : dict
        Cache for integrals, or None
    mailboxes : dict
        Dictionary storing the symbolic mailboxes of each variable given evidence
    primal : PrimalGraph instance
        The primal graph of 'formula'
    rand_gen : np.random.RandomState instance
        The random number generator
    smt_solver : pysmt.Solver instance
        The SMT solver used by MPWMI
    tolerance : float
        The tolerance parameter

    Methods
    -------
    compute_volumes(queries=None, evidence=None, cache=False)
        Computes the partition function value and the unnormalized probabilities
        of uni/bivariate literals in 'queries'.

    """

    def __init__(self, formula, weight, smt_solver=SMT_SOLVER, rand_gen=None, tolerance=0.0, #tolerance=1.49e-8,
                 n_processes=1):
        """
        Parameters
        ----------
        formula : pysmt.FNode instance
            The input formula representing the support of the distribution
        weight : pysmt.FNode instance
            Polynomial potentials attached to literal values
        rand_gen : np.random.RandomState instance, optional
            The random number generator (default: RandomState(mpwmi.RAND_SEED))
        """

        if rand_gen is None:
            from pympwmi.utils import RAND_SEED
            rand_gen = np.random.RandomState(RAND_SEED)

        self.rand_gen = rand_gen
        self.smt_solver = smt_solver
        self.tolerance = tolerance

        self.primal = PrimalGraph(formula, weight)

        self.mailboxes = dict()
        self.cache = None
        self.cache_hit = None

        self.n_processes = n_processes

    def compute_volumes(self, queries=None, evidence=None, cache=True):
        """Computes the unnormalized probabilities of univariate and
        bivariate literals in 'queries' associated to univariate
        literals and a list of uni/bivariate clauses representing the
        'evidence'.

        Returns (Z given evidence, list[volumes of queries given evidence]).

        Raises NotImplementedError if the literals are not uni/bivariate.

        Parameters
        ----------
        queries : list of pysmt.FNode instances (optional)
            Uni/bivariate literals, default: None
        evidence : iterable of pysmt.FNode instances (optional)
            Uni/bivariate clauses, default: None
        cache : bool (optional)
            If True, integrals are cached, default: True
        """

        if not nx.is_forest(self.primal.G):
            raise NotImplementedError("MP requires a forest-shaped primal graph")

        if queries is None:
            queries = []
        else:
            queries = [flip_negated_literals_cnf(q) for q in queries]

        if cache is False:
            self.cache = None
        elif cache is True and self.cache is None:
            self.cache = Manager().dict()
            self.cache_hit = [0, 0]
        else:
            self.cache = Manager().dict(self.cache) # needed?


        # message passing is parallelized over connected components in
        # the primal graph
        components = list(connected_components(self.primal.G))
        subproblems = []
        pysmt_env = get_env()
        for comp_vars in components:
            subprimal = self.primal.subprimal(comp_vars)
            subvars = {subprimal.nodes()[n]['var'] for n in subprimal.nodes()}

            if evidence is None:
                subevidence = None
            else:
                subevidence = [e for e in evidence
                               if set(e.get_free_symbols()).issubset(subvars)]
            subproblems.append((subprimal, self.smt_solver, self.cache,
                                self.tolerance, self.rand_gen, pysmt_env, subevidence))

        with Pool(processes=self.n_processes) as pool:
            results = pool.starmap(MPWMI3._message_passing, subproblems)

        for messages, ch in results:
            self.mailboxes.update(messages)
            if self.cache is not None:
                self.cache_hit[True] += ch[True]
                self.cache_hit[False] += ch[False]
        
        Z_components = []
        for comp_vars in components:
            x = list(comp_vars)[0]
            symbolic_marginal = Message.intersect(list(self.mailboxes[x].values()),
                                                  self.tolerance)
        
            comp_Z, ch = Message.integrate(self.cache, symbolic_marginal, x)
            if self.cache is not None:
                self.cache_hit[True] += ch[True]
                self.cache_hit[False] += ch[False]

            Z_components.append(comp_Z)

        query_volumes = []
        for q in queries:
            q_vars = list(q.get_free_variables())
            if not all([qv.symbol_type() == REAL for qv in q_vars]):
                raise NotImplementedError("Supporting lra queries only")

            x = q_vars[0].symbol_name()
            if len(q_vars) == 1:

                # MP messages + univariate query
                msgs = list(self.mailboxes[x].values())
                l, u = domains_to_intervals(q)[0]
                msgs.append([(l, u, Message.ONE)])                
                query_marginal = Message.intersect(msgs, self.tolerance)
                
                q_vol, ch = Message.integrate(self.cache, query_marginal, x)
                if self.cache is not None:
                    self.cache_hit[True] += ch[True]
                    self.cache_hit[False] += ch[False]

                # account for the volume of unconnected variables
                for i, comp_vars in enumerate(components):
                    if x not in comp_vars:
                        q_vol *= Z_components[i]

                query_volumes.append(q_vol)

            elif len(q_vars) == 2:

                # bivariate query
                y = q_vars[1].symbol_name()

                # creates a new message using the query 'q' as evidence
                q_msg, ch = MPWMI3._compute_message(self.primal, self.mailboxes,
                                                         self.smt_solver, self.cache,
                                                         self.tolerance, x, y, evidence=[q])

                if self.cache is not None:
                    self.cache_hit[True] += ch[True]
                    self.cache_hit[False] += ch[False]

                msgs =  [q_msg] + [self.mailboxes[y][z]
                                   for z in self.mailboxes[y] if z != x]
                query_marginal = Message.intersect(msgs, self.tolerance)
                q_vol, ch = Message.integrate(self.cache, query_marginal, y)

                if self.cache is not None:
                    self.cache_hit[True] += ch[True]
                    self.cache_hit[False] += ch[False]

                # account for the volume of unconnected variables
                for i, comp_vars in enumerate(components):
                    if x not in comp_vars:
                        q_vol *= Z_components[i]

                query_volumes.append(q_vol)

            else:
                raise NotImplementedError(
                    "Queries of ariety > 2 aren't supported")

        Z = 1.0
        for Z_comp in Z_components:
            Z *= Z_comp

        if self.cache is not None:
            # TODO: check if cache_hit index should be True or False
            print("\tHITS: {}/{} (ratio {})".format(self.cache_hit[True],
                                                    sum(self.cache_hit),
                                                    self.cache_hit[True] /
                                                    sum(self.cache_hit)))

        Z = float(Z.as_expr())
        query_volumes = [float(qv.as_expr()) for qv in query_volumes]
        return Z, query_volumes

    @staticmethod
    def _message_passing(primal, smt_solver, cache, tolerance,
                         rand_gen, pysmt_env, evidence=None):
        """
        Computes the symbolic piecewise integrals representing the marginal of
        each node. Performs a message passing step:
        1) The primal forest is randomly directed
        2) Nodes send/receive messages to/from every neighbor
           accounting for potentials and evidence when computing their mailboxes

        Parameters
        ----------
        evidence : list (optional)
            List of uni/bivariate clauses considered as evidence when computing
            the mailboxes
        """

        push_env(pysmt_env)

        if cache is not None:
            cache_hit = [0, 0]
        else:
            cache_hit = None
            
        if evidence is None:
            evidence = []

        # initialize mailbox of each node
        mailboxes = {n : dict() for n in primal.nodes()}

        # get a random directed forest from the primal graph
        # (nodes having at most in-degree 1)
        topdown = nx.DiGraph()
        left = set(primal.nodes())
        while len(left) > 0:
            root = rand_gen.choice(list(left))
            newtree = nx.bfs_tree(primal.G, root)
            topdown = nx.compose(topdown, newtree)
            left = left.difference(set(newtree.nodes))

        # bottom-up pass first
        bottomup = nx.DiGraph(topdown).reverse()
        # pick an arbitrary topological node order in the bottom-up graph
        exec_order = [n for n in nx.topological_sort(bottomup)]
        for n in exec_order:

            # account for univariate bounds/potentials first
            mailboxes[n][n] = MPWMI3._basecase_msg(primal, n)
            if len(primal.nodes()[n]['potentials']) > 0:
                aggr = [mailboxes[n][n]] + Message.potentials_to_messages(
                    primal.nodes()[n]['potentials'],
                    primal.nodes()[n]['var'])
                mailboxes[n][n] = Message.intersect(aggr, tolerance)
            
            parents = list(bottomup.neighbors(n))
            assert (len(parents) < 2), "this shouldn't happen"
            if len(parents) == 1:
                parent = parents[0]
                mailboxes[parent][n], ch = MPWMI3._compute_message(primal, mailboxes,
                                                                   smt_solver, cache,
                                                                   tolerance, n, parent,
                                                                   evidence=evidence)
                if cache is not None:
                    cache_hit[True] += ch[True]
                    cache_hit[False] += ch[False]


        # top-down pass
        exec_order.reverse()
        for n in exec_order:
            for child in topdown.neighbors(n):
                mailboxes[child][n], ch = MPWMI3._compute_message(primal, mailboxes, smt_solver,
                                                                  cache, tolerance, n, child,
                                                                  evidence=evidence)
                if ch is not None:
                    cache_hit[True] += ch[True]
                    cache_hit[False] += ch[False]

        return mailboxes, cache_hit

    @staticmethod
    def _basecase_msg(primal, x):
        """
        Computes the piecewise integral for a leaf node.

        Parameters
        ----------
        x : str
            Name of the variable in the primal graph
        """

        #assert(primal.G.degree[x] <= 1)
        intervals = domains_to_intervals(primal.get_univariate_formula(x))
        return list(map(lambda i: (i[0], i[1], Message.ONE), intervals))

    @staticmethod
    def _compute_message(primal, mailboxes, smt_solver, cache, tolerance,
                         x, y, evidence=None):
        """
        Returns a message from node 'x' to node 'y', possibly accounting for
        potentials and evidence.

        Parameters
        ----------
        x : str
        y : str
            The nodes on the primal graph having an edge connecting them.
        evidence : list (optional)
            List of uni/bivariate clauses, default: None
        """
        assert((x, y) in primal.edges())

        if evidence is None:
            evidence = []

        # gather previously received messages
        aggr = [mailboxes[x][z] for z in mailboxes[x] if z != y]
        new_integrand = Message.intersect(aggr, tolerance)


        # compute the new pieces using the x-, y-, x/y-clauses in f + evidence
        xvar = primal.nodes()[x]['var']
        yvar = primal.nodes()[y]['var']

        evidence_x = And([clause for clause in evidence
                          if set(clause.get_free_variables()) == {xvar}])
        evidence_y = And([clause for clause in evidence
                          if set(clause.get_free_variables()) == {yvar}])
        evidence_x_y = And([clause for clause in evidence
                            if
                            set(clause.get_free_variables()) == {xvar, yvar}])

        delta_x = Or([And(LE(Real(l), xvar), LE(xvar, Real(u)))
                      for l, u, _ in new_integrand])
        delta_x = simplify(And(delta_x, evidence_x))

        delta_y = primal.get_univariate_formula(y)
        delta_y = simplify(And(delta_y, evidence_y))

        # clauses containing exclusively x and y
        delta_x_y = primal.get_bivariate_formula(x, y)
        delta_x_y = simplify(And(delta_x_y, evidence_x_y))

        cpts = find_edge_critical_points(delta_x, delta_y, delta_x_y)
        cpts = sorted(list(set([float(c) for c in cpts])))
        new_msg = []
        for lc, uc in zip(cpts, cpts[1:]):
            f = And(delta_x, delta_x_y, delta_y,
                    LT(Real(lc), yvar), LT(yvar, Real(uc)))

            # we shouldn't check for the smt solver at each iteration
            #if self.smt_solver and not is_sat(f, solver_name=self.smt_solver):
            if not is_sat(f, solver_name=smt_solver):
                continue
            pwintegral = [(ls, us, new_integrand[ip][2])
                          for ls, us, ip in find_symbolic_bounds(new_integrand,
                                                                 xvar,
                                                                 (uc + lc) / 2,
                                                                 delta_x_y)]
            x_y_potentials = primal.edges()[(x, y)]['potentials']
            if len(x_y_potentials) > 0:
                subs = {yvar: Real((uc + lc) / 2)}
                potential_msgs = Message.potentials_to_messages(
                    x_y_potentials, xvar, subs
                )
                num_pwintegral = [(simplify(substitute(l, subs)),
                                   simplify(substitute(u, subs)),
                                   p)
                                  for l, u, p in pwintegral]
                num_pwintegral = Message.intersect(
                    potential_msgs + [num_pwintegral],
                    tolerance)
                bds = dict()
                for l, u, _ in pwintegral:
                    l_num = simplify(substitute(l, subs))
                    bds[l_num.constant_value()] = l
                    u_num = simplify(substitute(u, subs))
                    bds[u_num.constant_value()] = u
                for lit, _ in x_y_potentials:
                    k, _, _ = literal_to_bounds(lit)[xvar]
                    k_num = simplify(substitute(k, subs))
                    bds[k_num.constant_value()] = k
                pwintegral = [
                    (bds[l], bds[u], p) for l, u, p in num_pwintegral
                ]
            P, cache_hit = Message.integrate(cache, pwintegral, x, y)
            new_msg.append((lc, uc, P))

        return new_msg, cache_hit



if __name__ == '__main__':
    from pysmt.shortcuts import *
    from sys import argv
    from wmipa import WMI


    x = Symbol("x", REAL)
    y = Symbol("y", REAL)
    z = Symbol("z", REAL)


    f = And(LE(Real(0), x), LE(x, Real(1)),
            LE(Real(0), y), LE(y, Real(2)),
            LE(Real(-100), z), LE(z, Real(+100)),
            Or(LE(x, z), LE(Real(1/2),x)),
            LE(z, y))

    w = Times(Ite(LE(x, z), Plus(x,z), Real(1)),
              Ite(LE(y, Real(1/2)), Times(y,y), Real(1)))

    queries = [LE(x, Real(3/2)), LE(y, Real(3/2)), LE(x, z)]
    
    mpwmi = MPWMI3(f, w, n_processes=3)

    
    Z_mp, pq_mp = mpwmi.compute_volumes(queries=queries, cache=bool(int(argv[1])))

    wmipa = WMI(f, w)
    Z_pa, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)
    print("==================================================")
    print(f"Z\t\t\t\t{Z_mp}\t{Z_pa}")
    for i, q in enumerate(queries):
        pq_pa, _ = wmipa.computeWMI(q, mode=WMI.MODE_PA)
        print(f"{q}\t\t\t{pq_mp[i]}\t{pq_pa}")
