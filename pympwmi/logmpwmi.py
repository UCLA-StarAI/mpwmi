
from functools import reduce
from fractions import Fraction
from multiprocessing import Pool, Manager
import networkx as nx
from networkx.algorithms.components import connected_components
import numpy as np
from pysmt.shortcuts import And, LE, LT, Not, Or, Real, is_sat
from pysmt.environment import get_env, push_env

from pympwmi.logprimal import LogPrimalGraph
from pympwmi.utils import *


class LogMPWMI:
    """A class that implements the Message Passing Weighted Model
    Integration exact inference algorithm with ***log-linear
    weights***. Works with SMT-LRA formula in CNF that have a
    forest-shaped primal graph, raises NotImplementedError otherwise.

    The weight has to be a list [..(l_i, w_i) ]
    where w_i = k_i*exp(p_i(x_i)) and l_i = (a_i * x_i <= b_i)

    otherwise, ValueError is raised.

    Attributes
    ----------
    cache : dict
        Cache for integrals, or None
    mailboxes : dict
        Dictionary storing the symbolic mailboxes of each variable given evidence
    primal : LogPrimalGraph instance
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
        weight : list(pysmt.FNode, tuple)
            List of log-polynomial potentials k*exp(a1*x1+..+an*xn) attached to literal values.
            Potentials are in the form tuple(float(k), np.array(a1,..an))
        rand_gen : np.random.RandomState instance, optional
            The random number generator (default: RandomState(mpwmi.RAND_SEED))
        """

        if rand_gen is None:
            from pympwmi.utils import RAND_SEED
            rand_gen = np.random.RandomState(RAND_SEED)

        self.rand_gen = rand_gen
        self.smt_solver = smt_solver
        self.tolerance = tolerance

        self.primal = LogPrimalGraph(formula, weight)

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

        if queries is None: # preprocess the negated atoms in the queries, e.g. !(a <= b) -> (a > b)
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
            results = pool.starmap(LogMPWMI._message_passing, subproblems)


        for messages, ch in results:
            self.mailboxes.update(messages)
            if self.cache is not None:
                self.cache_hit[True] += ch[True]
                self.cache_hit[False] += ch[False]
        
        Z_components = []
        for comp_vars in components:
            x = list(comp_vars)[0]
            #full_marginal = LogMPWMI._get_full_marginal(self.primal, self.mailboxes,
            #                                          self.tolerance, x)

            if len(mailboxes[x].values()) > 0:
                msgs = list(mailboxes[x].values())
            else:
                msgs = [LogMPWMI._basecase_msg(primal, x)]

            if len(self.primal.nodes()[x]['potentials']) > 0:
                msgs.append(LogMPWMI._parse_potentials(x_potentials,
                                                       primal.nodes()[x]['var']))

            symbolic_marginal = LogMPWMI._get_msgs_intersection(msgs, tolerance)

            
            ix = self.primal.nodes()[x]['index']
            comp_Z, ch = LogMPWMI._piecewise_symbolic_integral(self.cache,
                                                               symbolic_marginal, ix)
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

                if len(mailboxes[x].values()) > 0:
                    msgs = list(mailboxes[x].values())
                else:
                    msgs = [LogMPWMI._basecase_msg(primal, x)]

                if len(self.primal.nodes()[x]['potentials']) > 0:
                    msgs.append(LogMPWMI._parse_potentials(x_potentials,
                                                           primal.nodes()[x]['var']))

                # univariate query
                l, u = domains_to_intervals(q)[0]
                one = (1.0,
                       np.zeros(2), #np.zeros(len(self.primal.nodes())),
                       np.zeros(2)) #np.zeros(len(self.primal.nodes()))) #1*exp(0)
                msgs.append((l, u, one))
                    
                query_marginal = LogMPWMI._get_msgs_intersection(msgs, self.tolerance)

                ix = self.primal.nodes()[x]['index']
                q_vol, ch = LogMPWMI._piecewise_symbolic_integral(self.cache,
                                                                  query_marginal,
                                                                  ix)
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
                q_marginal, ch = LogMPWMI._compute_message(self.primal, self.mailboxes,
                                                         self.smt_solver, self.cache,
                                                         self.tolerance, x, y, evidence=[q])

                if self.cache is not None:
                    self.cache_hit[True] += ch[True]
                    self.cache_hit[False] += ch[False]

                marg_not_x =  [self.mailboxes[y][z] for z in self.mailboxes[y] if z != x]
                q_marginal = LogMPWMI._get_msgs_intersection([q_marginal] + marg_not_x,
                                                           self.tolerance)

                y_potentials = self.primal.nodes()[y]['potentials']
                if len(y_potentials) > 0:
                    potential_msgs = LogMPWMI._parse_potentials(
                        #self.primal.nvars,
                        y_potentials,
                        self.primal.nodes()[y]['var']
                    )
                    q_marginal = self._get_msgs_intersection(
                        potential_msgs + [q_marginal], self.tolerance)

                iy = self.primal.nodes()[y]['index']
                q_vol, ch = LogMPWMI._piecewise_symbolic_integral(self.cache, q_marginal, iy)

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
        Symbolic message-passing:
        1) The primal forest is randomly directed
        2) Nodes send/receive messages to/from every neighbor
           accounting for potentials and evidence when computing their marginals

        Parameters
        ----------
        evidence : list (optional)
            List of uni/bivariate clauses considered as evidence when computing
            the marginals
        """

        push_env(pysmt_env)

        if cache is not None:
            cache_hit = [0, 0]
        else:
            cache_hit = None
            
        if evidence is None:
            evidence = []

        # initialize marginals and the mailbox of each node
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
            parents = list(bottomup.neighbors(n))
            assert (len(parents) < 2), "this shouldn't happen"
            if len(parents) == 1:
                parent = parents[0]
                mailboxes[parent][n], ch = LogMPWMI._compute_message(primal, mailboxes,
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
                mailboxes[child][n], ch = LogMPWMI._compute_message(primal, mailboxes, smt_solver,
                                                                  cache, tolerance, n, child,
                                                                  evidence=evidence)
                if ch is not None:
                    cache_hit[True] += ch[True]
                    cache_hit[False] += ch[False]

        return mailboxes, cache_hit

    @staticmethod
    def _basecase_msg(primal, x):
        """
        Computes the base message for a leaf node x.

        Parameters
        ----------
        primal : LogPrimalGraph         
        x : str
        """

        assert(primal.G.degree[x] <= 1)
        intervals = domains_to_intervals(primal.get_univariate_formula(x))
        # one = Poly(1, symvar(x), symvar("aux_y"), domain="QQ") # becomes
        one = (1.0,
               np.zeros(2), #np.zeros(len(primal.nodes())),
               np.zeros(2)) #np.zeros(len(primal.nodes()))) #1*exp(0)
        return list(map(lambda i: (i[0], i[1], one), intervals))

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

        # gather previously received messages
        if primal.G.degree[x] == 1:
            new_integrand = LogMPWMI._basecase_msg(primal, x) # leaf
        else:
            # aggregate msgs not coming from the recipient
            aggr = [mailboxes[x][z] for z in mailboxes[x] if z != y]
            new_integrand = LogMPWMI._get_msgs_intersection(aggr, tolerance)

        # account for potentials associated with this variable
        x_potentials = primal.nodes()[x]['potentials']

        if len(x_potentials) > 0:
            potential_msgs = LogMPWMI._parse_potentials(
                #primal.nvars,
                x_potentials,
                primal.nodes()[x]['var']
            )
            new_integrand = LogMPWMI._get_msgs_intersection(
                potential_msgs + [new_integrand], tolerance
            )

        if evidence is None:
            evidence = []

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
                potential_msgs = LogMPWMI._parse_potentials(
                    #primal.nvars,
                    x_y_potentials,
                    xvar,
                    subs
                )
                num_pwintegral = [(simplify(substitute(l, subs)),
                                   simplify(substitute(u, subs)),
                                   p)
                                  for l, u, p in pwintegral]
                num_pwintegral = LogMPWMI._get_msgs_intersection(
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

            ix = primal.nodes()[x]['index']
            iy = primal.nodes()[y]['index']
            P, cache_hit = LogMPWMI._piecewise_symbolic_integral(cache, pwintegral, ix, iy)
            new_msg.append((lc, uc, P))

        return new_msg, cache_hit

    @staticmethod
    def _antiderivative(f, ix):
        return (f[0]/f[1][ix], f[1])

    @staticmethod
    def _substitute(f, ix, iy, p):
        assert(len(p) == 2), f"{p} != ax + b"
        newexp = np.array(f[1])
        newexp[ix] = 0
        newexp[iy] += f[1][ix]*p[0]
        return(f[0]* nep**(f[1][ix]*p[1]), newexp)


    @staticmethod
    def _piecewise_symbolic_integral(cache, integrand, ix, iy=None):
        """
        Computes the symbolic integral of 'x' of a piecewise 'integrand'.

        Parameters
        ----------
        integrand : list
            A list of (lower bound, upper bound, logweight)
        ix : int
            Index of the integration variable
        iy : int
            Index of the recipient variable ????
        """
        cache_hit = [0, 0] if (cache is not None) else None
        res = 0
        for l, u, p in integrand:            
            if cache is not None:  # for cache = True
                raise NotImplementedError
            else:  # for cache = False

                antidrv = LogMPWMI._antiderivative(p, ix)
                lower = LogMPWMI._substitute(antidrv, ix, iy, lcoeff)
                lower = LogMPWMI._substitute(antidrv, ix, iy, lcoeff)
                upper = antidrv.subs(symx, symu)
                symintegral = upper - lower

            res += symintegral
            #print("integral:", symintegral.as_expr())
            #print()

        #print("RESULT:", res)
        #print("**************************************************")
        return res, cache_hit


    @staticmethod
    #def _parse_potentials(nvars, potentials, xvar, subs=None):
    def _parse_potentials(potentials, xvar, subs=None):
        msgs = []
        #one = Poly(1, symx, symvar("aux_y"), domain="QQ") # becomes
        one = (1.0, np.zeros(2), np.zeros(2)) #1*exp(0)
        for lit, f in potentials:
            msg = []
            k, is_lower, _ = literal_to_bounds(lit)[xvar]
            if subs is not None:
                k = simplify(substitute(k, subs))
            k = k.constant_value()
            if is_lower:
                msg.append((float('-inf'), k, one))
                msg.append((k, float('inf'), f))
            else:
                msg.append((float('-inf'), k, f))
                msg.append((k, float('inf'), one))
            msgs.append(msg)
        return msgs


    # TODO: document and refactor this
    @staticmethod
    def _line_sweep(points: list, n_msgs: int):

        print("--------------------------------------------------")
        print("LINE SWEEP msgs:")
        for m in points:
            print(m)
        print("--------------------------------------------------")

        intersection = []  # elements: tuple (start, end, integrand)
        factors = [None] * n_msgs 
        n_factors = 0
        l = None
        for x, is_lower, f, msgid in points:
            if not is_lower:
                assert(factors[msgid] is not None)
                if n_factors == n_msgs:
                    assert(l is not None)
                    assert(l <= x)
                    integrand = reduce(lambda x, y: (x[0] * y[0], x[1] + y[1]),
                                       factors[1:], factors[0])
                    intersection.append([l, x, integrand])

                factors[msgid] = None
                l = None
                n_factors -= 1

            else:
                assert(factors[msgid] is None)
                factors[msgid] = f
                l = x
                n_factors += 1

        assert(n_factors == 0)
        assert(factors == [None] * n_msgs)

        return intersection


    @staticmethod
    def _get_msgs_intersection(msgs: list, tolerance):
        '''
        Given a list of messages, each one encoding a piecewise integral:
            (x, [[lb1(x), ub1(x), f1(x)], ... , [lbn(x), ubn(x), fn(x)]])

        = sum_{i=1}^{n} int_{lbi}^{ubi} fi(x) dx


        '''

        points = []  # element: [point, isright, integrand, index]
        list_num = len(msgs)

        if len(msgs) == 1:
            return msgs[0]

        for msgid, msg in enumerate(msgs):
            for start, end, integrand in msg:
                if type(start) == FNode:
                    start = start.constant_value()
                if type(end) == FNode:
                    end = end.constant_value()
                if abs(end - start) < tolerance:  # TODO: should i put it here?
                    continue
                if float(start) == float(end):
                    print("DEGENERATE INTERVAL HERE!!!!!")
                    print(f"with degenerate interval [{start}, {end}]")
                    continue
                p1 = [start, True, integrand, msgid]
                p2 = [end, False, integrand, msgid]
                points.append(p1)
                points.append(p2)

        points.sort(key=lambda p: (float(p[0]), p[1]), reverse=False)
        intersection = LogMPWMI._line_sweep(points, len(msgs))
        intersection = LogMPWMI._filter_msgs(intersection, tolerance)
        return intersection

    @staticmethod
    def _filter_msgs(msgs: list, tolerance):
        f_msgs = []
        for start, end, integrand in msgs:
            if abs(end - start) < tolerance:
                continue
            f_msgs.append((start, end, integrand))
        return f_msgs

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
            LE(x, z),
            LE(z, y))

    w = Real(1)#Ite(LE(x, y), Plus(x,y), Real(1))

    symw = []#(LE(x, y), (3,[0,0,0])), (LE(x, z), (5,[0,0,0]))]

    queries = [LE(y,x)] 
    
    mpwmi = LogMPWMI(f, symw, n_processes=3)

    
    Z_mp, pq_mp = mpwmi.compute_volumes(queries=queries, cache=bool(int(argv[1])))

    wmipa = WMI(f, w)
    Z_pa, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)
    print("==================================================")
    print(f"Z\t\t\t\t{Z_mp}\t{Z_pa}")
    for i, q in enumerate(queries):
        pq_pa, _ = wmipa.computeWMI(q, mode=WMI.MODE_PA)
        print(f"{q}\t\t\t{pq_mp[i]}\t{pq_pa}")


"""
    @staticmethod
    def _get_full_marginal(primal, mailboxes, tolerance, x):
        incoming = list(mailboxes[x].values())
        if len(incoming) > 0:
            full_marginal = LogMPWMI._get_msgs_intersection(incoming, tolerance)
        else:
            full_marginal = LogMPWMI._basecase_msg(primal, x)

        x_potentials = primal.nodes()[x]['potentials']
        if len(x_potentials) > 0:
            potential_msgs = LogMPWMI._parse_potentials(
                #primal.nvars,
                x_potentials,
                primal.nodes()[x]['var']
            )
            potential_msgs.append(full_marginal)
            full_marginal = LogMPWMI._get_msgs_intersection(potential_msgs, tolerance)

        return full_marginal
"""
