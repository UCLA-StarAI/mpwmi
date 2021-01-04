
from functools import reduce
from fractions import Fraction
from multiprocessing import Pool, Manager
import networkx as nx
from networkx.algorithms.components import connected_components
import numpy as np
from pysmt.shortcuts import And, LE, LT, Not, Or, Real, is_sat
from pysmt.environment import get_env, push_env

from pympwmi.primal import PrimalGraph
from pympwmi.utils import *
from sympy import integrate as symbolic_integral, Poly
from sympy import sympify
from sympy.core.mul import Mul as symbolic_mul
from sympy.core.symbol import Symbol as symvar
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement


class MP2WMI:
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
    marginals : dict
        Dictionary storing the symbolic marginals of each variable given evidence
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

        self.marginals = dict()
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


        # compute the partition function as the product of the marginals of any node
        # for each connected component in the primal graph
        components = list(connected_components(self.primal.G))
        subproblems = []
        pysmt_env = get_env()
        for comp_vars in components:
            subprimal = PrimalGraph.from_graph(self.primal.G.subgraph(comp_vars))
            subvars = {subprimal.nodes()[n]['var'] for n in subprimal.nodes()}
            
            submarginals = {k : v for k, v in self.marginals.items() if k in comp_vars}

            """
            if self.cache is not None:
                subcache = self.cache
            else:
                subcache = None
            """

            if evidence is None:
                subevidence = None
            else:
                subevidence = [e for e in evidence
                               if set(e.get_free_symbols()).issubset(subvars)]
            subproblems.append((subprimal, submarginals, self.smt_solver, self.cache,
                                self.tolerance, self.rand_gen, pysmt_env, subevidence))

        with Pool(processes=self.n_processes) as pool:
            results = pool.starmap(MP2WMI._compute_marginals, subproblems)


        for submarginals, ch in results:
            self.marginals.update(submarginals)
            if self.cache is not None:
                self.cache_hit[True] += ch[True]
                self.cache_hit[False] += ch[False]
        
        Z_components = []
        for comp_vars in components:
            x = list(comp_vars)[0]
            full_marginal = MP2WMI._get_full_marginal(self.primal, self.marginals,
                                                      self.tolerance, x)
            comp_Z, ch = MP2WMI._piecewise_symbolic_integral(self.cache,
                                                             full_marginal, x)
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

                # univariate query
                l, u = domains_to_intervals(q)[0]
                q_msg = [(l, u, 1)]

                # intersecting with the node symbolic marginal
                q_marginal = MP2WMI._get_msgs_intersection(
                    [MP2WMI._get_full_marginal(self.primal, self.marginals, self.tolerance,x), q_msg],
                    self.tolerance)
                q_vol, ch = MP2WMI._piecewise_symbolic_integral(self.cache, q_marginal, x)
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
                q_marginal, ch = MP2WMI._compute_message(self.primal, self.marginals,
                                                         self.smt_solver, self.cache,
                                                         self.tolerance, x, y, evidence=[q])

                if self.cache is not None:
                    self.cache_hit[True] += ch[True]
                    self.cache_hit[False] += ch[False]

                marg_not_x =  [self.marginals[y][z] for z in self.marginals[y] if z != x]
                q_marginal = MP2WMI._get_msgs_intersection([q_marginal] + marg_not_x,
                                                           self.tolerance)

                y_potentials = self.primal.nodes()[y]['potentials']
                if len(y_potentials) > 0:
                    potential_msgs = MP2WMI._parse_potentials(
                        y_potentials, self.primal.nodes()[y]['var']
                    )
                    q_marginal = self._get_msgs_intersection(
                        potential_msgs + [q_marginal], self.tolerance)

                q_vol, ch = MP2WMI._piecewise_symbolic_integral(self.cache, q_marginal, y)

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
    def _compute_marginals(primal, marginals, smt_solver, cache, tolerance,
                           rand_gen, pysmt_env, evidence=None):
        """
        Computes the symbolic piecewise integrals representing the marginal of
        each node. Performs a message passing step:
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
        for n in primal.nodes():
            marginals[n] = dict()

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
        print("EXEC-ORDER", exec_order)
        for n in exec_order:
            parents = list(bottomup.neighbors(n))
            assert (len(parents) < 2), "this shouldn't happen"
            if len(parents) == 1:
                parent = parents[0]
                marginals[parent][n], ch = MP2WMI._compute_message(primal, marginals,
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
                marginals[child][n], ch = MP2WMI._compute_message(primal, marginals, smt_solver,
                                                                  cache, tolerance, n, child,
                                                                  evidence=evidence)
                if ch is not None:
                    cache_hit[True] += ch[True]
                    cache_hit[False] += ch[False]

        return marginals, cache_hit

    @staticmethod
    def _basecase_msg(primal, x):
        """
        Computes the piecewise integral for a leaf node.

        Parameters
        ----------
        x : str
            Name of the variable in the primal graph
        """

        assert(primal.G.degree[x] <= 1)
        intervals = domains_to_intervals(primal.get_univariate_formula(x))
        one = Poly(1, symvar(x), symvar("aux_y"), domain="QQ")
        return list(map(lambda i: (i[0], i[1], one), intervals))

    @staticmethod
    def _compute_message(primal, marginals, smt_solver, cache, tolerance,
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
            new_integrand = MP2WMI._basecase_msg(primal, x) # leaf
        else:
            # aggregate msgs not coming from the recipient
            aggr = [marginals[x][z] for z in marginals[x] if z != y]
            new_integrand = MP2WMI._get_msgs_intersection(aggr, tolerance)

        # account for potentials associated with this variable
        x_potentials = primal.nodes()[x]['potentials']

        if len(x_potentials) > 0:
            potential_msgs = MP2WMI._parse_potentials(
                x_potentials, primal.nodes()[x]['var']
            )
            new_integrand = MP2WMI._get_msgs_intersection(
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
                potential_msgs = MP2WMI._parse_potentials(
                    x_y_potentials, xvar, subs
                )
                num_pwintegral = [(simplify(substitute(l, subs)),
                                   simplify(substitute(u, subs)),
                                   p)
                                  for l, u, p in pwintegral]
                num_pwintegral = MP2WMI._get_msgs_intersection(
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
            P, cache_hit = MP2WMI._piecewise_symbolic_integral(cache, pwintegral, x, y)
            new_msg.append((lc, uc, P))

        return new_msg, cache_hit


    @staticmethod
    def _piecewise_symbolic_integral(cache, integrand, x, y=None):
        """
        Computes the symbolic integral of 'x' of a piecewise polynomial 'integrand'.
        The result might be a sympy expression or a numerical value.

        Parameters
        ----------
        integrand : list
            A list of (lower bound, upper bound, polynomial)
        x : object
            A string/sympy expression representing the integration variable
        """
        cache_hit = [0, 0] if (cache is not None) else None

        res = 0
        for l, u, p in integrand:
            symx = symvar(x)
            symy = symvar(y) if y else symvar("aux_y")
            syml = Poly(to_sympy(l), symy, domain="QQ")            
            symu = Poly(to_sympy(u), symy, domain="QQ")

            if type(p) != Poly:
                symp = Poly(to_sympy(p), symx, domain="QQ")
            else:
                symp = Poly(p.as_expr(), symx, symy, domain="QQ")

            #print("integrating", symp.as_expr(), f"in d{symx} with bounds", [syml.as_expr(), symu.as_expr()])
            if cache is not None:  # for cache = True
                """ hierarchical cache, where we cache:
                 - the anti-derivatives for integrands, retrieved by:
                       (None, None, integrand key)
                 - the partial integration term, retrieved by:
                       (lower bound key, None, integrand key)
                       (None, upper bound key, integrand key)
                 - the whole integration, retrieved by:
                       (lower bound key, upper bound key, integrand key)
                """
                # cache keys for bounds
                k_lower = MP2WMI.sympy_to_tuple(syml)
                k_upper = MP2WMI.sympy_to_tuple(symu)
                k_poly = MP2WMI.sympy_to_tuple(symp)  # cache key for integrand polynomial
                k_full = (k_lower, k_upper, k_poly)

                #print("========= KEYS =========")
                #print("lower:", syml.as_expr(), "-->", k_lower)
                #print("upper:", symu.as_expr(), "-->", k_upper)
                #print("poly:", symp.as_expr(), "-->", k_poly)
                #print("========================")
                if k_full in cache:
                    # retrieve the whole integration                    
                    cache_hit[True] += 1
                    symintegral = MP2WMI.tuple_to_sympy(cache[k_full], symx, symy)
                    symintegral = symintegral.subs(symintegral.gens[0], symy)

                else:
                    # retrieve partial integration terms
                    terms = [None, None]
                    k_part_l = (k_lower, k_poly)
                    k_part_u = (k_upper, k_poly)
                    if k_part_l in cache:
                        partial_l = MP2WMI.tuple_to_sympy(cache[k_part_l], symx, symy)
                        terms[0] = partial_l.subs(partial_l.gens[0], symy)

                    if k_part_u in cache:
                        partial_u = MP2WMI.tuple_to_sympy(cache[k_part_u], symx, symy)
                        terms[1] = partial_u.subs(partial_u.gens[0], symy)

                    if None not in terms:
                        cache_hit[True] += 1
                    else:
                        # retrieve anti-derivative
                        k_anti = (k_poly,)
                        if k_anti in cache:                            
                            cache_hit[True] += 1
                            antidrv = MP2WMI.tuple_to_sympy(cache[k_anti], symx, symy)

                        else:
                            cache_hit[False] += 1
                            antidrv = symp.integrate(symx)
                            cache[k_anti] = MP2WMI.sympy_to_tuple(antidrv)

                        # cache partial integration terms
                        if terms[0] is None:
                            terms[0] = Poly(antidrv.as_expr(), symx,
                                            domain=f'QQ[{symy}]').eval({symx: syml.as_expr()})
                            terms[0] = Poly(terms[0].as_expr(), symx, symy, domain="QQ")
                            cache[k_part_l] = MP2WMI.sympy_to_tuple(terms[0])

                        if terms[1] is None:
                            terms[1] = Poly(antidrv.as_expr(), symx,
                                            domain=f'QQ[{symy}]').eval({symx: symu.as_expr()})
                            terms[1] = Poly(terms[1].as_expr(), symx, symy, domain="QQ")
                            cache[k_part_u] = MP2WMI.sympy_to_tuple(terms[1])

                    #print("subs: (", terms[1].as_expr(), ") - (", terms[0].as_expr(), ")")
                    symintegral = terms[1] - terms[0]
                    if not isinstance(symintegral, Poly):
                        symintegral = Poly(symintegral, symx, symy, domain='QQ')
                    cache[k_full] = MP2WMI.sympy_to_tuple(symintegral)

            else:  # for cache = False
                antidrv = symp.integrate(symx)
                lower = Poly(antidrv.as_expr(), symx,
                             domain=f'QQ[{symy}]').eval({symx: syml.as_expr()})
                lower = Poly(lower.as_expr(), symx, symy, domain="QQ")
                upper = Poly(antidrv.as_expr(), symx,
                             domain=f'QQ[{symy}]').eval({symx: symu.as_expr()})
                upper = Poly(upper.as_expr(), symx, symy, domain="QQ")
                symintegral = upper - lower

            res += symintegral
            #print("integral:", symintegral.as_expr())
            #print()

        #print("RESULT:", res)
        #print("**************************************************")
        return res, cache_hit

    # TODO: document and refactor this
    @staticmethod
    def _account_for_potential(piecewise_integral, literal, f):
        """
        Accounts for a potential 'f' associated with the univariate
        'literal', returning a modified 'piecewise_integral'.

        Parameters
        ----------
        piecewise_integral : list
            The input pw integral [(lower, upper, polynomial)] over x
        literal : pysmt.FNode
            The univariate literal on x
        f : pysmt.FNode instance
            The polynomial potential associated to 'literal' being True
        """

        assert (is_literal(literal))

        var, k, is_lower, k_included = parse_univariate_literal(literal)
        new_msg = []

        fsym = to_sympy(f)

        for piece in piecewise_integral:
            l, u, p = piece
            assert (l != u), f"degenerate interval: [{l}, {u}]!!!!!"
            if (is_lower and k >= u) or (not is_lower and k <= l):
                # no intersection
                new_msg.append(piece)

            else:
                if k > l and k < u:
                    # k in ]l,u[
                    # the piece must be split
                    if is_lower:
                        new_msg.append((l, k, p))
                        new_msg.append((k, u, symbolic_mul(p, fsym)))

                    else:
                        new_msg.append((l, k, symbolic_mul(p, fsym)))
                        new_msg.append((k, u, p))

                else:
                    new_msg.append((l, u, symbolic_mul(p, fsym)))

        return new_msg

    @staticmethod
    def _parse_potentials(potentials, xvar, subs=None):
        msgs = []
        symx = symvar(str(xvar))
        one = Poly(1, symx, symvar("aux_y"), domain="QQ")
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

    @staticmethod
    def _account_for_edge_potential(piecewise_integral,
                                    literal,
                                    f,
                                    xvar,
                                    yvar,
                                    yval):
        assert (is_literal(literal))

        ksym, is_lower, k_included = literal_to_bounds(literal)[xvar]
        fsym = to_sympy(f)
        k = simplify(substitute(ksym, {yvar: Real(yval)}))
        k = k.constant_value()

        new_msg = []
        for piece in piecewise_integral:
            lsym, usym, p = piece
            p = sympify(p)
            l = simplify(substitute(lsym, {yvar: Real(yval)}))
            l = l.constant_value()
            u = simplify(substitute(usym, {yvar: Real(yval)}))
            u = u.constant_value()
            assert (l != u), f"degenerate interval: [{l}, {u}]!!!!!"
            if (is_lower and k >= u) or (not is_lower and k <= l):
                # no intersection
                new_msg.append(piece)

            else:
                if k > l and k < u:
                    # k in ]l,u[
                    # the piece must be split
                    if is_lower:
                        new_msg.append((lsym, ksym, p))
                        new_msg.append((ksym, usym, symbolic_mul(p, fsym)))

                    else:
                        new_msg.append((lsym, ksym, symbolic_mul(p, fsym)))
                        new_msg.append((ksym, usym, p))

                else:
                    new_msg.append((lsym, usym, symbolic_mul(p, fsym)))

        return new_msg

    # TODO: document and refactor this
    @staticmethod
    def _line_sweep(points: list, n_msgs: int):
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
                    integrand = reduce(lambda x, y: x * y,
                                       factors[1:],
                                       factors[0])
                    intersection.append([l, x, integrand])

                factors[msgid] = None
                l = None
                n_factors -= 1

            else:
                assert(factors[msgid] is None), f"fuck {factors[msgid]}"
                factors[msgid] = sympify(f)
                l = x
                n_factors += 1

        assert(n_factors == 0)
        assert(factors == [None] * n_msgs)

        return intersection

    @staticmethod
    def _get_full_marginal(primal, marginals, tolerance, x):
        incoming = list(marginals[x].values())
        if len(incoming) > 0:
            full_marginal = MP2WMI._get_msgs_intersection(incoming, tolerance)
        else:
            full_marginal = MP2WMI._basecase_msg(primal, x)

        x_potentials = primal.nodes()[x]['potentials']
        if len(x_potentials) > 0:
            potential_msgs = MP2WMI._parse_potentials(
                x_potentials, primal.nodes()[x]['var']
            )
            potential_msgs.append(full_marginal)
            full_marginal = MP2WMI._get_msgs_intersection(potential_msgs, tolerance)

        return full_marginal


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
        intersection = MP2WMI._line_sweep(points, len(msgs))
        intersection = MP2WMI._filter_msgs(intersection, tolerance)
        return intersection

    @staticmethod
    def _filter_msgs(msgs: list, tolerance):
        f_msgs = []
        for start, end, integrand in msgs:
            if abs(end - start) < tolerance:
                continue
            f_msgs.append((start, end, integrand))
        return f_msgs

    @staticmethod
    def sympy_to_tuple(p):
        d = p.as_dict(native=True)
        if d == {}:
            assert (p == 0)
            return (((0,0), 0),) # the Zero tuple

        else:
            l = []
            for k, v in d.items():
                tk = tuple(map(int, k))
                if isinstance(v, PolyElement):
                    tv = int(v.coeff(1))
                elif isinstance(v, FracElement):
                    tv = Fraction(int(v.numer.coeff(1)), int(v.denom.coeff(1)))
                else:
                    tv = v

                l.append((tk, tv))
                
            #return tuple(sorted((tuple(map(int, k)), float(v)) for k, v in d.items()))
            t = tuple(sorted(l))
            return t

    @staticmethod
    def tuple_to_sympy(t, x, y):
        d = dict(t)
        nvars = len(t[0][0])        
        if nvars == 1:
            return Poly.from_dict(d, x, domain='QQ')
        elif nvars == 2:
            return Poly.from_dict(d, x, y, domain='QQ')
        else:
            raise ValueError("Expected univariate or bivariate expression")


if __name__ == '__main__':
    from pysmt.shortcuts import *
    from sys import argv
    from wmipa import WMI


    w = Symbol("w", REAL)
    x = Symbol("x", REAL)
    y = Symbol("y", REAL)
    z = Symbol("z", REAL)
    """

    f = And(LE(Real(0), w), LE(w, Real(1)),
            LE(Real(0), x), LE(x, Real(1)),
            LE(Real(0), y), LE(y, Real(1)),
            LE(Real(0), z), LE(z, Real(1)),
            LE(Plus(w, x), Real(1)),
            LE(Plus(x, y), Real(1)),
            LE(Plus(y, z), Real(1)))

    w = Ite(LE(x, y), Real(2), Real(1))
    queries = [LE(w, Real(0.5)),
               LE(x, Real(0.5)),
               LE(y, Real(0.5)),
               LE(z, Real(0.5)),
               LE(Plus(w, x), Real(0.5)),
               LE(Plus(x, y), Real(0.5)),
               LE(Plus(y, z), Real(0.5))]
    """

    f = And(LE(Real(1), x), LE(x, Real(2)),
            LE(Real(1), y), LE(y, Real(2)),
            LE(Real(1), z), LE(z, Real(2)))

    w = Ite(LE(x, y), Plus(x,y), Real(1))

    queries = [] #[LE(x, Real(3/2)), LE(y, Real(3/2)), LE(x, y)]
    
    mpwmi = MP2WMI(f, w, n_processes=3)

    
    Z_mp, pq_mp = mpwmi.compute_volumes(queries=queries, cache=bool(int(argv[1])))

    wmipa = WMI(f, w)
    Z_pa, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)
    print("==================================================")
    print(f"Z\t\t\t\t{Z_mp}\t{Z_pa}")
    for i, q in enumerate(queries):
        pq_pa, _ = wmipa.computeWMI(q, mode=WMI.MODE_PA)
        print(f"{q}\t\t\t{pq_mp[i]}\t{pq_pa}")
