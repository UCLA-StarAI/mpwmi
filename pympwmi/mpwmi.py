import logging
from functools import reduce
import networkx as nx
from networkx.algorithms.components import connected_components
import numpy as np
from pysmt.fnode import FNode
from pysmt.shortcuts import And, LE, LT, Not, Or, Real, is_sat

from pympwmi import logger
from pympwmi.primal import PrimalGraph
from pympwmi.utils import domains_to_intervals, find_edge_critical_points, flip_negated_literals_cnf, \
    find_symbolic_bounds, parse_univariate_literal, to_sympy, \
    literal_to_bounds, weight_to_lit_potentials, ordered_variables, symSymbol, \
    SMT_SOLVER, cache_key2
from sympy import integrate as symbolic_integral, Poly
from sympy import sympify
from sympy.core.mul import Mul as symbolic_mul
from sympy.core.symbol import Symbol as symvar


class MPWMI:
    """
    A class that implements the Message Passing Model Integration exact
    inference algorithm. Works with SMT-LRA formula in CNF that have a
    forest-shaped primal graph, raises NotImplementedError otherwise.

    The weight expression has to be:
        Times(Ite(lit_1, w_1, Real(1)), ..., Ite(lit_n, w_n, Real(1)))

    otherwise, ValueError is raised.

    Attributes
    ----------
    cache : dict
        Cache for integrals, or None
    components : list
        The list of connected components (sets of nodes) in 'primal'
    marginals : dict
        Dictionary storing the symbolic marginals of each variable given evidence
    potentials : dict
        Mapping variables to the list of potentials associated with them
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

    def __init__(self, formula, weight, smt_solver=SMT_SOLVER, rand_gen=None, tolerance=1.49e-8):
        """
        Parameters
        ----------
        formula : pysmt.FNode instance
            The input formula representing the support of the distribution
        weight : pysmt.FNode instance
            Polynomial potentials attached to literal values
        rand_gen : np.random.RandomState instance, optional
            The random number generator (default: RandomState(pympwmi.RAND_SEED))
        """

        if rand_gen is None:
            from pympwmi.utils import RAND_SEED
            rand_gen = np.random.RandomState(RAND_SEED)

        self.rand_gen = rand_gen
        self.smt_solver = smt_solver
        self.tolerance = tolerance
        self.potentials = weight_to_lit_potentials(weight)

        # flip negated literals in potentials (TODO: remove this)

        def flip_potential_pair(lw):
            return (flip_negated_literals_cnf(lw[0]), lw[1])

        for vs, vs_potentials in self.potentials.items():
            self.potentials[vs] = list(map(flip_potential_pair, vs_potentials))

        # bivariate conditions are added to formula ###############################
        # TODO kinda of hack, refactor asap.
        formula_atoms = formula.get_atoms()
        conditions = []
        for vs in self.potentials:
            if len(vs) == 2:
                for lit, _ in self.potentials[vs]:
                    if lit not in formula_atoms:
                        conditions.append(Or(lit, Not(lit)))
        formula = And(list(formula.args()) + conditions)
        logging.debug("\tFinished preprocessing formula")
        ############################################################################

        self.primal = PrimalGraph(formula)
        if not nx.is_forest(self.primal.G):
            raise NotImplementedError("MP requires a forest-shaped primal graph")
        logging.debug("\tCreated primal graph")

        assert (all(
            [e in self.primal.edges() for e in self.potentials if len(e) == 2]))

        self.components = list(connected_components(self.primal.G))
        self.marginals = dict()
        self.cache = None
        self.cache_hit = None

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
            Uni/bivariate literals
        evidence : iterable of pysmt.FNode instances (optional)
            Uni/bivariate clauses, default: None
        cache : bool (optional)
            If True, integrals are cached, default: False
        """

        if queries is None:
            queries = []
        else:
            queries = [flip_negated_literals_cnf(q) for q in queries]

        if cache is True and self.cache is None:
            self.cache = dict()
            self.cache_hit = [0, 0]

        else:
            self.cache = None

        # send around messages, possibly accounting for 'evidence'
        self._compute_marginals(evidence=evidence)
        logging.debug("\tMarginals computed")

        # compute the partition function as the product of the marginals of any node
        # for each connected component in the primal graph
        Z_components = []
        for c, comp_vars in enumerate(self.components):
            x = list(comp_vars)[0]
            full_marginal = self._get_full_marginal(x)
            logging.debug(f"\t\t got full marginals for component {c}")
            comp_Z = self.piecewise_symbolic_integral(full_marginal, x)
            logging.debug(f"\t\t Z_c computed by simbolic integration")

            Z_components.append(comp_Z)
        logging.debug("\tCollecting partial results from components: done")

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
                q_marginal = self._get_msgs_intersection(
                    [self._get_full_marginal(x), q_msg]
                )

                q_vol = self.piecewise_symbolic_integral(q_marginal, x)

                # account for the volume of unconnected variables
                for i, comp_vars in enumerate(self.components):
                    if x not in comp_vars:
                        q_vol *= Z_components[i]

                query_volumes.append(q_vol)

            elif len(q_vars) == 2:

                # bivariate query
                y = q_vars[1].symbol_name()
                if (x, y) not in self.primal.edges():
                    # TODO: a method for getting the auxiliary variable name:
                    # aux(x:str, y:str) -> str should be offered by PrimalGraph
                    xc = f"{x}_{y}"
                    yc = f"{y}_{x}"
                    if (xc, y) in self.primal.edges():
                        x = xc
                    elif (x, yc) in self.primal.edges():
                        y = yc
                    else:
                        raise NotImplementedError("Can't answer this query")

                # creates a new message using the query 'q' as evidence
                q_marginal = self._compute_message(x, y, evidence=[q])
                q_marginal = self._get_msgs_intersection([q_marginal] +
                                                          [self.marginals[y][z]
                                                           for z in
                                                           self.marginals[y]
                                                           if z != x])

                if (y,) in self.potentials:
                    potential_msgs = self._parse_potentials(
                        self.potentials[(y,)], self.primal.nodes()[y]['var']
                    )
                    q_marginal = self._get_msgs_intersection(
                        potential_msgs + [q_marginal]
                    )

                q_vol = self.piecewise_symbolic_integral(q_marginal, y)

                # account for the volume of unconnected variables
                for i, comp_vars in enumerate(self.components):
                    if x not in comp_vars:
                        q_vol *= Z_components[i]

                query_volumes.append(q_vol)

            else:
                raise NotImplementedError(
                    "Queries of ariety > 2 aren't supported")

        Z = 1.0
        for Z_comp in Z_components:
            Z *= Z_comp

        if self.cache_hit is not None:
            # TODO: check if cache_hit index should be True or False
            logging.debug("HITS: {}/{} (ratio {})".format(self.cache_hit[True],
                                                          sum(self.cache_hit),
                                                          self.cache_hit[True] /
                                                          sum(self.cache_hit)))

        Z = float(Z.as_expr())
        query_volumes = [float(qv.as_expr()) for qv in query_volumes]
        return Z, query_volumes

    def _compute_marginals(self, evidence=None):
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
        if evidence is None:
            evidence = []

        # initialize marginals and the mailbox of each node
        for n in self.primal.nodes():
            self.marginals[n] = dict()

        # get a random directed forest from the primal graph
        # (nodes having at most in-degree 1)
        topdown = nx.DiGraph()
        left = set(self.primal.nodes())
        while len(left) > 0:
            root = self.rand_gen.choice(list(left))
            newtree = nx.bfs_tree(self.primal.G, root)
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
                self._send_message(n, parent, evidence=evidence)

        logging.debug("\t\tBottom-up pass done")

        # top-down pass
        exec_order.reverse()
        for n in exec_order:
            for child in topdown.neighbors(n):
                self._send_message(n, child, evidence=evidence)
        logging.debug("\t\tTop-down pass done")

    def _basecase_msg(self, x):
        """
        Computes the piecewise integral for a leaf node.

        Parameters
        ----------
        x : str
            Name of the variable in the primal graph
        """

        assert(self.primal.G.degree[x] <= 1)
        intervals = domains_to_intervals(self.primal.get_univariate_formula(x))
        one = Poly(1, symvar(x), domain="QQ")
        return list(map(lambda i: (i[0], i[1], one), intervals))

    def _compute_message(self, x, y, evidence=None):
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
        assert((x, y) in self.primal.edges())

        # gather previously received messages
        if self.primal.G.degree[x] == 1:
            new_integrand = self._basecase_msg(x)
        else:
            # aggregate msgs not coming from the recipient
            aggr = [self.marginals[x][z] for z in self.marginals[x] if z != y]
            new_integrand = self._get_msgs_intersection(aggr)

        # account for potentials associated with this variable

        if (x,) in self.potentials:
            potential_msgs = self._parse_potentials(
                self.potentials[(x,)], self.primal.nodes()[x]['var']
            )
            new_integrand = self._get_msgs_intersection(
                potential_msgs + [new_integrand]
            )

        if evidence is None:
            evidence = []

        # compute the new pieces using the x-, y-, x/y-clauses in f + evidence

        xvar = self.primal.nodes()[x]['var']
        yvar = self.primal.nodes()[y]['var']

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

        delta_y = self.primal.get_univariate_formula(y)
        delta_y = simplify(And(delta_y, evidence_y))

        # clauses containing exclusively x and y
        delta_x_y = self.primal.get_bivariate_formula(x, y)
        delta_x_y = simplify(And(delta_x_y, evidence_x_y))

        cpts = find_edge_critical_points(delta_x, delta_y, delta_x_y)
        cpts = sorted(list(set([float(c) for c in cpts])))
        new_msg = []
        for lc, uc in zip(cpts, cpts[1:]):
            f = And(delta_x, delta_x_y, delta_y,
                    LT(Real(lc), yvar), LT(yvar, Real(uc)))

            if self.smt_solver and not is_sat(f, solver_name=self.smt_solver):
                continue
            pwintegral = [(ls, us, new_integrand[ip][2])
                          for ls, us, ip in find_symbolic_bounds(new_integrand,
                                                                 xvar,
                                                                 (uc + lc) / 2,
                                                                 delta_x_y)]
            if tuple(sorted([x, y])) in self.potentials:
                subs = {yvar: Real((uc + lc) / 2)}
                potential_msgs = MPWMI._parse_potentials(
                    self.potentials[tuple(sorted([x, y]))], xvar, subs
                )
                num_pwintegral = [(simplify(substitute(l, subs)),
                                   simplify(substitute(u, subs)),
                                   p)
                                  for l, u, p in pwintegral]
                num_pwintegral = self._get_msgs_intersection(
                    potential_msgs + [num_pwintegral]
                )
                bds = dict()
                for l, u, _ in pwintegral:
                    l_num = simplify(substitute(l, subs))
                    bds[l_num.constant_value()] = l
                    u_num = simplify(substitute(u, subs))
                    bds[u_num.constant_value()] = u
                for lit, _ in self.potentials[tuple(sorted([x, y]))]:
                    k, _, _ = literal_to_bounds(lit)[xvar]
                    k_num = simplify(substitute(k, subs))
                    bds[k_num.constant_value()] = k
                pwintegral = [
                    (bds[l], bds[u], p) for l, u, p in num_pwintegral
                ]
            P = self.piecewise_symbolic_integral(pwintegral, x, y)
            new_msg.append((lc, uc, P))

        return new_msg

    def _send_message(self, x, y, evidence=None):
        """
        Sends a message from node 'x' to node 'y', possibly accounting for
        potentials and evidence.

        Parameters
        ----------
        x : str
        y : str
            The nodes on the primal graph having an edge connecting them.
        evidence : list (optional)
            List of uni/bivariate clauses, default: None
        """
        self.marginals[y][x] = self._compute_message(x, y, evidence=evidence)

    def piecewise_symbolic_integral(self, integrand, x, y=None):
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

        res = 0
        for l, u, p in integrand:
            symx = symvar(x)
            symy = symvar(y) if y else symvar("aux_y")

            syml = Poly(to_sympy(l), symy, domain="QQ")
            symu = Poly(to_sympy(u), symy, domain="QQ")
            if type(p) != Poly:
                symp = Poly(to_sympy(p), symx, domain="QQ")
            else:
                symp = Poly(p.as_expr(), symx, domain=f"QQ[{symy}]") if y else p

            if self.cache is not None:
                # for bounds (l, u)
                bds_ks = [cache_key2(syml)[0], cache_key2(symu)[0]]
                bds = [syml.as_expr(), symu.as_expr()]
                p_ks = cache_key2(symp)
                trm_ks = [(bds_ks[0], p_ks[0]), (bds_ks[1], p_ks[0])]
                if (bds_ks[0], bds_ks[1], p_ks[0]) in self.cache:
                    self.cache_hit[True] += 1
                    symintegral = self.cache[(bds_ks[0], bds_ks[1], p_ks[0])]
                    symintegral = symintegral.subs(symintegral.gens[0], symy)
                else:
                    terms = []
                    for tk in trm_ks:  # retrieve terms
                        if tk in self.cache:
                            trm = self.cache[tk]
                            trm = trm.subs(trm.gens[0], symy)
                            terms.append(trm)
                        else:
                            terms.append(None)

                    if None not in terms:
                        self.cache_hit[True] += 1
                    else:
                        if p_ks[0] in self.cache:  # retrieve antiderivative
                            antidrv = self.cache[p_ks[0]]
                            antidrv_expr = antidrv.as_expr().subs(antidrv.gens[0], symx)
                            antidrv = Poly(antidrv_expr, symx,
                                           domain=f"QQ[{symy}]") if y \
                                else Poly(antidrv_expr, symx, domain="QQ")
                        else:
                            self.cache_hit[False] += 1
                            antidrv = symp.integrate(symx)
                            for k in p_ks:  # cache antiderivative
                                self.cache[k] = antidrv

                        for i in range(len(terms)):
                            if terms[i] is not None:
                                continue
                            terms[i] = antidrv.eval({symx: bds[i]})
                            terms[i] = Poly(terms[i].as_expr(), symy, domain="QQ")
                            for k in p_ks:  # cache terms
                                self.cache[(bds_ks[i], k)] = terms[i]

                    symintegral = terms[1] - terms[0]
                    for k in p_ks:  # cache integration
                        self.cache[(bds_ks[0], bds_ks[1], k)] = symintegral

            else:
                symintegral = symbolic_integral(symp, (symx, syml, symu))
                logging.debug(f"\t\t\t\t\t sym integral done for symx: {symx} symp: {symp}")

            res += symintegral
        return res

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
        one = Poly(1, symx, domain="QQ")
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
                assert(factors[msgid] is None)
                factors[msgid] = sympify(f)
                l = x
                n_factors += 1

        assert(n_factors == 0)
        assert(factors == [None] * n_msgs)

        return intersection

    def _get_full_marginal(self, x):
        incoming = list(self.marginals[x].values())
        if len(incoming) > 0:
            full_marginal = self._get_msgs_intersection(incoming)
        else:
            full_marginal = self._basecase_msg(x)

        if (x,) in self.potentials:
            potential_msgs = self._parse_potentials(
                self.potentials[(x,)], self.primal.nodes()[x]['var']
            )
            full_marginal = self._get_msgs_intersection(
                potential_msgs + [full_marginal]
            )

        return full_marginal

    # TODO: document and refactor this
    def _get_msgs_intersection(self, msgs: list):
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
                if abs(end - start) < self.tolerance:  # TODO: should i put it here?
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
        intersection = MPWMI._line_sweep(points, len(msgs))
        intersection = self._filter_msgs(intersection)
        return intersection

    def _filter_msgs(self, msgs: list):
        f_msgs = []
        for start, end, integrand in msgs:
            if abs(end - start) < self.tolerance:
                continue
            f_msgs.append((start, end, integrand))
        return f_msgs

    @staticmethod
    def cache_key(l, u, poly, x):
        ls = sympify(l)
        us = sympify(u)
        polys = sympify(poly)
        ord_vars = ordered_variables(polys)
        renaming = {v: symvar(f"aux_{i}") for i, v in enumerate(ord_vars)}

        return (ls.subs(renaming),
                us.subs(renaming),
                polys.subs(renaming),
                x.subs(renaming))


if __name__ == '__main__':
    from pysmt.shortcuts import *
    from wmipa import WMI
    from pympwmi import set_logger_debug

    w = Symbol("w", REAL)
    x = Symbol("x", REAL)
    y = Symbol("y", REAL)
    z = Symbol("z", REAL)

    f = And(LE(Real(0), w), LE(w, Real(1)),
            LE(Real(0), x), LE(x, Real(1)),
            LE(Real(0), y), LE(y, Real(1)),
            LE(Real(0), z), LE(z, Real(1)),
            LE(Plus(w, x), Real(1)),
            LE(Plus(x, y), Real(1)),
            LE(Plus(y, z), Real(1)))

    queries = [LE(w, Real(0.5)),
               LE(x, Real(0.5)),
               LE(y, Real(0.5)),
               LE(z, Real(0.5)),
               LE(Plus(w, x), Real(0.5)),
               LE(Plus(x, y), Real(0.5)),
               LE(Plus(y, z), Real(0.5))]
    w = Ite(LE(x, y), Real(2), Real(1))
    mpwmi = MPWMI(f, w)

    Z_mp, pq_mp = mpwmi.compute_volumes(queries=queries, cache=True)

    wmipa = WMI(f, w)
    Z_pa, _ = wmipa.computeWMI(Bool(True), mode=WMI.MODE_PA)
    print("==================================================")
    print(f"Z\t\t\t{Z_mp}\t{Z_pa}")
    for i, q in enumerate(queries):
        pq_pa, _ = wmipa.computeWMI(q, mode=WMI.MODE_PA)
        print(f"{q}\t\t\t{pq_mp[i]}\t{pq_pa}")
