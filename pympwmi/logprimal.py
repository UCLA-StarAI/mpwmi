
import networkx as nx
from pympwmi.utils import flip_negated_literals_cnf, get_boolean_variables, is_literal, to_sympy
from pysmt.shortcuts import *
from pympwmi.sympysmt import sympy2pysmt



MSG_NOT_CNF = "the formula must be in CNF"
MSG_NOT_CLAUSE = "the formula must be a clause"

class LogPrimalGraph:
    """
    A class that implements the primal graph construction from a SMT-LRA 
    'formula' and a 'weight' mapping literals to log-polynomials.

    Raises NotImplementedError if 'formula':
    - is not in CNF
    - contains clauses of ariety > 2

    Raises NotImplementedError if 'weight':
    - does not map literals to log-polynomials
    - contains literals of ariety > 2

    Attributes
    ----------
    G : networkx.Graph instance
        The structure of the primal graph
    nvars : int
        The total number of variables in the original problem

    Methods
    -------
    edges()
        Returns the edges (dependencies) in the primal graph
    nodes()
        Returns the nodes (variables) in the primal graph
    get_univariate_formula(x)
        Returns the conjunction of the univariate clauses on 'x'
    get_bivariate_formula(x, y)
        Returns the conjunction of:
        - the univariate clauses on 'x'
        - the univariate clauses on 'y'
        - the bivariate clauses on 'x' and 'y'
    """

    def __init__(self, formula, weight):
        """
        Parameters
        ----------
        formula : pysmt.FNode instance
            The input formula representing the support of the distribution
        weight : list(pysmt.FNode, sympy.Mul)
            List of log-polynomial potentials attached to literal values.
            Each potential is k*exp(p(x)) where p(x) is linear, i.e.:
                p(x) = a1 * x1 + ... + an * xn

            Each weight is represented as:
                tuple(float(k), numpy.array([a1,..,an]))
            
        """

        if formula is not None and weight is not None:

            variables = set(formula.get_free_variables())
            self.nvars = len(variables)

            # TODO: remove *flipping negated literals*
            formula = flip_negated_literals_cnf(simplify(formula))

            potentials = {}
            for lit, w in weight:
                assert(sum(c != 0 for c in w[1]) <= 2)
                smt_vars = list(lit.get_free_variables())
                variables.union(smt_vars)
                str_vars = tuple(sorted(map(lambda x: x.symbol_name(), smt_vars)))

                if str_vars not in potentials:
                    potentials[str_vars] = []

                # TODO: automated conversion sympy -> numpy data structures
                potentials[str_vars].append((flip_negated_literals_cnf(lit), w))

            # ariety assumption
            assert(all(len(dom) in [1,2] for dom in potentials))

            # initializing the nodes
            self.G = nx.Graph()
            for ivar, var in enumerate(variables):
                varname = var.symbol_name()

                if (varname,) in potentials:
                    varp = potentials[(varname,)]
                else:
                    varp = []

                # using str as node type
                self.G.add_node(varname,
                                var=var,
                                index=ivar,
                                clauses=set(),
                                potentials=varp)

            # initializing the edges
            for x, y in [dom for dom in potentials if len(dom) == 2]:

                # TODO: this shouldn't be necessary
                cls = {Or(cond, flip_negated_literals_cnf(Not(cond)))
                       for cond, _ in potentials[(x, y)]}
                self.G.add_edge(x, y, clauses=cls, potentials=potentials[(x, y)])


            if is_literal(formula) or formula.is_or():
                self._add_clause(formula)

            elif formula.is_and():
                for clause in formula.args():
                    if is_literal(clause) or clause.is_or():
                        self._add_clause(clause)
                    else:
                        raise NotImplementedError(MSG_NOT_CNF)
            else:
                raise NotImplementedError(MSG_NOT_CNF)


    def nodes(self):
        """
        Returns the nodes (variables) in the primal graph.
        """
        
        return self.G.nodes

    def edges(self):
        """
        Returns the edges (dependencies) in the primal graph.
        """

        return self.G.edges

    def _add_clause(self, clause):
        """
        Private method. Add 'clause' to LogPrimalGraph.
        Raises NotImplementedError if 'clause':
        - is not a literal or a disjunction of literals
        - has ariety > 2 (TODO: this limitation should be lifted)

        Parameters
        ----------
        clause : pysmt.FNode instance
            The clause to add (a literal or a disjunction of literals)
        """

        # collect vars in the clause to later add "clause edges"
        cvars = set()
        if is_literal(clause):
            cvars = set(map(lambda var: var.symbol_name(),
                            clause.get_free_variables()))

        elif clause.is_or():
            for l in clause.args():
                if not is_literal(l):
                    raise NotImplementedError(MSG_NOT_CLAUSE)

                lvars = set(map(lambda var: var.symbol_name(),
                                l.get_free_variables()))
                cvars = cvars.union(lvars)


        assert(len(cvars) > 0)
        if len(cvars) == 1:
            # univariate clause
            x = list(cvars)[0]
            self.G.nodes[x]['clauses'].add(clause)

        else:
            # TODO: this limitation should be lifted
            if len(cvars) > 2 :
                raise NotImplementedError("clause ariety > 2")

            # multivariate clause
            for x, y in [[u, v] for u in cvars for v in cvars if u < v]:
                if (x, y) not in self.G.edges:
                    self.G.add_edge(x, y,
                                    clauses=set(),
                                    potentials=[])

                self.G.edges[[x, y]]['clauses'].add(clause)

    def get_univariate_formula(self, x):
        """
        Returns the conjunction of the univariate clauses on 'x'.
        The returned CNF formula is a pysmt.FNode instance.

        Parameters

        ----------
        x : str
            The variable name
        """

        return And(self.G.nodes[x]['clauses'])

    def get_bivariate_formula(self, x, y):
        """
        Returns the conjunction of:
        - the univariate clauses on 'x'
        - the univariate clauses on 'y'
        - the bivariate clauses on 'x' and 'y'
        The returned CNF formula is a pysmt.FNode instance.

        Parameters
        ----------
        x : str
            The first variable name
        y : str
            The second variable name
        """
        return And(list(self.G.nodes[x]['clauses'])+
                   #list(self.G.nodes[y]['clauses'])+
                   list(self.G.edges[(x, y)]['clauses']))


    def get_full_formula(self):
        return And([unicl for x in self.G.nodes for unicl in self.G.nodes[x]['clauses']] +
                   [bicl for e in self.G.edges for bicl in self.G.edges[e]['clauses']])

    def get_wmi_problem(self):
        # TODO
        raise NotImplementedError
        all_potentials = []
        for x in self.G.nodes: all_potentials.extend(self.G.nodes[x]['potentials'])
        for e in self.G.edges: all_potentials.extend(self.G.edges[e]['potentials'])

        formula = self.get_full_formula()
        if len(all_potentials) > 0:
            weight = Times([Ite(lit, sympy2pysmt(w), Real(1)) for lit, w in all_potentials])
        else:
            weight = Real(1)

        return formula, weight

    def subprimal(self, subvars):
        assert(all(v in self.G.nodes for v in subvars))
        subp = LogPrimalGraph(None, None)
        subp.G = self.G.subgraph(subvars)
        subp.nvars = self.nvars
        return subp
