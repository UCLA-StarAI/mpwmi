
import networkx as nx
from pympwmi.utils import flip_negated_literals_cnf, get_boolean_variables, is_literal
from pysmt.shortcuts import *
from pympwmi import logger


MSG_NOT_CNF = "the formula must be in CNF"
MSG_NOT_CLAUSE = "the formula must be a clause"

class PrimalGraph:
    """
    A class that implements the primal graph construction from a SMT-LRA 
    'formula' and its manipulation. 

    Raises NotImplementedError if 'formula':
    - is not in CNF
    - contains clauses of ariety > 3

    Attributes
    ----------
    G : networkx.Graph instance
        The structure of the primal graph

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

    def __init__(self, formula):
        """
        Parameters
        ----------
        formula : pysmt.FNode instance
            The input formula representing the support of the distribution
        """

        # TODO: why is this needed?
        formula = flip_negated_literals_cnf(formula)

        # builds the primal graph structure
        self.G = nx.Graph()
        for var in formula.get_free_variables():
            vname = var.symbol_name()
            self.G.add_node(vname, var=var, clauses=set(), literals=set())

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
        Private method. Add 'clause' to PrimalGraph.
        Raises NotImplementedError if 'clause':
        - is not a literal or a disjunction of literals
        - has ariety > 2 (TODO: this limitation should be lifted)

        Parameters
        ----------
        clause : pysmt.FNode instance
            The clause to add (a literal or a disjunction of literals)
        """

        # collect vars in the clause to later add "clause edges"
        #logger.debug(f"add clause: {serialize(clause)}")
        cvars = set()
        if is_literal(clause):
            cvars = set(map(lambda var: var.symbol_name(), clause.get_free_variables()))

        elif clause.is_or():
            for l in clause.args():
                if not is_literal(l):
                    raise NotImplementedError(MSG_NOT_CLAUSE)

                lvars = set(map(lambda var: var.symbol_name(), l.get_free_variables()))
                cvars = cvars.union(lvars)

            #logger.debug(f"is or with cvars: {cvars}")

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
                if [x, y] not in self.G.edges:
                    self.G.add_edge(x, y, clauses=set())

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
                   list(self.G.nodes[x]['clauses'])+
                   list(self.G.edges[(x, y)]['clauses']))

    def get_subformula(self, variables):
        """
        Returns the conjunction of clauses over (a subset of) 'variables'.
        The returned CNF formula is a pysmt.FNode instance.

        Parameters
        ----------
        variables : list of str
            The list of the variables' names
        """
        # TODO: the structure should change in order to implement this
        raise NotImplementedError("clauses of ariety > 2 are not supported yet")
