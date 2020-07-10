
import networkx as nx
import numpy as np
from pysmt.shortcuts import *
from pywmi import Domain
from generators.utils import RAND_SEED

class ForestGenerator:
    """A class that implements a generator of synthetic
    SMT-LRA formulas in CNF with forest-shaped Primal Graphs.

    The generated formulas contain univariate and bivariate LRA
    literals only. The clauses in the formula involve at most 2
    continuous variables.

    Attributes
    ----------
    rand_gen : object
        The random number generator (impl. random() and choice())
    variable_name : function (int -> str)
        A function returning a variable name given the index
    univ_l : float
        The lower bound for continuous vars
    univ_u : float
        The upper bound for continuous vars
    max_m : float
        The maximum linear coefficient
    p_univ : float (in [0,1])
        The probability of drawing a univariate lit
    p_pos : float (in [0,1])
        The probability of drawing a positive lit

    """

    # DEFAULT PARAMS
    # univariate bounds for continuous vars
    UNIV_L, UNIV_U = -1.0, 1.0

    # linear coefficient
    MAX_M = 1000.0

    # Pr(univariate_lit), Pr(positive_lit)
    P_UNIV = P_POS = 0.5

    VARIABLE_NAME = lambda i : f"x{i}"

    # shape enum
    SHAPE_ARGSEP = '-'
    SHAPE_PATH = 'p'
    SHAPE_RAND = 'r' # not for now tho
    SHAPE_SNOW = 'sn'
    SHAPE_STAR = 'st'
    SHAPE_TREE = 't'
    SHAPES = [SHAPE_TREE, SHAPE_PATH, SHAPE_SNOW, SHAPE_STAR] #, SHAPE_RAND] not for now

    def __init__(self, rand_gen=None, variable_name=None,
                 univ_lu=None, max_m=None, p_univ=None, p_pos=None):
        """
        Parameters
        ----------
        rand_gen : object
            The random number generator
        variable_name : function (int -> str)
            A function returning a variable name given the index
        univ_lu : tuple(float)
            The lower/upper bounds for continuous vars
        max_m : float
            The maximum linear coefficient
        p_univ : float (in [0,1])
            The probability of drawing a univariate lit
        p_pos : float (in [0,1])
            The probability of drawing a positive lit
        """
        if rand_gen is None:
            rand_gen = np.random.RandomState(RAND_SEED)

        if variable_name is None:
            variable_name = ForestGenerator.VARIABLE_NAME

        if univ_lu is None:
            univ_lu = ForestGenerator.UNIV_L, ForestGenerator.UNIV_U

        if max_m is None:
            max_m = ForestGenerator.MAX_M

        if p_univ is None:
            p_univ = ForestGenerator.P_UNIV

        if p_pos is None:
            p_pos = ForestGenerator.P_POS

        self.rand_gen = rand_gen
        self.variable_name = variable_name
        self.univ_l, self.univ_u = univ_lu
        self.max_m = max_m
        self.p_univ = p_univ        
        self.p_pos = p_pos

    def random_univ_literal(self, x, lower=None, upper=None):
        if lower is None:
            lower = self.univ_l
        if upper is None:
            upper = self.univ_u

        D = abs(upper - lower)
        c = Real(self.rand_gen.random() * D + lower)
        rel = self.rand_gen.choice([LT, LE, GT, GE])
        atom = rel(x, c)
        pos = self.rand_gen.random() < self.p_pos  # positive or negative lit
        return atom if pos else Not(atom)    


    def random_biv_literal(self, x, y, lower=None, upper=None):
        if lower is None:
            lower = self.univ_l
        if upper is None:
            upper = self.univ_u

        D = abs(upper - lower)
        m = Real(self.rand_gen.random() * self.max_m)
        q = Real(self.rand_gen.random() * D + lower)
        rel = self.rand_gen.choice([LT, LE, GT, GE])
        atom = rel(Plus(Times(m, x), q), y)
        pos = self.rand_gen.random() < self.p_pos  # positive or negative lit
        return atom if pos else Not(atom)


    def random_literal(self, x, y, lower=None, upper=None):
        if lower is None:
            lower = self.univ_l
        if upper is None:
            upper = self.univ_u
        '''
        TODO: in setup.py add the dependency from numpy 1.17+
        otherwise, RandomState does not have the 'random' method
        '''
        if self.rand_gen.random() < self.p_univ:  # univariate or bivariate lit
            v = self.rand_gen.choice([x, y])
            return self.random_univ_literal(v, lower, upper)
        else:
            return self.random_biv_literal(x, y, lower, upper)

        
    def random_formula(self, nvars, nclauses, nlits, shape=None, allow_unsat=False):
        """Generates a random SMT-LRA formula in CNF with a forest-shaped
        Primal Graph.

        For each edge in the dependency graph, 'nclauses' are generated.

        If 'allow_unsat' is False (by default), ensures that the formula is SAT.

        Returns the formula as a pysmt.FNode instance and
        the pywmi.Domain.
        
        Parameters
        ----------
        nvars : int
            Number of continuous variables (> 1)
        nclauses : int
            Number of clauses for each dependency
        nlits : int
            Number of literals in each clause
        shape : str in ForestGenerator.SHAPES
            Shape of the resulting Primal Graph
        allow_unsat : bool
            If True, the procedure doesn't check for satisfiability
        """
        assert(nvars > 1)
        variables = [Symbol(self.variable_name(i), REAL) for i in range(nvars)]        
        f = self.random_formula_with_vars(variables, nclauses, nlits, shape)

        tries = 1
        if not allow_unsat:
            while not is_sat(f):
                print(f"formula was UNSAT: {tries}")
                tries += 1
                f = self.random_formula_with_vars(variables, nclauses, nlits, shape)
            
        varnames = [v.symbol_name() for v in variables]
        dom = Domain(varnames,
                     {vname : REAL for vname in varnames},
                     {vname : [self.univ_l, self.univ_u] for vname in varnames})
        return f, dom


    def random_formula_with_vars(self, variables, nclauses, nlits, shape=None):
        """Generates a random SMT-LRA formula in CNF with a forest-shaped
        Primal Graph using the input variables.

        For each edge in the dependency graph, 'nclauses' are generated.

        Returns the formula as a pysmt.FNode.
        
        Parameters
        ----------
        variables : list(pysmt.FNode)
            The list of Symbol instances
        nclauses : int
            Number of clauses
        nlits : int
            Number of literals in each clause
        shape : str in ForestGenerator.SHAPES
            Shape of the resulting Primal Graph

        """        
        if shape is None:
            print("WARNING: No shape specified, using SHAPE_TREE")
            shape = ForestGenerator.SHAPE_TREE

        L, U = self.univ_l, self.univ_u
        D = abs(U - L)
        nvars = len(variables)
        clauses = []

        # adding univariate bounds
        for i in range(nvars):            
            clauses.append(LE(Real(L), variables[i]))
            clauses.append(LE(variables[i], Real(U)))

        shapeargs = shape.split(ForestGenerator.SHAPE_ARGSEP)

        if shapeargs[0] == ForestGenerator.SHAPE_PATH:
            deps = list(nx.generators.path_graph(nvars).edges)
        elif shapeargs[0] == ForestGenerator.SHAPE_RAND:
            nedges = shapeargs[1]
            deps = list(nx.generators.trees.random_tree(nvars, seed=self.rand_gen).edges)
            candidate_edges = set([(u, v)
                                   for u in range(0, nvars-1)
                                   for v in range(u, nvars)
                                   if (u, v) not in deps])
                                          
            while len(deps) < nedges or len(candidate_edges) == 0:
                edge = self.rand_gen.choice(candidate_edges)
                deps.append(edge)
                candidate_edges.remove(edge)
   
        elif shapeargs[0] == ForestGenerator.SHAPE_SNOW:
            r = int(shapeargs[1])
            deps = list(nx.generators.classic.full_rary_tree(r, nvars).edges)
        elif shapeargs[0] == ForestGenerator.SHAPE_STAR:
            deps = list(nx.generators.star_graph(nvars-1).edges)
        elif shapeargs[0] == ForestGenerator.SHAPE_TREE:
            deps = list(nx.generators.trees.random_tree(nvars, seed=self.rand_gen).edges)
        else:
            raise NotImplementedError(f"shape not in ForestGenerator.SHAPES = {ForestGenerator.SHAPES}")

        for ix, iy in deps:
            for _ in range(nclauses):
                x = variables[ix]
                y = variables[iy]
                clause = []
                for h in range(nlits):
                    if h == 0:
                        literal = self.random_biv_literal(x, y, L, U)
                    else:
                        literal = self.random_literal(x, y, L, U)

                    clause.append(literal)

                clauses.append(Or(clause))

        formula = And(clauses)
        return formula
