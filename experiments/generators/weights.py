

import numpy as np
from generators.utils import RAND_SEED
from pysmt.shortcuts import *


class RandomPolynomials:
    """A class that implements a generator of synthetic polynomial weight
    functions.

    Attributes
    ----------
    rand_gen : object
        The random number generator (impl. random() and choice())
    n_monomials : int
        Number of monomials in each polynomial
    const_l : float
        Lower bound for random non-zero constant values
    const_u : float
        Upper bound for random non-zero constant values

    """

    # DEFAULT PARAMS
    # bounds for random constants
    CONST_L, CONST_U = -100.0, 100.0

    # number of monomials
    N_MONOMIALS = 4

    def __init__(self, rand_gen=None, n_monomials=None, const_lu=None):
        """
        Parameters
        ----------
        rand_gen : object
            The random number generator
        const_lu : tuple(float)
            The lower/upper bounds for random constants
        """

        if rand_gen is None:            
            rand_gen = np.random.RandomState(RAND_SEED)

        if n_monomials is None:
            n_monomials = RandomPolynomials.N_MONOMIALS
            
        if const_lu is None:
            const_lu = RandomPolynomials.CONST_L, RandomPolynomials.CONST_U

        self.rand_gen = rand_gen
        self.n_monomials = n_monomials
        self.const_l, self.const_u = const_lu


    def random_constant(self, lower=None, upper=None):
        """Generates a non-zero constant in ['lower','upper'[.

        Returns a pysmt.FNode instance.

        Parameters
        ----------
        lower : float
            Lower bound for the constant value (optional)
        upper : float
            Upper bound for the constant value (optional)

        """        
        coeff = 0
        if lower is None:
            lower = RandomPolynomials.CONST_L
        if upper is None:
            upper = RandomPolynomials.CONST_U

        while coeff == 0:
            coeff = self.rand_gen.random()*abs(upper-lower) + lower
        return Real(coeff)


    def random_monomial(self, variables, degree):
        """Generates a random monomial over continuous 'variables' with a
        given 'degree':
        - each variable has degree >= 1
        - the sum of the degrees is 'degree'

        Returns a pysmt.FNode instance.

        Parameters
        ----------
        variables : list(pysmt.Symbol)
            The list of variables
        degree : int
            Degree of the monomial

        """
        nvars = len(variables)
        assert(nvars > 0)
        if nvars > degree:
            variables = self.rand_gen.choice(variables, degree, replace=False)
            nvars = degree

        # sample the exponents s.t. the monomial has the right 'degree'
        exps = []
        while len(exps) < (nvars - 1):
            maxd = 2 + degree - (nvars - len(exps))
            e = self.rand_gen.randint(1, maxd)
            exps.append(e)

        exps.append(degree - sum(exps))

        pows = [self.random_constant()]
        for i in range(nvars):
            pows.append(Pow(variables[i], Real(exps[i])))

        return Times(pows)


    def random_polynomial(self, variables, degree, nonnegative=True):
        """Generates a random polynomial over continuous 'variables' with a
        given 'degree'.
        
        If 'nonnegative' is True, the actual degree is floor(n/2)*2.

        Returns a pysmt.FNode instance.

        Parameters
        ----------
        variables : list(pysmt.Symbol)
            The list of variables
        degree : int
            The degree of the polynomial
        nonnegative : bool
            Ensures the non-negativity of the polynomial (default True)

        """
        
        if nonnegative:
            return Plus([Times(m, m) for m in
                         [self.random_monomial(variables, int(degree/2))
                          for _ in range(self.n_monomials)]])

        else:
            return Plus([self.random_monomial(variables, degree)
                         for _ in range(self.n_monomials)])
