
from collections import defaultdict
from fractions import Fraction
from typing import List, Dict

from pympwmi import logger
from pympwmi.sympysmt import pysmt2sympy

from pysmt.fnode import FNode
from pysmt.operators import POW
from pysmt.shortcuts import And, Equals, LE, LT, Or, REAL, Real, simplify, substitute, \
    Plus, Solver, Times
from sympy import Add, Mul, Pow, Poly
from sympy import Symbol as symSymbol
from sympy.core.symbol import Symbol as symvar
from sympy.polys.domains.mpelements import RealElement as symReal

# default value for the random seed if not provided by the user
RAND_SEED = 1337

# default SMT solver to be used in pySMT
SMT_SOLVER = "msat"
# SMT_SOLVER = "z3"
# SMT_SOLVER = "cvc4"

#SAFEEXP_CLAMPING = float_info.max
# try decreasing the clamping whenever compensate(..) crashes due to
# NaN potentials
# TODO: (maybe) set this value according to the number of dimensions
SAFEEXP_CLAMPING = 1000
LOG_ZERO = -1000

UPPER = 1
LOWER = 2
NEITHER = 0


def atom_to_bound(atom: FNode, x_symbol: FNode):
    """
    obtain bounds on variable x from inequality atom
    :param atom: must be of plus form or a single term on each side
    :param x_symbol:
    :return:
    """
    assert atom.is_le() or atom.is_lt()
    variables = list(get_real_variables(atom))
    if x_symbol not in variables:
        return [atom, NEITHER]

    lhs, rhs = atom.arg(0), atom.arg(1)

    lhs_coef, lhs_const = get_coefficients(lhs), get_constants(lhs)
    rhs_coef, rhs_const = get_coefficients(rhs), get_constants(rhs)

    lhs_x_coef = lhs_coef[x_symbol] if lhs_coef.get(x_symbol) else Real(0)
    rhs_x_coef = rhs_coef[x_symbol] if rhs_coef.get(x_symbol) else Real(0)
    x_coef_diff = simplify(lhs_x_coef - rhs_x_coef)

    assert float(x_coef_diff.constant_value()) != 0
    flag = simplify(x_coef_diff > 0)
    bound = simplify((rhs_const - lhs_const) / x_coef_diff)
    bound_type = UPPER if flag.is_true() else LOWER

    if len(variables) == 1:
        return [bound, bound_type]

    y_symbol = variables[0] if x_symbol == variables[1] else variables[1]
    lhs_y_coef = lhs_coef[y_symbol] if lhs_coef.get(y_symbol) else Real(0)
    rhs_y_coef = rhs_coef[y_symbol] if rhs_coef.get(y_symbol) else Real(0)
    y_coef_diff = simplify(lhs_y_coef - rhs_y_coef)

    if float(y_coef_diff.constant_value()) == 0:
        return [bound, bound_type]

    new_y_coef = simplify(y_coef_diff * Real(-1) / x_coef_diff)
    bound = Plus(bound, Times(new_y_coef, y_symbol))
    return [bound, bound_type]


def categorize_bounds(formula: FNode, sender_symbol: FNode):
    """
    categorize symbolic bounds for a given variable
    :param formula: FNode formula in CNF form
    :param sender_symbol: FNode of the bounded variable
    :return: upper_bounds, lower_bounds, parent_atoms
    """
    clauses = set()
    formula = simplify(formula)
    if formula.is_or() or is_literal(formula):
        clauses.add(formula)
    else:  # formula of CNF form
        for clause in formula.args():
            assert is_literal(clause) or clause.is_or()
            clauses.add(clause)

    bounds = defaultdict(list)
    for clause in clauses:
        if clause.is_le() or clause.is_lt():  # single literal clause
            bound, bound_type = atom_to_bound(clause, sender_symbol)
            bounds[bound_type].append(bound)
            continue
        for atom in clause.get_atoms():  # atom: inequality a * x + b * y < c
            assert atom.is_le() or atom.is_lt()
            bound, bound_type = atom_to_bound(atom, sender_symbol)
            bounds[bound_type].append(bound)
    return list(set(bounds[UPPER])), list(set(bounds[LOWER])), \
        list(set(bounds[NEITHER]))



def clause_to_intervals(clause: FNode,
                        sender_symbol: FNode,
                        test_point: float):
    """
    turn clause into a list of  intervals,
    where there are at most two intervals
    :param clause: OR FNode or a single atom
    :param sender_symbol:
    :param test_point: an initiation value for recipient variable
    :return: list of tuples of form [lower, upper]
    """
    upper_bounds, lower_bounds, recipient_atoms = categorize_bounds(
        clause, sender_symbol
    )
    if recipient_atoms:
        parent_symbol = list(get_real_variables(recipient_atoms[0]))
        for atom in recipient_atoms:
            bound = simplify(substitute(
                atom, {parent_symbol[0]: Real(test_point)})
            )
            if bound.is_true():
                return []

    num_uppers = [initiate_bound(upper, test_point) for upper in upper_bounds]
    num_lowers = [initiate_bound(lower, test_point) for lower in lower_bounds]
    if not num_uppers and not num_lowers:
        return []
    if not num_uppers:
        return [[min(num_lowers), float('inf')]]
    if not num_lowers:
        return [[float('-inf'), max(num_uppers)]]

    max_upper, min_lower = max(num_uppers), min(num_lowers)
    if max_upper < min_lower:
        return [[float('-inf'), max_upper], [min_lower, float('inf')]]
    else:
        return []


# TODO: documentation is incomplete/ambiguous. method might need refactoring too
def construct_bound_dict(sender_constraints: list,
                         sender_symbol: FNode,
                         test_point: float,
                         formula: FNode):
    """
    read from constrains and formula Delta_x,y to obtain integration bound
    dictionary from numeric ones to symbolic ones
    :param sender_constraints: list of tuples of form [l, u, p]
    with l, u float bounds and p a polynomial
    :param sender_symbol:
    :param test_point:
    :param formula: formula: edge formula, Delta_x,y
    :return: two dictionary for upper bounds and lower bounds respectively
    """
    upper_bounds, lower_bounds, _ = categorize_bounds(formula, sender_symbol)
    num_upper_bounds, num_lower_bounds = dict(), dict()
    for upper in upper_bounds:
        num_upper_bounds[initiate_bound(upper, test_point)] = upper
    for lower in lower_bounds:
        num_lower_bounds[initiate_bound(lower, test_point)] = lower
    for lower, upper, _ in sender_constraints:
        num_upper_bounds[upper] = Real(upper)
        num_lower_bounds[lower] = Real(lower)
    return num_upper_bounds, num_lower_bounds

def domains_to_intervals(domains):
    """
    turn formula domain formula into list of domain interval points
    this work for non-trivial coefficient formulas
    :param domains: FNode formula
    :return: list of interval points, type float, e.g. [[1,2], [3,4]] which means domain (1<x<2 or 3<x<4)
    """
    domains = simplify(domains)
    real_variables = list(get_real_variables(domains))
    assert len(real_variables) == 1
    var = real_variables[0]
    bounds = []
    for atom in domains.get_atoms():
        solution = solve_equation(atom.arg(0), atom.arg(1))
        bounds.append(solution[var])

    bounds = list(sorted({float(bound.constant_value()) for bound in bounds}))
    intervals = filter_intervals(bounds, var, domains)
    return intervals


def filter_intervals(bounds: List, var, domains, solver_name='msat'):
    min_bound, max_bound = min(bounds), max(bounds)
    extended_bounds = bounds + [min_bound - 1, max_bound + 1]  # replace inf
    extended_bounds.sort()

    intervals = list(zip(extended_bounds, extended_bounds[1:]))
    res_intervals = []
    for start, end in intervals:
        test_point = (start + end) * 0.5
        test = Equals(var, Real(test_point))
        problem = domains.And(test)
        with Solver(name=solver_name) as solver:
            solver.add_assertion(problem)
            if solver.solve():
                if start < min_bound:
                    res_intervals.append([float('-inf'), end])
                elif end > max_bound:
                    res_intervals.append([start, float('inf')])
                else:
                    res_intervals.append([start, end])
    return res_intervals


def find_edge_critical_points(sender_domains: FNode,
                              recipient_domains: FNode,
                              edge_formula: FNode):
    """
    collect critical points for recipents
    :param sender_domains: CNF involving child variable only
    :param recipient_domains: CNF involving parent variable only
    :param edge_formula: CNF involving both
    :return critical_points: a list of fraction constants
    """
    sender_symbol = list(get_real_variables(sender_domains))
    sender_symbol = sender_symbol[0]
    recipient_symbol = list(get_real_variables(recipient_domains))
    recipient_symbol = recipient_symbol[0]

    upper_bounds, lower_bounds, recipient_atoms = categorize_bounds(
        edge_formula, sender_symbol
    )

    critical_points = []
    # collect critical points from recipient atoms
    for atom in recipient_atoms:
        lower, upper = atom.arg(0), atom.arg(1)
        solution = solve_equation(lower, upper, domains=recipient_domains)
        if solution:
            critical_points.append(solution[recipient_symbol])

    # collect critical points from recipient domain
    intervals = domains_to_intervals(recipient_domains)
    critical_points.extend(intervals_to_points(intervals))

    # collect critical points from solving bounds for sender
    intervals = domains_to_intervals(sender_domains)
    points = intervals_to_points(intervals)
    bounds = list(set(upper_bounds + lower_bounds + points))
    critical_points.extend(
        get_critical_points(bounds, domains=recipient_domains))
    critical_points = list(set(c.constant_value() for c in critical_points))

    return critical_points



NOT_CSTR = 0  # not constraints
LEFT_END = 1
RIGHT_END = 0


# TODO: documentation is incomplete/ambiguous. method might need refactoring too
def find_symbolic_bounds(sender_constraints: list,
                         sender_symbol: FNode,
                         test_point: float,
                         formula: FNode):
    """
    find the symbolic integration bounds and also index of integrand
    :param sender_constraints: list of tuples of form [l, u, p]
    with l, u float bounds and p a polynomial
    :param sender_symbol: FNode of variable
    :param test_point: mid point of an interval of recipient
    :param formula: edge formula, Delta_x,y
    :return: list of tuples of form [symbolic bounds, index]
    """
    interval_list = formula_to_interval_set(formula, sender_symbol, test_point)
    list_num = len(interval_list)
    points = []
    for i in range(list_num):
        for interval in interval_list[i]:
            start, end = interval
            points.append([start, NOT_CSTR, i, LEFT_END])
            points.append([end, NOT_CSTR, i, RIGHT_END])

    for idx, interval in enumerate(sender_constraints):
        start, end, _ = interval
        points.append([start, idx, list_num, LEFT_END])
        points.append([end, idx, list_num, RIGHT_END])
    points.sort(key=lambda p: (p[0], p[3]), reverse=False)
    intersection = line_sweep(points, list_num + 1)
    num_upper_bounds, num_lower_bounds = construct_bound_dict(
        sender_constraints, sender_symbol, test_point, formula
    )

    symbolic_bounds = []
    for start, end, idx in intersection:
        symbolic_bounds.append(
            [num_lower_bounds[start], num_upper_bounds[end], idx]
        )
    return symbolic_bounds


def get_boolean_variables(formula):
    return {a for a in formula.get_free_variables() if a.get_type() == BOOL}


def get_coefficients(atom: FNode):
    """
    obtain coefficient of variable x in atom of form a * x + b * y + const
    note that when there is a * x + b * x, simplify() doesn't do multiplication
    but still here we return (a + b)
    :param atom:  FNode, formula
    :param x:  FNode, symbol of variable
    :return: dict with keys as variable symbol and values as coefficient in atom
    """
    variables = list(get_real_variables(atom))
    coefficients = defaultdict(int)

    if len(variables) == 0:
        return coefficients

    const = get_constants(atom)
    atom = simplify(Plus(atom, -const))

    sub_dict = dict().fromkeys(variables, Real(0))
    for i in range(len(variables)):
        sub_dict[variables[i]] = Real(1)
        coefficient = simplify(atom.substitute(sub_dict))
        coefficients[variables[i]] = coefficient
        sub_dict[variables[i]] = Real(0)
    return coefficients


def get_constants(atom: FNode):
    """
    obtain constant term in a linear real arithmetics
    :param atom: FNode formula of form a * x + b * y + c
    :return:
    """
    variables = list(get_real_variables(atom))
    new_atom = atom
    for var in variables:
        new_atom = new_atom.substitute({var: Real(0)})
    return simplify(new_atom)


def get_critical_points(bounds: List[FNode],
                        domains: FNode = None) -> List[FNode]:
    """
    get critical points in variable domain from a set of bounds
    :param bounds: list of FNode atoms
    :param domains: domain of variable
    :return:
    """
    bounds = list(set(bounds))
    critical_points = []
    for i in range(len(bounds)):
        for j in range(i):
            lhs, rhs = bounds[i], bounds[j]
            real_variables = list(get_real_variables(lhs.Equals(rhs)))
            if len(real_variables) == 0:
                continue
            solution = solve_equation(lhs, rhs, domains=domains)
            if solution:
                critical_points.append(solution[real_variables[0]])
    critical_points = list(set(critical_points))
    return critical_points


def get_real_variables(formula):
    return {a for a in formula.get_free_variables() if a.get_type() == REAL}

def initiate_bound(bound: FNode,
                   initiation: float) -> float:
    """
    initiate an atom with only one variable with value initiation
    :param bound: FNode atom
    :param initiation: initiation of variable, type float
    :return: initiated bound, type float
    """
    real_variables = list(get_real_variables(bound))
    assert len(real_variables) < 2
    if real_variables:
        # print(real_variables[0], Real(initiation))
        bound = simplify(substitute(
            bound, {real_variables[0]: Real(initiation)})
        )
    return float(bound.constant_value())


def intervals_to_points(intervals: List[List[float]]) -> List[FNode]:
    """
    turn list of interval bounds to a list of endpoints
    :param intervals: list of lists, type float
    :return: list of bound points, type FNode
    """
    points = []
    for interval in intervals:
        points.extend(interval)
    points = sorted(list(set(points)))
    points = [Real(point) for point in points]
    return points


def is_atom(f):
    """Returns True if 'f' is an atom."""
    return f.is_theory_relation() or (f.is_symbol() and
                                      f.symbol_type() == BOOL)


def is_literal(f):
    """Returns True if 'f' is a literal."""
    return is_atom(f) or (f.is_not() and is_atom(f.args()[0]))


def line_sweep(points, list_num):
    num_set = sum_degree = 0
    start = 0
    intersection = []  # element: [start, end, sum of degrees]
    for x, degree, index, side in points:
        if side == LEFT_END:
            num_set += 1
            sum_degree += degree
            if num_set == list_num:
                start = x
        elif side == RIGHT_END:
            if num_set == list_num:  # end record
                intersection.append([start, x, sum_degree])
            num_set -= 1
            sum_degree -= degree
    return intersection


def literal_to_bounds(f: FNode):
    """
    obtain bounds on variable x from inequality atom
    :param literal: ax + by + c < dx + ey + f
    :param x: variable
    :return: symbolic bound for x, is_lower, k_inclduded

    for example, given literal 3x + 2y + 4 < x + y + 1,
    first move all terms to left hand side: 2x + y + 3 < 0,
    then it returns [-1/2 y - 3/2, False]
    since the literal is equivalent to x < -1/2 y - 3/2
    """
    assert (is_literal(f))
    assert f.is_le() or f.is_lt()
    variables = list(get_real_variables(f))

    k_included = f.is_le()
    f = simplify(f)
    lhs, rhs = f.arg(0), f.arg(1)

    lhs_coef, lhs_const = get_coefficients(lhs), get_constants(lhs)
    rhs_coef, rhs_const = get_coefficients(rhs), get_constants(rhs)
    # move all terms to lhs
    const = simplify(lhs_const - rhs_const)
    coef = dict()
    for v in variables:
        coef[v] = simplify(lhs_coef[v] - rhs_coef[v])

    bounds = dict()
    for v in variables:
        if float(coef[v].constant_value()) == 0:
            continue

        bound_terms = [simplify(const * Real(-1) / coef[v])]
        for w in variables:
            if w == v or float(coef[w].constant_value()) == 0:
                continue
            new_w_coef = simplify(coef[w] * Real(-1) / coef[v])
            bound_terms.append(Times(new_w_coef, w))
        bound = Plus(bound_terms)
        is_lower = coef[v].constant_value() < 0
        bounds[v] = bound, is_lower, k_included

    return bounds


def flip_ra_atom(atom):
    """
    Flips an LRA/NRA atom, i.e.:
    - if 'atom' is (x < y), returns (y <= x)
    - if 'atom' is (x <= y), returns (y < x)
    otherwise, raises ValueError.

    Parameters
    ----------
    atom : pysmt.FNode instance
        The LRA/NRA atom
    """
    if atom.is_le():
        return LT(atom.args()[1], atom.args()[0])
    elif atom.is_lt():
        return LE(atom.args()[1], atom.args()[0])
    else:
        raise ValueError("Negated literals outside LE and LT are not supported")


def flip_negated_literals_cnf(f):
    """
    Removes the negated LRA/NRA literals in 'f' by pushing the negation into the
    relation, while leaving the negated boolean atoms unaltered.
    Returns a pysmt.FNode instance that is equivalent to 'f'.
    Raises ValueError if the 'f' is not in CNF.

    Parameters
    ----------
    f : pysmt.FNode instance
    """

    if is_literal(f):
        if is_ra_literal(f) and f.is_not():
            return flip_ra_atom(f.args()[0])

        else:
            return f

    elif f.is_or():
        return Or([flip_negated_literals_cnf(l) for l in f.args()])

    elif f.is_and():
        return And([flip_negated_literals_cnf(c) for c in f.args()])

    else:
        raise ValueError("Formula not in CNF {}".format(f))


# TODO: documentation is incomplete/ambiguous. method might need refactoring too
def formula_to_interval_set(formula: FNode,
                            sender_symbol: FNode,
                            test_point: float):
    clauses = set()
    if formula.is_or() or is_literal(formula):
        clauses.add(formula)
    else:  # formula of CNF form
        for clause in formula.args():
            assert is_literal(clause) or clause.is_or()
            clauses.add(clause)

    interval_list = []
    for clause in clauses:
        intervals = clause_to_intervals(clause, sender_symbol, test_point)
        if intervals:
            interval_list.append(intervals)
    return interval_list


def is_ra_literal(f):
    """
    Tests whether 'f' is an LRA/NRA literal.

    Parameters
    ----------
    f : pysmt.FNode instance
        Input formula
    """
    return is_literal(f) and all(v.symbol_type() == REAL
                                 for v in f.get_free_variables())


def is_sympy_expr(expr):
    """
    Tests whether 'expr' is a sympy expression.

    Parameters
    ----------
    expr : any object
        Input formula
    """
    return 'sympy' in type(expr).__module__


def parse_univariate_literal(f):
    """
    Not complete. Given a univariate literal f, returns:
    - the variable x
    - the simplified endpoint k
    - a flag that is True iff f is a lower bound to x
    - a flag that is True iff the endpoint k is included

    WARNING: again, not complete, i.e. it doesn't work for arbitrary
    univariate literals.
    """
    assert (is_literal(f))
    assert (len(f.get_free_variables()) == 1)
    # TODO: is this needed, given that we flip every xRA literal as a
    # preprocessing step?
    if f.is_not():
        f = flip_ra_atom(f.args()[0])

    f = simplify(f)
    x = list(f.get_free_variables())[0]
    k_included = f.is_le()

    l, r = f.args()
    if l.is_real_constant():
        is_lower = True
        k = l.constant_value()
        ax = r

    elif r.is_real_constant():
        is_lower = False
        k = r.constant_value()
        ax = l

    else:
        raise ValueError(f"Can't parse the univariate literal {f.serialize()}")

    if ax.is_times():
        assert (len(ax.args()) == 2)
        assert (ax.args()[0].is_symbol())
        assert (ax.args()[1].is_real_constant())
        k = k / ax.args()[1].constant_value()

    elif ax.is_plus():
        assert (len(ax.args()) == 2)
        assert (ax.args()[0].is_symbol())
        assert (ax.args()[1].is_real_constant())
        k = k - ax.args()[1].constant_value()

    elif not ax.is_symbol():
        raise ValueError(f"Can't parse the univariate literal {f.serialize()}")

    return x, k, is_lower, k_included


def to_sympy(expr):
    """
    Tries to convert an object into something that is amenable to symbolic
    integration with sympy.

    Parameters
    ----------
    expr : object
        The algebraic expression to convert
    """

    try:
        return Fraction(str(expr))
    except ValueError:
        pass
    if type(expr) == FNode:
        return pysmt2sympy(expr)
    else:
        # TODO: here we might want to add some sanity checks on the input
        # instead of just returning it unaltered.
        return expr


def weight_to_lit_potentials(expr):
    """Converts a pysmt.FNode instance into the MPWMI data structure for
    weights if the input encodes potentials associated to literals, i.e.:

    Times(Ite(lit_1, w_1, Real(1)), ..., Ite(lit_n, w_n, Real(1)))

    Or the constant Real(1).

    Raises ValueError otherwise.


    Parameters
    ---------- 
    expr : pysmt.FNode 
        The pysmt expression
    """
    def ite_to_potential(expr):

        if expr.is_ite():
            cond, wpos, wneg = expr.args()
            if (is_literal(cond) and wneg.is_real_constant() and wneg.constant_value() == 1):
                return cond, wpos

        raise ValueError(
            "The expression is not Times(Ite(lit_1, w_1, Real(1)), ..., Ite(lit_n, w_n, Real(1)))")

    if expr.is_times():
        wls = [ite_to_potential(a) for a in expr.args()]
    elif expr.is_real_constant() and expr.constant_value() == 1:
        return {}
    else:
        wls = [ite_to_potential(expr)]

    potentials = {}
    for lit, w in wls:
        variables = list(lit.get_free_variables())
        v = tuple(sorted(map(lambda x: x.symbol_name(), variables)))
        assert(len(v) in [1, 2]), "Not implemented for ternary atoms"

        if v not in potentials:
            potentials[v] = []

        symv = tuple([symvar(x) for x in v])
        symw = Poly(to_sympy(w), symv, domain="RR")
        potentials[v].append((lit, symw))

    return potentials


def ordered_variables(expr):
    """Given an input sympy 'expr', returns an ordered list of its
    variables.

    Parameters
    ---------- 
    expr : sympy.something
        The sympy expression

    """
    if expr.is_symbol:
        return [expr]
    elif len(expr.args) == 0:
        return []
    else:
        res = []
        for a in expr.args:
            res.extend(ordered_variables(a))

        return res
        # return list(reduce(lambda x, y: x.extend(y),
        #                    [ordered_variables(a) for a in expr.args]))


def dict_to_tuple(d):
    if d == {}:
        return [(((), symReal(0)))]

    t = list(d.keys())
    assert len(t[0]) < 3, f"t[0] = {t[0]} should not happen"
    if len(t[0]) == 1:
        return [tuple(sorted(
            (tuple((k, v)) for k, v in d.items()), key=lambda p: p[0][0]
        ))]
    elif len(t[0]) == 2:
        return [tuple(sorted(
            (tuple((k, v)) for k, v in d.items()),
            key=lambda p: (p[0][0], p[0][1])
        )), tuple(sorted(
            (tuple(((k[1], k[0]), v)) for k, v in d.items()),
            key=lambda p: (p[0][0], p[0][1])
        ))]


def safeexp(x):
    return min(np.exp(x), SAFEEXP_CLAMPING)


def safelog(x):
    x = float(x)
    if np.isclose(x, 0.0):
        return LOG_ZERO
    else:
        return np.log(x)


def solve_equation(lhs: FNode,
                   rhs: FNode,
                   domains: FNode = None,
                   solver_name: str = 'msat'):
    """
    solve FNode equations
    :param lhs: FNode formula
    :param rhs: FNode formula
    :param domains: domain of variable, type FNode formula
    :param solver_name:
    :return: solution of variable, dict with key as variable and value as solution, both type FNode
    """
    problem = lhs.Equals(rhs)
    formula = problem if not domains else domains.And(problem)
    real_variables = get_real_variables(formula)

    with Solver(name=solver_name) as solver:
        solver.add_assertion(formula)
        if solver.solve():
            solution = {}
            for rvar in real_variables:
                solution[rvar] = solver.get_value(rvar)
            return solution
        else:
            return None
