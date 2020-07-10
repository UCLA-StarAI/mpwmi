
from fractions import Fraction
import sympy as sym
from sympy.polys.polytools import Poly
import pysmt.shortcuts as smt
from pysmt.operators import AND, OR, NOT, PLUS, TIMES, POW


SMT2SYM = {AND : sym.And,
           OR : sym.Or,
           NOT : sym.Not,
           PLUS : sym.Add,
           TIMES : sym.Mul,
           POW : sym.Pow}

def pysmt2sympy(expr):
    op = expr.node_type()
    if expr.is_bool_constant() or expr.is_real_constant() or expr.is_int_constant():
        return expr.constant_value()
    elif expr.is_symbol():
        return sym.Symbol(expr.symbol_name())
    elif expr.node_type() in SMT2SYM:
        sympyargs = [pysmt2sympy(c) for c in expr.args()]
        return SMT2SYM[op](*sympyargs)

    raise NotImplementedError(f"PYSMT -> SYMPY Not implemented for op: {op}")


SYM2SMT = {sym.And : smt.And,
           sym.Or : smt.Or,
           sym.Not : smt.Not,
           sym.Add : smt.Plus,
           sym.Mul : smt.Times,
           sym.Pow : smt.Pow}

OP_TYPE = {sym.And : smt.BOOL,
           sym.Or : smt.BOOL,
           sym.Not : smt.BOOL,
           sym.Add : smt.REAL,
           sym.Mul : smt.REAL,
           sym.Pow : smt.REAL}
        

def sympy2pysmt(expr, expr_type=None):
    if type(expr) == Poly: # turn Poly instances into generic sympy expressions
        expr = expr.as_expr()

    op = type(expr)

    if len(expr.free_symbols) == 0:
        if expr.is_Boolean:
            return smt.Bool(bool(expr))
        elif expr.is_number:
            return smt.Real(float(expr))

    elif op == sym.Symbol:
        if expr_type is None:
            raise ValueError("Can't create a pysmt Symbol without type information")

        return smt.Symbol(expr.name, expr_type)

    elif op in SYM2SMT:
        if expr_type is None:
            expr_type = OP_TYPE[op]

        smtargs = [sympy2pysmt(c, expr_type) for c in expr.args]
        return SYM2SMT[op](*smtargs)

    raise NotImplementedError(f"SYMPY -> PYSMT Not implemented for op: {op}")

if __name__ == '__main__':
    A, B, C = sym.Symbol("A"), sym.Symbol("B"), sym.Symbol("C")
    f = ~((~(A) | B) & C)

    print("========== LOGIC ==========")
    print(f"type(f) {type(f)}")
    print(f"f {f}")
    print("----------")
    
    f2 = sympy2pysmt(f)
    print(f"type(f2) {type(f2)}")
    print(f"f2 {f2}")
    print("----------")

    f3 = pysmt2sympy(f2)
    print(f"type(f3) {type(f3)}")
    print(f"f3 {f3}")

    
    print("========= ALGEBRA =========")
    x, y, z = sym.Symbol("x"), sym.Symbol("y"), sym.Symbol("z")
    f = (3 * x**2) + (y * z + 1)

    print(f"type(f) {type(f)}")
    print(f"f {f}")
    print("----------")
    
    f2 = sympy2pysmt(f)
    print(f"type(f2) {type(f2)}")
    print(f"f2 {f2}")
    print("----------")

    f3 = pysmt2sympy(f2)
    print(f"type(f3) {type(f3)}")
    print(f"f3 {f3}")
    print("----------")

