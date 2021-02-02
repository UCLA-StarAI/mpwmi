
from functools import reduce
from itertools import product
from math import e as numexp
from pysmt.shortcuts import simplify, substitute
from pysmt.fnode import FNode

from sympy import Poly
from sympy.core.symbol import Symbol as symvar
from sympy.core.numbers import One
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement

from pympwmi.utils import literal_to_bounds, to_sympy


class Message:

    #ONE = One()

    @staticmethod
    def product(x, y):
        raise NotImplementedError("Can't use the base class")

    @classmethod
    def potentials_to_messages(cls, potentials, xvar, subs=None):
        msgs = []
        for lit, f in potentials:
            msg = []
            k, is_lower, _ = literal_to_bounds(lit)[xvar]
            if subs is not None:
                k = simplify(substitute(k, subs))
            k = k.constant_value()
            if is_lower:
                msg.append((float('-inf'), k, cls.ONE))
                msg.append((k, float('inf'), f))
            else:
                msg.append((float('-inf'), k, f))
                msg.append((k, float('inf'), cls.ONE))
            msgs.append(msg)

        print("POTENTIALS TO MESSAGES:", msgs)
        return msgs

    @classmethod
    def intersect(cls, msgs, tolerance):
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
                print("----------")
                print("start", start)
                print("end", end)
                print("integrand", integrand)
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
        intersection = cls._line_sweep(points, len(msgs))
        intersection = cls._filter_msgs(intersection, tolerance)
        return intersection

    @staticmethod
    def _filter_msgs(msgs: list, tolerance): # TODO: improve
        f_msgs = []
        for start, end, integrand in msgs:
            if abs(end - start) < tolerance:
                continue
            f_msgs.append((start, end, integrand))
        return f_msgs

    @classmethod
    def _line_sweep(cls, points, n_msgs):
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
                    integrand = reduce(lambda x, y: cls.product(x, y),
                                       factors[1:], factors[0])
                    intersection.append([l, x, integrand])

                factors[msgid] = None
                l = None
                n_factors -= 1

            else:
                assert(factors[msgid] is None), f"damn {factors[msgid]}"                
                factors[msgid] = f
                l = x
                n_factors += 1

        assert(n_factors == 0)
        assert(factors == [None] * n_msgs)

        return intersection


    @classmethod
    def integrate(cls, cache, message, x, y=None):
        """
        Computes the symbolic integral of 'x' of a piecewise polynomial 'message'.
        The result might be a sympy expression or a numerical value.

        Parameters
        ----------
        message : list
            A list of (lower bound, upper bound, polynomial)
        x : object
            A string/sympy expression representing the integration variable
        """
        cache_hit = [0, 0] if (cache is not None) else None

        print("----------------")
        print("integrating variable:", x)
        res = 0
        x, y = cls.preprocess_vars(x, y)
        for l, u, p in message:
            print("L", type(l), "--", l)
            print("U", type(u), "--", u)
            print("P", type(p), "--", p)
            print()

            l, u, p = cls.preprocess_piece(l, u, p, x, y)

            if cache is not None:  # for cache = True
                integral = cls.integrate_cache(cache, cache_hit, l, u, p, x, y)
            else:  # for cache = False
                antidrv = cls.antiderivative(p, x)
                lower = cls.substitute(antidrv, x, y, l)
                upper = cls.substitute(antidrv, x, y, u)
                print("p", p)
                print("l", l)
                print("u", u)
                print("antidrv", antidrv)
                print("upper", upper)
                print("lower", lower)
                integral = cls.substract(upper, lower)

            res = cls.sum(res, integral)

        return res, cache_hit



    @staticmethod
    def to_cache(p):
        raise NotImplementedError("Can't use the base class")


    @staticmethod
    def from_cache(t, x, y):
        raise NotImplementedError("Can't use the base class")


class SympyMessage(Message):

    ONE = One()

    @staticmethod
    def preprocess_vars(x, y):
        x = symvar(x)
        y = symvar(y) if y else symvar("aux_y")
        return x, y

    @staticmethod
    def preprocess_piece(l, u, p, x, y):

        l = Poly(to_sympy(l), y, domain="QQ")            
        u = Poly(to_sympy(u), y, domain="QQ")

        if type(p) != Poly:
            p = Poly(to_sympy(p), x, domain="QQ")
        else:
            p = Poly(p.as_expr(), x, y, domain="QQ")

        return l, u, p

    @staticmethod
    def product(x, y):
        return x * y

    @staticmethod
    def sum(x, y):
        return x + y
    
    @staticmethod
    def subtract(x, y):
        return x - y

    @staticmethod
    def antiderivative(f, var):
        return f.integrate(var)

    @staticmethod
    def substitute(f, x, y, sub):
        f2 = Poly(f.as_expr(), x,
                     domain=f'QQ[{y}]').eval({x: sub.as_expr()})
        return Poly(f2.as_expr(), x, y, domain="QQ")

    @classmethod
    def integrate_cache(cls, cache, cache_hit, l, u, p, x, y):

        """ hierarchical cache, where we cache:
        - the anti-derivatives for messages, retrieved by:
                (None, None, message key)
        - the partial integration term, retrieved by:
                (lower bound key, None, message key)
                (None, upper bound key, message key)
        - the whole integration, retrieved by:
                (lower bound key, upper bound key, message key)
        """

        # cache keys for bounds
        k_lower = cls.to_cache(l)
        k_upper = cls.to_cache(u)
        k_poly = cls.to_cache(p)  # cache key for message polynomial
        k_full = (k_lower, k_upper, k_poly)

        if k_full in cache:
            # retrieve the whole integration                    
            cache_hit[True] += 1
            integral = cls.from_cache(cache[k_full], x, y)
            integral = integral.subs(integral.gens[0], y)

        else:
            # retrieve partial integration terms
            terms = [None, None]
            k_part_l = (k_lower, k_poly)
            k_part_u = (k_upper, k_poly)
            if k_part_l in cache:
                partial_l = cls.from_cache(cache[k_part_l], x, y)
                terms[0] = partial_l.subs(partial_l.gens[0], y)

            if k_part_u in cache:
                partial_u = cls.from_cache(cache[k_part_u], x, y)
                terms[1] = partial_u.subs(partial_u.gens[0], y)

            if None not in terms:
                cache_hit[True] += 1
            else:
                # retrieve anti-derivative
                k_anti = (k_poly,)
                if k_anti in cache:                            
                    cache_hit[True] += 1
                    antidrv = cls.from_cache(cache[k_anti], x, y)

                else:
                    cache_hit[False] += 1
                    antidrv = cls.antiderivative(p, x)
                    cache[k_anti] = cls.to_cache(antidrv)

                # cache partial integration terms
                if terms[0] is None:
                    terms[0] = cls.substitute(antidrv, x, y, l)
                    cache[k_part_l] = cls.to_cache(terms[0])

                if terms[1] is None:
                    terms[1] = cls.substitute(antidrv, x, y, u)
                    cache[k_part_u] = cls.to_cache(terms[1])

            integral = terms[1] - terms[0]
            if not isinstance(integral, Poly):
                integral = Poly(integral, x, y, domain='QQ')
            cache[k_full] = cls.to_cache(integral)

        return integral



    @staticmethod
    def to_cache(p):
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

            t = tuple(sorted(l))
            return t

    @staticmethod
    def from_cache(t, x, y):
        d = dict(t)
        nvars = len(t[0][0])        
        if nvars == 1:
            return Poly.from_dict(d, x, domain='QQ')
        elif nvars == 2:
            return Poly.from_dict(d, x, y, domain='QQ')
        else:
            raise ValueError("Expected univariate or bivariate expression")


class NumMessage(Message):
    """
    Encodes:
        F(x,y) = sum_i P_i(x,y) * exp(alpha_i * x + beta_i * y)
        P(x, y) = sum_j k_j * x^gamma_j * y^theta_j

    as:
    {(alpha_i, beta_i) : {(gamma_j, theta_j) : k_j}}

    """
    PONE = {(0,0) : 1.0}
    ONE = {(0,0) : PONE}


    @staticmethod
    def preprocess_vars(x, y):
        return x, y

    @staticmethod
    def preprocess_piece(l, u, p, x, y):
        def fnode2linear(f):
            A, B = 0.0, 0.0
            if f.is_constant():
                B = f.constant_value()
            elif f.is_plus():
                raise NotImplementedError()


            return (A, B)
                
        return fnode2linear(l), fnode2linear(u), p

    @staticmethod
    def reverse(f):
        return {(j,i) : {(l,k) : f[(i,j)][(k,l)] for k,l in f[(i,j)]}
                for i,j in f}

    @staticmethod
    def is_univariate(f):
        return all([(E[1] == 0) and all([(e[1] == 0) for e in p])
                    for E,p in f.items()])

    @staticmethod
    def polysum(p1, p2):
        psum = {}
        for P in set(p1.keys()).union(set(p2.keys())):
            newk = p1.get(P, 0.0) + p2.get(P, 0.0)
            if newk != 0.0:
                psum[P] = newk

        return psum

    @staticmethod
    def polyprod(p1, p2):
        pprod = {}
        for e1, e2 in product(p1.keys(), p2.keys()):
            eprod = (e1[0]+e2[0], e1[1]+e2[1])
            pprod[eprod] = pprod.get(eprod, 0.0) + p1[e1] * p2[e2]

        return pprod

    @staticmethod
    def polysub(p, s):
        psub = {}
        for e in p:
            if e[0] == 0:
                newk = psub.get(e, 0.0) + p[e]
                if newk != 0:
                    psub[e] = newk
                else:
                    psub.pop(e, None)
            else:
                if s[0] != 0:
                    newe = (0, e[0]+e[1])
                    newk = psub.get(newe, 0.0) + p[e] * (s[0]**e[0])
                    if newk != 0:
                        psub[newe] = newk

                if s[1] != 0:
                    newe = (0, e[1])
                    newk = psub.get(newe, 0.0) + p[e] * (s[1]**e[0])
                    if newk != 0:
                        psub[newe] = newk

        return psub

    @staticmethod
    def polyantider(p):
        panti = {}
        for e in p:
            panti[(e[0]+1, e[1])] = p[e] / (e[0]+1)

        return panti

    @staticmethod
    def polyder(p):
        pder = {}
        for e in p:
            if e[0] > 0:                
                pder[(e[0]-1, e[1])] = p[e] * e[0]

        return pder

    
    @staticmethod
    def polypart(p, k):
        p2 = {pe : pk/k for pe,pk in p.items()}
        pder = NumMessage.polyder(p2)
        if pder != {}:
            p3 = {e : -k for e, k in NumMessage.polypart(pder, k).items()}
            return NumMessage.polysum(p2, p3)
        else:
            return p2
        


    @staticmethod
    def minus(f):
        return {E : {e : -k for e,k in P.items()} for E,P in f.items()}

    @staticmethod
    def sum(f1, f2):
        fsum = {}
        for E in set(f1.keys()).union(set(f2.keys())):
            newpoly = NumMessage.polysum(f1.get(E, {}), f2.get(E, {}))
            if len(newpoly) > 0:
                fsum[E] = newpoly

        return fsum

    @staticmethod
    def substract(f1, f2):
        return NumMessage.sum(f1, NumMessage.minus(f2))

    @staticmethod
    def product(f1, f2):
        fprod = {}
        for E1, E2 in product(f1.keys(), f2.keys()):
            Eprod = (E1[0]+E2[0], E1[1]+E2[1])

            newpoly = NumMessage.polyprod(f1[E1], f2[E2])
            if Eprod not in fprod:
                fprod[Eprod] = newpoly
            else:
                fprod[Eprod] = NumMessage.polysum(fprod[Eprod], newpoly)

        return fprod

    @staticmethod
    def substitute(f, x, y, s):
        fsub = {}
        for E in f:
            newp = NumMessage.polysub(f[E], s)
            if E[0] == 0:
                newE = E
            else:
                newE = (0, E[1]+E[0]*s[0])
                newK = numexp ** (E[0]*s[1])
                newp = {e : k * newK for e,k in newp.items()}

            if newE not in fsub:
                fsub[newE] = newp
            else:
                fsub[newE] = NumMessage.polysum(fsub[newE], newp)

        return fsub

    @staticmethod
    def antiderivative(f, x):
        fanti = {}
        
        for E in f:
            if E[0] == 0:
                fanti[E] = NumMessage.polyantider(f[E])
            else:
                fanti[E] = NumMessage.polypart(f[E], E[0])

        return fanti


if __name__ == '__main__':
    from sympy import integrate, symbols, exp, simplify
    from numpy import isclose

    x, y = symbols("x y")
    def convert(f):
        sym = 0
        for fe, p in f.items():
            symexp = exp(x * fe[0] + y * fe[1])
            sympoly = 0
            for pe, k in p.items():
                sympoly = sympoly + k * x**pe[0] * y**pe[1]

            sym = simplify(sym + sympoly * symexp)

        return sym

    def test_binary(fs):
        tests = [("sum",
                  NumMessage.sum,
                  lambda f1, f2 : f1+f2),
                        
                 ("product",
                  NumMessage.product,
                  lambda f1, f2 : f1*f2),
                        
                 ("sub",
                  lambda f1, f2 : NumMessage.sum(f1, NumMessage.minus(f2)),
                  lambda f1, f2 : f1-f2)]
        
        for label, numop, symop in tests:
            print("========== testing", label)

            for i in range(len(fs)):
                for j in range(i, len(fs)):
                    numres = simplify(convert(numop(fs[i], fs[j])))
                    symres = simplify(symop(convert(fs[i]), convert(fs[j])))
                    if numres.equals(symres):
                        print("---", f"({i},{j})", "---","OK")
                    else:
                        print("---", f"({i},{j})", "---","ERROR")
                        exit()


    def test_substitute(fs):
        for A, B in [(0, 0), (0, 7), (3, 0), (3, 7)]:
            print(f"========== testing substitution with ({A}y + {B})")
            for i in range(len(fs)):
                numres = simplify(convert(NumMessage.substitute(fs[i], None, None, (A, B))))
                symres = simplify(convert(fs[i]).subs({x: (A*y + B)}))

                numres0, symres0 = float(numres.subs({y : 0})), float(symres.subs({y : 0}))
                numres1, symres1 = float(numres.subs({y : 1})), float(symres.subs({y : 1}))
                if isclose(numres0, symres0) and isclose(numres1, symres1):
                    print("---", f"{i}", "---","OK")
                else:
                    print("---", f"{i}", "---","ERROR")
                    exit()

    def test_antiderivative(fs):

        print("========== testing antiderivative")
        for i in range(len(fs)):
            numres = simplify(convert(NumMessage.antiderivative(fs[i], None)))
            symres = simplify(integrate(convert(fs[i]), x))

            if numres.equals(symres):
                print("---", f"{i}", "---","OK")
            else:
                print("---", f"{i}", "---","ERROR")
                exit()





    f1 = {(0,0) : {(0,0) : 2}}

    f2 = {(3,0) : {(0,0) : 2}}

    f3 = {(3,5) : {(0,0) : 2}}

    f4 = {(0,0) : {(1,0) : 3, (0,1) : 5}}

    f5 = {(1,1) : {(1,0) : 3, (0,0): 7}}
    
    f6 = {(1,1) : {(0,1) : 5, (0,0): 11}, (0,0) : {(1,0) : 3, (0,1) : 5}}
    
    fs = [f1, f2, f3, f4, f5, f6, NumMessage.product(f6,NumMessage.sum(f5,f6))]

    for i,f in enumerate(fs):
        print(i, ":", f, "-->", convert(f))


    #test_binary(fs)
    #test_substitute(fs)
    test_antiderivative(fs)
