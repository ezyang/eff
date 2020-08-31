from eff import handler, run, Eff, Resume
from dataclasses import dataclass
from typing import TypeVar, Callable, cast, Generic, Any

X = TypeVar('X')
R = TypeVar('R')

# This is a port of some Frank code from Jesse Sigal

def const(f: float) -> Eff[X]:
    return cast(X, (yield (const, (f,))))

def negate(x: X) -> Eff[X]:
    return cast(X, (yield (negate, (x,))))

def plus(x: X, y: X) -> Eff[X]:
    return cast(X, (yield (plus, (x, y))))

def times(x: X, y: X) -> Eff[X]:
    return cast(X, (yield (times, (x, y))))

def term(x: X, y: X) -> Eff[X]:
    # 1 + x ** 3 - y ** 2
    return (yield from plus(
        (yield from const(1)),
        (yield from plus(
            (yield from times(
                (yield from times(x, x)),
                x
            )),
            (yield from negate(
                (yield from times(y, y))
            ))
        ))))

def evaluate(g: Eff[R]) -> R:
    def h_const(f: float, *, resume: Resume[R]) -> Eff[R]:
        return (yield from resume(f))
    def h_negate(x: float, *, resume: Resume[R]) -> Eff[R]:
        return (yield from resume(-x))
    def h_plus(x: float, y: float, *, resume: Resume[R]) -> Eff[R]:
        return (yield from resume(x + y))
    def h_times(x: float, y: float, *, resume: Resume[R]) -> Eff[R]:
        return (yield from resume(x * y))
    return run(handler(g, {
        const: h_const,
        negate: h_negate,
        plus: h_plus,
        times: h_times,
    }))

print(evaluate(term(2, 4)))

# TODO: higher order differentiation by making grad a Variable
@dataclass
class Variable(Generic[X]):
    data: X
    grad: X  # mutable

def reversep(g: Eff[None]) -> Eff[None]:
    def h_const(f: float, *, resume: Resume[None]) -> Eff[None]:
        # Ordinarily, this would come from the effect type, but we don't
        # have one, so just get out of jail free
        r: Variable[Any] = Variable((yield from const(f)), (yield from const(0)))
        yield from resume(r)
    def h_negate(x: Variable[X], *, resume: Resume[None]) -> Eff[None]:
        r = Variable((yield from negate(x.data)), (yield from const(0)))
        yield from resume(r)
        x.grad = yield from plus(x.grad, r.grad)
    def h_plus(x: Variable[X], y: Variable[X], *, resume: Resume[None]) -> Eff[None]:
        r = Variable((yield from plus(x.data, y.data)), (yield from const(0)))
        yield from resume(r)
        x.grad = yield from plus(x.grad, r.grad)
        y.grad = yield from plus(y.grad, r.grad)
    def h_times(x: Variable[X], y: Variable[X], *, resume: Resume[None]) -> Eff[None]:
        r = Variable((yield from times(x.data, y.data)), (yield from const(0)))
        yield from resume(r)
        x.grad = yield from plus(x.grad, (yield from times(y.data, r.grad)))
        y.grad = yield from plus(y.grad, (yield from times(x.data, r.grad)))
    return handler(g, {
        const: h_const,
        negate: h_negate,
        plus: h_plus,
        times: h_times,
    })

def grad(f: Callable[[Variable[X]], Eff[Variable[X]]], x: X) -> Eff[X]:
    z = Variable(x, (yield from const(0)))
    c1: X
    c1 = yield from const(1)
    def go() -> Eff[None]:
        (yield from f(z)).grad = c1
    yield from reversep(go())
    return z.grad

def cube(x: X) -> Eff[X]:
    return (yield from times(
        (yield from times(x, x)),
        x
    ))

print(evaluate(grad(cube, 2)))

def t1_body() -> Eff[float]:
    def t1_y(y: Variable[float]) -> Eff[Variable[float]]:
        return (yield from grad(cube, y))
    return (yield from grad(t1_y, float(3)))

print(evaluate(t1_body()))
