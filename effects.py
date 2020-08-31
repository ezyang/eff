from typing import Generator, TypeVar, Any, Callable, cast, Dict, Tuple

# Simple deep algebraic effects implementation in Python, using
# generators.  Don't use this for anything serious: you'll almost
# certainly blow out your stack.
#
# I couldn't think of any way to make use of mypy's Protocol structural
# subtyping support to get some sort of structural subtyping system
# going.  https://dl.acm.org/doi/pdf/10.1145/3276481 has a Java based
# effect system that works by explicitly passing a context object
# around; while this would work, I thought it was pretty inelegant,
# so I just did as Pythoners do and made up some less precise types.

R = TypeVar("R")

Eff = Generator[Tuple[Any, ...], Any, R]
Operation = Callable[..., Any]
# Should take kwarg resume: Callable[..., Eff[R]])
Handler = Callable[..., Eff[R]]
Resume = Callable[..., Eff[R]]

def handler(
    g: Eff[R],
    h: Dict[Operation, Handler[R]],
    r: object = None
) -> Eff[R]:
    try:
        op, args = g.send(r)

        def resume(r: object = None) -> Eff[R]:
            return (yield from handler(g, h, r))

        if op in h:
            return (yield from h[op](*args, resume=resume))
        else:
            return (yield from resume((yield op, args)))
    except StopIteration as e:
        return cast(R, e.value)


def run(g: Eff[R]) -> R:
    try:
        op, args = next(g)
        raise RuntimeError(f'unhandled effect {op}({args})')
    except StopIteration as e:
        return cast(R, e.value)
