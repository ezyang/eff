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

T = TypeVar("T")
R = TypeVar("R")

Eff = Generator[Tuple[Any, ...], Any, R]
Operation = Callable[..., Any]
# Should take kwarg resume: Callable[..., Eff[R]])
Handler = Callable[..., Eff[R]]

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
            r = yield op, args
            return (yield from resume(r))
    except StopIteration as e:
        return cast(R, e.value)


def run(g: Eff[R]) -> R:
    try:
        op, args = next(g)
        raise RuntimeError(f'unhandled effect {op}({args})')
    except StopIteration as e:
        return cast(R, e.value)


# Some basic test cases

def output(arg: int) -> Eff[None]:
    return cast(None, (yield (output, (arg,))))


def input() -> Eff[int]:
    return cast(int, (yield (input, ())))


def sample() -> Eff[None]:
    i = yield from input()
    yield from output(i)
    yield from output(i + 1)
    yield from output(i + 2)


def output_handler(arg: int, *, resume: Callable[[], Eff[R]]) -> Eff[R]:
    print(arg)
    # NB: this form is inefficient, because we always push a stack
    # frame even when it's not necessary
    return (yield from resume())


def input_handler(*, resume: Callable[[int], Eff[R]]) -> Eff[R]:
    return (yield from resume(3))


run(
    handler(
        sample(),
        {input: input_handler,
         output: output_handler
         }
    )
)

run(
    handler(
        handler(
            sample(),
            {output: output_handler}
        ),
        {input: input_handler}
    )
)


def sample2() -> Eff[int]:
    yield from error()
    print("omitted")
    return 2


def error() -> Eff[T]:
    return cast(T, (yield (error, ())))


def error_handler(*, resume: Callable[[int], Eff[R]]) -> Eff[int]:
    # ignore resume
    yield from []
    return 2


run(handler(sample2(), {error: error_handler}))


# Reverse mode AD


