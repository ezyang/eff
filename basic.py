from eff import handler, run, Eff
from typing import Callable, TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")

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
