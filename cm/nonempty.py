"""A sequence with at least one element in it.

A machine has at least one card, an answer from the estimator speaks about at least one
device, and a placement leaves room on at least one. Code reading any of those reads the
first one without asking whether there is one, so the type says so rather than a check
somewhere upstream: it is built from the first element and the rest, and there is no way
to write down an empty one.
"""

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class NonEmpty(Generic[T]):
    """An immutable sequence of one element or more, in the order it was given."""

    __slots__ = ("_items",)

    def __init__(self, first: T, *rest: T) -> None:
        object.__setattr__(self, "_items", (first, *rest))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} cannot be changed")

    @property
    def first(self) -> T:
        return self._items[0]

    @property
    def last(self) -> T:
        return self._items[-1]

    def map(self, change: Callable[[T], U]) -> "NonEmpty[U]":
        first, *rest = (change(one) for one in self._items)
        return NonEmpty(first, *rest)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NonEmpty) and self._items == other._items

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return f"NonEmpty{self._items!r}"
