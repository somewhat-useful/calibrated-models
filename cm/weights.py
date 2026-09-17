"""What a model's own file says about the weights that are never held in memory.

An architecture may mark a tensor as read row by row: llama.cpp maps the file and
fetches only the rows a token asks for, so the tensor never becomes resident, whatever
the load mode says. Both architectures that carry a table of per-layer embeddings do
it, and the table is the larger part of such a file -- 27 GiB of 87 in the one this was
written for.

The estimator counts that table as system memory all the same. Its memory-fit pass
allocates nothing and maps nothing, so the tensor is a host buffer like any other, and
a model ends up refused for memory that is never taken. Nothing in llama.cpp answers
the question either: the loader says so in the log of a real load and nowhere else. So
the file is read here instead.

Bytes in, values out. The last two functions open files and do nothing else.
"""

import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import files
from .units import Bytes, Mib

MAGIC = b"GGUF"

# The tensors llama.cpp reads row by row instead of holding, by the name its loader
# gives them. One name today: the table of per-layer embeddings, marked by the two
# architectures that carry one and by nothing else.
READ_LAZILY = frozenset({"per_layer_token_embd.weight"})

# Below this a marked tensor is held in memory anyway: reading a small one row by row
# costs more than it saves. llama.cpp's own floor, for the mode a machine is left in.
LAZY_FLOOR = Bytes(4 * 1024 ** 3)

# What the settings file writes to make llama.cpp hold such a tensor regardless, and
# what makes the reading below the wrong answer.
HELD = "off"

# What a header is read in. The first file of a model carries its tokenizer and runs to
# megabytes; the rest carry a tensor table and little else. Reading grows until the
# header is whole, and stops before a read that is not a header at all.
FIRST_READ = Bytes(1 << 20)
MOST_READ = Bytes(64 << 20)

# How many bytes a file says its tensor data is aligned to, and what the format says
# when the file says nothing.
_ALIGNMENT = "general.alignment"
_DEFAULT_ALIGNMENT = 32

# How many files a model is stored in, where it is stored in several.
_SPLIT_COUNT = "split.count"

# The types a value in a header may be, as the format numbers them.
(_U8, _I8, _U16, _I16, _U32, _I32, _F32,
 _BOOL, _STRING, _ARRAY, _U64, _I64, _F64) = range(13)

_FIXED = {_U8: "<B", _I8: "<b", _U16: "<H", _I16: "<h", _U32: "<I", _I32: "<i",
          _F32: "<f", _BOOL: "<?", _U64: "<Q", _I64: "<q", _F64: "<d"}

_MIB = 1024 * 1024


class Unreadable(Exception):
    """A file that is not a model file, with the one line saying so."""


class Short(Exception):
    """The bytes in hand stop before the header does. Read more of the file."""


@dataclass(frozen=True)
class Tensor:
    """One tensor of a file: what the loader calls it, and how many bytes it takes.

    The size is the distance to whatever is stored after it, so it carries the padding
    up to the next alignment boundary -- tens of bytes on a tensor of gibibytes. Read
    that way rather than from the shape and the quantisation: what every quantisation
    weighs is llama.cpp's arithmetic and moves with it, while the offsets are the
    file's own.
    """

    name: str
    size: Bytes


@dataclass(frozen=True)
class Header:
    """The front of one file: what it says about itself, and what it carries."""

    said: Mapping[str, object]
    tensors: tuple[Tensor, ...]


def header(blob: bytes, whole: Bytes) -> Header:
    """The header of a file whose first bytes these are and whose size is `whole`."""
    reading = _Reading(blob)
    if reading.take(len(MAGIC)) != MAGIC:
        raise Unreadable(f"not a {MAGIC.decode()} file")

    reading.number(_U32)                      # the format version, which nothing here reads
    count = reading.number(_U64)
    said = _said(reading, reading.number(_U64))

    stated = tuple((reading.string(), _shape(reading), reading.number(_U32),
                    reading.number(_U64)) for _ in range(count))
    alignment = _alignment(said)
    start = reading.at + (-reading.at % alignment)

    return Header(said=said, tensors=_tensors(stated, Bytes(max(0, whole - start))))


def volumes(first: Path, said: Mapping[str, object]) -> tuple[Path, ...]:
    """Every file a model is stored in, the first one named and the rest worked out.

    A settings file names the first volume and llama.cpp opens the others through it,
    so a tensor of the second volume is one nothing has looked at yet.
    """
    count = said.get(_SPLIT_COUNT)
    if not isinstance(count, int) or count <= 1:
        return (first,)

    stem = first.stem[:-len(f"-{1:05d}-of-{count:05d}")]

    return tuple(first.with_name(f"{stem}-{one:05d}-of-{count:05d}{first.suffix}")
                 for one in range(1, count + 1))


def lazily_read(headers: Iterable[Header]) -> Mib:
    """How much of a model llama.cpp reads off the disk instead of holding in memory."""
    taken = sum(one.size for head in headers for one in head.tensors
                if one.name in READ_LAZILY and one.size > LAZY_FLOOR)

    return Mib(taken // _MIB)


def on_demand(path: Path) -> Mib:
    """The same, for the model stored at this path, every volume of it read."""
    first = _head(path)

    return lazily_read([first, *(_head(one) for one in volumes(path, first.said)[1:])])


def _head(path: Path) -> Header:
    """One file's header, reading as much of the file as it takes to have all of it."""
    want = FIRST_READ
    while True:
        blob = files.head(path, want)
        try:
            return header(blob, files.length(path))
        except Short:
            if len(blob) < want or want >= MOST_READ:
                raise Unreadable(f"{path.name}: the header does not end in the first "
                                 f"{len(blob)} bytes")
            want = Bytes(want * 4)


def _tensors(stated: Sequence[tuple[str, Sequence[int], int, int]],
             data: Bytes) -> tuple[Tensor, ...]:
    """What each tensor takes, from where the next one starts and where the file ends."""
    order = sorted(stated, key=lambda one: one[3])
    ends = [one[3] for one in order[1:]] + [data]

    return tuple(Tensor(name=one[0], size=Bytes(max(0, end - one[3])))
                 for one, end in zip(order, ends))


def _said(reading: "_Reading", count: int) -> Mapping[str, object]:
    return {reading.string(): reading.value(reading.number(_U32)) for _ in range(count)}


def _shape(reading: "_Reading") -> tuple[int, ...]:
    return tuple(reading.number(_U64) for _ in range(reading.number(_U32)))


def _alignment(said: Mapping[str, object]) -> int:
    stated = said.get(_ALIGNMENT, _DEFAULT_ALIGNMENT)

    return stated if isinstance(stated, int) and stated > 0 else _DEFAULT_ALIGNMENT


class _Reading:
    """A walk through the bytes in hand, which stops rather than reads past their end."""

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.at = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.at + count > len(self.blob):
            raise Short()
        taken = self.blob[self.at:self.at + count]
        self.at += count

        return taken

    def number(self, kind: int) -> int:
        shape = _FIXED[kind]

        return struct.unpack(shape, self.take(struct.calcsize(shape)))[0]

    def string(self) -> str:
        return self.take(self.number(_U64)).decode("utf-8", "replace")

    def value(self, kind: int) -> object:
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            of = self.number(_U32)
            return tuple(self.value(of) for _ in range(self.number(_U64)))
        if kind not in _FIXED:
            raise Unreadable(f"a header value of an unknown kind: {kind}")

        return self.number(kind)
