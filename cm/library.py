r"""Where the files in the model library came from, and what each of them is called.

The library is laid out publisher\repository\file.gguf: LM Studio's layout, and the one
a Hugging Face repository unpacks into. So a path already written in the settings file
names the repository the file was published in, and no second copy of that name has to
be kept in step by hand.

A file whose place is not laid out that way came from somewhere this cannot name. That
is a case rather than an error: a GGUF put there by hand is served exactly like any
other, and only a person can say where it should be fetched from again.

The same layout says what a model is called. A file's name carries the model and the
quantisation of its weights, the directory above it carries the variant the file name
often drops, and both are things a person choosing between two files decides by. So a
name is built from them rather than asked for, and scan writes it into the settings file
where it can then be edited like anything else there.

Nothing is opened here and nothing is asked over the network: a path in, an origin and a
name out, and one answer read for what a repository holds now.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePath
from typing import NewType

HOST = "https://huggingface.co"

# What a model is kept in. Everything else under the library belongs to something
# else, or sits beside a model rather than being one.
SUFFIX = ".gguf"

# How the library lays a model out, as a person reading a message sees it.
LAYOUT = r"publisher\repository\file"

# "unsloth/Qwen3.8-27B-GGUF": publisher and repository, as the project spells them.
Repository = NewType("Repository", str)

# The commit a repository's main branch is at.
Commit = NewType("Commit", str)


@dataclass(frozen=True)
class Published:
    """A file that came from a repository, and which one."""

    repository: Repository


@dataclass(frozen=True)
class Unpublished:
    """A file whose place in the library names no repository."""

    place: PurePath


Origin = Published | Unpublished


@dataclass(frozen=True)
class Revision:
    """What a repository holds now: the commit it is at, and when it last moved.

    The commit is what identifies a build. Size does not: repositories are re-uploaded
    under the same names, and two files of the same length are not the same weights.
    """

    commit: Commit
    when: datetime


@dataclass(frozen=True)
class Unanswered:
    """Nothing could be read about the repository, and what stood in the way."""

    why: str


Reported = Revision | Unanswered


def origin(place: PurePath) -> Origin:
    """Which repository a file came from, out of where it sits in the library.

    Split on both separators. The layout is written with backslashes and read on
    Windows, and where a file came from is not a thing that should change with the
    system this is asked on.
    """
    parts = [part for part in str(place).replace("\\", "/").split("/") if part]

    if len(parts) != 3:
        return Unpublished(place)

    publisher, repository, _ = parts

    return Published(Repository(f"{publisher}/{repository}"))


# ---------------------------------------------------------------------------
# What the library holds, and what each of it is called
# ---------------------------------------------------------------------------

# What a person asks the router for, before a profile's own numbers are added to it.
Key = NewType("Key", str)


@dataclass(frozen=True)
class Held:
    """One model in the library: where its file sits, and what it is called."""

    key: Key
    place: PurePath


# The quantisation is the tail of a file's name: a dynamic-quant prefix where there is
# one, a width, and the block layout under it -- UD-IQ4_XS, Q4_K_M, Q8_0, BF16. What
# comes before it names the model. A file whose name ends in no such tail is named by
# the whole of it, which is a case rather than an error: it is still a model.
_QUANTISED = re.compile(
    r"^(?P<stem>.+?)-(?P<quant>(?:UD-)?(?:IQ|Q|BF|F)[0-9]+(?:_[A-Za-z0-9]+)*)$",
    re.IGNORECASE)

# A model split across volumes. The first names it and llama.cpp opens the rest through
# it, so the others are not models of their own.
_VOLUME = re.compile(r"^(?P<stem>.+)-(?P<volume>[0-9]{5})-of-[0-9]{5}$")
_FIRST = "00001"

# Files that only ever sit beside a model: a vision projector, which nothing here loads,
# and a prediction head published on its own, which this reads out of the model's own
# file instead.
_BESIDE = ("mmproj-", "mtp-")

# What a repository directory is called over and above the model in it.
_PUBLISHED = "-gguf"

# A layout part saying the weights carry no layout at all, which reads as noise in a
# name: Q4_0 is q4 and Q8_0 is q8, while Q4_K_M keeps both of its letters.
_PLAIN = "0"

# How many names a file answers to: the built one, the built one behind its publisher,
# and where it sits, which nothing else can claim.
_DEPTHS = 3


def holds(places: Sequence[PurePath], taken: frozenset[Key]) -> tuple[Held, ...]:
    """Every model the library holds, named, out of where each of its files sits.

    Names are built rather than chosen, so a library that gained a file has one more
    name in it and nothing else moved. Where two files would be called the same, both
    take a longer name that tells them apart rather than one of them quietly winning;
    `taken` is the names already spoken for, which the same rule steps around.
    """
    named = tuple(_named(place) for place in places if _is_a_model(place))

    return tuple(sorted((Held(key, one.place) for one, key in _apart(named, taken).items()),
                        key=lambda held: held.key))


@dataclass(frozen=True)
class _Named:
    """One file, with the names it would answer to, shortest first."""

    place: PurePath
    candidates: tuple[Key, ...]


def _is_a_model(place: PurePath) -> bool:
    """Whether a file in the library is a model, or something that sits beside one."""
    if any(place.name.lower().startswith(beside) for beside in _BESIDE):
        return False

    volume = _VOLUME.match(place.stem)

    return volume is None or volume.group("volume") == _FIRST


def _named(place: PurePath) -> _Named:
    """One file with every name it would answer to, from the plainest outwards."""
    stem, quant = _says(place)
    plain = Key(f"{stem}-{quant}" if quant else stem)

    match origin(place):
        case Published(repository):
            publisher = repository.split("/")[0].lower()
            return _Named(place, (plain, Key(f"{publisher}-{plain}"), _sits(place, plain)))
        case Unpublished():
            return _Named(place, (plain, plain, _sits(place, plain)))


def stem(place: PurePath) -> str:
    """What a file's name calls the model in it, with the quantisation left off.

    Two quantisations of one model share it, and so do two publishers' copies of one
    model, which is what makes it what a recommendation is looked up by: what a model
    is recommended to run with is about the model, not about which repository a file
    was fetched from or how heavily its weights were rounded.
    """
    return _says(place)[0]


def _says(place: PurePath) -> tuple[str, str]:
    """What a file's name calls the model in it, and how its weights are quantised."""
    volume = _VOLUME.match(place.stem)
    whole = volume.group("stem") if volume else place.stem

    quantised = _QUANTISED.match(whole)
    if quantised is None:
        return _stem(place, whole), ""

    return _stem(place, quantised.group("stem")), _quant(quantised.group("quant"))


def _stem(place: PurePath, said: str) -> str:
    """What the model is called, taking the repository's word for it where that says
    more. A repository directory often carries a variant the file name drops -- A3B,
    Instruct, Thinking -- and that variant is part of which model this is."""
    directory = place.parent.name
    published = (directory[:-len(_PUBLISHED)]
                 if directory.lower().endswith(_PUBLISHED) else directory).lower()
    said = said.lower()

    return published if published.startswith(said) and len(published) > len(said) else said


def _quant(tag: str) -> str:
    """Q4_K_M as q4km. The underscores go, and a layout of 0 goes with them: it says
    the weights carry no layout, where K and M say which one they carry."""
    width, *layout = tag.split("_")

    return "".join([width, *(part for part in layout if part != _PLAIN)]).lower()


def _sits(place: PurePath, plain: Key) -> Key:
    """Where a file sits, as a name: the directories above it, and what it is called.

    Two files cannot share a place, so this is where naming stops. The name itself is
    still the built one -- a raw file stem here would spell its quantisation differently
    from every other name.
    """
    return Key("-".join([*(part.lower() for part in place.parent.parts), plain]))


def _apart(named: Sequence[_Named], taken: frozenset[Key]) -> dict[_Named, Key]:
    """One name each: the shortest a file answers to that nothing else does.

    Two files that would be called the same both move on to a longer name. Letting one
    of them keep the short one would mean the name a person types depends on which file
    was walked first.
    """
    chosen: dict[_Named, Key] = {}
    left = list(named)

    for depth in range(_DEPTHS - 1):
        counted: dict[Key, int] = {}
        for one in left:
            counted[one.candidates[depth]] = counted.get(one.candidates[depth], 0) + 1

        spoken = taken | frozenset(chosen.values())
        settled = [one for one in left
                   if counted[one.candidates[depth]] == 1
                   and one.candidates[depth] not in spoken]

        chosen.update((one, one.candidates[depth]) for one in settled)
        left = [one for one in left if one not in chosen]

    # Where it sits, for whatever is still sharing a name. A profile whose name meets
    # another's is caught when the preset is written, and says so there.
    chosen.update((one, one.candidates[-1]) for one in left)

    return chosen


def page_url(published: Published) -> str:
    """The repository's page, for a person to fetch a file from."""
    return f"{HOST}/{published.repository}"


def api_url(published: Published) -> str:
    """Where the project answers what that repository holds now."""
    return f"{HOST}/api/models/{published.repository}"


def revision(answer: object) -> Reported:
    """What a repository holds now, out of the answer to api_url.

    Two fields of the many it carries. An answer shaped any other way says so and is
    not guessed at: a revision reported wrongly is worse than one not reported, because
    it is what a person decides by.
    """
    if not isinstance(answer, Mapping):
        return Unanswered("the answer was not a record")

    commit = answer.get("sha")
    if not isinstance(commit, str) or not commit:
        return Unanswered("the answer named no commit")

    when = answer.get("lastModified")
    if not isinstance(when, str):
        return Unanswered("the answer did not say when it last changed")

    try:
        moved = datetime.fromisoformat(when)
    except ValueError:
        return Unanswered(f"the answer dated it {when!r}, which is not a date")

    return Revision(Commit(commit), moved)
