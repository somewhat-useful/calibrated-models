"""Reading settings.toml into what the rest of the program works with.

Parsing inward is fallible: every way the file can be wrong ends here, as one message
saying what to fix. Nothing downstream re-checks any of it.

No path is opened and no device is read. Whether model_root exists is a fact about the
machine rather than about the text, so it is checked where the machine is.
"""

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath

from .library import Key
from .lmstudio import Found, Library, Missing
from .machine import GIBIBYTE, Budget, Fitted, Fixed, Share
from . import library, rpc
from .place import (DEFAULT_AMPLE_CTX, DEFAULT_MIN_CTX, DEFAULT_MULTI_GPU_RESERVE,
                    DEFAULT_NO_DESKTOP_RESERVE, DEFAULT_RESERVE, DEFAULT_SLAVE_RESERVE,
                    EVERYTHING, Allowed, CacheType, Endpoint, Worker)
from .recommended import Advised, Recommended, Setting, Unknown
from .serving import (DEFAULT_HOST, DEFAULT_IDLE, DEFAULT_PORT, DEFAULT_RESIDENT, LOGS,
                      Serving)
from .units import Mib, Port, Seconds, Tokens
from .upstream import Cuda
from .workspace import RECOMMENDED

# What a value may be in a preset file. A router flag is a word, a number or a switch.
Value = str | int | float | bool

# Keys calibrate works out and writes itself. A model's own settings may not carry one:
# a placement quietly overridden is exactly the failure this program exists to prevent,
# and it would be invisible -- the router would start, and hold a different window than
# the one the card was measured for.
DERIVED = frozenset({
    "model", "ctx-size", "cache-type-k", "cache-type-v", "gpu-layers", "n-cpu-moe",
    "fit", "fit-target", "spec-type", "spec-draft-n-max", "spec-draft-type-k",
    "spec-draft-type-v", "threads", "threads-batch", "cache-ram",
    "device", "split-mode", "tensor-split", "ubatch-size", "rpc", "override-tensor",
})


@dataclass(frozen=True)
class _Flag:
    """One true/false key an entry may carry, and what it means where it is absent."""

    name: str
    default: bool


# Whether the model is offered at all. A model taken out this way keeps its entry rather
# than losing it, which is what makes the removal stick: scan writes an entry for a file
# that no entry names, so a deleted entry comes back at the next run and a hidden one
# does not.
HIDDEN = _Flag("hidden", False)

# Whether the settings in this entry are the person's own. scan brings an entry's
# sampler values up to date with the recommendations in the repository; one marked this
# way it does not touch, however far the two have drifted apart.
MANUAL = _Flag("manual", False)

# Both are keys of the entry rather than sampler values, and TOML gives a bare key to
# the last table header above it -- so one written under [models."x".settings] is a
# sampler value called `hidden` that hides nothing. Caught rather than ignored.
FLAGS = frozenset({HIDDEN.name, MANUAL.name})

DEFAULT_PRESET = "llamacpp.models.ini"

# Which CUDA build of a release to install, and how many releases to keep once a newer
# one is unpacked. Two leaves the one that was running to fall back to; older ones are
# some 670 MB each.
DEFAULT_CUDA = Cuda("13.3")
DEFAULT_KEPT = 2


class ConfigError(Exception):
    """A settings file that cannot be acted on, with the one line saying why."""


@dataclass(frozen=True)
class Runtime:
    """What every model runs with that moves how much memory a placement needs.

    An estimate asked at one batch size and served at another was asked about a machine
    that does not exist: the compute buffers grow with the batch, and flash attention
    changes what is allocated for attention at all. These four are therefore read out of
    the shared block and passed to the estimator, rather than left at its defaults.
    """

    batch: int
    ubatch: int
    parallel: int
    flash_attn: str


# What llama.cpp itself would use for a shared block that names none of them.
DEFAULT_RUNTIME = Runtime(batch=2048, ubatch=512, parallel=1, flash_attn="auto")


@dataclass(frozen=True)
class Model:
    """One GGUF the router should serve, and everything the file says about it."""

    key: Key
    path: Path
    vendor: Mapping[str, Value]
    allowed: Allowed
    # Whether these settings are the person's own, which scan leaves alone.
    manual: bool


@dataclass(frozen=True)
class NoSlave:
    """The file names no slave: every placement is on this machine's own cards."""


@dataclass(frozen=True)
class Config:
    """The settings file, read."""

    model_root: Path
    cuda: Cuda
    keep_releases: int
    preset_path: Path
    serving: Serving
    reserve: Mib
    min_ctx: Tokens
    ample_ctx: Tokens
    cache_ram: Budget
    runtime: Runtime
    shared: Mapping[str, Value]
    models: tuple[Model, ...]
    # Entries set aside with hidden = true. Nothing is placed or served for them; they
    # are here so scan can see that their files are already named and leave them alone.
    withheld: tuple[Model, ...]
    # What to leave on each card driving a monitor, where the machine has several.
    reserve_multi_gpu: Mib = DEFAULT_MULTI_GPU_RESERVE
    # What to leave on each card driving none, where the machine has several.
    reserve_no_desktop: Mib = DEFAULT_NO_DESKTOP_RESERVE
    # The machine lending its card, where the file names one.
    slave: NoSlave | Worker = NoSlave()


def parse(text: str, library: Library) -> Config:
    """The settings file as text, read into the settings file as values.

    The library is where this machine keeps its weights, found before this was called. A
    settings file naming model_root does not use it; one that leaves it out is asking
    for the machine's own library, which is why it has to be here rather than looked up
    later -- every model's path is built from it.

    A TOMLDecodeError passes through: its message already says which line and why, and
    nothing here can say it better.
    """
    raw = tomllib.loads(text)

    model_root = _model_root(raw, library)

    shared = dict(raw.get("shared", {}))

    # A settings file naming no model is a machine that has not run scan yet, not a
    # machine that is wrong. What needs a model says so where it needs one.
    entries = raw.get("models", {})
    if not isinstance(entries, dict):
        raise ConfigError('models must be a table: one [models."name"] entry per model')

    read = tuple(_entry(key, entry, model_root) for key, entry in entries.items())

    return Config(
        model_root=model_root,
        cuda=Cuda(_text(raw, "cuda_version", DEFAULT_CUDA.version)),
        keep_releases=_whole(raw, "keep_releases", DEFAULT_KEPT),
        preset_path=Path(str(raw.get("preset_path", DEFAULT_PRESET))),
        serving=_serving(raw),
        reserve=Mib(_whole(raw, "reserve_mib", DEFAULT_RESERVE)),
        min_ctx=Tokens(_whole(raw, "min_ctx_tokens", DEFAULT_MIN_CTX)),
        ample_ctx=Tokens(_whole(raw, "ample_ctx_tokens", DEFAULT_AMPLE_CTX)),
        cache_ram=_budget(raw),
        runtime=_runtime(shared),
        shared=shared,
        models=tuple(one.model for one in read if not one.hidden),
        withheld=tuple(one.model for one in read if one.hidden),
        reserve_multi_gpu=Mib(_whole(raw, "reserve_multi_gpu_mib",
                                     DEFAULT_MULTI_GPU_RESERVE)),
        reserve_no_desktop=Mib(_whole(raw, "reserve_no_desktop_mib",
                                      DEFAULT_NO_DESKTOP_RESERVE)),
        slave=_slave(raw),
    )


@dataclass(frozen=True)
class Lending:
    """What a machine lending its card reads from its settings file: which releases it
    installs and keeps, and where its worker writes."""

    cuda: Cuda
    keep_releases: int
    logs: Path


def lending(text: str) -> Lending:
    """The settings text as a slave reads it.

    None of it has to be there, and neither does the file: a machine lending its card
    names no model and keeps no library, and every key a slave reads has a default.
    """
    raw = tomllib.loads(text)

    return Lending(cuda=Cuda(_text(raw, "cuda_version", DEFAULT_CUDA.version)),
                   keep_releases=_whole(raw, "keep_releases", DEFAULT_KEPT),
                   logs=_logs(raw))


def naming_the_library(text: str, models: Path) -> str:
    """The settings text, with model_root naming this directory.

    Where LM Studio is not installed there is nothing on the machine that says where
    the models are, and install asks. This is where the answer goes: into the copy it
    just made, rather than being held for that one run and asked for again by the next
    command.

    The key replaces the one the template carries commented out, wherever the template
    puts it, and is prepended where there is none -- a top-level key after the first
    table header would belong to that table.

    Written as a basic string with the escapes JSON uses, which TOML reads the same way.
    A Windows path is full of backslashes, and TOML's literal strings, which the
    template uses, cannot hold an apostrophe -- which a directory under somebody's name
    may well have in it.
    """
    written = f"model_root = {json.dumps(str(models))}"
    lines = text.splitlines()

    for index, line in enumerate(lines):
        if line.lstrip().removeprefix("#").lstrip().startswith("model_root"):
            lines[index] = written
            return "\n".join(lines) + "\n"

    return f"{written}\n{text}"


def unnamed(named: Sequence[Model], places: Sequence[PurePath],
            model_root: Path) -> tuple[PurePath, ...]:
    """The files in the library that no entry in the settings file names.

    Matched on the file rather than on the name, so an entry that was renamed is still
    the entry for its file. Compared without case, because this is a Windows library
    and a file written two ways there is one file.

    Names are worked out for these and not for the rest, so a file already named cannot
    push the name it holds away from itself.
    """
    already = frozenset(str(one.path).lower() for one in named)

    return tuple(place for place in places
                 if str(model_root / place).lower() not in already)


# What an entry runs on where nothing in the repository covers its model. Three of these
# say a penalty is off -- code repeats itself, and a long chain of thought revisits the
# same identifiers -- and the rest are what most publishers ask for. They are a floor
# rather than a recommendation, and the block says so where they are used.
#
# Written out even where llama.cpp would arrive at the same number by itself. Three of
# them it would not -- it defaults to temp 0.8, top-k 40 and min-p 0.05 -- and a value
# that is only a default is one nobody can see in this file to disagree with.
NEUTRAL: Mapping[str, Setting] = {
    "min-p": "0",
    "presence-penalty": "0",
    "repeat-penalty": "1.0",
    "temp": "1.0",
    "top-k": "20",
    "top-p": "0.95",
}


@dataclass(frozen=True)
class Writing:
    """One entry as scan means it to stand: the file it names, and what says how to run
    it -- a row in the repository, or nothing yet written for this model."""

    key: Key
    place: PurePath
    advice: Advised


@dataclass(frozen=True)
class Untouched:
    """One entry left exactly as it stands, and the one line saying why.

    Half-editing somebody's file is worse than not editing it, so an entry written in a
    way this cannot rewrite is not rewritten -- and not reported as brought up to date
    either, which is what this is for.
    """

    key: Key
    why: str


@dataclass(frozen=True)
class Retuning:
    """The settings text after every entry that could be brought up to date was, and
    the ones that could not."""

    text: str
    untouched: tuple[Untouched, ...]


@dataclass(frozen=True)
class Rename:
    """One entry's key as it stands, and as the library names the file it holds."""

    was: Key
    now: Key


@dataclass(frozen=True)
class Renaming:
    """The settings text after every entry that could be renamed was, and the ones that
    could not."""

    text: str
    untouched: tuple[Untouched, ...]


def naming(text: str, adding: Sequence[Writing]) -> str:
    """The settings text with an entry appended for each of these models.

    Appended rather than rewritten. Everything above is the person's file -- their
    comments, their order, their own numbers -- and reproducing it through a writer
    would be a second parser to keep in step with the reader.

    Both the key and the file are quoted. A derived name carries the dots of a version
    number, which a bare key would read as one table nested inside another.
    """
    if not adding:
        return text

    return "{}\n\n\n{}".format(text.rstrip("\n"),
                               "\n".join(_entry_written(one) for one in adding))


def retuning(text: str, updating: Sequence[Writing]) -> Retuning:
    """The settings text with each of these entries' settings block brought up to date.

    Everything between that block's header and whatever follows it belongs to scan and
    is replaced, the provenance comment included: a block saying where its numbers came
    from is worth having only if it cannot go stale. A note written above the header is
    the person's and survives, and an entry marked manual never reaches here at all.

    An entry that cannot be found, or that writes its settings in a way this does not
    recognise, comes back untouched and named. Rewriting it blind is how a settings
    file ends up saying one thing twice and then reading as nothing at all, and scan
    reports what it did rather than what it meant to.
    """
    untouched = []

    for one in updating:
        match _retuned(text, one):
            case _Retuned(written):
                text = written
            case Untouched(_, _) as left:
                untouched.append(left)

    return Retuning(text=text, untouched=tuple(untouched))


def renames(named: Sequence[Model], model_root: Path) -> tuple[Rename, ...]:
    """Every entry whose key is not the name the library builds for the file it holds.

    Built for the whole set at once, so that two files which would be called the same
    both take a longer name. An entry marked manual is left out of it, and so is one
    whose file sits outside the library or is not a model at all: their keys stay as
    they are, and are handed to the naming as spoken for so that no built name lands on
    one of them.
    """
    keyed: dict[PurePath, Key] = {}
    taken = []

    for one in named:
        place = _under(model_root, one)
        if one.manual or place is None:
            taken.append(one.key)
            continue
        keyed[place] = one.key

    held = library.holds(tuple(keyed), frozenset(taken))
    # A file the library does not count as a model -- a projector, a volume after the
    # first -- is named by nothing, so its entry keeps its key, and the second pass is
    # what tells the naming to step around it.
    kept = frozenset(one.place for one in held)
    passed = frozenset(key for place, key in keyed.items() if place not in kept)
    if passed:
        held = library.holds(tuple(keyed), frozenset(taken) | passed)

    return tuple(Rename(was=keyed[one.place], now=one.key)
                 for one in held if keyed[one.place] != one.key)


def _under(model_root: Path, model: Model) -> PurePath | None:
    """Where a model's file sits under the library, or nothing where it sits elsewhere.

    A file kept somewhere else has no place for the naming rule to read a name out of,
    so its entry keeps the key it has.
    """
    try:
        return PurePath(model.path).relative_to(model_root)
    except ValueError:
        return None


def renaming(text: str, renames: Sequence[Rename]) -> Renaming:
    """The settings text with each of these entries keyed as the library names its file.

    Two lines of an entry change and nothing else: its own header and its settings
    block's. What a person wrote under either of them is theirs and stays where it is,
    and so does the file the entry names -- a rename says what a model is called here,
    never which file that is.

    A name another entry still holds is waited for rather than written over: the entry
    holding it may be about to move off it, and two entries under one key is a file TOML
    reads as nothing at all. So renames are applied in rounds, each round writing the
    ones whose name is free, until a round writes none -- the rest are then holding one
    another's names in a ring, and every one of them is left alone and named.

    An entry this cannot find, one that opens its settings twice, or one carrying a
    table of its own besides them is left alone and named as well. A key rewritten in
    one header and left in another declares two models where there was one.
    """
    untouched = []
    left = list(renames)

    while left:
        waiting, written = [], False
        for one in left:
            match _renamed(text, one):
                case _Renamed(said):
                    text, written = said, True
                case _Held():
                    waiting.append(one)
                case Untouched(_, _) as refused:
                    untouched.append(refused)

        if not written:
            untouched.extend(Untouched(one.was, f"{one.now} is another entry's key")
                             for one in waiting)
            break

        left = waiting

    return Renaming(text=text, untouched=tuple(untouched))


def _entry_written(writing: Writing) -> str:
    return (f"[models.{_quoted(writing.key)}]\n"
            f"file = {_quoted(str(writing.place))}\n"
            f"\n{_block(writing)}")


def _block(writing: Writing) -> str:
    """The settings sub-table: its header, where the numbers came from, and them."""
    values = _values(writing.advice)
    widest = max(len(name) for name in values)

    return "".join([f"[models.{_quoted(writing.key)}.settings]\n",
                    _provenance(writing.advice),
                    *(f"{name:<{widest}} = {_quoted(value)}\n"
                      for name, value in sorted(values.items()))])


def _values(advice: Advised) -> Mapping[str, Setting]:
    """What the block runs on: the floor, with what the model's own page says over it.

    A page states the values it has an opinion about and says nothing about the rest.
    Saying nothing is not the same as recommending llama.cpp's own numbers -- min-p 0.05
    is nobody's choice here -- so the floor fills in whatever the page leaves open, and
    anything the page names wins.
    """
    match advice:
        case Recommended(settings, _, _):
            return {**NEUTRAL, **settings}
        case Unknown(_):
            return NEUTRAL


def _provenance(advice: Advised) -> str:
    """Where this block's numbers came from, said inside the block.

    Inside rather than above it, so that the whole of what scan wrote is one span it can
    replace: a comment above the header would be either stale or somebody's own note
    overwritten, and there is no telling which from the text.
    """
    match advice:
        case Recommended(_, source, pattern):
            said = (f"# {pattern}, over neutral defaults for whatever it leaves open:\n"
                    f"# {source}\n")
        case Unknown(stem):
            said = (f"# Nothing in {RECOMMENDED} covers {stem}, so these are neutral "
                    "values rather than\n# anybody's recommendation.\n")

    return said + ("# Change any of it here; manual = true above keeps scan out of "
                   "this block for good.\n")


# The sub-table an entry's sampler values are written under, as the file spells it.
_SETTINGS_KEY = "settings"

# A model's own table, and the settings sub-table under it. A key may be written in
# either kind of quotes or, where it holds nothing that needs them, in none.
_KEY = r"(?P<key>'[^']*'|\"[^\"]*\"|[A-Za-z0-9_-]+)"
_ENTRY = re.compile(rf"^\[models\.{_KEY}\]\s*$")
_SETTINGS = re.compile(rf"^\[models\.{_KEY}\.{_SETTINGS_KEY}\]\s*$")
_UNDER = re.compile(rf"^\[models\.{_KEY}\.")
_TABLE = re.compile(r"^\[")

# A key given a value, either outright or through a dot. Both are ways of writing a
# sub-table that no header opens.
_ASSIGNED = re.compile(rf"^\s*{_KEY}\s*[.=]")

# A line that says nothing itself. A run of them at the end of a table introduces
# whatever comes after it rather than belonging to the table they follow.
_INTRODUCES = re.compile(r"^\s*(#|$)")


@dataclass(frozen=True)
class _Retuned:
    """The whole text, with this one entry's settings block as scan means it."""

    text: str


def _retuned(text: str, writing: Writing) -> _Retuned | Untouched:
    lines = text.splitlines()
    key = writing.key

    entry = _headers(lines, _ENTRY, key)
    if len(entry) != 1:
        return Untouched(key, f'no one [models."{key}"] line to write under')

    opened = entry[0]
    closed = _closed(lines, opened, key)
    blocks = _headers(lines, _SETTINGS, key)
    inside = tuple(index for index in blocks if opened < index < closed)

    if len(inside) != len(blocks):
        return Untouched(key, "its settings block is not under its entry")

    if _otherwise(lines, range(opened + 1, closed), key, inside):
        return Untouched(key, "its settings are written in a form scan cannot rewrite")

    written = _block(writing).splitlines()

    match inside:
        case ():
            ends = _before(lines, range(opened + 1, closed))
            return _Retuned("\n".join([*lines[:ends], "", *written,
                                       *lines[ends:]]) + "\n")
        case (block,):
            ends = _before(lines, range(block + 1, closed))
            return _Retuned("\n".join([*lines[:block], *written,
                                       *lines[ends:]]) + "\n")
        case _:
            return Untouched(key, "it opens its settings block more than once")


@dataclass(frozen=True)
class _Renamed:
    """The whole text, with one entry keyed as the library names its file."""

    text: str


@dataclass(frozen=True)
class _Held:
    """The name this rename takes is another entry's, and may yet come free: that entry
    may itself be waiting to be renamed off it."""


def _renamed(text: str, rename: Rename) -> _Renamed | _Held | Untouched:
    lines = text.splitlines()

    if _headers(lines, _ENTRY, rename.now):
        return _Held()

    entry = _headers(lines, _ENTRY, rename.was)
    if len(entry) != 1:
        return Untouched(rename.was, f'no one [models."{rename.was}"] line to rename')

    blocks = _headers(lines, _SETTINGS, rename.was)
    if len(blocks) > 1:
        return Untouched(rename.was, "it opens its settings block more than once")

    under = tuple(index for index, line in enumerate(lines)
                  if (said := _UNDER.match(line)) is not None
                  and _unquoted(said.group("key")) == rename.was)
    if any(index not in blocks for index in under):
        return Untouched(rename.was, "it carries a table besides its settings")

    written = list(lines)
    written[entry[0]] = f"[models.{_quoted(rename.now)}]"
    for block in blocks:
        written[block] = f"[models.{_quoted(rename.now)}.{_SETTINGS_KEY}]"

    return _Renamed("\n".join(written) + "\n")


def _headers(lines: Sequence[str], header: re.Pattern[str], key: Key) -> tuple[int, ...]:
    """Every line in these that opens this key's table, written as a header this
    recognises.

    All of them rather than the first. None and more than one are both answers about
    the file rather than the absence of one, and each of them is a reason to leave the
    entry alone.
    """
    return tuple(index for index, line in enumerate(lines)
                 if (said := header.match(line)) is not None
                 and _unquoted(said.group("key")) == key)


def _otherwise(lines: Sequence[str], span: range, key: Key,
               inside: Sequence[int]) -> bool:
    """Whether the entry says something about its settings that this cannot rewrite.

    Rewriting works by replacing the run of lines under one recognised header. An entry
    that writes its settings some other way -- as an inline table, as dotted keys, or
    under a header carrying a comment after it, which is a header the pattern does not
    match -- would keep every one of them and be handed a second block besides. TOML
    reads a table declared twice as no file at all, so the whole of the person's
    settings would stop being readable to fix what one entry runs with.
    """
    for index in span:
        if index in inside:
            continue

        under = _UNDER.match(lines[index])
        if under is not None and _unquoted(under.group("key")) == key:
            return True

        assigned = _ASSIGNED.match(lines[index])
        if assigned is not None and _unquoted(assigned.group("key")) == _SETTINGS_KEY:
            return True

    return False


def _closed(lines: Sequence[str], opened: int, key: Key) -> int:
    """Where this model's entry ends: the next header that is not a table under it.

    Matched on the key rather than on the text of the header, because a file may quote
    a key either way and both name the same model.
    """
    for index in range(opened + 1, len(lines)):
        if not _TABLE.match(lines[index]):
            continue

        under = _UNDER.match(lines[index])
        if under is None or _unquoted(under.group("key")) != key:
            return index

    return len(lines)


def _before(lines: Sequence[str], span: range) -> int:
    """Where a span ends once what introduces whatever follows it is given back.

    The blank lines and comments at the end of a table are not that table's: a note
    written above the next entry sits there, and rewriting the table would take it with
    it. So a span this program replaces ends at the last line that says something.
    """
    ends = span.stop
    while ends > span.start and _INTRODUCES.match(lines[ends - 1]):
        ends -= 1

    return ends


def _unquoted(said: str) -> str:
    if said.startswith("'"):
        return said[1:-1]
    if said.startswith('"'):
        return json.loads(said)

    return said


def _quoted(value: str) -> str:
    """A TOML string, literal wherever it can be.

    A Windows path is full of backslashes and a literal string keeps them as they are,
    which is what someone reading this file expects to see. A value carrying an
    apostrophe cannot be written that way at all, and takes the basic string instead,
    whose escapes are JSON's.
    """
    return f"'{value}'" if "'" not in value else json.dumps(value)


def _serving(raw: Mapping[str, object]) -> Serving:
    """How the router is to be run. Every part of it has a defensible default: this is
    a machine serving one card, and nothing about it is particular to a machine."""
    port = _whole(raw, "port", DEFAULT_PORT)
    if not 1 <= port <= 65535:
        raise ConfigError("port must be between 1 and 65535")

    resident = _whole(raw, "models_max", DEFAULT_RESIDENT)
    if resident < 1:
        raise ConfigError("models_max must be at least 1, or the router may hold no "
                          "model and serves nothing")

    idle = _whole(raw, "sleep_idle_seconds", DEFAULT_IDLE)
    if idle < 0:
        raise ConfigError("sleep_idle_seconds cannot be negative")

    return Serving(host=_text(raw, "listen_host", DEFAULT_HOST),
                   port=Port(port),
                   resident=resident,
                   idle=Seconds(idle),
                   logs=_logs(raw))


def _logs(raw: Mapping[str, object]) -> Path:
    """Where the router writes. Beside this file unless the file says otherwise, and
    relative like everything else it names: the logs belong to the working directory
    rather than to the release that happened to write them, which is what lets a
    release be replaced without taking them with it."""
    named = raw.get("log_dir")
    if isinstance(named, str) and named.strip():
        return Path(named.strip())

    return Path(LOGS)


def _runtime(shared: Mapping[str, Value]) -> Runtime:
    """The shared flags the estimator has to be told about, or llama.cpp's own."""
    return Runtime(batch=_count(shared, "batch-size", DEFAULT_RUNTIME.batch),
                   ubatch=_count(shared, "ubatch-size", DEFAULT_RUNTIME.ubatch),
                   parallel=_count(shared, "parallel", DEFAULT_RUNTIME.parallel),
                   flash_attn=str(shared.get("flash-attn",
                                             DEFAULT_RUNTIME.flash_attn)))


def _count(shared: Mapping[str, Value], key: str, fallback: int) -> int:
    value = shared.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"shared: {key} must be a whole number above zero")
    return value


def _text(raw: Mapping[str, object], key: str, fallback: str) -> str:
    value = raw.get(key, fallback)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be written in quotes, and not left empty")
    return value.strip()


def _model_root(raw: Mapping[str, object], library: Library) -> Path:
    """Where the weights are: what the file says, or this machine's own library.

    Leaving the key out is the ordinary case rather than the fallback. LM Studio is how
    these files get onto a machine, it records where it put them, and a settings file
    that repeats that path is a settings file that goes stale when the library moves.
    """
    named = raw.get("model_root")
    if isinstance(named, str) and named.strip():
        return Path(named)

    match library:
        case Found(root):
            return root
        case Missing(looked):
            raise ConfigError(
                f"model_root is not set and there is no LM Studio library at {looked}. "
                "Set model_root to the directory the weights are under.")


# How a size may be written: 32, 32G, 32Gb and 32GiB are all gibibytes, 50% is a share
# of the installed memory. The unit is optional because a bare number is what a person
# writes first, and gibibytes is the only unit anyone means at this size.
_SIZE = re.compile(r"^(\d+)\s*(%|g|gb|gib)?$", re.IGNORECASE)

_FORMS = "cache_ram: write it as 32, '32G', '32Gb' or '50%'"


def _budget(raw: Mapping[str, object]) -> Budget:
    """What the settings file allows the prompt cache, or that it should be worked out.

    An absent key is not a missing value: it says the machine should decide, which is
    what it can do better than a person -- it knows what the placements leave.
    """
    if "cache_ram" not in raw:
        return Fitted()

    written = raw["cache_ram"]
    if isinstance(written, str):
        return _written(written)
    if isinstance(written, int) and not isinstance(written, bool):
        return Fixed(Mib(written * GIBIBYTE))

    raise ConfigError(_FORMS)


def _written(text: str) -> Budget:
    read = _SIZE.match(text.strip())
    if read is None:
        raise ConfigError(_FORMS)

    amount, unit = int(read[1]), (read[2] or "").lower()
    if unit != "%":
        return Fixed(Mib(amount * GIBIBYTE))

    if amount > 100:
        raise ConfigError("cache_ram: a share over 100% is more memory than there is")
    return Share(amount)


SLAVE = "slave"

_SLAVE_MEMORY = "slave: memory is the size of its card in gibibytes: 12, '12G' or '12GiB'"


def _slave(raw: Mapping[str, object]) -> NoSlave | Worker:
    """The machine lending its card, where the file names one.

    The address and the memory have to be written; what is left on the card has a
    default. The memory is a person's round figure for that card, since nothing on this
    machine can read it.
    """
    if SLAVE not in raw:
        return NoSlave()

    table = raw[SLAVE]
    if not isinstance(table, dict):
        raise ConfigError("slave must be a table: [slave], with address = and memory = "
                          "under it")

    said = table.get("address")
    if not isinstance(said, str) or not said.strip():
        raise ConfigError("slave: address is not set")

    match rpc.endpoint(said, rpc.DEFAULT_PORT):
        case rpc.Unreadable(why):
            raise ConfigError(f"slave: address {why}")
        case Endpoint() as reached:
            pass

    reserve = table.get("reserve_mib", DEFAULT_SLAVE_RESERVE)
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise ConfigError("slave: reserve_mib must be a whole number")

    return Worker(endpoint=reached, memory=gibibytes(table.get("memory")),
                  reserve=Mib(reserve))


def gibibytes(size: object) -> Mib:
    """A slave's memory as a person writes it: 12, '12G', '12Gb' or '12GiB'."""
    if isinstance(size, int) and not isinstance(size, bool) and size > 0:
        return Mib(size * GIBIBYTE)

    if isinstance(size, str):
        read = _SIZE.match(size.strip())
        if read is not None and read[2] != "%" and int(read[1]) > 0:
            return Mib(int(read[1]) * GIBIBYTE)

    raise ConfigError(_SLAVE_MEMORY)


# The slave's own table, and the slave written any way at all: as that table, as a table
# under it, as a key given a value, or through a dot.
_SLAVE_TABLE = re.compile(r"^\[slave\]\s*$")
_SLAVE_ANYHOW = re.compile(r"^\s*(\[slave[.\]\s]|slave\s*[.=])")


def slaved(text: str, worker: Worker) -> str:
    """The settings text naming this slave, in place of any it named before.

    Appended at the end: a table can stand anywhere after the keys at the top, and the
    end is the one place that is never inside somebody's own table.
    """
    block = ["[slave]",
             f"address     = {_quoted(rpc.written(worker.endpoint))}",
             f"memory      = {_quoted(f'{worker.memory // GIBIBYTE}G')}",
             f"reserve_mib = {worker.reserve}"]

    return "\n".join([unslaved(text).rstrip("\n"), "", "", *block]) + "\n"


def unslaved(text: str) -> str:
    """The settings text naming no slave.

    A slave written in a way this cannot take out whole -- inline, dotted, or opened more
    than once -- is refused rather than half removed: TOML reads a table declared twice
    as no file at all. What introduces whatever follows the table stays.
    """
    lines = text.splitlines()
    opened = tuple(index for index, line in enumerate(lines) if _SLAVE_TABLE.match(line))
    anyhow = tuple(index for index, line in enumerate(lines) if _SLAVE_ANYHOW.match(line))

    if anyhow != opened or len(opened) > 1:
        raise ConfigError("the settings file names its slave in a form that cannot be "
                          "rewritten; edit its [slave] by hand")
    if not opened:
        return text

    start = opened[0]
    closed = next((index for index in range(start + 1, len(lines))
                   if _TABLE.match(lines[index])), len(lines))
    ends = _before(lines, range(start + 1, closed))
    while start > 0 and not lines[start - 1].strip():
        start -= 1

    return "\n".join([*lines[:start], *lines[ends:]]) + "\n"


def _whole(raw: Mapping[str, object], key: str, fallback: int) -> int:
    value = raw.get(key, fallback)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be a whole number")
    return value


@dataclass(frozen=True)
class _Entry:
    """One [models] entry, read: the model it names, and whether it is hidden."""

    model: Model
    hidden: bool


def _entry(key: str, entry: object, model_root: Path) -> _Entry:
    if not isinstance(entry, dict):
        raise ConfigError(
            f'{key}: an entry is a table -- [models."{key}"] with file = under it, '
            f"rather than {key} given a value of its own")

    return _Entry(_model(key, entry, model_root), _flag(key, entry, HIDDEN))


def _flag(key: str, entry: Mapping[str, object], flag: _Flag) -> bool:
    value = entry.get(flag.name, flag.default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key}: {flag.name} must be true or false")

    return value


def _model(key: str, entry: Mapping[str, object], model_root: Path) -> Model:
    file = entry.get("file")
    if not isinstance(file, str) or not file:
        raise ConfigError(f"{key}: file is not set")

    vendor = entry.get("settings", {})
    if not isinstance(vendor, dict):
        raise ConfigError(f"{key}: settings must be a table")
    for name in vendor:
        if name in DERIVED:
            raise ConfigError(f"{key}: {name} is derived; remove it")
        if name in FLAGS:
            raise ConfigError(
                f"{key}: {name} is a flag of the entry and not a sampler value, but it "
                f'is written under [models."{key}".settings], where it does nothing. '
                "Move it above that line.")

    return Model(key=Key(key),
                 path=model_root / file,
                 vendor=dict(vendor),
                 allowed=_allowed(key, entry),
                 manual=_flag(key, entry, MANUAL))


def _allowed(key: str, entry: Mapping[str, object]) -> Allowed:
    """What the person permits for this file. Absent keys permit what the file can do."""
    listed = entry.get("cache")
    if listed is None:
        caches = EVERYTHING.caches
    else:
        if not isinstance(listed, list) or not listed:
            raise ConfigError(f"{key}: cache is empty")
        caches = frozenset(_cache(key, one) for one in listed)

    head = entry.get("mtp", EVERYTHING.head)
    if not isinstance(head, bool):
        raise ConfigError(f"{key}: mtp must be true or false")

    return Allowed(caches=caches, head=head)


def _cache(key: str, name: object) -> CacheType:
    try:
        return CacheType(name)
    except ValueError:
        raise ConfigError(f"{key}: unknown cache {name}") from None
