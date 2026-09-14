"""Where a model sits on this machine's cards.

Nothing here starts a process or reads a device. The core says which configuration it
wants the estimator asked about next, and decides once the answers come back; cli.py does
the asking and makes no decision of its own.

Two model classes, two levers.

A dense model keeps every layer on the card, always. Moving layers of a dense model into
system memory is not a placement, it is a different and much slower machine, so a dense
model is served whole or not at all. That leaves the window, the precision of the
attention cache, and whether a prediction head runs -- and since the last two are choices
rather than answers, one file yields up to four profiles, each with the longest window
its combination can hold.

A mixture of experts is the opposite. Its experts are read a few per token, so leaving
some in system memory costs a bounded amount of speed and returns a large amount of
memory. The window stays where it is -- a mixture is worth its size for the window it
holds, and one cut short is a worse dense model -- and the count of offloaded layers is
the lever.

Both levers are searched rather than solved. What a configuration needs moves in one
direction along its lever and never turns back, which is all a search needs, and it is
the only thing about the shape of that curve worth relying on: fitting a straight line
through two points was tried and it lied by three thousand tokens, because the compute
buffers hold flat over most of the range and jump near the end. Every step of the search
is half a second of arithmetic that loads nothing, and calibration is rare.

Everything about the cards is a list with an element per device, and a machine with one
card is the list with one element in it. There is one way through this module whatever
the length.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .estimate import Needs, Refused, Requirement
from .facts import Head, ModelFacts
from .machine import CudaIndex
from .nonempty import NonEmpty
from .units import Halvings, Layers, Mib, Tokens

# Video memory the CUDA driver holds before a single tensor is placed. It appears in no
# log the loader writes and belongs to the driver rather than to any model, so it is
# written down once. To re-measure it: take the estimate for a configuration, load
# exactly that configuration, and subtract the estimate from the rise in
# nvidia-smi --query-gpu=memory.used.
DRIVER_CONTEXT = Mib(400)

# Windows are written, named and searched in thousands of tokens. A step of one costs
# some twenty megabytes against a reserve of a thousand -- far too little to be worth an
# unreadable number in a profile's name. Nothing is aligned to anything beyond that: a
# window is a count of tokens, its cache is allocated per token, and a power of two buys
# it nothing.
PAGE = 1000

# What a mixture of experts is asked to hold. Long enough to be why the model was chosen.
# The trained length behind it is 131072, written on the page like every other window:
# the seventy-two tokens over are worth less than a number a person can read.
MIXTURE_WINDOW = Tokens(131000)

# Video memory to leave for everything that is not a model, unless the settings file
# says otherwise. A target to land near rather than a line to clear.
DEFAULT_RESERVE = Mib(1024)

# The least any device is left for the system that drives it, whatever the settings file
# asks. Windows keeps part of every card to itself, a desktop or not; tens of megabytes
# either way make no difference, and a model that fails to allocate is lost outright.
# A target like any reserve, not a line.
WINDOWS_SHARE = Mib(1024)

# Shortest window worth writing out, unless the settings file says otherwise. Below some
# such length a window stops being useful, but where exactly is a judgement about the
# work rather than a fact about the card, which is why it is a setting.
DEFAULT_MIN_CTX = Tokens(25000)

# Window past which a coarser cache is not worth its precision. Below it a longer window
# is what the work is short of; above it the window is already ample and the cache is
# the only thing left to lose. Where the line falls is, again, a judgement about the
# work, so it is a setting.
DEFAULT_AMPLE_CTX = Tokens(100000)


class CacheType(Enum):
    """Precision of the attention cache.

    Only these: the released CUDA builds carry flash-attention kernels for F16/F16,
    Q4_0/Q4_0 and Q8_0/Q8_0, and any other pair silently falls back to materialising the
    attention matrix, many times slower with nothing logged.
    """

    Q8_0 = "q8_0"
    Q4_0 = "q4_0"
    F16 = "f16"

    @property
    def bytes_per_element(self) -> float:
        return {"q4_0": 18 / 32, "q8_0": 34 / 32, "f16": 2.0}[self.value]


# Finest first. A coarser cache costs precision, so it is only ever taken for what it
# buys: a longer window, or a model fitting at all.
PRECISION = (CacheType.F16, CacheType.Q8_0, CacheType.Q4_0)


@dataclass(frozen=True)
class WholeCard:
    """Every layer on the card."""


@dataclass(frozen=True)
class ExpertsOnCpu:
    """Experts of this many layers left in system memory."""

    layers: Layers


Placement = WholeCard | ExpertsOnCpu


@dataclass(frozen=True)
class Local:
    """A card in this machine: the number llama.cpp knows it by, and how much it has."""

    index: CudaIndex
    total: Mib


Device = Local


# A tensor name nothing in a model is called. Telling llama.cpp where to put it moves
# nothing, and any such override at all is what turns its pipeline parallelism off.
NO_TENSOR = "no-tensor-is-called-this"


class Pipeline(Enum):
    """Whether consecutive pieces of a prompt are processed on several cards at once.

    llama.cpp does it by itself on two local cards or more, and pays for it with extra
    copies of the working buffers on every card. On one card there is nothing to run at
    once, so a single card is always OFF.
    """

    ON = "on"
    OFF = "off"


@dataclass(frozen=True)
class Layout:
    """Where the layers of one configuration go, and the micro-batch they run at.

    layers counts per device the way the loader counts: the model's blocks, and on the
    last device also the output and a prediction head's block. The last device is the one
    that carries both.
    """

    devices: NonEmpty[Device]
    layers: NonEmpty[Layers]
    halvings: Halvings
    pipeline: Pipeline

    def __post_init__(self) -> None:
        if len(self.devices) != len(self.layers):
            raise ValueError("a layout names a count of layers for every device it uses")


@dataclass(frozen=True)
class Question:
    """A configuration to ask the estimator about.

    It carries no head. The estimator has no flag to be told one will run and answers the
    same either way, so the head is arithmetic the core adds afterwards -- which also
    means the variant with a head and the one without share a single answer.
    """

    ctx: Tokens
    cache: CacheType
    placement: Placement
    layout: Layout


@dataclass(frozen=True)
class Point:
    """One setting of a model's lever: the window, and where its experts are."""

    ctx: Tokens
    placement: Placement


@dataclass(frozen=True)
class Variant:
    """One of the ways a single file can be run, before the card has its say."""

    cache: CacheType
    head: bool


@dataclass(frozen=True)
class Allowed:
    """What a person permits for this file, narrowing what its shape makes possible.

    Neither of these is visible to the estimator or to any memory figure. A coarser
    cache can cost quality on one model and cost nothing on another; a prediction head
    can accept so little of what it drafts that verifying it is a loss. Both are
    judgements about the model's answers rather than about the card, so both are ruled
    out by hand or not at all.

    It only ever narrows. Permitting a head on a file that carries none yields no head.
    """

    caches: frozenset[CacheType]
    head: bool


# What a file may do when nobody has said otherwise.
EVERYTHING = Allowed(caches=frozenset({CacheType.Q8_0, CacheType.Q4_0}), head=True)


@dataclass(frozen=True)
class Seat:
    """One device a placement may use: which it is, what it may hold, what to leave.

    available is a bound: past it a model does not run slowly, it fails to allocate.
    reserve is a target, and missing it from either side is the same miss.
    """

    device: Device
    available: Mib
    reserve: Mib


# The devices one placement runs across, in the order it runs through them.
Chain = NonEmpty[Seat]


@dataclass(frozen=True)
class Limits:
    """What placements may spend, what is worth having, and what they run with.

    Every chain is placed on its own and yields profiles of its own. ubatch is the
    micro-batch the settings file runs with; min_ctx is the human's call about how short
    a window stops being useful, and ample_ctx about how long a window stops wanting a
    coarser cache to lengthen it.
    """

    chains: NonEmpty[Chain]
    ubatch: int
    min_ctx: Tokens
    ample_ctx: Tokens


@dataclass(frozen=True)
class Settings:
    """One way to run one model across one chain of devices."""

    ctx: Tokens
    cache: CacheType
    head: bool
    placement: Placement
    spare: NonEmpty[Mib]
    layout: Layout


def local_seat(index: CudaIndex, total: Mib, reserve: Mib) -> Seat:
    """A card in this machine, with what the driver keeps taken off the top."""
    return Seat(device=Local(index, total),
                available=Mib(total - DRIVER_CONTEXT),
                reserve=Mib(max(reserve, WINDOWS_SHARE)))


def limits_for(chains: NonEmpty[Chain], ubatch: int, min_ctx: Tokens,
               ample_ctx: Tokens) -> Limits:
    return Limits(chains=chains, ubatch=ubatch, min_ctx=min_ctx, ample_ctx=ample_ctx)


def holds(settings: Settings) -> NonEmpty[Mib]:
    """The video memory this placement holds on each device once it is loaded.

    The other side of `spare`: everything the placement counted, plus the driver's own
    context. This is the figure a free-memory reading has to be compared against, and
    the one written into the preset for `vram` to read back.
    """
    held = [Mib(device.total - spare)
            for device, spare in zip(settings.layout.devices, settings.spare)]
    return NonEmpty(*held)


def micro_batch(ubatch: int, halvings: Halvings) -> int:
    """The micro-batch a layout runs at, out of the one the settings file runs with."""
    return ubatch >> halvings


def device_name(device: Device) -> str:
    """What llama.cpp calls the device on its command line."""
    return f"CUDA{device.index}"


def head_cost(facts: ModelFacts, cache: CacheType, ctx: Tokens) -> Mib:
    """What a prediction head adds: its weights, and its own cache over the window.

    It is one more layer, so its cache follows the window exactly as the model's does.
    """
    if not isinstance(facts.head, Head):
        return Mib(0)

    elements = ctx * facts.head.cache_per_token * cache.bytes_per_element
    return Mib(round((facts.head.weight_bytes + elements) / 1048576))


def requirement(facts: ModelFacts, variant: Variant, ctx: Tokens,
                answer: Needs) -> NonEmpty[Mib]:
    """What the estimator counted on each device, plus what it was never told to count.

    The head is the last layer of the model, so it is counted on the last device.
    """
    if not variant.head:
        return answer.cards

    *before, last = answer.cards
    return NonEmpty(*before, Mib(last + head_cost(facts, variant.cache, ctx)))


def _ceiling(facts: ModelFacts) -> Tokens:
    """The longest window worth asking for, on the page windows are written in.

    Past what the model was trained for the weights are asked about positions they never
    saw, which costs quality rather than memory, so a larger card buys nothing there.
    """
    return Tokens(facts.n_ctx_train - facts.n_ctx_train % PAGE)


def _floor(facts: ModelFacts, limits: Limits) -> Tokens:
    """The shortest window worth writing out, on the same page.

    A model trained for less than the settings ask for cannot be asked for more than it
    knows, so for it the ceiling is also the floor and the only question left is whether
    it fits there.
    """
    shortest = Tokens(-(-limits.min_ctx // PAGE) * PAGE)
    return Tokens(min(shortest, _ceiling(facts)))


def variants(facts: ModelFacts, allowed: Allowed) -> tuple[Variant, ...]:
    """Every way this file may be run, before the card rules any of them out.

    A head is a choice only where the file carries one -- the tensors are dropped by
    some quantisations whatever the upstream config declares -- and only where it is
    permitted.
    """
    caches = tuple(cache for cache in PRECISION if cache in allowed.caches)
    if not caches:
        return ()

    if facts.n_expert > 0:
        # One profile, at the most precise cache still permitted: a mixture is placed by
        # moving experts, and its window does not move to pay for a coarser cache.
        return (Variant(caches[0], head=False),)

    heads = ((False, True)
             if allowed.head and isinstance(facts.head, Head) else (False,))
    return tuple(Variant(cache, head) for cache in caches for head in heads)


def grid(facts: ModelFacts, limits: Limits) -> tuple[Point, ...]:
    """Every setting the lever can be put at, ordered by falling spare.

    First entry leaves the card the most, last the least. That is the only property the
    search relies on, and it holds for both levers: a longer window costs more, and an
    expert moved off the card costs less.
    """
    if facts.n_expert > 0:
        # A model trained for less is asked for what it knows, on the same page.
        window = Tokens(min(_ceiling(facts), MIXTURE_WINDOW))
        return tuple(Point(window, ExpertsOnCpu(Layers(offloaded)))
                     for offloaded in range(facts.n_layer, -1, -1))

    floor = _floor(facts, limits)
    windows = range(floor, _ceiling(facts) + 1, PAGE)

    return tuple(Point(Tokens(window), WholeCard()) for window in windows)


def _trailing(facts: ModelFacts) -> Layers:
    """The layers the loader counts after the model's blocks: the output, and the head's
    own block where the file carries one. Both go on the last device."""
    return Layers(1 + (1 if isinstance(facts.head, Head) else 0))


@dataclass(frozen=True)
class _Search:
    """One search along the lever: a variant, on a chain, at a micro-batch."""

    variant: Variant
    chain: Chain
    layout: Layout


def _searches(facts: ModelFacts, allowed: Allowed,
              limits: Limits) -> tuple[_Search, ...]:
    """Every search the placements are settled by, chain by chain."""
    found = []
    for chain in limits.chains:
        layout = Layout(devices=chain.map(lambda seat: seat.device),
                        layers=NonEmpty(Layers(facts.n_layer + _trailing(facts))),
                        halvings=Halvings(0),
                        pipeline=Pipeline.OFF)
        found.extend(_Search(variant, chain, layout)
                     for variant in variants(facts, allowed))

    return tuple(found)


@dataclass(frozen=True)
class _Known:
    """What the answers say about one point of one search."""

    needed: NonEmpty[Mib]
    spare: NonEmpty[Mib]
    fits: bool
    clears_reserve: bool
    # How far the placement lands from the reserve, from either side.
    miss: int


@dataclass(frozen=True)
class _Ask:
    question: Question


@dataclass(frozen=True)
class _Unanswerable:
    """The estimator would not answer for a point the search needs, so this search
    settles nothing. It is arithmetic, and asking again gets the same refusal."""


def _known_at(facts: ModelFacts, search: _Search, points: Sequence[Point], index: int,
              answers: Mapping[Question, Requirement]) -> _Known | _Ask | _Unanswerable:
    point = points[index]
    question = Question(point.ctx, search.variant.cache, point.placement, search.layout)
    if question not in answers:
        return _Ask(question)

    match answers[question]:
        case Refused():
            return _Unanswerable()
        case Needs() as answer if len(answer.cards) != len(search.chain):
            # An answer about other devices than the ones asked about is no answer.
            return _Unanswerable()
        case Needs() as answer:
            needed = requirement(facts, search.variant, point.ctx, answer)

    spare = NonEmpty(*(Mib(seat.available - need)
                       for seat, need in zip(search.chain, needed)))
    gap = spare.first - search.chain.first.reserve

    return _Known(needed=needed,
                  spare=spare,
                  fits=all(one >= 0 for one in spare),
                  clears_reserve=gap >= 0,
                  miss=abs(gap))


def next_questions(facts: ModelFacts, allowed: Allowed, limits: Limits,
                   answers: Mapping[Question, Requirement]) -> tuple[Question, ...]:
    """What the estimator should be asked next. Empty when there is nothing left to ask.

    Every search that has not finished contributes one question, so a round of asking
    costs one pass whatever the file. A refusal ends that search rather than being
    retried: the estimator is arithmetic and gives the same answer twice.
    """
    points = grid(facts, limits)
    if not points:
        return ()

    wanted: list[Question] = []
    for search in _searches(facts, allowed, limits):
        match _step(facts, search, points, answers):
            case _Ask(question) if question not in wanted and question not in answers:
                wanted.append(question)
            case _Ask() | _Settled() | _Hopeless() | _Unanswerable():
                pass

    return tuple(wanted)


def settings(facts: ModelFacts, allowed: Allowed, limits: Limits,
             answers: Mapping[Question, Requirement]) -> tuple[Settings, ...]:
    """The placements the answers settled on.

    A variant with nothing that fits contributes nothing: a dense model that cannot sit
    on this card whole is skipped rather than offloaded, and there is no Settings
    describing a placement that does not fit.
    """
    points = grid(facts, limits)
    if not points:
        return ()

    kept: list[Settings] = []
    for chain in limits.chains:
        chosen = []
        for search in _searches(facts, allowed, limits):
            if search.chain != chain:
                continue
            match _step(facts, search, points, answers):
                case _Settled(index, known):
                    chosen.append(Settings(ctx=points[index].ctx,
                                           cache=search.variant.cache,
                                           head=search.variant.head,
                                           placement=points[index].placement,
                                           spare=known.spare,
                                           layout=search.layout))
                case _Ask() | _Hopeless() | _Unanswerable():
                    pass

        kept.extend(_worth_offering(chosen, limits))

    return tuple(kept)


def resident(chosen: Sequence[Settings],
             answers: Mapping[Question, Requirement]) -> Mib:
    """The most system memory any of these placements keeps off the card.

    The router holds one model at a time, so the heaviest placement is the one the
    prompt cache has to leave room for, and the others cost nothing while it is loaded.

    A placement names the question it was settled by -- the window, the cache, where
    the experts went and where the layers went are the question -- so the answer is
    looked up rather than carried around. An answer that is not a requirement
    contributes nothing: no placement was ever settled on one.
    """
    most = 0
    for settings in chosen:
        question = Question(settings.ctx, settings.cache, settings.placement,
                            settings.layout)
        if question not in answers:
            continue
        match answers[question]:
            case Needs(_, host):
                most = max(most, host)
            case Refused():
                pass

    return Mib(most)


def _worth_offering(chosen: list[Settings], limits: Limits) -> tuple[Settings, ...]:
    """Without the coarser caches that bought nothing worth having.

    A coarser attention cache is precision given up, and what it can buy is window. It
    earns its place only in the middle: where the finer cache leaves the window short.
    Once the window is ample there is nothing left to buy, and two profiles differing
    only in how exactly they hold the same conversation are not a choice anyone can
    make. Where the finer cache did not fit at all, there is nothing to compare against
    and the coarser one stands -- it is what makes the model servable.
    """
    kept = []
    for head in (False, True):
        longest = 0
        for cache in PRECISION:
            for settings in chosen:
                if settings.head is not head or settings.cache is not cache:
                    continue
                if longest >= limits.ample_ctx or settings.ctx <= longest:
                    continue
                longest = settings.ctx
                kept.append(settings)

    return tuple(kept)


@dataclass(frozen=True)
class _Settled:
    index: int
    known: _Known


@dataclass(frozen=True)
class _Hopeless:
    """Not one point of this search fits on the card."""


_Step = _Ask | _Settled | _Hopeless | _Unanswerable


def _step(facts: ModelFacts, search: _Search, points: Sequence[Point],
          answers: Mapping[Question, Requirement]) -> _Step:
    """Where one search stands: what to ask next, or what it settled on.

    Pure in the answers, so the same table always gives the same step and the search can
    be resumed, replayed or tested without anything to hold on to between calls.

    Spare falls along the grid, so the two ends answer the two questions worth asking
    first: the last point is the most the card could be asked to hold, and the first is
    the least. Between them the answer is where spare crosses the reserve, and a
    bisection finds that in the logarithm of the grid's length.
    """
    last = len(points) - 1

    match _known_at(facts, search, points, last, answers):
        case _Ask() | _Unanswerable() as waiting:
            return waiting
        case _Known() as generous if generous.clears_reserve:
            # Even the most this lever can be asked for leaves the reserve behind: there
            # is nothing to trade and no reason to look further.
            return _Settled(last, generous)
        case _Known() as generous:
            pass

    match _known_at(facts, search, points, 0, answers):
        case _Ask() | _Unanswerable() as waiting:
            return waiting
        case _Known() as frugal if not frugal.fits:
            return _Hopeless()
        case _Known() as frugal if not frugal.clears_reserve:
            # Nowhere on the grid is the reserve reached, so the point leaving the most
            # is the nearest to it that fits.
            return _Settled(0, frugal)
        case _Known() as frugal:
            pass

    low, under, high, over = 0, frugal, last, generous
    while high - low > 1:
        middle = (low + high) // 2
        match _known_at(facts, search, points, middle, answers):
            case _Ask() | _Unanswerable() as waiting:
                return waiting
            case _Known() as known if known.clears_reserve:
                low, under = middle, known
            case _Known() as known:
                high, over = middle, known

    return _nearer(low, under, high, over)


def _nearer(low: int, under: _Known, high: int, over: _Known) -> _Settled:
    """Of the pair the reserve falls between, the one that misses it by less.

    The reserve is a target rather than a boundary, so the point just past it is a
    candidate exactly like the one just short of it. The card is a boundary, so a point
    overrunning it is not a candidate at any distance.
    """
    if not over.fits:
        return _Settled(low, under)
    if over.miss < under.miss:
        return _Settled(high, over)
    return _Settled(low, under)
