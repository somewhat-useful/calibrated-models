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
the length: on one card the layers have one place to go, pieces of a prompt have nothing
to run beside, and what is left is the search above.

On several devices each point of the lever is also a question of where the layers go.
The chain is filled from its end, the fastest device taking all it has room for, and the
point is judged by the first device, which takes what is left. A card left nothing to hold
is not used at all. The micro-batch is a third
lever, and whether the cards run pieces of a prompt at once a fourth; both are given up
only for a window worth what they cost.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from .estimate import Needs, Refused, Requirement
from .facts import Head, ModelFacts
from .machine import CudaIndex, Installed
from .nonempty import NonEmpty
from .units import Halvings, Layers, Mib, Port, Tokens

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

# What to leave on each card that drives a monitor, where the machine has several cards:
# one of them can be worked at while the others serve, and a desktop wants room.
DEFAULT_MULTI_GPU_RESERVE = Mib(2048)

# What to leave on a slave's card: everything its own machine keeps, in one round figure.
DEFAULT_SLAVE_RESERVE = Mib(2048)

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

# The smallest micro-batch a placement is asked at. Below it prefill slows for little
# memory: the buffers that shrink with it are already small.
SMALLEST_UBATCH = 128

# What one halving of the micro-batch has to buy, as a share of the ample window. Halving
# costs roughly a tenth of prefill speed, so it is worth taking only for a tenth of the
# window a profile is meant to reach -- and past ample a window buys nothing.
PREFILL_PER_HALVING = Fraction(1, 10)

# How much longer a window has to be before the cards stop running pieces of a prompt at
# once. Generation is several per cent faster with it, so it is given up only for a
# window a good deal longer than it allows.
WORTH_RUNNING_APART = Fraction(13, 10)

# A tensor name nothing in a model is called. Telling llama.cpp where to put it moves
# nothing, and any such override at all is what turns its pipeline parallelism off.
NO_TENSOR = "no-tensor-is-called-this"


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
class Endpoint:
    """Where a slave's llama.cpp worker listens."""

    host: str
    port: Port


@dataclass(frozen=True)
class Local:
    """A card in this machine: the number llama.cpp knows it by, and how much it has."""

    index: CudaIndex
    total: Mib


@dataclass(frozen=True)
class Remote:
    """The card of a slave, lent through its worker, and how much memory it was named
    as having."""

    endpoint: Endpoint
    memory: Mib


Device = Local | Remote


@dataclass(frozen=True, kw_only=True)
class Worker:
    """A slave as the settings file names it."""

    endpoint: Endpoint
    memory: Mib
    reserve: Mib


@dataclass(frozen=True, kw_only=True)
class Reserves:
    """What to leave on a machine's own cards.

    alone is for a machine's only card. with_others is for each card driving a monitor
    where there are several; a card driving none is left WINDOWS_SHARE.
    """

    alone: Mib
    with_others: Mib


class Pipeline(Enum):
    """Whether consecutive pieces of a prompt are processed on several cards at once.

    llama.cpp does it by itself on two local cards or more, and pays for it with extra
    copies of the working buffers on every card. Anywhere else there is nothing to run at
    once, and a layout says OFF.
    """

    ON = "on"
    OFF = "off"


class Among(Enum):
    """How many cards of its own the machine chose a layout's cards from.

    A machine with one card has never been told which card to use, and is not told now.
    Where there are several, a layout on one of them names it: left unnamed, llama.cpp
    would take the first card it counts, whichever that is.
    """

    ONE = "one"
    SEVERAL = "several"


@dataclass(frozen=True)
class Layout:
    """Where the layers of one configuration go, and the micro-batch they run at.

    devices are the ones that hold something, in the order the layers run through them.
    layers counts per device the way the loader counts: the model's blocks, and on the
    last device also the output and a prediction head's block. The last device is the one
    that carries both.
    """

    devices: NonEmpty[Device]
    layers: NonEmpty[Layers]
    halvings: Halvings
    pipeline: Pipeline
    among: Among

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


@dataclass(frozen=True, kw_only=True)
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

    Every chain is placed on its own and yields profiles of its own. The first is the
    machine's own cards; one after it adds a slave to them, and is only worth a profile
    where it holds a longer window than the first does. ubatch is the micro-batch the
    settings file runs with; min_ctx is the human's call about how short a window stops
    being useful, and ample_ctx about how long a window stops wanting a coarser cache to
    lengthen it.
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
                reserve=_at_least_windows(reserve))


def remote_seat(worker: Worker) -> Seat:
    """A slave's card. Nothing is taken off the top: the memory is a round figure a
    person named, and what is left on it covers everything its own machine keeps."""
    return Seat(device=Remote(worker.endpoint, worker.memory),
                available=worker.memory,
                reserve=_at_least_windows(worker.reserve))


def _at_least_windows(reserve: Mib) -> Mib:
    return Mib(max(reserve, WINDOWS_SHARE))


def chains(cards: NonEmpty[Installed], workers: Sequence[Worker],
           reserves: Reserves) -> NonEmpty[Chain]:
    """Every chain of devices a model is placed across on this machine.

    The machine's own cards first, all of them, the earliest generation first. A chain
    is filled from its end, so the last card holds the most layers and the output and
    the head besides, and the fastest card is the one to give them to. Then, for each
    slave, the same cards behind its card: reached over the network, it is slower than
    any of them, and it goes first.
    """
    several = len(cards) > 1
    earliest_first = sorted(cards, key=lambda one: (one.capability, one.index))

    first, *rest = (local_seat(one.index, one.card.total, _reserve(one, several, reserves))
                    for one in earliest_first)
    local = NonEmpty(first, *rest)

    return NonEmpty(local, *(NonEmpty(remote_seat(worker), *local) for worker in workers))


def _reserve(card: Installed, several: bool, reserves: Reserves) -> Mib:
    if not several:
        return reserves.alone
    if card.drives_display:
        return reserves.with_others
    return Mib(0)


def limits_for(chains: NonEmpty[Chain], ubatch: int, min_ctx: Tokens,
               ample_ctx: Tokens) -> Limits:
    return Limits(chains=chains, ubatch=ubatch, min_ctx=min_ctx, ample_ctx=ample_ctx)


def holds(settings: Settings) -> NonEmpty[Mib]:
    """The video memory this placement holds on each device once it is loaded.

    The other side of `spare`: everything the placement counted, plus, on a card of this
    machine, the driver's own context. This is the figure a free-memory reading has to be
    compared against, and the one written into the preset for `vram` to read back.
    """
    first, *rest = (_held(device, spare)
                    for device, spare in zip(settings.layout.devices, settings.spare))
    return NonEmpty(first, *rest)


def _held(device: Device, spare: Mib) -> Mib:
    match device:
        case Local(_, total):
            return Mib(total - spare)
        case Remote(_, memory):
            return Mib(memory - spare)


def micro_batch(ubatch: int, halvings: Halvings) -> int:
    """The micro-batch a layout runs at, out of the one the settings file runs with."""
    return ubatch >> halvings


def device_name(device: Device) -> str:
    """What llama.cpp calls the device on its command line. A slave's card is the one
    device reached over RPC, so it is the first of those."""
    match device:
        case Local(index, _):
            return f"CUDA{index}"
        case Remote():
            return "RPC0"


def endpoints(layout: Layout) -> tuple[Endpoint, ...]:
    """The workers a layout reaches over the network, in the order it uses them."""
    return tuple(device.endpoint for device in layout.devices
                 if isinstance(device, Remote))


def runs_apart_by_override(layout: Layout) -> bool:
    """Whether a layout has to override something to keep its cards from running pieces
    of a prompt at once: several cards, all local, told OFF. Where a worker is in the
    chain llama.cpp keeps them apart by itself."""
    return (len(layout.devices) > 1 and layout.pipeline is Pipeline.OFF
            and all(isinstance(device, Local) for device in layout.devices))


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


def next_questions(facts: ModelFacts, allowed: Allowed, limits: Limits,
                   answers: Mapping[Question, Requirement]) -> tuple[Question, ...]:
    """What the estimator should be asked next. Empty when there is nothing left to ask.

    Every variant on every chain that has not finished contributes one question, so a
    round of asking costs one pass whatever the file. A refusal ends that search rather
    than being retried: the estimator is arithmetic and gives the same answer twice.
    """
    points = grid(facts, limits)
    if not points:
        return ()

    wanted: list[Question] = []
    for chain in limits.chains:
        for variant in variants(facts, allowed):
            match _best(facts, limits, variant, chain, points, answers):
                case _Ask(question) if question not in wanted and question not in answers:
                    wanted.append(question)
                case _Ask() | _Chosen() | _Nothing():
                    pass

    return tuple(wanted)


def settings(facts: ModelFacts, allowed: Allowed, limits: Limits,
             answers: Mapping[Question, Requirement]) -> tuple[Settings, ...]:
    """The placements the answers settled on.

    A variant with nothing that fits contributes nothing: a dense model that cannot sit
    on this card whole is skipped rather than offloaded, and there is no Settings
    describing a placement that does not fit.

    A chain with a slave in it is slower than the machine's own cards, so a placement on
    it is kept only where it uses the slave's card at all, and its window is longer than
    the one the first chain settled on for the same variant.
    """
    points = grid(facts, limits)
    if not points:
        return ()

    kept: list[Settings] = []
    own: dict[Variant, Tokens] = {}
    for position, chain in enumerate(limits.chains):
        chosen = []
        for variant in variants(facts, allowed):
            match _best(facts, limits, variant, chain, points, answers):
                case _Chosen(index, known):
                    window = points[index].ctx
                    layout = known.arranged.layout
                    if position == 0:
                        own[variant] = window
                    elif not endpoints(layout):
                        continue
                    elif variant in own and window <= own[variant]:
                        continue
                    chosen.append(Settings(ctx=window,
                                           cache=variant.cache,
                                           head=variant.head,
                                           placement=points[index].placement,
                                           spare=_used(chain, known.spare, layout),
                                           layout=layout))
                case _Ask() | _Nothing():
                    pass

        kept.extend(_worth_offering(chosen, limits))

    return tuple(kept)


def _used(chain: Chain, amounts: NonEmpty[Mib], layout: Layout) -> NonEmpty[Mib]:
    """Of an amount for every device of the chain, the amounts of the devices the layout
    uses, in the same order."""
    first, *rest = (amount for seat, amount in zip(chain, amounts)
                    if seat.device in layout.devices)
    return NonEmpty(first, *rest)


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


# ---------------------------------------------------------------------------- the search


@dataclass(frozen=True)
class _Search:
    """One search along the lever: a variant, on a chain, at a micro-batch, with the
    cards running pieces of a prompt at once or not."""

    variant: Variant
    chain: Chain
    halvings: Halvings
    pipeline: Pipeline


@dataclass(frozen=True)
class _Ask:
    question: Question


@dataclass(frozen=True)
class _Unanswerable:
    """Nothing this search could be told would settle it: the estimator refused a point
    it needs, which asking again would not change."""


@dataclass(frozen=True)
class _Arranged:
    """Where the layers go at one point, and what that needs on every device.

    short says, per device, that it holds a single block and still falls short of its
    reserve: nowhere left to give a block to, which no other point of the lever mends.
    """

    layout: Layout
    needed: NonEmpty[Mib]
    short: NonEmpty[bool]


@dataclass(frozen=True)
class _Known:
    """What the answers say about one point of one search."""

    arranged: _Arranged
    spare: NonEmpty[Mib]
    fits: bool
    clears_reserve: bool
    # How far the placement lands from the reserve, from either side.
    miss: int


@dataclass(frozen=True)
class _Settled:
    index: int
    known: _Known


@dataclass(frozen=True)
class _Hopeless:
    """Not one point of this search fits on the cards."""


@dataclass(frozen=True)
class _Took:
    """How many blocks a device takes, and whether one was already too many."""

    blocks: Layers
    short: bool


@dataclass(frozen=True)
class _Spare:
    amount: Mib


@dataclass(frozen=True)
class _Chosen:
    """The placement a variant settles on for a chain, at its micro-batch and pipeline."""

    index: int
    known: _Known


@dataclass(frozen=True)
class _Nothing:
    """No placement of this variant fits this chain."""


@dataclass(frozen=True)
class _Within:
    """A search that might beat what is already chosen, and is worth running."""


@dataclass(frozen=True)
class _Beyond:
    """A search that cannot beat what is already chosen, whatever it finds."""


@dataclass(frozen=True)
class _OneWay:
    pipeline: Pipeline


@dataclass(frozen=True)
class _TwoWays:
    together: Pipeline
    apart: Pipeline


def _ways(chain: Chain) -> _OneWay | _TwoWays:
    """How the cards of a chain may run a prompt.

    Where llama.cpp runs pieces of it on the cards at once by itself -- two local cards
    or more, and no worker in the chain -- keeping them apart is the other way, bought
    with an override. Anywhere else there is one way.
    """
    if len(chain) > 1 and all(isinstance(seat.device, Local) for seat in chain):
        return _TwoWays(together=Pipeline.ON, apart=Pipeline.OFF)
    return _OneWay(Pipeline.OFF)


def _halvings(ubatch: int) -> tuple[Halvings, ...]:
    """The micro-batch the settings file runs with, then each halving of it down to the
    smallest a placement is asked at. The file's own is always among them."""
    steps = [Halvings(0)]
    while micro_batch(ubatch, Halvings(len(steps))) >= SMALLEST_UBATCH:
        steps.append(Halvings(len(steps)))
    return tuple(steps)


def _trailing(facts: ModelFacts) -> Layers:
    """The layers the loader counts after the model's blocks: the output, and the head's
    own block where the file carries one. Both go on the last device."""
    return Layers(1 + (1 if isinstance(facts.head, Head) else 0))


def _best(facts: ModelFacts, limits: Limits, variant: Variant, chain: Chain,
          points: Sequence[Point],
          answers: Mapping[Question, Requirement]) -> _Ask | _Chosen | _Nothing:
    """The placement of one variant on one chain, every lever considered."""
    match _ways(chain):
        case _OneWay(pipeline):
            return _rung(facts, limits, variant, chain, pipeline, points, answers)
        case _TwoWays(together, apart):
            match _rung(facts, limits, variant, chain, together, points, answers):
                case _Ask() as asking:
                    return asking
                case _Nothing():
                    return _rung(facts, limits, variant, chain, apart, points, answers)
                case _Chosen() as running:
                    return _apart_if_worth(facts, limits, variant, chain, running, apart,
                                           points, answers)


def _apart_if_worth(facts: ModelFacts, limits: Limits, variant: Variant, chain: Chain,
                    together: _Chosen, apart: Pipeline, points: Sequence[Point],
                    answers: Mapping[Question, Requirement]) -> _Ask | _Chosen:
    """The cards kept from running pieces of a prompt at once, where the window that
    buys is WORTH_RUNNING_APART times the one they have running together.

    The one question that can rule it out is asked first, at the smallest micro-batch,
    which holds the longest window any search of it could find.
    """
    window = points[together.index].ctx
    longest = _Search(variant, chain, _halvings(limits.ubatch)[-1], apart)

    match _probe(facts, longest, points, answers,
                 lambda ctx: ctx >= window * WORTH_RUNNING_APART):
        case _Ask() as asking:
            return asking
        case _Beyond():
            return together
        case _Within():
            pass

    match _rung(facts, limits, variant, chain, apart, points, answers):
        case _Ask() as asking:
            return asking
        case _Chosen() as found if points[found.index].ctx >= window * WORTH_RUNNING_APART:
            return found
        case _Chosen() | _Nothing():
            return together


def _rung(facts: ModelFacts, limits: Limits, variant: Variant, chain: Chain,
          pipeline: Pipeline, points: Sequence[Point],
          answers: Mapping[Question, Requirement]) -> _Ask | _Chosen | _Nothing:
    """The micro-batch that runs this variant best, and the placement at it.

    The settings file's own micro-batch first, then each halving. A halving is kept only
    where its window, counted no further than ample, is longer by PREFILL_PER_HALVING of
    ample for every halving it took; equal is not enough, since the faster prefill is
    worth having for nothing. A halving that could not reach that far is not searched.
    """
    best: _Chosen | _Nothing = _Nothing()
    for halvings in _halvings(limits.ubatch):
        search = _Search(variant, chain, halvings, pipeline)

        match _reach(facts, limits, search, best, points, answers):
            case _Ask() as asking:
                return asking
            case _Beyond():
                continue
            case _Within():
                pass

        match _step(facts, search, points, answers):
            case _Ask() as asking:
                return asking
            case _Settled(index, known):
                best = _better(limits, points, best, _Chosen(index, known))
            case _Hopeless() | _Unanswerable():
                pass

    return best


def _better(limits: Limits, points: Sequence[Point], best: _Chosen | _Nothing,
            found: _Chosen) -> _Chosen:
    match best:
        case _Nothing():
            return found
        case _Chosen() if _score(limits, points, found) > _score(limits, points, best):
            return found
        case _Chosen():
            return best


def _score(limits: Limits, points: Sequence[Point], chosen: _Chosen) -> Fraction:
    """A placement's window as a share of the ample one, counted no further than ample,
    less what its halvings of the micro-batch cost."""
    window = min(points[chosen.index].ctx, limits.ample_ctx)
    halvings = chosen.known.arranged.layout.halvings
    return Fraction(window, limits.ample_ctx) - PREFILL_PER_HALVING * halvings


def _reach(facts: ModelFacts, limits: Limits, search: _Search, best: _Chosen | _Nothing,
           points: Sequence[Point],
           answers: Mapping[Question, Requirement]) -> _Ask | _Within | _Beyond:
    """Whether a search at this micro-batch could beat the placement already chosen."""
    match best:
        case _Nothing():
            return _Within()
        case _Chosen():
            needed = ((_score(limits, points, best) + PREFILL_PER_HALVING * search.halvings)
                      * limits.ample_ctx)

    if needed >= limits.ample_ctx:
        return _Beyond()

    return _probe(facts, search, points, answers, lambda ctx: ctx > needed)


def _probe(facts: ModelFacts, search: _Search, points: Sequence[Point],
           answers: Mapping[Question, Requirement],
           enough: Callable[[Tokens], bool]) -> _Ask | _Within | _Beyond:
    """Whether a search could settle on a window that is enough, asked of one point.

    A search settles on a point that leaves the reserve, or on the one just past such a
    point. Either way the point just short of the first window that is enough has to
    leave the reserve, so where it does not, nothing the search finds is enough.
    """
    reaching = [index for index, point in enumerate(points) if enough(point.ctx)]
    if not reaching:
        return _Beyond()
    if reaching[0] == 0:
        return _Within()

    match _known_at(facts, search, points, reaching[0] - 1, answers):
        case _Ask() as asking:
            return asking
        case _Unanswerable():
            return _Beyond()
        case _Known() as known if known.clears_reserve:
            return _Within()
        case _Known():
            return _Beyond()


def _step(facts: ModelFacts, search: _Search, points: Sequence[Point],
          answers: Mapping[Question, Requirement]
          ) -> _Ask | _Settled | _Hopeless | _Unanswerable:
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


def _known_at(facts: ModelFacts, search: _Search, points: Sequence[Point], index: int,
              answers: Mapping[Question, Requirement]) -> _Known | _Ask | _Unanswerable:
    """One point of one search, judged.

    Every device after the first has already been brought as near its reserve as whole
    blocks allow, so it is the first, holding what was left, that says how far the point
    is from the reserve -- unless a later device holds a single block and still falls
    short, which no layout of this point can mend and which is then what the point misses
    by. On one card the first device is the card, and this is the card's own spare
    against its reserve.
    """
    match _arranged(facts, search, points[index], answers):
        case _Ask() | _Unanswerable() as waiting:
            return waiting
        case _Arranged() as arranged:
            pass

    chain = search.chain
    first, *rest = (Mib(seat.available - need) for seat, need in zip(chain, arranged.needed))
    spare = NonEmpty(first, *rest)

    gap = spare.first - chain.first.reserve
    shortfalls = tuple(amount - seat.reserve
                       for seat, amount, short in zip(chain, spare, arranged.short)
                       if short and amount < seat.reserve)

    return _Known(arranged=arranged,
                  spare=spare,
                  fits=all(one >= 0 for one in spare),
                  clears_reserve=gap >= 0 and not shortfalls,
                  miss=max((abs(gap), *(abs(one) for one in shortfalls))))


def _arranged(facts: ModelFacts, search: _Search, point: Point,
              answers: Mapping[Question, Requirement]) -> _Arranged | _Ask | _Unanswerable:
    """Where the layers go at one point of the lever, and what that needs on each device.

    The chain is filled from its end. The last device, the fastest, takes blocks from the
    end of the model for as long as each one brings what it leaves nearer its reserve,
    and never one it has no room for; the device before it does the same with what is
    left, and the first takes the rest. The reserve is a target here as everywhere, so a
    device that can take one more block by missing it slightly takes the block.
    """
    count = len(search.chain)
    after: tuple[Layers, ...] = ()
    short: tuple[bool, ...] = ()
    left = facts.n_layer
    for position in range(count - 1, 0, -1):
        match _blocks_for(facts, search, point, position, left, after, answers):
            case _Ask() | _Unanswerable() as waiting:
                return waiting
            case _Took(blocks, fell_short):
                after = (blocks, *after)
                short = (fell_short, *short)
                left -= blocks

    question = _question(facts, search, point, NonEmpty(Layers(left), *after))
    match _needed(facts, search, point, question, answers):
        case _Ask() | _Unanswerable() as waiting:
            return waiting
        case NonEmpty() as needed:
            return _Arranged(question.layout, needed, NonEmpty(False, *short))


def _blocks_for(facts: ModelFacts, search: _Search, point: Point, position: int,
                left: int, after: tuple[Layers, ...],
                answers: Mapping[Question, Requirement]) -> _Took | _Ask | _Unanswerable:
    """How many of the blocks still to place the device at `position` takes.

    What a device needs grows with every block it holds, so the count that leaves its
    reserve is found by bisection, and of it and the count one past, the nearer to the
    reserve is taken. A device may take none, and is then not used: what it would have
    held fits on the devices after it. The last device carries the output and takes at
    least one.
    """
    reserve = search.chain[position].reserve
    least = 1 if position == len(search.chain) - 1 else 0
    most = left

    def spare_with(blocks: int) -> _Spare | _Ask | _Unanswerable:
        return _spare_with(facts, search, point, position, left, after, blocks, answers)

    match spare_with(least):
        case _Ask() | _Unanswerable() as waiting:
            return waiting
        case _Spare(one) if one < reserve:
            # Holding nothing, a device short of its reserve is short of nothing it holds.
            return _Took(Layers(least), short=least > 0)
        case _Spare(one):
            pass

    match spare_with(most):
        case _Ask() | _Unanswerable() as waiting:
            return waiting
        case _Spare(full) if full >= reserve:
            return _Took(Layers(most), short=False)
        case _Spare(full):
            pass

    low, above, high, below = least, one, most, full
    while high - low > 1:
        middle = (low + high) // 2
        match spare_with(middle):
            case _Ask() | _Unanswerable() as waiting:
                return waiting
            case _Spare(amount) if amount >= reserve:
                low, above = middle, amount
            case _Spare(amount):
                high, below = middle, amount

    if below >= 0 and reserve - below < above - reserve:
        return _Took(Layers(high), short=False)
    return _Took(Layers(low), short=False)


def _spare_with(facts: ModelFacts, search: _Search, point: Point, position: int,
                left: int, after: tuple[Layers, ...], blocks: int,
                answers: Mapping[Question, Requirement]) -> _Spare | _Ask | _Unanswerable:
    """What the device at `position` leaves holding this many blocks, the devices after
    it holding what they already took and the rest of the model on the first device.
    Holding none, it leaves all it has, and nothing needs asking."""
    seat = search.chain[position]
    if blocks == 0:
        return _Spare(seat.available)

    between = [Layers(0)] * (position - 1)
    trial = NonEmpty(Layers(left - blocks), *between, Layers(blocks), *after)

    match _needed(facts, search, point, _question(facts, search, point, trial), answers):
        case _Ask() | _Unanswerable() as waiting:
            return waiting
        case NonEmpty() as needed:
            return _Spare(Mib(seat.available - needed[position]))


def _question(facts: ModelFacts, search: _Search, point: Point,
              blocks: NonEmpty[Layers]) -> Question:
    """The question about this many blocks on each device of the search's chain.

    A device holding none is not in it. Where one device is left, nothing runs beside it,
    whichever way the search was running the cards.
    """
    last = len(search.chain) - 1
    used = [(seat.device, Layers(count + (_trailing(facts) if position == last else 0)))
            for position, (seat, count) in enumerate(zip(search.chain, blocks))
            if count > 0 or position == last]
    devices, layers = zip(*used)
    own = sum(1 for seat in search.chain if isinstance(seat.device, Local))

    layout = Layout(devices=NonEmpty(*devices),
                    layers=NonEmpty(*layers),
                    halvings=search.halvings,
                    pipeline=search.pipeline if len(used) > 1 else Pipeline.OFF,
                    among=Among.ONE if own == 1 else Among.SEVERAL)

    return Question(point.ctx, search.variant.cache, point.placement, layout)


def _needed(facts: ModelFacts, search: _Search, point: Point, question: Question,
            answers: Mapping[Question, Requirement]
            ) -> NonEmpty[Mib] | _Ask | _Unanswerable:
    """What a question's answer says each device of the chain needs, the head counted
    in. A device the question leaves out needs nothing."""
    if question not in answers:
        return _Ask(question)

    match answers[question]:
        case Refused():
            return _Unanswerable()
        case Needs() as answer if len(answer.cards) != len(question.layout.devices):
            # An answer about other devices than the ones asked about is no answer.
            return _Unanswerable()
        case Needs() as answer:
            held = dict(zip(question.layout.devices,
                            requirement(facts, search.variant, point.ctx, answer)))
            first, *rest = (held.get(seat.device, Mib(0)) for seat in search.chain)
            return NonEmpty(first, *rest)
