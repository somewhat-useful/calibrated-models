"""Invariants of placing one model across several devices.

The estimator is a law written here, as everywhere in this suite: every block weighs the
same, every fourth attends over the window, each card keeps working buffers that grow
with the micro-batch and with the window and are larger where the cards run pieces of a
prompt at once, and the last card carries the output. What a device needs depends only
on the blocks it holds, which is the one property of the real estimator the layout
search relies on.

The strongest invariants are checked against the core itself run with every other way
refused: what the micro-batch ladder or the pipeline settles on has to be what the rule
says, given what each way alone would have held.
"""

import unittest
from fractions import Fraction

from cm import place
from cm.estimate import Needs, Refused
from cm.facts import Head, ModelFacts, NoHead
from cm.machine import Capability, Card, CudaIndex, Installed, PciAddress
from cm.nonempty import NonEmpty
from cm.place import (WINDOWS_SHARE, CacheType, Endpoint, ExpertsOnCpu, Local, Pipeline,
                      Remote, Reserves, Worker)
from cm.units import Layers, Mib, Port, Tokens

UBATCH = 512
MIN_CTX = Tokens(25000)
AMPLE_CTX = Tokens(100000)

BLOCK = 200
EXPERTS = 150
ATTENTION_PER_TOKEN = 0.002
STATE = 3
OUTPUT = 500

DENSE = ModelFacts(n_expert=0, n_layer=Layers(64), n_ctx_train=Tokens(262144),
                   head=NoHead())

DENSE_WITH_HEAD = ModelFacts(n_expert=0, n_layer=Layers(64), n_ctx_train=Tokens(262144),
                             head=Head(weight_bytes=600_000_000, cache_per_token=2048))

MIXTURE = ModelFacts(n_expert=128, n_layer=Layers(40), n_ctx_train=Tokens(262144),
                     head=NoHead())

RESERVES = Reserves(alone=Mib(1024), with_others=Mib(2048))

WORKER = Worker(endpoint=Endpoint("worker", Port(50052)), memory=Mib(12288),
                reserve=Mib(2048))


def card(index, total, capability, display) -> Installed:
    return Installed(index=CudaIndex(index), card=Card(f"card {index}", Mib(total)),
                     capability=capability, drives_display=display,
                     address=PciAddress(index + 1, 0, 0))


def two(fast=16303, slow=8192) -> NonEmpty[Installed]:
    """A machine's cards as nvidia-smi numbers them: the fast one with the monitor first."""
    return NonEmpty(card(0, fast, Capability(12, 0), True),
                    card(1, slow, Capability(7, 5), False))


def law(facts, per_token_compute=0.00001, together=1.6):
    """What the estimator answers about a question, device by device."""
    trailing = 2 if isinstance(facts.head, Head) else 1

    def answer(question):
        layout = question.layout
        micro = place.micro_batch(UBATCH, layout.halvings)
        counts = [*layout.layers[:-1], layout.layers[-1] - trailing]
        rows, works, host, start = [], [], 0, 0
        for position, count in enumerate(counts):
            need, held = row(question, start, count, position == len(counts) - 1, micro,
                             per_token_compute, together)
            rows.append(Mib(round(need)))
            works.append(Mib(round(working(question, micro, per_token_compute, together))))
            host += held
            start += count
        return Needs(cards=NonEmpty(*rows), working=NonEmpty(*works),
                     host=Mib(round(host)))

    return answer


def working(question, micro, per_token_compute=0.00001, together=1.6):
    """One device's working buffers, larger where the cards run pieces of a prompt at
    once."""
    factor = together if question.layout.pipeline is Pipeline.ON else 1.0
    return factor * (micro * 0.25 + (0.004 + micro * per_token_compute) * question.ctx)


def row(question, start, count, last, micro, per_token_compute=0.00001, together=1.6):
    """One device: its blocks, their cache and state, its working buffers."""
    offloaded = (question.placement.layers
                 if isinstance(question.placement, ExpertsOnCpu) else 0)
    moved = max(0, min(start + count, offloaded) - start)
    attention = sum(1 for index in range(start, start + count) if index % 4 == 3)
    share = question.cache.bytes_per_element / CacheType.Q8_0.bytes_per_element

    need = (count * BLOCK - moved * EXPERTS
            + attention * ATTENTION_PER_TOKEN * question.ctx * share
            + (count - attention) * STATE
            + working(question, micro, per_token_compute, together))
    if last:
        need += OUTPUT
    return need, moved * EXPERTS


def refusing(answer, keep):
    """The same law, refusing every question `keep` does not accept."""
    return lambda question: answer(question) if keep(question) else Refused()


def run(facts, chains, answer, ubatch=UBATCH, ample_ctx=AMPLE_CTX,
        allowed=place.EVERYTHING):
    limits = place.limits_for(chains, ubatch, MIN_CTX, ample_ctx)
    answers = {}
    for _ in range(5000):
        asking = place.next_questions(facts, allowed, limits, answers)
        if not asking:
            break
        for question in asking:
            answers[question] = answer(question)
    else:
        raise AssertionError("the search did not settle")

    return place.settings(facts, allowed, limits, answers), answers


def local(cards, workers=()):
    return place.chains(cards, workers, RESERVES)


class TheChainsAddTheMachinesCardsOneAtATimeThenTheSlave(unittest.TestCase):
    def test_one_card_is_one_chain_of_one_left_what_it_is_alone(self):
        chains = local(NonEmpty(card(0, 16303, Capability(12, 0), True)))

        self.assertEqual(1, len(chains))
        self.assertEqual([place.local_seat(CudaIndex(0), Mib(16303), Mib(1024))],
                         list(chains.first))

    def test_the_latest_generation_is_alone_first_and_each_chain_puts_the_next_in_front(self):
        three = NonEmpty(card(0, 8192, Capability(7, 5), False),
                         card(1, 16303, Capability(12, 0), True),
                         card(2, 12288, Capability(8, 6), False))

        self.assertEqual([[CudaIndex(1)], [CudaIndex(2), CudaIndex(1)],
                          [CudaIndex(0), CudaIndex(2), CudaIndex(1)]],
                         [[seat.device.index for seat in chain] for chain in local(three)])

    def test_of_one_generation_the_card_with_more_memory_is_alone_first(self):
        same = NonEmpty(card(0, 8192, Capability(12, 0), True),
                        card(1, 16303, Capability(12, 0), False))

        self.assertEqual([[CudaIndex(1)], [CudaIndex(0), CudaIndex(1)]],
                         [[seat.device.index for seat in chain] for chain in local(same)])

    def test_a_later_generation_goes_first_whatever_the_memory_or_the_monitor(self):
        mixed = NonEmpty(card(0, 24576, Capability(8, 6), True),
                         card(1, 8192, Capability(8, 9), False))

        self.assertEqual([CudaIndex(1)],
                         [seat.device.index for seat in local(mixed).first])

    def test_of_several_cards_one_with_a_monitor_keeps_the_multi_gpu_reserve_alone(self):
        for chain in local(two()):
            seats = {seat.device.index: seat for seat in chain}
            with self.subTest(chain=list(seats)):
                self.assertEqual(Mib(2048), seats[CudaIndex(0)].reserve)

    def test_of_several_cards_one_without_a_monitor_keeps_what_windows_keeps_alone(self):
        """The monitors moved onto the slower card, so the fastest serves with all of its
        memory but what Windows keeps -- whatever a machine's only card would be left."""
        moved = NonEmpty(card(0, 16303, Capability(12, 0), False),
                         card(1, 8192, Capability(7, 5), True))
        chains = place.chains(moved, (), Reserves(alone=Mib(3072), with_others=Mib(2048)))

        self.assertEqual([CudaIndex(0)], [seat.device.index for seat in chains.first])
        for chain in chains:
            seats = {seat.device.index: seat for seat in chain}
            with self.subTest(chain=list(seats)):
                self.assertEqual(WINDOWS_SHARE, seats[CudaIndex(0)].reserve)
                if CudaIndex(1) in seats:
                    self.assertEqual(Mib(2048), seats[CudaIndex(1)].reserve)

    def test_beside_others_a_card_with_a_monitor_is_left_the_multi_gpu_reserve(self):
        seats = {seat.device.index: seat for seat in local(two())[1]}

        self.assertEqual(Mib(2048), seats[CudaIndex(0)].reserve)

    def test_beside_others_a_card_with_no_monitor_is_left_what_windows_keeps(self):
        seats = {seat.device.index: seat for seat in local(two())[1]}

        self.assertEqual(WINDOWS_SHARE, seats[CudaIndex(1)].reserve)

    def test_no_reserve_is_ever_below_what_windows_keeps(self):
        stingy = Reserves(alone=Mib(0), with_others=Mib(0))

        for cards in (two(), NonEmpty(card(0, 8192, Capability(7, 5), False))):
            for chain in place.chains(cards, (), stingy):
                for seat in chain:
                    with self.subTest(card=seat.device):
                        self.assertEqual(WINDOWS_SHARE, seat.reserve)

    def test_monitors_on_every_card_leave_every_card_the_reserve_beside_others(self):
        both = NonEmpty(card(0, 16303, Capability(12, 0), True),
                        card(1, 8192, Capability(7, 5), True))

        for seat in local(both)[1]:
            with self.subTest(card=seat.device):
                self.assertEqual(Mib(2048), seat.reserve)

    def test_the_driver_is_taken_off_a_local_card_and_nothing_off_a_slave(self):
        chains = local(two(), (WORKER,))
        slave = chains.last.first

        self.assertEqual(Remote(WORKER.endpoint, WORKER.memory), slave.device)
        self.assertEqual(WORKER.memory, slave.available)
        for chain in chains:
            for seat in chain:
                if isinstance(seat.device, Local):
                    with self.subTest(card=seat.device):
                        self.assertEqual(seat.device.total - place.DRIVER_CONTEXT,
                                         seat.available)

    def test_a_slave_goes_first_and_all_the_machines_cards_follow_it(self):
        chains = local(two(), (WORKER,))

        self.assertEqual(3, len(chains))
        self.assertEqual(list(chains[1]), list(chains.last)[1:])

    def test_a_machine_with_several_cards_names_them_and_one_with_a_slave_does_not(self):
        several = place.limits_for(local(two()), UBATCH, MIN_CTX, AMPLE_CTX)
        one = place.limits_for(local(NonEmpty(two().first), (WORKER,)), UBATCH, MIN_CTX,
                               AMPLE_CTX)

        self.assertIs(place.Among.SEVERAL, several.among)
        self.assertIs(place.Among.ONE, one.among)


class TheLayersAreAllPlacedAndOnlyOnce(unittest.TestCase):
    def test_every_question_places_every_block_with_one_or_more_on_each_device(self):
        for facts in (DENSE, DENSE_WITH_HEAD):
            trailing = 2 if isinstance(facts.head, Head) else 1
            _, answers = run(facts, local(two(), (WORKER,)), law(facts))

            for question in answers:
                layers = list(question.layout.layers)
                with self.subTest(facts=facts.head, layers=layers):
                    self.assertEqual(facts.n_layer + trailing, sum(layers))
                    self.assertTrue(all(count >= 1 for count in layers[:-1]))
                    self.assertGreaterEqual(layers[-1], 1 + trailing)

    def test_a_layout_names_the_devices_of_one_chain_in_order(self):
        chains = local(two(), (WORKER,))
        chosen, _ = run(DENSE, chains, law(DENSE))

        orders = {tuple(seat.device for seat in chain) for chain in chains}
        self.assertTrue(chosen)
        for settings in chosen:
            devices = tuple(settings.layout.devices)
            with self.subTest(devices=devices):
                self.assertIn(devices, orders)
                self.assertEqual(len(devices), len(settings.spare))


SMALL = ModelFacts(n_expert=0, n_layer=Layers(48), n_ctx_train=Tokens(32768), head=NoHead())


TIGHT = ModelFacts(n_expert=0, n_layer=Layers(32), n_ctx_train=Tokens(262144), head=NoHead())


class ADeviceIsAddedOnlyForTheWindowItBuys(unittest.TestCase):
    """Every device added costs speed. The fastest card alone is placed first, and each
    chain after it adds profiles only where they hold a longer window than every chain
    before it did."""

    def test_the_fastest_card_alone_is_placed_as_one_card_left_the_same_is(self):
        def shape(settings):
            return (settings.ctx, settings.cache, settings.head, settings.placement,
                    tuple(settings.spare), tuple(settings.layout.layers),
                    settings.layout.halvings)

        answer = law(DENSE)
        same = Reserves(alone=RESERVES.with_others, with_others=RESERVES.with_others)
        alone, _ = run(DENSE, place.chains(NonEmpty(two().first), (), same), answer)
        both, _ = run(DENSE, local(two()), answer)

        self.assertTrue(alone)
        self.assertEqual([shape(one) for one in alone],
                         [shape(one) for one in both if len(one.layout.devices) == 1])

    def test_every_chain_after_the_first_buys_a_longer_window_for_its_variant(self):
        for facts in (DENSE, DENSE_WITH_HEAD):
            chosen, _ = run(facts, local(two(fast=12288, slow=8192), (WORKER,)), law(facts))

            windows = {}
            for settings in chosen:
                windows.setdefault((settings.cache, settings.head), []).append(settings.ctx)
            self.assertTrue(windows)
            for variant, found in windows.items():
                with self.subTest(facts=facts.head, variant=variant):
                    self.assertEqual(sorted(set(found)), found)

    def test_a_model_whole_on_the_fastest_card_at_its_longest_runs_there_alone(self):
        chosen, answers = run(SMALL, local(two()), law(SMALL))

        self.assertTrue(chosen)
        for settings in chosen:
            with self.subTest(cache=settings.cache):
                self.assertEqual(32000, settings.ctx)
                self.assertEqual((Local(CudaIndex(0), Mib(16303)),),
                                 tuple(settings.layout.devices))
                self.assertIs(place.Among.SEVERAL, settings.layout.among)
                self.assertIs(Pipeline.OFF, settings.layout.pipeline)
                self.assertEqual(1, len(settings.spare))

    def test_the_only_card_of_a_machine_is_placed_on_as_it_always_was(self):
        chosen, answers = run(SMALL, local(NonEmpty(two().first)), law(SMALL))

        self.assertTrue(chosen)
        for question in answers:
            with self.subTest(layers=question.layout.layers):
                self.assertIs(place.Among.ONE, question.layout.among)

    def test_a_slave_adds_profiles_and_changes_none_of_the_machines_own(self):
        for facts in (DENSE, SMALL, DENSE_WITH_HEAD):
            answer = law(facts)
            without, _ = run(facts, local(two(fast=12288, slow=8192)), answer)
            with_slave, _ = run(facts, local(two(fast=12288, slow=8192), (WORKER,)), answer)

            with self.subTest(facts=facts):
                self.assertEqual(list(without),
                                 [one for one in with_slave
                                  if not place.endpoints(one.layout)])


class TheCoarseCacheAndTheHeadFollowTheCards(unittest.TestCase):
    """A coarse cache is precision given up. The fastest card alone adds it beside the
    precise one while the precise one falls short of ample, and takes it alone where the
    precise one does not fit; a chain of more devices never gives precision up, however
    short its window. Where a second card can buy window, a prediction head is never
    given up for it."""

    def test_alone_a_card_adds_a_coarse_cache_only_while_the_fine_one_falls_short(self):
        answer = law(TIGHT)
        short, _ = run(TIGHT, local(NonEmpty(card(0, 10000, Capability(12, 0), True))),
                       answer)
        roomy, _ = run(TIGHT, local(NonEmpty(card(0, 16303, Capability(12, 0), True))),
                       answer)

        self.assertEqual({CacheType.Q8_0, CacheType.Q4_0}, {one.cache for one in short})
        self.assertLess(max(one.ctx for one in short if one.cache is CacheType.Q8_0),
                        AMPLE_CTX)
        self.assertEqual([CacheType.Q8_0], [one.cache for one in roomy])

    def test_alone_a_card_the_fine_cache_does_not_fit_takes_the_coarse_one(self):
        tight = NonEmpty(card(0, 7850, Capability(12, 0), True))
        chosen, _ = run(TIGHT, local(tight), law(TIGHT))

        self.assertEqual([CacheType.Q4_0], [one.cache for one in chosen])

    def test_with_a_second_card_the_fine_cache_is_placed_across_cards_never_coarsened(self):
        """The fine cache does not fit on the fastest card: alone it would take the
        coarse one, beside a second card the model is placed on both instead."""
        chosen, _ = run(TIGHT, local(two(fast=7850, slow=8192)), law(TIGHT))

        self.assertEqual([(2, CacheType.Q8_0)],
                         [(len(one.layout.devices), one.cache) for one in chosen])

    def test_no_chain_of_several_devices_is_ever_asked_about_a_coarse_cache(self):
        for facts in (DENSE, DENSE_WITH_HEAD, TIGHT):
            _, answers = run(facts, local(two(fast=10000, slow=8192), (WORKER,)), law(facts))

            for question in answers:
                if len(question.layout.devices) > 1:
                    with self.subTest(facts=facts, devices=question.layout.devices):
                        self.assertIs(CacheType.Q8_0, question.cache)

    def test_with_a_second_card_every_placement_runs_the_head(self):
        chosen, _ = run(DENSE_WITH_HEAD, local(two()), law(DENSE_WITH_HEAD))

        self.assertTrue(chosen)
        self.assertTrue(all(one.head for one in chosen))

    def test_a_machine_with_one_card_still_offers_both(self):
        roomy = NonEmpty(card(0, 20480, Capability(12, 0), True))
        chosen, _ = run(DENSE_WITH_HEAD, local(roomy), law(DENSE_WITH_HEAD))

        self.assertEqual({False, True}, {one.head for one in chosen})


class NoDeviceIsEverOverrun(unittest.TestCase):
    def test_every_device_of_every_placement_fits(self):
        for fast, slow in ((16303, 8192), (8192, 6144)):
            chosen, _ = run(DENSE, local(two(fast, slow), (WORKER,)), law(DENSE))

            for settings in chosen:
                for spare in settings.spare:
                    with self.subTest(fast=fast, slow=slow, spare=spare):
                        self.assertGreaterEqual(spare, 0)

    def test_what_a_placement_holds_is_the_device_less_what_it_leaves(self):
        chosen, _ = run(DENSE, local(two(), (WORKER,)), law(DENSE))

        self.assertTrue(chosen)
        for settings in chosen:
            for device, spare, held in zip(settings.layout.devices, settings.spare,
                                           place.holds(settings)):
                size = device.total if isinstance(device, Local) else device.memory
                with self.subTest(device=device):
                    self.assertEqual(size - spare, held)


class TheFastestCardLandsNearestItsReserve(unittest.TestCase):
    """It takes blocks while each one brings it nearer its reserve, and never one it has
    no room for: one block more or one fewer would never have been nearer."""

    def test_neither_one_block_more_nor_one_fewer_would_have_been_nearer(self):
        answer = law(DENSE)
        for fast in (12288, 14336, 16303):
            chains = local(two(fast=fast))
            chosen, _ = run(DENSE, chains, answer)
            seat = chains.last.last

            for settings in chosen:
                if len(settings.layout.devices) == 1:
                    continue
                question = place.Question(settings.ctx, settings.cache,
                                          settings.placement, settings.layout)
                micro = place.micro_batch(UBATCH, settings.layout.halvings)
                held = settings.layout.layers.last - 1
                start = DENSE.n_layer - held

                def miss(count):
                    need, _ = row(question, DENSE.n_layer - count, count, True, micro)
                    spare = seat.available - round(need)
                    return abs(spare - seat.reserve) if spare >= 0 else None

                chosen_miss = miss(held)
                for other in (held - 1, held + 1):
                    if other < 1 or other > DENSE.n_layer - 1 or miss(other) is None:
                        continue
                    with self.subTest(fast=fast, ctx=settings.ctx, held=held, other=other):
                        self.assertLessEqual(chosen_miss, miss(other))
                self.assertEqual(start + held, DENSE.n_layer)


class ASecondCardIsWindow(unittest.TestCase):
    def test_two_cards_hold_at_least_the_window_the_faster_one_holds_alone(self):
        answer = law(DENSE)
        alone, _ = run(DENSE, local(NonEmpty(two().first)), answer)
        both, _ = run(DENSE, local(two()), answer)

        for head in (False, True):
            longest_alone = max((one.ctx for one in alone if one.head is head), default=0)
            longest_both = max((one.ctx for one in both if one.head is head), default=0)
            with self.subTest(head=head):
                self.assertGreaterEqual(longest_both, longest_alone)

    def test_a_model_too_big_for_either_card_is_placed_across_both(self):
        big = ModelFacts(n_expert=0, n_layer=Layers(96), n_ctx_train=Tokens(262144),
                         head=NoHead())
        answer = law(big)

        fast_alone, _ = run(big, local(NonEmpty(two().first)), answer)
        both, _ = run(big, local(two()), answer)

        self.assertEqual((), fast_alone)
        self.assertTrue(both)


class AMixtureFillsEveryCardBeforeSystemMemory(unittest.TestCase):
    """Experts read from system memory are slower than on any card, so a mixture leaves
    experts off the cards only once every card of the machine holds layers of it."""

    def test_a_mixture_the_fastest_card_holds_whole_runs_there_alone(self):
        chosen, _ = run(MIXTURE, local(two()), law(MIXTURE))

        self.assertEqual([((Local(CudaIndex(0), Mib(16303)),), ExpertsOnCpu(Layers(0)))],
                         [(tuple(one.layout.devices), one.placement) for one in chosen])

    def test_what_the_fastest_card_alone_would_offload_goes_on_the_next_card_instead(self):
        answer = law(MIXTURE)
        alone, _ = run(MIXTURE, local(NonEmpty(two(fast=12288).first)), answer)
        both, _ = run(MIXTURE, local(two(fast=12288)), answer)

        self.assertTrue(alone)
        self.assertTrue(all(one.placement.layers > 0 for one in alone))
        self.assertEqual([(2, ExpertsOnCpu(Layers(0)))],
                         [(len(one.layout.devices), one.placement) for one in both])

    def test_no_card_is_left_out_while_experts_are_in_system_memory(self):
        machines = (two(fast=12288), two(fast=8192, slow=6144),
                    NonEmpty(card(0, 8192, Capability(12, 0), True),
                             card(1, 6144, Capability(8, 6), False),
                             card(2, 6144, Capability(7, 5), False)))

        offloaded = []
        for cards in machines:
            chosen, _ = run(MIXTURE, local(cards, (WORKER,)), law(MIXTURE))

            self.assertTrue(chosen)
            for settings in chosen:
                if settings.placement.layers == 0:
                    continue
                offloaded.append(settings)
                own = sum(1 for one in settings.layout.devices if isinstance(one, Local))
                with self.subTest(cards=len(cards), devices=settings.layout.devices):
                    self.assertEqual(len(cards), own)

        self.assertTrue(offloaded)


class AHalvingOfTheMicroBatchHasToBuyItsShareOfWindow(unittest.TestCase):
    """What the ladder settles on, against every rung of it searched on its own."""

    def rungs(self, facts, chains, answer):
        by_rung = {}
        for halvings in (0, 1, 2):
            micro = place.micro_batch(UBATCH, halvings)
            alone = refusing(answer, lambda question, m=micro: place.micro_batch(
                UBATCH, question.layout.halvings) == m)
            chosen, _ = run(facts, chains, alone)
            by_rung[halvings] = {(one.cache, one.head): one for one in chosen}
        return by_rung

    def score(self, settings):
        window = min(settings.ctx, AMPLE_CTX)
        return (Fraction(window, AMPLE_CTX)
                - place.PREFILL_PER_HALVING * settings.layout.halvings)

    def check(self, facts, chains, answer):
        by_rung = self.rungs(facts, chains, answer)
        ladder, _ = run(facts, chains, answer)

        for settings in ladder:
            variant = (settings.cache, settings.head)
            found = [by_rung[h][variant] for h in (0, 1, 2) if variant in by_rung[h]]
            best = max(found, key=lambda one: (self.score(one), -one.layout.halvings))
            with self.subTest(variant=variant):
                self.assertEqual((best.ctx, best.layout.halvings),
                                 (settings.ctx, settings.layout.halvings))
        return ladder

    def test_the_ladder_takes_what_the_rule_says_on_one_card(self):
        for per_token in (0.00001, 0.0001):
            for total in (14336, 20480):
                with self.subTest(per_token=per_token, total=total):
                    self.check(DENSE, local(NonEmpty(card(0, total, Capability(12, 0),
                                                          True))),
                               law(DENSE, per_token_compute=per_token))

    def test_a_halving_that_makes_the_model_fit_is_taken(self):
        heavy = law(DENSE, per_token_compute=0.0001)
        chosen = self.check(DENSE, local(NonEmpty(card(0, 15500, Capability(12, 0),
                                                       True))), heavy)

        self.assertTrue(chosen)
        self.assertTrue(all(one.layout.halvings > 0 for one in chosen))

    def test_a_micro_batch_that_changes_nothing_is_never_halved(self):
        flat = law(DENSE, per_token_compute=0)

        def ignoring(question):
            same = place.Question(question.ctx, question.cache, question.placement,
                                  place.Layout(question.layout.devices,
                                               question.layout.layers,
                                               place.Halvings(0),
                                               question.layout.pipeline,
                                               question.layout.among))
            return flat(same)

        chosen, _ = run(DENSE, local(two()), ignoring)

        self.assertTrue(chosen)
        for settings in chosen:
            self.assertEqual(0, settings.layout.halvings)

    def test_nothing_is_ever_asked_below_the_smallest_micro_batch(self):
        for ubatch in (512, 256, 1024):
            _, answers = run(DENSE, local(two()), law(DENSE, per_token_compute=0.0001),
                             ubatch=ubatch)

            for question in answers:
                with self.subTest(ubatch=ubatch):
                    self.assertGreaterEqual(
                        place.micro_batch(ubatch, question.layout.halvings),
                        place.SMALLEST_UBATCH)

    def test_a_micro_batch_below_the_smallest_is_used_as_it_is(self):
        _, answers = run(DENSE, local(two()), law(DENSE), ubatch=64)

        self.assertEqual({0}, {question.layout.halvings for question in answers})

    def test_a_window_already_ample_is_never_halved_for(self):
        roomy = NonEmpty(card(0, 32768, Capability(12, 0), True))
        chosen, answers = run(DENSE, local(roomy), law(DENSE, per_token_compute=0.0001))

        self.assertTrue(chosen)
        for settings in chosen:
            self.assertGreaterEqual(settings.ctx, AMPLE_CTX)
            self.assertEqual(0, settings.layout.halvings)


class TheCardsRunPiecesOfAPromptAtOnceUnlessApartIsMuchLonger(unittest.TestCase):
    def isolated(self, facts, chains, answer, pipeline):
        """Only this way of running several cards. A question about one card is about no
        way of running them, and every search needs it answered."""
        alone = refusing(answer, lambda question: len(question.layout.devices) == 1
                         or question.layout.pipeline is pipeline)
        chosen, _ = run(facts, chains, alone)
        return {(one.cache, one.head): one.ctx for one in chosen
                if len(one.layout.devices) > 1}

    def test_apart_is_taken_exactly_where_its_window_is_long_enough(self):
        for together in (1.6, 5.0):
            answer = law(DENSE, together=together)
            chains = local(two())
            on = self.isolated(DENSE, chains, answer, Pipeline.ON)
            off = self.isolated(DENSE, chains, answer, Pipeline.OFF)
            chosen, _ = run(DENSE, chains, answer)

            for settings in chosen:
                if len(settings.layout.devices) == 1:
                    continue
                variant = (settings.cache, settings.head)
                expected = (Pipeline.OFF
                            if variant not in on
                            or off[variant] >= on[variant] * place.WORTH_RUNNING_APART
                            else Pipeline.ON)
                with self.subTest(together=together, variant=variant):
                    self.assertEqual(expected, settings.layout.pipeline)

    def test_one_card_and_a_chain_with_a_slave_are_never_told_to_run_together(self):
        _, answers = run(DENSE, local(NonEmpty(two().first), (WORKER,)), law(DENSE))

        for question in answers:
            with self.subTest(devices=question.layout.devices):
                self.assertIs(Pipeline.OFF, question.layout.pipeline)
                self.assertFalse(place.runs_apart_by_override(question.layout))

    def test_only_local_cards_are_kept_apart_by_an_override(self):
        chosen, _ = run(DENSE, local(two(), (WORKER,)), law(DENSE, together=5.0))

        for settings in chosen:
            has_slave = any(isinstance(one, Remote) for one in settings.layout.devices)
            with self.subTest(devices=settings.layout.devices):
                self.assertEqual(settings.layout.pipeline is Pipeline.OFF and not has_slave
                                 and len(settings.layout.devices) > 1,
                                 place.runs_apart_by_override(settings.layout))


class ASlaveIsOnlyWorthAProfileForALongerWindow(unittest.TestCase):
    def test_a_slave_profile_is_longer_than_the_machines_own_for_its_variant(self):
        chosen, _ = run(DENSE_WITH_HEAD, local(two(fast=12288, slow=8192), (WORKER,)),
                        law(DENSE_WITH_HEAD))

        own = {(one.cache, one.head): one.ctx for one in chosen
               if not place.endpoints(one.layout)}
        slave = [one for one in chosen if place.endpoints(one.layout)]

        self.assertTrue(slave)
        for settings in slave:
            with self.subTest(cache=settings.cache, head=settings.head):
                self.assertGreater(settings.ctx, own.get((settings.cache, settings.head), 0))

    def test_a_slave_that_buys_no_window_yields_no_profile(self):
        """A mixture holds the same window wherever it runs."""
        chosen, _ = run(MIXTURE, local(two(), (WORKER,)), law(MIXTURE))

        self.assertTrue(chosen)
        self.assertEqual([], [one for one in chosen if place.endpoints(one.layout)])

    def test_the_machines_own_profiles_come_first(self):
        chosen, _ = run(DENSE, local(two(fast=12288, slow=8192), (WORKER,)), law(DENSE))

        slaved = [bool(place.endpoints(one.layout)) for one in chosen]
        self.assertEqual(sorted(slaved), slaved)

    def test_a_slave_that_refuses_everything_leaves_the_machines_own_profiles(self):
        answer = law(DENSE)
        without, _ = run(DENSE, local(two()), answer)
        refused, _ = run(DENSE, local(two(), (WORKER,)),
                         refusing(answer, lambda question: not place.endpoints(
                             question.layout)))

        self.assertEqual(without, refused)


class DevicesAreNamedTheWayLlamaCppNamesThem(unittest.TestCase):
    def test_a_local_card_by_its_number_and_the_slave_as_the_first_rpc_device(self):
        self.assertEqual("CUDA1", place.device_name(Local(CudaIndex(1), Mib(8192))))
        self.assertEqual("RPC0", place.device_name(Remote(WORKER.endpoint, Mib(12288))))


if __name__ == "__main__":
    unittest.main()
