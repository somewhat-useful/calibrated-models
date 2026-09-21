"""Invariants of what a profile is called.

Naming is arithmetic on a profile's own numbers, so most of these are checked on
profiles written down in the test. The one that is not is distinctness: whether two
profiles of one model can ever collide is a question about what the core produces, so it
is asked of the core, over a range of cards, rather than of a list chosen by hand.
"""

import unittest

from cm import name, place
from cm.facts import Head, ModelFacts, NoHead
from cm.machine import CudaIndex
from cm.name import names
from cm.nonempty import NonEmpty
from cm.place import (Among, CacheType, Endpoint, ExpertsOnCpu, Layout, Local, Pipeline,
                      Remote, Settings, WholeCard)
from cm.units import Halvings, Layers, Mib, Port, Tokens
from one_card import LAYOUT, UBATCH, chains, needs

RESERVE = Mib(1024)
MIN_CTX = Tokens(25000)
AMPLE_CTX = Tokens(100000)

DENSE_WITH_HEAD = ModelFacts(n_expert=0, n_layer=Layers(64), n_ctx_train=Tokens(262144),
                             head=Head(weight_bytes=1_400_000_000,
                                       cache_per_token=2048))

MIXTURE = ModelFacts(n_expert=128, n_layer=Layers(40), n_ctx_train=Tokens(262144),
                     head=NoHead())


def law(question):
    """What the estimator would answer if cost rose evenly along the lever."""
    share = question.cache.bytes_per_element / CacheType.Q8_0.bytes_per_element
    needed = 8000 + 0.06 * question.ctx * share
    if isinstance(question.placement, ExpertsOnCpu):
        needed -= 150 * question.placement.layers
    return needs(Mib(round(needed)), Mib(0))


def run(facts, card):
    """Drive the core the way cli.py will, and hand back what it settled on."""
    limits = place.limits_for(chains(card, RESERVE), UBATCH, MIN_CTX, AMPLE_CTX)
    answers = {}

    for _ in range(40):
        asking = place.next_questions(facts, place.EVERYTHING, limits, answers)
        if not asking:
            break
        for question in asking:
            answers[question] = law(question)
    else:
        raise AssertionError("the search did not settle")

    return place.settings(facts, place.EVERYTHING, limits, answers)


def profile(ctx, cache=CacheType.Q8_0, head=False):
    return Settings(ctx=Tokens(ctx), cache=cache, head=head,
                    placement=WholeCard(), spare=NonEmpty(Mib(1024)), layout=LAYOUT)


EVERY_SHAPE = tuple(profile(ctx, cache, head)
                    for ctx in (25000, 45000)
                    for cache in (CacheType.Q8_0, CacheType.Q4_0)
                    for head in (False, True))

FAST = Local(CudaIndex(0), Mib(16303))
SLOW = Local(CudaIndex(1), Mib(8192))
SLOWEST = Local(CudaIndex(2), Mib(6144))
SLAVE = Remote(Endpoint("worker", Port(50052)), Mib(12288))

# The ways a dense file carrying a head may run, and the ways of a mixture, which never
# runs one.
HEADED = place.variants(DENSE_WITH_HEAD, place.EVERYTHING)
HEADLESS = place.variants(MIXTURE, place.EVERYTHING)


def across(devices, ctx, cache=CacheType.Q8_0, head=False):
    """A profile over several devices, the way a machine with more than one card lays
    them out."""
    return Settings(ctx=Tokens(ctx), cache=cache, head=head,
                    placement=WholeCard(),
                    spare=NonEmpty(*(Mib(1024) for _ in devices)),
                    layout=Layout(devices=NonEmpty(*devices),
                                  layers=NonEmpty(*(Layers(20) for _ in devices)),
                                  halvings=Halvings(0), pipeline=Pipeline.OFF,
                                  among=Among.SEVERAL))


class ASingleProfileIsTheModelItself(unittest.TestCase):
    """One way to run a model needs no window to tell it apart from the others."""

    def test_one_profile_giving_nothing_up_is_named_by_the_key_alone(self):
        named = names("ornith-1.5-35b", (profile(131000),), HEADLESS)

        self.assertEqual(["ornith-1.5-35b"], [one.name for one in named])

    def test_one_profile_carries_no_window_whatever_it_holds(self):
        for settings in EVERY_SHAPE:
            with self.subTest(settings=settings):
                named = names("gemma4-12b", (settings,), HEADED)

                self.assertNotIn(f"{settings.ctx // name.THOUSAND}k", named[0].name)

    def test_what_one_profile_gives_up_is_still_said(self):
        named = names("gemma4-12b", (profile(25000, CacheType.Q4_0),), HEADED)

        self.assertEqual(["gemma4-12b-q4-nomtp"], [one.name for one in named])

    def test_no_profiles_are_named_nothing_rather_than_the_key(self):
        self.assertEqual((), names("qwen3.8-q4km", (), HEADED))


class ProfilesOfOneModelAreToldApart(unittest.TestCase):
    """Two sections of one name is a preset file that serves one of them."""

    def test_the_core_never_produces_two_profiles_of_one_name(self):
        for facts in (DENSE_WITH_HEAD, MIXTURE):
            for card in range(10000, 26001, 500):
                with self.subTest(facts=facts, card=card):
                    named = names("qwen3.8", run(facts, Mib(card)),
                                  place.variants(facts, place.EVERYTHING))
                    given = [one.name for one in named]

                    self.assertEqual(len(set(given)), len(given))

    def test_profiles_differing_in_any_number_get_different_names(self):
        given = [one.name for one in names("qwen3.8", EVERY_SHAPE, HEADED)]

        self.assertEqual(len(EVERY_SHAPE), len(set(given)))

    def test_each_profile_is_named_once_and_keeps_what_it_says(self):
        named = names("qwen3.8", EVERY_SHAPE, HEADED)

        self.assertEqual(list(EVERY_SHAPE), [one.settings for one in named])


class TheNameIsTheNumbers(unittest.TestCase):
    """Nothing outside the profile's own figures reaches its name, besides whether the
    file could have run a head."""

    def test_the_same_profile_among_others_is_always_named_the_same(self):
        one = profile(45000, CacheType.Q4_0, head=True)

        first = names("qwen3.8", (one, profile(25000)), HEADED)
        second = names("qwen3.8", (profile(30000), one, profile(60000)), HEADED)

        self.assertEqual(first[0].name, second[1].name)

    def test_the_window_appears_in_thousands(self):
        named = names("qwen3.8", (profile(45000), profile(85000)), HEADLESS)

        self.assertIn("45k", named[0].name)
        self.assertIn("85k", named[1].name)

    def test_a_coarser_cache_is_said_and_q8_is_not(self):
        named = names("qwen3.8", (profile(45000, CacheType.Q8_0),
                                  profile(85000, CacheType.Q4_0)), HEADLESS)

        self.assertEqual(["qwen3.8-45k", "qwen3.8-85k-q4"], [one.name for one in named])

    def test_a_cache_finer_than_q8_is_said_too(self):
        named = names("qwen3.8", (profile(25000, CacheType.F16),
                                  profile(45000, CacheType.Q8_0)), HEADLESS)

        self.assertEqual(["qwen3.8-25k-f16", "qwen3.8-45k"], [one.name for one in named])

    def test_a_head_left_out_is_said_exactly_where_it_does_not_run(self):
        for one in names("qwen3.8", EVERY_SHAPE, HEADED):
            with self.subTest(name=one.name):
                self.assertEqual(not one.settings.head, one.name.endswith("-nomtp"))

    def test_a_head_that_runs_is_not_said(self):
        for one in names("qwen3.8", EVERY_SHAPE, HEADED):
            with self.subTest(name=one.name):
                self.assertNotIn("-mtp", one.name)

    def test_nothing_is_left_out_where_the_file_could_run_no_head(self):
        named = names("ornith-1.5-35b", (profile(45000), profile(85000, CacheType.Q4_0)),
                      HEADLESS)

        self.assertEqual(["ornith-1.5-35b-45k", "ornith-1.5-35b-85k-q4"],
                         [one.name for one in named])

    def test_nothing_is_left_out_where_the_head_is_ruled_out_by_hand(self):
        ruled_out = place.variants(DENSE_WITH_HEAD,
                                   place.Allowed(caches=place.EVERYTHING.caches,
                                                 head=False))
        named = names("qwen3.8", (profile(45000), profile(85000, CacheType.Q4_0)),
                      ruled_out)

        self.assertEqual(["qwen3.8-45k", "qwen3.8-85k-q4"], [one.name for one in named])

    def test_what_is_given_up_comes_in_order_after_the_window(self):
        named = names("qwen3.8", (profile(57000, CacheType.Q4_0),
                                  across((SLOW, FAST), 120000)), HEADED)

        self.assertEqual(["qwen3.8-57k-q4-nomtp", "qwen3.8-120k-nomtp-2gpu"],
                         [one.name for one in named])

    def test_a_key_that_reads_like_a_name_is_still_only_a_key(self):
        named = names("qwen3.8-45k-q8", (profile(45000), profile(85000)), HEADLESS)

        self.assertEqual(["qwen3.8-45k-q8-45k", "qwen3.8-45k-q8-85k"],
                         [one.name for one in named])


class TheCardsPastTheFirstAreSaid(unittest.TestCase):
    """Every card a profile adds to the fastest one costs speed, and a card reached over
    the network costs the most."""

    def test_one_card_says_nothing(self):
        named = names("qwen3.8", (profile(25000, head=True),
                                  across((SLOW, FAST), 60000, head=True)), HEADED)

        self.assertEqual(["qwen3.8-25k", "qwen3.8-60k-2gpu"], [one.name for one in named])

    def test_this_machines_cards_are_counted(self):
        named = names("qwen3.8", (across((SLOW, FAST), 60000, head=True),
                                  across((SLOWEST, SLOW, FAST), 90000, head=True)),
                      HEADED)

        self.assertEqual(["qwen3.8-60k-2gpu", "qwen3.8-90k-3gpu"],
                         [one.name for one in named])

    def test_one_profile_on_two_of_this_machines_cards_says_two(self):
        named = names("ornith-1.5-35b", (across((SLOW, FAST), 150000),), HEADLESS)

        self.assertEqual(["ornith-1.5-35b-2gpu"], [one.name for one in named])

    def test_a_name_ends_in_rpc_exactly_when_a_slaves_card_is_used(self):
        chosen = (profile(25000, head=True), across((SLOW, FAST), 60000, head=True),
                  across((SLAVE, FAST), 90000, head=True),
                  across((SLAVE, SLOW, FAST), 120000, head=True))

        for one in names("qwen3.8", chosen, HEADED):
            with self.subTest(name=one.name):
                self.assertEqual(isinstance(one.settings.layout.devices.first, Remote),
                                 one.name.endswith("-rpc"))

    def test_a_slaves_card_is_all_that_is_said_of_the_cards(self):
        named = names("qwen3.8", (profile(25000, head=True),
                                  across((SLAVE, SLOW, FAST), 120000, head=True)), HEADED)

        self.assertEqual(["qwen3.8-25k", "qwen3.8-120k-rpc"], [one.name for one in named])

    def test_rpc_is_the_last_thing_in_a_name(self):
        named = names("qwen3.8", (profile(25000, head=True),
                                  across((SLAVE, FAST), 120000, CacheType.Q4_0)), HEADED)

        self.assertEqual("qwen3.8-120k-q4-nomtp-rpc", named[1].name)

    def test_a_single_profile_on_a_slaves_card_is_the_key_and_rpc(self):
        named = names("ornith-1.5-35b", (across((SLAVE, SLOW, FAST), 150000),), HEADLESS)

        self.assertEqual(["ornith-1.5-35b-rpc"], [one.name for one in named])

    def test_the_same_numbers_here_and_across_a_slave_are_two_names(self):
        named = names("qwen3.8", (across((SLOW, FAST), 120000),
                                  across((SLAVE, FAST), 120000)), HEADLESS)

        self.assertEqual(["qwen3.8-120k-2gpu", "qwen3.8-120k-rpc"],
                         [one.name for one in named])


if __name__ == "__main__":
    unittest.main()
