"""Invariants of what a profile is called.

Naming is arithmetic on a profile's own numbers, so most of these are checked on
profiles written down in the test. The one that is not is distinctness: whether two
profiles of one model can ever collide is a question about what the core produces, so it
is asked of the core, over a range of cards, rather than of a list chosen by hand.
"""

import unittest

from cm import name, place
from cm.facts import Head, ModelFacts, NoHead
from cm.name import names
from cm.nonempty import NonEmpty
from cm.place import CacheType, ExpertsOnCpu, Settings, WholeCard
from cm.units import Layers, Mib, Tokens
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


class ASingleProfileIsTheModelItself(unittest.TestCase):
    """One way to run a model needs nothing to tell it apart from."""

    def test_one_profile_is_named_by_the_key_alone(self):
        named = names("ornith-1.5-35b", (profile(131000),))

        self.assertEqual(["ornith-1.5-35b"], [one.name for one in named])

    def test_that_holds_whatever_the_numbers_are(self):
        for settings in EVERY_SHAPE:
            with self.subTest(settings=settings):
                named = names("gemma4-12b", (settings,))

                self.assertEqual("gemma4-12b", named[0].name)

    def test_no_profiles_are_named_nothing_rather_than_the_key(self):
        self.assertEqual((), names("qwen3.8-q4km", ()))


class ProfilesOfOneModelAreToldApart(unittest.TestCase):
    """Two sections of one name is a preset file that serves one of them."""

    def test_the_core_never_produces_two_profiles_of_one_name(self):
        for facts in (DENSE_WITH_HEAD, MIXTURE):
            for card in range(10000, 26001, 500):
                with self.subTest(facts=facts, card=card):
                    named = names("qwen3.8", run(facts, Mib(card)))
                    given = [one.name for one in named]

                    self.assertEqual(len(set(given)), len(given))

    def test_profiles_differing_in_any_number_get_different_names(self):
        given = [one.name for one in names("qwen3.8", EVERY_SHAPE)]

        self.assertEqual(len(EVERY_SHAPE), len(set(given)))

    def test_each_profile_is_named_once_and_keeps_what_it_says(self):
        named = names("qwen3.8", EVERY_SHAPE)

        self.assertEqual(list(EVERY_SHAPE), [one.settings for one in named])


class TheNameIsTheNumbers(unittest.TestCase):
    """Nothing outside the profile's own figures reaches its name."""

    def test_the_same_profile_among_others_is_always_named_the_same(self):
        one = profile(45000, CacheType.Q4_0, head=True)

        first = names("qwen3.8", (one, profile(25000)))
        second = names("qwen3.8", (profile(30000), one, profile(60000)))

        self.assertEqual(first[0].name, second[1].name)

    def test_the_window_appears_in_thousands(self):
        named = names("qwen3.8", (profile(45000), profile(85000)))

        self.assertIn("45k", named[0].name)
        self.assertIn("85k", named[1].name)

    def test_the_cache_appears_by_its_width(self):
        named = names("qwen3.8", (profile(45000, CacheType.Q8_0),
                                  profile(85000, CacheType.Q4_0)))

        self.assertIn("q8", named[0].name)
        self.assertNotIn("q4", named[0].name)
        self.assertIn("q4", named[1].name)
        self.assertNotIn("q8", named[1].name)

    def test_the_head_is_named_exactly_when_it_runs(self):
        named = names("qwen3.8", EVERY_SHAPE)

        for one in named:
            with self.subTest(name=one.name):
                self.assertEqual(one.settings.head, one.name.endswith("-mtp"))

    def test_a_key_that_reads_like_a_name_is_still_only_a_key(self):
        named = names("qwen3.8-45k-q8", (profile(45000), profile(85000)))

        self.assertEqual(["qwen3.8-45k-q8-45k-q8", "qwen3.8-45k-q8-85k-q8"],
                         [one.name for one in named])


if __name__ == "__main__":
    unittest.main()
