"""Invariants of choosing where a model sits on this machine's card.

The estimator is not here. Every test answers the questions the core asks from a law
written in the test itself, so the answers are arbitrary on purpose: an invariant that
only holds for one model's numbers is not an invariant.

The strongest of them is checked by brute force. The core visits a handful of points on
the grid; the test walks every one of them and asserts that nothing the core skipped
would have been a better answer.
"""

import math
import unittest

from cm import place
from cm.estimate import Refused
from cm.facts import Head, ModelFacts, NoHead
from cm.nonempty import NonEmpty
from cm.place import CacheType, ExpertsOnCpu, Question, WholeCard
from cm.units import Layers, Mib, Tokens
from one_card import LAYOUT, UBATCH, chains, needs

CARD = Mib(16303)
RESERVE = Mib(1024)
MIN_CTX = Tokens(25000)
AMPLE_CTX = Tokens(100000)

DENSE = ModelFacts(n_expert=0, n_layer=Layers(64), n_ctx_train=Tokens(262144),
                   head=NoHead())

DENSE_WITH_HEAD = ModelFacts(n_expert=0, n_layer=Layers(64), n_ctx_train=Tokens(262144),
                             head=Head(weight_bytes=1_400_000_000,
                                       cache_per_token=2048))

SHORT_TRAIN = ModelFacts(n_expert=0, n_layer=Layers(8), n_ctx_train=Tokens(20000),
                         head=NoHead())

MIXTURE = ModelFacts(n_expert=128, n_layer=Layers(40), n_ctx_train=Tokens(262144),
                     head=NoHead())

# A search that has not finished after this many rounds is not converging, and a test
# that waits for it forever reports nothing.
ROUND_LIMIT = 40


def straight_line(fixed, per_token, per_offloaded_layer=0):
    """What the estimator would answer if cost rose evenly along the lever.

    The real curve is not this tidy -- the compute buffers hold flat and then jump --
    which is why the core searches instead of fitting a line. Monotonicity is all that
    is shared between the two, and it is all these tests rely on.
    """

    def law(question):
        # per_token is quoted at q8_0; a coarser cache costs its share of that.
        share = (question.cache.bytes_per_element
                 / CacheType.Q8_0.bytes_per_element)
        needed = fixed + per_token * question.ctx * share
        held = 0.0
        if isinstance(question.placement, ExpertsOnCpu):
            # What leaves the card arrives in system memory. The fake says so, or a
            # test could not tell a placement that offloads from one that does not.
            needed -= per_offloaded_layer * question.placement.layers
            held = per_offloaded_layer * question.placement.layers
        return needs(Mib(round(needed)), Mib(round(held)))

    return law


def run(facts, card, reserve, law, min_ctx=MIN_CTX, ample_ctx=AMPLE_CTX,
        allowed=place.EVERYTHING):
    """Drive the core the way cli.py will: ask what it asks, hand back the answers."""
    limits = place.limits_for(chains(card, reserve), UBATCH, min_ctx, ample_ctx)
    answers = {}

    for _ in range(ROUND_LIMIT):
        asking = place.next_questions(facts, allowed, limits, answers)
        if not asking:
            break
        for question in asking:
            answers[question] = law(question)
    else:
        raise AssertionError(f"the search did not settle in {ROUND_LIMIT} rounds")

    return place.settings(facts, allowed, limits, answers), answers, limits


def every_point(facts, limits, variant, law):
    """What each point of the grid would leave the card, if it fits at all."""
    for point in place.grid(facts, limits):
        question = Question(point.ctx, variant.cache, point.placement, LAYOUT)
        needed = place.requirement(facts, variant, point.ctx, law(question)).first
        spare = limits.chains.first.first.available - needed
        if spare >= 0:
            yield point, spare


class TheCardIsNeverExceeded(unittest.TestCase):
    """The reserve is a target and may be missed from either side. The card may not.

    A placement needing more than the card holds does not run slowly, it fails to
    allocate, so there is no distance from the target at which it becomes a candidate.
    """

    def test_no_settings_ever_spends_more_than_the_card_has(self):
        law = straight_line(fixed=12000, per_token=0.021)

        for card in range(8000, 24001, 500):
            with self.subTest(card=card):
                chosen, _, _ = run(DENSE, Mib(card), RESERVE, law)

                for settings in chosen:
                    self.assertGreaterEqual(settings.spare.first, 0)

    def test_a_card_too_small_for_the_weights_yields_nothing(self):
        """Dense is whole on the card or not served: there is no third answer."""
        law = straight_line(fixed=12000, per_token=0.021)

        chosen, _, _ = run(DENSE, Mib(4096), RESERVE, law)

        self.assertEqual(chosen, ())


class NothingOnTheGridWouldHaveBeenBetter(unittest.TestCase):
    """Checked against every point, not against the few the search visited."""

    def test_the_chosen_point_misses_the_reserve_by_the_least(self):
        law = straight_line(fixed=12000, per_token=0.021)

        for card in range(13000, 24001, 250):
            limits = place.limits_for(chains(Mib(card), RESERVE), UBATCH, MIN_CTX, AMPLE_CTX)
            chosen, _, _ = run(DENSE_WITH_HEAD, Mib(card), RESERVE, law)

            for settings in chosen:
                variant = place.Variant(settings.cache, settings.head)
                best = min(abs(spare - RESERVE)
                           for _, spare in every_point(DENSE_WITH_HEAD, limits,
                                                       variant, law))
                with self.subTest(card=card, cache=settings.cache, head=settings.head):
                    self.assertEqual(abs(settings.spare.first - RESERVE), best)

    def test_the_finest_cache_yields_nothing_only_when_no_point_fits(self):
        """Stated of q8_0 alone: a coarser cache is dropped for its own reasons."""
        law = straight_line(fixed=12000, per_token=0.021)
        variant = place.Variant(CacheType.Q8_0, head=False)

        for card in range(8000, 24001, 250):
            limits = place.limits_for(chains(Mib(card), RESERVE), UBATCH, MIN_CTX, AMPLE_CTX)
            chosen, _, _ = run(DENSE, Mib(card), RESERVE, law)
            offered = {(s.cache, s.head) for s in chosen}

            fits = any(every_point(DENSE, limits, variant, law))
            with self.subTest(card=card):
                self.assertEqual((variant.cache, variant.head) in offered, fits)


class TheSearchIsCheap(unittest.TestCase):
    def test_it_asks_about_a_handful_of_points_not_all_of_them(self):
        """Two ends and a bisection: the log of the grid, not its length."""
        law = straight_line(fixed=12000, per_token=0.021)
        limits = place.limits_for(chains(CARD, RESERVE), UBATCH, MIN_CTX, AMPLE_CTX)
        points = len(place.grid(DENSE_WITH_HEAD, limits))

        _, answers, _ = run(DENSE_WITH_HEAD, CARD, RESERVE, law)

        per_variant = 2 + math.ceil(math.log2(points))
        offered = place.variants(DENSE_WITH_HEAD, place.EVERYTHING)
        self.assertLessEqual(len(answers), per_variant * len(offered))
        self.assertLess(len(answers), points)


class DenseStaysWholeOnTheCard(unittest.TestCase):
    def test_every_dense_settings_keeps_all_layers_on_the_card(self):
        law = straight_line(fixed=12000, per_token=0.021)

        for card in range(12000, 24001, 500):
            with self.subTest(card=card):
                chosen, _, _ = run(DENSE, Mib(card), RESERVE, law)

                for settings in chosen:
                    self.assertEqual(settings.placement, WholeCard())

    def test_no_question_about_a_dense_model_offloads_anything(self):
        law = straight_line(fixed=12000, per_token=0.021)

        _, answers, _ = run(DENSE, CARD, RESERVE, law)

        for question in answers:
            self.assertEqual(question.placement, WholeCard())


class AHeadRunsOnlyOnAWholeCard(unittest.TestCase):
    def test_head_implies_the_whole_model_is_on_the_card(self):
        law = straight_line(fixed=12000, per_token=0.021, per_offloaded_layer=400)

        for facts in (DENSE_WITH_HEAD, MIXTURE):
            with self.subTest(model="mixture" if facts.n_expert else "dense"):
                chosen, _, _ = run(facts, CARD, RESERVE, law)

                for settings in chosen:
                    if settings.head:
                        self.assertEqual(settings.placement, WholeCard())

    def test_a_mixture_never_runs_one(self):
        """Its context refuses to be built with experts off the card."""
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)

        chosen, _, _ = run(MIXTURE, CARD, RESERVE, law)

        for settings in chosen:
            self.assertFalse(settings.head)

    def test_a_file_without_a_head_never_produces_one(self):
        law = straight_line(fixed=12000, per_token=0.021)

        chosen, _, _ = run(DENSE, CARD, RESERVE, law)

        self.assertTrue(chosen)
        for settings in chosen:
            self.assertFalse(settings.head)

    def test_a_file_with_a_head_produces_both_answers(self):
        """With and without: the head roughly doubles speed or the window, not both."""
        law = straight_line(fixed=12000, per_token=0.021)

        chosen, _, _ = run(DENSE_WITH_HEAD, CARD, RESERVE, law)

        self.assertEqual({settings.head for settings in chosen}, {True, False})

    def test_a_head_costs_window(self):
        """Its weights and its own cache come out of the same card."""
        law = straight_line(fixed=12000, per_token=0.05)

        chosen, _, _ = run(DENSE_WITH_HEAD, CARD, RESERVE, law)

        by_variant = {(s.cache, s.head): s.ctx for s in chosen}
        for cache in (CacheType.Q8_0, CacheType.Q4_0):
            with self.subTest(cache=cache):
                self.assertLess(by_variant[(cache, True)], by_variant[(cache, False)])


class ARefusalIsNotAnAmount(unittest.TestCase):
    def test_refusing_everything_gives_no_settings_rather_than_an_error(self):
        chosen, _, _ = run(DENSE, CARD, RESERVE, lambda question: Refused())

        self.assertEqual(chosen, ())

    def test_a_refusal_ends_a_variant_instead_of_being_counted_as_free(self):
        law = straight_line(fixed=12000, per_token=0.021)

        def refuses_q4(question):
            return Refused() if question.cache is CacheType.Q4_0 else law(question)

        chosen, _, _ = run(DENSE, CARD, RESERVE, refuses_q4)

        self.assertEqual({settings.cache for settings in chosen}, {CacheType.Q8_0})
        for settings in chosen:
            self.assertGreaterEqual(settings.spare.first, 0)


class TheWindowStaysWithinWhatTheModelWasTrainedFor(unittest.TestCase):
    def test_no_settings_asks_for_more_than_n_ctx_train(self):
        """A card with room to spare past the ceiling buys nothing past it."""
        law = straight_line(fixed=500, per_token=0.0001)

        for facts in (DENSE, SHORT_TRAIN):
            for card in range(8000, 24001, 2000):
                with self.subTest(train=facts.n_ctx_train, card=card):
                    chosen, _, _ = run(facts, Mib(card), RESERVE, law)

                    for settings in chosen:
                        self.assertLessEqual(settings.ctx, facts.n_ctx_train)

    def test_a_ceiling_reached_is_not_read_as_not_fitting(self):
        """Room left over past the trained window is a fit, not a failure."""
        law = straight_line(fixed=500, per_token=0.0001)

        chosen, _, _ = run(SHORT_TRAIN, CARD, RESERVE, law)

        self.assertTrue(chosen)

    def test_the_window_stays_on_the_page_even_at_the_ceiling(self):
        """The trained maximum is not a round number, and the window still is one.

        262144 tokens of training buys a window of 262000: the 144 left over are worth
        less than a number a person can read.
        """
        law = straight_line(fixed=500, per_token=0.0001)

        chosen, _, _ = run(DENSE, CARD, RESERVE, law)

        self.assertTrue(chosen)
        for settings in chosen:
            self.assertEqual(settings.ctx % place.PAGE, 0)
            self.assertEqual(settings.ctx, Tokens(262000))


class EveryWindowIsWrittenInThousands(unittest.TestCase):
    """A window is a count of tokens, and the count is one a person can read.

    A step of a thousand costs some twenty megabytes against a reserve of a thousand,
    so the rounding is free and it applies to every window the core yields -- the ones
    it searched for, the ones a trained ceiling handed it, and the fixed one a mixture
    holds.
    """

    def test_every_settings_lands_on_the_page(self):
        law = straight_line(fixed=8000, per_token=0.06, per_offloaded_layer=400)
        odd_mixture = ModelFacts(n_expert=128, n_layer=Layers(40),
                                 n_ctx_train=Tokens(40960), head=NoHead())

        for facts in (DENSE, DENSE_WITH_HEAD, SHORT_TRAIN, MIXTURE, odd_mixture):
            for card in range(10000, 26001, 1000):
                with self.subTest(train=facts.n_ctx_train, card=card):
                    chosen, _, _ = run(facts, Mib(card), RESERVE, law)

                    for settings in chosen:
                        self.assertEqual(settings.ctx % place.PAGE, 0)

    def test_no_question_asks_about_a_window_off_the_page(self):
        """Not one estimator call is spent on a window that could never be written."""
        law = straight_line(fixed=8000, per_token=0.06, per_offloaded_layer=400)

        for facts in (DENSE_WITH_HEAD, MIXTURE):
            _, answers, _ = run(facts, CARD, RESERVE, law)

            for question in answers:
                self.assertEqual(question.ctx % place.PAGE, 0)


class AWindowTooShortIsNotAProfile(unittest.TestCase):
    """A few thousand tokens is not a cheaper way to run the model, it is a useless one.

    Where the line falls is the human's call, so it comes from the settings file; that
    it is enforced at all is not.
    """

    def test_no_settings_is_shorter_than_the_floor(self):
        law = straight_line(fixed=12000, per_token=0.021)

        for card in range(8000, 24001, 250):
            with self.subTest(card=card):
                chosen, _, _ = run(DENSE, Mib(card), RESERVE, law)

                for settings in chosen:
                    self.assertGreaterEqual(settings.ctx, MIN_CTX)

    def test_a_card_holding_only_a_short_window_yields_nothing(self):
        """The weights fit and a four-thousand-token window fits. That is not a fit."""
        law = straight_line(fixed=12000, per_token=0.021)

        chosen, _, limits = run(DENSE, Mib(12440), RESERVE, law)

        self.assertGreaterEqual(limits.chains.first.first.available,
                                12000 + round(0.021 * 4096))
        self.assertEqual(chosen, ())

    def test_the_floor_moves_with_the_setting(self):
        """Asking for less makes a card that served nothing serve something."""
        law = straight_line(fixed=12000, per_token=0.021)

        strict, _, _ = run(DENSE, Mib(12440), RESERVE, law, min_ctx=Tokens(25000))
        lenient, _, _ = run(DENSE, Mib(12440), RESERVE, law, min_ctx=Tokens(10000))

        self.assertEqual(strict, ())
        self.assertTrue(lenient)
        for settings in lenient:
            self.assertGreaterEqual(settings.ctx, 10000)

    def test_a_model_trained_shorter_than_the_floor_is_still_served(self):
        """It cannot be asked for more than it knows, so its ceiling is its floor."""
        law = straight_line(fixed=500, per_token=0.0001)

        chosen, _, _ = run(SHORT_TRAIN, CARD, RESERVE, law)

        self.assertTrue(chosen)
        for settings in chosen:
            self.assertEqual(settings.ctx, Tokens(20000))


class ACoarserCacheHasToBuySomething(unittest.TestCase):
    """It is precision given up. Given up for nothing, it is not a profile.

    What it can buy is a longer window, or a model fitting at all. Two profiles at the
    same window differing only in how exactly they hold the conversation are not a
    choice anyone can make.
    """

    def test_the_same_window_at_a_coarser_cache_is_dropped(self):
        """Both run into the trained ceiling, so q4 bought no tokens at all."""
        law = straight_line(fixed=500, per_token=0.0001)

        chosen, _, _ = run(DENSE, CARD, RESERVE, law)

        self.assertEqual([settings.cache for settings in chosen], [CacheType.Q8_0])

    def test_a_longer_window_keeps_it(self):
        law = straight_line(fixed=12000, per_token=0.05)

        chosen, _, _ = run(DENSE, CARD, RESERVE, law)

        windows = {settings.cache: settings.ctx for settings in chosen}
        self.assertEqual(set(windows), {CacheType.Q8_0, CacheType.Q4_0})
        self.assertGreater(windows[CacheType.Q4_0], windows[CacheType.Q8_0])

    def test_it_stands_alone_when_the_finer_one_did_not_fit(self):
        """Nothing to compare against: the coarse cache is what makes this servable."""
        law = straight_line(fixed=12000, per_token=0.021)

        chosen, _, _ = run(DENSE, Mib(12440), RESERVE, law, min_ctx=Tokens(10000))

        self.assertEqual([settings.cache for settings in chosen], [CacheType.Q4_0])

    def test_the_rule_is_applied_within_a_head_setting_not_across_them(self):
        """A head is a different profile, not a coarser one."""
        law = straight_line(fixed=12000, per_token=0.05)

        chosen, _, _ = run(DENSE_WITH_HEAD, CARD, RESERVE, law)

        self.assertEqual(
            {(settings.cache, settings.head) for settings in chosen},
            {(CacheType.Q8_0, False), (CacheType.Q4_0, False),
             (CacheType.Q8_0, True), (CacheType.Q4_0, True)})

    def test_an_ample_window_at_the_finer_cache_ends_the_matter(self):
        """q8 already reaches past ample, so the longer q4 window buys nothing wanted."""
        law = straight_line(fixed=12000, per_token=0.021)

        chosen, _, _ = run(DENSE, CARD, RESERVE, law)

        self.assertEqual([settings.cache for settings in chosen], [CacheType.Q8_0])
        self.assertGreater(chosen[0].ctx, AMPLE_CTX)

    def test_the_threshold_moves_with_the_setting(self):
        """Say a longer window is still wanted, and the coarser cache comes back."""
        law = straight_line(fixed=12000, per_token=0.021)

        chosen, _, _ = run(DENSE, CARD, RESERVE, law, ample_ctx=Tokens(200000))

        self.assertEqual({settings.cache for settings in chosen},
                         {CacheType.Q8_0, CacheType.Q4_0})

    def test_no_two_profiles_ever_share_a_window_and_a_head(self):
        law = straight_line(fixed=12000, per_token=0.021)
        ceiling_bound = straight_line(fixed=500, per_token=0.0001)

        for card in range(13000, 24001, 250):
            for which in (law, ceiling_bound):
                chosen, _, _ = run(DENSE_WITH_HEAD, Mib(card), RESERVE, which)
                offered = [(settings.ctx, settings.head) for settings in chosen]

                with self.subTest(card=card):
                    self.assertEqual(len(offered), len(set(offered)))


class WhatAPersonRulesOutStaysRuledOut(unittest.TestCase):
    """Quality at a coarser cache, and whether a head earns its verification, are not
    things any memory figure shows. They are ruled out by hand or not at all.
    """

    def test_forbidding_the_coarse_cache_drops_it_even_where_it_would_buy_window(self):
        law = straight_line(fixed=12000, per_token=0.05)
        only_precise = place.Allowed(caches=frozenset({CacheType.Q8_0}), head=True)

        with_it, _, _ = run(DENSE, CARD, RESERVE, law)
        without, _, _ = run(DENSE, CARD, RESERVE, law, allowed=only_precise)

        self.assertIn(CacheType.Q4_0, {settings.cache for settings in with_it})
        self.assertEqual([settings.cache for settings in without], [CacheType.Q8_0])

    def test_forbidding_the_head_drops_it_even_where_the_file_carries_one(self):
        law = straight_line(fixed=12000, per_token=0.05)
        no_head = place.Allowed(caches=place.EVERYTHING.caches, head=False)

        with_it, _, _ = run(DENSE_WITH_HEAD, CARD, RESERVE, law)
        without, _, _ = run(DENSE_WITH_HEAD, CARD, RESERVE, law, allowed=no_head)

        self.assertIn(True, {settings.head for settings in with_it})
        self.assertEqual({settings.head for settings in without}, {False})

    def test_forbidding_both_leaves_one_profile(self):
        law = straight_line(fixed=12000, per_token=0.05)
        plainest = place.Allowed(caches=frozenset({CacheType.Q8_0}), head=False)

        chosen, _, _ = run(DENSE_WITH_HEAD, CARD, RESERVE, law, allowed=plainest)

        self.assertEqual(len(chosen), 1)
        self.assertEqual((chosen[0].cache, chosen[0].head), (CacheType.Q8_0, False))

    def test_permitting_a_head_a_file_has_not_got_yields_none(self):
        """Permission narrows; it cannot add what the quantisation dropped."""
        law = straight_line(fixed=12000, per_token=0.05)

        chosen, _, _ = run(DENSE, CARD, RESERVE, law)

        self.assertEqual({settings.head for settings in chosen}, {False})

    def test_a_mixture_takes_the_most_precise_cache_still_permitted(self):
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)
        only_coarse = place.Allowed(caches=frozenset({CacheType.Q4_0}), head=True)

        chosen, _, _ = run(MIXTURE, CARD, RESERVE, law, allowed=only_coarse)

        self.assertEqual([settings.cache for settings in chosen], [CacheType.Q4_0])

    def test_forbidding_every_cache_yields_nothing_and_asks_nothing(self):
        law = straight_line(fixed=12000, per_token=0.05)
        nothing = place.Allowed(caches=frozenset(), head=True)

        chosen, answers, _ = run(DENSE, CARD, RESERVE, law, allowed=nothing)

        self.assertEqual(chosen, ())
        self.assertEqual(answers, {})


class AMixtureKeepsItsWindowAndMovesExperts(unittest.TestCase):
    def test_the_window_is_the_same_whatever_the_card(self):
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)

        windows = set()
        for card in (Mib(8192), Mib(12288), Mib(16303)):
            chosen, _, _ = run(MIXTURE, card, RESERVE, law)
            windows.update(settings.ctx for settings in chosen)

        self.assertEqual(windows, {place.MIXTURE_WINDOW})

    def test_a_shorter_trained_window_is_not_exceeded(self):
        short = ModelFacts(n_expert=128, n_layer=Layers(40),
                           n_ctx_train=Tokens(40960), head=NoHead())
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)

        chosen, _, _ = run(short, CARD, RESERVE, law)

        for settings in chosen:
            self.assertEqual(settings.ctx, Tokens(40000))

    def test_a_smaller_card_moves_more_experts(self):
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)

        offloaded = []
        for card in (Mib(16303), Mib(12288), Mib(8192)):
            chosen, _, _ = run(MIXTURE, card, RESERVE, law)
            self.assertEqual(len(chosen), 1)
            offloaded.append(chosen[0].placement.layers)

        self.assertEqual(offloaded, sorted(offloaded))
        self.assertLess(offloaded[0], offloaded[-1])

    def test_the_least_offload_that_lands_nearest_is_taken(self):
        """Every step above it moves experts the card had room for into system RAM."""
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)
        limits = place.limits_for(chains(CARD, RESERVE), UBATCH, MIN_CTX, AMPLE_CTX)

        chosen, _, _ = run(MIXTURE, CARD, RESERVE, law)

        variant = place.variants(MIXTURE, place.EVERYTHING)[0]
        best = min(abs(spare - RESERVE)
                   for _, spare in every_point(MIXTURE, limits, variant, law))
        self.assertEqual(abs(chosen[0].spare.first - RESERVE), best)

    def test_a_mixture_produces_exactly_one_placement(self):
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)

        chosen, _, _ = run(MIXTURE, CARD, RESERVE, law)

        self.assertEqual(len(chosen), 1)
        self.assertIsInstance(chosen[0].placement, ExpertsOnCpu)


class WhatStaysInSystemMemoryIsCountedToo(unittest.TestCase):
    """The prompt cache is sized around it, so a placement has two halves worth reading.

    The router holds one model at a time, so what matters of a set of profiles is the
    heaviest of them: the others cost nothing while it is loaded.
    """

    def test_a_model_that_sits_on_the_card_whole_holds_nothing(self):
        law = straight_line(fixed=12000, per_token=0.021)
        chosen, answers, _ = run(DENSE, CARD, RESERVE, law)

        self.assertEqual(Mib(0), place.resident(chosen, answers))

    def test_a_mixture_holds_what_it_moved_off_the_card(self):
        law = straight_line(fixed=20000, per_token=0.011, per_offloaded_layer=400)
        chosen, answers, _ = run(MIXTURE, CARD, RESERVE, law)

        offloaded = chosen[0].placement.layers

        self.assertEqual(Mib(400 * offloaded), place.resident(chosen, answers))

    def test_the_heaviest_profile_is_the_one_that_counts(self):
        light = Question(Tokens(30000), CacheType.Q8_0, WholeCard(), LAYOUT)
        heavy = Question(Tokens(60000), CacheType.Q8_0, WholeCard(), LAYOUT)
        answers = {light: needs(Mib(9000), Mib(600)),
                   heavy: needs(Mib(12000), Mib(4000))}

        chosen = [place.Settings(ctx=question.ctx, cache=question.cache, head=False,
                                 placement=question.placement,
                                 spare=NonEmpty(Mib(1000)), layout=question.layout)
                  for question in (light, heavy)]

        self.assertEqual(Mib(4000), place.resident(chosen, answers))

    def test_an_answer_that_is_not_a_requirement_holds_nothing(self):
        question = Question(Tokens(30000), CacheType.Q8_0, WholeCard(), LAYOUT)
        settings = place.Settings(ctx=question.ctx, cache=question.cache, head=False,
                                  placement=question.placement,
                                  spare=NonEmpty(Mib(1000)), layout=question.layout)

        self.assertEqual(Mib(0), place.resident([settings], {question: Refused()}))

    def test_a_model_with_no_placement_holds_nothing(self):
        self.assertEqual(Mib(0), place.resident([], {}))


if __name__ == "__main__":
    unittest.main()
