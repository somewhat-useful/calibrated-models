"""Invariants of reading a model's shape out of llama-fit-params' verbose output.

Every input here is written by hand and is only as large as the invariant needs. What a
real run printed is deliberately not a test input, however easy it would be to keep one:
a test reading it pins one model's numbers where the subject is a property of the parser,
and it fails on the day the estimator changes a word it never had to say.
"""

import unittest

from cm.facts import Head, MissingFact, NoHead, VariesByLayer, parse_facts

# The estimator prefixes every line with elapsed time and a level. It is kept in these
# inputs because a parser anchored to the start of a line passes without it and fails on
# the real output.
PREFIX = "0.01.241.081 I "

# The three a placement always needs. A file's cache geometry is only needed when it
# carries a head, so it is not here.
REQUIRED = (
    ("n_ctx_train", 262144),
    ("n_layer", 64),
    ("n_expert", 0),
)

CACHE = (
    ("n_embd_k_gqa", 1024),
    ("n_embd_v_gqa", 512),
)

UNUSED = "0.01.246.143 W model has unused tensor {name} (size = {size} bytes) -- ignoring"


def verbose(pairs, extra=()):
    """The estimator's stderr for a shape, in its real line format."""
    lines = [f"{PREFIX}print_info: {name:22}= {value}" for name, value in pairs]
    lines.extend(extra)
    return "\n".join(lines) + "\n"


def head_tensors(*sizes):
    return [UNUSED.format(name=f"blk.64.attn_q.weight", size=size) for size in sizes]


def without(pairs, field):
    return tuple((name, value) for name, value in pairs if name != field)


class FieldsAreNotConfusedWithTheirNeighbours(unittest.TestCase):
    """n_layer_all and n_expert_used sit beside the fields wanted, and differ from them.

    Both orders are tried: a parser that takes the first line matching a substring is
    right in one order and wrong in the other, and the real output does not promise an
    order.
    """

    def assert_reads(self, field, wanted, decoy, decoy_value):
        for pairs in (
            without(REQUIRED, field) + ((field, wanted), (decoy, decoy_value)),
            without(REQUIRED, field) + ((decoy, decoy_value), (field, wanted)),
        ):
            with self.subTest(first=pairs[-2][0]):
                self.assertEqual(getattr(parse_facts(verbose(pairs)), field), wanted)

    def test_n_layer_is_not_n_layer_all(self):
        self.assert_reads("n_layer", 64, "n_layer_all", 65)

    def test_n_expert_is_not_n_expert_used(self):
        self.assert_reads("n_expert", 128, "n_expert_used", 8)


class AHeadIsEitherThereWithBothItsNumbersOrNotAtAll(unittest.TestCase):
    """No file has a head weight without a cache cost, or a cache cost without a head.

    Keeping them in one value is what stops a head being priced from numbers that were
    never read.
    """

    def test_no_unused_tensors_means_no_head(self):
        facts = parse_facts(verbose(REQUIRED))

        self.assertEqual(facts.head, NoHead())

    def test_the_head_weighs_every_tensor_the_estimator_skipped(self):
        text = verbose(REQUIRED + CACHE,
                       extra=head_tensors(20480, 51609600, 43008000))

        facts = parse_facts(text)

        self.assertEqual(facts.head, Head(weight_bytes=20480 + 51609600 + 43008000,
                                          cache_per_token=1024 + 512))

    def test_the_head_cache_is_both_halves(self):
        """Asymmetric halves: doubling either one gives a different, wrong answer."""
        facts = parse_facts(verbose(REQUIRED + CACHE, extra=head_tensors(1000)))

        self.assertEqual(facts.head.cache_per_token, 1536)
        self.assertNotIn(facts.head.cache_per_token, (2048, 1024))

    def test_a_head_without_its_cache_geometry_is_reported(self):
        for field, _ in CACHE:
            with self.subTest(field=field):
                text = verbose(REQUIRED + without(CACHE, field),
                               extra=head_tensors(1000))

                with self.assertRaises(MissingFact) as caught:
                    parse_facts(text)

                self.assertEqual(caught.exception.field, field)


class ALayerListIsNotANumber(unittest.TestCase):
    """A model whose layers differ gets a list here, not a value.

    Some of its layers attend over the whole history and some over a sliding window, so
    there is no single number, and taking the first entry would silently price the head
    against whichever layer happened to be printed first.
    """

    PER_LAYER = f"{PREFIX}print_info: n_embd_k_gqa          = [2048, 2048, 512, 2048]"

    def test_a_head_cannot_be_weighed_against_a_list(self):
        text = verbose(REQUIRED + (("n_embd_v_gqa", 512),),
                       extra=[self.PER_LAYER, *head_tensors(1000)])

        with self.assertRaises(VariesByLayer) as caught:
            parse_facts(text)

        self.assertEqual(caught.exception.field, "n_embd_k_gqa")

    def test_a_file_without_a_head_does_not_care(self):
        """Nothing needs the number, so a list where one would be is not a failure."""
        text = verbose(REQUIRED, extra=[self.PER_LAYER])

        facts = parse_facts(text)

        self.assertEqual(facts.head, NoHead())


class AMissingFieldNamesItself(unittest.TestCase):
    """No zero, no None, no default: a shape that cannot be read stops the run.

    A default here is the worst kind of wrong -- every placement downstream is derived
    from these numbers, and a plausible one produces a plausible answer.
    """

    def test_every_required_field_is_reported_by_name(self):
        for field, _ in REQUIRED:
            with self.subTest(field=field):
                with self.assertRaises(MissingFact) as caught:
                    parse_facts(verbose(without(REQUIRED, field)))

                self.assertEqual(caught.exception.field, field)


class ThePrefixDoesNotHideAField(unittest.TestCase):
    def test_fields_are_found_mid_line(self):
        """Real lines carry a timestamp and a level before print_info."""
        text = verbose(REQUIRED)
        self.assertFalse(text.startswith("print_info"))

        facts = parse_facts(text)

        self.assertEqual(facts.n_ctx_train, 262144)


if __name__ == "__main__":
    unittest.main()
