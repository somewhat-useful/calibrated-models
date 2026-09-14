"""Invariants of a sequence that always has a first element."""

import unittest

from cm.nonempty import NonEmpty


class ThereIsAlwaysAFirstOne(unittest.TestCase):
    def test_an_empty_one_cannot_be_written_down(self):
        with self.assertRaises(TypeError):
            NonEmpty()

    def test_one_element_is_first_and_last(self):
        one = NonEmpty(7)

        self.assertEqual((7, 7, 1), (one.first, one.last, len(one)))


class ItIsTheSequenceItWasGiven(unittest.TestCase):
    def test_the_order_is_kept(self):
        given = NonEmpty(3, 1, 2)

        self.assertEqual([3, 1, 2], list(given))
        self.assertEqual((3, 2), (given.first, given.last))
        self.assertEqual(1, given[1])

    def test_changing_every_element_keeps_order_and_length(self):
        self.assertEqual(NonEmpty(6, 2, 4), NonEmpty(3, 1, 2).map(lambda one: one * 2))


class ItIsAValue(unittest.TestCase):
    def test_equal_contents_are_equal_and_hash_alike(self):
        self.assertEqual(NonEmpty(1, 2), NonEmpty(1, 2))
        self.assertEqual({NonEmpty(1, 2): "a"}[NonEmpty(1, 2)], "a")

    def test_other_contents_or_another_order_are_not(self):
        self.assertNotEqual(NonEmpty(1, 2), NonEmpty(2, 1))
        self.assertNotEqual(NonEmpty(1, 2), NonEmpty(1, 2, 3))

    def test_a_tuple_of_the_same_elements_is_something_else(self):
        self.assertNotEqual(NonEmpty(1, 2), (1, 2))

    def test_it_cannot_be_changed_once_made(self):
        given = NonEmpty(1, 2)

        with self.assertRaises(AttributeError):
            given._items = ()


if __name__ == "__main__":
    unittest.main()
