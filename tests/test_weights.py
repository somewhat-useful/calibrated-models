"""Invariants of what a model file is read for.

One question is asked of a model's own bytes: how much of it llama.cpp never holds in
memory, because an architecture marked a tensor as read row by row. The estimator
counts such a tensor as memory held, so the answer here is what its answer has to be
corrected by, and a wrong one is a model refused for memory nothing takes -- or worse,
a model placed against memory that is taken after all.

Headers are built in the test, byte for byte as the format writes them. Nothing is
opened: a file on this machine would make the rules depend on what happens to be in a
library.
"""

import struct
import unittest
from pathlib import PureWindowsPath

from cm.units import Bytes, Mib
from cm.weights import (LAZY_FLOOR, Header, Short, Tensor, Unreadable, header,
                        lazily_read, volumes)
from places import somewhere

LAZY = "per_layer_token_embd.weight"
GIB = 1024 ** 3
ALIGNMENT = 32

_U32, _STRING = 4, 8


def _text(said: str) -> bytes:
    raw = said.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _said(keys: dict[str, int | str]) -> bytes:
    blob = b""
    for key, value in keys.items():
        blob += _text(key)
        if isinstance(value, str):
            blob += struct.pack("<I", _STRING) + _text(value)
        else:
            blob += struct.pack("<I", _U32) + struct.pack("<I", value)
    return blob


def written(tensors: tuple[tuple[str, int], ...],
            keys: dict[str, int | str] | None = None) -> tuple[bytes, Bytes]:
    """A file's header as GGUF writes one, and the size of the file carrying it.

    Every tensor is stored where the one before it ended, rounded up to the alignment,
    which is what makes the distance between two offsets the size of the first.
    """
    keys = keys or {}
    blob = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(keys)) + _said(keys)

    at = 0
    for name, size in tensors:
        blob += _text(name) + struct.pack("<I", 2) + struct.pack("<QQ", 1, 1)
        blob += struct.pack("<I", 0) + struct.pack("<Q", at)
        at += size + (-size % ALIGNMENT)

    start = len(blob) + (-len(blob) % ALIGNMENT)

    return blob + bytes(start - len(blob)), Bytes(start + at)


def read(tensors, keys=None) -> Header:
    blob, whole = written(tensors, keys)
    return header(blob, whole)


class ATableReadFromDiskIsNotMemoryTheModelHolds(unittest.TestCase):
    """llama.cpp maps such a tensor and fetches rows, whatever the load mode says, so
    its bytes are never resident and must come off what a placement is said to need."""

    def test_the_marked_tensor_is_what_is_read_from_disk(self):
        head = read(((LAZY, 27 * GIB), ("blk.0.ffn_down_exps.weight", GIB)))

        self.assertEqual(Mib(27 * 1024), lazily_read([head]))

    def test_a_file_that_carries_no_such_tensor_reads_nothing_from_disk(self):
        head = read((("token_embd.weight", 8 * GIB), ("blk.0.attn_qkv.weight", GIB)))

        self.assertEqual(Mib(0), lazily_read([head]))

    def test_a_marked_tensor_under_the_floor_is_held_like_any_other(self):
        """Reading a small one row by row costs more than it saves, so llama.cpp keeps
        it, and counting it as read from disk would place against memory that is used."""
        for size in (1, LAZY_FLOOR // 2, LAZY_FLOOR):
            with self.subTest(size=size):
                self.assertEqual(Mib(0), lazily_read([read(((LAZY, size),))]))

    def test_one_byte_over_the_floor_is_read_from_disk(self):
        head = read(((LAZY, LAZY_FLOOR + ALIGNMENT),))

        self.assertEqual(Mib(4096), lazily_read([head]))


class WhatATensorTakesIsWhereTheNextOneStarts(unittest.TestCase):
    """Read from the offsets rather than from the shape and the quantisation: what each
    quantisation weighs is llama.cpp's arithmetic, and it moves with llama.cpp."""

    def test_every_tensor_is_measured_by_the_one_stored_after_it(self):
        head = read((("first", 2 * GIB), ("second", GIB), (LAZY, 5 * GIB)))

        self.assertEqual([Tensor("first", Bytes(2 * GIB)),
                          Tensor("second", Bytes(GIB)),
                          Tensor(LAZY, Bytes(5 * GIB))], list(head.tensors))

    def test_the_last_tensor_is_measured_by_the_end_of_the_file(self):
        blob, whole = written(((LAZY, 5 * GIB),))

        self.assertEqual(Mib(5 * 1024), lazily_read([header(blob, whole)]))

    def test_tensors_stored_out_of_order_are_measured_in_the_order_stored(self):
        """Nothing says a header lists them in the order the data is written."""
        blob, whole = written((("first", 2 * GIB), (LAZY, 5 * GIB)))
        forward = header(blob, whole)

        blob, whole = written(((LAZY, 5 * GIB), ("first", 2 * GIB)))

        self.assertEqual(lazily_read([forward]), lazily_read([header(blob, whole)]))


class AModelStoredInSeveralFilesIsReadInAllOfThem(unittest.TestCase):
    """A settings file names the first volume and llama.cpp opens the rest through it,
    so the table is regularly in a file nothing has looked at."""

    FIRST = somewhere("models", "publisher", "repo", "model-00001-of-00003.gguf")

    def test_the_other_volumes_are_named_as_llama_cpp_splits_them(self):
        found = volumes(self.FIRST, {"split.count": 3})

        self.assertEqual([self.FIRST,
                          self.FIRST.with_name("model-00002-of-00003.gguf"),
                          self.FIRST.with_name("model-00003-of-00003.gguf")], list(found))

    def test_a_model_in_one_file_is_that_file(self):
        one = somewhere("models", "publisher", "repo", "model.gguf")

        self.assertEqual([one], list(volumes(one, {})))
        self.assertEqual([one], list(volumes(one, {"split.count": 1})))

    def test_a_table_in_the_second_volume_is_found(self):
        first = read((("token_embd.weight", GIB),), {"split.count": 2})
        second = read(((LAZY, 27 * GIB),))

        self.assertEqual(Mib(27 * 1024), lazily_read([first, second]))


class AFileThatIsNotOneIsRefusedRatherThanGuessedAt(unittest.TestCase):
    def test_bytes_that_do_not_open_with_the_magic_are_refused(self):
        for blob in (b"", b"GGUF"[:3], b"NOPE" + bytes(64)):
            with self.subTest(blob=blob):
                with self.assertRaises((Unreadable, Short)):
                    header(blob, Bytes(1024))

    def test_a_header_that_does_not_end_in_the_bytes_in_hand_asks_for_more(self):
        blob, whole = written(((LAZY, 5 * GIB), ("second", GIB)))

        with self.assertRaises(Short):
            header(blob[:len(blob) // 2], whole)

    def test_a_windows_path_is_split_the_same_way(self):
        """The library this reads is somebody's disk, and this one runs on Windows."""
        first = PureWindowsPath(r"X:\models\repo\model-00001-of-00002.gguf")

        self.assertEqual(r"model-00002-of-00002.gguf",
                         volumes(first, {"split.count": 2})[1].name)


if __name__ == "__main__":
    unittest.main()
