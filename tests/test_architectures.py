"""Invariants of reading which cards the published builds carry finished code for.

The source is written here in the shape the project writes it -- a defaults block,
appended to a CUDA version at a time -- rather than copied. What is checked is that
every way that shape can be read is read, and that every way it cannot ends in saying
so rather than in a list that only looks right.
"""

import unittest

from cm.architectures import (NOTHING_KNOWN, Architectures, Before, Compiled, From,
                              Unread, finished, read)
from cm.cuda import Cuda
from cm.machine import Capability

SOURCE = """
if (CUDAToolkit_FOUND)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == whatever is in this machine
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            if (CUDAToolkit_VERSION VERSION_LESS "13")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual)
            endif ()

            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real 90-virtual)
            endif()

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
            endif()
        endif()
    endif()

    enable_language(CUDA)
endif()
"""

WORKFLOW = """
jobs:
  windows-cpu:
    steps:
      - run: cmake -DGGML_CPU=ON

  windows-cuda:
    strategy:
      matrix:
        include:
          - cuda: '13.4'
    steps:
      - run: cmake -S . -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON

  release:
    steps:
      - run: echo CMAKE_CUDA_ARCHITECTURES is not this job's
"""

READ = Architectures(compiled=(
    Compiled(Capability(8, 6), ()),
    Compiled(Capability(8, 9), (From(Cuda(11, 8)),)),
    Compiled(Capability(12, 0), (From(Cuda(12, 8)),)),
))


def defaults(*body: str) -> str:
    """A defaults block holding these lines, and nothing else."""
    return "\n".join(("if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)", *body, "endif()"))


class WhatTheSourceCompilesToFinishedCodeIsRead(unittest.TestCase):
    def test_every_finished_architecture_with_the_versions_it_is_compiled_under(self):
        self.assertEqual(READ, read(SOURCE, WORKFLOW))

    def test_what_a_build_carries_follows_its_cuda_version(self):
        known = read(SOURCE, WORKFLOW)

        self.assertEqual(frozenset({Capability(8, 6), Capability(8, 9)}),
                         finished(Cuda(12, 4), known))
        self.assertEqual(frozenset({Capability(8, 6), Capability(8, 9), Capability(12, 0)}),
                         finished(Cuda(13, 4), known))

    def test_ptx_is_not_finished_code(self):
        known = read(SOURCE, WORKFLOW)

        for card in (Capability(7, 5), Capability(8, 0), Capability(9, 0)):
            with self.subTest(card=card):
                self.assertNotIn(card, finished(Cuda(13, 4), known))

    def test_an_architecture_named_without_a_suffix_is_finished_as_well(self):
        """No suffix compiles both PTX and finished code."""
        known = read(defaults("list(APPEND CMAKE_CUDA_ARCHITECTURES 80)"), WORKFLOW)

        self.assertEqual(frozenset({Capability(8, 0)}), finished(Cuda(13, 4), known))

    def test_one_compiled_only_below_a_version_is_not_above_it(self):
        known = read(defaults('if (CUDAToolkit_VERSION VERSION_LESS "13")',
                              "list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)",
                              "endif()"), WORKFLOW)

        self.assertEqual(Architectures((Compiled(Capability(7, 0),
                                                 (Before(Cuda(13, 0)),)),)), known)
        self.assertIn(Capability(7, 0), finished(Cuda(12, 9), known))
        self.assertNotIn(Capability(7, 0), finished(Cuda(13, 0), known))

    def test_the_else_of_a_version_is_the_versions_the_other_side(self):
        known = read(defaults('if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")',
                              "list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)",
                              "else()",
                              "list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real)",
                              "endif()"), WORKFLOW)

        self.assertEqual(frozenset({Capability(12, 0)}), finished(Cuda(13, 4), known))
        self.assertEqual(frozenset({Capability(8, 9)}), finished(Cuda(12, 4), known))


class WhatCannotBeReadIsSaidRatherThanGuessed(unittest.TestCase):
    def assertUnread(self, said, *words):
        self.assertIsInstance(said, Unread)
        for word in words:
            self.assertIn(word, said.why)

    def test_a_source_without_the_defaults_block(self):
        self.assertUnread(read("project(ggml)\n", WORKFLOW), "no")

    def test_windows_builds_that_name_their_own_architectures(self):
        """The defaults would then say nothing about them."""
        own = WORKFLOW.replace("-DGGML_CUDA=ON",
                               '-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;86"')

        self.assertUnread(read(SOURCE, own), "windows-cuda", "architectures of its own")

    def test_a_workflow_with_no_windows_cuda_job(self):
        self.assertUnread(read(SOURCE, "jobs:\n  linux:\n    steps: []\n"), "windows-cuda")

    def test_an_architecture_added_under_a_condition_this_does_not_read(self):
        said = read(defaults("if (GGML_CUDA_FA_ALL_QUANTS)",
                             "list(APPEND CMAKE_CUDA_ARCHITECTURES 90-real)",
                             "endif()"), WORKFLOW)

        self.assertUnread(said, "GGML_CUDA_FA_ALL_QUANTS")

    def test_elseif(self):
        said = read(defaults('if (CUDAToolkit_VERSION VERSION_LESS "12")',
                             "list(APPEND CMAKE_CUDA_ARCHITECTURES 70-real)",
                             'elseif (CUDAToolkit_VERSION VERSION_LESS "13")',
                             "list(APPEND CMAKE_CUDA_ARCHITECTURES 75-real)",
                             "endif()"), WORKFLOW)

        self.assertUnread(said, "elseif")

    def test_an_architecture_named_some_other_way(self):
        said = read(defaults("list(APPEND CMAKE_CUDA_ARCHITECTURES sm_90)"), WORKFLOW)

        self.assertUnread(said, "sm_90")

    def test_the_list_set_rather_than_appended_to(self):
        said = read(defaults('set(CMAKE_CUDA_ARCHITECTURES "86-real;89-real")'), WORKFLOW)

        self.assertUnread(said, "sets the architectures")

    def test_a_block_that_never_closes(self):
        said = read("if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)\n"
                    "list(APPEND CMAKE_CUDA_ARCHITECTURES 86-real)\n", WORKFLOW)

        self.assertUnread(said, "does not close")

    def test_a_block_compiling_nothing_finished(self):
        said = read(defaults("list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual)"), WORKFLOW)

        self.assertUnread(said, "no architecture")

    def test_nothing_known_is_no_card_ready_under_any_version(self):
        self.assertEqual(frozenset(), finished(Cuda(13, 4), NOTHING_KNOWN))


if __name__ == "__main__":
    unittest.main()
