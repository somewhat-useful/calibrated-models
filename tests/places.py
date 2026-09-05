"""Places for tests to name, taken from the system instead of written down.

Nothing here is opened: this suite is text in, values out, and all it needs of a path
is that it be one -- absolute, spelt the way this system spells one, and belonging to
no machine in particular. The system's own scratch directory is such a place, and it is
the same kind of thing wherever these tests run. A drive letter written into a test
would be neither: it names somebody's machine, and it names it wrongly everywhere else.
"""

import tempfile
from pathlib import Path

_SCRATCH = Path(tempfile.gettempdir())


def somewhere(*names: str) -> Path:
    """A directory named for the part it plays, under the system's own scratch."""
    return Path(_SCRATCH, *names)
