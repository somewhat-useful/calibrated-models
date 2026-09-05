"""`python -m cm`: which command comes next.

Nothing is read and nothing is decided. guide.py holds the order and this prints it.
"""

import sys

from . import guide


def main() -> int:
    for line in guide.path():
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
