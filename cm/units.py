"""Quantities this program is expressed in.

A card's size, a window's length, a count of layers and the size of a download are all
integers, and a signature taking two bare integers lets a caller swap them with nothing
to notice. Naming each meaning is what makes the swap visible.
"""

from typing import NewType

Mib = NewType("Mib", int)
Tokens = NewType("Tokens", int)
Layers = NewType("Layers", int)

# What a file is measured in where the measuring is somebody else's: a release states
# the size of its archives in bytes, and a finished download is compared with it.
Bytes = NewType("Bytes", int)

# What a router is reached on and how long it waits before it lets a model go. Both are
# whole numbers of very different kinds, and a run started on port 900 would listen
# where nothing calls.
Port = NewType("Port", int)
Seconds = NewType("Seconds", int)
