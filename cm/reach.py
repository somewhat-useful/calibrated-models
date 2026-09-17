"""Whether a machine on the network answers on a port. One question, asked of the network,
with nothing decided here."""

import socket

from .place import Endpoint

# Long enough for a machine on the local network to answer, short enough that one that is
# switched off is reported while a person is still watching.
TIMEOUT = 3


def reachable(endpoint: Endpoint) -> bool:
    """Whether something accepts a connection there. What it is, is not asked."""
    try:
        with socket.create_connection((endpoint.host, endpoint.port), timeout=TIMEOUT):
            return True
    except OSError:
        return False
