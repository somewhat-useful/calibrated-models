"""The one way a command stops short of doing what it was asked.

Every command has the same two endings: it did the thing, or it did not and there is a
line saying why. The line is written where the decision was taken -- the module that
knows what was not done and what to do about it -- and carried out to main(), which
prints it and returns 1.

One kind for all of them rather than one per command. Every handler does the same thing
with it, so a second kind is a second name for one ending -- and a command that grows a
call into a step raising somebody else's kind has to remember to name that kind too, or
what should have been a line on stderr comes out as a traceback.
"""


class Refusal(Exception):
    """Something this run will not do, with the one line saying why."""
