"""Invariants of the rule that lets other machines reach the router.

One of them is not a matter of taste: the rule admits the local subnet. Nothing stands
in front of this endpoint, so a rule written for any address hands the card, and every
conversation on it, to whoever else can route to this machine.
"""

import unittest

from cm.lan import PROFILE, SUBNET, closed, opened, rule, shown
from cm.units import Port

PORT = Port(18081)
ELSEWHERE = Port(18082)


def asked(argv):
    """What netsh is told, as name=value pairs, so a test can name one of them."""
    return dict(part.split("=", 1) for part in argv if "=" in part)


class TheRuleIsNamedForThePortItAdmits(unittest.TestCase):
    def test_the_port_is_in_the_name(self):
        self.assertIn(str(PORT), rule(PORT))

    def test_a_router_on_another_port_is_another_rule(self):
        """Replacing the rule for one port must not leave the other port open."""
        self.assertNotEqual(rule(PORT), rule(ELSEWHERE))

    def test_the_rule_added_is_the_rule_asked_after_and_taken_away(self):
        self.assertEqual(asked(opened(PORT))["name"], asked(shown(PORT))["name"])
        self.assertEqual(asked(opened(PORT))["name"], asked(closed(PORT))["name"])


class NothingWiderThanTheLocalSubnet(unittest.TestCase):
    def test_the_rule_admits_the_local_subnet_and_not_any_address(self):
        self.assertEqual(SUBNET, asked(opened(PORT))["remoteip"])
        self.assertNotEqual("any", asked(opened(PORT))["remoteip"])

    def test_it_applies_to_the_private_profile_only(self):
        self.assertEqual(PROFILE, asked(opened(PORT))["profile"])
        self.assertNotIn("public", asked(opened(PORT))["profile"])

    def test_it_admits_that_one_port_over_tcp(self):
        said = asked(opened(PORT))

        self.assertEqual(str(PORT), said["localport"])
        self.assertEqual("TCP", said["protocol"])

    def test_inbound_and_allow(self):
        self.assertIn("dir=in", opened(PORT))
        self.assertIn("action=allow", opened(PORT))


if __name__ == "__main__":
    unittest.main()
