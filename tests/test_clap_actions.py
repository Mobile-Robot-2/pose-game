import unittest

from action_counter import ClapDetector
from test_rhythm_actions import pose


class ClapActionsTest(unittest.TestCase):
    def test_initial_contact_requires_separation_before_clap(self):
        detector = ClapDetector()
        shoulders = ((.6, .4), (.4, .4))
        separated = pose(*shoulders, (.7, .6), (.3, .6))
        contact = pose(*shoulders, (.51, .6), (.49, .6))
        self.assertFalse(detector.update(contact, 0))
        self.assertFalse(detector.update(contact, .2))
        self.assertFalse(detector.update(separated, .4))
        self.assertTrue(detector.update(contact, .6))

    def test_contact_after_long_gap_requires_new_separation(self):
        detector = ClapDetector()
        shoulders = ((.6, .4), (.4, .4))
        separated = pose(*shoulders, (.7, .6), (.3, .6))
        contact = pose(*shoulders, (.51, .6), (.49, .6))
        detector.update(separated, 0)
        self.assertTrue(detector.update(contact, .2))
        self.assertFalse(detector.update(contact, 1.5))
        self.assertFalse(detector.update(separated, 1.7))
        self.assertTrue(detector.update(contact, 1.9))

    def test_out_of_frame_hand_may_arm_but_cannot_trigger(self):
        detector = ClapDetector()
        shoulders = ((.6, .4), (.4, .4))
        at_edge = pose(*shoulders, (.7, .6), (-.1, .6))
        at_edge[16].visibility = .1
        contact = pose(*shoulders, (.51, .6), (.49, .6))
        self.assertFalse(detector.update(at_edge, 0))
        self.assertTrue(detector.update(contact, .2))
        self.assertFalse(detector.update(contact, .4))

    def test_partially_visible_separation_cannot_trigger_by_itself(self):
        detector = ClapDetector()
        shoulders = ((.6, .4), (.4, .4))
        partial = pose(*shoulders, (.7, .6), (.3, .6))
        partial[16].visibility = .1
        contact = pose(*shoulders, (.51, .6), (.49, .6))
        self.assertFalse(detector.update(partial, 0))
        self.assertTrue(detector.update(contact, .2))

    def test_clap_with_slight_wrist_gap_is_one_event(self):
        detector = ClapDetector()
        shoulders = ((.6, .4), (.4, .4))
        separated = pose(*shoulders, (.7, .6), (.3, .6))
        near = pose(*shoulders, (.57, .6), (.43, .6))
        self.assertFalse(detector.update(separated, 0))
        self.assertTrue(detector.update(near, .2))
        self.assertFalse(detector.update(near, .4))

    def test_clap_emits_once_on_contact_and_rearms_after_separation(self):
        detector = ClapDetector()
        shoulders = ((.6, .4), (.4, .4))
        separated = pose(*shoulders, (.7, .6), (.3, .6))
        contact = pose(*shoulders, (.51, .6), (.49, .6))
        self.assertFalse(detector.update(separated, 0))
        self.assertTrue(detector.update(contact, .2))
        self.assertFalse(detector.update(contact, .4))
        self.assertFalse(detector.update(contact, .6))
        self.assertFalse(detector.update(separated, .8))
        self.assertFalse(detector.update(separated, 1.0))
        self.assertTrue(detector.update(contact, 1.2))
        self.assertFalse(detector.update(separated, 1.4))

    def test_brief_pose_loss_does_not_repeat_contact(self):
        shoulders = ((.6, .4), (.4, .4))
        separated = pose(*shoulders, (.7, .6), (.3, .6))
        contact = pose(*shoulders, (.51, .6), (.49, .6))
        detector = ClapDetector()
        detector.update(separated, 0)
        self.assertTrue(detector.update(contact, .2))
        self.assertFalse(detector.update(None, .4))
        self.assertFalse(detector.update(contact, .6))
        self.assertFalse(detector.update(separated, .8))
        self.assertTrue(detector.update(contact, 1.0))

    def test_large_frame_gap_starts_new_observation(self):
        detector = ClapDetector()
        shoulders = ((.6, .4), (.4, .4))
        separated = pose(*shoulders, (.7, .6), (.3, .6))
        contact = pose(*shoulders, (.51, .6), (.49, .6))
        detector.update(separated, 0)
        self.assertTrue(detector.update(contact, .2))
        self.assertFalse(detector.update(separated, 1.5))
        self.assertTrue(detector.update(contact, 1.7))


if __name__ == "__main__":
    unittest.main()
