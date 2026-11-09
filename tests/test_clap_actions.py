import unittest

from action_counter import ClapDetector
from test_rhythm_actions import pose


class ClapActionsTest(unittest.TestCase):
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
