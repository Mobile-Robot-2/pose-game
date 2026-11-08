import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fall_detector import (
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    FallDetector,
)


def pose(nose, shoulders, hips, visibility=0.99):
    landmarks = [None] * 33
    landmarks[NOSE] = SimpleNamespace(x=nose[0], y=nose[1], visibility=visibility)
    for point, left, right in (
        (shoulders, LEFT_SHOULDER, RIGHT_SHOULDER),
        (hips, LEFT_HIP, RIGHT_HIP),
    ):
        x, y = point
        landmarks[left] = SimpleNamespace(x=x - 0.06, y=y, visibility=visibility)
        landmarks[right] = SimpleNamespace(x=x + 0.06, y=y, visibility=visibility)
    return landmarks


STANDING = pose((0.50, 0.22), (0.50, 0.35), (0.50, 0.57))
LYING = pose((0.20, 0.70), (0.40, 0.72), (0.67, 0.73))
SEATED = pose((0.50, 0.31), (0.50, 0.45), (0.50, 0.68))
BENDING = pose((0.22, 0.51), (0.42, 0.40), (0.50, 0.60))
OCCLUDED = pose((0.50, 0.22), (0.50, 0.35), (0.50, 0.57), visibility=0.1)
LOW_VISIBILITY = pose((0.50, 0.22), (0.50, 0.35), (0.50, 0.57), visibility=0.52)


class FallDetectorTest(unittest.TestCase):
    def setUp(self):
        self.detector = FallDetector()
        clock = patch("fall_detector.time.monotonic")
        self.clock = clock.start()
        self.addCleanup(clock.stop)

    def observe(self, timestamp, landmarks, expected_status):
        self.clock.return_value = timestamp
        if landmarks is None:
            state = self.detector.update_no_pose()
        else:
            state = self.detector.update(landmarks)
        self.assertEqual(state["fall_status"], expected_status, f"at {timestamp} s")
        self.assertEqual(state["fall_detected"], expected_status == "FALL_CONFIRMED")
        return state

    def begin_fall(self):
        self.observe(0.0, STANDING, "NORMAL")
        state = self.observe(0.5, LYING, "FALL_SUSPECTED")
        self.assertTrue(state["fall_metrics"]["sudden_drop"])
        self.assertIsNone(state["fall_event"]["event_id"])

    def confirm_fall(self):
        self.begin_fall()
        self.observe(1.0, LYING, "FALL_SUSPECTED")
        state = self.observe(1.5, LYING, "FALL_CONFIRMED")
        self.assertTrue(state["fall_event"]["new_event"])
        self.assertIsNotNone(state["fall_event"]["event_id"])
        return state["fall_event"]["event_id"]

    def test_already_lying(self):
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            state = self.observe(timestamp, LYING, "NORMAL")
            self.assertIsNone(state["fall_event"]["event_id"])

    def test_standing(self):
        for timestamp in (0.0, 0.5, 1.0, 1.5):
            self.observe(timestamp, STANDING, "NORMAL")

    def test_fast_fall_and_no_duplicate_event(self):
        event_id = self.confirm_fall()
        for timestamp in (2.0, 2.5, 3.0):
            state = self.observe(timestamp, LYING, "FALL_CONFIRMED")
            self.assertFalse(state["fall_event"]["new_event"])
            self.assertEqual(state["fall_event"]["event_id"], event_id)

    def test_slow_lying_down(self):
        self.observe(0.0, STANDING, "NORMAL")
        stages = (
            pose((0.46, 0.27), (0.50, 0.39), (0.53, 0.60)),
            pose((0.40, 0.34), (0.50, 0.46), (0.57, 0.62)),
            pose((0.34, 0.44), (0.49, 0.53), (0.60, 0.65)),
            pose((0.27, 0.54), (0.44, 0.60), (0.64, 0.67)),
            pose((0.23, 0.62), (0.42, 0.66), (0.66, 0.70)),
            LYING,
            LYING,
            LYING,
        )
        for frame, landmarks in enumerate(stages, start=1):
            self.observe(frame * 0.5, landmarks, "NORMAL")

    def test_sitting_and_bending(self):
        self.observe(0.0, STANDING, "NORMAL")
        self.observe(0.5, SEATED, "NORMAL")
        self.observe(1.0, SEATED, "NORMAL")
        self.observe(1.5, STANDING, "NORMAL")
        self.observe(2.0, BENDING, "FALL_SUSPECTED")
        self.observe(2.5, STANDING, "NORMAL")

    def test_long_gap_cancels_suspicion(self):
        self.begin_fall()
        self.observe(1.5, LYING, "NORMAL")
        self.observe(2.0, LYING, "NORMAL")

    def test_missing_or_unreliable_pose_cancels_suspicion(self):
        for landmarks in (None, OCCLUDED, LOW_VISIBILITY):
            with self.subTest(landmarks=landmarks):
                self.detector.reset()
                self.begin_fall()
                self.observe(0.75, landmarks, "NORMAL")
                self.observe(1.0, LYING, "NORMAL")
                self.observe(1.5, LYING, "NORMAL")

    def test_old_upright_reference_expires_without_a_frame_gap(self):
        self.observe(0.0, STANDING, "NORMAL")
        self.observe(0.5, SEATED, "NORMAL")
        self.observe(1.0, SEATED, "NORMAL")
        self.observe(1.5, LYING, "NORMAL")
        self.observe(2.0, LYING, "NORMAL")

    def test_confirmation_timer_starts_when_low_pose_is_observed(self):
        self.observe(0.0, STANDING, "NORMAL")
        self.observe(0.5, BENDING, "FALL_SUSPECTED")
        self.observe(1.0, LYING, "FALL_SUSPECTED")
        self.observe(1.5, LYING, "FALL_SUSPECTED")
        self.observe(2.0, LYING, "FALL_CONFIRMED")

    def test_non_low_pose_interrupts_confirmation_timer(self):
        self.begin_fall()
        self.observe(0.75, BENDING, "FALL_SUSPECTED")
        self.observe(1.0, LYING, "FALL_SUSPECTED")
        self.observe(1.5, LYING, "FALL_SUSPECTED")
        self.observe(2.0, LYING, "FALL_CONFIRMED")

    def test_lower_angle_can_sustain_but_not_start_suspicion(self):
        # A low nose and shoulders, upper-body angle about 45 degrees,
        # and a vertical torso reproduce the intermediate Temi frames.
        low_bent = pose((0.38, 0.58), (0.50, 0.46), (0.50, 0.69))
        self.observe(0.0, STANDING, "NORMAL")
        self.observe(0.5, low_bent, "NORMAL")
        self.observe(1.0, low_bent, "NORMAL")

        self.detector.reset()
        self.begin_fall()
        self.observe(1.0, low_bent, "FALL_SUSPECTED")
        self.observe(1.5, low_bent, "FALL_CONFIRMED")

    def test_frontal_drop_confirms_with_low_in_frame_pose(self):
        low_pose = pose((0.50, 0.70), (0.50, 0.62), (0.50, 0.75))
        self.observe(0.0, STANDING, "NORMAL")
        state = self.observe(0.4, low_pose, "FALL_SUSPECTED")
        self.assertFalse(state["fall_metrics"]["upper_body_horizontal"])
        self.assertFalse(state["fall_metrics"]["horizontal"])
        self.observe(0.8, low_pose, "FALL_SUSPECTED")
        self.observe(1.2, low_pose, "FALL_SUSPECTED")
        self.observe(1.55, low_pose, "FALL_SUSPECTED")
        self.observe(1.65, low_pose, "FALL_CONFIRMED")

    def test_frontal_drop_does_not_use_expired_upright_sample(self):
        middle_pose = pose((0.50, 0.37), (0.50, 0.47), (0.50, 0.65))
        low_pose = pose((0.50, 0.70), (0.50, 0.62), (0.50, 0.75))
        self.observe(0.0, STANDING, "NORMAL")
        self.observe(0.4, middle_pose, "NORMAL")
        self.observe(0.8, low_pose, "NORMAL")

    def test_out_of_frame_hips_cannot_sustain_angle_free_confirmation(self):
        out_of_frame = pose((0.50, 0.70), (0.50, 0.62), (0.50, 1.10))
        self.begin_fall()
        self.observe(1.0, out_of_frame, "FALL_SUSPECTED")
        self.observe(1.5, out_of_frame, "NORMAL")

    def test_missing_or_unreliable_pose_interrupts_recovery(self):
        for landmarks in (None, OCCLUDED, LOW_VISIBILITY):
            with self.subTest(landmarks=landmarks):
                self.detector.reset()
                event_id = self.confirm_fall()
                self.observe(2.0, STANDING, "FALL_CONFIRMED")
                for timestamp in (2.5, 3.0, 3.5):
                    state = self.observe(timestamp, landmarks, "FALL_CONFIRMED")
                    self.assertFalse(state["fall_event"]["new_event"])
                    self.assertEqual(state["fall_event"]["event_id"], event_id)
                self.observe(4.0, STANDING, "FALL_CONFIRMED")

    def test_upright_recovery_then_normal(self):
        self.confirm_fall()
        for timestamp in (2.0, 2.5, 3.0):
            self.observe(timestamp, STANDING, "FALL_CONFIRMED")
        for timestamp in (3.5, 4.0, 4.5, 5.0):
            self.observe(timestamp, STANDING, "RECOVERED")
        self.observe(5.5, STANDING, "NORMAL")

    def test_reset_clears_pending_motion_and_event(self):
        self.confirm_fall()
        state = self.detector.reset()
        self.assertEqual(state["fall_status"], "NORMAL")
        self.assertIsNone(state["fall_event"]["event_id"])
        self.observe(2.0, LYING, "NORMAL")


if __name__ == "__main__":
    unittest.main()
