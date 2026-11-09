import unittest
from types import SimpleNamespace

from action_counter import ActionCounter, classify_actions, is_arms_open, is_left_hand_up, is_right_hand_up


def pose(left_shoulder, right_shoulder, left_wrist, right_wrist, visibility=1.0):
    landmarks = [SimpleNamespace(x=0, y=0, visibility=visibility) for _ in range(33)]
    for index, (x, y) in zip((11, 12, 15, 16), (left_shoulder, right_shoulder, left_wrist, right_wrist)):
        landmarks[index] = SimpleNamespace(x=x, y=y, visibility=visibility)
    for index, shoulder, wrist in ((13, left_shoulder, left_wrist), (14, right_shoulder, right_wrist)):
        landmarks[index] = SimpleNamespace(x=(shoulder[0] + wrist[0]) / 2,
                                           y=(shoulder[1] + wrist[1]) / 2, visibility=visibility)
    return landmarks


class RhythmActionsTest(unittest.TestCase):
    def test_bent_elbow_and_wrist_slightly_above_shoulder(self):
        landmarks = pose((.6, .4), (.4, .4), (.65, .37), (.35, .37))
        landmarks[13].y = .6
        landmarks[14].y = .6
        self.assertTrue(is_left_hand_up(landmarks))
        self.assertTrue(is_right_hand_up(landmarks))

    def test_finger_can_confirm_hand_near_shoulder(self):
        landmarks = pose((.6, .4), (.4, .4), (.65, .42), (.35, .42))
        landmarks[19] = SimpleNamespace(x=.65, y=.29, visibility=.9)
        landmarks[20] = SimpleNamespace(x=.35, y=.29, visibility=.9)
        self.assertTrue(is_left_hand_up(landmarks))
        self.assertTrue(is_right_hand_up(landmarks))
        landmarks[19].visibility = .1
        landmarks[20].x = .9
        self.assertFalse(is_left_hand_up(landmarks))
        self.assertFalse(is_right_hand_up(landmarks))

    def test_hands_below_shoulders_are_not_raised(self):
        landmarks = pose((.6, .4), (.4, .4), (.65, .55), (.35, .55))
        landmarks[19] = SimpleNamespace(x=.65, y=.30, visibility=.9)
        landmarks[20] = SimpleNamespace(x=.35, y=.30, visibility=.9)
        self.assertFalse(is_left_hand_up(landmarks))
        self.assertFalse(is_right_hand_up(landmarks))

    def test_prediction_priorities_and_ambiguous_or_missing_pose(self):
        cases = [
            (["ARMS_OPEN", "LEFT_HAND_UP", "RIGHT_HAND_UP"], "ARMS_OPEN", "detected"),
            (["CLAP", "LEFT_HAND_UP", "RIGHT_HAND_UP"], "CLAP", "detected"),
            (["CLAP", "ARMS_OPEN", "LEFT_HAND_UP"], None, "conflict"),
            (["LEFT_HAND_UP", "RIGHT_HAND_UP"], None, "conflict"),
            (["RIGHT_HAND_UP"], "RIGHT_HAND_UP", "detected"),
            ([], None, "no_action"),
        ]
        for actions, expected, status in cases:
            with self.subTest(actions=actions):
                self.assertEqual(classify_actions(actions), (expected, status))
        self.assertEqual(classify_actions([], pose_detected=False), (None, "pose_not_detected"))

    def test_first_clap_event_is_counted_once(self):
        counter = ActionCounter()
        first = counter.update({"CLAP": True}, elapsed_time=16)
        self.assertEqual(first["counts"]["CLAP"], 1)
        self.assertEqual(first["detected_action"], "CLAP")
        self.assertEqual(counter.update({"CLAP": True}, elapsed_time=16.1)["counts"]["CLAP"], 1)

    def test_open_arms_in_both_screen_directions(self):
        for left, right in ((0.6, 0.4), (0.4, 0.6)):
            direction = 1 if left > right else -1
            self.assertTrue(is_arms_open(pose((left, .4), (right, .4),
                                              (left + .2 * direction, .4),
                                              (right - .2 * direction, .4))))
        self.assertFalse(is_arms_open(pose((.6, .4), (.4, .4), (.5, .4), (.5, .4))))

    def test_invisible_or_out_of_frame_wrist_is_not_open(self):
        self.assertFalse(is_arms_open(pose((.6, .4), (.4, .4), (.8, .4), (.2, 1.2))))
        self.assertFalse(is_arms_open(pose((.6, .4), (.4, .4), (.8, .4), (.2, .4), visibility=.1)))

    def test_lowered_elbow_is_not_open(self):
        landmarks = pose((.6, .4), (.4, .4), (.8, .4), (.2, .4))
        landmarks[13].y = .7
        self.assertFalse(is_arms_open(landmarks))

if __name__ == "__main__":
    unittest.main()
