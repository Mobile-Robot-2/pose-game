import unittest
from types import SimpleNamespace

from action_counter import is_arms_open


def pose(left_shoulder, right_shoulder, left_wrist, right_wrist, visibility=1.0):
    landmarks = [SimpleNamespace(x=0, y=0, visibility=visibility) for _ in range(33)]
    for index, (x, y) in zip((11, 12, 15, 16), (left_shoulder, right_shoulder, left_wrist, right_wrist)):
        landmarks[index] = SimpleNamespace(x=x, y=y, visibility=visibility)
    for index, shoulder, wrist in ((13, left_shoulder, left_wrist), (14, right_shoulder, right_wrist)):
        landmarks[index] = SimpleNamespace(x=(shoulder[0] + wrist[0]) / 2,
                                           y=(shoulder[1] + wrist[1]) / 2, visibility=visibility)
    return landmarks


class RhythmActionsTest(unittest.TestCase):
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
