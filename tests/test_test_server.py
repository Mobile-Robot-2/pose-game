import sys
import types
import unittest
from unittest.mock import patch

# The test exercises session logic without requiring a camera or MediaPipe runtime.
fake_pose = types.ModuleType("pose_analyzer")
fake_pose.PoseAnalyzer = object
sys.modules.setdefault("pose_analyzer", fake_pose)

import test_server


class FakeAnalyzer:
    def __init__(self):
        self.closed = False

    def analyze_image_bytes(self, image):
        return {"detected": True, "landmarks": [],
                "detections": {"LEFT_HAND_UP": True, "RIGHT_HAND_UP": True}}

    def close(self):
        self.closed = True


class TestSessions(unittest.TestCase):
    def tearDown(self):
        test_server.shutdown()

    @patch.object(test_server, "PoseAnalyzer", FakeAnalyzer)
    def test_session_reset_and_ambiguous_actions(self):
        result = test_server.analyze("rhythm", "one", b"image")
        first = test_server.sessions["rhythm"]["analyzer"]
        self.assertEqual(len(result["current_actions"]), 2)
        self.assertIsNone(result["detected_action"])
        test_server.analyze("rhythm", "one", b"image")
        self.assertIs(test_server.sessions["rhythm"]["analyzer"], first)
        test_server.analyze("rhythm", "two", b"image")
        self.assertTrue(first.closed)
        self.assertIsNot(test_server.sessions["rhythm"]["analyzer"], first)

    def test_empty_image_rejected(self):
        with self.assertRaises(ValueError):
            test_server.analyze("fall", "one", b"")

    @patch.object(test_server, "PoseAnalyzer", FakeAnalyzer)
    def test_landmarks_are_returned_and_empty_when_pose_is_lost(self):
        point = types.SimpleNamespace(x=0.2, y=0.4, visibility=0.8)
        with patch.object(FakeAnalyzer, "analyze_image_bytes", return_value={
                "detected": True, "landmarks": [point], "detections": {}}):
            result = test_server.analyze("rhythm", "one", b"image")
            self.assertEqual(result["landmarks"], [{"x": 0.2, "y": 0.4, "visibility": 0.8}])
        with patch.object(FakeAnalyzer, "analyze_image_bytes", return_value={
                "detected": False, "landmarks": None, "detections": {}}):
            result = test_server.analyze("rhythm", "one", b"image")
            self.assertEqual(result["landmarks"], [])


if __name__ == "__main__":
    unittest.main()
