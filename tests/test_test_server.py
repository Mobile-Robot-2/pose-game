import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import os

# The test exercises session logic without requiring a camera or MediaPipe runtime.
fake_pose = types.ModuleType("pose_analyzer")
fake_pose.PoseAnalyzer = object
sys.modules.setdefault("pose_analyzer", fake_pose)

import test_server
from rhythm_trials import RhythmTrials
from experiment_manager import ExperimentManager


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
    def test_raw_candidates_and_conflicts_are_reported(self):
        raw = {"LEFT_HAND_UP": True, "ARMS_OPEN": True, "CLAP": True}
        with patch.object(FakeAnalyzer, "analyze_image_bytes", return_value={
                "detected": True, "landmarks": [], "detections": {"CLAP": True}, "raw_detections": raw}):
            result = test_server.analyze("rhythm", "one", b"image")
        self.assertEqual(result["current_actions"], list(raw))
        self.assertIsNone(result["detected_action"])
        self.assertEqual(result["prediction_status"], "conflict")

    @patch.object(test_server, "PoseAnalyzer", FakeAnalyzer)
    def test_open_arms_wins_over_hand_raises(self):
        raw = {"LEFT_HAND_UP": True, "RIGHT_HAND_UP": True, "ARMS_OPEN": True, "CLAP": False}
        with patch.object(FakeAnalyzer, "analyze_image_bytes", return_value={
                "detected": True, "landmarks": [], "detections": raw}):
            result = test_server.analyze("rhythm", "one", b"image")
        self.assertEqual(result["detected_action"], "ARMS_OPEN")
        self.assertEqual(result["prediction_status"], "detected")

    @patch.object(test_server, "PoseAnalyzer", FakeAnalyzer)
    def test_rhythm_trial_routes_capture_frame_and_save(self):
        with tempfile.TemporaryDirectory(dir="data") as directory, \
                patch.object(test_server, "rhythm_trials", RhythmTrials(Path(directory) / "trials.csv")):
            test_server.analyze("rhythm", "temi", b"image")
            trial = test_server.start_rhythm_trial(test_server.RhythmTrialStart(
                participant_id="P01", actual_action="LEFT_HAND_UP", trial_number=1))
            with patch.object(FakeAnalyzer, "analyze_image_bytes", return_value={
                    "detected": True, "landmarks": [], "detections": {"LEFT_HAND_UP": True}}):
                test_server.analyze("rhythm", "temi", b"image")
            result = test_server.end_rhythm_trial(test_server.RhythmTrialEnd(trial_id=trial["trial_id"]))
            self.assertEqual(result["actual_action"], "LEFT_HAND_UP")
            self.assertEqual(result["detected_action"], "LEFT_HAND_UP")
            self.assertTrue(test_server.rhythm_trials.path.is_file())

    def test_fall_dashboard_rejects_new_session_during_trial(self):
        class DiagnosticAnalyzer(FakeAnalyzer):
            def analyze_image_bytes(self, image, *, diagnostics=None):
                return {"detected": True, "landmarks": [], "detections": {}}

        class FakeFall:
            def reset(self):
                pass

            def update(self, landmarks, *, diagnostics=None):
                return {"fall_status": "NORMAL", "fall_detected": False,
                        "fall_metrics": {"head_shoulders_visible": True,
                                         "reliable_upper_body_visible": True}}

        with tempfile.TemporaryDirectory(dir="data") as directory:
            manager = ExperimentManager(Path(directory), test_server.lock)
            try:
                with patch.object(test_server, "experiments", manager), \
                        patch.object(test_server, "PoseAnalyzer", DiagnosticAnalyzer), \
                        patch.object(test_server, "FallDetector", FakeFall):
                    test_server.analyze("fall", "old", b"image")
                    manager.create("a" * 32, "P01", "fall", ["front"], 1, 10)
                    trial = manager.snapshot()["current"]["trials"][0]
                    manager.start_trial("a" * 32, trial["trial_id"],
                                        test_server.sessions["fall"]["fall"].reset, "old",
                                        "posture_experimental")
                    self.assertEqual(manager.snapshot()["current"]["trials"][0]["fall_confirmation_policy"],
                                     "posture_experimental")
                    response = test_server.analyze("fall", "new", b"image")
                    self.assertFalse(response["detected"])
                    self.assertEqual(response["fall_status"], "UNKNOWN")
                    self.assertEqual(manager.snapshot()["current"]["trials"][0]["session_mismatch_frames"], 1)
            finally:
                manager.close()

    def test_experimental_fall_policy_is_visible_and_applied_to_session(self):
        class FakeFall:
            def __init__(self, *, require_fall_posture=False):
                self.require_fall_posture = require_fall_posture

            def update(self, landmarks, *, diagnostics=None):
                return {"fall_status": "NORMAL", "fall_detected": False,
                        "fall_metrics": {"head_shoulders_visible": True,
                                         "reliable_upper_body_visible": True}}

        with patch.object(test_server, "fall_confirmation_policy", "posture_experimental"), \
                patch.object(test_server, "PoseAnalyzer", FakeAnalyzer), \
                patch.object(test_server, "FallDetector", FakeFall):
            response = test_server.analyze("fall", "temi", b"image")
            self.assertEqual(response["confirmation_policy"], "posture_experimental")
            self.assertTrue(test_server.sessions["fall"]["fall"].require_fall_posture)
            self.assertEqual(test_server.test_config()["fall_confirmation_policy"], "posture_experimental")

    def test_unknown_fall_policy_fails_before_starting_experiment_manager(self):
        with patch.dict(os.environ, {"FALL_CONFIRMATION_POLICY": "typo"}), \
                patch.object(test_server, "fall_confirmation_policy", "baseline"):
            with self.assertRaises(ValueError):
                test_server.startup()

    def test_dashboard_start_passes_selected_confirmation_policy(self):
        with patch.object(test_server, "fall_confirmation_policy", "posture_experimental"), \
                patch.object(test_server, "experiments") as manager:
            test_server.experiment_start(test_server.TrialCommand(experiment_id="run", trial_id="trial"))
            self.assertEqual(manager.start_trial.call_args.args[-1], "posture_experimental")

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
