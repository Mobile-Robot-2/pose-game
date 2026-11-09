import csv
from pathlib import Path
import tempfile
import unittest

from rhythm_trials import RhythmTrials


class RhythmTrialsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="data")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "trials.csv"
        self.trials = RhythmTrials(self.path)

    def start(self):
        return self.trials.start("P01", "CLAP", 1, "temi")

    def record(self, status, action=None, session="temi"):
        self.trials.record(session, {"detected": status != "pose_not_detected",
                                     "prediction_status": status, "detected_action": action})

    def test_saves_one_trial_and_clap_event_without_using_truth_to_select(self):
        trial = self.start()
        self.record("no_action")
        self.record("detected", "CLAP")
        row = self.trials.finish(trial["trial_id"])
        self.assertEqual((row["detected_action"], row["prediction_status"]), ("CLAP", "detected"))
        with self.path.open(encoding="utf-8-sig", newline="") as file:
            saved = list(csv.DictReader(file))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["actual_action"], "CLAP")
        self.assertEqual(saved[0]["detected_actions"], "CLAP:1")
        frame_path = self.path.parent / saved[0]["frames_file"]
        with frame_path.open(encoding="utf-8-sig", newline="") as file:
            frames = list(csv.DictReader(file))
        self.assertEqual([frame["prediction_status"] for frame in frames], ["no_action", "detected"])
        self.assertEqual(frames[1]["detected_action"], "CLAP")
        self.assertIsNone(self.trials.snapshot())

    def test_different_predictions_or_conflict_are_not_forced_to_truth(self):
        trial = self.start()
        self.record("detected", "LEFT_HAND_UP")
        self.record("detected", "CLAP")
        row = self.trials.finish(trial["trial_id"])
        self.assertEqual((row["detected_action"], row["prediction_status"]), ("", "conflict"))

    def test_missing_frames_and_changed_session_are_invalid(self):
        trial = self.start()
        self.record("detected", "CLAP", "other")
        row = self.trials.finish(trial["trial_id"])
        self.assertEqual(row["prediction_status"], "invalid")
        self.assertEqual(row["session_mismatch_frames"], 1)

    def test_connection_gap_makes_trial_invalid(self):
        now = [0.0]
        self.trials = RhythmTrials(self.path, clock=lambda: now[0])
        trial = self.start()
        self.record("detected", "CLAP")
        now[0] = 2.1
        row = self.trials.finish(trial["trial_id"])
        self.assertEqual(row["prediction_status"], "invalid")
        self.assertEqual(row["connection_gaps"], 1)

    def test_pose_loss_and_no_action_are_distinct(self):
        trial = self.start()
        self.record("pose_not_detected")
        self.assertEqual(self.trials.finish(trial["trial_id"])["prediction_status"], "pose_not_detected")
        trial = self.start()
        self.record("no_action")
        self.assertEqual(self.trials.finish(trial["trial_id"])["prediction_status"], "no_action")
        with self.path.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["participant_id"], "P01")

    def test_trial_identity_and_single_active_trial(self):
        with self.assertRaises(ValueError):
            self.trials.start("P01", "CLAP", 1, None)
        trial = self.start()
        with self.assertRaises(ValueError):
            self.start()
        with self.assertRaises(ValueError):
            self.trials.finish("wrong")
        self.assertEqual(self.trials.snapshot()["trial_id"], trial["trial_id"])

    def test_failed_save_keeps_trial_available_for_retry(self):
        trial = self.start()
        self.record("detected", "CLAP")
        self.trials.path = Path(self.tmp.name)
        with self.assertRaises(OSError):
            self.trials.finish(trial["trial_id"])
        self.assertEqual(self.trials.snapshot()["trial_id"], trial["trial_id"])
        self.trials.path = self.path
        self.assertEqual(self.trials.finish(trial["trial_id"])["detected_action"], "CLAP")


if __name__ == "__main__":
    unittest.main()
