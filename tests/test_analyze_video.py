import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from analyze_video import analyze_video
from test_fall_detector import LYING, STANDING


class FakeAnalyzer:
    timestamps = []

    def analyze_image_bytes(self, image, *, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        return {"detected": True, "landmarks": STANDING if timestamp_ms == 0 else LYING,
                "detections": {"LEFT_HAND_UP": True, "RIGHT_HAND_UP": True}}

    def close(self):
        pass


class AnalyzeVideoTest(unittest.TestCase):
    def setUp(self):
        test_root = Path("data")
        test_root.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=test_root)
        self.addCleanup(self.directory.cleanup)
        self.video = Path(self.directory.name) / "sample.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 64))
        self.assertTrue(writer.isOpened())
        for _ in range(7):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        FakeAnalyzer.timestamps = []

    @patch("analyze_video.PoseAnalyzer", FakeAnalyzer)
    def test_fall_uses_video_time_even_when_processed_quickly(self):
        output = Path(self.directory.name) / "fall.csv"
        self.assertEqual(analyze_video(self.video, "fall", output), 7)
        with output.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(FakeAnalyzer.timestamps, [0, 200, 400, 600, 800, 1000, 1200])
        self.assertEqual(rows[1]["fall_status"], "FALL_SUSPECTED")
        self.assertEqual(rows[-1]["fall_status"], "FALL_CONFIRMED")

    @patch("analyze_video.PoseAnalyzer", FakeAnalyzer)
    def test_rhythm_reports_all_candidates_and_does_not_overwrite(self):
        output = Path(self.directory.name) / "rhythm.csv"
        analyze_video(self.video, "rhythm", output, interval=0.4)
        with output.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(FakeAnalyzer.timestamps, [0, 400, 800, 1200])
        self.assertEqual(rows[0]["current_actions"], "LEFT_HAND_UP|RIGHT_HAND_UP")
        self.assertEqual(rows[0]["detected_action"], "")
        with self.assertRaises(FileExistsError):
            analyze_video(self.video, "rhythm", output)


if __name__ == "__main__":
    unittest.main()
