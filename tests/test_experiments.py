import csv
from pathlib import Path
import tempfile
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

import cv2
from fastapi.testclient import TestClient
import numpy as np

from experiment_manager import ExperimentManager
from fall_detector import FallDetector
import test_server


def complete_pose(*, lying=False):
    points = [SimpleNamespace(x=0.5, y=0.5, visibility=0.99, presence=0.99) for _ in range(33)]
    if lying:
        shoulders = ((0.29, 0.66), (0.41, 0.66))
        hips = ((0.59, 0.69), (0.71, 0.69))
        knees = ((0.75, 0.71), (0.87, 0.71))
        ankles = ((0.91, 0.73), (1.0, 0.73))
    else:
        shoulders = ((0.44, 0.30), (0.56, 0.30))
        hips = ((0.44, 0.55), (0.56, 0.55))
        knees = ((0.44, 0.75), (0.56, 0.75))
        ankles = ((0.44, 0.95), (0.56, 0.95))
    for index, (x, y) in zip((11, 12), shoulders):
        points[index].x, points[index].y = x, y
    for index, (x, y) in zip((23, 24), hips):
        points[index].x, points[index].y = x, y
    for index, (x, y) in zip((25, 26), knees):
        points[index].x, points[index].y = x, y
    for index, (x, y) in zip((27, 28), ankles):
        points[index].x, points[index].y = x, y
    return points


class FakeAnalyzer:
    landmarks = complete_pose()

    def analyze_image_bytes(self, image, *, diagnostics=None):
        if diagnostics is not None:
            diagnostics.update(width=64, height=64, inference_ms=1)
        points = self.landmarks
        if points is not None:
            points = [
                point if point is not None else SimpleNamespace(x=0.5, y=0.5, visibility=0, presence=0)
                for point in points
            ]
        return dict(detected=points is not None, landmarks=points, detections={})

    def close(self):
        pass


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        root = Path("data")
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.now = 0.0
        self.manager = ExperimentManager(self.temp.name, test_server.lock,
                                         save_frames=True, clock=lambda: self.now)
        self.addCleanup(self.manager.close)
        self.addCleanup(test_server.sessions.clear)
        self.manager_patch = patch.object(test_server, "experiments", self.manager)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        analyzer = patch.object(test_server, "PoseAnalyzer", FakeAnalyzer)
        analyzer.start()
        self.addCleanup(analyzer.stop)
        timer = patch("fall_detector.time.monotonic", side_effect=lambda: self.now)
        timer.start()
        self.addCleanup(timer.stop)
        FakeAnalyzer.landmarks = complete_pose()
        self.client = TestClient(test_server.app)
        self.addCleanup(self.client.close)
        ok, encoded = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertTrue(ok)
        self.image = encoded.tobytes()

    def post(self, route, data):
        response = self.client.post("/experiment/api/" + route, json=data)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def create(self, kind="fall", actions=None, sets=1, seconds=10):
        return self.post("create", dict(
            experiment_id=uuid4().hex, participant_id="P01", experiment_type=kind,
            actions=actions or ["front"], sets=sets, observation_seconds=seconds,
        ))

    def command(self, state):
        run = state["current"]
        return dict(experiment_id=run["id"], trial_id=run["trials"][run["index"]]["trial_id"])

    def frame(self, session="temi"):
        response = self.client.post("/test/fall", data={"session_id": session},
                                    files={"image": ("frame.jpg", self.image, "image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def record_result(self, detected):
        # Tests persistence/statistics independently of the recognition model.
        with self.manager.lock:
            ticket = self.manager.ticket("temi")
            with self.manager.capture_frame(ticket, self.image) as diagnostics:
                result = dict(detected=True, fall_status="FALL_CONFIRMED" if detected else "NORMAL",
                              fall_detected=detected, fall_metrics={"reliable_upper_body_visible": True})
                self.manager.record_frame(ticket, result, diagnostics)

    def test_default_sequence_and_idempotent_commands(self):
        config = dict(experiment_id=uuid4().hex, participant_id="P01", experiment_type="fall",
                      actions=["front", "left", "right", "back"], sets=5, observation_seconds=10)
        state = self.post("create", config)
        self.post("create", config)
        self.assertEqual(len(self.manager.runs), 1)
        self.assertEqual([t["action"] for t in state["current"]["trials"]],
                         ["front", "left", "right", "back"] * 5)
        self.assertEqual(state["current"]["trials"][-1]["set_number"], 5)
        command = self.command(state)
        self.post("start", command)
        self.now = 1
        self.post("start", command)
        self.assertEqual(self.manager.started_clock, 0)
        self.post("end", command)
        self.post("end", command)
        self.post("next", command)
        self.post("next", command)
        self.assertEqual(self.manager.snapshot()["current"]["index"], 1)
        self.post("start", self.command(self.manager.snapshot()))
        self.assertEqual(self.manager._trial(self.manager._run())["frame_count"], 0)

    def test_real_fsm_and_trial_debug_images_are_connected(self):
        self.frame()  # Preparation frame must not count toward the trial.
        self.assertEqual(self.client.get("/experiment").status_code, 200)
        state = self.create()
        command = self.command(state)
        test_server.sessions["fall"]["fall"].status = "FALL_CONFIRMED"
        self.post("start", command)
        self.assertEqual(test_server.sessions["fall"]["fall"].status, "NORMAL")
        sequence = [(0.0, False)] + [(0.4 * index, True) for index in range(1, 13)]
        for now, is_lying in sequence:
            self.now = now
            FakeAnalyzer.landmarks = complete_pose(lying=is_lying)
            result = self.frame()
        self.assertTrue(result["fall_detected"])
        self.assertEqual(self.manager._trial(self.manager._run())["status"], "running")
        FakeAnalyzer.landmarks = None
        self.now = 5.2
        self.assertEqual(self.frame()["fall_status"], "UNKNOWN")
        state = self.post("end", command)
        trial = state["current"]["trials"][0]
        self.assertEqual(trial["frame_count"], 14)
        self.assertEqual(trial["valid_pose_frames"], 13)
        self.assertEqual(trial["unknown_frames"], 1)
        self.assertEqual(trial["debug_errors"], 0)
        directory = self.manager.root / trial["debug_directory"]
        with (directory / "frames.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 14)
        self.assertTrue(all(row["trial_id"] == command["trial_id"] for row in rows))
        self.assertEqual(rows[-1]["response_fall_status"], "UNKNOWN")
        self.assertTrue(all((directory / row["comparison_image"]).is_file() for row in rows))
        response = self.client.get("/experiment/api/preview/" + command["experiment_id"] + "/" + command["trial_id"])
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR))
        self.assertEqual(self.client.get("/experiment/api/results.csv").status_code, 200)

    def test_timeout_without_frames_and_restart_preserves_results(self):
        state = self.create(actions=["front", "left"], seconds=2)
        command = self.command(state)
        self.post("start", command)
        self.now = 3
        self.manager.tick()
        trial = self.manager._trial(self.manager._run())
        self.assertEqual(trial["end_reason"], "timeout")
        self.assertEqual(trial["frame_count"], 0)
        self.assertEqual(trial["connection_gaps"], 1)
        self.assertEqual(self.manager.summary()["excluded"], 1)
        self.post("review", dict(**command, actual_action="전방", actual_fall=True, valid=True, notes=""))
        self.assertEqual(self.manager.summary()["TP"], 0)  # No observations cannot inflate metrics.
        self.post("next", command)
        self.post("start", self.command(self.manager.snapshot()))
        # Simulate a process crash: persist running state, but skip orderly trial closure.
        self.manager.recorder.close()
        self.manager.recorder = None
        reopened = ExperimentManager(self.temp.name, RLock(), save_frames=False)
        try:
            trials = reopened.snapshot()["current"]["trials"]
            self.assertEqual(trials[0]["end_reason"], "timeout")
            self.assertEqual(trials[1]["status"], "interrupted")
            self.assertFalse(trials[1]["valid"])
        finally:
            reopened.close()

    def test_late_requests_and_changed_sessions_do_not_mix_trials(self):
        state = self.create(actions=["front", "left"])
        first = self.command(state)
        stale = self.manager.ticket("temi")  # Received before trial start.
        self.post("start", first)
        test_server.analyze("fall", "temi", self.image, experiment_ticket=stale)
        self.assertEqual(self.manager._trial(self.manager._run())["frame_count"], 0)
        old_ticket = self.manager.ticket("temi")
        self.frame("temi")
        detector = test_server.sessions["fall"]["fall"]
        detector.status = "FALL_SUSPECTED"
        self.frame("different-session")
        self.assertIs(test_server.sessions["fall"]["fall"], detector)
        self.assertEqual(detector.status, "FALL_SUSPECTED")
        self.assertEqual(self.manager._trial(self.manager._run())["session_mismatch_frames"], 1)
        self.post("end", first)
        self.post("next", first)
        second = self.command(self.manager.snapshot())
        self.post("start", second)
        test_server.analyze("fall", "temi", self.image, experiment_ticket=old_ticket)
        self.assertEqual(self.manager._trial(self.manager._run())["frame_count"], 0)
        self.assertEqual(self.manager._run()["trials"][0]["frame_count"], 1)

    def test_reviewed_confusion_matrix_and_invalid_trial_exclusion(self):
        for kind, expected, detected in (
            ("fall", True, True), ("fall", True, False),
            ("nonfall", False, True), ("nonfall", False, False),
        ):
            state = self.create(kind=kind, actions=["front" if kind == "fall" else "sit"])
            command = self.command(state)
            self.post("start", command)
            self.record_result(detected)
            self.post("end", command)
            self.post("review", dict(**command, actual_action="검증 동작",
                                     actual_fall=expected, valid=True, notes=""))
            self.post("next", command)
        summary = self.manager.summary()
        self.assertEqual([summary[key] for key in ("TP", "FN", "FP", "TN")], [1, 1, 1, 1])
        self.assertEqual([summary[key] for key in ("precision", "recall", "specificity", "f1")], [0.5] * 4)
        state = self.manager.snapshot()
        command = self.command(state)
        self.post("review", dict(**command, actual_action="검증 동작",
                                 actual_fall=False, valid=False, notes="잘못 수행"))
        self.assertEqual(self.manager.summary()["TN"], 0)
        self.assertEqual(self.manager.summary()["excluded"], 1)

    def test_invalid_actions_and_unsafe_participant_id_rejected(self):
        config = dict(experiment_id=uuid4().hex, participant_id="../P01", experiment_type="fall",
                      actions=["front"], sets=5, observation_seconds=10)
        self.assertEqual(self.client.post("/experiment/api/create", json=config).status_code, 422)
        config.update(participant_id="P01", actions=["sit"])
        self.assertEqual(self.client.post("/experiment/api/create", json=config).status_code, 422)


if __name__ == "__main__":
    unittest.main()
