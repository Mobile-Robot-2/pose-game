import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import count
import json
import logging
from pathlib import Path
from queue import Full, Queue
from threading import Thread
import time
from uuid import uuid4

import cv2
import numpy as np

from fall_detector import (
    LEFT_HIP, LEFT_SHOULDER, MIN_LANDMARK_VISIBILITY, MIN_RELIABLE_UPPER_BODY_VISIBILITY,
    NOSE, RIGHT_HIP, RIGHT_SHOULDER,
)


LOGGER = logging.getLogger(__name__)
JOINTS = {
    "nose": NOSE, "left_shoulder": LEFT_SHOULDER, "right_shoulder": RIGHT_SHOULDER,
    "left_hip": LEFT_HIP, "right_hip": RIGHT_HIP,
}
CONNECTIONS = (
    (0, 11), (0, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
)
FIELDS = (
    "frame_id", "timestamp", "endpoint", "session_id", "frame_interval_ms",
    "observation_gap_ms", "pose_gap_grace", "inference_ms", "handler_ms", "width", "height",
    "pose_detected", "nose_y", "shoulder_y", "hip_y", "body_center_y",
    "torso_angle", "upper_body_angle", "upright_pose", "upright_history_count",
    "upright_samples_expired", "center_drop", "sudden_drop", "horizontal",
    "upper_body_horizontal", "low_position", "low_fall_pose", "fall_motion",
    "angle_fall_motion", "head_shoulder_drop",
    "low_pose_ms", "fall_status", "response_fall_status", "fall_detected", "landmark_visibility",
    "landmark_presence", "landmark_xy", "observation_reset_reason", "processing_error",
    "comparison_image",
)


class FallDebugRecorder:
    def __init__(self, root: Path, *, save_frames: bool) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = root / f"{stamp}-{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        self.save_frames = save_frames
        self._ids = count(1)
        self._last_request = {}
        self._queue = Queue(maxsize=16)
        self._dropped = 0
        self._file = (self.directory / "frames.csv").open("w", newline="", encoding="utf-8")
        self._worker = Thread(target=self._write, name="fall-debug", daemon=True)
        self._worker.start()
        LOGGER.info("Fall diagnostics directory: %s", self.directory)

    @contextmanager
    def capture(self, endpoint: str, session_id: str | None = None):
        started = time.perf_counter()
        previous_session, previous_time = self._last_request.get(endpoint, (None, None))
        self._last_request[endpoint] = (session_id, started)
        row = {
            "frame_id": next(self._ids),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "endpoint": endpoint,
            "session_id": session_id,
            "save_frames": self.save_frames,
            "frame_interval_ms": (
                (started - previous_time) * 1000
                if previous_time is not None and previous_session == session_id else None
            ),
        }
        try:
            yield row
        except Exception as exc:
            row["processing_error"] = type(exc).__name__
            raise
        finally:
            row["handler_ms"] = (time.perf_counter() - started) * 1000
            try:
                self._queue.put_nowait(row)
            except Full:
                self._dropped += 1
                if self._dropped == 1:
                    LOGGER.warning("Fall diagnostics queue full; dropping debug records")

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join()
        if self._dropped:
            LOGGER.warning("Dropped %d fall diagnostic records", self._dropped)

    def _write(self) -> None:
        with self._file:
            writer = csv.DictWriter(self._file, fieldnames=FIELDS, extrasaction="ignore")
            try:
                writer.writeheader()
                self._file.flush()
            except OSError:
                LOGGER.exception("Could not initialize fall diagnostics CSV")
            while (row := self._queue.get()) is not None:
                try:
                    self._write_row(writer, row)
                    self._file.flush()
                except Exception:
                    LOGGER.exception("Could not write fall diagnostics for frame %s", row["frame_id"])

    def _write_row(self, writer, row: dict) -> None:
        landmarks = row.get("landmarks")
        if landmarks:
            for field in ("visibility", "presence"):
                row[f"landmark_{field}"] = json.dumps({
                    name: getattr(landmarks[index], field, None) for name, index in JOINTS.items()
                })
            row["landmark_xy"] = json.dumps({
                name: [landmarks[index].x, landmarks[index].y] for name, index in JOINTS.items()
            })
            row.update(
                nose_y=landmarks[NOSE].y,
                shoulder_y=(landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2,
                hip_y=(landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2,
            )
        if self.save_frames and row.get("image_bytes"):
            try:
                self._save_comparison(row, landmarks)
            except (OSError, cv2.error):
                LOGGER.exception("Could not save fall comparison for frame %s", row["frame_id"])
        writer.writerow(row)

    def _save_comparison(self, row: dict, landmarks) -> None:
        raw = cv2.imdecode(np.frombuffer(row["image_bytes"], np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            return
        height, width = raw.shape[:2]
        overlay = raw.copy()
        if landmarks:
            pixels = {
                i: (int(p.x * width), int(p.y * height))
                for i, p in enumerate(landmarks)
                if 0 <= p.x < 1 and 0 <= p.y < 1
            }
            for first, second in CONNECTIONS:
                if first in pixels and second in pixels:
                    cv2.line(overlay, pixels[first], pixels[second], (70, 220, 120), 2)
            for index, pixel in pixels.items():
                visibility = getattr(landmarks[index], "visibility", None)
                threshold = (
                    MIN_RELIABLE_UPPER_BODY_VISIBILITY
                    if index in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)
                    else MIN_LANDMARK_VISIBILITY
                )
                color = (
                    (80, 240, 100) if visibility is not None and visibility >= threshold
                    else (0, 160, 255)
                )
                cv2.circle(overlay, pixel, 3, color, -1)
            nose = landmarks[NOSE]
            left_shoulder, right_shoulder = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
            left_hip, right_hip = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]
            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_x = (left_hip.x + right_hip.x) / 2
            centers = (
                ("N", nose.x, nose.y), ("S", shoulder_x, row["shoulder_y"]),
                ("H", hip_x, row["hip_y"]),
                ("C", (nose.x + shoulder_x + hip_x) / 3,
                 (nose.y + row["shoulder_y"] + row["hip_y"]) / 3),
            )
            for label, x, y in centers:
                if 0 <= x < 1 and 0 <= y < 1:
                    pixel = (int(x * width), int(y * height))
                    cv2.circle(overlay, pixel, 5, (255, 220, 0), -1)
                    cv2.putText(overlay, label, pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        comparison = np.hstack((raw, overlay))
        comparison = cv2.copyMakeBorder(comparison, 0, 82, 0, 0, cv2.BORDER_CONSTANT)
        lines = (
            f"#{row['frame_id']}   {row.get('fall_status', '-')}   fall={row.get('fall_detected', '-')}",
            f"torso={row.get('torso_angle', '-')}  upper={row.get('upper_body_angle', '-')}  drop={row.get('center_drop', '-')}",
            f"upright={row.get('upright_pose', '-')}  sudden_drop={row.get('sudden_drop', '-')}"
            f"  reset={row.get('observation_reset_reason', '-')}",
        )
        for index, line in enumerate(lines):
            cv2.putText(comparison, line, (8, height + 22 + index * 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        filename = f"frame-{row['frame_id']:06}.jpg"
        # imencode + pathlib supports Windows paths with non-ASCII characters.
        ok, encoded = cv2.imencode(".jpg", comparison)
        if ok:
            (self.directory / filename).write_bytes(encoded.tobytes())
            row["comparison_image"] = filename
