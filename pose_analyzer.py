from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from action_counter import ACTION_NAMES, detect_actions


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"


class PoseAnalyzer:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Download pose_landmarker_lite.task into the models folder."
            )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.start_time = time.time()
        self.last_timestamp_ms = -1

    def analyze_image_bytes(self, image_bytes: bytes, *, diagnostics: dict | None = None) -> dict:
        bgr_image = self._decode_image(image_bytes)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        timestamp_ms = self._next_timestamp_ms()
        if diagnostics is not None:
            diagnostics["height"], diagnostics["width"] = bgr_image.shape[:2]
            inference_start = time.perf_counter()
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if diagnostics is not None:
            diagnostics["inference_ms"] = (time.perf_counter() - inference_start) * 1000
        if not result.pose_landmarks:
            return {
                "detected": False,
                "detections": {action: False for action in ACTION_NAMES},
                "landmarks": None,
            }

        landmarks = result.pose_landmarks[0]
        return {
            "detected": True,
            "detections": detect_actions(landmarks),
            "landmarks": landmarks,
        }

    def close(self) -> None:
        self.landmarker.close()

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError("Could not decode image bytes. Send a valid JPEG or PNG image.")
        return bgr_image

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms
