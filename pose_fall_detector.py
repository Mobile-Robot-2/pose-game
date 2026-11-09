"""Pose-feature two-stage fall detector using MediaPipe Pose landmarks.

Stage 1 follows the symbols and thresholds in Han et al. (2020). The paper's
Stage 2 uses a trained SVM over [gamma, epsilon, tau]. This project does not
train/load that SVM; its explicit rule-based replacement is documented in
docs/paper-fall-detector.md and must not be described as the paper's classifier.
"""
from __future__ import annotations

import math
import time


LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

ANGLE_LIMIT = 30.0
ANGLE_LOWER = 30.0
ANGLE_UPPER = 50.0
RATIO_BOUNDARY = 1.0
UNSTABLE_ENTRY_FRAMES = 3
STABLE_EXIT_FRAMES = 3
NSPR_FEATURE_FRAMES = 10


def _point(landmarks, index: int) -> tuple[float, float]:
    point = landmarks[index]
    x, y = float(point.x), float(point.y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Non-finite pose landmark")
    return x, y


def _midpoint(first: tuple[float, float], second: tuple[float, float]):
    return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)


def _length(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _vector_angle_from_vertical(vector: tuple[float, float]) -> float:
    """Unsigned deflection from gravity (vertical), in [0, 90] degrees.

    Image y increases downwards. Taking absolute components makes a segment's
    angle independent of endpoint order, while vertical remains 0 degrees.
    """
    dx, dy = vector
    if math.hypot(dx, dy) <= 1e-12:
        raise ValueError("Degenerate posture vector")
    return math.degrees(math.atan2(abs(dx), abs(dy)))


class PoseFallDetector:
    """Stateful detector; update() accepts MediaPipe's 33 normalized landmarks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> dict:
        self.status = "NORMAL"
        self.unstable_streak = 0
        self.stable_streak = 0
        self.nspr_active = False
        self.nspr_frames: list[dict] = []
        self.reference_midhip = None
        self.reference_spine_length = None
        self.last_timestamp = None
        self.final_result = "NON-FALL"
        self.last_features = self._empty_features()
        return self.get_state()

    def update(self, landmarks, *, timestamp: float | None = None) -> dict:
        now = time.monotonic() if timestamp is None else float(timestamp)
        frame_interval = None if self.last_timestamp is None else now - self.last_timestamp
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError("Timestamps must be strictly increasing")
        self.last_timestamp = now

        try:
            features = self._extract_features(landmarks)
        except (IndexError, TypeError, ValueError, AttributeError):
            self.unstable_streak = 0
            self.stable_streak = 0
            self.last_features = self._empty_features()
            self.last_features["frame_interval_seconds"] = frame_interval
            self.last_features["stage1_state"] = "UNKNOWN"
            self.last_features["nspr_active"] = self.nspr_active
            self.last_features["nspr_frame_count"] = len(self.nspr_frames)
            self.last_features["final_result"] = self.final_result
            self.last_features["fall_status"] = self.status
            return self.get_state()

        self.last_features = features
        features["frame_interval_seconds"] = frame_interval
        state = features["stage1_state"]

        if state == "Stable":
            self.stable_streak += 1
            self.unstable_streak = 0
            if not self.nspr_active:
                self._update_reference(features)
                if self.status == "RECOVERED":
                    self.status = "NORMAL"
            elif self.stable_streak >= STABLE_EXIT_FRAMES:
                self.nspr_active = False
                self.nspr_frames.clear()
                was_confirmed = self.status == "FALL_CONFIRMED"
                if not was_confirmed:
                    self.final_result = "NON-FALL"
                self.status = "RECOVERED" if was_confirmed else "NORMAL"
                self._update_reference(features)
        else:
            self.stable_streak = 0
            self.unstable_streak += 1
            if not self.nspr_active and self.unstable_streak >= UNSTABLE_ENTRY_FRAMES:
                self.nspr_active = True
                self.status = "FALL_SUSPECTED"
                self.final_result = "NON-FALL"

        if self.nspr_active and self.status != "FALL_CONFIRMED":
            record = dict(features)
            record["midhip"] = features["midhip"]
            record["lower_limb_y"] = features["lower_limb_y"]
            self.nspr_frames.append(record)
            self._add_temporal_features()
            if state == "Stable" and self.stable_streak >= STABLE_EXIT_FRAMES:
                # The third stable frame closes the NSPR as NON-FALL.
                self.last_features["nspr_frame_count"] = len(self.nspr_frames)
            elif len(self.nspr_frames) >= NSPR_FEATURE_FRAMES:
                self._finalize_rule_based_window()

        features["nspr_active"] = self.nspr_active
        features["nspr_frame_count"] = len(self.nspr_frames)
        features["unstable_streak"] = self.unstable_streak
        features["stable_streak"] = self.stable_streak
        features["final_result"] = self.final_result
        features["fall_status"] = self.status
        return self.get_state()

    def update_no_pose(self, *, timestamp: float | None = None) -> dict:
        """A missing pose breaks consecutive-frame runs and is not an NSPR sample."""
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self.last_timestamp is not None and now <= self.last_timestamp:
            raise ValueError("Timestamps must be strictly increasing")
        interval = None if self.last_timestamp is None else now - self.last_timestamp
        self.last_timestamp = now
        self.unstable_streak = 0
        self.stable_streak = 0
        self.last_features = self._empty_features()
        self.last_features["frame_interval_seconds"] = interval
        self.last_features["stage1_state"] = "UNKNOWN"
        self.last_features["nspr_active"] = self.nspr_active
        self.last_features["nspr_frame_count"] = len(self.nspr_frames)
        self.last_features["final_result"] = self.final_result
        self.last_features["fall_status"] = self.status
        return self.get_state()

    def _extract_features(self, landmarks) -> dict:
        ls, rs = _point(landmarks, LEFT_SHOULDER), _point(landmarks, RIGHT_SHOULDER)
        lh, rh = _point(landmarks, LEFT_HIP), _point(landmarks, RIGHT_HIP)
        lk, rk = _point(landmarks, LEFT_KNEE), _point(landmarks, RIGHT_KNEE)
        la, ra = _point(landmarks, LEFT_ANKLE), _point(landmarks, RIGHT_ANKLE)
        neck, midhip = _midpoint(ls, rs), _midpoint(lh, rh)
        spine = (neck[0] - midhip[0], neck[1] - midhip[1])
        waist_width = (rh[0] - lh[0], rh[1] - lh[1])
        spine_length, waist_length = _length(spine), _length(waist_width)
        if waist_length <= 1e-12:
            raise ValueError("Degenerate waist-width vector")

        left_thigh = _vector_angle_from_vertical((lk[0] - lh[0], lk[1] - lh[1]))
        right_thigh = _vector_angle_from_vertical((rk[0] - rh[0], rk[1] - rh[1]))
        left_calf = _vector_angle_from_vertical((la[0] - lk[0], la[1] - lk[1]))
        right_calf = _vector_angle_from_vertical((ra[0] - rk[0], ra[1] - rk[1]))
        spine_angle = _vector_angle_from_vertical(spine)
        thigh_angle = (left_thigh + right_thigh) / 2
        calf_angle = (left_calf + right_calf) / 2
        spine_ratio = spine_length / waist_length

        if spine_angle < ANGLE_LOWER and spine_ratio > RATIO_BOUNDARY:
            trend = "T1"
        elif spine_angle > ANGLE_UPPER or spine_ratio < RATIO_BOUNDARY:
            trend = "T3"
        else:
            trend = "T2"

        pairs = (
            (spine_angle < ANGLE_LIMIT and left_thigh < ANGLE_LIMIT),
            (spine_angle < ANGLE_LIMIT and right_thigh < ANGLE_LIMIT),
            (left_calf < ANGLE_LIMIT and left_thigh < ANGLE_LIMIT),
            (right_calf < ANGLE_LIMIT and right_thigh < ANGLE_LIMIT),
        )
        stability = "S1" if any(pairs) else "S2"
        if stability == "S1" and trend == "T1":
            state = "Stable"
        elif stability == "S2" and trend == "T3":
            state = "Disordered"
        else:
            state = "Fluctuating"

        return {
            "spine_deflection_angle": spine_angle,
            "thigh_deflection_angle": thigh_angle,
            "left_thigh_angle": left_thigh,
            "right_thigh_angle": right_thigh,
            "calf_deflection_angle": calf_angle,
            "left_calf_angle": left_calf,
            "right_calf_angle": right_calf,
            "spine_ratio": spine_ratio,
            "trend_symbol": trend,
            "stability_symbol": stability,
            "stage1_state": state,
            "midhip": midhip,
            "spine_length": spine_length,
            "lower_limb_y": (lk[1], la[1], rk[1], ra[1]),
            "thresholds": {
                "angle_limit_degrees": ANGLE_LIMIT,
                "spine_angle_lower_degrees": ANGLE_LOWER,
                "spine_angle_upper_degrees": ANGLE_UPPER,
                "spine_ratio_boundary": RATIO_BOUNDARY,
            },
        }

    def _update_reference(self, features: dict) -> None:
        self.reference_midhip = features["midhip"]
        self.reference_spine_length = features["spine_length"]

    def _finalize_rule_based_window(self) -> None:
        """Variant rule: sustained NSPR for 10 samples is FALL.

        The paper instead classifies [gamma, epsilon, tau] with a trained SVM.
        No additional, unpublished numerical cutoff is introduced here.
        """
        self.final_result = "FALL"
        self.status = "FALL_CONFIRMED"
        self._add_temporal_features()

    def _add_temporal_features(self) -> None:
        if not self.nspr_frames:
            return
        if self.reference_midhip is None or not self.reference_spine_length:
            gamma = None
        else:
            hx, hy = self.reference_midhip
            height = self.reference_spine_length
            gamma = 0.0
            for sample in self.nspr_frames:
                x, y = sample["midhip"]
                # Paper Eq. (5): softsign(reference_y - current_midhip_y)
                # times normalized distance from the stable reference point.
                delta_y = hy - y
                softsign = delta_y / (1.0 + abs(delta_y))
                gamma += softsign * math.hypot(x - hx, y - hy) / height
        means = [
            sum(sample["lower_limb_y"][i] for sample in self.nspr_frames) / len(self.nspr_frames)
            for i in range(4)
        ]
        height = self.reference_spine_length
        epsilon = None if not height else sum(
            abs(sample["lower_limb_y"][i] - means[i]) / height
            for sample in self.nspr_frames for i in range(4)
        )
        score_map = {"Stable": 0, "Fluctuating": 1, "Disordered": 2}
        tau = sum(score_map[sample["stage1_state"]] for sample in self.nspr_frames)
        self.last_features.update(gamma=gamma, epsilon=epsilon, tau=tau)

    def get_state(self) -> dict:
        return {
            "fall_status": self.status,
            "fall_detected": self.final_result == "FALL",
            "final_result": self.final_result,
            "nspr_active": self.nspr_active,
            "nspr_frame_count": len(self.nspr_frames),
            "unstable_streak": self.unstable_streak,
            "stable_streak": self.stable_streak,
            "features": dict(self.last_features),
        }

    @staticmethod
    def _empty_features() -> dict:
        return {
            "spine_deflection_angle": None,
            "thigh_deflection_angle": None,
            "left_thigh_angle": None,
            "right_thigh_angle": None,
            "calf_deflection_angle": None,
            "left_calf_angle": None,
            "right_calf_angle": None,
            "spine_ratio": None,
            "trend_symbol": None,
            "stability_symbol": None,
            "stage1_state": "UNKNOWN",
            "frame_interval_seconds": None,
            "nspr_active": False,
            "nspr_frame_count": 0,
            "final_result": "NON-FALL",
            "fall_status": "NORMAL",
            "thresholds": {
                "angle_limit_degrees": ANGLE_LIMIT,
                "spine_angle_lower_degrees": ANGLE_LOWER,
                "spine_angle_upper_degrees": ANGLE_UPPER,
                "spine_ratio_boundary": RATIO_BOUNDARY,
            },
        }
