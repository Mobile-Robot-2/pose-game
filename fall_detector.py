from collections import deque
import math
import time
from datetime import datetime


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

MIN_LANDMARK_VISIBILITY = 0.5

FALL_CONFIRM_SECONDS = 0.8
RECOVERY_CONFIRM_SECONDS = 1.5
RECOVERED_DISPLAY_SECONDS = 2.0
EVIDENCE_GRACE_SECONDS = 0.6

DROP_WINDOW_SECONDS = 1.0
MIN_CENTER_DROP = 0.12
HORIZONTAL_ANGLE_DEGREES = 45.0
STRONG_HORIZONTAL_ANGLE_DEGREES = 75.0
UPPER_BODY_HORIZONTAL_ANGLE_DEGREES = 50.0
LOW_HEAD_Y = 0.36
LOW_HIP_Y = 0.55
LOW_SHOULDER_Y = 0.42
BODY_CENTER_LOW_Y = 0.50
MIN_UPPER_BODY_VISIBILITY = 0.50
MIN_RELIABLE_UPPER_BODY_VISIBILITY = 0.55


class FallDetector:
    """Stateful fall-suspicion detector using normalized pose landmarks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> dict:
        self.status = "NORMAL"
        self.suspected_since = None
        self.recovery_since = None
        self.recovered_since = None
        self.last_evidence_time = 0.0
        self.last_event = None
        self._new_event = False
        self.center_history = deque()
        self.upright_seen = False
        self.last_metrics = self._empty_metrics()
        return self.get_state()

    def update(self, landmarks) -> dict:
        previous_status = self.status
        now = time.monotonic()
        metrics = self._calculate_metrics(landmarks, now)
        self.last_metrics = metrics
        self._new_event = False

        strong_horizontal = (
            metrics["torso_angle"] >= STRONG_HORIZONTAL_ANGLE_DEGREES
        )
        fallen_pose = (
            metrics["reliable_upper_body_visible"]
            and metrics["horizontal"]
            and (
                metrics["low_position"]
                or (strong_horizontal and metrics["upper_body_low"])
            )
        )
        obvious_lying_pose = (
            metrics["reliable_upper_body_visible"]
            and metrics["upper_body_horizontal"]
            and metrics["upper_body_low"]
        )
        # 먼저 서 있거나 상체를 세운 자세를 확인해야 낙상 감지를 활성화한다.
        if (
            metrics["reliable_upper_body_visible"]
            and not metrics["horizontal"]
            and not metrics["upper_body_horizontal"]
            and not metrics["low_position"]
        ):
            self.upright_seen = True
            
        lying_pose = fallen_pose or obvious_lying_pose

        # 급격한 하강은 낙상 의심을 시작할 때만 확인한다.
        fall_trigger = (
            self.upright_seen
            and metrics["sudden_drop"]
            and lying_pose
        )

        if self.status == "NORMAL":
            if fall_trigger:
                self.status = "FALL_SUSPECTED"
                self.suspected_since = now
                self.last_evidence_time = now

        elif self.status == "FALL_SUSPECTED":
            # 의심 상태에서는 누운 자세가 유지되는지 확인한다.
            if lying_pose:
                self.last_evidence_time = now

                if (
                    self.suspected_since is not None
                    and now - self.suspected_since >= FALL_CONFIRM_SECONDS
                ):
                    self.status = "FALL_CONFIRMED"
                    self.recovery_since = None
                    self._create_event()

            elif now - self.last_evidence_time > EVIDENCE_GRACE_SECONDS:
                self.status = "NORMAL"
                self.suspected_since = None

        elif self.status == "FALL_CONFIRMED":
            recovered_pose = (
                not metrics["horizontal"]
                and not metrics["upper_body_horizontal"]
                and not metrics["low_position"]
            )
            if recovered_pose:
                if self.recovery_since is None:
                    self.recovery_since = now
                elif now - self.recovery_since >= RECOVERY_CONFIRM_SECONDS:
                    self.status = "RECOVERED"
                    self.recovered_since = now
                    self.suspected_since = None
                    if self.last_event is not None:
                        self.last_event["status"] = "CLOSED"
            else:
                self.recovery_since = None

        elif self.status == "RECOVERED":
            if fall_trigger:
                self.status = "FALL_SUSPECTED"
                self.suspected_since = now
                self.last_evidence_time = now
                self.recovered_since = None
            elif (
                self.recovered_since is not None
                and now - self.recovered_since >= RECOVERED_DISPLAY_SECONDS
            ):
                self.status = "NORMAL"
                self.recovered_since = None
                
        if self.status != previous_status:
            print(
                f"fall_state_transition: {previous_status} -> {self.status}, "
                f"upright_seen={self.upright_seen}, "
                f"sudden_drop={metrics['sudden_drop']}, "
                f"fall_trigger={fall_trigger}, "
                f"lying_pose={lying_pose}",
                flush=True,
            )
            
        return self.get_state()

    def update_no_pose(self) -> dict:
        self.last_metrics = self._empty_metrics()
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "fall_detected": self.status == "FALL_CONFIRMED",
            "fall_status": self.status,
            "fall_confidence": self._confidence(),
            "fall_metrics": dict(self.last_metrics),
            "fall_event": {
                "new_event": self._new_event,
                "event_id": self.last_event["event_id"] if self.last_event else None,
                "evidence_image": self.last_event.get("evidence_image") if self.last_event else None,
            },
        }

    def attach_evidence_image(self, evidence_image: str | None) -> None:
        if self.last_event is not None:
            self.last_event["evidence_image"] = evidence_image

    def mark_event_handled(self) -> None:
        self._new_event = False

    def _create_event(self) -> None:
        if self.last_event and self.last_event.get("status") == "OPEN":
            return

        event_id = datetime.now().strftime("fall_%Y%m%d_%H%M%S")
        self.last_event = {
            "event_id": event_id,
            "evidence_image": None,
            "status": "OPEN",
        }
        self._new_event = True

    def _calculate_metrics(self, landmarks, now: float) -> dict:
        nose = landmarks[NOSE]
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]

        # 상체(코/어깨)와 골반이 실제로 보일 때만 자세를 판단한다.
        # 화면 밖 관절은 MediaPipe가 추정한 좌표를 채워 넣으므로,
        # 이를 그대로 쓰면 차렷 자세(허리 아래만 보임)도 낙상으로 오인한다.
        upper_body_visible = (
            self._is_visible(nose)
            and self._is_visible(left_shoulder)
            and self._is_visible(right_shoulder)
        )
        hips_visible = self._is_visible(left_hip) and self._is_visible(right_hip)
        if not (upper_body_visible and hips_visible):
            self.center_history.clear()
            return self._empty_metrics()

        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_x = (left_hip.x + right_hip.x) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        body_center_y = (nose.y + shoulder_y + hip_y) / 3

        torso_dx = abs(shoulder_x - hip_x)
        torso_dy = abs(shoulder_y - hip_y)
        torso_angle = math.degrees(math.atan2(torso_dx, max(torso_dy, 1e-6)))
        horizontal = torso_angle >= HORIZONTAL_ANGLE_DEGREES

        upper_body_dx = abs(nose.x - shoulder_x)
        upper_body_dy = abs(nose.y - shoulder_y)
        upper_body_angle = math.degrees(
            math.atan2(upper_body_dx, max(upper_body_dy, 1e-6))
        )
        upper_body_horizontal = (
            upper_body_angle >= UPPER_BODY_HORIZONTAL_ANGLE_DEGREES
        )
        head_shoulders_visible = (
            getattr(nose, "visibility", 1.0) >= MIN_UPPER_BODY_VISIBILITY
            and getattr(left_shoulder, "visibility", 1.0) >= MIN_UPPER_BODY_VISIBILITY
            and getattr(right_shoulder, "visibility", 1.0) >= MIN_UPPER_BODY_VISIBILITY
        )
        reliable_upper_body_visible = (
            getattr(nose, "visibility", 1.0) >= MIN_RELIABLE_UPPER_BODY_VISIBILITY
            and getattr(left_shoulder, "visibility", 1.0) >= MIN_RELIABLE_UPPER_BODY_VISIBILITY
            and getattr(right_shoulder, "visibility", 1.0) >= MIN_RELIABLE_UPPER_BODY_VISIBILITY
        )

        low_position = (
            hip_y >= LOW_HIP_Y
            or shoulder_y >= LOW_SHOULDER_Y
            or body_center_y >= BODY_CENTER_LOW_Y
        )
        upper_body_low = (
            nose.y >= LOW_HEAD_Y
            and shoulder_y >= LOW_SHOULDER_Y
        )

        self.center_history.append((now, body_center_y))
        while self.center_history and now - self.center_history[0][0] > DROP_WINDOW_SECONDS:
            self.center_history.popleft()

        previous_center_y = min(
            (center_y for _, center_y in self.center_history),
            default=body_center_y,
        )
        center_drop = max(0.0, body_center_y - previous_center_y)
        sudden_drop = center_drop >= MIN_CENTER_DROP

        return {
            "torso_angle": round(torso_angle, 1),
            "center_drop": round(center_drop, 3),
            "upper_body_angle": round(upper_body_angle, 1),
            "body_center_y": round(body_center_y, 3),
            "head_y": round(nose.y, 3),
            "shoulder_y": round(shoulder_y, 3),
            "hip_y": round(hip_y, 3),
            "horizontal": horizontal,
            "upper_body_horizontal": upper_body_horizontal,
            "head_shoulders_visible": head_shoulders_visible,
            "reliable_upper_body_visible": reliable_upper_body_visible,
            "upper_body_low": upper_body_low,
            "low_position": low_position,
            "sudden_drop": sudden_drop,
            "rapid_drop": sudden_drop,
        }

    def _confidence(self) -> float:
        score = 0.0
        if self.last_metrics["horizontal"]:
            score += 0.4
        if self.last_metrics["upper_body_horizontal"]:
            score += 0.4
        if self.last_metrics["head_shoulders_visible"]:
            score += 0.2
        if self.last_metrics["reliable_upper_body_visible"]:
            score += 0.15
        if self.last_metrics["upper_body_low"]:
            score += 0.15
        if self.last_metrics["low_position"]:
            score += 0.35
        if self.last_metrics["sudden_drop"]:
            score += 0.25
        if self.status == "FALL_CONFIRMED":
            score = max(score, 0.8)
        return round(min(score, 1.0), 2)

    @staticmethod
    def _is_visible(landmark) -> bool:
        # MediaPipe NormalizedLandmark 는 visibility/presence 를 제공한다.
        # 속성이 없으면(다른 입력 형식) 보이는 것으로 간주한다.
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            return True
        return visibility >= MIN_LANDMARK_VISIBILITY

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "torso_angle": 0.0,
            "center_drop": 0.0,
            "upper_body_angle": 0.0,
            "body_center_y": 0.0,
            "head_y": 0.0,
            "shoulder_y": 0.0,
            "hip_y": 0.0,
            "horizontal": False,
            "upper_body_horizontal": False,
            "head_shoulders_visible": False,
            "reliable_upper_body_visible": False,
            "upper_body_low": False,
            "low_position": False,
            "sudden_drop": False,
            "rapid_drop": False,
        }
