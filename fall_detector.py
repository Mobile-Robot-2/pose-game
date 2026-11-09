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
# Temi sends at least 0.5 s apart. Allow 0.25 s of jitter without bridging
# a missed frame at that cadence or confirming from a single interval.
MAX_OBSERVATION_GAP_SECONDS = 0.75
SUSPECTED_MISSING_FRAME_GRACE_SECONDS = 1.0

# At the nominal 0.5 s cadence, require the drop within one interval plus
# jitter instead of accumulating gradual descent across two intervals.
DROP_WINDOW_SECONDS = 0.75
HEAD_SHOULDER_DROP_WINDOW_SECONDS = 0.85
MIN_CENTER_DROP = 0.12
MIN_HEAD_DROP = 0.20
MIN_SHOULDER_DROP = 0.15
HORIZONTAL_ANGLE_DEGREES = 45.0
STRONG_HORIZONTAL_ANGLE_DEGREES = 75.0
UPPER_BODY_HORIZONTAL_ANGLE_DEGREES = 50.0
UPPER_BODY_LOW_HOLD_ANGLE_DEGREES = 40.0
LOW_HEAD_Y = 0.36
LOW_HIP_Y = 0.55
LOW_SHOULDER_Y = 0.42
BODY_CENTER_LOW_Y = 0.50
MIN_UPPER_BODY_VISIBILITY = 0.50
MIN_RELIABLE_UPPER_BODY_VISIBILITY = 0.55

# [진단] 전방/후방 낙상 방향 판별 가능성 검증용 로깅.
# FSM 판정에는 전혀 관여하지 않고, FALL_CONFIRMED 순간에 후보 신호만 한 줄 출력한다.
# 낙하 직전~확정까지를 담도록 관찰 윈도우보다 넉넉히 잡는다.
DIRECTION_HISTORY_SECONDS = 2.0


class FallDetector:
    """Stateful fall-suspicion detector using normalized pose landmarks."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> dict:
        self.status = "NORMAL"
        self.low_pose_since = None
        self.recovery_since = None
        self.recovered_since = None
        self.last_evidence_time = 0.0
        self.last_event = None
        self._new_event = False
        self.upright_center_history = deque()
        self.upright_head_shoulder_history = deque()
        self.last_observation_time = None
        self.missing_pose_pending = False
        self.last_metrics = self._empty_metrics()
        self._signal_history = deque()  # [진단] 방향 판별용 최근 프레임 신호
        return self.get_state()

    def update(self, landmarks, *, diagnostics: dict | None = None, timestamp: float | None = None) -> dict:
        now = time.monotonic() if timestamp is None else timestamp
        observation_gap = (
            now - self.last_observation_time if self.last_observation_time is not None else 0.0
        )
        if diagnostics is not None and self.last_observation_time is not None:
            diagnostics["observation_gap_ms"] = observation_gap * 1000
        max_gap = (
            SUSPECTED_MISSING_FRAME_GRACE_SECONDS
            if self.missing_pose_pending else MAX_OBSERVATION_GAP_SECONDS
        )
        if observation_gap > max_gap:
            self._break_observation()
            if diagnostics is not None:
                self._record_reset(diagnostics, "observation_gap")

        history_size = len(self.upright_center_history) if diagnostics is not None else 0
        metrics, upright_pose = self._calculate_metrics(landmarks, now)
        if not metrics["reliable_upper_body_visible"]:
            if diagnostics is not None:
                self._record_reset(diagnostics, "low_landmark_visibility")
            return self.update_no_pose(allow_grace=False, timestamp=now)

        self.last_observation_time = now
        self.last_metrics = metrics
        self._record_direction_sample(landmarks, metrics, now)  # [진단]
        self._new_event = False

        in_frame = all(
            0 <= landmarks[index].x <= 1 and 0 <= landmarks[index].y <= 1
            for index in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
        )
        head_shoulder_drop = self._head_shoulders_dropped(landmarks, now, upright_pose, in_frame)
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
        low_fall_pose = fallen_pose or obvious_lying_pose
        if self.status == "FALL_SUSPECTED":
            low_fall_pose = low_fall_pose or (
                metrics["upper_body_low"]
                and (
                    metrics["upper_body_angle"] >= UPPER_BODY_LOW_HOLD_ANGLE_DEGREES
                    or (in_frame and metrics["body_center_y"] >= BODY_CENTER_LOW_Y)
                )
            )
        if self.missing_pose_pending:
            self.missing_pose_pending = False
            resumed = in_frame and low_fall_pose and not upright_pose
            if resumed:
                # Neither end of the unobserved interval adds time to the hold.
                if self.low_pose_since is not None:
                    self.low_pose_since += observation_gap
            else:
                self.status = "NORMAL"
                self.low_pose_since = None
            if diagnostics is not None:
                diagnostics["pose_gap_grace"] = "resumed" if resumed else "rejected"
        angle_fall_motion = (
            metrics["sudden_drop"]
            and (metrics["horizontal"] or metrics["upper_body_horizontal"])
        )
        fall_motion = angle_fall_motion or head_shoulder_drop
        if diagnostics is not None:
            diagnostics.update(
                upright_pose=upright_pose,
                upright_history_count=len(self.upright_center_history),
                upright_samples_expired=(
                    history_size + int(upright_pose) - len(self.upright_center_history)
                ),
                low_fall_pose=low_fall_pose,
                fall_motion=fall_motion,
                angle_fall_motion=angle_fall_motion,
                head_shoulder_drop=head_shoulder_drop,
            )

        if self.status in ("NORMAL", "RECOVERED"):
            if fall_motion:
                self.status = "FALL_SUSPECTED"
                self.last_evidence_time = now
                self.low_pose_since = now if low_fall_pose else None
                self.recovered_since = None
            elif (
                self.recovered_since is not None
                and now - self.recovered_since >= RECOVERED_DISPLAY_SECONDS
            ):
                self.status = "NORMAL"
                self.recovered_since = None

        elif self.status == "FALL_SUSPECTED":
            if fall_motion or low_fall_pose:
                self.last_evidence_time = now
            if low_fall_pose:
                if self.low_pose_since is None:
                    self.low_pose_since = now
                elif now - self.low_pose_since >= FALL_CONFIRM_SECONDS:
                    self.status = "FALL_CONFIRMED"
                    self.recovery_since = None
                    self._create_event()
                    self._log_fall_direction(now)  # [진단]
            else:
                self.low_pose_since = None
                if upright_pose or now - self.last_evidence_time > EVIDENCE_GRACE_SECONDS:
                    self.status = "NORMAL"

        elif self.status == "FALL_CONFIRMED":
            if upright_pose:
                if self.recovery_since is None:
                    self.recovery_since = now
                elif now - self.recovery_since >= RECOVERY_CONFIRM_SECONDS:
                    self.status = "RECOVERED"
                    self.recovered_since = now
                    if self.last_event is not None:
                        self.last_event["status"] = "CLOSED"
            else:
                self.recovery_since = None

        if diagnostics is not None:
            diagnostics["low_pose_ms"] = (
                (now - self.low_pose_since) * 1000
                if low_fall_pose and self.low_pose_since is not None
                else None
            )
        return self.get_state()

    def update_no_pose(
        self, *, diagnostics: dict | None = None, allow_grace: bool = True,
        timestamp: float | None = None,
    ) -> dict:
        now = time.monotonic() if timestamp is None else timestamp
        if (
            allow_grace
            and self.status == "FALL_SUSPECTED"
            and not self.missing_pose_pending
            and self.last_observation_time is not None
            and now - self.last_observation_time <= SUSPECTED_MISSING_FRAME_GRACE_SECONDS
        ):
            self.missing_pose_pending = True
            self.upright_center_history.clear()
            self.upright_head_shoulder_history.clear()
            if diagnostics is not None:
                diagnostics["pose_gap_grace"] = "waiting"
        else:
            if diagnostics is not None:
                self._record_reset(diagnostics, "pose_not_detected")
            self._break_observation()
        self.last_metrics = self._empty_metrics()
        self._new_event = False
        return self.get_state()

    @staticmethod
    def _record_reset(diagnostics: dict, reason: str) -> None:
        previous = diagnostics.get("observation_reset_reason")
        diagnostics["observation_reset_reason"] = f"{previous};{reason}" if previous else reason
        diagnostics["upright_history_count"] = 0
        diagnostics.update(
            upright_pose=False, low_fall_pose=False, fall_motion=False,
            angle_fall_motion=False, head_shoulder_drop=False,
        )

    def _break_observation(self) -> None:
        self.upright_center_history.clear()
        self.upright_head_shoulder_history.clear()
        self._signal_history.clear()  # [진단] 관찰이 끊기면 방향 기준선도 초기화
        self.last_observation_time = None
        self.missing_pose_pending = False
        self.low_pose_since = None
        self.recovery_since = None
        if self.status == "FALL_SUSPECTED":
            self.status = "NORMAL"

    def _head_shoulders_dropped(self, landmarks, now, upright_pose, in_frame) -> bool:
        history = self.upright_head_shoulder_history
        if not in_frame:
            history.clear()
            return False
        while history and now - history[0][0] > HEAD_SHOULDER_DROP_WINDOW_SECONDS:
            history.popleft()

        nose_y = landmarks[NOSE].y
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
        if upright_pose:
            history.append((now, nose_y, shoulder_y))
            return False
        return any(
            nose_y - previous_nose >= MIN_HEAD_DROP
            and shoulder_y - previous_shoulder >= MIN_SHOULDER_DROP
            for _, previous_nose, previous_shoulder in history
        )

    def _record_direction_sample(self, landmarks, metrics: dict, now: float) -> None:
        """[진단] 방향 판별 후보 신호(세로 낙폭 + 깊이 z)를 최근 윈도우에 쌓는다.

        FSM 판정과 완전히 분리된 기록 전용 로직이다. reliable_upper_body_visible
        프레임에서만 호출되므로 상체(코·어깨·골반) 좌표는 신뢰할 수 있다.
        MediaPipe 의 z 는 골반 중심을 원점으로 한 상대 깊이(작을수록 카메라에 가까움).
        """
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        nose = landmarks[NOSE]
        self._signal_history.append({
            "t": now,
            "nose_y": metrics["head_y"],
            "shoulder_y": metrics["shoulder_y"],
            "hip_y": metrics["hip_y"],
            "nose_z": float(getattr(nose, "z", 0.0)),
            "shoulder_z": float((getattr(left_shoulder, "z", 0.0)
                                 + getattr(right_shoulder, "z", 0.0)) / 2),
            "hip_z": float((getattr(left_hip, "z", 0.0)
                            + getattr(right_hip, "z", 0.0)) / 2),
        })
        while (
            self._signal_history
            and now - self._signal_history[0]["t"] > DIRECTION_HISTORY_SECONDS
        ):
            self._signal_history.popleft()

    def _log_fall_direction(self, now: float) -> None:
        """[진단] FALL_CONFIRMED 순간, 전방/후방 판별 후보 신호를 한 줄로 출력한다.

        label 은 ? 로 두니, 전방낙상 'F' / 후방낙상 'B' 로 바꿔 적어 붙여넣으면 된다.
        부호 규약: 세로 낙폭(head_drop/hip_drop)은 아래로 떨어질수록 +,
        dz_*(깊이 변화)는 카메라로 다가올수록 - / 멀어질수록 +.
        """
        history = self._signal_history
        if len(history) < 2:
            print(f"fall_dir_debug label=? n={len(history)} (표본부족: 기준선 없음)")
            return
        base, current = history[0], history[-1]
        span = current["t"] - base["t"]
        head_drop = current["nose_y"] - base["nose_y"]
        hip_drop = current["hip_y"] - base["hip_y"]
        denom = hip_drop if abs(hip_drop) > 1e-3 else (1e-3 if hip_drop >= 0 else -1e-3)
        head_hip_ratio = head_drop / denom
        dz_nose = current["nose_z"] - base["nose_z"]
        dz_shoulder = current["shoulder_z"] - base["shoulder_z"]
        print(
            "fall_dir_debug label=? "
            f"n={len(history)} span={span:.2f}s "
            f"head_drop={head_drop:+.3f} hip_drop={hip_drop:+.3f} "
            f"head_hip_ratio={head_hip_ratio:+.2f} "
            f"dz_nose={dz_nose:+.3f} dz_shoulder={dz_shoulder:+.3f} "
            f"nose_z0={base['nose_z']:+.3f} nose_z1={current['nose_z']:+.3f} "
            f"sh_z0={base['shoulder_z']:+.3f} sh_z1={current['shoulder_z']:+.3f}"
        )

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

    def _calculate_metrics(self, landmarks, now: float) -> tuple[dict, bool]:
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
            return self._empty_metrics(), False

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

        # A standing person's hips can be below LOW_HIP_Y in the camera view.
        # Require both upper-body landmarks to be high and vertically ordered.
        upright_pose = (
            reliable_upper_body_visible
            and not horizontal
            and not upper_body_horizontal
            and nose.y < LOW_HEAD_Y
            and shoulder_y < LOW_SHOULDER_Y
            and nose.y < shoulder_y < hip_y
        )
        if upright_pose:
            self.upright_center_history.append((now, body_center_y))
        while (
            self.upright_center_history
            and now - self.upright_center_history[0][0] > DROP_WINDOW_SECONDS
        ):
            self.upright_center_history.popleft()

        previous_center_y = min(
            (center_y for _, center_y in self.upright_center_history),
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
        }, upright_pose

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
