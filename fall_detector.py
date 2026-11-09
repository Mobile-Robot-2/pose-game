from collections import deque
import math
import time
from datetime import datetime


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

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
# [①] 낙상 확정에 요구하는 '상체 전체가 바닥까지 내려감' 기준 (테미 실측 보정).
# 머리·어깨·골반 평균(body_center_y)이 단일 관절보다 분리력이 크다:
#   테미 실측 — 진짜 낙상 ≥0.528, 쭈그림/줍기 오탐 ≤0.512 (간격 ~0.016).
# 낙상은 상체 전체가 내려가지만 쭈그림은 골반만 내려가고 머리·어깨는 덜 내려가 평균에서 갈린다.
CONFIRM_CENTER_Y = 0.52
# [②] 낙상다운 '하강 낙폭' 하한. 테미 실측 — 낙상 center_drop ≥0.258, 쭈그림/줍기 ≤0.214.
# 의심 구간 중 최대 낙폭(peak)이 이 값 미만이면 '천천히 내려온 것'으로 보고 확정하지 않는다.
CONFIRM_MIN_DROP = 0.23
# (아래는 로그/참고용 보조 임계)
CONFIRM_SHOULDER_Y = 0.49
CONFIRM_HIP_Y = 0.64
MIN_UPPER_BODY_VISIBILITY = 0.50
MIN_RELIABLE_UPPER_BODY_VISIBILITY = 0.55


class FallDetector:
    """Stateful fall-suspicion detector using normalized pose landmarks."""

    def __init__(self, *, require_fall_posture: bool = False) -> None:
        self.require_fall_posture = require_fall_posture
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
        self.fall_posture_seen = False
        self._prev_center_y = None      # [측정] 직전 프레임 몸중심 (순간 낙폭용)
        self._last_frame_drop = 0.0     # [측정] 직전→현재 몸중심 변화(+가 하강)
        self._episode_peak_drop = 0.0   # [②] 의심 구간 중 최대 하강 낙폭
        self._episode_base_center = 0.0 # [②] 의심 시작 시점의 기준(선 자세) 몸중심
        self.last_metrics = self._empty_metrics()
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
        # [측정] 직전 프레임 대비 순간 낙폭(하강 속도 proxy). 미검출/관찰끊김 뒤엔 prev 없음.
        cur_center = metrics["body_center_y"]
        self._last_frame_drop = (cur_center - self._prev_center_y) if self._prev_center_y is not None else 0.0
        self._prev_center_y = cur_center
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
        # [①] 테미 실측 보정: 확정은 '상체 전체(머리+어깨+골반 평균)가 충분히 내려감'을 요구한다.
        # body_center_y≥0.52 일 때만 확정. 깊은 쭈그림/줍기는 골반만 내려가고 머리·어깨가
        # 덜 내려가 평균이 0.512를 못 넘어 차단된다. (torso 수평은 요구 안 함 — 테미 낙상도 40°↓)
        low_fall_pose = low_fall_pose and (metrics["body_center_y"] >= CONFIRM_CENTER_Y)
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
        posture_evidence = low_fall_pose and (
            metrics["horizontal"]
            or metrics["upper_body_angle"] >= UPPER_BODY_LOW_HOLD_ANGLE_DEGREES
            or (in_frame and landmarks[NOSE].y >= (
                landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2)
        )
        if self.status == "FALL_SUSPECTED" and posture_evidence:
            self.fall_posture_seen = True
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
                fall_posture_evidence=posture_evidence,
                fall_posture_seen=self.fall_posture_seen or (fall_motion and posture_evidence),
            )

        if self.status in ("NORMAL", "RECOVERED"):
            if fall_motion:
                self.fall_posture_seen = posture_evidence
                self.status = "FALL_SUSPECTED"
                self.last_evidence_time = now
                self.low_pose_since = now if low_fall_pose else None
                self.recovered_since = None
                # [②] 의심 시작 시점의 '선 자세' 몸중심을 기준선으로 고정(만료 안 됨).
                # center_drop = 현재중심 − 최근 upright중심 이므로, 기준선 = 현재중심 − center_drop.
                self._episode_base_center = metrics["body_center_y"] - metrics["center_drop"]
                self._episode_peak_drop = metrics["center_drop"]
                self._log_case("SUSPECT", metrics)  # [0단계]
            elif (
                self.recovered_since is not None
                and now - self.recovered_since >= RECOVERED_DISPLAY_SECONDS
            ):
                self.status = "NORMAL"
                self.recovered_since = None

        elif self.status == "FALL_SUSPECTED":
            # [②] 기준선(선 자세) 대비 현재 총 하강폭의 최대값을 누적 (2단계 붕괴도 포착).
            self._episode_peak_drop = max(
                self._episode_peak_drop,
                metrics["body_center_y"] - self._episode_base_center,
            )
            if fall_motion or low_fall_pose:
                self.last_evidence_time = now
            if low_fall_pose:
                if self.low_pose_since is None:
                    self.low_pose_since = now
                elif (now - self.low_pose_since >= FALL_CONFIRM_SECONDS
                      and (not self.require_fall_posture or self.fall_posture_seen)
                      and self._episode_peak_drop >= CONFIRM_MIN_DROP):  # [②] 낙상다운 급하강 필수
                    self.status = "FALL_CONFIRMED"
                    self.recovery_since = None
                    self._create_event()
                    self._log_case("CONFIRM", metrics)  # [0단계]
            else:
                self.low_pose_since = None
                if upright_pose or now - self.last_evidence_time > EVIDENCE_GRACE_SECONDS:
                    self._log_case("REJECT", metrics)  # [0단계]
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
        self.last_observation_time = None
        self.missing_pose_pending = False
        self.fall_posture_seen = False
        self.low_pose_since = None
        self.recovery_since = None
        self._prev_center_y = None  # [측정] 관찰 끊기면 순간 낙폭 기준도 초기화
        self._episode_peak_drop = 0.0  # [②] 관찰 끊기면 하강 낙폭 누적도 초기화
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

        # [0단계 검증] 다리(무릎·발목)로 '다리 폄(누움) vs 접힘(쭈그림)' 판별 가능성 측정.
        # FSM 판정엔 아직 쓰지 않고 로그로만 확인한다.
        n = len(landmarks)
        left_knee = landmarks[LEFT_KNEE] if n > LEFT_KNEE else None
        right_knee = landmarks[RIGHT_KNEE] if n > RIGHT_KNEE else None
        left_ankle = landmarks[LEFT_ANKLE] if n > LEFT_ANKLE else None
        right_ankle = landmarks[RIGHT_ANKLE] if n > RIGHT_ANKLE else None
        knee_vis = round((self._vis(left_knee) + self._vis(right_knee)) / 2, 2)
        ankle_vis = round((self._vis(left_ankle) + self._vis(right_ankle)) / 2, 2)
        knee_angles = []
        for h, k, a in ((left_hip, left_knee, left_ankle), (right_hip, right_knee, right_ankle)):
            if k is not None and a is not None and self._is_visible(k) and self._is_visible(a):
                ang = self._joint_angle(h, k, a)
                if ang is not None:
                    knee_angles.append(ang)
        legs_visible = len(knee_angles) > 0
        knee_angle = round(sum(knee_angles) / len(knee_angles), 1) if knee_angles else None

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
            "hips_on_floor": CONFIRM_HIP_Y <= hip_y <= 1.0,  # [①] 골반이 바닥까지 내려감
            "shoulder_low": CONFIRM_SHOULDER_Y <= shoulder_y <= 1.0,  # [①] 어깨(상체)까지 내려감
            "sudden_drop": sudden_drop,
            "rapid_drop": sudden_drop,
            "legs_visible": legs_visible,
            "knee_angle": knee_angle,
            "knee_vis": knee_vis,
            "ankle_vis": ankle_vis,
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
    def _vis(landmark) -> float:
        """[0단계] landmark visibility(없거나 None이면 0으로)."""
        if landmark is None:
            return 0.0
        return float(getattr(landmark, "visibility", 1.0))

    @staticmethod
    def _joint_angle(a, b, c):
        """[0단계] b에서의 내각(a-b-c), 도 단위. 다리 폄≈180°, 접힘<120°. 계산 불가 시 None."""
        v1x, v1y = a.x - b.x, a.y - b.y
        v2x, v2y = c.x - b.x, c.y - b.y
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cos = (v1x * v2x + v1y * v2y) / (n1 * n2)
        cos = max(-1.0, min(1.0, cos))
        return math.degrees(math.acos(cos))

    def _log_case(self, tag: str, metrics: dict) -> None:
        """[0단계 검증] 상태 전이 시점의 자세·다리 신호를 한 줄로 출력(판정엔 무관)."""
        print(
            f"fall_dbg[{tag}] "
            f"hip_y={metrics['hip_y']} torso_angle={metrics['torso_angle']} "
            f"horizontal={metrics['horizontal']} upper_body_low={metrics['upper_body_low']} "
            f"head_y={metrics['head_y']} shoulder_y={metrics['shoulder_y']} "
            f"legs_visible={metrics['legs_visible']} knee_angle={metrics['knee_angle']} "
            f"knee_vis={metrics['knee_vis']} ankle_vis={metrics['ankle_vis']} "
            f"body_center_y={metrics['body_center_y']} center_drop={metrics['center_drop']} "
            f"frame_drop={self._last_frame_drop:+.3f} peak_drop={self._episode_peak_drop:.3f} "
            f"sudden_drop={metrics['sudden_drop']}"
        )

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
            "hips_on_floor": False,
            "shoulder_low": False,
            "sudden_drop": False,
            "rapid_drop": False,
            "legs_visible": False,
            "knee_angle": None,
            "knee_vis": 0.0,
            "ankle_vis": 0.0,
        }
