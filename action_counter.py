import math
import time


TARGET_COUNT = None
TIME_LIMIT_SECONDS = 60
COUNT_COOLDOWN_SECONDS = 0.8

ACTION_NAMES = ("LEFT_HAND_UP", "RIGHT_HAND_UP", "ARMS_OPEN", "CLAP")

ACTION_TIMELINE = [
    {"start": 0.0, "end": 5.0, "action": "LEFT_HAND_UP", "score": 10},
    {"start": 5.0, "end": 10.0, "action": "RIGHT_HAND_UP", "score": 10},
    {"start": 10.0, "end": 15.0, "action": "ARMS_OPEN", "score": 10},
    {"start": 15.0, "end": 20.0, "action": "CLAP", "score": 10},
]

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16

MIN_RAISE_MARGIN_Y = 0.08
ARMS_OPEN_Y_TOLERANCE = 0.28
MIN_SHOULDER_WIDTH = 0.10


def _visibility_ok(landmarks, indexes: tuple[int, ...], threshold: float = 0.35) -> bool:
    for index in indexes:
        visibility = getattr(landmarks[index], "visibility", 1.0)
        if visibility < threshold:
            return False
    return True


def _shoulder_width(landmarks) -> float:
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    width = abs(right_shoulder.x - left_shoulder.x)
    return max(width, MIN_SHOULDER_WIDTH)


def is_left_hand_up(landmarks) -> bool:
    if not _visibility_ok(landmarks, (LEFT_SHOULDER, LEFT_WRIST)):
        return False
    left_shoulder = landmarks[LEFT_SHOULDER]
    left_wrist = landmarks[LEFT_WRIST]
    return left_wrist.y < left_shoulder.y - MIN_RAISE_MARGIN_Y


def is_right_hand_up(landmarks) -> bool:
    if not _visibility_ok(landmarks, (RIGHT_SHOULDER, RIGHT_WRIST)):
        return False
    right_shoulder = landmarks[RIGHT_SHOULDER]
    right_wrist = landmarks[RIGHT_WRIST]
    return right_wrist.y < right_shoulder.y - MIN_RAISE_MARGIN_Y


def is_arms_open(landmarks) -> bool:
    if not _visibility_ok(
        landmarks,
        (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST),
    ):
        return False

    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]
    left_elbow = landmarks[LEFT_ELBOW]
    right_elbow = landmarks[RIGHT_ELBOW]

    if not all(0 <= point.x <= 1 and 0 <= point.y <= 1
               for point in (left_elbow, right_elbow, left_wrist, right_wrist)):
        return False

    shoulder_width = _shoulder_width(landmarks)
    wrists_width = abs(right_wrist.x - left_wrist.x)
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

    direction = 1 if left_shoulder.x > right_shoulder.x else -1
    left_open = (left_wrist.x - left_shoulder.x) * direction > shoulder_width * 0.20
    right_open = (right_wrist.x - right_shoulder.x) * direction < -shoulder_width * 0.20
    wrists_near_shoulder_height = (
        abs(left_wrist.y - shoulder_y) <= ARMS_OPEN_Y_TOLERANCE
        and abs(right_wrist.y - shoulder_y) <= ARMS_OPEN_Y_TOLERANCE
    )
    elbows_near_shoulder_height = (
        abs(left_elbow.y - left_shoulder.y) <= shoulder_width * 0.5
        and abs(right_elbow.y - right_shoulder.y) <= shoulder_width * 0.5
    )
    wide_enough = wrists_width >= shoulder_width * 1.45

    return left_open and right_open and wrists_near_shoulder_height and elbows_near_shoulder_height and wide_enough


def is_clap(landmarks) -> bool:
    if not _visibility_ok(landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST)):
        return False

    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]
    shoulder_width = _shoulder_width(landmarks)

    wrist_distance = math.hypot(
        left_wrist.x - right_wrist.x,
        left_wrist.y - right_wrist.y,
    )
    wrist_y_distance = abs(left_wrist.y - right_wrist.y)

    return (
        wrist_distance <= shoulder_width * 0.65
        and wrist_y_distance <= shoulder_width * 0.55
    )


def detect_actions(landmarks) -> dict[str, bool]:
    if not landmarks or len(landmarks) <= RIGHT_WRIST:
        return {action: False for action in ACTION_NAMES}

    return {
        "LEFT_HAND_UP": is_left_hand_up(landmarks),
        "RIGHT_HAND_UP": is_right_hand_up(landmarks),
        "ARMS_OPEN": is_arms_open(landmarks),
        "CLAP": is_clap(landmarks),
    }


class ClapDetector:
    """Emit one event when the visible hands move close together."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.contact = False
        self.last_seen = None

    def update(self, landmarks, timestamp: float) -> bool:
        if (not landmarks or len(landmarks) <= RIGHT_WRIST
                or not _visibility_ok(landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST))):
            if self.last_seen is not None and timestamp - self.last_seen > 1.0:
                self.reset()
            return False

        left, right = landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST]
        if not all(0 <= point.x <= 1 and 0 <= point.y <= 1 for point in (left, right)):
            if self.last_seen is not None and timestamp - self.last_seen > 1.0:
                self.reset()
            return False

        if self.last_seen is not None and (timestamp <= self.last_seen or timestamp - self.last_seen > 1.0):
            self.reset()
        self.last_seen = timestamp

        distance = math.hypot(left.x - right.x, left.y - right.y) / _shoulder_width(landmarks)
        if distance >= 0.9:
            self.contact = False
        elif not self.contact and is_clap(landmarks):
            self.contact = True
            return True
        return False


class ActionCounter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> dict:
        self.counts = {action: 0 for action in ACTION_NAMES}
        self.action_active = {action: False for action in ACTION_NAMES}
        self.action_armed = {action: False for action in ACTION_NAMES}
        self.last_count_time = {action: 0.0 for action in ACTION_NAMES}
        self.scored_segments = set()
        self.total_pose_score = 0
        self.start_time = time.monotonic()
        return self.get_state(current_actions=[])

    def update(
        self,
        detections: dict[str, bool],
        elapsed_time: float | None = None,
    ) -> dict:
        current_actions = [action for action, detected in detections.items() if detected]
        detected_action = self._select_detected_action(current_actions, elapsed_time)
        expected_segment = self.get_expected_segment(elapsed_time)
        expected_action = expected_segment["action"] if expected_segment else None
        action_correct = bool(detected_action and expected_action == detected_action)
        action_score = 0
        now = time.monotonic()

        for action in ACTION_NAMES:
            detected = detections.get(action, False)
            if detected and self.action_armed[action] and not self.action_active[action]:
                can_count = now - self.last_count_time[action] >= COUNT_COOLDOWN_SECONDS
                if can_count:
                    self.counts[action] += 1
                    self.last_count_time[action] = now
                self.action_active[action] = True
            elif not detected:
                self.action_active[action] = False
                self.action_armed[action] = True

        if action_correct and expected_segment:
            segment_key = (
                expected_segment["start"],
                expected_segment["end"],
                expected_segment["action"],
            )
            if segment_key not in self.scored_segments:
                action_score = int(expected_segment["score"])
                self.total_pose_score += action_score
                self.scored_segments.add(segment_key)

        return self.get_state(
            current_actions=current_actions,
            detected_action=detected_action,
            expected_action=expected_action,
            action_correct=action_correct,
            action_score=action_score,
            elapsed_time=elapsed_time,
        )

    def mark_no_pose(self, elapsed_time: float | None = None) -> dict:
        for action in ACTION_NAMES:
            self.action_active[action] = False
        expected_segment = self.get_expected_segment(elapsed_time)
        return self.get_state(
            current_actions=[],
            detected_action=None,
            expected_action=expected_segment["action"] if expected_segment else None,
            action_correct=False,
            action_score=0,
            elapsed_time=elapsed_time,
        )

    def get_expected_segment(self, elapsed_time: float | None = None) -> dict | None:
        elapsed = self._elapsed_time(elapsed_time)
        for segment in ACTION_TIMELINE:
            if segment["start"] <= elapsed < segment["end"]:
                return segment
        return None

    def get_state(
        self,
        current_actions: list[str] | None = None,
        detected_action: str | None = None,
        expected_action: str | None = None,
        action_correct: bool = False,
        action_score: int = 0,
        elapsed_time: float | None = None,
    ) -> dict:
        elapsed = self._elapsed_time(elapsed_time)
        remaining_time = max(0, int(TIME_LIMIT_SECONDS - elapsed))
        if expected_action is None:
            expected_segment = self.get_expected_segment(elapsed_time)
            expected_action = expected_segment["action"] if expected_segment else None

        return {
            "current_actions": current_actions or [],
            "detected_action": detected_action,
            "expected_action": expected_action,
            "action_correct": action_correct,
            "action_score": action_score,
            "total_pose_score": self.total_pose_score,
            "counts": dict(self.counts),
            "target_count": TARGET_COUNT,
            "remaining_time": remaining_time,
            "clear": False,
            "game_clear": False,
            "time_over": remaining_time <= 0,
            "action_timeline": list(ACTION_TIMELINE),
        }

    def _select_detected_action(
        self,
        current_actions: list[str],
        elapsed_time: float | None,
    ) -> str | None:
        if not current_actions:
            return None

        expected_segment = self.get_expected_segment(elapsed_time)
        if expected_segment and expected_segment["action"] in current_actions:
            return expected_segment["action"]

        for action in ACTION_NAMES:
            if action in current_actions:
                return action
        return current_actions[0]

    def _elapsed_time(self, elapsed_time: float | None) -> float:
        if elapsed_time is not None:
            return max(0.0, elapsed_time)
        return max(0.0, time.monotonic() - self.start_time)
