"""Single-device, operator-bounded rhythm trials for the test server."""
import csv
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

FIELDS = (
    "trial_id", "participant_id", "actual_action", "trial_number", "session_id",
    "started_at", "ended_at", "frames", "pose_frames", "no_action_frames",
    "conflict_frames", "session_mismatch_frames", "connection_gaps", "max_frame_gap_ms",
    "detected_actions", "prediction_status", "detected_action", "frames_file",
)
FRAME_FIELDS = ("elapsed_ms", "pose_detected", "current_actions", "prediction_status", "detected_action")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class RhythmTrials:
    def __init__(self, path: Path, *, clock=time.monotonic):
        self.path = path
        self.current = None
        self.clock = clock
        self.last_frame_at = None
        self.started_at = None
        self.frames = []

    def start(self, participant_id, actual_action, trial_number, session_id):
        if self.current is not None:
            raise ValueError("진행 중인 리듬게임 시행을 먼저 종료하세요.")
        if not session_id:
            raise ValueError("Temi에서 리듬게임 테스트 화면을 먼저 여세요.")
        self.current = dict(trial_id=uuid4().hex, participant_id=participant_id,
                            actual_action=actual_action, trial_number=trial_number,
                            session_id=session_id, started_at=utc_now(), frames=0,
                            pose_frames=0, no_action_frames=0, conflict_frames=0,
                            session_mismatch_frames=0, connection_gaps=0,
                            max_frame_gap_ms=0, detected_actions=Counter())
        self.last_frame_at = self.clock()
        self.started_at = self.last_frame_at
        self.frames = []
        return self.snapshot()

    def record(self, session_id, result):
        trial = self.current
        if trial is None:
            return
        if session_id != trial["session_id"]:
            trial["session_mismatch_frames"] += 1
            return
        self._record_gap()
        self.frames.append(dict(elapsed_ms=max(0, round((self.last_frame_at - self.started_at) * 1000)),
                                pose_detected=result["detected"],
                                current_actions="|".join(result.get("current_actions", [])),
                                prediction_status=result["prediction_status"],
                                detected_action=result["detected_action"] or ""))
        trial["frames"] += 1
        if result["detected"]:
            trial["pose_frames"] += 1
        status = result["prediction_status"]
        if status == "no_action":
            trial["no_action_frames"] += 1
        elif status == "conflict":
            trial["conflict_frames"] += 1
        elif status == "detected":
            trial["detected_actions"][result["detected_action"]] += 1

    def snapshot(self):
        if self.current is None:
            return None
        return {**self.current, "detected_actions": dict(self.current["detected_actions"])}

    def _record_gap(self):
        now = self.clock()
        gap_ms = max(0, round((now - self.last_frame_at) * 1000))
        self.current["max_frame_gap_ms"] = max(self.current["max_frame_gap_ms"], gap_ms)
        if gap_ms > 2000:
            self.current["connection_gaps"] += 1
        self.last_frame_at = now

    def finish(self, trial_id):
        if self.current is None or trial_id != self.current["trial_id"]:
            raise ValueError("진행 중인 시행 ID를 확인하세요.")
        self._record_gap()
        row = self.snapshot()
        actions = row["detected_actions"]
        if row["session_mismatch_frames"] or row["connection_gaps"] or not row["frames"]:
            status, prediction = "invalid", ""
        elif not row["pose_frames"]:
            status, prediction = "pose_not_detected", ""
        elif row["conflict_frames"] or len(actions) > 1:
            status, prediction = "conflict", ""
        elif actions:
            status, prediction = "detected", next(iter(actions))
        else:
            status, prediction = "no_action", ""
        row.update(ended_at=utc_now(), prediction_status=status, detected_action=prediction,
                   detected_actions="|".join(f"{key}:{value}" for key, value in sorted(actions.items())))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame_dir = self.path.parent / "rhythm_frames"
        frame_dir.mkdir(exist_ok=True)
        frame_file = frame_dir / (trial_id + ".csv")
        temporary = frame_file.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FRAME_FIELDS)
            writer.writeheader()
            writer.writerows(self.frames)
        temporary.replace(frame_file)
        row["frames_file"] = str(Path("rhythm_frames") / frame_file.name)
        existed = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if not existed:
                writer.writeheader()
            writer.writerow(row)
        self.current = None
        self.frames = []
        return row
