"""Persistent single-device fall experiments, coordinated with the frame lock."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sqlite3
from threading import Event, Thread
import time
from uuid import uuid4

from fall_debug import FallDebugRecorder


FALL_ACTIONS = {"front": "전방", "left": "좌측", "right": "우측", "back": "후방"}
NONFALL_ACTIONS = {"sit": "의자에 앉기", "lie": "의도적으로 눕기",
                   "pick_up": "물건 줍기", "squat": "쪼그려 앉기"}
SUMMARY_FIELDS = (
    "experiment_id", "participant_id", "experiment_type", "trial_id", "trial_number",
    "set_number", "action", "action_label", "status", "started_at", "ended_at",
    "end_reason", "observation_seconds", "elapsed_seconds", "fall_detected",
    "first_confirmed_at", "first_confirmation_seconds", "frame_count", "valid_pose_frames",
    "unknown_frames", "session_id", "fall_confirmation_policy", "session_mismatch_frames", "connection_gaps",
    "processing_errors",
    "max_frame_gap_seconds", "actual_action", "actual_fall", "valid", "notes",
    "debug_directory", "debug_dropped", "debug_errors",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        # Free-form operator input must remain text when opened in Excel.
        writer.writerow({
            key: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value
            for key, value in row.items() if key in SUMMARY_FIELDS
        })
    return "\ufeff" + output.getvalue()


class ExperimentConflict(ValueError):
    pass


class ExperimentManager:
    def __init__(self, root, lock, *, save_frames=False, clock=time.monotonic):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = lock
        self.clock = clock
        self.save_frames = save_frames
        self.connection = sqlite3.connect(self.root / "experiments.sqlite3", check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self.runs = {
            run_id: json.loads(data) for run_id, data in self.connection.execute("SELECT id, data FROM runs")
        }
        self.current_id = max(self.runs, key=lambda key: self.runs[key]["created_at"], default=None)
        self.recorder = None
        self.started_clock = None
        self.last_trial_frame = None
        self.last_received = None
        self.latest = {}
        self.storage_error = None
        self._stop = Event()
        # A process restart cannot establish continuity with a former live trial.
        for run in self.runs.values():
            for trial in run["trials"]:
                if trial["status"] == "running":
                    trial.update(status="interrupted", ended_at=utc_now(), end_reason="server_restart",
                                 valid=False, notes="서버 재시작으로 중단됨")
            self._save(run, export=True)
        self._timer = None

    def start_monitor(self):
        self._timer = Thread(target=self._monitor, name="experiment-deadline", daemon=True)
        self._timer.start()

    def _monitor(self):
        while not self._stop.wait(0.25):
            with self.lock:
                try:
                    self.tick()
                except Exception as exc:
                    self.storage_error = str(exc)

    def _save(self, run, *, export=False):
        try:
            with self.connection:
                self.connection.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)",
                                        (run["id"], json.dumps(run, ensure_ascii=False)))
        except sqlite3.Error as exc:
            self.storage_error = str(exc)
            raise
        if export:
            directory = self.root / run["participant_id"] / run["id"]
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "trials.csv"
            temporary = directory / "trials.csv.tmp"
            try:
                temporary.write_text(csv_text(run["trials"]), encoding="utf-8")
                temporary.replace(target)
                self.storage_error = None
            except OSError as exc:
                # SQLite remains authoritative, including when Excel locks the CSV.
                self.storage_error = str(exc)

    def _run(self, run_id=None):
        run = self.runs.get(run_id or self.current_id)
        if run is None:
            raise ExperimentConflict("실험을 먼저 시작하세요.")
        return run

    def _trial(self, run):
        return run["trials"][run["index"]]

    def _require_current(self, run_id, trial_id):
        run = self._run(run_id)
        if run["id"] != self.current_id:
            raise ExperimentConflict("다른 실험이 선택되어 있습니다. 화면을 새로 확인하세요.")
        trial = self._trial(run)
        if trial["trial_id"] != trial_id:
            raise ExperimentConflict("시행이 이미 변경되었습니다. 현재 시행을 확인하세요.")
        return run, trial

    def create(self, run_id, participant_id, experiment_type, actions, sets, observation_seconds):
        with self.lock:
            self.tick()
            config = dict(participant_id=participant_id, experiment_type=experiment_type,
                          actions=actions, sets=sets, observation_seconds=observation_seconds)
            if run_id in self.runs:
                if self.runs[run_id]["config"] != config:
                    raise ExperimentConflict("같은 요청 ID에 다른 설정이 전달됐습니다.")
                return run_id
            if self.current_id and self._run()["status"] == "active":
                raise ExperimentConflict("진행 중인 실험을 완료하거나 실험 종료 버튼을 누르세요.")
            labels = FALL_ACTIONS if experiment_type == "fall" else NONFALL_ACTIONS
            trials = []
            for set_number in range(1, sets + 1):
                for action in actions:
                    trials.append(dict(
                        experiment_id=run_id, participant_id=participant_id,
                        experiment_type=experiment_type, trial_id=uuid4().hex,
                        trial_number=len(trials) + 1, set_number=set_number,
                        action=action, action_label=labels[action], status="ready",
                        started_at=None, ended_at=None, end_reason=None,
                        observation_seconds=observation_seconds, elapsed_seconds=0,
                        fall_detected=False, first_confirmed_at=None, first_confirmation_seconds=None,
                        frame_count=0, valid_pose_frames=0, unknown_frames=0, session_id=None,
                        fall_confirmation_policy=None,
                        processing_errors=0,
                        session_mismatch_frames=0, connection_gaps=0, max_frame_gap_seconds=0,
                        actual_action="", actual_fall=None, valid=None, notes="",
                        debug_directory=None, debug_dropped=0, debug_errors=0,
                    ))
            run = dict(id=run_id, participant_id=participant_id, experiment_type=experiment_type,
                       config=config, created_at=utc_now(), status="active", index=0, trials=trials)
            self.runs[run_id] = run
            self.current_id = run_id
            self._save(run, export=True)
            return run_id

    def start_trial(self, run_id, trial_id, reset_detector, session_id, confirmation_policy="baseline"):
        with self.lock:
            self.tick()
            run, trial = self._require_current(run_id, trial_id)
            if trial["status"] != "ready":
                return  # Retried start must not reset or reopen a completed trial.
            if run["status"] != "active":
                raise ExperimentConflict("종료된 실험입니다.")
            root = self.root / run["participant_id"] / run["id"] / f"trial-{trial['trial_number']:03d}"
            recorder = FallDebugRecorder(root, save_frames=self.save_frames)
            reset_detector()
            self.recorder = recorder
            self.started_clock = self.clock()
            self.last_trial_frame = None
            trial.update(status="running", started_at=utc_now(), session_id=session_id,
                         fall_confirmation_policy=confirmation_policy,
                         debug_directory=str(recorder.directory.relative_to(self.root)))
            self._save(run, export=True)

    def tick(self):
        if not self.current_id:
            return
        run = self._run()
        trial = self._trial(run)
        if trial["status"] == "running" and self.clock() - self.started_clock >= trial["observation_seconds"]:
            self._end(run, trial, "timeout")

    def _end(self, run, trial, reason):
        now = self.clock()
        if self.recorder:
            self.recorder.close()
            trial.update(debug_dropped=self.recorder.dropped_records, debug_errors=self.recorder.write_errors)
            self.recorder = None
        last = self.last_trial_frame if self.last_trial_frame is not None else self.started_clock
        gap = now - last
        trial["max_frame_gap_seconds"] = max(trial["max_frame_gap_seconds"], round(gap, 3))
        if gap > 2:
            trial["connection_gaps"] += 1
        trial.update(status="completed", ended_at=utc_now(), end_reason=reason,
                     elapsed_seconds=round(min(now - self.started_clock, trial["observation_seconds"]), 3))
        self.started_clock = None
        self._save(run, export=True)

    def end_trial(self, run_id, trial_id):
        with self.lock:
            self.tick()
            run, trial = self._require_current(run_id, trial_id)
            if trial["status"] == "running":
                self._end(run, trial, "manual")

    def next_trial(self, run_id, trial_id):
        with self.lock:
            self.tick()
            run = self._run(run_id)
            if run_id != self.current_id:
                raise ExperimentConflict("현재 실험을 확인하세요.")
            if self._trial(run)["trial_id"] != trial_id:
                return  # A duplicated next request cannot skip another trial.
            if self._trial(run)["status"] not in {"completed", "interrupted"}:
                raise ExperimentConflict("시행을 먼저 종료하세요.")
            if run["index"] + 1 < len(run["trials"]):
                run["index"] += 1
            else:
                run["status"] = "completed"
            self._save(run, export=True)

    def finish(self, run_id):
        with self.lock:
            self.tick()
            run = self._run(run_id)
            if run_id != self.current_id:
                raise ExperimentConflict("현재 실험을 확인하세요.")
            if self._trial(run)["status"] == "running":
                self._end(run, self._trial(run), "experiment_closed")
            run["status"] = "closed"
            self._save(run, export=True)

    def annotate(self, run_id, trial_id, actual_action, actual_fall, valid, notes):
        with self.lock:
            run = self._run(run_id)
            trial = next((item for item in run["trials"] if item["trial_id"] == trial_id), None)
            if trial is None or trial["status"] not in {"completed", "interrupted"}:
                raise ExperimentConflict("종료된 시행만 검토할 수 있습니다.")
            if valid is True and (not actual_action.strip() or actual_fall is None):
                raise ExperimentConflict("유효한 시행에는 실제 행동과 정답 라벨을 입력하세요.")
            trial.update(actual_action=actual_action.strip(), actual_fall=actual_fall, valid=valid, notes=notes)
            self._save(run, export=True)

    def select(self, run_id):
        with self.lock:
            self.tick()
            if self.current_id and self._trial(self._run())["status"] == "running":
                raise ExperimentConflict("기록 중에는 실험을 변경할 수 없습니다.")
            self._run(run_id)
            self.current_id = run_id

    def ticket(self, session_id):
        with self.lock:
            self.tick()
            trial = self._trial(self._run()) if self.current_id else None
            return dict(trial_id=trial["trial_id"] if trial and trial["status"] == "running" else None,
                        received=self.clock(), session_id=session_id)

    def is_recording(self):
        return bool(self.current_id and self._trial(self._run())["status"] == "running")

    @contextmanager
    def capture_frame(self, ticket, image_bytes):
        # Called under the shared frame lock, so trial closure cannot race the writer.
        self.tick()
        self.last_received = ticket["received"]
        trial = self._trial(self._run()) if self.current_id else None
        eligible = (
            trial is not None and trial["status"] == "running"
            and ticket["trial_id"] == trial["trial_id"]
            and ticket["received"] >= self.started_clock
        )
        if not eligible:
            yield None
            return
        if trial["session_id"] is None:
            trial["session_id"] = ticket["session_id"]
        if trial["session_id"] != ticket["session_id"]:
            trial["session_mismatch_frames"] += 1
            self._save(self._run())
            yield None
            return
        with self.recorder.capture("/test/fall", ticket["session_id"]) as diagnostics:
            diagnostics.update(
                experiment_id=trial["experiment_id"], participant_id=trial["participant_id"],
                trial_id=trial["trial_id"], trial_number=trial["trial_number"],
                set_number=trial["set_number"], action=trial["action"], experiment_phase="recording",
            )
            if self.save_frames:
                diagnostics["image_bytes"] = image_bytes
            try:
                yield diagnostics
            except Exception:
                trial["processing_errors"] += 1
                self._save(self._run())
                raise

    def record_frame(self, ticket, result, diagnostics):
        self.latest = dict(fall_status=result["fall_status"], pose_detected=result["detected"],
                           session_id=ticket["session_id"])
        if not diagnostics or not diagnostics.get("trial_id"):
            return
        run = self._run()
        trial = self._trial(run)
        now = self.clock()
        previous = self.last_trial_frame if self.last_trial_frame is not None else self.started_clock
        gap = max(0, ticket["received"] - previous)
        trial["max_frame_gap_seconds"] = max(trial["max_frame_gap_seconds"], round(gap, 3))
        if gap > 2:
            trial["connection_gaps"] += 1
        self.last_trial_frame = ticket["received"]
        trial["frame_count"] += 1
        trial["elapsed_seconds"] = round(min(now - self.started_clock, trial["observation_seconds"]), 3)
        if result["detected"] and result["fall_metrics"].get("reliable_upper_body_visible"):
            trial["valid_pose_frames"] += 1
        else:
            trial["unknown_frames"] += 1
        if result.get("fall_detected") and not trial["fall_detected"]:
            trial.update(fall_detected=True, first_confirmed_at=utc_now(),
                         first_confirmation_seconds=round(now - self.started_clock, 3))
        diagnostics["response_fall_status"] = result["fall_status"]
        self._save(run)

    def snapshot(self):
        with self.lock:
            self.tick()
            run = self._run() if self.current_id else None
            trial = self._trial(run) if run else None
            elapsed = self.clock() - self.started_clock if self.started_clock is not None else None
            return dict(
                current=json.loads(json.dumps(run)) if run else None,
                elapsed_seconds=elapsed, latest=dict(self.latest),
                frame_age_seconds=self.clock() - self.last_received if self.last_received is not None else None,
                storage_error=self.storage_error, save_frames=self.save_frames,
                runs=[dict(id=item["id"], participant_id=item["participant_id"],
                           experiment_type=item["experiment_type"], created_at=item["created_at"],
                           status=item["status"]) for item in self.runs.values()],
                summary=self.summary(),
            )

    def summary(self):
        counts = dict(TP=0, FN=0, FP=0, TN=0)
        excluded = 0
        by_action = {}
        for run in self.runs.values():
            for trial in run["trials"]:
                if trial["status"] not in {"completed", "interrupted"}:
                    continue
                key = trial["experiment_type"] + "/" + trial["action"]
                group = by_action.setdefault(key, dict(action_label=trial["action_label"], trials=0,
                                                       confirmed=0, not_confirmed=0, reviewed_valid=0))
                group["trials"] += 1
                group["confirmed" if trial["fall_detected"] else "not_confirmed"] += 1
                if trial["status"] != "completed" or trial["valid"] is not True or trial["actual_fall"] is None or not trial["valid_pose_frames"]:
                    excluded += 1
                    continue
                group["reviewed_valid"] += 1
                key = ("TP" if trial["fall_detected"] else "FN") if trial["actual_fall"] else (
                    "FP" if trial["fall_detected"] else "TN"
                )
                counts[key] += 1
        tp, fn, fp, tn = (counts[k] for k in ("TP", "FN", "FP", "TN"))
        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None
        return dict(
            **counts, excluded=excluded, by_action=by_action,
            precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
            specificity=ratio(tn, tn + fp), f1=ratio(2 * tp, 2 * tp + fp + fn),
        )

    def preview(self, run_id, trial_id):
        with self.lock:
            run = self._run(run_id)
            trial = next((t for t in run["trials"] if t["trial_id"] == trial_id), None)
            if trial is None or not trial["debug_directory"]:
                return None
            return max((self.root / trial["debug_directory"]).glob("frame-*.jpg"), default=None)

    def close(self):
        self._stop.set()
        if self._timer:
            self._timer.join()
        with self.lock:
            if self.current_id and self._trial(self._run())["status"] == "running":
                self._end(self._run(), self._trial(self._run()), "server_shutdown")
            for run in self.runs.values():
                self._save(run, export=True)
            self.connection.close()
