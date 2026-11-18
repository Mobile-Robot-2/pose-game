"""Isolated, single-device pose diagnostics; no Firebase or robot actions."""
import logging
import os
from pathlib import Path
from threading import RLock
from contextlib import nullcontext
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from action_counter import classify_actions
from experiment_manager import (
    ExperimentConflict, ExperimentManager, FALL_ACTIONS, NONFALL_ACTIONS, csv_text,
)
from fall_debug import FallDebugRecorder
from fall_detector import FallDetector
from pose_analyzer import PoseAnalyzer
from rhythm_trials import RhythmTrials

app = FastAPI(title="Pose Test API")
lock = RLock()
sessions = {}
fall_debug: FallDebugRecorder | None = None
fall_confirmation_policy = "baseline"
experiments: ExperimentManager | None = None
BASE_DIR = Path(__file__).resolve().parent
rhythm_trials = RhythmTrials(BASE_DIR / "data" / "experiments" / "rhythm_trials.csv")
app.mount("/experiment/assets", StaticFiles(directory=str(BASE_DIR / "static")), name="experiment-assets")


def analyze(mode, session_id, image_bytes, *, diagnostics=None, experiment_ticket=None):
    with lock:
        manager = experiments if mode == "fall" else None
        ticket = experiment_ticket
        if manager and ticket is None:
            ticket = manager.ticket(session_id)
        context = manager.capture_frame(ticket, image_bytes) if manager else nullcontext()
        with context as trial_diagnostics:
            primary = diagnostics if diagnostics is not None else trial_diagnostics
            if primary is not None and mode == "fall":
                primary["fall_confirmation_policy"] = fall_confirmation_policy
            if manager and manager.is_recording() and trial_diagnostics is None:
                # A delayed preparation frame or foreign session cannot change
                # the active trial's MediaPipe tracker or fall detector.
                result = dict(detected=False, fall_status="UNKNOWN", fall_metrics={}, landmarks=[])
                if primary is not None:
                    primary.update(processing_error="experiment_frame_excluded",
                                   response_fall_status="UNKNOWN")
            else:
                result = _analyze_locked(mode, session_id, image_bytes, diagnostics=primary)
            if mode == "fall":
                result["confirmation_policy"] = fall_confirmation_policy
            if manager:
                if trial_diagnostics is not None and trial_diagnostics is not primary:
                    trial_diagnostics.update({
                        key: value for key, value in primary.items()
                        if key not in {"frame_id", "timestamp", "endpoint", "session_id", "save_frames"}
                    })
                manager.record_frame(ticket, result, trial_diagnostics)
            if diagnostics is not None:
                diagnostics["experiment_phase"] = "recording" if trial_diagnostics else "preparation"
                if trial_diagnostics is not None:
                    for key in ("experiment_id", "participant_id", "trial_id", "trial_number",
                                "set_number", "action"):
                        diagnostics[key] = trial_diagnostics[key]
            return result


def _analyze_locked(mode, session_id, image_bytes, *, diagnostics=None):
    if not image_bytes:
        raise ValueError("Empty image file")
    with lock:
        session = sessions.get(mode)
        if session is None or session["id"] != session_id:
            if session is not None:
                sessions.pop(mode)
                session["analyzer"].close()
            detector = (FallDetector(require_fall_posture=True)
                        if mode == "fall" and fall_confirmation_policy == "posture_experimental"
                        else FallDetector())
            session = {"id": session_id, "analyzer": PoseAnalyzer(), "fall": detector}
            sessions[mode] = session
            if diagnostics is not None:
                diagnostics["observation_reset_reason"] = "new_session"
        analyzer = session["analyzer"]
        if diagnostics is None:
            result = analyzer.analyze_image_bytes(image_bytes)
        else:
            result = analyzer.analyze_image_bytes(image_bytes, diagnostics=diagnostics)
            diagnostics.update(pose_detected=result["detected"], landmarks=result["landmarks"])
        landmarks = [
            {"x": point.x, "y": point.y, "visibility": getattr(point, "visibility", None)}
            for point in (result["landmarks"] or [])
        ]
        if mode == "fall":
            if not result["detected"]:
                state = session["fall"].update_no_pose(diagnostics=diagnostics)
                _record_fall_state(diagnostics, state)
                return {"detected": False, "fall_status": "UNKNOWN", "fall_metrics": {}, "landmarks": landmarks}
            state = session["fall"].update(result["landmarks"], diagnostics=diagnostics)
            _record_fall_state(diagnostics, state)
            if not state["fall_metrics"]["head_shoulders_visible"]:
                return {"detected": False, "fall_status": "UNKNOWN", "fall_metrics": state["fall_metrics"], "landmarks": landmarks}
            return {"detected": True, "fall_status": state["fall_status"],
                    "fall_detected": state["fall_detected"], "fall_metrics": state["fall_metrics"],
                    "landmarks": landmarks}
        actions = [name for name, detected in result.get("raw_detections", result["detections"]).items() if detected]
        prediction, prediction_status = classify_actions(actions, pose_detected=result["detected"])
        response = {"detected": result["detected"], "current_actions": actions,
                    "detected_action": prediction, "prediction_status": prediction_status, "landmarks": landmarks}
        rhythm_trials.record(session_id, response)
        return response


def _record_fall_state(diagnostics, state):
    if diagnostics is not None:
        metrics = state["fall_metrics"]
        if metrics["reliable_upper_body_visible"]:
            diagnostics.update(metrics)
        diagnostics.update(fall_status=state["fall_status"], fall_detected=state["fall_detected"])


async def analyze_upload(mode, image, session_id):
    ticket = experiments.ticket(session_id) if mode == "fall" and experiments else None
    try:
        if mode == "fall" and fall_debug is not None:
            with fall_debug.capture("/test/fall", session_id) as diagnostics:
                image_bytes = await image.read()
                if diagnostics["save_frames"]:
                    diagnostics["image_bytes"] = image_bytes
                result = analyze(mode, session_id, image_bytes, diagnostics=diagnostics, experiment_ticket=ticket)
                diagnostics["response_fall_status"] = result["fall_status"]
                return result
        return analyze(mode, session_id, await image.read(), experiment_ticket=ticket)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/test/fall")
async def test_fall(image: UploadFile = File(...), session_id: str = Form(...)):
    return await analyze_upload("fall", image, session_id)


@app.get("/test/config")
def test_config():
    return {"fall_confirmation_policy": fall_confirmation_policy}


@app.post("/test/rhythm")
async def test_rhythm(image: UploadFile = File(...), session_id: str = Form(...)):
    return await analyze_upload("rhythm", image, session_id)


class RhythmTrialStart(BaseModel):
    participant_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    actual_action: Literal["LEFT_HAND_UP", "RIGHT_HAND_UP", "ARMS_OPEN", "CLAP"]
    trial_number: int = Field(ge=1, le=10000)


class RhythmTrialEnd(BaseModel):
    trial_id: str


@app.post("/experiment/rhythm/start")
def start_rhythm_trial(request: RhythmTrialStart):
    with lock:
        try:
            session = sessions.get("rhythm")
            return rhythm_trials.start(request.participant_id, request.actual_action,
                                       request.trial_number, session["id"] if session else None)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc


@app.get("/experiment/rhythm/current")
def current_rhythm_trial():
    with lock:
        return rhythm_trials.snapshot()


@app.post("/experiment/rhythm/end")
def end_rhythm_trial(request: RhythmTrialEnd):
    with lock:
        try:
            return rhythm_trials.finish(request.trial_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, "시행 저장 실패: " + str(exc)) from exc


@app.get("/experiment/rhythm/results.csv")
def rhythm_results():
    if not rhythm_trials.path.exists():
        raise HTTPException(404, "저장된 리듬게임 시행이 없습니다.")
    return FileResponse(rhythm_trials.path, media_type="text/csv; charset=utf-8",
                        filename="rhythm_trials.csv")


@app.on_event("startup")
def startup():
    global fall_debug, experiments, fall_confirmation_policy
    fall_confirmation_policy = os.environ.get("FALL_CONFIRMATION_POLICY", "baseline").strip().lower()
    if fall_confirmation_policy not in {"baseline", "posture_experimental"}:
        raise ValueError("FALL_CONFIRMATION_POLICY must be baseline or posture_experimental")
    if fall_confirmation_policy == "posture_experimental":
        logging.getLogger(__name__).warning(
            "Experimental fall policy can miss falls; use only for comparison, not caregiver calls")
    mode = os.environ.get("FALL_DEBUG", "").strip().lower()
    experiments = ExperimentManager(BASE_DIR / "data" / "experiments", lock, save_frames=mode == "frames")
    experiments.start_monitor()
    if mode in {"csv", "frames"}:
        try:
            root = Path(__file__).resolve().parent / "data" / "fall_debug"
            fall_debug = FallDebugRecorder(root, save_frames=mode == "frames")
        except OSError:
            logging.getLogger(__name__).exception("Could not start fall diagnostics")


@app.on_event("shutdown")
def shutdown():
    global fall_debug, experiments
    if rhythm_trials.current is not None:
        logging.getLogger(__name__).warning("Unfinished rhythm trial discarded during shutdown")
        rhythm_trials.current = None
        rhythm_trials.frames = []
    if experiments is not None:
        experiments.close()
        experiments = None
    if fall_debug is not None:
        fall_debug.close()
        fall_debug = None
    for session in sessions.values():
        session["analyzer"].close()
    sessions.clear()


class ExperimentConfig(BaseModel):
    experiment_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    participant_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,32}$")
    experiment_type: str = Field(pattern=r"^(fall|nonfall)$")
    actions: list[str] = Field(min_length=1, max_length=4)
    sets: int = Field(default=5, ge=1, le=100)
    observation_seconds: float = Field(default=10, ge=1, le=300, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_actions(self):
        allowed = FALL_ACTIONS if self.experiment_type == "fall" else NONFALL_ACTIONS
        if len(set(self.actions)) != len(self.actions) or any(action not in allowed for action in self.actions):
            raise ValueError("실험 동작 목록이 올바르지 않습니다.")
        return self


class TrialCommand(BaseModel):
    experiment_id: str
    trial_id: str


class Review(TrialCommand):
    actual_action: str = Field(default="", max_length=200)
    actual_fall: bool | None = None
    valid: bool | None = None
    notes: str = Field(default="", max_length=1000)


def experiment_manager():
    if experiments is None:
        raise HTTPException(503, "실험 관리가 시작되지 않았습니다.")
    return experiments


@app.exception_handler(ExperimentConflict)
async def experiment_conflict(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/experiment", include_in_schema=False)
def experiment_page():
    return FileResponse(BASE_DIR / "templates" / "experiment.html")


@app.get("/experiment/api/state")
def experiment_state():
    return experiment_manager().snapshot()


@app.post("/experiment/api/create")
def experiment_create(config: ExperimentConfig):
    manager = experiment_manager()
    manager.create(config.experiment_id, config.participant_id, config.experiment_type,
                   config.actions, config.sets, config.observation_seconds)
    return manager.snapshot()


@app.post("/experiment/api/start")
def experiment_start(command: TrialCommand):
    with lock:
        session = sessions.get("fall")
        manager = experiment_manager()
        manager.start_trial(
            command.experiment_id, command.trial_id,
            session["fall"].reset if session else lambda: None,
            session["id"] if session else None,
            fall_confirmation_policy,
        )
        return manager.snapshot()


@app.post("/experiment/api/end")
def experiment_end(command: TrialCommand):
    manager = experiment_manager()
    manager.end_trial(command.experiment_id, command.trial_id)
    return manager.snapshot()


@app.post("/experiment/api/next")
def experiment_next(command: TrialCommand):
    manager = experiment_manager()
    manager.next_trial(command.experiment_id, command.trial_id)
    return manager.snapshot()


@app.post("/experiment/api/finish")
def experiment_finish(command: TrialCommand):
    manager = experiment_manager()
    manager.finish(command.experiment_id)
    return manager.snapshot()


@app.post("/experiment/api/review")
def experiment_review(review: Review):
    manager = experiment_manager()
    manager.annotate(review.experiment_id, review.trial_id, review.actual_action,
                     review.actual_fall, review.valid, review.notes)
    return manager.snapshot()


@app.post("/experiment/api/select")
def experiment_select(command: TrialCommand):
    manager = experiment_manager()
    manager.select(command.experiment_id)
    return manager.snapshot()


@app.get("/experiment/api/preview/{run_id}/{trial_id}")
def experiment_preview(run_id: str, trial_id: str):
    path = experiment_manager().preview(run_id, trial_id)
    if path is None:
        raise HTTPException(404, "저장된 이미지가 아직 없습니다.")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/experiment/api/results.csv")
def experiment_csv():
    manager = experiment_manager()
    with lock:
        rows = [trial for run in manager.runs.values() for trial in run["trials"]]
        return Response(csv_text(rows), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="experiment-results.csv"'})
