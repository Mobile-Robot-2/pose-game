"""Isolated, single-device pose diagnostics; no Firebase or robot actions."""
import logging
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from fall_debug import FallDebugRecorder
from fall_detector import FallDetector
from pose_analyzer import PoseAnalyzer

app = FastAPI(title="Pose Test API")
lock = Lock()
sessions = {}
fall_debug: FallDebugRecorder | None = None


def analyze(mode, session_id, image_bytes, *, diagnostics=None):
    if not image_bytes:
        raise ValueError("Empty image file")
    with lock:
        session = sessions.get(mode)
        if session is None or session["id"] != session_id:
            if session is not None:
                sessions.pop(mode)
                session["analyzer"].close()
            session = {"id": session_id, "analyzer": PoseAnalyzer(), "fall": FallDetector()}
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
        actions = [name for name, detected in result["detections"].items() if detected]
        return {"detected": result["detected"], "current_actions": actions,
                "detected_action": actions[0] if len(actions) == 1 else None, "landmarks": landmarks}


def _record_fall_state(diagnostics, state):
    if diagnostics is not None:
        metrics = state["fall_metrics"]
        if metrics["reliable_upper_body_visible"]:
            diagnostics.update(metrics)
        diagnostics.update(fall_status=state["fall_status"], fall_detected=state["fall_detected"])


async def analyze_upload(mode, image, session_id):
    try:
        if mode == "fall" and fall_debug is not None:
            with fall_debug.capture("/test/fall", session_id) as diagnostics:
                image_bytes = await image.read()
                if diagnostics["save_frames"]:
                    diagnostics["image_bytes"] = image_bytes
                result = analyze(mode, session_id, image_bytes, diagnostics=diagnostics)
                diagnostics["response_fall_status"] = result["fall_status"]
                return result
        return analyze(mode, session_id, await image.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/test/fall")
async def test_fall(image: UploadFile = File(...), session_id: str = Form(...)):
    return await analyze_upload("fall", image, session_id)


@app.post("/test/rhythm")
async def test_rhythm(image: UploadFile = File(...), session_id: str = Form(...)):
    return await analyze_upload("rhythm", image, session_id)


@app.on_event("startup")
def startup():
    global fall_debug
    mode = os.environ.get("FALL_DEBUG", "").strip().lower()
    if mode in {"csv", "frames"}:
        try:
            root = Path(__file__).resolve().parent / "data" / "fall_debug"
            fall_debug = FallDebugRecorder(root, save_frames=mode == "frames")
        except OSError:
            logging.getLogger(__name__).exception("Could not start fall diagnostics")


@app.on_event("shutdown")
def shutdown():
    global fall_debug
    if fall_debug is not None:
        fall_debug.close()
        fall_debug = None
    for session in sessions.values():
        session["analyzer"].close()
    sessions.clear()
