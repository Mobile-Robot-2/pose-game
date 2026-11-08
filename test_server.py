"""Isolated, single-device pose diagnostics; no Firebase or robot actions."""
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from fall_detector import FallDetector
from pose_analyzer import PoseAnalyzer

app = FastAPI(title="Pose Test API")
lock = Lock()
sessions = {}


def analyze(mode, session_id, image_bytes):
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
        result = session["analyzer"].analyze_image_bytes(image_bytes)
        if mode == "fall":
            if not result["detected"]:
                if session["fall"].status == "FALL_SUSPECTED":
                    session["fall"].reset()
                session["fall"].update_no_pose()
                return {"detected": False, "fall_status": "UNKNOWN", "fall_metrics": {}}
            state = session["fall"].update(result["landmarks"])
            if not state["fall_metrics"]["head_shoulders_visible"]:
                if session["fall"].status == "FALL_SUSPECTED":
                    session["fall"].reset()
                return {"detected": False, "fall_status": "UNKNOWN", "fall_metrics": state["fall_metrics"]}
            return {"detected": True, "fall_status": state["fall_status"],
                    "fall_detected": state["fall_detected"], "fall_metrics": state["fall_metrics"]}
        actions = [name for name, detected in result["detections"].items() if detected]
        return {"detected": result["detected"], "current_actions": actions,
                "detected_action": actions[0] if len(actions) == 1 else None}


async def analyze_upload(mode, image, session_id):
    try:
        return analyze(mode, session_id, await image.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/test/fall")
async def test_fall(image: UploadFile = File(...), session_id: str = Form(...)):
    return await analyze_upload("fall", image, session_id)


@app.post("/test/rhythm")
async def test_rhythm(image: UploadFile = File(...), session_id: str = Form(...)):
    return await analyze_upload("rhythm", image, session_id)


@app.on_event("shutdown")
def shutdown():
    for session in sessions.values():
        session["analyzer"].close()
    sessions.clear()
