from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from action_counter import ActionCounter
from fall_detector import FallDetector
from medication_store import MedicationLogRequest, MedicationStore
from pose_analyzer import MODEL_PATH, PoseAnalyzer

import time
from robot_manager import RobotCommandConflict, RobotManager
from robot_models import RobotCommandRequest, RobotCommandType, RobotStatusUpdate

import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("firebase-key.json")

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL":
        "https://mobile-robot-2-56458-default-rtdb.firebaseio.com/"
    }
)

def _publish_game_state(result: dict) -> None:
    """게임 상태를 Firebase에 기록. 실패해도 낙상/분석 응답은 막지 않는다."""
    firebase_data = {
        "timestamp": time.time(),
        "counts": result["counts"],
        "elapsed_time": result["elapsed_time"],
        "game_clear": result["game_clear"],
        "fall_detected": result["fall_detected"],
    }
    try:
        db.reference("game/current").set(firebase_data)
        db.reference("game/history").push(firebase_data)
    except Exception as exc:
        print(f"[firebase] game state publish failed (ignored): {exc!r}")


app = FastAPI(title="Pose Rhythm Game API")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CONTROL_PAGE = BASE_DIR / "templates" / "control.html"
EVIDENCE_DIR = BASE_DIR / "evidence"
DATA_DIR = BASE_DIR / "data"
MEDICATION_LOG_PATH = DATA_DIR / "medication_logs.json"

game_start_time = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

counter = ActionCounter()
fall_detector = FallDetector()
robot_manager = RobotManager()
medication_store = MedicationStore(MEDICATION_LOG_PATH)
analyzer: PoseAnalyzer | None = None
startup_error: str | None = None
processing_lock = Lock()
current_fall_session_id: str | None = None
current_fall_location: str | None = None
seen_fall_session_ids: set[str] = set()
fall_frame_counter = 0
consecutive_fall_count = 0
last_fall_result: dict | None = None


@app.get("/")
def root() -> dict:
    return health()


@app.on_event("startup")
def startup() -> None:
    global analyzer, startup_error
    try:
        analyzer = PoseAnalyzer()
        startup_error = None
    except FileNotFoundError as exc:
        analyzer = None
        startup_error = str(exc)


@app.on_event("shutdown")
def shutdown() -> None:
    if analyzer is not None:
        analyzer.close()


@app.get("/health")
def health() -> dict:
    if startup_error:
        return {
            "status": "error",
            "message": startup_error,
            "model_path": str(MODEL_PATH),
        }
    return {"status": "ok"}


@app.get("/control", include_in_schema=False)
def control_page() -> FileResponse:
    return FileResponse(CONTROL_PAGE)


@app.post("/control/start_patrol")
def control_start_patrol(robot_id: str = "temi-01") -> dict:
    request = RobotCommandRequest(
        robot_id=robot_id,
        command=RobotCommandType.START_PATROL,
        location=None,
    )
    return create_robot_command(request)


@app.post("/control/medication_alarm")
def control_medication_alarm(
    hour: int = Form(...),
    minute: int = Form(...),
    robot_id: str = Form("temi-01"),
) -> dict:
    request = RobotCommandRequest(
        robot_id=robot_id,
        command=RobotCommandType.SET_MEDICATION_ALARM,
        hour=hour,
        minute=minute,
    )
    return create_robot_command(request)


@app.post("/control/cancel_medication_alarm")
def control_cancel_medication_alarm(robot_id: str = Form("temi-01")) -> dict:
    request = RobotCommandRequest(
        robot_id=robot_id,
        command=RobotCommandType.CANCEL_MEDICATION_ALARM,
    )
    return create_robot_command(request)


@app.post("/medication/log")
def add_medication_log(log: MedicationLogRequest) -> dict:
    entry = medication_store.add_log(log)
    print(f"Medication log saved: {entry}")
    return {
        "success": True,
        "log": entry,
    }


@app.get("/medication/logs")
def get_medication_logs(robot_id: str = "temi-01") -> dict:
    return {
        "success": True,
        "logs": medication_store.list_logs(robot_id),
    }


@app.post("/robot/command")
def create_robot_command(request: RobotCommandRequest) -> dict:
    if request.command == RobotCommandType.START_PATROL:
        with processing_lock:
            _reset_fall_tracking_locked()

    try:
        command = robot_manager.create_command(request)
    except RobotCommandConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "success": True,
        "command": command,
    }


@app.get("/robot/command")
def get_robot_command(
    robot_id: str = "temi-01",
    last_command_id: str | None = None,
) -> dict:
    command = robot_manager.get_command(robot_id, last_command_id)
    return {
        "success": True,
        "has_command": command is not None,
        "command": command,
    }


@app.post("/robot/status")
def update_robot_status(update: RobotStatusUpdate) -> dict:
    status = robot_manager.update_status(update)
    return {
        "success": True,
        "status": status,
    }


@app.get("/robot/status")
def get_robot_status() -> dict:
    return {
        "success": True,
        "status": robot_manager.get_status(),
        "active_command": robot_manager.get_active_command(),
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
    elapsed_time: float | None = Form(None),
) -> dict:
    upload = image or file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing image file. Use multipart field name 'image' or 'file'.",
        )
    result = await _analyze_upload(upload, elapsed_time)

    _publish_game_state(result)

    return result


@app.post("/analyze_frame")
async def analyze_frame(
    image: UploadFile = File(...),
    elapsed_time: float | None = Form(None),
    session_id: str | None = Form(None),
    location: str | None = Form(None),
) -> dict:
    result = await _analyze_upload(
        image,
        elapsed_time=elapsed_time,
        session_id=session_id,
        location=location,
    )

    if result.get("stale"):
        _log_fall_frame(result)
        return result

    _publish_game_state(result)

    _log_fall_frame(result)
    return result


async def _analyze_upload(
    upload: UploadFile,
    elapsed_time: float | None = None,
    session_id: str | None = None,
    location: str | None = None,
) -> dict:
    if analyzer is None:
        raise HTTPException(
            status_code=500,
            detail={
                "message": startup_error or "Pose analyzer is not initialized.",
                "model_path": str(MODEL_PATH),
            },
        )

    image_bytes = await upload.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        with processing_lock:
            session_state = _prepare_fall_session_locked(session_id, location)
            if session_state["stale"]:
                return _stale_analyze_response(session_state, elapsed_time)

            analysis = analyzer.analyze_image_bytes(image_bytes)
            if analysis["detected"]:
                state = counter.update(analysis["detections"], elapsed_time=elapsed_time)
                fall_state = fall_detector.update(analysis["landmarks"])
                pose_not_found = False
            else:
                state = counter.mark_no_pose(elapsed_time=elapsed_time)
                if session_state["session_id"] is not None:
                    fall_state = _pose_not_found_fall_state()
                else:
                    fall_state = fall_detector.update_no_pose()
                pose_not_found = True

            fall_event = fall_state.get("fall_event", {})
            if fall_event.get("new_event"):
                evidence_image = _save_fall_evidence_image(
                    image_bytes=image_bytes,
                    event_id=fall_event["event_id"],
                )
                fall_detector.attach_evidence_image(evidence_image)
                fall_state["fall_event"]["evidence_image"] = evidence_image
                fall_detector.mark_event_handled()

            session_state = _update_fall_tracking_locked(session_state, fall_state)
            image_shape = _image_shape_from_bytes(image_bytes) if pose_not_found else None
            if pose_not_found and session_state["session_id"] is not None:
                _log_pose_not_found(session_state, image_shape)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pose analysis failed: {exc}") from exc

    if game_start_time is not None:
        elapsed_time = round(
            time.time() - game_start_time,
            2
        )


    return {
        "success": True,
        "detected": analysis["detected"],
        "message": None if analysis["detected"] else "No pose detected",
        "detected_action": state["detected_action"],
        "expected_action": state["expected_action"],
        "action_correct": state["action_correct"],
        "action_score": state["action_score"],
        "total_pose_score": state["total_pose_score"],
        "current_actions": state["current_actions"],
        "counts": state["counts"],
        "target_count": state["target_count"],
        "remaining_time": state["remaining_time"],
        "clear": state["clear"],
        "game_clear": state["game_clear"],
        "time_over": state["time_over"],
        **fall_state,
        "confidence": fall_state["fall_confidence"],
        "pose_not_found": pose_not_found,
        "image_shape": image_shape,
        **session_state,
        "elapsed_time": elapsed_time,
    }


def _normalize_form_value(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _reset_fall_tracking_locked() -> dict:
    global current_fall_session_id
    global current_fall_location
    global fall_frame_counter
    global consecutive_fall_count
    global last_fall_result

    fall_state = fall_detector.reset()
    current_fall_session_id = None
    current_fall_location = None
    fall_frame_counter = 0
    consecutive_fall_count = 0
    last_fall_result = None
    return fall_state


def _prepare_fall_session_locked(
    session_id: str | None,
    location: str | None,
) -> dict:
    global current_fall_session_id
    global current_fall_location
    global seen_fall_session_ids
    global fall_frame_counter
    global consecutive_fall_count
    global last_fall_result

    normalized_session_id = _normalize_form_value(session_id)
    normalized_location = _normalize_form_value(location)
    if normalized_session_id is None:
        return {
            "session_id": None,
            "location": normalized_location,
            "stale": False,
            "fall_counter": fall_frame_counter,
            "consecutive_fall_count": consecutive_fall_count,
            "confirm_count": consecutive_fall_count,
            "last_session_id": current_fall_session_id,
            "last_location": current_fall_location,
        }

    if (
        normalized_session_id != current_fall_session_id
        and normalized_session_id in seen_fall_session_ids
    ):
        return {
            "session_id": normalized_session_id,
            "location": normalized_location,
            "stale": True,
            "current_session_id": current_fall_session_id,
            "current_location": current_fall_location,
            "fall_counter": fall_frame_counter,
            "consecutive_fall_count": consecutive_fall_count,
            "confirm_count": consecutive_fall_count,
            "last_session_id": current_fall_session_id,
            "last_location": current_fall_location,
        }

    if normalized_session_id != current_fall_session_id:
        fall_detector.reset()
        current_fall_session_id = normalized_session_id
        current_fall_location = normalized_location
        seen_fall_session_ids.add(normalized_session_id)
        fall_frame_counter = 0
        consecutive_fall_count = 0
        last_fall_result = None
    elif current_fall_location is None:
        current_fall_location = normalized_location
    elif normalized_location and normalized_location != current_fall_location:
        return {
            "session_id": normalized_session_id,
            "location": normalized_location,
            "stale": True,
            "current_session_id": current_fall_session_id,
            "current_location": current_fall_location,
            "fall_counter": fall_frame_counter,
            "consecutive_fall_count": consecutive_fall_count,
            "confirm_count": consecutive_fall_count,
            "last_session_id": current_fall_session_id,
            "last_location": current_fall_location,
        }

    return {
        "session_id": normalized_session_id,
        "location": current_fall_location or normalized_location,
        "stale": False,
        "fall_counter": fall_frame_counter,
        "consecutive_fall_count": consecutive_fall_count,
        "confirm_count": consecutive_fall_count,
        "last_session_id": current_fall_session_id,
        "last_location": current_fall_location,
    }


def _update_fall_tracking_locked(session_state: dict, fall_state: dict) -> dict:
    global fall_frame_counter
    global consecutive_fall_count
    global last_fall_result

    if session_state["session_id"] is not None:
        fall_frame_counter += 1
        if fall_state["fall_detected"]:
            consecutive_fall_count += 1
        else:
            consecutive_fall_count = 0

    session_state["fall_counter"] = fall_frame_counter
    session_state["consecutive_fall_count"] = consecutive_fall_count
    session_state["confirm_count"] = consecutive_fall_count
    session_state["last_session_id"] = current_fall_session_id
    session_state["last_location"] = current_fall_location
    last_fall_result = {
        "fall_detected": fall_state["fall_detected"],
        "fall_status": fall_state["fall_status"],
        "confidence": fall_state["fall_confidence"],
        "session_id": session_state["session_id"],
        "location": session_state["location"],
        "fall_counter": fall_frame_counter,
        "consecutive_fall_count": consecutive_fall_count,
        "confirm_count": consecutive_fall_count,
    }
    return session_state


def _stale_analyze_response(session_state: dict, elapsed_time: float | None) -> dict:
    current_state = counter.get_state()
    return {
        "success": True,
        "detected": False,
        "message": "Stale patrol frame ignored.",
        "detected_action": current_state["detected_action"],
        "expected_action": current_state["expected_action"],
        "action_correct": current_state["action_correct"],
        "action_score": current_state["action_score"],
        "total_pose_score": current_state["total_pose_score"],
        "current_actions": current_state["current_actions"],
        "counts": current_state["counts"],
        "target_count": current_state["target_count"],
        "remaining_time": current_state["remaining_time"],
        "clear": current_state["clear"],
        "game_clear": current_state["game_clear"],
        "time_over": current_state["time_over"],
        "fall_detected": False,
        "fall_status": "NORMAL",
        "fall_confidence": 0.0,
        "confidence": 0.0,
        "pose_not_found": False,
        "image_shape": None,
        "fall_metrics": {},
        "fall_event": {
            "new_event": False,
            "event_id": None,
            "evidence_image": None,
        },
        **session_state,
        "elapsed_time": elapsed_time,
    }


def _log_fall_frame(result: dict) -> None:
    metrics = result.get("fall_metrics") or {}
    print(
        "fall_frame "
        f"session_id={result.get('session_id')} "
        f"location={result.get('location')} "
        f"pose_not_found={result.get('pose_not_found')} "
        f"buffer_filled={metrics.get('buffer_filled')}/{metrics.get('window')} "
        f"fall_detected={result.get('fall_detected')} "
        f"fall_status={result.get('fall_status')} "
        f"confidence={result.get('confidence', result.get('fall_confidence'))} "
        f"confirm_count={result.get('confirm_count')} "
        f"fall_counter={result.get('fall_counter')} "
        f"consecutive_fall_count={result.get('consecutive_fall_count')} "
        f"stale={result.get('stale')}"
    )


def _pose_not_found_fall_state() -> dict:
    return {
        "fall_detected": False,
        "fall_status": "NORMAL",
        "fall_confidence": 0.0,
        "fall_metrics": {},
        "fall_event": {
            "new_event": False,
            "event_id": None,
            "evidence_image": None,
        },
    }


def _image_shape_from_bytes(image_bytes: bytes) -> list[int] | None:
    try:
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1
        return [height, width, channels]
    except Exception:
        return None


def _log_pose_not_found(session_state: dict, image_shape: list[int] | None) -> None:
    print(
        "pose_not_found "
        f"session_id={session_state.get('session_id')} "
        f"location={session_state.get('location')} "
        f"image_shape={image_shape}"
    )


def _save_fall_evidence_image(image_bytes: bytes, event_id: str) -> str | None:
    try:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        filename = f"{event_id}.jpg"
        path = EVIDENCE_DIR / filename
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            return None
        return f"evidence/{filename}"
    except Exception:
        return None


@app.post("/reset")
def reset() -> dict:
    global game_start_time

    # Firebase 이전 게임 데이터 삭제
    db.reference("game/history").delete()
    db.reference("game/current").delete()

    with processing_lock:
        state = counter.reset()
        fall_state = _reset_fall_tracking_locked()
    game_start_time = time.time()
    print(f"게임 시작: {game_start_time}")
    return {
        "success": True,
        "message": "Game state reset.",
        "current_actions": state["current_actions"],
        "detected_action": state["detected_action"],
        "expected_action": state["expected_action"],
        "action_correct": state["action_correct"],
        "action_score": state["action_score"],
        "total_pose_score": state["total_pose_score"],
        "counts": state["counts"],
        "target_count": state["target_count"],
        "remaining_time": state["remaining_time"],
        "clear": state["clear"],
        "game_clear": state["game_clear"],
        "time_over": state["time_over"],
        **fall_state,
    }


@app.post("/reset_fall")
def reset_fall() -> dict:
    """게임 데이터(Firebase/counter)는 건드리지 않고 낙상 검출기 상태만 초기화한다.

    순찰이 홈베이스로 복귀할 때 호출된다. 낙상이 한 번 FALL_CONFIRMED 되면
    프레임 전송이 멈춰 검출기가 스스로 회복하지 못하고 상태가 남기 때문에,
    다음 순찰 도착 즉시 fall_detected=True 가 반환되는 문제를 막기 위함이다.
    """
    with processing_lock:
        fall_state = _reset_fall_tracking_locked()
    return {
        "success": True,
        "message": "Fall detector reset.",
        "session_id": None,
        "location": None,
        "stale": False,
        "fall_counter": 0,
        "consecutive_fall_count": 0,
        "confirm_count": 0,
        "last_session_id": None,
        "last_location": None,
        "confidence": fall_state["fall_confidence"],
        **fall_state,
    }


@app.get("/state")
def state() -> dict:
    with processing_lock:
        current_state = counter.get_state()
        fall_state = fall_detector.get_state()
    
    elapsed_time = None

    if game_start_time is not None:
        elapsed_time = round(
            time.time() - game_start_time,
            2
        )
        
    return {
        "success": True,
        "current_actions": current_state["current_actions"],
        "detected_action": current_state["detected_action"],
        "expected_action": current_state["expected_action"],
        "action_correct": current_state["action_correct"],
        "action_score": current_state["action_score"],
        "total_pose_score": current_state["total_pose_score"],
        "counts": current_state["counts"],
        "target_count": current_state["target_count"],
        "remaining_time": current_state["remaining_time"],
        "clear": current_state["clear"],
        "game_clear": current_state["game_clear"],
        "time_over": current_state["time_over"],
        **fall_state,
        "elapsed_time": elapsed_time,
    }
