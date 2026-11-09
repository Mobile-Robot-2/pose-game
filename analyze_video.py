"""Analyze a local video without starting the server or writing to Firebase."""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
from uuid import uuid4

import cv2

from action_counter import classify_actions
from fall_detector import FallDetector
from pose_fall_detector import PoseFallDetector
from pose_analyzer import PoseAnalyzer


FIELDS = (
    "video", "mode", "frame", "video_time_seconds", "sample_interval_seconds", "pose_detected",
    "fall_status", "fall_detected", "fall_event", "torso_angle", "center_drop",
    "pose_fall_status", "pose_fall_detected", "pose_final_result", "pose_stage1_state",
    "pose_trend_symbol", "pose_stability_symbol", "pose_nspr_active", "pose_nspr_frames",
    "pose_frame_interval_seconds", "spine_deflection_angle", "thigh_deflection_angle",
    "left_thigh_angle", "right_thigh_angle", "calf_deflection_angle", "left_calf_angle",
    "right_calf_angle", "spine_ratio", "pose_angle_limit_degrees",
    "pose_spine_angle_lower_degrees", "pose_spine_angle_upper_degrees",
    "pose_spine_ratio_boundary", "pose_gamma", "pose_epsilon", "pose_tau",
    "current_actions", "detected_action", "prediction_status", "fall_confirmation_policy",
)


def analyze_video(video_path: Path, mode: str, output_path: Path, interval: float = 0.04,
                  require_fall_posture: bool = False, display: bool = False) -> int:
    if mode not in ("fall", "rhythm"):
        raise ValueError("Mode must be fall or rhythm.")
    if require_fall_posture and mode != "fall":
        raise ValueError("Posture evidence policy is only available for fall analysis.")
    if not math.isfinite(interval) or interval < 0.001:
        raise ValueError("Frame interval must be at least 0.001 seconds.")
    if not video_path.is_file():
        raise ValueError(f"Video not found: {video_path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("Output must be different from the input video.")

    capture = cv2.VideoCapture(str(video_path))
    analyzer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Could not determine video FPS.")
        analyzer = PoseAnalyzer()
        detector = FallDetector(require_fall_posture=require_fall_posture) if mode == "fall" else None
        pose_detector = PoseFallDetector() if mode == "fall" else None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed = 0
        next_sample = 0.0
        frame_index = 0
        # shortcut: frame index / FPS assumes constant frame rate, convert variable-rate videos before analysis.
        with output_path.open("x", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                frame_index += 1
                if timestamp + 1e-9 < next_sample:
                    continue
                next_sample = timestamp + interval
                encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not encoded:
                    raise ValueError(f"Could not encode video frame {frame_index}.")
                analysis = analyzer.analyze_image_bytes(jpeg.tobytes(), timestamp_ms=round(timestamp * 1000))
                row = {"video": video_path.name, "mode": mode, "frame": frame_index,
                       "video_time_seconds": round(timestamp, 3), "sample_interval_seconds": interval,
                       "pose_detected": analysis["detected"]}
                if mode == "fall":
                    row["fall_confirmation_policy"] = (
                        "posture_experimental" if require_fall_posture else "baseline")
                    state = (detector.update(analysis["landmarks"], timestamp=timestamp)
                             if analysis["detected"] else detector.update_no_pose(timestamp=timestamp))
                    pose_state = (pose_detector.update(analysis["landmarks"], timestamp=timestamp)
                                  if analysis["detected"] else pose_detector.update_no_pose(timestamp=timestamp))
                    metrics = state["fall_metrics"]
                    reliable = analysis["detected"] and metrics["reliable_upper_body_visible"]
                    features = pose_state["features"]
                    thresholds = features.get("thresholds", {})
                    row.update(fall_status=state["fall_status"] if reliable else "UNKNOWN",
                               fall_detected=state["fall_detected"] if reliable else "",
                               fall_event=state["fall_event"]["new_event"],
                               torso_angle=metrics["torso_angle"] if reliable else "",
                               center_drop=metrics["center_drop"] if reliable else "",
                               pose_fall_status=pose_state["fall_status"],
                               pose_fall_detected=pose_state["fall_detected"],
                               pose_final_result=pose_state["final_result"],
                               pose_stage1_state=features.get("stage1_state"),
                               pose_trend_symbol=features.get("trend_symbol"),
                               pose_stability_symbol=features.get("stability_symbol"),
                               pose_nspr_active=pose_state["nspr_active"],
                               pose_nspr_frames=pose_state["nspr_frame_count"],
                               pose_frame_interval_seconds=features.get("frame_interval_seconds"),
                               spine_deflection_angle=features.get("spine_deflection_angle"),
                               thigh_deflection_angle=features.get("thigh_deflection_angle"),
                               left_thigh_angle=features.get("left_thigh_angle"),
                               right_thigh_angle=features.get("right_thigh_angle"),
                               calf_deflection_angle=features.get("calf_deflection_angle"),
                               left_calf_angle=features.get("left_calf_angle"),
                               right_calf_angle=features.get("right_calf_angle"),
                               spine_ratio=features.get("spine_ratio"),
                               pose_angle_limit_degrees=thresholds.get("angle_limit_degrees"),
                               pose_spine_angle_lower_degrees=thresholds.get("spine_angle_lower_degrees"),
                               pose_spine_angle_upper_degrees=thresholds.get("spine_angle_upper_degrees"),
                               pose_spine_ratio_boundary=thresholds.get("spine_ratio_boundary"),
                               pose_gamma=features.get("gamma"),
                               pose_epsilon=features.get("epsilon"),
                               pose_tau=features.get("tau"))
                else:
                    actions = [name for name, detected in analysis.get("raw_detections", analysis["detections"]).items() if detected]
                    prediction, prediction_status = classify_actions(actions, pose_detected=analysis["detected"])
                    row.update(current_actions="|".join(actions),
                               detected_action=prediction or "", prediction_status=prediction_status)
                if display:
                    preview = frame.copy()
                    if mode == "fall":
                        _draw_fall_diagnostics(preview, analysis.get("landmarks"), pose_state)
                    cv2.imshow("Pose Fall Comparison" if mode == "fall" else "Pose Analysis", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        writer.writerow(row)
                        processed += 1
                        break
                writer.writerow(row)
                processed += 1
            if processed == 0:
                raise ValueError("Video contains no readable frames.")
        return processed
    finally:
        if analyzer is not None:
            analyzer.close()
        capture.release()
        if display:
            cv2.destroyAllWindows()


def _draw_fall_diagnostics(frame, landmarks, pose_state: dict) -> None:
    """Draw the mapped torso/leg skeleton and pose detector diagnostics."""
    height, width = frame.shape[:2]
    features = pose_state["features"]
    if landmarks:
        connections = ((11, 12), (23, 24), (11, 23), (12, 24),
                       (23, 25), (25, 27), (24, 26), (26, 28))
        indices = {index for pair in connections for index in pair}
        pixels = {
            index: (int(landmarks[index].x * width), int(landmarks[index].y * height))
            for index in indices if index < len(landmarks)
        }
        for first, second in connections:
            if first in pixels and second in pixels:
                cv2.line(frame, pixels[first], pixels[second], (0, 220, 255), 2)
        for point in pixels.values():
            cv2.circle(frame, point, 4, (50, 255, 120), -1)

    def shown(value, digits=1):
        return "--" if value is None else f"{value:.{digits}f}"

    lines = (
        f"Spine {shown(features.get('spine_deflection_angle'))} deg | Ratio {shown(features.get('spine_ratio'), 3)}",
        f"Thigh {shown(features.get('thigh_deflection_angle'))} deg | Calf {shown(features.get('calf_deflection_angle'))} deg",
        "Limits angle 30/30/50 deg | ratio 1.0",
        f"Trend {features.get('trend_symbol')} | Stability {features.get('stability_symbol')} | Stage1 {features.get('stage1_state')}",
        f"NSPR {pose_state['nspr_active']} | frames {pose_state['nspr_frame_count']}/10 | result {pose_state['final_result']}",
        f"Pose state {pose_state['fall_status']} | dt {shown(features.get('frame_interval_seconds'), 3)}s",
    )
    for index, line in enumerate(lines):
        y = 28 + index * 26
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved video using the pose test algorithms.")
    parser.add_argument("mode", choices=("fall", "rhythm"))
    parser.add_argument("video", type=Path, help="Path to a constant-frame-rate video")
    parser.add_argument("--interval", type=float, help="Minimum sampling interval in seconds (fall: 0.04, rhythm: 0.1)")
    parser.add_argument("--output", type=Path, help="CSV output path; existing files are not overwritten")
    parser.add_argument("--require-fall-posture", action="store_true",
                        help="Experimental fall policy; can miss falls and is for comparison only")
    parser.add_argument("--display", action="store_true",
                        help="Show pose skeleton and pose-detector diagnostics while analyzing")
    args = parser.parse_args()
    output = args.output or Path("data/video_analysis") / (
        f"{args.video.stem}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.csv"
    )
    try:
        interval = args.interval if args.interval is not None else (0.1 if args.mode == "rhythm" else 0.04)
        count = analyze_video(args.video, args.mode, output, interval,
                              args.require_fall_posture, args.display)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"Analyzed frames: {count}\nResult CSV: {output.resolve()}")


if __name__ == "__main__":
    main()
