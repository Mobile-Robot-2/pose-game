"""Analyze a local video without starting the server or writing to Firebase."""
import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
from uuid import uuid4

import cv2

from fall_detector import FallDetector
from pose_analyzer import PoseAnalyzer


FIELDS = (
    "video", "mode", "frame", "video_time_seconds", "sample_interval_seconds", "pose_detected",
    "fall_status", "fall_detected", "fall_event", "torso_angle", "center_drop",
    "current_actions", "detected_action",
)


def analyze_video(video_path: Path, mode: str, output_path: Path, interval: float = 0.2) -> int:
    if mode not in ("fall", "rhythm"):
        raise ValueError("Mode must be fall or rhythm.")
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
        detector = FallDetector() if mode == "fall" else None
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
                    state = (detector.update(analysis["landmarks"], timestamp=timestamp)
                             if analysis["detected"] else detector.update_no_pose(timestamp=timestamp))
                    metrics = state["fall_metrics"]
                    reliable = analysis["detected"] and metrics["reliable_upper_body_visible"]
                    row.update(fall_status=state["fall_status"] if reliable else "UNKNOWN",
                               fall_detected=state["fall_detected"] if reliable else "",
                               fall_event=state["fall_event"]["new_event"],
                               torso_angle=metrics["torso_angle"] if reliable else "",
                               center_drop=metrics["center_drop"] if reliable else "")
                else:
                    actions = [name for name, detected in analysis["detections"].items() if detected]
                    row.update(current_actions="|".join(actions),
                               detected_action=actions[0] if len(actions) == 1 else "")
                writer.writerow(row)
                processed += 1
            if processed == 0:
                raise ValueError("Video contains no readable frames.")
        return processed
    finally:
        if analyzer is not None:
            analyzer.close()
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved video using the pose test algorithms.")
    parser.add_argument("mode", choices=("fall", "rhythm"))
    parser.add_argument("video", type=Path, help="Path to a constant-frame-rate video")
    parser.add_argument("--interval", type=float, help="Minimum sampling interval in seconds (fall: 0.2, rhythm: 0.1)")
    parser.add_argument("--output", type=Path, help="CSV output path; existing files are not overwritten")
    args = parser.parse_args()
    output = args.output or Path("data/video_analysis") / (
        f"{args.video.stem}_{args.mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.csv"
    )
    try:
        interval = args.interval if args.interval is not None else (0.1 if args.mode == "rhythm" else 0.2)
        count = analyze_video(args.video, args.mode, output, interval)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")
    print(f"Analyzed frames: {count}\nResult CSV: {output.resolve()}")


if __name__ == "__main__":
    main()
