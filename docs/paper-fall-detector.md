# Pose feature two stage fall detector

`pose_fall_detector.py` is an independent MediaPipe Pose implementation of
posture features and Stage 1 rules adapted from Han et al. (2020). The existing
`FallDetector` remains unchanged. `analyze_video.py fall ...` feeds the same
landmarks and source-video timestamps to both detectors and writes both
decisions and the pose detector's feature diagnostics to one CSV. The server
response also contains a `pose_fall` object, and `FALL_DEBUG=csv` records the
per-frame diagnostics.

## Landmark and feature mapping

- Neck: midpoint of MediaPipe shoulders 11 and 12.
- MidHip: midpoint of MediaPipe hips 23 and 24.
- Spine vector: Neck minus MidHip.
- Waist width: right hip minus left hip.
- Thigh vectors: each Hip to its Knee (25/26).
- Calf vectors: each Knee to its Ankle (27/28).
- Spine Ratio: spine length divided by waist-width length.

Angles are unsigned deflections from vertical gravity, calculated as
`degrees(atan2(abs(dx), abs(dy)))`. Image y grows downwards; absolute components
make the result independent of segment endpoint order and give vertical = 0°.
The paper's printed cosine equation (2) appears to repeat the gravity-vector
magnitude in its denominator; implementation uses the standard dot-product
denominator `|posture vector| * |gravity vector|`, equivalently the angle above.
MediaPipe bilateral midpoint estimates differ from OpenPose's single Neck and
MidHip landmarks, so results are not numerically identical to the paper.

## Stage 1

Strict inequalities from the paper are preserved: T1 is spine angle `< 30°`
and ratio `> 1.0`; T3 is spine angle `> 50°` or ratio `< 1.0`; all other cases
are T2. S1 is assigned if any of the paper's four spine/thigh or thigh/calf
pairs has both angles `< 30°`; otherwise S2. S1+T1 is Stable, S2+T3 is
Disordered, and all other combinations are Fluctuating. Bilateral thigh and
calf values are each retained; their arithmetic means are included as summary
features only.

## Stage 2 and explicit project variation

Three consecutive Fluctuating/Disordered pose frames enter NSPR. Three
consecutive Stable frames exit it as NON-FALL. A missing-pose observation
breaks either consecutive run; it is not added to the NSPR feature window.
During NSPR, up to ten valid pose samples accumulate. The detector also reports
the paper's gamma, epsilon and tau descriptive values when their reference
measurements are available.

**The paper's final decision is a trained SVM on `[gamma, epsilon, tau]`. This
project does not train or load that SVM.** The implemented replacement rule is
explicitly a project variation: if an NSPR reaches ten valid pose samples
without first closing through three stable frames, the result is FALL. This
rule uses the paper's ten-sample window but is not equivalent to its SVM, and
its performance has not been established by the paper's reported metrics. No
additional numeric cutoff was introduced. If no stable reference pose exists,
gamma/epsilon are unavailable (CSV blank) rather than normalized with an
invented reference size.

The project states map as follows: before the NSPR trigger the state is
NORMAL; after trigger it is FALL_SUSPECTED; the ten-sample rule produces
FALL_CONFIRMED; three stable frames after confirmation produce RECOVERED.
These project states are an integration mapping, not states defined by the
paper.

## FPS and timestamps

The saved-video analyzer keeps its 0.04 second default interval (nominally 25
FPS). Detector frame counters count sampled pose observations, while
`frame_interval_seconds` records the actual timestamp difference. The source
timestamp remains `frame_index / source_fps`; changing analysis interval does
not synthesize frames. With lower-FPS or dropped-frame input, the same frame
count spans more time, which is visible in the CSV. The paper's dataset was
25 FPS; its three- and ten-frame intervals therefore correspond to about
0.12 s and 0.40 s at that cadence, but those durations vary if sampling does.

Run a comparison on one video:

```bash
python analyze_video.py fall "data/videos/fall.mp4"
python analyze_video.py fall "data/videos/fall.mp4" --display
```

`--display` opens a live preview with the mapped torso/leg skeleton, feature
values, Stage 1 symbols, NSPR sample count, final result, and sampled timestamp
interval. Press `q` to stop the preview and finish the CSV.

The CSV includes the original detector columns plus `pose_*`, angle, ratio,
symbol, NSPR, frame interval, and temporal-feature columns. The original
`fall_status` / `fall_detected` fields remain the original detector's output.
