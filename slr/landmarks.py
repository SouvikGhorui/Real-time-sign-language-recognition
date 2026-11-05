"""MediaPipe landmark extraction, shared by dataset preprocessing and the live app.

Each frame becomes a 218-value vector with the same layout as vision_test.py:

    pose: 23 landmarks * (x, y, z, visibility) = 92
    left hand: 21 * (x, y, z) = 63
    right hand: 21 * (x, y, z) = 63

Differences from vision_test.py that matter for recognition:

* num_hands=2. The MediaPipe default of 1 would drop a hand from two-handed signs.
* Missing values are NaN, not 0, because 0 is a real image coordinate.
* VIDEO running mode, so landmarks are tracked between frames. This is faster and
  steadier than detecting from scratch on every frame.
* Frames must NOT be mirrored before they get here. WLASL videos are unmirrored, so
  a mirrored webcam frame would look like a different-handed signer to the model.
  Mirror the picture only when displaying it.
* Hand slots are chosen by the nearest pose wrist, falling back on MediaPipe's
  handedness label only when there is no pose.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

POSE_COUNT = 23
HAND_COUNT = 21
POSE_DIM = POSE_COUNT * 4
HAND_DIM = HAND_COUNT * 3
RAW_DIM = POSE_DIM + 2 * HAND_DIM  # 218

POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
DUPLICATE_HAND_RATIO = 0.25  # mean joint gap, as a fraction of hand size

ROOT = Path(__file__).resolve().parent.parent


def is_same_hand(first: np.ndarray, second: np.ndarray) -> bool:
    """True if two detections sit on top of each other, joint for joint.

    MediaPipe sometimes reports one hand twice. Two real hands held close together
    (as in BOOK) still have clearly different joint positions, so they don't match.
    """
    size = max(float(np.ptp(first[:, :2], axis=0).max()), 1e-3)
    gap = float(np.linalg.norm(first[:, :2] - second[:, :2], axis=1).mean())
    return gap < DUPLICATE_HAND_RATIO * size


def assign_hands(hands: list[np.ndarray], labels: list[str], pose: np.ndarray | None):
    """Return (left, right) hand arrays of shape (21, 3), or None for a missing hand.

    `pose` has shape (23, 4) or is None. `left` means the signer's own left hand.
    """
    if len(hands) == 2 and is_same_hand(hands[0], hands[1]):
        hands, labels = hands[:1], labels[:1]
    if not hands:
        return None, None

    if pose is not None:
        left_wrist = pose[POSE_LEFT_WRIST, :2]
        right_wrist = pose[POSE_RIGHT_WRIST, :2]
        to_left = [float(np.linalg.norm(h[0, :2] - left_wrist)) for h in hands]
        to_right = [float(np.linalg.norm(h[0, :2] - right_wrist)) for h in hands]
        if len(hands) == 1:
            return (hands[0], None) if to_left[0] <= to_right[0] else (None, hands[0])
        if to_left[0] + to_right[1] <= to_left[1] + to_right[0]:
            return hands[0], hands[1]
        return hands[1], hands[0]

    # MediaPipe labels assume a mirrored image. Frames here are unmirrored, so the
    # labels are swapped relative to the signer's real hands.
    left = right = None
    for hand, label in zip(hands, labels):
        if label == "Right" and left is None:
            left = hand
        elif label == "Left" and right is None:
            right = hand
    return left, right


class LandmarkExtractor:
    """Runs the hand and pose models over consecutive frames of ONE video.

    Timestamps must strictly increase, so create a new extractor for each video.
    """

    def __init__(self, hand_model: Path = ROOT / "hand_landmarker.task",
                 pose_model: Path = ROOT / "pose_landmarker.task"):
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=str(hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        ))
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.VIDEO,
        ))
        self._last_ms = -1

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Return the 218-value vector for one unmirrored BGR frame."""
        timestamp_ms = max(int(timestamp_ms), self._last_ms + 1)
        self._last_ms = timestamp_ms

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hand_result = self._hands.detect_for_video(image, timestamp_ms)
        pose_result = self._pose.detect_for_video(image, timestamp_ms)

        pose = None
        if pose_result.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in pose_result.pose_landmarks[0][:POSE_COUNT]], dtype=np.float32)

        hands = [np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                 for landmarks in hand_result.hand_landmarks]
        labels = [h[0].category_name for h in hand_result.handedness]
        left, right = assign_hands(hands, labels, pose)

        output = np.full(RAW_DIM, np.nan, dtype=np.float32)
        if pose is not None:
            output[:POSE_DIM] = pose.ravel()
        if left is not None:
            output[POSE_DIM:POSE_DIM + HAND_DIM] = left.ravel()
        if right is not None:
            output[POSE_DIM + HAND_DIM:] = right.ravel()
        return output

    def close(self) -> None:
        self._hands.close()
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
