"""Turns raw landmark sequences into model inputs.

Training and the live app both call `prepare_clip`, then `sample_frames` and
`to_vector`, so a webcam segment gets exactly the same treatment as a training clip.

Body coordinates: origin at the mid-shoulder point, one unit = one shoulder width,
x corrected for aspect ratio. This removes the signer's position, distance from the
camera, and the frame shape, all of which vary a lot between WLASL's sources.
"""
from dataclasses import dataclass

import numpy as np

from .landmarks import HAND_COUNT, HAND_DIM, POSE_COUNT, POSE_DIM

NUM_FRAMES = 40  # every clip is resampled to this many frames
MAX_GAP = 6  # hand dropouts up to this many frames are interpolated

NOSE, LEFT_EAR, RIGHT_EAR = 0, 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
POSE_JOINTS = np.array([NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, 13, 14, 15, 16])
# Indices into POSE_JOINTS of each joint and its left/right partner, used for mirroring.
POSE_MIRROR = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])
WRIST_SLOTS = (7, 8)  # POSE_JOINTS positions of the signer's left and right wrists

# Per frame: 9 joints xy + their velocity, then for each hand:
# presence + wrist xy + wrist velocity xy + 21 wrist-relative xyz.
HAND_FEATURES = 1 + 2 + 2 + HAND_COUNT * 3
FEATURE_DIM = 9 * 2 * 2 + 2 * HAND_FEATURES

REST_LINE = 2.2  # hands lower than this many shoulder widths below the shoulders are "at rest"


@dataclass
class Clip:
    pose: np.ndarray  # (T, 9, 2), body coordinates
    hands: np.ndarray  # (T, 2, 21, 3), body coordinates
    present: np.ndarray  # (T, 2), whether each hand was seen


def split_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 218) -> pose (T, 23, 4), hands (T, 2, 21, 3)."""
    pose = raw[:, :POSE_DIM].reshape(-1, POSE_COUNT, 4)
    left = raw[:, POSE_DIM:POSE_DIM + HAND_DIM].reshape(-1, HAND_COUNT, 3)
    right = raw[:, POSE_DIM + HAND_DIM:].reshape(-1, HAND_COUNT, 3)
    return pose, np.stack([left, right], axis=1)


def interpolate_short_gaps(values: np.ndarray, max_gap: int = MAX_GAP) -> np.ndarray:
    """Linearly fill runs of NaN frames up to max_gap long. Leaves longer gaps as NaN."""
    missing = np.isnan(values.reshape(len(values), -1)).any(axis=1)
    if not missing.any() or missing.all():
        return values
    result = values.copy()
    seen = np.flatnonzero(~missing)
    for before, after in zip(seen[:-1], seen[1:]):
        gap = after - before - 1
        if 0 < gap <= max_gap:
            for step in range(1, gap + 1):
                weight = step / (gap + 1)
                result[before + step] = (1 - weight) * values[before] + weight * values[after]
    return result


def frame_activity(raw: np.ndarray, aspect: float) -> np.ndarray:
    """(T, 218) -> (T,) bool: is at least one hand visible and raised above rest?

    Uses each frame's own shoulders, so the live app can call it one frame at a time.
    y grows downward and is in units of image height, so the shoulder width is
    converted to those units with `aspect` (frame width / height). With no pose in a
    frame, any visible hand counts as raised.
    """
    pose, hands = split_raw(raw)
    left, right = pose[:, LEFT_SHOULDER], pose[:, RIGHT_SHOULDER]
    width = np.maximum(np.abs(left[:, 0] - right[:, 0]) * aspect, 0.05)
    rest_y = (left[:, 1] + right[:, 1]) / 2 + REST_LINE * width
    rest_y = np.where(np.isnan(rest_y), np.inf, rest_y)
    wrist_y = hands[:, :, 0, 1]  # NaN compares False, so a missing hand is never raised
    return (wrist_y < rest_y[:, None]).any(axis=1)


def trim_to_activity(raw: np.ndarray, aspect: float, padding: int = 2) -> np.ndarray:
    """Cut off the rest pose before and after signing, as the live segmenter does."""
    active = np.flatnonzero(frame_activity(raw, aspect))
    if len(active) == 0:
        return raw
    return raw[max(active[0] - padding, 0): active[-1] + padding + 1]


def prepare_clip(raw: np.ndarray, aspect: float) -> Clip | None:
    """Convert raw landmarks to body coordinates. Returns None if there's no usable data.

    `aspect` is frame width / height.
    """
    raw = trim_to_activity(raw, aspect)
    pose_all, hands = split_raw(raw)

    pose_xy = pose_all[:, :, :2].copy()
    if np.isnan(pose_xy).all():
        return None
    pose_xy = interpolate_short_gaps(pose_xy, max_gap=len(pose_xy))
    pose_xy[..., 0] *= aspect

    hands = np.stack([interpolate_short_gaps(hands[:, side]) for side in (0, 1)], axis=1)
    hands = hands.copy()
    hands[..., 0] *= aspect
    hands[..., 2] *= aspect  # MediaPipe's z has about the same scale as image x

    left, right = pose_xy[:, LEFT_SHOULDER], pose_xy[:, RIGHT_SHOULDER]
    center = np.nanmedian((left + right) / 2, axis=0)
    scale = float(np.nanmedian(np.linalg.norm(left - right, axis=1)))
    if not np.isfinite(scale) or scale < 1e-3:
        return None

    pose = (pose_xy[:, POSE_JOINTS] - center) / scale
    pose = np.nan_to_num(pose)

    present = ~np.isnan(hands[:, :, 0, 0])
    if not present.any():
        return None
    hands_body = hands.copy()
    hands_body[..., :2] = (hands_body[..., :2] - center) / scale
    hands_body[..., 2] /= scale
    hands_body = np.nan_to_num(hands_body)
    return Clip(pose.astype(np.float32), hands_body.astype(np.float32), present)


def sample_frames(clip: Clip, indices: np.ndarray) -> Clip:
    return Clip(clip.pose[indices], clip.hands[indices], clip.present[indices])


def uniform_indices(length: int, count: int = NUM_FRAMES) -> np.ndarray:
    return np.round(np.linspace(0, length - 1, count)).astype(np.int64)


def to_vector(clip: Clip) -> np.ndarray:
    """Clip -> (T, FEATURE_DIM) float32 array."""
    frames = len(clip.pose)
    pose_velocity = np.diff(clip.pose, axis=0, prepend=clip.pose[:1])
    parts = [clip.pose.reshape(frames, -1), pose_velocity.reshape(frames, -1)]

    for side in (0, 1):
        present = clip.present[:, side]
        # Where a hand is missing, its position comes from the pose wrist.
        wrist = np.where(present[:, None], clip.hands[:, side, 0, :2], clip.pose[:, WRIST_SLOTS[side]])
        velocity = np.diff(wrist, axis=0, prepend=wrist[:1])
        relative = clip.hands[:, side] - clip.hands[:, side, :1]
        size = np.linalg.norm(relative[:, :, :2], axis=2).max(axis=1, keepdims=True)
        shape = relative / np.maximum(size, 1e-3)[:, :, None]
        shape = shape * present[:, None, None]
        parts += [present[:, None].astype(np.float32), wrist, velocity, shape.reshape(frames, -1)]

    return np.concatenate(parts, axis=1).astype(np.float32)


def augment(clip: Clip, rng: np.random.Generator) -> Clip:
    """Random mirror, rotation, scale, shift, jitter, and dropped hand detections."""
    pose, hands, present = clip.pose.copy(), clip.hands.copy(), clip.present.copy()

    if rng.random() < 0.5:  # a left-handed signer looks like a mirrored right-handed one
        pose = pose[:, POSE_MIRROR]
        pose[..., 0] *= -1
        hands = hands[:, ::-1].copy()
        hands[..., 0] *= -1
        present = present[:, ::-1].copy()

    angle = rng.uniform(-0.2, 0.2)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    scale = rng.uniform(0.85, 1.15) * np.array([rng.uniform(0.9, 1.1), 1.0], dtype=np.float32)
    shift = rng.normal(0, 0.1, size=2).astype(np.float32)

    pose = (pose @ rotation.T) * scale + shift + rng.normal(0, 0.01, pose.shape).astype(np.float32)
    hand_xy = (hands[..., :2] @ rotation.T) * scale + shift
    hands = np.concatenate([hand_xy, hands[..., 2:]], axis=-1)
    hands = hands + rng.normal(0, 0.004, hands.shape).astype(np.float32)

    # Simulate the detector losing a hand for a frame, as it often does on a webcam.
    present = present & (rng.random(present.shape) > 0.05)
    return Clip(pose.astype(np.float32), hands.astype(np.float32), present)


def random_indices(length: int, rng: np.random.Generator, count: int = NUM_FRAMES) -> np.ndarray:
    """Time augmentation: a random crop of the clip, sampled with jitter, in order."""
    keep = max(int(round(length * rng.uniform(0.8, 1.0))), 1)
    start = int(rng.integers(0, length - keep + 1))
    positions = np.linspace(start, start + keep - 1, count)
    positions = positions + rng.uniform(-0.5, 0.5, count) * max(keep / count, 1.0) * 0.5
    return np.clip(np.sort(np.round(positions).astype(np.int64)), 0, length - 1)
