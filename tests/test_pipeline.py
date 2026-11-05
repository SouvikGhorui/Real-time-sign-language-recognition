"""Checks for the parts where a silent bug would quietly ruin accuracy.

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slr.features import (FEATURE_DIM, NUM_FRAMES, augment, frame_activity, interpolate_short_gaps,
                          prepare_clip, random_indices, sample_frames, to_vector, uniform_indices)
from slr.landmarks import HAND_DIM, POSE_DIM, RAW_DIM, assign_hands
from slr.segmenter import Segmenter


def make_pose(left_wrist=(0.6, 0.5), right_wrist=(0.4, 0.5)) -> np.ndarray:
    """A person facing the camera: the signer's left side is on the image's right."""
    pose = np.zeros((23, 4), dtype=np.float32)
    pose[:, :2] = (0.5, 0.2)
    pose[11, :2] = (0.6, 0.3)  # left shoulder
    pose[12, :2] = (0.4, 0.3)  # right shoulder
    pose[15, :2] = left_wrist
    pose[16, :2] = right_wrist
    return pose


def make_hand(x: float, y: float) -> np.ndarray:
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[:, 0] = x + np.linspace(0, 0.03, 21)
    hand[:, 1] = y - np.linspace(0, 0.05, 21)
    return hand


def make_raw(left=None, right=None, pose=True) -> np.ndarray:
    raw = np.full(RAW_DIM, np.nan, dtype=np.float32)
    if pose:
        raw[:POSE_DIM] = make_pose().ravel()
    if left is not None:
        raw[POSE_DIM:POSE_DIM + HAND_DIM] = make_hand(*left).ravel()
    if right is not None:
        raw[POSE_DIM + HAND_DIM:] = make_hand(*right).ravel()
    return raw


class AssignHands(unittest.TestCase):
    def test_nearest_pose_wrist_decides(self):
        near_left, near_right = make_hand(0.61, 0.5), make_hand(0.39, 0.5)
        left, right = assign_hands([near_right, near_left], ["Left", "Left"], make_pose())
        self.assertIs(left, near_left)
        self.assertIs(right, near_right)

    def test_single_hand(self):
        near_right = make_hand(0.39, 0.5)
        left, right = assign_hands([near_right], ["Left"], make_pose())
        self.assertIsNone(left)
        self.assertIs(right, near_right)

    def test_one_hand_detected_twice_fills_one_slot(self):
        hand = make_hand(0.61, 0.5)
        left, right = assign_hands([hand, hand + 0.002], ["Left", "Right"], make_pose())
        self.assertIs(left, hand)
        self.assertIsNone(right)

    def test_two_hands_close_together_are_both_kept(self):
        near_left, near_right = make_hand(0.53, 0.5), make_hand(0.47, 0.5)
        left, right = assign_hands([near_left, near_right], ["Left", "Right"], make_pose())
        self.assertIs(left, near_left)
        self.assertIs(right, near_right)

    def test_labels_are_swapped_without_pose(self):
        hand = make_hand(0.5, 0.5)
        left, right = assign_hands([hand], ["Left"], None)
        self.assertIsNone(left)
        self.assertIs(right, hand)


class Activity(unittest.TestCase):
    def test_raised_hand_is_active_and_low_hand_is_not(self):
        raw = np.stack([make_raw(left=(0.6, 0.4)), make_raw(left=(0.6, 0.95)), make_raw()])
        self.assertEqual(frame_activity(raw, 1.0).tolist(), [True, False, False])

    def test_aspect_moves_the_rest_line(self):
        # Shoulders 0.2 wide in x. At aspect 1 the rest line is 0.3 + 2.2*0.2 = 0.74.
        # At aspect 2 the same image-x width is twice as far, so it's 0.3 + 2.2*0.4 = 1.18.
        raw = make_raw(left=(0.6, 0.9))[None]
        self.assertFalse(frame_activity(raw, 1.0)[0])
        self.assertTrue(frame_activity(raw, 2.0)[0])

    def test_no_pose_counts_any_visible_hand(self):
        self.assertTrue(frame_activity(make_raw(left=(0.6, 0.99), pose=False)[None], 1.0)[0])


class Gaps(unittest.TestCase):
    def test_short_gap_is_filled_and_long_gap_is_not(self):
        values = np.array([0, np.nan, 2, np.nan, np.nan, np.nan, 6], dtype=np.float32)[:, None]
        filled = interpolate_short_gaps(values, max_gap=2)
        self.assertAlmostEqual(float(filled[1, 0]), 1.0)
        self.assertTrue(np.isnan(filled[3:6]).all())


class Features(unittest.TestCase):
    def setUp(self):
        self.raw = np.stack([make_raw(left=(0.6, 0.4 + i * 0.002)) for i in range(50)])

    def test_output_shape_is_finite(self):
        clip = prepare_clip(self.raw, 1.0)
        vector = to_vector(sample_frames(clip, uniform_indices(len(clip.pose))))
        self.assertEqual(vector.shape, (NUM_FRAMES, FEATURE_DIM))
        self.assertTrue(np.isfinite(vector).all())

    def test_body_coordinates_ignore_position_and_zoom(self):
        moved = self.raw.copy()
        for slice_ in (slice(0, 92, 4), slice(1, 92, 4)):
            moved[:, slice_] = moved[:, slice_] * 0.5 + 0.1
        for start in (POSE_DIM, POSE_DIM + HAND_DIM):
            moved[:, start::3][:, :21] = self.raw[:, start::3][:, :21] * 0.5 + 0.1
            moved[:, start + 1::3][:, :21] = self.raw[:, start + 1::3][:, :21] * 0.5 + 0.1
            moved[:, start + 2::3][:, :21] = self.raw[:, start + 2::3][:, :21] * 0.5
        original, shifted = prepare_clip(self.raw, 1.0), prepare_clip(moved, 1.0)
        np.testing.assert_allclose(original.pose, shifted.pose, atol=1e-4)
        np.testing.assert_allclose(original.hands, shifted.hands, atol=1e-4)

    def test_no_hands_or_no_pose_is_rejected(self):
        self.assertIsNone(prepare_clip(np.stack([make_raw()] * 10), 1.0))
        self.assertIsNone(prepare_clip(np.stack([make_raw(left=(0.5, 0.5), pose=False)] * 10), 1.0))

    def test_mirror_swaps_sides(self):
        clip = prepare_clip(self.raw, 1.0)
        self.assertTrue(clip.present[:, 0].all())
        self.assertFalse(clip.present[:, 1].any())
        seen_swapped = False
        for seed in range(20):
            augmented = augment(clip, np.random.default_rng(seed))
            if augmented.present[:, 1].any():
                seen_swapped = True
                self.assertFalse(augmented.present[:, 0].any())
                # The left wrist was at x > 0, so it should now sit near negative x, in the right slot.
                self.assertLess(augmented.hands[:, 1, 0, 0].mean(), 0)
        self.assertTrue(seen_swapped)

    def test_random_indices_are_ordered_and_in_range(self):
        rng = np.random.default_rng(0)
        for length in (1, 3, 40, 200):
            indices = random_indices(length, rng)
            self.assertEqual(len(indices), NUM_FRAMES)
            self.assertTrue((np.diff(indices) >= 0).all())
            self.assertTrue(indices.min() >= 0 and indices.max() < length)


class Segmentation(unittest.TestCase):
    def feed(self, segmenter, raised_frames, fps=30.0, start=0):
        results, time_s = [], start
        for raised in raised_frames:
            frame = make_raw(left=(0.6, 0.4)) if raised else make_raw()
            segment = segmenter.push(frame, time_s / fps)
            if segment is not None:
                results.append(segment)
            time_s += 1
        return results

    def test_sign_between_rests_gives_one_segment(self):
        results = self.feed(Segmenter(1.0), [False] * 10 + [True] * 30 + [False] * 20)
        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(len(results[0]), 30)

    def test_two_signs_give_two_segments(self):
        pattern = [True] * 30 + [False] * 20 + [True] * 30 + [False] * 20
        self.assertEqual(len(self.feed(Segmenter(1.0), pattern)), 2)

    def test_brief_flicker_is_ignored(self):
        self.assertEqual(self.feed(Segmenter(1.0), [False] * 5 + [True] * 3 + [False] * 30), [])

    def test_short_dropout_does_not_split_a_sign(self):
        pattern = [True] * 20 + [False] * 5 + [True] * 20 + [False] * 20
        self.assertEqual(len(self.feed(Segmenter(1.0), pattern)), 1)

    def test_long_sign_is_capped(self):
        results = self.feed(Segmenter(1.0, max_duration_s=2.0), [True] * 90)
        self.assertEqual(len(results), 1)
        self.assertLessEqual(len(results[0]), 62)


if __name__ == "__main__":
    unittest.main()
