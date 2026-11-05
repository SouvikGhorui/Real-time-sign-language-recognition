"""Recognise signs live from a webcam, or from a video file for testing.

    python realtime.py                                  # webcam 0
    python realtime.py --source clip.mp4 --headless     # no window, prints predictions

Raise your hands to sign, then lower them for a moment so the sign can be classified.
Keys: q = quit, c = clear the text.
"""
import argparse
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from slr.landmarks import HAND_COUNT, HAND_DIM, POSE_DIM, LandmarkExtractor
from slr.model import SignPredictor
from slr.segmenter import Segmenter

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "wlasl100_official" / "best.pt"
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class LatestFrameReader:
    """Reads the camera on its own thread so a slow model never builds up a backlog."""

    def __init__(self, capture: cv2.VideoCapture):
        self.capture = capture
        self.frame = None
        self.fresh = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                self.running = False
                self.fresh.set()
                return
            self.frame = frame
            self.fresh.set()

    def read(self):
        self.fresh.wait()
        self.fresh.clear()
        return self.frame if self.running else None

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=2)


def draw_hands(frame: np.ndarray, raw: np.ndarray) -> None:
    """Draw hand skeletons on an already mirrored frame."""
    height, width = frame.shape[:2]
    for offset, color in ((POSE_DIM, (80, 220, 80)), (POSE_DIM + HAND_DIM, (80, 160, 255))):
        hand = raw[offset:offset + HAND_DIM].reshape(HAND_COUNT, 3)
        if np.isnan(hand).any():
            continue
        points = np.stack([(1 - hand[:, 0]) * width, hand[:, 1] * height], axis=1).astype(int)
        for a, b in HAND_LINKS:
            cv2.line(frame, tuple(map(int, points[a])), tuple(map(int, points[b])), color, 2)


def draw_overlay(frame: np.ndarray, status: str, top: list, words: deque, fps: float) -> None:
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 34), (30, 30, 30), -1)
    cv2.putText(frame, f"{status}   {fps:.0f} fps", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for row, (gloss, probability) in enumerate(top[:3]):
        y = 60 + row * 26
        cv2.putText(frame, f"{gloss} {probability:.0%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255) if row == 0 else (200, 200, 200), 2 if row == 0 else 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, height - 40), (width, height), (30, 30, 30), -1)
    cv2.putText(frame, " ".join(words), (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", default="0", help="camera number or video file path")
    parser.add_argument("--threshold", type=float, default=0.35, help="minimum confidence to accept a sign")
    parser.add_argument("--headless", action="store_true", help="print results only, with no window")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"No checkpoint at {args.checkpoint}. Train one first with train.py.")

    torch.set_num_threads(2)
    predictor = SignPredictor(args.checkpoint)

    is_camera = args.source.isdigit()
    capture = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect = width / height if height else 4 / 3
    file_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    reader = LatestFrameReader(capture) if is_camera else None
    segmenter = Segmenter(aspect)
    words: deque[str] = deque(maxlen=8)
    top: list = []
    frame_times: deque[float] = deque(maxlen=30)
    frame_index = 0
    started = time.monotonic()

    def classify(segment: np.ndarray) -> None:
        nonlocal top
        begin = time.perf_counter()
        result = predictor.predict(segment, aspect)
        elapsed = (time.perf_counter() - begin) * 1000
        if not result:
            return
        top = result
        gloss, probability = result[0]
        accepted = probability >= args.threshold
        if accepted:
            words.append(gloss)
        options = ", ".join(f"{g} {p:.0%}" for g, p in result[:3])
        print(f"[{'accepted' if accepted else 'rejected'}] {options} "
              f"({len(segment)} frames, classified in {elapsed:.1f} ms)", flush=True)

    with LandmarkExtractor() as extractor:
        while True:
            if reader:
                frame = reader.read()
            else:
                ok, frame = capture.read()
                frame = frame if ok else None
            if frame is None:
                break

            time_s = time.monotonic() - started if is_camera else frame_index / file_fps
            frame_index += 1
            raw = extractor.process(frame, int(time_s * 1000))  # the unmirrored frame, like the training videos
            segment = segmenter.push(raw, time_s)
            if segment is not None:
                classify(segment)

            frame_times.append(time.perf_counter())
            if args.headless:
                continue

            view = cv2.flip(frame, 1)
            draw_hands(view, raw)
            fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-3)
            draw_overlay(view, "SIGNING..." if segmenter.signing else "Ready", top, words, fps)
            cv2.imshow("Sign language recognition", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                words.clear()
                top = []

    segment = segmenter.flush()
    if segment is not None:
        classify(segment)

    if reader:
        reader.stop()
    capture.release()
    if frame_times and len(frame_times) > 1:
        print(f"Processing speed: {(len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-3):.1f} fps")
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
