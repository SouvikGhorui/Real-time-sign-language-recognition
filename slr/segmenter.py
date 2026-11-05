"""Finds where each sign starts and ends in a live stream of landmarks.

A sign starts when a hand is raised above the rest line and ends when the hands have
been down or out of view for `end_gap_s`. It uses the same rule that trims training
clips. Signs therefore need a short pause with the hands lowered between them.
"""
from collections import deque

import numpy as np

from .features import frame_activity


class Segmenter:
    def __init__(self, aspect: float, min_duration_s: float = 0.3, end_gap_s: float = 0.4,
                 max_duration_s: float = 4.0, preroll_frames: int = 3):
        self.aspect = aspect
        self.min_duration_s = min_duration_s
        self.end_gap_s = end_gap_s
        self.max_duration_s = max_duration_s
        self._preroll = deque(maxlen=preroll_frames)
        self._frames: list[np.ndarray] = []
        self._start = 0.0
        self._last_active = 0.0

    @property
    def signing(self) -> bool:
        return bool(self._frames)

    def push(self, raw_frame: np.ndarray, time_s: float) -> np.ndarray | None:
        """Add one frame. Returns the finished segment (T, 218) when a sign ends."""
        active = bool(frame_activity(raw_frame[None], self.aspect)[0])

        if not self._frames:
            if active:
                self._frames = [*self._preroll, raw_frame]
                self._preroll.clear()
                self._start = self._last_active = time_s
            else:
                self._preroll.append(raw_frame)
            return None

        self._frames.append(raw_frame)
        if active:
            self._last_active = time_s

        ended = time_s - self._last_active >= self.end_gap_s
        too_long = time_s - self._start >= self.max_duration_s
        if not (ended or too_long):
            return None

        duration = self._last_active - self._start
        segment = np.stack(self._frames)
        self._frames = []
        return segment if duration >= self.min_duration_s else None

    def flush(self) -> np.ndarray | None:
        """Return any sign still in progress, for when the input stream ends."""
        if not self._frames or self._last_active - self._start < self.min_duration_s:
            self._frames = []
            return None
        segment = np.stack(self._frames)
        self._frames = []
        return segment
