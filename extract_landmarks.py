"""Run MediaPipe over the downloaded WLASL videos and save landmarks for each clip.

    python extract_landmarks.py --subset 100

Writes data/wlasl/landmarks/<video_id>.npz with:
    raw: (T, 218) float32 landmarks, NaN where missing
    aspect: frame width / height
    fps: source frame rate

Existing files are skipped, so this can be stopped and resumed.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
META = ROOT / "data" / "wlasl" / "meta" / "WLASL_v0.3.json"
RAW = ROOT / "data" / "wlasl" / "raw"
OUT = ROOT / "data" / "wlasl" / "landmarks"
WLASL_FPS = 25.0  # frame_start and frame_end are defined at 25 fps
MAX_SIDE = 960  # larger frames are shrunk; the landmark models work on small inputs anyway


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def find_source(url: str) -> Path | None:
    for path in RAW.glob(f"{url_key(url)}.*"):
        # key.f140.m4a style names are single-stream leftovers from a failed yt-dlp merge.
        if path.suffix not in {".part", ".ytdl", ".tmp"} and len(path.suffixes) == 1:
            return path
    return None


def extract(job: tuple[dict, str]) -> tuple[str, str]:
    """Process one clip. Returns (video_id, 'ok' or an error description)."""
    from slr.landmarks import LandmarkExtractor  # imported here so each worker loads its own

    instance, source = job
    video_id = instance["video_id"]
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        return video_id, "cannot open video"

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps < 1 or fps > 120:
        fps = WLASL_FPS

    # Convert the 1-indexed 25 fps frame range to this file's frame numbers.
    first = int(round((instance["frame_start"] - 1) / WLASL_FPS * fps))
    last = None
    if instance["frame_end"] > 0:
        last = int(round((instance["frame_end"] - 1) / WLASL_FPS * fps))
    if first > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)

    rows, aspect = [], None
    index = first
    with LandmarkExtractor() as extractor:
        while last is None or index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            aspect = width / height
            longest = max(height, width)
            if longest > MAX_SIDE:
                ratio = MAX_SIDE / longest
                frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
            rows.append(extractor.process(frame, int((index - first) * 1000 / fps)))
            index += 1
    capture.release()

    if len(rows) < 5:
        return video_id, f"only {len(rows)} frames decoded"
    temp = OUT / f"{video_id}.tmp.npz"
    np.savez_compressed(temp, raw=np.stack(rows), aspect=np.float32(aspect), fps=np.float32(fps))
    temp.replace(OUT / f"{video_id}.npz")
    return video_id, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    parser.add_argument("--workers", type=int, default=max((os.cpu_count() or 4) // 2, 1))
    parser.add_argument("--limit", type=int, default=0, help="only process N clips (for testing)")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.loads(META.read_text(encoding="utf-8"))[: args.subset]

    jobs, no_source, done = [], 0, 0
    for entry in entries:
        for instance in entry["instances"]:
            source = find_source(instance["url"])
            if source is None:
                no_source += 1
            elif (OUT / f"{instance['video_id']}.npz").exists():
                done += 1
            else:
                jobs.append((instance, str(source)))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} clips to process, {done} already done, {no_source} have no source video", flush=True)

    errors = {}
    with multiprocessing.Pool(args.workers) as pool:
        for count, (video_id, result) in enumerate(pool.imap_unordered(extract, jobs), 1):
            if result != "ok":
                errors[video_id] = result
            if count % 50 == 0 or count == len(jobs):
                print(f"{count}/{len(jobs)} processed, {len(errors)} errors", flush=True)

    (ROOT / "data" / "wlasl" / "extract_errors.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    print(f"Finished: {len(jobs) - len(errors)} ok, {len(errors)} errors")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("GLOG_minloglevel", "2")
    main()
