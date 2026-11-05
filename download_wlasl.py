"""Download the raw videos for a WLASL subset, safely resumable.

    python download_wlasl.py --subset 100

Videos are saved to data/wlasl/raw/<key>.<ext>, where <key> is a hash of the URL,
so instances that share a source video only download it once. Results are logged
to data/wlasl/download_status.json. Permanent failures are not retried unless you
pass --retry-failed.
"""
import argparse
import collections
import concurrent.futures as futures
import hashlib
import json
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
META = ROOT / "data" / "wlasl" / "meta" / "WLASL_v0.3.json"
RAW = ROOT / "data" / "wlasl" / "raw"
STATUS = ROOT / "data" / "wlasl" / "download_status.json"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"
ASLPRO_REFERER = "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi"
TIMEOUT_S = 45


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


def find_existing(key: str) -> Path | None:
    for path in RAW.glob(f"{key}.*"):
        # key.f140.m4a style names are single-stream leftovers from a failed yt-dlp merge.
        is_leftover = path.suffix in {".part", ".ytdl", ".tmp"} or len(path.suffixes) > 1
        if not is_leftover and path.stat().st_size > 0:
            return path
    return None


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def fetch_direct(url: str, key: str) -> Path:
    headers = {"User-Agent": USER_AGENT}
    if "aslpro" in url:
        headers["Referer"] = ASLPRO_REFERER
    request = urllib.request.Request(url, headers=headers)
    suffix = Path(urlparse(url).path).suffix.lower() or ".mp4"
    target = RAW / f"{key}{suffix}"
    partial = target.with_name(target.name + ".tmp")
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    if partial.stat().st_size == 0:
        partial.unlink()
        raise ValueError("empty response")
    partial.replace(target)
    return target


def fetch_youtube(url: str, key: str) -> Path:
    command = [
        sys.executable, "-m", "yt_dlp", url,
        "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]/b",
        "--merge-output-format", "mp4",
        "--no-playlist", "--no-progress", "--quiet", "--no-warnings",
        "--socket-timeout", str(TIMEOUT_S), "--retries", "2",
        "-o", str(RAW / f"{key}.%(ext)s"),
    ]
    done = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=600)
    if done.returncode != 0:
        lines = done.stderr.strip().splitlines()
        raise RuntimeError(lines[-1][:300] if lines else f"yt-dlp exit {done.returncode}")
    found = find_existing(key)
    if found is None:
        raise RuntimeError("yt-dlp reported success but produced no file")
    return found


class HostUnreachable(Exception):
    """The host was skipped for this run after repeated connection failures."""


class HostThrottle:
    """One request at a time per host, with a pause, and a cutoff for dead hosts.

    After MAX_CONNECTION_FAILURES in a row with no HTTP response from a host, the
    rest of its URLs are skipped for this run. They are not marked as failed, so
    the next run tries them again.
    """

    MAX_CONNECTION_FAILURES = 5

    def __init__(self):
        self._locks = collections.defaultdict(threading.Lock)
        self._failures = collections.Counter()

    def run(self, host: str, call):
        with self._locks[host]:
            if self._failures[host] >= self.MAX_CONNECTION_FAILURES:
                raise HostUnreachable(host)
            try:
                result = call()
            except urllib.error.HTTPError:
                self._failures[host] = 0  # the server answered, so the host is alive
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                self._failures[host] += 1
                raise
            finally:
                time.sleep(random.uniform(0.5, 1.5))
            self._failures[host] = 0
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=100, choices=[100, 300, 1000, 2000])
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (one per host at a time)")
    parser.add_argument("--limit", type=int, default=0, help="only try N URLs per host (for testing)")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    entries = json.loads(META.read_text(encoding="utf-8"))[: args.subset]
    urls = sorted({inst["url"] for entry in entries for inst in entry["instances"]})

    status: dict = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    lock = threading.Lock()
    throttle = HostThrottle()

    todo = []
    per_host = collections.Counter()
    for url in urls:
        host = urlparse(url).netloc
        if find_existing(url_key(url)):
            status[url] = {"ok": True}
            continue
        if url in status and not status[url]["ok"] and not args.retry_failed:
            continue
        if args.limit and per_host[host] >= args.limit:
            continue
        per_host[host] += 1
        todo.append(url)
    random.Random(0).shuffle(todo)
    print(f"{len(urls)} unique URLs; {len(todo)} to download", flush=True)

    def work(url: str) -> None:
        key = url_key(url)
        fetch = fetch_youtube if is_youtube(url) else fetch_direct
        host = "youtube" if is_youtube(url) else urlparse(url).netloc
        try:
            path = throttle.run(host, lambda: fetch(url, key))
            result = {"ok": True, "file": path.name}
        except HostUnreachable:
            return
        except (urllib.error.URLError, OSError, ValueError, RuntimeError,
                subprocess.TimeoutExpired) as error:
            result = {"ok": False, "error": f"{type(error).__name__}: {error}"[:300]}
        with lock:
            status[url] = result

    with futures.ThreadPoolExecutor(args.workers) as pool:
        for index, _ in enumerate(pool.map(work, todo), 1):
            if index % 25 == 0 or index == len(todo):
                with lock:
                    STATUS.write_text(json.dumps(status, indent=1), encoding="utf-8")
                    okay = sum(s["ok"] for s in status.values())
                print(f"{index}/{len(todo)} attempted; {okay} available so far", flush=True)

    STATUS.write_text(json.dumps(status, indent=1), encoding="utf-8")

    summary = collections.defaultdict(collections.Counter)
    for url in urls:
        host = urlparse(url).netloc
        summary[host]["ok" if status.get(url, {}).get("ok") else "missing"] += 1
    print("\nHost results:")
    for host, counts in sorted(summary.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {host:40s} ok={counts['ok']:4d} missing={counts['missing']:4d}")
    available = sum(c["ok"] for c in summary.values())
    print(f"\nTotal: {available}/{len(urls)} source videos available")


if __name__ == "__main__":
    main()
