"""
Download every AVA video that carries AVA-ActiveSpeaker labels, and the
annotation files the four components need.

WHY THIS REPLACES THE OLD SCRIPT
--------------------------------
The previous downloader walked the AVA-Speech file list in alphabetical order.
That list is not the list we need. Our features come from AVA-ActiveSpeaker,
whose public labels cover 153 videos (120 train + 33 val). Walking the wrong
list is how a video with no ActiveSpeaker labels ended up on disk while 114
labelled ones are still missing.

This script builds its target list from the ActiveSpeaker annotation archives
themselves, so "complete" means exactly "every video we have labels for".

WHAT IT DOES TO FILES ALREADY ON DISK
-------------------------------------
The old script wrote straight to the final filename. If a download was ever
interrupted, the truncated file looks complete and nothing would notice until
feature extraction failed weeks later. So before trusting an existing file,
this script asks the server for the true size (an HTTP HEAD request) and:

    same size        -> keep it
    smaller          -> it was truncated; resume it from where it stopped
    size unknown     -> keep it, but say so

RESUMING
--------
Every download is written to <name>.part first. If the connection drops, the
next run asks the server only for the missing bytes (an HTTP Range request) and
appends them, so a 2 GB film interrupted at 1.9 GB costs 100 MB, not 2 GB.
A file is renamed to its final name only once its size matches the server's.

USAGE (from the repository, on the machine that holds the data)
---------------------------------------------------------------
    python download_ava.py --list          # what is present / missing, no download
    python download_ava.py --verify        # check files already on disk against the server
    python download_ava.py                 # download everything still missing
    python download_ava.py --limit 10      # download at most 10 videos, then stop
    python download_ava.py --split val     # only the 33 validation videos

Safe to stop with Ctrl+C and re-run at any time.

Standard library only, so it runs on any Python 3.8+ without installing anything.
"""

import argparse
import csv
import datetime as dt
import os
import socket
import sys
import tarfile
import time
import urllib.error
import urllib.request

# --- where things go -------------------------------------------------------
# Corpora live OUTSIDE the git repository, next to it, so they can never be
# committed by accident:
#     Research Project/
#     |-- AVA/          <- annotations here, videos in AVA/ava_videos/
#     `-- equvid/       <- the repo; this file is equvid/data/download_ava.py
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "AVA"))
VIDEO_DIR = os.path.join(OUT_DIR, "ava_videos")
LOG = os.path.join(OUT_DIR, "download_log.csv")

S3 = "https://s3.amazonaws.com/ava-dataset"

# ActiveSpeaker archives: their member names ARE the target list.
AS_ARCHIVES = {
    "train": "ava_activespeaker_train_v1.0.tar.bz2",
    "val": "ava_activespeaker_val_v1.0.tar.bz2",
}

# Other annotation files the group uses. Small; fetched if missing.
#   ava_v2.2.zip              - AVA actions (context; not load-bearing)
#   ava_speech_labels_v1.csv  - AVA-Speech clean/music/noise labels, used by C3
#                               to gate the transcript and by C4 for its audio
#                               features. Fetched best-effort: if the name has
#                               changed upstream the script says so and carries on.
EXTRA_ANNOTATIONS = ["ava_v2.2.zip", "ava_speech_labels_v1.csv"]

# Published lists that map a video ID to its real filename. AVA ships mixed
# containers (.mp4, .mkv, .webm), so the extension cannot be guessed.
NAME_LISTS = [
    ("trainval", "ava_file_names_trainval_v2.1.txt"),
    ("trainval", "ava_speech_file_names_v1.txt"),
    ("test", "ava_file_names_test_v2.1.txt"),
]

TIMEOUT_S = 60          # a stalled connection fails instead of hanging forever
RETRIES = 4             # per file, with growing pauses between attempts
CHUNK = 1 << 20         # read 1 MB at a time

socket.setdefaulttimeout(TIMEOUT_S)

# ---------------------------------------------------------------------------
# small helpers


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def log(video_id, name, size, status):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "video_id", "file", "bytes", "status"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), video_id, name, size, status])


def remote_size(url):
    """True size of a file on the server, from an HTTP HEAD request.
    Returns None if the server will not say, which is not an error."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n is not None else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(url)
        return None
    except Exception:
        return None


def fetch(url, out, expected=None):
    """
    Download url to out, resuming any earlier partial attempt.

    Writes to out + '.part'. If that file already exists, asks the server for
    only the remaining bytes (Range request). If the server ignores the range
    and sends the whole file (status 200 instead of 206), starts again cleanly
    rather than appending a second copy - appending would silently corrupt it.
    """
    part = out + ".part"
    for attempt in range(1, RETRIES + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if expected is not None and have == expected:
            os.replace(part, out)
            return True, expected
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req) as r:
                if have and r.status != 206:      # server ignored the range
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                total = expected
                if total is None:
                    cl = r.headers.get("Content-Length")
                    total = (have + int(cl)) if cl is not None else None
                done = have
                last = time.time()
                with open(part, mode) as fh:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        fh.write(buf)
                        done += len(buf)
                        if time.time() - last > 1:
                            pct = f"{100 * done / total:5.1f}%" if total else ""
                            print(f"\r      {human(done):>10} {pct}", end="", flush=True)
                            last = time.time()
            print("\r" + " " * 40 + "\r", end="")
            size = os.path.getsize(part)
            if total is not None and size != total:
                raise IOError(f"got {size} bytes, expected {total}")
            os.replace(part, out)
            return True, size
        except FileNotFoundError:
            raise
        except urllib.error.HTTPError as e:
            if e.code == 416 and expected is not None and have == expected:
                os.replace(part, out)                   # it was already complete
                return True, expected
            if e.code == 404:
                return False, "not found on server (404)"
            err = f"HTTP {e.code}"
        except (OSError, IOError) as e:
            err = str(e)
        if attempt < RETRIES:
            wait = 5 * attempt
            print(f"\r      attempt {attempt} failed ({err}); retrying in {wait}s ...")
            time.sleep(wait)
    return False, err


def ensure(url, out, label):
    """
    Make sure `out` is a complete copy of `url`.
    Returns one of: 'ok', 'downloaded', 'resumed', 'unverified', or 'FAILED: ...'.
    """
    try:
        expected = remote_size(url)
    except FileNotFoundError:
        return "FAILED: not found on server (404)"

    if os.path.exists(out):
        have = os.path.getsize(out)
        if expected is None:
            return "unverified"                         # cannot check; leave it alone
        if have == expected:
            return "ok"
        if have < expected:
            # Truncated by an earlier interrupted download: turn it back into a
            # .part file and fetch only the missing tail.
            print(f"      {label}: truncated ({human(have)} of {human(expected)}) - resuming")
            os.replace(out, out + ".part")
            ok, info = fetch(url, out, expected)
            return "resumed" if ok else f"FAILED: {info}"
        return f"FAILED: larger than the server copy ({human(have)} > {human(expected)})"

    ok, info = fetch(url, out, expected)
    return "downloaded" if ok else f"FAILED: {info}"


# ---------------------------------------------------------------------------
# what we need


def get_annotations():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Annotations")
    for name in list(AS_ARCHIVES.values()) + EXTRA_ANNOTATIONS:
        out = os.path.join(OUT_DIR, name)
        status = ensure(f"{S3}/annotations/{name}", out, name)
        note = ""
        if status.startswith("FAILED") and name in EXTRA_ANNOTATIONS:
            note = "  (optional - check the AVA download page for the current name)"
        print(f"  {name:<44} {status}{note}")
    print()


def target_ids(split):
    """Video IDs that carry ActiveSpeaker labels, read from the archives'
    member names (<id>-activespeaker.csv). Nothing is extracted."""
    ids = {}
    for sp, archive in AS_ARCHIVES.items():
        if split not in ("all", sp):
            continue
        path = os.path.join(OUT_DIR, archive)
        if not os.path.exists(path):
            sys.exit(f"Missing {archive}. Run once without --list to fetch annotations.")
        with tarfile.open(path, "r:bz2") as tf:
            for m in tf.getmembers():
                base = os.path.basename(m.name)
                if base.endswith("-activespeaker.csv"):
                    ids[base[: -len("-activespeaker.csv")]] = sp
    return ids


def name_map():
    """video ID -> (folder on S3, filename with its real extension)."""
    mapping = {}
    for folder, listname in NAME_LISTS:
        cached = os.path.join(OUT_DIR, listname)
        if not os.path.exists(cached):
            try:
                urllib.request.urlretrieve(f"{S3}/annotations/{listname}", cached)
            except Exception as e:
                print(f"  (could not fetch {listname}: {e})")
                continue
        with open(cached) as fh:
            for line in fh:
                fn = line.strip()
                if fn:
                    vid = fn.rsplit(".", 1)[0]
                    mapping.setdefault(vid, (folder, fn))
    return mapping


def on_disk():
    if not os.path.isdir(VIDEO_DIR):
        return {}
    return {f.rsplit(".", 1)[0]: f for f in os.listdir(VIDEO_DIR)
            if not f.endswith(".part")}


# ---------------------------------------------------------------------------
# commands


def cmd_list(split):
    ids = target_ids(split)
    every = target_ids("all")
    have = on_disk()
    got = [i for i in ids if i in have]
    missing = [i for i in ids if i not in have]
    extra = [i for i in have if i not in every]
    size = sum(os.path.getsize(os.path.join(VIDEO_DIR, have[i])) for i in got)
    parts = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".part")] if os.path.isdir(VIDEO_DIR) else []
    print(f"ActiveSpeaker videos ({split}): {len(ids)}")
    print(f"  on disk   {len(got):4d}   {human(size)}")
    print(f"  missing   {len(missing):4d}")
    if parts:
        print(f"  partial   {len(parts):4d}   (re-run to finish)")
    if extra:
        print(f"  on disk but carrying no ActiveSpeaker labels: {', '.join(extra)}")
        print("    -> not needed for the features; keep only if C4 wants it for AVA-Speech")


def cmd_download(split, limit, verify_only):
    get_annotations()
    ids = target_ids(split)
    names = name_map()
    have = on_disk()

    unresolved = [i for i in ids if i not in names and i not in have]
    if unresolved:
        print(f"!! {len(unresolved)} labelled videos are in no published file list:")
        print("   " + ", ".join(sorted(unresolved)))
        print("   They cannot be downloaded by name; report them in the logbook.\n")

    # Files already on disk are verified first, then the missing ones fetched.
    present = [i for i in ids if i in have]
    missing = [i for i in ids if i not in have and i in names]
    queue = present if verify_only else present + missing[: (limit if limit else None)]

    print(f"{'Verifying' if verify_only else 'Checking'} {len(present)} on disk"
          + ("" if verify_only else f", then downloading {min(len(missing), limit or len(missing))}"
             f" of {len(missing)} missing") + "\n")

    tally = {}
    for n, vid in enumerate(queue, 1):
        if vid in have:
            fn = have[vid]
            folder = names.get(vid, ("trainval", fn))[0]
        else:
            folder, fn = names[vid]
        url = f"{S3}/{folder}/{fn}"
        out = os.path.join(VIDEO_DIR, fn)
        os.makedirs(VIDEO_DIR, exist_ok=True)
        print(f"  [{n}/{len(queue)}] {fn} ({ids[vid]})")
        status = ensure(url, out, fn)
        size = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"      {status}{'  ' + human(size) if size else ''}")
        log(vid, fn, size, status)
        key = status.split(":")[0]
        tally[key] = tally.get(key, 0) + 1

    print("\nSummary: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    print()
    cmd_list(split)


def main():
    p = argparse.ArgumentParser(description="AVA-ActiveSpeaker downloader (resumable).")
    p.add_argument("--list", action="store_true", help="report status only; download nothing")
    p.add_argument("--verify", action="store_true", help="check files already on disk; download nothing new")
    p.add_argument("--limit", type=int, default=0, help="download at most N missing videos this run")
    p.add_argument("--split", choices=["all", "train", "val"], default="all")
    a = p.parse_args()

    if a.list:
        cmd_list(a.split)
    else:
        cmd_download(a.split, a.limit, a.verify)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to resume where it left off.", file=sys.stderr)
