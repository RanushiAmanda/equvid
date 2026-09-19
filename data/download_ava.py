"""
Download AVA annotations and videos from the CVDF mirror.

AVA publishes no download button for video -- only a plain-text list of file
names and a URL pattern. This script walks that list, and also fetches the
ActiveSpeaker annotations, which are the part that actually makes AVA useful:

    entity_box  ->  face bounding box, already normalised (M1 features, no MTCNN)
    entity_id   ->  same person linked across frames (screen time, tracking)
    label       ->  SPEAKING_AND_AUDIBLE / SPEAKING_BUT_NOT_AUDIBLE /
                    NOT_SPEAKING, per frame (M2 speaking duration, no VAD)

Videos land OUTSIDE the git repository, alongside it:

    Research Project/
    |-- AVA/            <-- here
    `-- equvid/         <-- the repo; this script is equvid/data/download_ava.py

Corpora outside the repo cannot be committed by accident, even if someone runs
`git add -A` from the wrong directory.

Usage
-----
    python download_ava.py                 # annotations + top up to 60 videos
    python download_ava.py --target 100    # ... up to 100 videos
    python download_ava.py --annotations   # annotations only, no video
    python download_ava.py --list          # show what is present, download nothing

Safe to re-run. Completed files are skipped; interrupted downloads are written
to a .part file and only renamed on success, so a dropped connection never
leaves a truncated file that looks complete.
"""

import argparse
import os
import sys
import urllib.request

# --- configuration ---------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "AVA")
VIDEO_DIR = os.path.join(OUT_DIR, "ava_videos")

S3 = "https://s3.amazonaws.com/ava-dataset"
FILE_LIST_URL = f"{S3}/annotations/ava_speech_file_names_v1.txt"
VIDEO_URL = S3 + "/trainval/{name}"

# Small, and the single most valuable download in the corpus.
ANNOTATIONS = [
    f"{S3}/annotations/ava_activespeaker_train_v1.0.tar.bz2",
    f"{S3}/annotations/ava_activespeaker_val_v1.0.tar.bz2",
    f"{S3}/annotations/ava_v2.2.zip",
]

DEFAULT_TARGET = 60

# ---------------------------------------------------------------------------


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def tree_size(path):
    if not os.path.isdir(path):
        return 0
    return sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(path)
        for f in files
        if not f.endswith(".part")
    )


def fetch(url, out):
    """
    Download one file. Writes to <out>.part and renames on success.

    Without the .part dance, a connection dropped mid-transfer leaves a
    truncated file that the os.path.exists check would treat as complete on the
    next run, and the corruption stays hidden until feature extraction fails.
    """
    tmp = out + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, out)
        return True, os.path.getsize(out)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, str(exc)


def get_annotations():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Annotations")
    for url in ANNOTATIONS:
        name = url.rsplit("/", 1)[-1]
        out = os.path.join(OUT_DIR, name)
        if os.path.exists(out):
            print(f"  {name:<45} have it ({human(os.path.getsize(out))})")
            continue
        print(f"  {name:<45} downloading ...", end=" ", flush=True)
        ok, info = fetch(url, out)
        print(f"done ({human(info)})" if ok else f"FAILED: {info}")
    print()


def video_names():
    return urllib.request.urlopen(FILE_LIST_URL).read().decode().split()


def have_which(names):
    """AVA ships mixed containers, so match on the name as published."""
    present = set(os.listdir(VIDEO_DIR)) if os.path.isdir(VIDEO_DIR) else set()
    have = [n for n in names if n in present]
    missing = [n for n in names if n not in present]
    return have, missing


def get_videos(target):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    names = video_names()
    have, missing = have_which(names)

    print(f"Videos: {len(have)} of {len(names)} present, target {target}")
    todo = missing[: max(0, target - len(have))]
    if not todo:
        print("  target already met, nothing to do\n")
        return

    print(f"  fetching {len(todo)} more\n")
    for i, name in enumerate(todo, 1):
        out = os.path.join(VIDEO_DIR, name)
        print(f"  [{i}/{len(todo)}] {name} ...", end=" ", flush=True)
        ok, info = fetch(url=VIDEO_URL.format(name=name), out=out)
        print(f"{human(info)}" if ok else f"FAILED: {info}")


def report():
    names = video_names()
    have, _ = have_which(names)
    ann = [
        a.rsplit("/", 1)[-1]
        for a in ANNOTATIONS
        if os.path.exists(os.path.join(OUT_DIR, a.rsplit("/", 1)[-1]))
    ]
    print(f"\n  videos      {len(have)} / {len(names)}")
    print(f"  annotations {len(ann)} / {len(ANNOTATIONS)}  {ann}")
    print(f"  disk        {human(tree_size(OUT_DIR))}")
    stale = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".part")] if os.path.isdir(VIDEO_DIR) else []
    if stale:
        print(f"  !! {len(stale)} interrupted .part files -- re-run to finish them")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=DEFAULT_TARGET,
                   help=f"total videos to end up with (default {DEFAULT_TARGET})")
    p.add_argument("--annotations", action="store_true", help="annotations only")
    p.add_argument("--list", action="store_true", help="report status, download nothing")
    args = p.parse_args()

    if args.list:
        report()
        return

    get_annotations()
    if not args.annotations:
        get_videos(args.target)
    report()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted -- re-run to resume", file=sys.stderr)
