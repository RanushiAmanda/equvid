"""
Download AVA / AVA-ActiveSpeaker videos from the CVDF S3 mirror.

AVA publishes no download button for video -- only a plain-text list of file
names and a URL pattern. This script walks that list.

Videos are 15-minute movie segments. The full set of 160 is large, so download
in batches: set BATCH_START / BATCH_END, run, check disk usage, repeat.

Already-downloaded files are skipped, so the script is safe to re-run after an
interrupted connection.

Usage:
    python download_ava.py

Annotations are separate and tiny -- grab them by hand, see data/README.md.
"""

import os
import urllib.request

# --- configuration ---------------------------------------------------------

# Which slice of the file list to fetch this run.
# First run: 0, 20.  Then 20, 40.  Then 40, 60.  And so on.
BATCH_START = 0
BATCH_END = 20

# Where videos land: OUTSIDE the git repository, alongside it.
#   Research Project/
#   |-- AVA/      <-- here
#   `-- equvid/   <-- the repo, this script lives in equvid/data/
# Keeping corpora outside the repo means they cannot be committed by accident,
# even if someone runs `git add -A` from the wrong directory.
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "AVA")

FILE_LIST_URL = (
    "https://s3.amazonaws.com/ava-dataset/annotations/ava_speech_file_names_v1.txt"
)
VIDEO_URL = "https://s3.amazonaws.com/ava-dataset/trainval/{name}"

# ---------------------------------------------------------------------------


def human(n_bytes):
    """Bytes -> readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def folder_size(path):
    return sum(
        os.path.getsize(os.path.join(path, f))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # The file list is one name per line, e.g. "-5KQ66BBWC4.mkv"
    names = urllib.request.urlopen(FILE_LIST_URL).read().decode().split()
    print(f"{len(names)} videos in the full AVA list")

    batch = names[BATCH_START:BATCH_END]
    print(f"This run: items {BATCH_START} to {BATCH_END} ({len(batch)} files)\n")

    for i, name in enumerate(batch, 1):
        out = os.path.join(OUT_DIR, name)

        if os.path.exists(out):
            print(f"[{i}/{len(batch)}] {name} -- already have it, skipping")
            continue

        print(f"[{i}/{len(batch)}] {name} ...", end=" ", flush=True)
        try:
            # Download to a .part file first, then rename. Without this, an
            # interrupted download leaves a truncated file that looks complete
            # to the os.path.exists check above and gets silently skipped.
            tmp = out + ".part"
            urllib.request.urlretrieve(VIDEO_URL.format(name=name), tmp)
            os.replace(tmp, out)
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)

    total = folder_size(OUT_DIR)
    count = len([f for f in os.listdir(OUT_DIR) if not f.endswith(".part")])
    print(f"\n{OUT_DIR}: {count} files, {human(total)}")
    print(f"Next batch: set BATCH_START = {BATCH_END}, BATCH_END = {BATCH_END + 20}")


if __name__ == "__main__":
    main()
