"""
Download AMI meeting videos from a manifest exported by the Edinburgh chooser.

The AMI download page generates a wget script or a manifest of URLs. The wget
route needs wget on PATH, which is awkward on Windows. This script takes the
manifest instead.

To get the manifest:
    1. Go to https://groups.inf.ed.ac.uk/ami/download
    2. Select meetings (e.g. ES2002-ES2008, all of a b c d)
    3. Tick "Low-size DivX AVI videos"
    4. Press the button, then "copy manifest file to clipboard"
    5. Paste into data/ami_manifest.txt

Then:
    python download_ami.py

Why the camera filter matters: each meeting ships four Closeup cameras (one
face each) plus Corner and Overhead room views. Closeups are single-face fixed
shots -- useless for measuring visual prominence, and roughly two thirds of the
download. Filtering to room views cuts the transfer substantially.
"""

import os
import urllib.request
from urllib.parse import urlparse

# --- configuration ---------------------------------------------------------

MANIFEST = os.path.join(os.path.dirname(__file__), "ami_manifest.txt")

# Videos land OUTSIDE the git repository, alongside it -- see download_ava.py
# for the layout. Corpora outside the repo cannot be committed by accident.
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "AMI")

# Keep only these camera streams. Set to None to download everything.
KEEP_CAMERAS = ("Corner", "Overhead")

# ---------------------------------------------------------------------------


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    if not os.path.exists(MANIFEST):
        raise SystemExit(
            f"No manifest at {MANIFEST}\n"
            "Export one from https://groups.inf.ed.ac.uk/ami/download "
            "using 'copy manifest file to clipboard'."
        )

    urls = [u.strip() for u in open(MANIFEST) if u.strip().startswith("http")]
    print(f"{len(urls)} URLs in manifest")

    if KEEP_CAMERAS:
        urls = [u for u in urls if any(cam in u for cam in KEEP_CAMERAS)]
        print(f"{len(urls)} after keeping only {', '.join(KEEP_CAMERAS)}")

    for i, url in enumerate(urls, 1):
        # AMI URLs look like .../amicorpus/ES2002a/video/ES2002a.Corner.avi
        # Preserve that tree so the annotations line up with the signals.
        rel = urlparse(url).path.split("amicorpus/")[-1]
        out = os.path.join(OUT_DIR, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)

        if os.path.exists(out):
            print(f"[{i}/{len(urls)}] {rel} -- already have it, skipping")
            continue

        print(f"[{i}/{len(urls)}] {rel} ...", end=" ", flush=True)
        tmp = out + ".part"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, out)
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for root, _, files in os.walk(OUT_DIR)
        for f in files
    )
    print(f"\n{OUT_DIR}: {human(total)}")


if __name__ == "__main__":
    main()
