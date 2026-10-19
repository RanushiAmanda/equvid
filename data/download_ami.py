"""
Download the AMI Meeting Corpus video (and optionally audio) the group needs,
driven by the metadata AMI already ships - no download-form manifest required.

WHY THIS REPLACES THE OLD SCRIPT
--------------------------------
The old script needed a manifest exported by hand from the Edinburgh form, one
batch at a time. The URL scheme is regular, and the full list of meetings with
their participants is already on disk inside ami_public_manual_1.6.2.zip, so
this script generates the URLs itself.

WHICH MEETINGS - AND WHY NOT ALL OF THEM BY DEFAULT
---------------------------------------------------
EuqVid measures disparity BETWEEN genders present in the same material. A
meeting series in which all four participants share one gender has nothing to
compare, so it can never produce a disparity score, whatever we extract from it.

AMI's own participant metadata shows that this is common:

    35 scenario series (ES, IS, TS; 4 meetings each, same 4 people throughout)
    16 mixed-gender    -> usable
    19 single-gender   -> 14 all-male, 5 all-female
    TS series: 9 of 10 are all-male

So by default this script downloads only the mixed-gender series. That is 16
series rather than 35: every byte downloaded is usable, and the rest of the
corpus is one flag away (--all) if the group decides it wants it.

WHICH STREAMS
-------------
Video: the room-view cameras, whose names differ by recording site:
    ES (Edinburgh)  Corner, Overhead
    IS (Idiap)      C, L, R
    TS (TNO)        Overview1, Overview2
The four Closeup cameras are single-face fixed shots - the framing the
supervisor ruled out - and most of the download.
Audio (--audio): the headset mix, Mix-Headset.wav, which the audio component
needs. The video files do not carry it.

The transcripts C3 needs are NOT downloaded here: word-level transcripts for
all 171 AMI meetings are already inside ami_public_manual_1.6.2.zip.

GENDER METADATA GOTCHA
----------------------
participants.xml has no entries for the 40 TNO (TS) speakers at all; they
appear only in meetings.xml, with the role glued on ("MTD009PM"). Their only
gender information is the ID's leading F/M letter, which matches the recorded
sex for all 189 participants that do have an entry. This script uses that
letter as a fallback, so TS genders are id-prefix labels, NOT verified ones;
the group gender map must keep them flagged as such.

USAGE
-----
    python download_ami.py --list            # per-series gender mix and what is on disk
    python download_ami.py                   # video for every mixed-gender series still missing
    python download_ami.py --audio           # ... plus headset-mix audio for the same meetings
    python download_ami.py --audio-only      # audio only (e.g. for meetings whose video you have)
    python download_ami.py --verify          # check files already on disk against the server
    python download_ami.py --all             # every scenario series, single-gender included
    python download_ami.py --series ES2009,IS1000

Safe to stop with Ctrl+C and re-run; downloads resume where they stopped.
Standard library only.
"""

import argparse
import collections
import csv
import datetime as dt
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "AMI"))
CORPUS = os.path.join(OUT_DIR, "amicorpus")
ANN_ZIP = os.path.join(OUT_DIR, "ami_public_manual_1.6.2.zip")
LOG = os.path.join(OUT_DIR, "download_log.csv")

MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus"
ANN_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
DOME_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/dome_annotations_M1.csv"

# Room-view cameras per recording site, keyed by the first letter of the
# meeting ID. Checked against the mirror's directory listings (Oct 2026) for
# ES, IS and TS. The first version asked every site for Corner/Overhead, which
# exist only in Edinburgh, so every IS and TS file came back 404.
ROOM_VIEWS = {
    "E": ("Corner", "Overhead"),       # Edinburgh
    "I": ("C", "L", "R"),              # Idiap
    "T": ("Overview1", "Overview2"),   # TNO
}
AUDIO_STREAM = "Mix-Headset"

NS = "{http://nite.sourceforge.net/}"
TIMEOUT_S = 60
RETRIES = 4
CHUNK = 1 << 20
socket.setdefaulttimeout(TIMEOUT_S)

# ---------------------------------------------------------------------------
# download helpers (identical to download_ava.py so the two behave the same)


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def log(item, size, status):
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "file", "bytes", "status"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), item, size, status])


def remote_size(url):
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
    """Resumable download via <out>.part and HTTP Range; see download_ava.py."""
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
                if have and r.status != 206:
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                total = expected
                if total is None:
                    cl = r.headers.get("Content-Length")
                    total = (have + int(cl)) if cl is not None else None
                done, last = have, time.time()
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
                os.replace(part, out)
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
    try:
        expected = remote_size(url)
    except FileNotFoundError:
        return "FAILED: not found on server (404)"
    if os.path.exists(out):
        have = os.path.getsize(out)
        if expected is None:
            return "unverified"
        if have == expected:
            return "ok"
        if have < expected:
            print(f"      {label}: truncated ({human(have)} of {human(expected)}) - resuming")
            os.replace(out, out + ".part")
            ok, info = fetch(url, out, expected)
            return "resumed" if ok else f"FAILED: {info}"
        return f"FAILED: larger than the server copy ({human(have)} > {human(expected)})"
    ok, info = fetch(url, out, expected)
    return "downloaded" if ok else f"FAILED: {info}"


# ---------------------------------------------------------------------------
# metadata


def load_meetings():
    """
    From the shipped annotation zip: every meeting, its type, duration, and the
    sex of each participant. Returns (meetings, series) where
        meetings[obs] = {"type", "duration", "sexes": [..]}
        series[name]  = {"F": n, "M": n, "meetings": [obs, ...]}
    """
    if not os.path.exists(ANN_ZIP):
        print("Annotation zip missing - fetching it first (23 MB)")
        status = ensure(ANN_URL, ANN_ZIP, "annotations")
        if status.startswith("FAILED"):
            sys.exit(f"Could not get {ANN_URL}: {status}")
    with zipfile.ZipFile(ANN_ZIP) as z:
        parts = ET.fromstring(z.read("corpusResources/participants.xml"))
        meets = ET.fromstring(z.read("corpusResources/meetings.xml"))

    sex_of = {p.get(NS + "id"): p.get("sex") for p in parts.iter("participant")}

    def sex(global_name):
        if global_name in sex_of:
            return sex_of[global_name]
        bare = re.sub(r"(PM|ME|UID|UI|ID)$", "", global_name)   # harmless; TS ids are not in participants.xml
        if bare in sex_of:
            return sex_of[bare]
        return global_name[0] if global_name[:1] in ("F", "M") else "?"

    meetings, series = {}, collections.defaultdict(lambda: {"F": set(), "M": set(), "meetings": []})
    for m in meets.iter("meeting"):
        obs = m.get("observation")
        speakers = [(s.get("global_name"), sex(s.get("global_name"))) for s in m.iter("speaker")]
        meetings[obs] = {"type": m.get("type"), "duration": float(m.get("duration") or 0),
                         "speakers": speakers}
        if m.get("type") == "scenario":
            sr = series[obs[:6]]
            sr["meetings"].append(obs)
            for gn, g in speakers:
                if g in ("F", "M"):
                    sr[g].add(re.sub(r"(PM|ME|UID|UI|ID)$", "", gn))
    for sr in series.values():
        sr["meetings"].sort()
        sr["F"], sr["M"] = len(sr["F"]), len(sr["M"])
    return meetings, dict(series)


def files_for(obs, video, audio):
    out = []
    if video:
        for cam in ROOM_VIEWS[obs[0]]:
            fn = f"{obs}.{cam}.avi"
            out.append((f"{MIRROR}/{obs}/video/{fn}", os.path.join(CORPUS, obs, "video", fn)))
    if audio:
        fn = f"{obs}.{AUDIO_STREAM}.wav"
        out.append((f"{MIRROR}/{obs}/audio/{fn}", os.path.join(CORPUS, obs, "audio", fn)))
    return out


# ---------------------------------------------------------------------------
# commands


def cmd_list(meetings, series):
    print(f"{'series':<8} {'F':>2} {'M':>2}  {'usable':<7} {'video on disk':<14} {'audio on disk'}")
    totals = collections.Counter()
    for name in sorted(series):
        sr = series[name]
        usable = sr["F"] > 0 and sr["M"] > 0
        v = sum(all(os.path.exists(o) for _, o in files_for(m, True, False)) for m in sr["meetings"])
        a = sum(all(os.path.exists(o) for _, o in files_for(m, False, True)) for m in sr["meetings"])
        n = len(sr["meetings"])
        print(f"{name:<8} {sr['F']:>2} {sr['M']:>2}  {'yes' if usable else 'no':<7} "
              f"{v}/{n:<12} {a}/{n}")
        totals["series"] += 1
        totals["usable"] += usable
        totals["usable_video_done"] += usable and v == n
    print(f"\n{totals['usable']} of {totals['series']} scenario series are mixed-gender; "
          f"{totals['usable_video_done']} of those have complete room-view video on disk.")
    print("Transcripts for all meetings are already in ami_public_manual_1.6.2.zip.")


def cmd_download(meetings, series, wanted, video, audio, verify_only):
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Annotations")
    for url, out in [(DOME_URL, os.path.join(OUT_DIR, os.path.basename(DOME_URL)))]:
        print(f"  {os.path.basename(out):<36} {ensure(url, out, out)}")
    print()

    obs_list = [m for s in sorted(wanted) for m in series[s]["meetings"]]
    jobs = [(u, o) for m in obs_list for u, o in files_for(m, video, audio)]
    if verify_only:
        jobs = [(u, o) for u, o in jobs if os.path.exists(o)]
    hours = sum(meetings[m]["duration"] for m in obs_list) / 3600
    print(f"{len(wanted)} series, {len(obs_list)} meetings ({hours:.1f} h), {len(jobs)} files"
          f"{' already on disk to verify' if verify_only else ''}\n")

    tally = collections.Counter()
    for n, (url, out) in enumerate(jobs, 1):
        rel = os.path.relpath(out, OUT_DIR)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f"  [{n}/{len(jobs)}] {rel}")
        status = ensure(url, out, rel)
        size = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"      {status}{'  ' + human(size) if size else ''}")
        log(rel, size, status)
        tally[status.split(":")[0]] += 1
    print("\nSummary: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())))


def main():
    p = argparse.ArgumentParser(description="AMI downloader driven by shipped metadata (resumable).")
    p.add_argument("--list", action="store_true", help="per-series gender mix and disk status; no download")
    p.add_argument("--verify", action="store_true", help="check files on disk against the server only")
    p.add_argument("--all", action="store_true", help="every scenario series, single-gender included")
    p.add_argument("--series", help="comma-separated series, e.g. ES2009,IS1000")
    p.add_argument("--audio", action="store_true", help="also fetch headset-mix audio")
    p.add_argument("--audio-only", action="store_true", help="fetch headset-mix audio, no video")
    a = p.parse_args()

    meetings, series = load_meetings()
    if a.list:
        cmd_list(meetings, series)
        return

    if a.series:
        wanted = [s.strip() for s in a.series.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in series]
        if unknown:
            sys.exit(f"Unknown series: {', '.join(unknown)}")
    elif a.all:
        wanted = list(series)
    else:
        wanted = [s for s, sr in series.items() if sr["F"] > 0 and sr["M"] > 0]

    video = not a.audio_only
    audio = a.audio or a.audio_only
    cmd_download(meetings, series, wanted, video, audio, a.verify)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to resume where it left off.", file=sys.stderr)
