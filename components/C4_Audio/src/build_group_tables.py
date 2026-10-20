"""
C4 — build gender_map.csv and clips.csv for AMI, in M2's shared schema.

WHY THIS EXISTS
---------------
M2's conventions document specifies two shared files that every component joins
against:

    gender_map.csv   person_id, gender, gender_source, component
    clips.csv        clip_id, corpus, meeting, series, component, n_F, n_M,
                     n_rows, mixed_gender

M2 has already generated them, but they are not on my machine. Rather than wait,
this script regenerates the AMI half from AMI's own shipped metadata, using M2's
exact column names and ID formats. When M2's copy arrives the two reconcile
row-for-row, and any disagreement is itself worth knowing about.

Everything here comes from ami_public_manual_1.6.2.zip, which
download_ami.py fetches automatically. No network access, no teammate needed.

ID FORMATS (M2's conventions doc, "Shared IDs")
----------------------------------------------
    corpus      AMI
    clip_id     AMI:ES2002a          corpus:unit, unit = one meeting
    person_id   AMI:FEE005           corpus:participant code (global_name)
    series      ES2002               groups the four meetings of one team
    component   AMI_019              person-connected group; meetings that
                                     share a person are linked

The seat letters A-D are NOT persons: the same letter maps to different people
in different meetings. That is why person_id uses global_name.

THE TS GOTCHA
-------------
meetings.xml names TS speakers with the role glued on ("MTD009PM") while
participants.xml uses the bare ID ("MTD009"). A naive join silently drops all
40 TS speakers. We strip the role suffix, then fall back to the ID's leading
F/M letter. download_ami.py applies the same rule; keep them identical.

USAGE
-----
    python build_group_tables.py
    python build_group_tables.py --ami-dir ../../AMI --out-dir output
"""

import argparse
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

NS = "{http://nite.sourceforge.net/}"
ROLE_SUFFIX = re.compile(r"(PM|ME|UID|UI|ID)$")


def find_ami_dir(start):
    """Walk up from the script looking for the AMI corpus folder.

    The repo layout puts corpora BESIDE the repo, not inside it:

        Research Project/
        |-- equvid/                        <- the git repo
        |   |-- data/download_ami.py
        |   `-- components/C4_Audio/src/   <- this script
        |-- AMI/                           <- what we are looking for
        |-- MELD/
        `-- AVA/

    Searching upward means the script works wherever you put it, instead of
    depending on a hard-coded number of '..' steps.
    """
    d = os.path.abspath(start)
    for _ in range(8):
        cand = os.path.join(d, "AMI")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # Not found: return the conventional location so the error message names a
    # real path rather than something relative and confusing.
    return os.path.normpath(os.path.join(start, "..", "..", "..", "..", "AMI"))


def strip_role(global_name):
    """'MTD009PM' -> 'MTD009'. Leaves non-TS names untouched."""
    return ROLE_SUFFIX.sub("", global_name)


def load_metadata(ann_zip):
    """Return (sex_of, meetings) read from the shipped annotation zip."""
    if not os.path.exists(ann_zip):
        sys.exit(
            f"Annotation zip not found:\n  {ann_zip}\n\n"
            "Run this first, from equvid/data/:\n"
            "    python download_ami.py --list\n"
            "It fetches the zip (23 MB) automatically."
        )
    with zipfile.ZipFile(ann_zip) as z:
        parts = ET.fromstring(z.read("corpusResources/participants.xml"))
        meets = ET.fromstring(z.read("corpusResources/meetings.xml"))

    # participants.xml: <participant nite:id="FEE005" sex="F" age=".."/>
    sex_of = {p.get(NS + "id"): p.get("sex") for p in parts.iter("participant")}

    # meetings.xml: <meeting observation="ES2002a" type="scenario" duration="..">
    #                 <speaker nxt_agent="A" global_name="FEE005" role=".."/>
    meetings = {}
    for m in meets.iter("meeting"):
        obs = m.get("observation")
        if not obs:
            continue
        speakers = []
        for s in m.iter("speaker"):
            gname = s.get("global_name")
            if not gname:
                continue
            # The seat letter attribute is 'nxt_agent' in AMI's own files, but
            # accept the alternatives rather than silently dropping every row
            # if a future release renames it.
            seat = s.get("nxt_agent") or s.get("agent") or s.get("channel") or ""
            speakers.append({"global_name": gname, "seat": seat,
                             "role": s.get("role") or ""})
        meetings[obs] = {"type": m.get("type"),
                         "duration": float(m.get("duration") or 0),
                         "speakers": speakers}
    return sex_of, meetings


def resolve_sex(global_name, sex_of):
    """Three-step resolution; returns (gender, source)."""
    if global_name in sex_of and sex_of[global_name] in ("F", "M"):
        return sex_of[global_name], "participants_xml"
    bare = strip_role(global_name)
    if bare in sex_of and sex_of[bare] in ("F", "M"):
        return sex_of[bare], "participants_xml_stripped"
    if global_name[:1] in ("F", "M"):
        return global_name[0], "id_prefix_fallback"
    return "", "unresolved"


def connected_components(meetings):
    """
    Group meetings that share at least one person. M2 calls these 'components'
    and builds cross-validation folds from them, so that one person never
    appears in both training and test.

    Union-find over (meeting, person) membership.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for obs, m in meetings.items():
        key_m = ("meeting", obs)
        find(key_m)
        for s in m["speakers"]:
            key_p = ("person", strip_role(s["global_name"]))
            union(key_m, key_p)

    # Name each component deterministically: AMI_000, AMI_001, ... ordered by
    # the alphabetically first meeting it contains, so reruns are identical.
    roots = {}
    for obs in meetings:
        roots.setdefault(find(("meeting", obs)), []).append(obs)
    ordered = sorted(roots.items(), key=lambda kv: min(kv[1]))
    comp_of_meeting, comp_of_person = {}, {}
    for i, (root, obs_list) in enumerate(ordered):
        name = f"AMI_{i:03d}"
        for obs in obs_list:
            comp_of_meeting[obs] = name
            for s in meetings[obs]["speakers"]:
                comp_of_person[strip_role(s["global_name"])] = name
    return comp_of_meeting, comp_of_person


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Build AMI gender_map.csv and clips.csv in M2's schema.")
    ap.add_argument("--ami-dir", default=find_ami_dir(here),
                    help="folder holding ami_public_manual_1.6.2.zip (default: ../../../AMI)")
    ap.add_argument("--out-dir", default=os.path.normpath(os.path.join(here, "..", "output")),
                    help="where to write the CSVs (default: ./output)")
    ap.add_argument("--scenario-only", action="store_true",
                    help="keep only scenario meetings (ES/IS/TS), excluding the natural ones")
    a = ap.parse_args()

    ann_zip = os.path.join(a.ami_dir, "ami_public_manual_1.6.2.zip")
    os.makedirs(a.out_dir, exist_ok=True)

    sex_of, meetings = load_metadata(ann_zip)
    if a.scenario_only:
        meetings = {k: v for k, v in meetings.items() if v["type"] == "scenario"}

    comp_of_meeting, comp_of_person = connected_components(meetings)

    # ---- gender_map.csv -------------------------------------------------
    people = {}
    unresolved = []
    for obs, m in meetings.items():
        for s in m["speakers"]:
            gname = strip_role(s["global_name"])
            if gname in people:
                continue
            gender, source = resolve_sex(s["global_name"], sex_of)
            if not gender:
                unresolved.append(gname)
            people[gname] = {"person_id": f"AMI:{gname}",
                             "gender": gender,
                             "gender_source": source,
                             "component": comp_of_person.get(gname, "")}

    gm_path = os.path.join(a.out_dir, "gender_map_ami.csv")
    with open(gm_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["person_id", "gender", "gender_source", "component"])
        w.writeheader()
        for gname in sorted(people):
            w.writerow(people[gname])

    # ---- clips.csv ------------------------------------------------------
    rows = []
    for obs in sorted(meetings):
        m = meetings[obs]
        genders = [people[strip_role(s["global_name"])]["gender"] for s in m["speakers"]]
        n_f = sum(g == "F" for g in genders)
        n_m = sum(g == "M" for g in genders)
        rows.append({
            "clip_id": f"AMI:{obs}",
            "corpus": "AMI",
            "meeting": obs,
            "series": obs[:6],
            "component": comp_of_meeting.get(obs, ""),
            "n_F": n_f,
            "n_M": n_m,
            "n_rows": len(m["speakers"]),
            "mixed_gender": int(n_f > 0 and n_m > 0),
            "duration_s": round(m["duration"], 1),
            "type": m["type"],
        })

    clips_path = os.path.join(a.out_dir, "clips_ami.csv")
    with open(clips_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- report ---------------------------------------------------------
    mixed = [r for r in rows if r["mixed_gender"]]
    n_f_people = sum(p["gender"] == "F" for p in people.values())
    n_m_people = sum(p["gender"] == "M" for p in people.values())
    by_source = {}
    for p in people.values():
        by_source[p["gender_source"]] = by_source.get(p["gender_source"], 0) + 1

    print(f"Wrote {gm_path}")
    print(f"Wrote {clips_path}")
    print()
    print(f"  people              {len(people)}   ({n_f_people} F, {n_m_people} M, "
          f"{len(unresolved)} unresolved)")
    print(f"  meetings            {len(rows)}")
    print(f"  mixed-gender        {len(mixed)}  <- the usable set for EuqVid")
    print(f"  components          {len(set(comp_of_meeting.values()))}  "
          f"<- independent units for folds, NOT meetings")
    print(f"  usable hours        {sum(r['duration_s'] for r in mixed) / 3600:.1f}")
    print()
    print("  gender source breakdown:")
    for k in sorted(by_source):
        print(f"    {k:<28} {by_source[k]}")
    if unresolved:
        print(f"\n  UNRESOLVED (report these): {', '.join(sorted(set(unresolved))[:10])}")
    print()
    print("  Note for the thesis: AMI ships binary F/M labels only. Other gender")
    print("  identities are not represented. This is a declared limitation.")


if __name__ == "__main__":
    main()
