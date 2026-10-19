"""
C1 step 4b — cut the face crops for the blind validation sample.

Runs on the machine that holds the AVA videos.  Reads sample_manifest.csv,
seeks each video to the recorded timestamp, crops the annotated box with a
margin, and writes one JPEG per sample plus a small labelling page.

WHY A MARGIN
------------
A tight face box is hard to judge.  Hair, shoulders and a little context
are what a human actually uses, and the question being asked is what a
human would say -- so the crop is expanded by MARGIN on each side.  The
crop is only the annotator's view; no measurement is taken from it.

Requires: opencv-python  (already installed if pilot_meld.py ran)

Usage:
  python c1_make_crops.py sample_manifest.csv <videos_dir> <out_dir>
"""

import csv
import os
import sys

import cv2

MARGIN = 0.6      # expand the box by 60% of its size on each axis
OUT_PX = 320      # longest side of the saved crop


def find_video(videos_dir, vid):
    for ext in (".mp4", ".mkv", ".webm", ".avi"):
        p = os.path.join(videos_dir, vid + ext)
        if os.path.exists(p):
            return p
    return None


def main(manifest, videos_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))

    # Group by video so each file is opened once, not once per crop.
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)

    done, failed = 0, []
    for vid, items in sorted(by_video.items()):
        path = find_video(videos_dir, vid)
        if not path:
            failed.extend(r["sample_id"] for r in items)
            print("MISSING VIDEO: %s (%d samples skipped)" % (vid, len(items)))
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            failed.extend(r["sample_id"] for r in items)
            print("CANNOT OPEN : %s" % path)
            continue
        print("%s  (%d crops)" % (vid, len(items)))
        for r in sorted(items, key=lambda x: float(x["timestamp"])):
            t = float(r["timestamp"])
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                failed.append(r["sample_id"])
                continue
            H, W = frame.shape[:2]
            x1, y1 = float(r["x1"]) * W, float(r["y1"]) * H
            x2, y2 = float(r["x2"]) * W, float(r["y2"]) * H
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0, int(x1 - MARGIN * bw))
            y1 = max(0, int(y1 - MARGIN * bh))
            x2 = min(W, int(x2 + MARGIN * bw))
            y2 = min(H, int(y2 + MARGIN * bh))
            if x2 <= x1 or y2 <= y1:
                failed.append(r["sample_id"])
                continue
            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]
            s = OUT_PX / float(max(h, w))
            if s < 1:
                crop = cv2.resize(crop, (int(w * s), int(h * s)),
                                  interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, r["sample_id"] + ".jpg"), crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            done += 1
        cap.release()

    ids = [r["sample_id"] for r in rows
           if os.path.exists(os.path.join(out_dir, r["sample_id"] + ".jpg"))]
    write_page(os.path.join(out_dir, "label.html"), ids)

    print()
    print("crops written : %d" % done)
    print("failed        : %d" % len(failed))
    if failed[:10]:
        print("  e.g. %s" % ", ".join(failed[:10]))
    print()
    print("Open %s in Chrome, label every crop, then press" % os.path.join(out_dir, "label.html"))
    print("'Download CSV' and keep the file as manual_labels.csv")


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>C1 validation labelling</title>
<style>
 :root{--bg:#fff;--fg:#111;--line:#d8d8d8;--chip:#f3f3f3}
 @media (prefers-color-scheme:dark){:root:not([data-theme=light]){
   --bg:#161616;--fg:#eee;--line:#333;--chip:#242424}}
 body{margin:0;padding:16px;background:var(--bg);color:var(--fg);
   font:15px/1.5 system-ui,sans-serif}
 header{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;
   border-bottom:1px solid var(--line);z-index:5}
 h1{font-size:18px;margin:0 0 4px}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 button{font:inherit;padding:7px 14px;border:1px solid var(--line);
   border-radius:7px;background:var(--chip);color:var(--fg);cursor:pointer}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
   gap:14px;margin-top:16px}
 figure{margin:0;border:1px solid var(--line);border-radius:9px;
   overflow:hidden;background:var(--chip)}
 img{width:100%;height:190px;object-fit:contain;background:#000;display:block}
 .ctl{display:flex}
 .ctl button{flex:1;border:0;border-top:1px solid var(--line);border-radius:0;
   background:transparent;padding:9px 0;font-size:14px}
 .ctl button:not(:last-child){border-right:1px solid var(--line)}
 .ctl button.on{background:var(--fg);color:var(--bg);font-weight:600}
 figcaption{font-size:11px;padding:4px 7px;opacity:.6}
</style></head><body>
<header>
 <h1>C1 validation &mdash; label each face</h1>
 <div class="bar">
  <strong id="count">0 / 0</strong>
  <button id="dl">Download CSV</button>
  <span style="opacity:.65;font-size:13px">F = woman &middot; M = man &middot;
   U = cannot tell. Keyboard: click a crop then press f, m or u.</span>
 </div>
</header>
<div id="grid"></div>
<script>
const IDS = __IDS__;
const picked = {};
try{Object.assign(picked, JSON.parse(localStorage.getItem('c1labels')||'{}'));}
catch(e){}
const grid = document.getElementById('grid');
function save(){ try{localStorage.setItem('c1labels',JSON.stringify(picked));}
  catch(e){} tally(); }
function tally(){ document.getElementById('count').textContent =
  Object.keys(picked).length + ' / ' + IDS.length; }
IDS.forEach(id=>{
  const f=document.createElement('figure'); f.tabIndex=0;
  f.innerHTML='<img loading="lazy" src="'+id+'.jpg" alt="">'
   +'<div class="ctl"></div><figcaption>'+id+'</figcaption>';
  const ctl=f.querySelector('.ctl');
  ['F','M','U'].forEach(v=>{
    const b=document.createElement('button'); b.textContent=v;
    b.onclick=()=>{picked[id]=v;paint(f,id);save();};
    ctl.appendChild(b);
  });
  f.addEventListener('keydown',e=>{
    const k=e.key.toLowerCase();
    if('fmu'.includes(k)){picked[id]=k.toUpperCase();paint(f,id);save();}
  });
  grid.appendChild(f); paint(f,id);
});
function paint(f,id){
  f.querySelectorAll('.ctl button').forEach(b=>
    b.classList.toggle('on', picked[id]===b.textContent));
}
tally();
document.getElementById('dl').onclick=()=>{
  let out='sample_id,manual_gender\\n';
  IDS.forEach(id=>{ if(picked[id]) out+=id+','+picked[id]+'\\n'; });
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([out],{type:'text/csv'}));
  a.download='manual_labels.csv'; a.click();
};
</script></body></html>
"""


def write_page(path, ids):
    import json
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(PAGE.replace("__IDS__", json.dumps(ids)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
