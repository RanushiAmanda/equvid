import os
import base64
import json
import pandas as pd
import cv2

ROOT = os.environ.get('EUQ_ROOT', os.path.expanduser('~/Downloads/euqvid_temporal'))
D = ROOT + '/outputs/ava_gender'
V = D + '/verify'

key = pd.read_csv(V + '/verify_key.csv')[['id', 'file']]
items = []
for _, r in key.iterrows():
    im = cv2.imread(D + '/crops/' + r.file)
    if im is None:
        continue
    h, w = im.shape[:2]
    s = 320.0 / max(h, w)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
    ok, buf = cv2.imencode('.jpg', im, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    items.append({'id': int(r['id']), 'img': 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()})
print('faces embedded:', len(items))

HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>Face labelling</title>
<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#f4f4f4;margin:0;text-align:center}
#box{max-width:520px;margin:20px auto;padding:16px}
#num{font-size:22px;font-weight:bold;color:#b00}
img{max-width:360px;max-height:360px;border:2px solid #333;background:#000;margin:12px 0}
button{font-size:22px;padding:14px 26px;margin:6px;border-radius:8px;border:1px solid #444;cursor:pointer}
#bF{background:#ffd6e7}#bM{background:#cfe3ff}#bU{background:#e5e5e5}
#cur{font-size:18px;min-height:26px}
.small{font-size:14px;color:#444}
</style></head><body><div id="box">
<div id="num"></div>
<img id="face" alt="face">
<div id="cur"></div>
<div>
<button id="bF">F (woman)</button><button id="bM">M (man)</button><button id="bU">U (cannot tell)</button>
</div>
<p class="small">Keys: F, M, U to answer. Left arrow = go back. Judge only the face you see.<br>U = too blurry, too small, or you really cannot tell.</p>
<div><button id="back">Back</button><button id="dl">Download CSV</button></div>
<p id="prog" class="small"></p>
</div>
<script>
var ITEMS = __ITEMS__;
var labels = {};
var pos = 0;
try { var saved = localStorage.getItem('ava_verify_labels'); if (saved) { labels = JSON.parse(saved); } } catch (e) {}
function save() { try { localStorage.setItem('ava_verify_labels', JSON.stringify(labels)); } catch (e) {} }
function firstUnlabelled() {
  for (var i = 0; i < ITEMS.length; i++) { if (!labels[ITEMS[i].id]) { return i; } }
  return ITEMS.length - 1;
}
function show() {
  var it = ITEMS[pos];
  document.getElementById('num').textContent = 'Face ' + it.id + ' of ' + ITEMS.length;
  document.getElementById('face').src = it.img;
  document.getElementById('cur').textContent = labels[it.id] ? 'your answer: ' + labels[it.id] : '';
  var n = Object.keys(labels).length;
  document.getElementById('prog').textContent = 'labelled ' + n + ' / ' + ITEMS.length + (n === ITEMS.length ? '  -  all done, press Download CSV' : '');
}
function answer(v) {
  labels[ITEMS[pos].id] = v;
  save();
  if (pos < ITEMS.length - 1) { pos++; }
  show();
}
function back() { if (pos > 0) { pos--; } show(); }
function download() {
  var nl = String.fromCharCode(10);
  var out = 'id,true_gender' + nl;
  for (var i = 0; i < ITEMS.length; i++) { out += ITEMS[i].id + ',' + (labels[ITEMS[i].id] || '') + nl; }
  var a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([out], {type: 'text/csv'}));
  a.download = 'verify_filled.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}
document.getElementById('bF').onclick = function () { answer('F'); };
document.getElementById('bM').onclick = function () { answer('M'); };
document.getElementById('bU').onclick = function () { answer('U'); };
document.getElementById('back').onclick = back;
document.getElementById('dl').onclick = download;
document.addEventListener('keydown', function (e) {
  var k = e.key.toLowerCase();
  if (k === 'f') { answer('F'); } else if (k === 'm') { answer('M'); } else if (k === 'u') { answer('U'); } else if (e.key === 'ArrowLeft') { back(); }
});
pos = firstUnlabelled();
show();
</script></body></html>'''

open(V + '/label.html', 'w').write(HTML.replace('__ITEMS__', json.dumps(items)))
print('made', V + '/label.html')
