#!/bin/bash
cd ~/Downloads/euqvid_temporal
PY=/opt/anaconda3/envs/euqvid_face/bin/python
CLIPS=data/AVA_videos/clips
LOG=outputs/ava_gender/cvdf_loop_log.txt
while pgrep -f "ava_gender.py 1000" >/dev/null; do sleep 60; done
for name in $(cat data/AVA_videos/cvdf_files.txt); do
  id="${name%.*}"
  if [ ! -f "data/AVA/ava_activespeaker_train_v1.0/$id-activespeaker.csv" ] && [ ! -f "data/AVA/ava_activespeaker_test_v1.0/$id-activespeaker.csv" ]; then continue; fi
  if grep -q "^$id," outputs/ava_gender/entity_frames.csv 2>/dev/null; then echo "$id already_done" >> $LOG; continue; fi
  avail=$(df -g ~ | tail -1 | awk '{print $4}')
  if [ "$avail" -lt 4 ]; then echo "STOP low disk: $avail GB" >> $LOG; exit 1; fi
  f="$CLIPS/$name"
  got=0
  if ! ls $CLIPS/$id.* >/dev/null 2>&1; then
    if curl -sSL -f --retry 3 -o "$f.part" "https://s3.amazonaws.com/ava-dataset/trainval/$name"; then
      mv "$f.part" "$f"; got=1
    else
      echo "$id download_failed" >> $LOG; rm -f "$f.part"; continue
    fi
  fi
  $PY src/ava_gender.py 1 "$id" >> outputs/ava_gender/cvdf_run.txt 2>&1
  echo "$id processed" >> $LOG
  if [ "$got" = "1" ]; then rm -f "$f"; fi
done
echo "ALL DONE" >> $LOG
