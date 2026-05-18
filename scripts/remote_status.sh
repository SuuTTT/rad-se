#!/usr/bin/env bash
# remote_status.sh — one-shot snapshot of a remote (or local) training run.
#
# Usage:
#   scripts/remote_status.sh                                    # local default run dir
#   scripts/remote_status.sh -d runs/brax_ppo_CartpoleSwingup_s23_5M
#   scripts/remote_status.sh -h ssh1.vast.ai -p 34217 -u root \
#       -d /root/rad-se/runs/sac_3090_CartpoleSwingup_s23
#
# Flags:
#   -h HOST    remote ssh host (omit for local)
#   -p PORT    remote ssh port (default 22)
#   -u USER    remote ssh user (default root)
#   -d DIR     run work-dir containing train.log / metrics.jsonl
#   -n LINES   tail length for train.log (default 8)
#   -k KEY     extra ssh -i identity file
#
# Reports: process state, elapsed time, GPU mem/util, CPU/RAM, last SPS,
#          best episode reward (from metrics.jsonl), last N log lines.

set -u

HOST=""
PORT="22"
USER_="root"
DIR="runs/sac_3090_CartpoleSwingup_s23"
LINES=8
KEY=""

while getopts "h:p:u:d:n:k:" opt; do
  case "$opt" in
    h) HOST="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    u) USER_="$OPTARG" ;;
    d) DIR="$OPTARG" ;;
    n) LINES="$OPTARG" ;;
    k) KEY="$OPTARG" ;;
    *) echo "usage: $0 [-h HOST] [-p PORT] [-u USER] [-d DIR] [-n LINES] [-k KEYFILE]" >&2; exit 2 ;;
  esac
done

# Build the remote-side script (also runs locally if HOST is empty).
REMOTE_SCRIPT=$(cat <<'EOS'
set -u
DIR="__DIR__"
LINES=__LINES__
METRICS="$DIR/metrics.jsonl"
# Find the log file: prefer $DIR/train.log, else common sibling patterns
# created by `tee` / `> file.log` next to the workdir.
LOG=""
for cand in "$DIR/train.log" "${DIR%/}.log" "${DIR%/}.nohup.log" "${DIR%/}.out"; do
  [ -f "$cand" ] && LOG="$cand" && break
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
sec()  { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

sec "RUN DIR"
echo "$DIR"
if [ ! -d "$DIR" ]; then
  echo "  (missing)"; exit 0
fi
ls -la --time-style=+%H:%M:%S "$DIR" 2>/dev/null | head -8

sec "PROCESS"
# Find python process whose cmdline references this dir.
PIDS=$(pgrep -af "rad_brax_(sac|ppo)" 2>/dev/null | awk -v d="$DIR" 'index($0,d){print $1}')
if [ -z "$PIDS" ]; then
  # Fallback: any rad_brax_* process.
  PIDS=$(pgrep -af "rad_brax_" 2>/dev/null | awk '{print $1}')
fi
if [ -n "$PIDS" ]; then
  ps -o pid,etime,pcpu,pmem,stat,cmd -p $PIDS | head -20
else
  echo "  (no rad_brax_* process matches; run may have finished or not started)"
fi

sec "HOST"
uname -srm 2>/dev/null
echo "uptime:$(uptime 2>/dev/null | sed 's/.*up //; s/,.*//')"
echo "load:  $(awk '{print $1, $2, $3}' /proc/loadavg 2>/dev/null)"

sec "CPU"
nproc 2>/dev/null | awk '{print "cores:", $1}'
# Use top batch mode to get the snapshot %Cpu line.
top -bn1 2>/dev/null | awk '/^%Cpu|Cpu\(s\)/{print; exit}'

sec "MEMORY"
free -h 2>/dev/null | sed -n '1,3p'

sec "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
             --format=csv,noheader 2>/dev/null
else
  echo "  (nvidia-smi not available)"
fi

sec "PROGRESS (last SPS lines)"
if [ -n "$LOG" ]; then
  echo "log: $LOG"
  grep -E "S:[[:space:]]*[0-9]+.*SPS" "$LOG" 2>/dev/null | tail -3
  echo "---"
  LAST=$(grep -E "S:[[:space:]]*[0-9]+.*SPS" "$LOG" 2>/dev/null | tail -1)
  if [ -n "$LAST" ]; then
    echo "latest: $LAST"
  else
    echo "  (no SPS lines yet — still compiling / prefilling)"
  fi
else
  echo "  (no log file found — checked: $DIR/train.log, ${DIR%/}.log, ${DIR%/}.nohup.log)"
fi

sec "BEST EPISODE REWARD"
if [ -f "$METRICS" ]; then
  python3 - <<'PY' 2>/dev/null
import json, os, sys
path = os.environ.get("METRICS_PATH", "")
best = None
last = None
n = 0
keys = ("eval/episode_reward", "episode_reward", "ER", "eval_reward")
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: row = json.loads(line)
        except Exception: continue
        n += 1
        v = None
        for k in keys:
            if k in row:
                v = row[k]; break
        if v is None: continue
        try: v = float(v)
        except Exception: continue
        last = (row.get("step", row.get("num_steps", n)), v)
        if best is None or v > best[1]:
            best = last
if best:
    print(f"rows={n}  best ER={best[1]:.3f} @ step {best[0]}  last ER={last[1]:.3f} @ step {last[0]}")
else:
    print(f"rows={n}  (no episode_reward field yet)")
PY
else
  echo "  (no metrics.jsonl yet)"
fi

sec "LOG TAIL (last $LINES lines)"
if [ -n "$LOG" ]; then
  tail -n "$LINES" "$LOG"
else
  echo "  (no log file)"
fi
EOS
)

# Substitute parameters.
REMOTE_SCRIPT="${REMOTE_SCRIPT//__DIR__/$DIR}"
REMOTE_SCRIPT="${REMOTE_SCRIPT//__LINES__/$LINES}"

if [ -n "$HOST" ]; then
  SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10)
  [ -n "$KEY" ] && SSH_OPTS+=(-i "$KEY")
  # Export METRICS_PATH via env on remote.
  ssh "${SSH_OPTS[@]}" "${USER_}@${HOST}" "METRICS_PATH='${DIR}/metrics.jsonl' bash -s" <<<"$REMOTE_SCRIPT"
else
  METRICS_PATH="${DIR}/metrics.jsonl" bash -s <<<"$REMOTE_SCRIPT"
fi
