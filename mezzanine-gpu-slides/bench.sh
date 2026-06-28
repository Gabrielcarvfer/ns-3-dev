#!/usr/bin/env bash
#
# bench.sh -- reproduce the CPU-vs-GPU speedup tables for the WebGPU "mezzanine"
# GPU offload of the contrib/nr 3GPP TR 38.901 channel model.
#
# Portable: macOS (bash 3.2, BSD date, no `timeout`) and Linux. No Windows-isms.
#
# ---------------------------------------------------------------------------
# METHODOLOGY (the *corrected* one -- read this before trusting any number)
# ---------------------------------------------------------------------------
# The honest config is: stock DenseAmimo scenario, full 3GPP channel, and a
# real channel update period. Three things were found to silently fake a "GPU
# loss" in an earlier sweep; this script avoids all three:
#   1. NO antenna-shrinking env hack (it uses the stock DenseAmimo array).
#   2. ALWAYS passes --ns3::ThreeGppChannelModel::UpdatePeriod=100ms, so the
#      channel actually evolves and the GPU's channel-gen offload has work to
#      amortize. (Omitting it => static channel => GPU looks slow. This was the
#      single biggest artifact.)
#   3. Compares against a CLEAN build (asserts OFF, SLS profiling OFF) -- see
#      BUILD below.
#
# Speedup = CPU_wall / GPU_wall  (>1 means the GPU is faster).
#
# ---------------------------------------------------------------------------
# BUILD (do this first, once)
# ---------------------------------------------------------------------------
#   cd <ns-3-dev>
#   ./ns3 configure -d optimized --enable-examples \
#         -- -DNS3_ASSERT=OFF -DNS3_SLS_PROFILE=OFF
#   ./ns3 build cttc-nr-3gpp-calibration-user
#
#   - `-d optimized` gives -O3 and disables asserts/logging.
#   - NS3_SLS_PROFILE already defaults OFF, but pass it explicitly to be safe.
#   - The GPU backend (NS3_ENABLE_3GPP_GPU=ON, WEBGPU_BACKEND=DAWN) is fetched
#     via CMake FetchContent and builds on macOS (Metal) and Linux (Vulkan).
#     If the GPU rows error out, your platform's Dawn build is the thing to fix;
#     the CPU rows are independent and will still run.
#
# ---------------------------------------------------------------------------
# BANDWIDTH > 40 MHz
# ---------------------------------------------------------------------------
# The committed DenseAmimo preset hard-caps bandwidth at {5,10,20,40} MHz
# (NS_ABORT in cttc-nr-3gpp-calibration.cc, "Valid bandwidth values are ...").
# To reproduce the 100/200 MHz columns you must lift that abort and rebuild:
#   replace the NS_ABORT_MSG_IF(bandwidthMHz != 40 && ... ) check with
#   NS_ABORT_MSG_IF(bandwidthMHz < 1, "bad bw");
# Cells with bw>40 are SKIPPED automatically unless ALLOW_WIDE_BW=1 is set.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   ./bench.sh                 # default sweep (rings 1,3 x bw 10,40 x 3 modes)
#   REPO=/path/to/ns-3-dev ./bench.sh
#   RINGS="0 1 3" CELLS="10:0 40:0" MODES="cpu gpu gpu_batch" ./bench.sh
#   ALLOW_WIDE_BW=1 CELLS="10:0 40:0 100:1 200:2" ./bench.sh   # needs the patch
#   PARITY=1 ./bench.sh        # also diff the GPU sinr table against CPU
#
set -u

# ----------------------------- configuration -------------------------------
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"   # default: parent of this dir
BUILD="${BUILD:-$REPO/build}"
BIN="${BENCH_BIN:-}"                                  # auto-discovered if empty

# cells are "bandwidthMHz:numerology"; rings is a numRings list.
CELLS="${CELLS:-10:0 40:0}"
RINGS="${RINGS:-1 3}"
MODES="${MODES:-cpu gpu gpu_batch}"   # also available: gpu_batch_skip

APPGEN="${APPGEN:-5ms}"               # appGenerationTime; <too short> => no flows
UPDATE="${UPDATE:-100ms}"             # ThreeGppChannelModel::UpdatePeriod (KEEP)
SEED="${SEED:-1}"; RUN="${RUN:-1}"
TIMEOUT_S="${TIMEOUT_S:-1800}"        # per-run wall cap (best-effort)
PARITY="${PARITY:-0}"                 # 1 => sqlite3 sinr-table diff vs CPU
ALLOW_WIDE_BW="${ALLOW_WIDE_BW:-0}"   # 1 => attempt bw>40 (needs source patch)
OUT="${OUT:-$REPO/bench_out}"

# mode -> environment (post-commit: batch defaults ON under MEZ_USE_GPU, so the
# plain "gpu" mode must explicitly turn the batch OFF to isolate channel offload)
mode_env() {
  case "$1" in
    cpu)            echo "" ;;
    gpu)            echo "MEZ_USE_GPU=1 MEZ_BATCH_RECEPTION=0 MEZ_BATCH_GPU=0" ;;
    gpu_batch)      echo "MEZ_USE_GPU=1 MEZ_BATCH_RECEPTION=1 MEZ_BATCH_GPU=1" ;;
    gpu_batch_skip) echo "MEZ_USE_GPU=1 MEZ_BATCH_RECEPTION=1 MEZ_BATCH_GPU=1 MEZ_SKIP_MATCOPY=1" ;;
    *) echo "UNKNOWN_MODE" ;;
  esac
}

# ----------------------------- portability ---------------------------------
# sub-second clock: GNU date(%N) -> python3 -> integer seconds
HAVE_GNU_DATE=0
if [ "$(date +%N 2>/dev/null)" != "N" ] && [ -n "$(date +%N 2>/dev/null)" ]; then HAVE_GNU_DATE=1; fi
HAVE_PY=0; command -v python3 >/dev/null 2>&1 && HAVE_PY=1
now() {
  if [ "$HAVE_GNU_DATE" -eq 1 ]; then date +%s.%N
  elif [ "$HAVE_PY" -eq 1 ]; then python3 -c 'import time;print(time.time())'
  else date +%s; fi
}
# optional run-time cap
TO=""
command -v timeout  >/dev/null 2>&1 && TO="timeout"
command -v gtimeout >/dev/null 2>&1 && TO="gtimeout"

# ----------------------------- discover binary -----------------------------
if [ -z "$BIN" ]; then
  BIN=$(find "$BUILD" -type f -name '*cttc-nr-3gpp-calibration-user-optimized' 2>/dev/null | head -1)
fi
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  echo "ERROR: optimized binary not found under $BUILD" >&2
  echo "Build it first (see the BUILD section at the top of this script):" >&2
  echo "  ./ns3 configure -d optimized --enable-examples -- -DNS3_ASSERT=OFF -DNS3_SLS_PROFILE=OFF" >&2
  echo "  ./ns3 build cttc-nr-3gpp-calibration-user" >&2
  exit 1
fi
BINBASE=$(basename "$BIN")
echo "binary : $BIN"
echo "out    : $OUT"
echo "cells  : [$CELLS]   rings: [$RINGS]   modes: [$MODES]"
echo "config : DenseAmimo  appGen=$APPGEN  UpdatePeriod=$UPDATE  seed=$SEED/run=$RUN"
echo

mkdir -p "$OUT"
RES="$OUT/results.tsv"
: > "$RES"
printf "ring\tbw\tnum\tmode\twall_s\trc\tparity\n" >> "$RES"

# make sure no GPU/MEZ var leaks in from the caller
unset MEZ_USE_GPU MEZ_BATCH_RECEPTION MEZ_BATCH_GPU MEZ_SKIP_MATCOPY MEZ_DEFER_CHUNKS

# ----------------------------- run one cell --------------------------------
run_one() {  # $1=ring $2=bw $3=num $4=mode
  ring="$1"; bw="$2"; num="$3"; mode="$4"
  if [ "$bw" -gt 40 ] && [ "$ALLOW_WIDE_BW" != "1" ]; then
    printf "  %-14s SKIPPED (bw>40 needs the cap-lift patch; set ALLOW_WIDE_BW=1)\n" "$mode r$ring ${bw}MHz"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ring" "$bw" "$num" "$mode" "NA" "skip" "-" >> "$RES"
    return
  fi
  env_str=$(mode_env "$mode")
  tag="r${ring}_bw${bw}_n${num}_${mode}"
  d="$OUT/$tag"; rm -rf "$d"; mkdir -p "$d"

  pkill -f "$BINBASE" >/dev/null 2>&1 || true

  args="--configurationType=calibrationConf --technology=NR"
  args="$args --nrConfigurationScenario=DenseAmimo --numRings=$ring"
  args="$args --bandwidth=$bw --numerologyBwp=$num"
  args="$args --appGenerationTime=$APPGEN --RngSeed=$SEED --RngRun=$RUN"
  args="$args --ns3::ThreeGppChannelModel::UpdatePeriod=$UPDATE"
  args="$args --outputDir=$d --dbName=db"

  start=$(now)
  ( cd "$d" && env $env_str ${TO:+$TO ${TIMEOUT_S}s} "$BIN" $args >run.out 2>run.err )
  rc=$?
  end=$(now)
  wall=$(awk "BEGIN{printf \"%.1f\", $end-$start}")

  # optional parity vs the CPU run of the same cell
  par="-"
  if [ "$PARITY" = "1" ] && command -v sqlite3 >/dev/null 2>&1 && [ -f "$d/db.db" ]; then
    sqlite3 "$d/db.db" "SELECT * FROM sinr ORDER BY 1,2,3,4;" > "$d/sinr.txt" 2>/dev/null || true
    cpud="$OUT/r${ring}_bw${bw}_n${num}_cpu"
    if [ "$mode" != "cpu" ] && [ -f "$cpud/sinr.txt" ]; then
      if diff -q "$cpud/sinr.txt" "$d/sinr.txt" >/dev/null 2>&1; then par="MATCH"; else par="DIFFER"; fi
    fi
  fi

  printf "  %-16s rc=%-3s wall=%6ss  parity=%s\n" "$mode r$ring ${bw}MHz" "$rc" "$wall" "$par"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$ring" "$bw" "$num" "$mode" "$wall" "$rc" "$par" >> "$RES"
}

# ----------------------------- sweep ---------------------------------------
for ring in $RINGS; do
  for cell in $CELLS; do
    bw=${cell%%:*}; num=${cell##*:}
    echo "== ring $ring, ${bw}MHz (num $num) =="
    # run cpu first so PARITY has a reference
    for mode in $MODES; do [ "$mode" = "cpu" ] && run_one "$ring" "$bw" "$num" "$mode"; done
    for mode in $MODES; do [ "$mode" != "cpu" ] && run_one "$ring" "$bw" "$num" "$mode"; done
    echo
  done
done

# ----------------------------- markdown table ------------------------------
echo
echo "=================== SPEEDUP TABLE (Speedup = CPU/GPU) ==================="
awk -F'\t' '
  NR==1 { next }
  { key=$1"|"$2"|"$3; mode=$4; wall[key"|"mode]=$5; seen[key]=1; order[++n]=key }
  END {
    # de-dup keys preserving first-seen order
    cnt=0
    for (i=1;i<=n;i++){ k=order[i]; if(!(k in done)){done[k]=1; keys[++cnt]=k} }
    printf "| Ring | BW (MHz) | Num | CPU (s) | GPU (s) | GPU+batch (s) | GPU x | batch x |\n"
    printf "|------|----------|-----|---------|---------|---------------|-------|---------|\n"
    for (i=1;i<=cnt;i++){
      k=keys[i]; split(k,a,"|"); r=a[1]; b=a[2]; nu=a[3]
      c=wall[k"|cpu"]; g=wall[k"|gpu"]; gb=wall[k"|gpu_batch"]
      if(c=="")c="-"; if(g=="")g="-"; if(gb=="")gb="-"
      gx="-"; bx="-"
      if(c!="-"&&c!="NA"&&g!="-"&&g!="NA"&&g+0>0) gx=sprintf("%.2f", c/g)
      if(c!="-"&&c!="NA"&&gb!="-"&&gb!="NA"&&gb+0>0) bx=sprintf("%.2f", c/gb)
      printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", r,b,nu,c,g,gb,gx,bx
    }
  }' "$RES"

echo
echo "raw results : $RES"
echo "(per-run stdout/stderr + sqlite db are under $OUT/<tag>/)"
