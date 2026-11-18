#!/usr/bin/env bash
#
# run_chain.sh — run the simulation reconstruction chain end to end, or
# any single stage, with one directory per stage and per-file logs.
#
#   OUTDIR/
#     clustered/   <stem>_clustered_partNNN.root
#     tracked/     <stem>_tracked_partNNN.root
#     golden/      <stem>_golden_partNNN.root
#     logs/<timestamp>/   chain.log, plus one log per file per stage
#
# Each stage is invoked once per file with an explicit -o, so outputs
# never collide and golden files always come from the tracked route.
#
# Written for the bash 3.2 that ships with macOS.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_chain.sh -o OUTDIR [-i INPUT] [options]

  -i INPUT   Geant4 HitData ROOT file, or a directory of them.
             Required for the cluster stage.
  -o OUTDIR  Output base directory.
  -s STAGE   all | cluster | track | golden          (default: all)
  -j N       Parallel jobs for the track and golden stages (default: 1).
             Clusterization of a single input file is always serial.
  -c         After the golden stage, run muraves-simCompare on the
             first clustered chunk as an end-to-end spot check.
  -f         Replace. Each stage that runs first deletes its existing
             output chunks, so nothing stale from a previous, longer
             run can survive into the next stage.
  -h         Show this help.

Without -f, the track and golden stages skip files whose output
already exists, so an interrupted run can be resumed by re-running the
same command. If a run was killed while a file was being written,
delete that one output before resuming.

Examples:
  run_chain.sh -i /data/sim/hits_full.root -o /data/sim/chain_v2 -j 4 -c
  run_chain.sh -o /data/sim/chain_v2 -s golden -j 4
EOF
}

# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------

INPUT=""
OUTDIR=""
STAGE="all"
JOBS=1
COMPARE=0
FORCE=0

while getopts "i:o:s:j:cfh" option; do
    case "$option" in
        i) INPUT="$OPTARG" ;;
        o) OUTDIR="$OPTARG" ;;
        s) STAGE="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        c) COMPARE=1 ;;
        f) FORCE=1 ;;
        h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 2; }

[[ -n "$OUTDIR" ]] || { usage; die "-o OUTDIR is required"; }

case "$STAGE" in
    all|cluster|track|golden) ;;
    *) die "unknown stage '$STAGE' (all, cluster, track, golden)" ;;
esac

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "-j must be a positive integer"

RUN_CLUSTER=0; RUN_TRACK=0; RUN_GOLDEN=0
case "$STAGE" in
    all)     RUN_CLUSTER=1; RUN_TRACK=1; RUN_GOLDEN=1 ;;
    cluster) RUN_CLUSTER=1 ;;
    track)   RUN_TRACK=1 ;;
    golden)  RUN_GOLDEN=1 ;;
esac

if [[ "$RUN_CLUSTER" == 1 ]]; then
    [[ -n "$INPUT" ]] || die "-i INPUT is required for the cluster stage"
    [[ -e "$INPUT" ]] || die "input not found: $INPUT"
fi

for command_name in muraves-simCluster muraves-simTrk muraves-simGTrk; do
    command -v "$command_name" >/dev/null \
        || die "$command_name not on PATH; activate the environment first"
done

if [[ "$COMPARE" == 1 ]]; then
    command -v muraves-simCompare >/dev/null \
        || die "muraves-simCompare not on PATH"
fi

# ----------------------------------------------------------------------
# Layout and logging
# ----------------------------------------------------------------------

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

CLUSTER_DIR="$OUTDIR/clustered"
TRACK_DIR="$OUTDIR/tracked"
GOLDEN_DIR="$OUTDIR/golden"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUTDIR/logs/$TIMESTAMP"
FAILFILE="$LOGDIR/failed_files.txt"

mkdir -p "$CLUSTER_DIR" "$TRACK_DIR" "$GOLDEN_DIR" \
         "$LOGDIR/cluster" "$LOGDIR/track" "$LOGDIR/golden"
: > "$FAILFILE"

exec > >(tee -a "$LOGDIR/chain.log") 2>&1

trap 'echo; echo "Interrupted."; exit 130' INT TERM

# Keep the Mac awake for exactly as long as this script runs.
if command -v caffeinate >/dev/null; then
    caffeinate -ims -w $$ &
fi

say() { echo "[$(date +%H:%M:%S)] $*"; }

hms() {
    local seconds=$1
    printf '%dh%02dm%02ds' \
        $((seconds / 3600)) $(((seconds % 3600) / 60)) $((seconds % 60))
}

# ----------------------------------------------------------------------
# Environment report
#
# Uses the interpreter the console scripts were installed with, so the
# report describes the code that will actually run — not whatever
# `python` happens to be first on PATH.
# ----------------------------------------------------------------------

PYTHON="$(head -1 "$(command -v muraves-simCluster)" | sed 's/^#!//')"
[[ -x "$PYTHON" ]] || PYTHON="python3"

PACKAGE_DIR="$("$PYTHON" -c \
    'import os, simReco_muraves; print(os.path.dirname(os.path.dirname(simReco_muraves.__file__)))' \
    2>/dev/null || echo "unknown")"

echo "=================================================="
echo "simReco_muraves chain run"
echo "=================================================="
echo "started      : $(date)"
echo "host         : $(hostname)"
echo "stage        : $STAGE"
echo "jobs         : $JOBS"
echo "input        : ${INPUT:-(not used)}"
echo "output       : $OUTDIR"
echo "logs         : $LOGDIR"
echo "python       : $PYTHON"
echo "package from : $PACKAGE_DIR"

if [[ "$PACKAGE_DIR" != "unknown" ]] \
   && git -C "$PACKAGE_DIR" rev-parse HEAD >/dev/null 2>&1; then

    echo "branch       : $(git -C "$PACKAGE_DIR" rev-parse --abbrev-ref HEAD)"
    echo "commit       : $(git -C "$PACKAGE_DIR" rev-parse HEAD)"

    if [[ -n "$(git -C "$PACKAGE_DIR" status --porcelain)" ]]; then
        echo
        echo "WARNING: the working tree has uncommitted changes."
        echo "         Every output will be tagged as produced from a"
        echo "         dirty tree. Commit first if these files matter."
    fi
else
    echo "commit       : (not a git checkout)"
fi

echo
df -h "$OUTDIR" | sed 's/^/disk         : /'
echo "=================================================="
echo

# ----------------------------------------------------------------------
# Per-file workers, exported so xargs can run them in parallel
# ----------------------------------------------------------------------

track_one() {
    local input="$1"
    local name; name="$(basename "$input")"
    local output="$TRACK_DIR/${name/_clustered/_tracked}"
    local file_log="$LOGDIR/track/${name%.root}.log"

    if [[ -e "$output" && "$FORCE" != 1 ]]; then
        echo "  skip    $name (output exists)"
        return 0
    fi

    local -a extra=()
    [[ "$FORCE" == 1 ]] && extra=(--force)

    if muraves-simTrk "$input" -o "$output" ${extra[@]+"${extra[@]}"} \
            > "$file_log" 2>&1; then
        echo "  ok      $name"
    else
        echo "  FAILED  $name  (log: ${file_log#$OUTDIR/})"
        echo "track $input" >> "$FAILFILE"
        return 1
    fi
}

golden_one() {
    local input="$1"
    local name; name="$(basename "$input")"
    local output="$GOLDEN_DIR/${name/_tracked/_golden}"
    local file_log="$LOGDIR/golden/${name%.root}.log"

    if [[ -e "$output" && "$FORCE" != 1 ]]; then
        echo "  skip    $name (output exists)"
        return 0
    fi

    local -a extra=()
    [[ "$FORCE" == 1 ]] && extra=(--force)

    if muraves-simGTrk "$input" --input-type tracked -o "$output" \
            ${extra[@]+"${extra[@]}"} > "$file_log" 2>&1; then
        echo "  ok      $name"
    else
        echo "  FAILED  $name  (log: ${file_log#$OUTDIR/})"
        echo "golden $input" >> "$FAILFILE"
        return 1
    fi
}

export -f track_one golden_one
export TRACK_DIR GOLDEN_DIR LOGDIR FAILFILE FORCE OUTDIR

run_each() {
    # run_each FUNCTION FILE... — at most $JOBS at a time.
    local worker="$1"; shift
    printf '%s\0' "$@" \
        | xargs -0 -n 1 -P "$JOBS" bash -c "$worker \"\$0\"" || true
}

collect() {
    # collect DIR PATTERN — print matching files, one per line, sorted.
    local directory="$1" pattern="$2"
    find "$directory" -maxdepth 1 -type f -name "$pattern" | sort
}

# ----------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------

CLUSTER_TIME=0; TRACK_TIME=0; GOLDEN_TIME=0
TOTAL_START=$SECONDS

if [[ "$RUN_CLUSTER" == 1 ]]; then
    say "Stage 1/3: clusterization"

    existing="$(collect "$CLUSTER_DIR" '*.root')"

    if [[ -n "$existing" ]]; then
        if [[ "$FORCE" == 1 ]]; then
            # Stale chunks from a longer previous run would otherwise
            # survive, be stamped with this run's provenance and flow
            # into tracking.
            say "removing existing chunks in clustered/ (-f)"
            find "$CLUSTER_DIR" -maxdepth 1 -type f -name '*.root' -delete
        else
            die "clustered/ already contains ROOT files; use -f to replace them"
        fi
    fi

    if [[ -d "$INPUT" ]]; then
        raw_inputs="$(collect "$INPUT" '*.root')"
    else
        raw_inputs="$INPUT"
    fi

    [[ -n "$raw_inputs" ]] || die "no ROOT files found in $INPUT"

    stage_start=$SECONDS

    while IFS= read -r raw_file; do
        raw_name="$(basename "$raw_file" .root)"
        say "clustering $raw_name"

        if ! muraves-simCluster "$raw_file" -o "$CLUSTER_DIR" \
                2>&1 | tee "$LOGDIR/cluster/$raw_name.log"; then
            echo "cluster $raw_file" >> "$FAILFILE"
            say "FAILED clustering $raw_name"
        fi
    done <<< "$raw_inputs"

    CLUSTER_TIME=$((SECONDS - stage_start))
    say "clusterization finished in $(hms $CLUSTER_TIME)"
    echo
fi

if [[ "$RUN_TRACK" == 1 ]]; then
    say "Stage 2/3: tracking"

    chunks="$(collect "$CLUSTER_DIR" '*_clustered_part*.root')"
    [[ -n "$chunks" ]] || die "no clustered chunks in $CLUSTER_DIR"

    if [[ "$FORCE" == 1 ]]; then
        say "removing existing chunks in tracked/ (-f)"
        find "$TRACK_DIR" -maxdepth 1 -type f -name '*.root' -delete
    fi

    say "$(echo "$chunks" | wc -l | tr -d ' ') chunk(s), $JOBS job(s)"

    stage_start=$SECONDS
    # shellcheck disable=SC2086
    run_each track_one $chunks
    TRACK_TIME=$((SECONDS - stage_start))

    say "tracking finished in $(hms $TRACK_TIME)"
    echo
fi

if [[ "$RUN_GOLDEN" == 1 ]]; then
    say "Stage 3/3: golden tracks (tracked route)"

    chunks="$(collect "$TRACK_DIR" '*_tracked_part*.root')"
    [[ -n "$chunks" ]] || die "no tracked chunks in $TRACK_DIR"

    if [[ "$FORCE" == 1 ]]; then
        say "removing existing chunks in golden/ (-f)"
        find "$GOLDEN_DIR" -maxdepth 1 -type f -name '*.root' -delete
    fi

    say "$(echo "$chunks" | wc -l | tr -d ' ') chunk(s), $JOBS job(s)"

    stage_start=$SECONDS
    # shellcheck disable=SC2086
    run_each golden_one $chunks
    GOLDEN_TIME=$((SECONDS - stage_start))

    say "golden finished in $(hms $GOLDEN_TIME)"
    echo
fi

COMPARE_STATUS="not run"

if [[ "$COMPARE" == 1 ]]; then
    first_chunk="$(collect "$CLUSTER_DIR" '*_clustered_part*.root' | head -1)"

    if [[ -n "$first_chunk" ]]; then
        say "spot check: muraves-simCompare on $(basename "$first_chunk")"

        if muraves-simCompare "$first_chunk" \
                > "$LOGDIR/compare.log" 2>&1; then
            COMPARE_STATUS="routes agree"
        else
            COMPARE_STATUS="ROUTES DIFFER (see logs/$TIMESTAMP/compare.log)"
        fi

        say "$COMPARE_STATUS"
        echo
    fi
fi

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

count() { collect "$1" "$2" | grep -c . || true; }

failures="$(grep -c . "$FAILFILE" || true)"

echo "=================================================="
echo "Summary"
echo "=================================================="
echo "clustered chunks : $(count "$CLUSTER_DIR" '*_clustered_part*.root')"
echo "tracked chunks   : $(count "$TRACK_DIR"   '*_tracked_part*.root')"
echo "golden chunks    : $(count "$GOLDEN_DIR"  '*_golden_part*.root')"
echo
echo "clusterization   : $(hms $CLUSTER_TIME)"
echo "tracking         : $(hms $TRACK_TIME)"
echo "golden           : $(hms $GOLDEN_TIME)"
echo "total            : $(hms $((SECONDS - TOTAL_START)))"
echo
echo "route spot check : $COMPARE_STATUS"
echo "failed files     : $failures"

if [[ "$failures" -gt 0 ]]; then
    echo
    sed 's/^/  /' "$FAILFILE"
fi

echo "finished         : $(date)"
echo "=================================================="

[[ "$failures" -eq 0 ]]
