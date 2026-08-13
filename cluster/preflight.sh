#!/usr/bin/env bash
# cluster/preflight.sh — one-shot readiness check for the EMRI cluster environment.
#
# Run BEFORE submitting anything to the cluster. It answers, in one structured
# block: is the repo on the right code? is the venv usable? is the galaxy
# catalog present with the expected schema? is the workspace about to expire?
# what is queued/failed? and what datasets (injections / CRBs / posteriors)
# currently exist on the workspace.
#
# Two ways to run it:
#   • From the dev box (recommended — no need to place it on the cluster):
#       ssh bwunicluster 'bash -s' < cluster/preflight.sh
#   • On the cluster directly:
#       bash ~/darksiren-emri/cluster/preflight.sh
#
# It is read-only: it never submits, cancels, edits, or deletes anything.

# --- config (overridable via env; falls back to cluster.env if present) --------
REPO="${CLUSTER_REPO:-$HOME/darksiren-emri}"
[ -f "$REPO/cluster/cluster.env" ] && . "$REPO/cluster/cluster.env" 2>/dev/null
REPO="${CLUSTER_REPO:-$HOME/darksiren-emri}"
WS_NAME="${CLUSTER_WORKSPACE_NAME:-emri}"
EXPECT_COLS="${EMRI_EXPECTED_CATALOG_COLS:-8}"
WARN_DAYS="${EMRI_WORKSPACE_WARN_DAYS:-14}"
TAG="${EMRI_COMMISSION_TAG:-commission-base}"
CATALOG="$REPO/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv"

PROBLEMS=()
note_problem() { PROBLEMS+=("$1"); }

echo "======================================================================"
echo " EMRI CLUSTER PREFLIGHT  —  $(hostname)  —  $(date '+%Y-%m-%d %H:%M %Z')"
echo "======================================================================"

# --- [REPO] --------------------------------------------------------------------
echo "[REPO]  $REPO"
if [ -d "$REPO/.git" ]; then
    cd "$REPO" || exit 1
    BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
    HD=$(git rev-parse --short HEAD 2>/dev/null)
    DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    HAS_TAG=$(git tag --list "$TAG" 2>/dev/null)
    git fetch origin --quiet 2>/dev/null
    BEHIND=$(git rev-list --count "HEAD..origin/$BR" 2>/dev/null || echo "?")
    AHEAD=$(git rev-list --count "origin/$BR..HEAD" 2>/dev/null || echo "?")
    echo "        branch=$BR head=$HD  ahead=$AHEAD behind=$BEHIND  dirty=$DIRTY files"
    if [ -n "$HAS_TAG" ]; then
        echo "        tag '$TAG' -> $(git rev-parse --short "$TAG" 2>/dev/null)"
    else
        echo "        tag '$TAG' -> (absent)"
    fi
    [ "$DIRTY" != "0" ] && echo "        note: working tree dirty — 'git status' before submitting"
    W=$(git worktree list 2>/dev/null | wc -l | tr -d ' ')
    [ "$W" != "1" ] && { echo "        WARNING: $W worktrees registered (expected 1 — no separate clones)"; note_problem "multiple git worktrees"; }
else
    echo "        MISSING repo/.git"; note_problem "repo missing"
fi

# --- [VENV] --------------------------------------------------------------------
echo "[VENV]  $REPO/.venv"
if [ -x "$REPO/.venv/bin/python" ]; then
    # The venv python needs the module-provided libpython — load modules quietly.
    . "$REPO/cluster/modules.sh" >/dev/null 2>&1
    IMP=$("$REPO/.venv/bin/python" - <<'PY' 2>&1
try:
    import darksiren_emri, numpy, scipy, pandas
    tail = ""
    try:
        import few  # GPU waveform pkg (present only in the --extra gpu venv)
        tail = " +few"
    except Exception:
        tail = " (no few — cpu venv)"
    print("import-ok numpy", numpy.__version__ + tail)
except Exception as e:
    print("IMPORT-FAIL:", e)
PY
)
    echo "        $IMP"
    case "$IMP" in *IMPORT-FAIL*) note_problem "venv import failed";; esac
else
    echo "        MISSING .venv/bin/python — run: source cluster/modules.sh && uv sync --extra gpu"
    note_problem "venv missing"
fi

# --- [CATALOG] -----------------------------------------------------------------
echo "[CATALOG]"
if [ -f "$CATALOG" ]; then
    SZ=$(du -h "$CATALOG" 2>/dev/null | cut -f1)
    COLS=$(head -1 "$CATALOG" 2>/dev/null | awk -F, '{print NF}')
    FLAG="OK"; [ "$COLS" != "$EXPECT_COLS" ] && { FLAG="SCHEMA-DRIFT (expected $EXPECT_COLS)"; note_problem "catalog schema $COLS != $EXPECT_COLS"; }
    echo "        reduced_galaxy_catalogue.csv: $SZ  cols=$COLS  [$FLAG]"
    # Provenance probe (TC-11): z_helio and z_cmb catalogues are BOTH 8-col and
    # full-depth — column count cannot discriminate. Fingerprint the first data
    # row's 4th field (redshift; the file has no header) instead. O(1), no scan.
    ROW1_Z=$(head -1 "$CATALOG" 2>/dev/null | awk -F, '{print $4}')
    EXPECT_ROW1_Z="${EMRI_CATALOG_ROW1_Z:-0.001733}"
    if [ "$ROW1_Z" = "$EXPECT_ROW1_Z" ]; then
        echo "        provenance: row1 z=$ROW1_Z — z_cmb frame ✓"
    elif [ "$ROW1_Z" = "0.000990570495285" ]; then
        echo "        provenance: row1 z=$ROW1_Z — STALE z_helio catalogue ✗"
        note_problem "STALE z_helio catalogue (row1 z=$ROW1_Z, expected $EXPECT_ROW1_Z)"
    else
        echo "        provenance: row1 z=$ROW1_Z — WARNING: unknown catalogue revision (expected $EXPECT_ROW1_Z)"
    fi
else
    echo "        reduced_galaxy_catalogue.csv: ABSENT"
    echo "        (auto-rebuilds from GLADE+.txt if present; else stage from dev box via rsync)"
    note_problem "reduced catalog absent"
fi
NPY="$REPO/darksiren_emri/galaxy_catalogue/m_th_map_nside32.npy"
[ -f "$NPY" ] && echo "        m_th_map_nside32.npy: present (git-tracked)" || { echo "        m_th_map_nside32.npy: ABSENT"; note_problem "nside32 map absent"; }
GLADE=$(find "$REPO/darksiren_emri/galaxy_catalogue" -maxdepth 1 -name 'GLADE+.txt' 2>/dev/null | head -1)
[ -n "$GLADE" ] && echo "        GLADE+.txt: present ($(du -h "$GLADE" | cut -f1))" || echo "        GLADE+.txt: absent (fine — only needed to regenerate the reduced csv)"

# --- [WORKSPACE] ---------------------------------------------------------------
echo "[WORKSPACE] '$WS_NAME'"
WS_PATH=$(ws_find "$WS_NAME" 2>/dev/null)
if [ -n "$WS_PATH" ]; then
    DAYS=$(ws_list "$WS_NAME" 2>/dev/null | grep -i 'remaining time' | grep -oE '[0-9]+ days?' | head -1 | grep -oE '[0-9]+')
    echo "        path=$WS_PATH"
    echo "        remaining=${DAYS:-?} days"
    if [ -n "$DAYS" ] && [ "$DAYS" -lt "$WARN_DAYS" ] 2>/dev/null; then
        echo "        WARNING: expires in <$WARN_DAYS days — ws_extend $WS_NAME 60 and copy results to persistent storage"
        note_problem "workspace expiring (<$WARN_DAYS d)"
    fi
else
    echo "        NOT FOUND — run cluster/setup.sh (ws_allocate)"; note_problem "workspace missing"
fi

# --- [QUEUE] -------------------------------------------------------------------
echo "[QUEUE]"
if command -v squeue >/dev/null 2>&1; then
    RUN=$(squeue -u "$USER" -h -t RUNNING 2>/dev/null | wc -l | tr -d ' ')
    PEND=$(squeue -u "$USER" -h -t PENDING 2>/dev/null | wc -l | tr -d ' ')
    DEAD=$(squeue -u "$USER" -h -o '%r' 2>/dev/null | grep -c 'DependencyNeverSatisfied')
    echo "        running=$RUN pending=$PEND dependency-dead=$DEAD"
    if [ "$DEAD" -gt 0 ] 2>/dev/null; then
        echo "        zombie jobs (will never run — consider scancel):"
        squeue -u "$USER" -h -o '          %.12i %.16j %.24R' 2>/dev/null | grep 'DependencyNeverSati'
        note_problem "$DEAD dependency-dead jobs in queue"
    fi
    [ $((RUN + PEND)) -gt 0 ] && squeue -u "$USER" -o '        %.12i %.16j %.9T %.10M %.20R' 2>/dev/null
else
    echo "        squeue unavailable (not on a SLURM node?)"
fi

# --- [DATASETS] live inventory scan (ground truth) -----------------------------
echo "[DATASETS]  (live scan of $WS_PATH)"
if [ -n "$WS_PATH" ]; then
    echo "        injections (P_det pools):"
    for d in "$WS_PATH"/injection_*; do
        [ -d "$d/simulations/injections" ] || continue
        n=$(find "$d/simulations/injections" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')
        echo "          $(basename "$d"): $n csv"
    done
    echo "        runs (crb / posteriors):"
    for R in "$WS_PATH"/run_*; do
        S="$R/simulations"; [ -d "$S" ] || continue
        crb="-"; [ -f "$S/cramer_rao_bounds.csv" ] && crb="crb"
        prp="-"; [ -f "$S/prepared_cramer_rao_bounds.csv" ] && prp="prep"
        p1=$(find "$S/posteriors" -maxdepth 1 -name 'h_*.json' 2>/dev/null | wc -l | tr -d ' ')
        p2=$(find "$S/posteriors_with_bh_mass" -maxdepth 1 -name 'h_*.json' 2>/dev/null | wc -l | tr -d ' ')
        cb="-"; [ -f "$S/posteriors/combined_posterior.json" ] && cb="combined"
        [ "$crb$prp$p1$p2" = "--00" ] && continue
        echo "          $(basename "$R"): $crb $prp post=$p1/$p2 $cb"
    done
    # A2-STALE-POOL-GATE (c): flag pre-depth-1.5 injection pools. Injection CSVs
    # are written by main.py:_flush_injection_results with header columns
    # ["z","M","phiS","qS","SNR","h_inj","luminosity_distance"] (main.py:631);
    # the awk below locates the "z" column by header name, not by guessing.
    # Cheap probe: one representative file (all files in a pool share z_cut).
    INJ_LINK="$REPO/simulations/injections"
    if [ -d "$INJ_LINK" ]; then
        SAMPLE_CSV=$(find -L "$INJ_LINK" -maxdepth 1 -name 'injection_h_*.csv' 2>/dev/null | head -1)
        if [ -n "$SAMPLE_CSV" ]; then
            MAXZ=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="z") c=i; next} c && $c+0>m {m=$c+0} END{if(c) printf "%.4f", m; else print "no-z-col"}' "$SAMPLE_CSV")
            if [ "$MAXZ" = "no-z-col" ]; then
                echo "        injection pool depth probe: no 'z' header in $(basename "$SAMPLE_CSV") — cannot probe (depth is gated in Python)"
            elif awk -v z="$MAXZ" 'BEGIN{exit !(z < 1.0)}'; then
                echo "        WARNING: shallow (pre-depth-1.5) injection pool detected (max z=$MAXZ in $(basename "$SAMPLE_CSV"))"
            else
                echo "        injection pool depth probe: max z=$MAXZ ($(basename "$SAMPLE_CSV")) — depth-1.5 compatible"
            fi
        fi
    fi
    echo "        (semantic map + provenance: cluster/datasets.yaml ; retirement status: DATA_INVENTORY.md)"
fi

# --- [VERDICT] -----------------------------------------------------------------
echo "----------------------------------------------------------------------"
if [ ${#PROBLEMS[@]} -eq 0 ]; then
    echo " VERDICT: READY ✓"
else
    echo " VERDICT: NOT READY — ${#PROBLEMS[@]} issue(s):"
    for p in "${PROBLEMS[@]}"; do echo "   • $p"; done
fi
echo "======================================================================"
