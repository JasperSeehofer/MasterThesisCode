#!/usr/bin/env bash
# migrate_local_rename.sh — Rebrand Migration Checklist §1/§3
#
# Renames the local dev-box checkout from MasterThesisCode to darksiren-emri
# and re-keys the two systems that key off the old filesystem path (Claude
# Code project memory, garden registry), relocates the venv, then applies the
# §3 mechanical reference fixes inside the renamed repo.
#
# PER-MACHINE, AND RE-RUNNABLE. Every dev box needs its own run, and every
# step is idempotent: a step whose work is already done reports "already
# migrated" and continues instead of aborting. That matters because the garden
# registry (step 4) is a *shared, synced* file — the first box to migrate it
# migrates it for all of them, so a "the old path must still be present"
# precondition can never hold on box number two.
#
# PRECONDITIONS (both hard requirements):
#   1. Run this script FROM $HOME.
#   2. Run it with NO Claude Code session open anywhere under the repo —
#      an open session holds file handles / working-directory state keyed
#      to the old path, and re-keying its project directory (step 3) out
#      from under a live session will orphan that session.
#
# Usage: cd ~ && bash ~/Repositories/MasterThesisCode/scripts/migrate_local_rename.sh
#   (or  bash ~/Repositories/darksiren-emri/scripts/migrate_local_rename.sh
#        if the directory rename already happened and you are re-running.)
#
# See docs/REBRAND_MIGRATION_CHECKLIST.md §1 and §3 for the source plan.

set -euo pipefail

OLD_DIR="/home/jasper/Repositories/MasterThesisCode"
NEW_DIR="/home/jasper/Repositories/darksiren-emri"
OLD_NAME="MasterThesisCode"
NEW_NAME="darksiren-emri"

# Extras to sync into the relocated venv (step 7). Dev boxes are CPU boxes;
# override with UV_SYNC_EXTRAS="--extra gpu" if that ever stops being true.
UV_SYNC_EXTRAS="${UV_SYNC_EXTRAS:---extra cpu --extra dev}"

# The sibling `-book` worktree (.../MasterThesisCode-book) is NOT renamed, and
# $OLD_DIR is a strict PREFIX of it. Every replacement below therefore uses a
# negative lookahead on `-book`; a plain sed would silently repoint live
# references at a directory that does not exist.
repl_abs() {  # repl_abs FILE...  — rewrite the absolute repo path, sparing -book
    perl -pi -e "s{\Q$OLD_DIR\E(?!-book)}{$NEW_DIR}g" "$@"
}
repl_name() {  # repl_name FILE... — rewrite the bare directory name, sparing -book
    perl -pi -e "s{\Q$OLD_NAME\E(?!-book)}{$NEW_NAME}g" "$@"
}

echo "=== migrate_local_rename.sh — Rebrand Migration Checklist §1/§3 ==="
echo

# --- Precondition: must be run from $HOME -----------------------------
if [ "$PWD" != "$HOME" ]; then
    echo "ABORT: must be run from \$HOME ($HOME); currently in $PWD" >&2
    echo "       cd ~ and re-run." >&2
    exit 1
fi
echo "[precondition] Running from \$HOME ($HOME) — OK."
echo "[precondition] REMINDER: this must run with no Claude Code session open"
echo "               in the repo. If one is open, close it now and re-run."
echo

# --- Step 1: preflight existence checks --------------------------------
echo "[1/9] Preflight: locating the checkout..."
if [ -d "$OLD_DIR" ] && [ -d "$NEW_DIR" ]; then
    echo "ABORT: BOTH $OLD_DIR and $NEW_DIR exist — ambiguous." >&2
    echo "       Resolve by hand (which one is the live checkout?) before re-running." >&2
    exit 1
fi
if [ -d "$NEW_DIR" ]; then
    NEEDS_MV=0
    echo "  $NEW_DIR already exists and $OLD_DIR does not — directory already renamed."
elif [ -d "$OLD_DIR" ]; then
    NEEDS_MV=1
    echo "  OK: $OLD_DIR exists, $NEW_DIR does not."
else
    echo "ABORT: neither $OLD_DIR nor $NEW_DIR exists — nothing to rename." >&2
    exit 1
fi
echo

# --- Step 2: rename the repo directory ----------------------------------
echo "[2/9] Renaming repo directory..."
if [ "$NEEDS_MV" -eq 1 ]; then
    mv "$OLD_DIR" "$NEW_DIR"
    echo "  mv: $OLD_DIR -> $NEW_DIR"
    if [ ! -d "$NEW_DIR" ]; then
        echo "ABORT: rename did not take effect — $NEW_DIR missing after mv." >&2
        exit 1
    fi
    echo "  verified: $NEW_DIR exists."
else
    echo "  SKIP: already renamed."
fi
echo

# --- Step 3: re-key Claude Code project memory / session state ---------
echo "[3/9] Re-keying Claude Code project memory/session state..."
CLAUDE_OLD="$HOME/.claude/projects/-home-jasper-Repositories-MasterThesisCode"
CLAUDE_NEW="$HOME/.claude/projects/-home-jasper-Repositories-darksiren-emri"
if [ -d "$CLAUDE_OLD" ] && [ -d "$CLAUDE_NEW" ]; then
    echo "ABORT: BOTH $CLAUDE_OLD and $CLAUDE_NEW exist." >&2
    echo "       Two project histories under two keys — merge them by hand (the new key" >&2
    echo "       is the one a session in $NEW_DIR will read), then re-run." >&2
    exit 1
fi
if [ -d "$CLAUDE_OLD" ]; then
    mv "$CLAUDE_OLD" "$CLAUDE_NEW"
    echo "  mv: $CLAUDE_OLD -> $CLAUDE_NEW"
elif [ -d "$CLAUDE_NEW" ]; then
    echo "  SKIP: already re-keyed ($CLAUDE_NEW exists)."
else
    echo "  WARNING: neither key exists — no project memory on this box, continuing."
fi
echo

# --- Step 4: garden registry Path column --------------------------------
#
# The garden's session-start hook (wiki/assets/claude-hooks/wiki-session-start.sh)
# finds a project by testing whether the registry Path column is a PREFIX of $CWD,
# and `exit 0`s silently when nothing matches. So a renamed directory with a stale
# registry Path does not error — it just stops briefing this repo, with no signal.
# That makes the POST-STATE a hard requirement: what must hold at the end of this
# step is "a Path column equals $NEW_DIR", not "an edit was performed here".
#
# The registry is shared and synced, so on any box after the first it is already
# migrated and there is nothing to edit. That is success, not failure.
#
# The registry lives at wiki/meta/registry.md, NOT at the repo root. (An earlier
# revision of this script pointed at $GARDEN/registry.md, which does not exist —
# it would have taken the "skipping" branch and produced exactly the silent
# no-briefing failure described above.)
echo "[4/9] Updating garden registry..."
GARDEN="$HOME/Repositories/garden"
REGISTRY="$GARDEN/wiki/meta/registry.md"
if [ ! -f "$REGISTRY" ]; then
    echo "ABORT: $REGISTRY not found." >&2
    echo "       The repo directory has already been moved to $NEW_DIR and Claude" >&2
    echo "       project memory re-keyed. Fix the garden checkout, then re-run —" >&2
    echo "       steps 1-3 are idempotent and will skip themselves." >&2
    exit 1
fi
if grep -q "$OLD_DIR" "$REGISTRY"; then
    cp "$REGISTRY" "$REGISTRY.bak"
    echo "  backup: $REGISTRY.bak"
    sed -i "s|$OLD_DIR|$NEW_DIR|g" "$REGISTRY"
    echo "  updated line(s):"
    grep -n "$NEW_DIR" "$REGISTRY" | sed 's/^/    /'
    echo "  TODO(manual): commit the garden change — the vault is a git repo and this"
    echo "        script deliberately does not commit on your behalf outside this repo."
    echo "        Remove $REGISTRY.bak once you have confirmed the edit."
else
    echo "  SKIP: no occurrences of $OLD_DIR — registry already migrated (it is a"
    echo "        shared, synced file; the first box to run this migrated it for all)."
fi
# Verify the hook's own prefix test succeeds against the new path, edit or not.
if awk -F '|' -v new="$NEW_DIR" '
      /\|/ && !/---/ && !/Project.*Path/ {
        gsub(/^ +| +$/,"",$3); if ($3 == new) found=1
      } END { exit found ? 0 : 1 }' "$REGISTRY"; then
    echo "  verified: a Path column equals $NEW_DIR (hook prefix test will match)"
else
    echo "ABORT: no registry Path column equals $NEW_DIR." >&2
    echo "       The briefing hook would silently stop firing for this repo. Fix the" >&2
    echo "       Path column by hand (restore $REGISTRY.bak if this run wrote one)." >&2
    exit 1
fi
echo "  NOTE: the vault slug stays 'master-thesis-code'. Downstream tables"
echo "        (interaction-feedback reminders, briefing-feedback, portfolio-health,"
echo "        context-budget, agent-weaknesses) key on the SLUG, not the path, so"
echo "        they keep working untouched. Renaming the slug is a separate,"
echo "        optional migration — see the garden's rename plan."
echo

# --- Step 5: §3 reference fixes inside the renamed repo -----------------
echo "[5/9] Applying §3 reference fixes inside $NEW_DIR..."
cd "$NEW_DIR"

TOUCHED=()

# Files carrying the literal old absolute path (checklist §3).
SIMPLE_FILES=(
    ".claude/skills/known-bugs/SKILL.md"
    ".claude/skills/physics-change/SKILL.md"
    "darksiren_emri_test/bayesian_inference/test_posterior_combination.py"
    "book/design/BOOK_DESIGN.md"
    "book/design/BOOK_SOURCES_MAP.md"
    "book/design/BOOK_TECH_DESIGN.md"
    "book/design/reviews/expert_A_ch00-06_museum.md"
    "book/design/reviews/expert_B_ch07-11_cellB.md"
    "book/README.md"
    # Local dev-box `REPO = Path(...)` constants. Only run_multi_truth_sweep.sh's
    # three lines are CLUSTER paths, and those belong to §2 / the cluster script.
    "scripts/bias_investigation/test_30_f4_estimator_smoothness.py"
    "scripts/bias_investigation/test_31_completion_term_characterization.py"
)
for f in "${SIMPLE_FILES[@]}"; do
    if [ -f "$f" ]; then
        if grep -qP "\Q$OLD_DIR\E(?!-book)" "$f"; then
            repl_abs "$f"
            echo "  fixed: $f"
            TOUCHED+=("$f")
        else
            echo "  SKIP (already fixed): $f"
        fi
    else
        echo "  WARNING: expected file not found, skipping: $f"
    fi
done

# .claude/settings.local.json carries the repo path in Bash allow-rules.
# It is GITIGNORED (~/.config/git/ignore), so it is fixed in place but never
# staged — `git add` on an ignored path fails, and under `set -e` that would
# take down the whole run at step 8.
SETTINGS=".claude/settings.local.json"
if [ -f "$SETTINGS" ]; then
    if grep -q "$OLD_DIR" "$SETTINGS"; then
        repl_abs "$SETTINGS"
        echo "  fixed (gitignored — not committed): $SETTINGS"
    else
        echo "  SKIP (already fixed): $SETTINGS"
    fi
else
    echo "  WARNING: expected file not found, skipping: $SETTINGS"
fi
echo

# book/generators/*.py — docstring run-commands plus the
# REPO_ROOT.parent / "MasterThesisCode" sibling-checkout fallback. Bare-name
# replace (not full-path) since these reference the directory name in prose and
# fallback logic, not always the full absolute path. The -book guard is what
# keeps `# .../MasterThesisCode-book` REPO_ROOT comments honest.
GEN_COUNT=0
for f in book/generators/*.py; do
    # Match the bare name only where it is NOT the -book worktree.
    if [ -f "$f" ] && grep -qP "\Q$OLD_NAME\E(?!-book)" "$f"; then
        repl_name "$f"
        echo "  fixed: $f"
        TOUCHED+=("$f")
        GEN_COUNT=$((GEN_COUNT + 1))
    fi
done
echo "  book/generators/*.py: $GEN_COUNT file(s) updated."
echo
echo "  NOT rewritten, on purpose: the sibling \`-book\` worktree paths (that"
echo "        worktree is not being renamed), the github.io/$OLD_NAME/ Pages URLs"
echo "        and other dated snapshots per checklist §4, and §2's three"
echo "        scripts/bias_investigation cluster-path lines (cluster-rename script)."
echo

# --- Step 6: repo linkage — remote URL and linked worktrees --------------
#
# Two more things `mv` does not fix, both of which fail quietly:
#
#   (a) `origin` still points at github.com:JasperSeehofer/MasterThesisCode.
#       GitHub redirects a renamed repo, so push/fetch keep WORKING and only
#       print a "This repository moved" notice — which is easy to miss in a
#       scripted push. RUNBOOK-9 §3.1(c): redirects are a grace period, not a
#       permanent alias. Repoint it now rather than discover it when the
#       grace period ends.
#   (b) A linked worktree stores an ABSOLUTE path back to the main repo in
#       its `.git` file, so the sibling `-book` worktree points into the old
#       directory and every git command inside it dies with
#       `fatal: not a git repository: (null)`. `git worktree repair` is the
#       supported fix and is a no-op when nothing is broken.
echo "[6/9] Repairing repo linkage (remote URL, linked worktrees)..."
OLD_REMOTE_RE="[:/]JasperSeehofer/$OLD_NAME(\.git)?\$"
NEW_REMOTE="git@github.com:JasperSeehofer/$NEW_NAME.git"
CUR_REMOTE="$(git remote get-url origin 2>/dev/null || true)"
if [ -z "$CUR_REMOTE" ]; then
    echo "  WARNING: no 'origin' remote — skipping."
elif printf '%s' "$CUR_REMOTE" | grep -qE "$OLD_REMOTE_RE"; then
    git remote set-url origin "$NEW_REMOTE"
    echo "  remote origin: $CUR_REMOTE -> $NEW_REMOTE"
    echo "  (was riding GitHub's rename redirect — see RUNBOOK-9 §3.1(c))"
else
    echo "  SKIP: origin already at $CUR_REMOTE"
fi
if git worktree list --porcelain 2>/dev/null | grep -q '^worktree '; then
    git worktree repair 2>&1 | sed 's/^/  worktree repair: /' || true
    BROKEN=0
    while read -r _ wt; do
        [ -n "$wt" ] || continue
        [ "$wt" = "$NEW_DIR" ] && continue
        git -C "$wt" rev-parse --git-dir >/dev/null 2>&1 || { echo "  WARNING: still broken: $wt"; BROKEN=1; }
    done < <(git worktree list --porcelain | grep '^worktree ')
    [ "$BROKEN" -eq 0 ] && echo "  verified: all linked worktrees resolve."
else
    echo "  SKIP: no linked worktrees."
fi
echo

# --- Step 7: relocate the venv ------------------------------------------
#
# `mv` does not fix a venv. Every .venv/bin console script has the absolute
# interpreter path baked into its shebang, and the activate* scripts hardcode
# VIRTUAL_ENV. After the rename, `uv run mypy` fails with a bare
# "Failed to spawn: mypy" and `source .venv/bin/activate` puts a nonexistent
# directory on PATH — while `.venv/bin/python` itself keeps working, which makes
# the failure look like a missing dependency rather than a stale path.
# `uv sync --reinstall` rewrites the shebangs; the activate* scripts need sed.
echo "[7/9] Relocating the venv..."
if [ ! -d ".venv" ]; then
    echo "  SKIP: no .venv on this box."
elif ! grep -rlq "$OLD_DIR" .venv/bin/ .venv/pyvenv.cfg 2>/dev/null; then
    echo "  SKIP: no stale paths in .venv."
else
    if command -v uv >/dev/null 2>&1; then
        echo "  uv sync --reinstall $UV_SYNC_EXTRAS (rewrites console-script shebangs)..."
        # shellcheck disable=SC2086
        uv sync --reinstall $UV_SYNC_EXTRAS >/dev/null 2>&1 \
            || echo "  WARNING: uv sync failed — rerun it by hand; continuing."
    else
        echo "  WARNING: uv not on PATH — skipping the reinstall, patching paths only."
    fi
    # activate* scripts and any script uv did not regenerate.
    mapfile -t STALE < <(grep -rl "$OLD_DIR" .venv/bin/ .venv/pyvenv.cfg 2>/dev/null || true)
    if [ "${#STALE[@]}" -gt 0 ]; then
        sed -i "s|$OLD_DIR/|$NEW_DIR/|g" "${STALE[@]}"
        echo "  path-patched ${#STALE[@]} file(s) (activate*, stragglers)."
    fi
    sed -i "s|^prompt = master-thesis-code$|prompt = $NEW_NAME|" .venv/pyvenv.cfg 2>/dev/null || true
    if grep -rlq "$OLD_DIR" .venv/bin/ .venv/pyvenv.cfg 2>/dev/null; then
        echo "  WARNING: stale paths remain in .venv — inspect by hand."
    else
        echo "  verified: no stale paths left in .venv/bin or pyvenv.cfg."
    fi
fi
echo

# --- Step 8: commit ------------------------------------------------------
echo "[8/9] Committing §1/§3 reference fixes..."
if [ "${#TOUCHED[@]}" -eq 0 ]; then
    echo "  No files were touched — nothing to commit."
else
    # Drop anything gitignored; `git add` on an ignored path is a hard error.
    STAGE=()
    for f in "${TOUCHED[@]}"; do
        if git check-ignore -q "$f"; then
            echo "  not staging (gitignored): $f"
        else
            STAGE+=("$f")
        fi
    done
    if [ "${#STAGE[@]}" -eq 0 ]; then
        echo "  Everything touched is gitignored — nothing to commit."
    else
        git add "${STAGE[@]}"
        if git diff --cached --quiet; then
            echo "  Staged files produced no diff — nothing to commit."
        else
            echo "  NOTE: the pre-commit hooks are whole-tree ruff + ruff-format + mypy."
            echo "        ruff-format may reformat a staged file that had pre-existing"
            echo "        drift and ABORT the first attempt — re-'git add' and re-run"
            echo "        the commit, that is expected behaviour, not a failure."
            git commit -m "$(cat <<'EOF'
docs+scripts: local-path references -> darksiren-emri (migration §1/§3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
            echo "  committed."
        fi
    fi
fi
echo

# --- Step 9: final verification instructions -----------------------------
echo "[9/9] Local rename complete. To verify and close out §1 on this box:"
echo "  1. Open a FRESH Claude Code session in $NEW_DIR."
echo "  2. Confirm it retrieves prior project memory (MEMORY.md index entries"
echo "     from before the rename should be visible/loadable) AND that the"
echo "     garden session-start briefing fires — the briefing is the live proof"
echo "     that the registry Path column resolves."
echo "  3. Confirm the toolchain: 'uv run mypy --version' and 'uv run pytest --version'"
echo "     both answer (they are the two that break on a stale venv)."
echo "  4. Confirm 'git fetch' prints NO 'This repository moved' notice, and that"
echo "     git commands work inside each linked worktree."
echo "  5. docs/REBRAND_MIGRATION_CHECKLIST.md §6 tracks §1 as [~] until EVERY"
echo "     dev box has run this. Note this box against it, and flip to [x] on"
echo "     the last one."
echo
echo "=== done ==="
