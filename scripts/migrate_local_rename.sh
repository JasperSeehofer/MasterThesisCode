#!/usr/bin/env bash
# migrate_local_rename.sh — Rebrand Migration Checklist §1/§3
#
# Renames the local dev-box checkout from MasterThesisCode to darksiren-emri
# and re-keys the two systems that key off the old filesystem path (Claude
# Code project memory, garden registry), then applies the §3 mechanical
# reference fixes inside the renamed repo.
#
# PRECONDITIONS (both hard requirements):
#   1. Run this script FROM $HOME.
#   2. Run it with NO Claude Code session open anywhere under the repo —
#      an open session holds file handles / working-directory state keyed
#      to the old path, and re-keying its project directory (step 3) out
#      from under a live session will orphan that session.
#
# Usage: cd ~ && bash /home/jasper/Repositories/MasterThesisCode/scripts/migrate_local_rename.sh
#
# See docs/REBRAND_MIGRATION_CHECKLIST.md §1 and §3 for the source plan.

set -euo pipefail

OLD_DIR="/home/jasper/Repositories/MasterThesisCode"
NEW_DIR="/home/jasper/Repositories/darksiren-emri"
OLD_NAME="MasterThesisCode"
NEW_NAME="darksiren-emri"

echo "=== migrate_local_rename.sh — Rebrand Migration Checklist §1/§3 ==="
echo

# --- Precondition: must be run from $HOME -----------------------------
if [ "$PWD" != "$HOME" ]; then
    echo "ABORT: must be run from \$HOME ($HOME); currently in $PWD" >&2
    echo "       cd ~ and re-run: bash $OLD_DIR/scripts/migrate_local_rename.sh" >&2
    exit 1
fi
echo "[precondition] Running from \$HOME ($HOME) — OK."
echo "[precondition] REMINDER: this must run with no Claude Code session open"
echo "               in the repo. If one is open, close it now and re-run."
echo

# --- Step 1: preflight existence checks --------------------------------
echo "[1/7] Preflight: verifying source exists and target does not..."
if [ -d "$NEW_DIR" ]; then
    echo "ABORT: target $NEW_DIR already exists — refusing to overwrite." >&2
    exit 1
fi
if [ ! -d "$OLD_DIR" ]; then
    echo "ABORT: source $OLD_DIR does not exist — nothing to rename." >&2
    exit 1
fi
echo "  OK: $OLD_DIR exists, $NEW_DIR does not."
echo

# --- Step 2: rename the repo directory ----------------------------------
echo "[2/7] Renaming repo directory..."
mv "$OLD_DIR" "$NEW_DIR"
echo "  mv: $OLD_DIR -> $NEW_DIR"
if [ ! -d "$NEW_DIR" ]; then
    echo "ABORT: rename did not take effect — $NEW_DIR missing after mv." >&2
    exit 1
fi
echo "  verified: $NEW_DIR exists."
echo

# --- Step 3: re-key Claude Code project memory / session state ---------
echo "[3/7] Re-keying Claude Code project memory/session state..."
CLAUDE_OLD="$HOME/.claude/projects/-home-jasper-Repositories-MasterThesisCode"
CLAUDE_NEW="$HOME/.claude/projects/-home-jasper-Repositories-darksiren-emri"
if [ -d "$CLAUDE_NEW" ]; then
    echo "ABORT: $CLAUDE_NEW already exists — refusing to overwrite Claude project state." >&2
    echo "       The repo directory has already been moved to $NEW_DIR; resolve the" >&2
    echo "       Claude-side conflict by hand, then re-run from step 4 manually." >&2
    exit 1
fi
if [ -d "$CLAUDE_OLD" ]; then
    mv "$CLAUDE_OLD" "$CLAUDE_NEW"
    echo "  mv: $CLAUDE_OLD -> $CLAUDE_NEW"
else
    echo "  WARNING: $CLAUDE_OLD not found — nothing to re-key here, continuing."
fi
echo

# --- Step 4: garden registry Path column --------------------------------
#
# The garden's session-start hook (wiki/assets/claude-hooks/wiki-session-start.sh)
# finds a project by testing whether the registry Path column is a PREFIX of $CWD,
# and `exit 0`s silently when nothing matches. So a renamed directory with a stale
# registry Path does not error — it just stops briefing this repo, with no signal.
# That makes this step a HARD requirement, not a nicety: abort rather than warn.
#
# The registry lives at wiki/meta/registry.md, NOT at the repo root. (An earlier
# revision of this script pointed at $GARDEN/registry.md, which does not exist —
# it would have taken the "skipping" branch and produced exactly the silent
# no-briefing failure described above.)
echo "[4/7] Updating garden registry..."
GARDEN="$HOME/Repositories/garden"
REGISTRY="$GARDEN/wiki/meta/registry.md"
if [ ! -f "$REGISTRY" ]; then
    echo "ABORT: $REGISTRY not found." >&2
    echo "       The repo directory has already been moved to $NEW_DIR and Claude" >&2
    echo "       project memory re-keyed. Fix the garden checkout, then update the" >&2
    echo "       Path column by hand and continue from step 5." >&2
    exit 1
fi
cp "$REGISTRY" "$REGISTRY.bak"
echo "  backup: $REGISTRY.bak"
if ! grep -q "$OLD_DIR" "$REGISTRY"; then
    echo "ABORT: zero occurrences of $OLD_DIR in $REGISTRY." >&2
    echo "       Either the Path column was already migrated, or the registry row" >&2
    echo "       is keyed differently than expected. Verify by hand before going on —" >&2
    echo "       a wrong Path here silently disables the vault briefing for this repo." >&2
    exit 1
fi
sed -i "s|$OLD_DIR|$NEW_DIR|g" "$REGISTRY"
echo "  updated line(s):"
grep -n "$NEW_DIR" "$REGISTRY" | sed 's/^/    /'
# Verify the hook's own prefix test now succeeds against the new path.
if awk -F '|' -v new="$NEW_DIR" '
      /\|/ && !/---/ && !/Project.*Path/ {
        gsub(/^ +| +$/,"",$3); if ($3 == new) found=1
      } END { exit found ? 0 : 1 }' "$REGISTRY"; then
    echo "  verified: a Path column now equals $NEW_DIR (hook prefix test will match)"
else
    echo "ABORT: no registry Path column exactly equals $NEW_DIR after the edit." >&2
    echo "       Restore $REGISTRY.bak and investigate — the briefing hook would" >&2
    echo "       silently stop firing for this repo." >&2
    exit 1
fi
echo "  NOTE: the vault slug stays 'master-thesis-code'. Downstream tables"
echo "        (interaction-feedback reminders, briefing-feedback, portfolio-health,"
echo "        context-budget, agent-weaknesses) key on the SLUG, not the path, so"
echo "        they keep working untouched. Renaming the slug is a separate,"
echo "        optional migration — see the garden's rename plan."
echo "  TODO(manual): commit the garden change — the vault is a git repo and this"
echo "        script deliberately does not commit on your behalf outside this repo."
echo

# --- Step 5: §3 reference fixes inside the renamed repo -----------------
echo "[5/7] Applying §3 reference fixes inside $NEW_DIR..."
cd "$NEW_DIR"

TOUCHED=()

# Files carrying the literal old absolute path (checklist §3, verbatim list).
SIMPLE_FILES=(
    ".claude/skills/known-bugs/SKILL.md"
    ".claude/skills/physics-change/SKILL.md"
    "darksiren_emri_test/bayesian_inference/test_posterior_combination.py"
    "book/design/BOOK_DESIGN.md"
    "book/design/BOOK_SOURCES_MAP.md"
    "book/design/BOOK_TECH_DESIGN.md"
    "book/design/reviews/expert_A_ch00-06_museum.md"
    "book/design/reviews/expert_B_ch07-11_cellB.md"
)
for f in "${SIMPLE_FILES[@]}"; do
    if [ -f "$f" ]; then
        if grep -q "$OLD_DIR" "$f"; then
            sed -i "s|$OLD_DIR|$NEW_DIR|g" "$f"
            echo "  fixed: $f"
            TOUCHED+=("$f")
        else
            echo "  WARNING: $f has no occurrences of $OLD_DIR — skipping (already fixed?)."
        fi
    else
        echo "  WARNING: expected file not found, skipping: $f"
    fi
done

# .claude/settings.local.json — lines 12,13 carry the repo path in Bash
# allow-rules; fix those. Line 9 is the sibling `-book` worktree path
# (MasterThesisCode-book) — leave it untouched here.
SETTINGS=".claude/settings.local.json"
if [ -f "$SETTINGS" ]; then
    if grep -q "$OLD_DIR" "$SETTINGS"; then
        sed -i "9!s|$OLD_DIR|$NEW_DIR|g" "$SETTINGS"
        echo "  fixed (line 9 intentionally skipped): $SETTINGS"
        TOUCHED+=("$SETTINGS")
    else
        echo "  WARNING: $SETTINGS has no occurrences of $OLD_DIR — skipping."
    fi
else
    echo "  WARNING: expected file not found, skipping: $SETTINGS"
fi
echo
echo "  NOTE: $SETTINGS line 9 (the sibling \`-book\` worktree path,"
echo "        MasterThesisCode-book) was left as-is. If that worktree"
echo "        directory is ALSO being renamed (to darksiren-emri-book),"
echo "        update this line by hand afterward — it is out of scope"
echo "        for this script."
echo

# book/generators/*.py — shebang comments + the
# REPO_ROOT.parent / "MasterThesisCode" sibling-checkout fallback.
# Word-level replace (not full-path) since these files reference the bare
# directory name in prose/fallback logic, not always the full absolute path.
GEN_COUNT=0
for f in book/generators/*.py; do
    if [ -f "$f" ] && grep -q "$OLD_NAME" "$f"; then
        sed -i "s|$OLD_NAME|$NEW_NAME|g" "$f"
        echo "  fixed: $f"
        TOUCHED+=("$f")
        GEN_COUNT=$((GEN_COUNT + 1))
    fi
done
echo "  book/generators/*.py: $GEN_COUNT file(s) updated."
echo

echo "  (§2's three scripts/bias_investigation cluster-path references are"
echo "   deferred to the cluster-rename migration script, per checklist §3.)"
echo

# --- Step 6: commit ------------------------------------------------------
echo "[6/7] Committing §1/§3 reference fixes..."
if [ "${#TOUCHED[@]}" -eq 0 ]; then
    echo "  No files were touched — nothing to commit."
else
    git add "${TOUCHED[@]}"
    if git diff --cached --quiet; then
        echo "  Staged files produced no diff — nothing to commit."
    else
        git commit -m "$(cat <<'EOF'
docs+scripts: local-path references -> darksiren-emri (migration §1/§3)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Vf92KFbg1F213zhYZKAKR2
EOF
)"
        echo "  committed."
    fi
fi
echo

# --- Step 7: final verification instructions -----------------------------
echo "[7/7] Local rename complete. To verify and close out §1:"
echo "  1. Open a FRESH Claude Code session in $NEW_DIR."
echo "  2. Confirm it retrieves prior project memory (MEMORY.md index entries"
echo "     from before the rename should be visible/loadable)."
echo "  3. Tick the §1 box in docs/REBRAND_MIGRATION_CHECKLIST.md:"
echo "     '§1: fresh Claude Code session in the renamed local directory"
echo "     retrieves prior project memory' -> [x], with today's date."
echo "  4. If the garden registry backup ($REGISTRY.bak) is no longer needed"
echo "     once the update is confirmed correct, remove it by hand."
echo
echo "=== done ==="
