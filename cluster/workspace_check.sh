#!/usr/bin/env bash
# cluster/workspace_check.sh — is the cluster workspace healthy, and did a move complete?
#
# The workspace is one of the two copies of record of the cluster data (the other is
# the off-site Storage Box, tag `hbox` in DATA_INVENTORY.md). bwHPC workspaces expire
# and are then deleted with no recovery, so the workspace is rolled forward: extend
# while extensions last, then allocate a new one and move the contents. The procedure
# is "Workspace lifecycle" in .claude/skills/cluster/SKILL.md; this script is its check.
#
#   bash ~/darksiren-emri/cluster/workspace_check.sh                       # check (read-only)
#   bash ~/darksiren-emri/cluster/workspace_check.sh --rewrite-links <old-ws-path>
#        # after a move: re-point symlinks inside the workspace from <old-ws-path>/...
#        # to the same path under the current workspace (only links whose target
#        # exists there; everything else is left and reported)
# Agents: cluster/agent_ssh.sh run 'bash ~/darksiren-emri/cluster/workspace_check.sh'
#
# Checks: (1) the name in cluster/cluster.env resolves; (2) expiry / extensions — the
# same "extend / plan the move / move now" rule as preflight.sh; (3) other workspaces
# still holding data (an unfinished move); (4) symlinks pointing into another
# workspace (they dangle once that one expires); (5) cluster/datasets.yaml
# meta.workspace and cluster.env agree with the live path and are committed.
# Exit 0 = OK (warnings allowed), 1 = problems, 2 = usage.

set -u
REPO="${CLUSTER_REPO:-$HOME/darksiren-emri}"
WS_NAME="${CLUSTER_WORKSPACE_NAME:-$( . "$REPO/cluster/cluster.env" >/dev/null 2>&1; echo "${CLUSTER_WORKSPACE_NAME:-}")}"
WARN_DAYS="${EMRI_WORKSPACE_WARN_DAYS:-14}"
LINK_DEPTH="${WS_CHECK_LINK_DEPTH:-6}"

PROBLEMS=(); WARNINGS=()
problem () { PROBLEMS+=("$1"); echo "        PROBLEM: $1"; }
warn ()    { WARNINGS+=("$1"); echo "        WARNING: $1"; }

# ws_list block for one workspace -> "days|expiry|extensions" (days=0 when under a day)
ws_facts () {
  local info rem days ext exp
  info=$(ws_list "$1" 2>/dev/null)
  rem=$(printf '%s\n' "$info" | grep -i 'remaining time' | head -1 | sed 's/^[^:]*: *//')
  days=$(printf '%s\n' "$rem" | grep -oE '[0-9]+ days?' | grep -oE '[0-9]+' | head -1)
  [ -z "$days" ] && [ -n "$rem" ] && days=0
  ext=$(printf '%s\n' "$info" | grep -i 'available extensions' | head -1 | grep -oE '[0-9]+' | head -1)
  exp=$(printf '%s\n' "$info" | grep -i 'expiration date' | head -1 | sed 's/^[^:]*: *//')
  echo "${days}|${exp}|${ext}"
}

mode=check; OLD=""
case "${1:-}" in
  "") ;;
  --rewrite-links) mode=rewrite; OLD="${2:?--rewrite-links needs the old workspace path}"; OLD="${OLD%/}" ;;
  -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
  *) echo "usage: $0 [--rewrite-links <old-ws-path>]" >&2; exit 2 ;;
esac

echo "======================================================================"
echo " WORKSPACE CHECK  —  $(hostname)  —  $(date '+%Y-%m-%d %H:%M %Z')"
echo "======================================================================"

# (1) name -> path
echo "[NAME]  CLUSTER_WORKSPACE_NAME='${WS_NAME:-<unset>}'  (cluster/cluster.env)"
WS_PATH=""
[ -n "$WS_NAME" ] && WS_PATH=$(ws_find "$WS_NAME" 2>/dev/null)
if [ -z "$WS_NAME" ]; then
  problem "CLUSTER_WORKSPACE_NAME is not set in $REPO/cluster/cluster.env"
elif [ -z "$WS_PATH" ]; then
  problem "workspace '$WS_NAME' not found (ws_find) — wrong name in cluster.env, or not allocated yet"
else
  echo "        path=$WS_PATH"
fi

if [ "$mode" = rewrite ]; then
  [ -n "$WS_PATH" ] || { echo "cannot rewrite links: no current workspace"; exit 1; }
  [ "$OLD" != "$WS_PATH" ] || { echo "old path equals the current workspace — nothing to do"; exit 2; }
  echo "[REWRITE]  links -> $OLD/...  become  -> $WS_PATH/..."
  n=0; skipped=0
  while IFS= read -r -d '' l; do
    t=$(readlink "$l")
    case "$t" in "$OLD"/*) ;; *) continue ;; esac
    nt="$WS_PATH/${t#"$OLD"/}"
    if [ -e "$nt" ]; then ln -sfn "$nt" "$l" && n=$((n+1))
    else echo "        kept (no target under the new workspace): $l -> $t"; skipped=$((skipped+1)); fi
  done < <(find "$WS_PATH" -maxdepth "$LINK_DEPTH" -type l -print0 2>/dev/null)
  echo "        rewritten: $n   kept: $skipped"
  echo "        re-run without options to check the result"
  exit 0
fi

if [ -n "$WS_PATH" ]; then
  # (2) expiry and extensions
  IFS='|' read -r DAYS EXP EXT <<< "$(ws_facts "$WS_NAME")"
  echo "[EXPIRY]  remaining=${DAYS:-?} days  expires=${EXP:-?}  extensions_left=${EXT:-?}"
  if [ -n "$DAYS" ] && [ "$DAYS" -lt "$WARN_DAYS" ]; then
    if [ "${EXT:-1}" -gt 0 ]; then problem "expires in ${DAYS} d — extend now: ws_extend $WS_NAME 60"
    else problem "expires in ${DAYS} d with 0 extensions left — MOVE NOW (cluster skill, 'Workspace lifecycle')"; fi
  elif [ "${EXT:-}" = 0 ]; then
    warn "0 extensions left — expiry ${EXP:-?} is final; schedule the move (cluster skill, 'Workspace lifecycle')"
  else
    echo "        ok"
  fi

  # (3) other workspaces — an old one that still holds data means the move is unfinished
  echo "[OTHER WORKSPACES]"
  others=$(ws_list 2>/dev/null | sed -n 's/^id: *//p' | grep -vx "$WS_NAME")
  if [ -z "$others" ]; then echo "        none"; fi
  for o in $others; do
    op=$(ws_find "$o" 2>/dev/null); IFS='|' read -r od oe _ <<< "$(ws_facts "$o")"
    cnt=$( [ -n "$op" ] && find "$op" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ' )
    echo "        $o  path=${op:-?}  entries=${cnt:-?}  expires=${oe:-?} (${od:-?} d)"
    [ "${cnt:-0}" -gt 0 ] && warn "workspace '$o' still holds ${cnt} entries — unfinished move? (it is deleted on expiry)"
  done

  # (4) symlinks that point into another workspace dangle once it expires
  echo "[SYMLINKS]  into other workspaces (depth ≤ $LINK_DEPTH)"
  # any absolute target under a .../workspace/... tree other than the current workspace
  foreign=$(find "$WS_PATH" -maxdepth "$LINK_DEPTH" -type l -printf '%p -> %l\n' 2>/dev/null \
            | awk -v cur="$WS_PATH/" '{ i = index($0, " -> "); t = substr($0, i + 4) }
                                      t ~ /^\/.*\/workspace\// && index(t, cur) != 1')
  nf=$(printf '%s' "$foreign" | grep -c .)
  if [ "$nf" -eq 0 ]; then echo "        none"
  else
    printf '%s\n' "$foreign" | head -5 | sed 's/^/        /'
    problem "$nf symlink(s) point into another workspace — fix: bash $REPO/cluster/workspace_check.sh --rewrite-links <old-ws-path>"
  fi
fi

# (5) registry + config agree and are committed
echo "[REGISTRY]"
if [ -n "$WS_PATH" ]; then
  reg=$(sed -n 's/^  workspace: *\([^ #]*\).*/\1/p' "$REPO/cluster/datasets.yaml" 2>/dev/null | head -1)
  if [ "$reg" = "$WS_PATH" ]; then echo "        datasets.yaml meta.workspace matches"
  else warn "cluster/datasets.yaml meta.workspace is '${reg:-?}', live path is '$WS_PATH' — update it and the DATA_INVENTORY.md workspace-lifetime table"; fi
fi
if git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  if [ -n "$(git -C "$REPO" status --porcelain -- cluster/cluster.env cluster/datasets.yaml 2>/dev/null)" ]; then
    warn "cluster/cluster.env or cluster/datasets.yaml has uncommitted changes on the cluster — commit on the dev box, push, git pull here"
  else echo "        cluster.env + datasets.yaml committed"; fi
fi

echo "----------------------------------------------------------------------"
if [ ${#PROBLEMS[@]} -gt 0 ]; then
  echo " VERDICT: PROBLEMS — ${#PROBLEMS[@]}"; for p in "${PROBLEMS[@]}"; do echo "   • $p"; done
  [ ${#WARNINGS[@]} -gt 0 ] && { echo " ALSO:"; for w in "${WARNINGS[@]}"; do echo "   • $w"; done; }
  exit 1
fi
if [ ${#WARNINGS[@]} -gt 0 ]; then echo " VERDICT: OK (WARN: ${#WARNINGS[@]})"; for w in "${WARNINGS[@]}"; do echo "   • $w"; done
else echo " VERDICT: OK ✓"; fi
exit 0
