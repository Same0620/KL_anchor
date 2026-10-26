#!/bin/bash
# Restores the files listed in REMOVED_BACKED_UP_2026-10-24.tsv (removed from loracle_em_exp1 on 2026-10-24 because they were
# byte-identical in the GH1 export) from GitHub: Same0620/KL_anchor, commit 62a0b832290d5bfacf9bae12658c53aea81c3eb9.
# The local export folder /work/u4161854/kl_line_export was deleted afterwards (it was fully pushed), so this clones from GitHub
# with the deploy key ~/.ssh/kl_line_export over ssh.github.com:443 (github.com:22 is blocked on the login nodes).
# Usage: bash restore_backed_up_2026-10-24.sh [TARGET_DIR]   (default: /work/u4161854/loracle_em_exp1)
set -euo pipefail
COMMIT=62a0b832290d5bfacf9bae12658c53aea81c3eb9
LIST=/work/u4161854/loracle_em_exp1/REMOVED_BACKED_UP_2026-10-24.tsv
TARGET=${1:-/work/u4161854/loracle_em_exp1}
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/kl_line_export -o IdentitiesOnly=yes -o HostName=ssh.github.com -o Port=443 -o UserKnownHostsFile=$HOME/.ssh/kl_line_export.known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"
TMP=$(mktemp -d /tmp/kl_anchor_restore.XXXXXX)
trap 'rm -rf "$TMP"' EXIT
git clone --quiet --no-checkout git@github.com:Same0620/KL_anchor.git "$TMP/repo"
test "$(git -C "$TMP/repo" rev-parse "$COMMIT^{commit}")" = "$COMMIT"
grep -v '^#' "$LIST" | cut -f2 | (cd "$TMP/repo" && xargs git archive "$COMMIT" --) | tar -x -C "$TARGET"
echo "restored $(grep -vc '^#' "$LIST") files into $TARGET"
