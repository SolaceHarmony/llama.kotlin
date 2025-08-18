#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: revert-copilot-merge.sh <merge-sha> [<merge-sha>...]

Reverts one or more merge commits into the current branch using mainline parent 1.
Intended for reverting merges that pulled in closed-unmerged Copilot PR branches.

Requirements:
- Clean working tree
- On the target integration branch (e.g., copilot/fix-45)

Notes:
- This performs one revert per SHA and commits after each revert.
- If a revert causes conflicts, fix them and run: git revert --continue

Examples:
  ./scripts/revert-copilot-merge.sh ba214cd6
  ./scripts/revert-copilot-merge.sh ba214cd6 b781be40
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

# Ensure clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is not clean. Stash or commit your changes first." >&2
  exit 2
fi

# Show current branch
branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $branch"

for sha in "$@"; do
  if ! git cat-file -e "$sha^{commit}" 2>/dev/null; then
    echo "Error: $sha is not a commit" >&2
    exit 3
  fi

  # Check it's a merge commit (2+ parents)
  parents=$(git rev-list --parents -n 1 "$sha" | wc -w)
  if [[ $parents -lt 3 ]]; then
    echo "Warning: $sha is not a merge commit (parents=$((parents-1))). Proceeding anyway." >&2
  else
    echo "Reverting merge commit $sha with mainline parent 1..."
  fi

  # Try revert with mainline parent 1, commit immediately
  if ! git revert -m 1 "$sha" --no-edit; then
    echo "Revert of $sha encountered conflicts. Resolve them, then run: git revert --continue" >&2
    exit 4
  fi
  echo "Reverted $sha"

done

echo "Done. Consider running: ./gradlew assemble"
