#!/usr/bin/env bash
# One-time backfill script to sync all existing kernel repos to their model repo counterparts.
# Run from the root of the kernels-community repo checkout:
#   cd /path/to/kernels-community && bash scripts/backfill-sync.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

pip install -U huggingface_hub

# Find all directories containing a build.toml — these are kernel directories.
# This skips .github, .claude, scripts, and any other non-kernel directories.
for build_toml in */build.toml; do
  [ -f "$build_toml" ] || continue  # guard against no matches

  kernel=$(dirname "$build_toml")
  repo_id="kernels-community/$kernel"
  version=$(grep -oP 'version\s*=\s*\K\d+' "$build_toml" | head -1)

  echo "==> Syncing $repo_id (v$version)..."
  python3 "$REPO_ROOT/.github/scripts/sync_kernel_to_model.py" "$repo_id" "v$version"

  if [ "$version" = "1" ]; then
    echo "    Also syncing $repo_id main branch..."
    python3 "$REPO_ROOT/.github/scripts/sync_kernel_to_model.py" "$repo_id" "main"
  fi

  echo ""
done

echo "Backfill complete."
