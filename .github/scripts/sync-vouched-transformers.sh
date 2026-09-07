#!/usr/bin/env bash
# Add members of the huggingface `transformers` and
# `transformers-core-maintainers` teams to .github/VOUCHED.td.
#
# Vouching is additive only: existing entries, header comments, and
# denouncements (-username reason) are preserved; new team members are
# merged in and the handle list is re-sorted case-insensitively.
#
# Requires: gh (authenticated with read access to the huggingface org).
#
# Usage:
#   .github/scripts/sync-vouched-transformers.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
VOUCHED="$REPO_ROOT/.github/VOUCHED.td"
TEAMS="transformers transformers-core-maintainers"

members="$(mktemp)"
trap 'rm -f "$members"' EXIT

for team in $TEAMS; do
  gh api --paginate "orgs/huggingface/teams/$team/members" --jq '.[].login' >>"$members"
done

{
  # Header comments, unchanged.
  grep '^#' "$VOUCHED" || true

  # Union of existing handles and team members. Existing entries come
  # first so their casing wins on case-insensitive duplicates.
  {
    grep -v '^#' "$VOUCHED" | grep -v '^-' || true
    cat "$members"
  } | sed '/^[[:space:]]*$/d' | awk '!seen[tolower($0)]++' | sort -f

  # Denouncements, unchanged.
  grep '^-' "$VOUCHED" || true
} >"$VOUCHED.new"

mv "$VOUCHED.new" "$VOUCHED"
