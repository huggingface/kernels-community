#!/usr/bin/env bash

set -euo pipefail

KERNELS=$(find . -maxdepth 2 -name "build.toml" -exec dirname {} \; | sed 's|^\./||' | sort)

FAILED_LIST=/tmp/verify-failed
: > "$FAILED_LIST"

{
  for kernel in $KERNELS; do
    echo "::group::$kernel"

    VERSION=$(awk -F'[[:space:]]*=[[:space:]]*' '/^version/ {print $2; exit}' "$kernel/build.toml")

    # Continue on failure to verify all kernels before reporting.
    if ! nix run -L github:huggingface/kernels#kernels -- \
      verify-signature --all-variants "kernels-community/$kernel" "$VERSION"; then
      echo "::warning::$kernel verification FAILED"
      echo "$kernel" >> "$FAILED_LIST"
    fi

    # Remove the HF cache to reclaim disk space before the next kernel.
    rm -rf ~/.cache/huggingface/hub

    echo "::endgroup::"
  done

  FAILURES=$(wc -l < "$FAILED_LIST" | tr -d '[:space:]')
  echo ""
  if [ "$FAILURES" -ne 0 ]; then
    echo "❌ Verification failed for $FAILURES kernel(s):"
    sed 's/^/  - /' "$FAILED_LIST"
    echo ""
    echo "See the corresponding groups above for details."
  else
    echo "✅ All kernels verified successfully."
  fi
} 2>&1 | tee /tmp/verify-output.log

[ -s "$FAILED_LIST" ] && exit 1 || exit 0
