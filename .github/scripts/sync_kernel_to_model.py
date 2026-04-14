"""Syncs a kernel repo to its model repo counterpart using huggingface_hub.

Usage: python sync_kernel_to_model.py <repo_id> <branch>
Example: python sync_kernel_to_model.py kernels-community/flash-attn3 v1

Requires: HF_TOKEN env var, huggingface_hub installed
"""

import sys

from huggingface_hub import HfApi


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <repo_id> <branch>")
        print(f"Example: {sys.argv[0]} kernels-community/flash-attn3 v1")
        sys.exit(1)

    repo_id = sys.argv[1]
    branch = sys.argv[2]

    print(f"Syncing kernel repo '{repo_id}' branch '{branch}' -> model repo...")

    api = HfApi()

    # Download snapshot from "kernel" repo type
    local_dir = api.snapshot_download(
        repo_id=repo_id,
        repo_type="kernel",
        revision=branch,
    )

    # Upload the same tree to "model" repo type
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        revision=branch,
        commit_message=f"sync: mirror kernel repo ({branch})",
    )

    print(f"Successfully synced {repo_id} ({branch}) from kernel -> model repo type.")


if __name__ == "__main__":
    main()
