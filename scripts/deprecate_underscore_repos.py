"""
For each kernels-community repo whose name contains an underscore, checks if its
dash-equivalent counterpart exists. If it does, prepends a deprecation warning to
the underscore repo's README.

Usage:
    python scripts/deprecate_underscore_repos.py [--dry-run]
"""
import argparse
import sys

from huggingface_hub import list_models, upload_file, hf_hub_download, repo_info
from huggingface_hub.utils import RepositoryNotFoundError

ORG = "kernels-community"

WARNING_TEMPLATE = """> [!WARNING]
> This repository will soon be deleted as it's now deprecated. Please use [{org}/{dash_name}](https://huggingface.co/{org}/{dash_name}).

"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying any READMEs.",
    )
    return parser.parse_args()


def _get_readme(repo_id: str) -> str:
    try:
        path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _insert_warning(readme: str, warning: str) -> str:
    """Insert warning after YAML front matter if present, otherwise at the top."""
    if readme.startswith("---"):
        # Find the closing '---' of the front matter
        close = readme.find("\n---", 3)
        if close != -1:
            end = close + len("\n---")
            return readme[:end] + "\n\n" + warning + readme[end:]
    return warning + readme


def main() -> int:
    args = parse_args()

    # Collect all repos and filter for those with underscores in the repo name
    all_repos = list(list_models(author=ORG))
    print(f"Found {len(all_repos)} total repos under {ORG}.")

    underscore_repos = [
        repo for repo in all_repos
        if "_" in repo.id.removeprefix(f"{ORG}/")
    ]
    print(f"Found {len(underscore_repos)} repos with underscore names.\n")

    opened_prs = []

    for repo in underscore_repos:
        repo_name = repo.id.removeprefix(f"{ORG}/")
        dash_name = repo_name.replace("_", "-")
        dash_repo_id = f"{ORG}/{dash_name}"

        # Check if the dash counterpart exists
        try:
            repo_info(repo_id=dash_repo_id, repo_type="model")
        except RepositoryNotFoundError:
            print(f"[skip] {repo_name}: no dash counterpart ({dash_name}) found.")
            continue

        print(f"[found] {repo_name} -> {dash_name} exists.")

        warning = WARNING_TEMPLATE.format(org=ORG, dash_name=dash_name)
        current_readme = _get_readme(repo.id)

        # Skip if the warning is already present
        if f"/{dash_name}" in current_readme and "[!WARNING]" in current_readme:
            print(f"  Warning already present in {repo_name}, skipping.")
            continue

        new_readme = _insert_warning(current_readme, warning)

        if args.dry_run:
            print(f"  [dry-run] Would open a PR to prepend deprecation warning to {repo.id}/README.md.")
            continue

        commit_info = upload_file(
            path_or_fileobj=new_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo.id,
            repo_type="model",
            commit_message=f"chore: mark as deprecated in favour of {dash_name}",
            create_pr=True,
        )
        pr_url = commit_info.pr_url
        opened_prs.append(pr_url)
        print(f"  Opened PR for {repo.id}: {pr_url}")

    if opened_prs:
        print(f"\nOpened {len(opened_prs)} PR(s):")
        for url in opened_prs:
            print(f"  {url}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
