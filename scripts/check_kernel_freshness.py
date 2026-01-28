import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request
import random


KERNEL_SOURCE_MAPPING = {
    "activation": "https://github.com/vllm-project/vllm",
    "causal-conv1d": "https://github.com/Dao-AILab/causal-conv1d",
    "deformable-detr": "https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch",
    "flash-attn2": "https://github.com/Dao-AILab/flash-attention",
    "flash-attn3": "https://github.com/Dao-AILab/flash-attention",
    "fp8-fbgemm": "https://github.com/pytorch/FBGEMM",
    "gpt-oss-metal-kernels": "https://github.com/openai/gpt-oss",
    "layer_norm": "https://github.com/Dao-AILab/flash-attention",
    "mamba-ssm": "https://github.com/state-spaces/mamba",
    "megablocks": "https://github.com/databricks/megablocks",
    "mra": "",  # TODO: Add source URL
    "paged-attention": "https://github.com/vllm-project/vllm",
    "punica-sgmv": "https://github.com/punica-ai/punica",
    "quantization-bitsandbytes": "https://github.com/bitsandbytes-foundation/bitsandbytes",
    "quantization-eetq": "https://github.com/NetEase-FuXi/EETQ",
    "quantization-gptq": "https://github.com/IST-DASLab/gptq",
    "rmsnorm": "https://github.com/intel/intel-extension-for-pytorch",
    "rotary": "https://github.com/Dao-AILab/flash-attention",
    "rwkv": "https://github.com/BlinkDL/RWKV-LM",
    "scattermoe": "https://github.com/shawntan/scattermoe",
    "tinygrad-rms": "https://github.com/tinygrad/tinygrad",
    "trimul_gpumode": "https://github.com/davidberard98/gpumode-trimul",
    "vllm-flash-attn3": "https://github.com/Dao-AILab/flash-attention",
    "yoso": "https://github.com/mlpen/YOSO",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory of the kernels-community repository.",
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Specify if the number of kernel to be checked should be limited. If specified, `limit` number of entries will be chosen randomly from `KERNEL_SOURCE_MAPPING`, Must specify `--dry-run` to enable."
    )
    parser.add_argument(
        "--slack-webhook", default=os.getenv("SLACK_WEBHOOK_URL"),
    )
    parser.add_argument(
        "--github-token", default=os.getenv("GITHUB_TOKEN"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without posting to Slack.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    if args.limit and not args.dry_run:
        raise ValueError("Must set `--dry-run` when supplying `--limit`.")

    return args

def _random_subdict(d: dict, n: int) -> dict:
    if n > len(d):
        raise ValueError("n cannot be larger than the dict size")
    keys = random.sample(list(d.keys()), n)
    return {k: d[k] for k in keys}


def _github_api_request(url: str, token: str | None = None) -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    req = request.Request(url, headers=headers)
    try:
        with request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except error.HTTPError as e:
        if e.code == 422:
            # 422 means the branch does not exist and we only check main/master
            # so we can ignore this error 
            raise
        print(f"HTTP error {e.code} for {url}: {e.reason}")
        raise
    except error.URLError as e:
        print(f"URL error for {url}: {e.reason}")
        raise


def _get_upstream_last_commit_date(repo_url: str, token: str | None = None) -> datetime | None:
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    for branch in ["main", "master"]:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}"
        try:
            data = _github_api_request(api_url, token)
            commit_date_str = data["commit"]["committer"]["date"]
            # ISO 8601 format: 2024-01-27T12:34:56Z
            commit_date = datetime.strptime(commit_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            logging.debug(f"Upstream {repo_url} ({branch}): last commit {commit_date}")
            return commit_date
        except (error.HTTPError, KeyError) as e:
            if branch == "master":
                print(f"Failed to get commit from {repo_url}: {e}")
                return None
            continue

    return None


def _get_local_kernel_last_commit_date(kernel_dir: Path) -> datetime | None:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(kernel_dir)],
            capture_output=True,
            text=True,
            check=True,
            cwd=kernel_dir.parent,
        )
        commit_date_str = result.stdout.strip()
        if not commit_date_str:
            print(f"No commits found for {kernel_dir}")
            return None

        # ISO 8601 format with timezone
        commit_date = datetime.fromisoformat(commit_date_str)
        logging.debug(f"Local {kernel_dir.name}: last commit {commit_date}")
        return commit_date
    except subprocess.CalledProcessError as e:
        print(f"Failed to get git log for {kernel_dir}: {e}")
        return None


def _check_single_kernel(
    kernel_dir: str,
    source_url: str,
    root_path: Path,
    github_token: str | None = None,
) -> dict | None:
    print(f"Checking {kernel_dir}...")

    kernel_path = root_path / kernel_dir
    if not kernel_path.exists():
        print(f"Kernel directory {kernel_path} does not exist, skipping")
        return None

    if not source_url:
        print(f"Skipping {kernel_dir}: no source URL configured")
        return None

    upstream_date = _get_upstream_last_commit_date(source_url, github_token)
    if upstream_date is None:
        print(f"Could not get upstream date for {kernel_dir}, skipping")
        return None

    local_date = _get_local_kernel_last_commit_date(kernel_path)
    if local_date is None:
        print(f"Could not get local date for {kernel_dir}, skipping")
        return None

    diff = upstream_date - local_date
    days_behind = diff.days

    if days_behind > 0:
        print(f"  {kernel_dir}: upstream is {days_behind} days newer")
        return {
            "kernel_dir": kernel_dir,
            "source_url": source_url,
            "days_behind": days_behind,
            "upstream_date": upstream_date,
            "local_date": local_date,
        }
    else:
        print(f"  {kernel_dir}: up to date")
        return None


def check_kernel_freshness(
    root_path: Path, 
    github_token: str | None = None, 
    max_workers: int = 10,
    limit: int = None
) -> list[dict]:
    mapping_to_use = KERNEL_SOURCE_MAPPING

    if limit:
        print(f"Limiting the check to {limit} kernels.")
        mapping_to_use = _random_subdict(mapping_to_use, limit)

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_kernel = {
            executor.submit(
                _check_single_kernel,
                kernel_dir,
                source_url,
                root_path,
                github_token,
            ): kernel_dir
            for kernel_dir, source_url in mapping_to_use.items()
        }

        for future in as_completed(future_to_kernel):
            kernel_dir = future_to_kernel[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f"Error checking {kernel_dir}: {exc}")

    return results


def _format_freshness_report(results: list[dict]) -> str:
    sorted_results = sorted(results, key=lambda x: x["days_behind"], reverse=True)

    heading = f"📊 Kernel Freshness Report - {len(sorted_results)} kernel(s) behind upstream"

    items = []
    for result in sorted_results:
        items.append(f"• {result['source_url']}: upstream is {result['days_behind']} days newer")

    report = f"{heading}\n\n" + "\n".join(items)

    repository = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if repository and run_id:
        action_url = f"https://github.com/{repository}/actions/runs/{run_id}"
        report += f"\n\nRun: {action_url}"

    return report


def _post_to_slack(webhook_url: str, message: str) -> None:
    payload = {"text": message}

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as response:
        response.read()


def main() -> int:
    args = parse_args()

    root_path = Path(args.root).resolve()
    print(f"Using root path: {root_path}")
    print(f"Using {args.max_workers} worker threads for parallel checking")

    results = check_kernel_freshness(root_path, args.github_token, args.max_workers, args.limit)

    if not results:
        print("\n✅ All kernel directories are up to date with their upstream sources!")
        return 0

    report = _format_freshness_report(results)
    print("\n" + report)

    if args.dry_run:
        print("Dry-run mode; skipping Slack notification.")
        return 0

    if not args.slack_webhook:
        print("Slack webhook URL is not provided; skipping Slack notification.")
        return 0

    try:
        _post_to_slack(args.slack_webhook, report)
        print("✅ Sent Slack notification for kernel freshness report.")
    except error.URLError as e:
        print(f"Failed to send Slack notification: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
