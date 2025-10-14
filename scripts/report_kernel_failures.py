import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from urllib import error, request

import run_kernels_checks


def parse_args() -> argparse.Namespace:
    parser = run_kernels_checks.build_parser()
    parser.description = "Run nightly kernel checks and post failing kernels to Slack."
    parser.add_argument(
        "--slack-webhook",
        default=os.getenv("SLACK_WEBHOOK_URL"),
        help="Slack incoming webhook URL. If not set and checks fail, no notification is sent.",
    )
    return parser.parse_args()


def _github_run_url() -> Optional[str]:
    repository = os.getenv("GITHUB_REPOSITORY")
    run_id = os.getenv("GITHUB_RUN_ID")
    if not repository or not run_id:
        return None
    return run_id


def _format_failure_message(failures: list[str]) -> str:
    failures_list = sorted(failures)
    kernels_failed = "\n".join(f"• `{run_kernels_checks.ORG}/{name}`" for name in failures_list)
    heading = f"❌ Nightly kernels check failed for {len(failures_list)} kernel(s)."
    run_url = _github_run_url()
    if run_url:
        return f"{heading}\nRun: {run_url}\n{kernels_failed}"
    return f"{heading}\n{kernels_failed}"


def _post_to_slack(webhook_url: str, message: str) -> None:
    payload = {"text": message}

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as response:
        response.read()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    root_path = Path(args.root).resolve()
    logging.debug(f"Using root path {root_path}")

    try:
        directories = run_kernels_checks.discover_kernel_dirs(root_path, args.exclude)
    except RuntimeError as err:
        logging.error("%s", err)
        return 1

    if not directories:
        logging.error("⛔️ No kernel directories found in %s.", root_path)
        return 1

    logging.info(f"🧪 Checking {len(directories)} kernel directories: {directories=}.")

    failures = run_kernels_checks.run_kernels_checks(directories, args.dry_run, args.clear_cache)
    if not failures:
        logging.info("✅ All kernels checks completed successfully.")
        return 0

    logging.error(
        "❌ kernels check failed for %d directories: %s",
        len(failures),
        ", ".join(sorted(failures)),
    )

    if args.dry_run:
        logging.info("Dry-run mode; skipping Slack notification.")
        return 1

    if not args.slack_webhook:
        logging.warning("Slack webhook URL is not provided; skipping Slack notification.")
        return 1

    message = _format_failure_message(failures)
    try:
        _post_to_slack(args.slack_webhook, message)
    except error.URLError as err:
        logging.error("Failed to send Slack notification: %s", err)
        return 1

    logging.info("Sent Slack notification for failing kernels.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
