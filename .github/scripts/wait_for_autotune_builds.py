"""Wait for the entire merge-and-upload dispatch set before starting autotuning."""

import json
import os
import subprocess
import time


def github_api(path):
    return json.loads(subprocess.check_output(["gh", "api", path]))


def wait_for_builds(request, *, timeout=21000, interval=30):
    repository = os.environ["GITHUB_REPOSITORY"]
    base = f"repos/{repository}/actions"
    builds = request["builds"]
    if not builds:
        raise ValueError("No builds were dispatched")
    deadline = time.monotonic() + timeout
    missing_deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        complete = True
        for build in builds:
            if "run_id" not in build:
                pages = json.loads(
                    subprocess.check_output(
                        [
                            "gh",
                            "api",
                            "--paginate",
                            "--slurp",
                            f"{base}/workflows/{build['workflow']}/runs?event=workflow_dispatch&branch=main&per_page=100",
                        ]
                    )
                )
                matches = [
                    run
                    for page in pages
                    for run in page["workflow_runs"]
                    if run["display_title"].endswith(f"request={build['dispatch_key']}")
                ]
                if len(matches) > 1:
                    raise ValueError("Multiple runs match one build dispatch")
                if not matches:
                    if time.monotonic() >= missing_deadline:
                        raise TimeoutError(f"Build did not appear: {build}")
                    complete = False
                    continue
                build["run_id"] = matches[0]["id"]
            run = github_api(f"{base}/runs/{build['run_id']}")
            if run["status"] != "completed":
                complete = False
            elif run["conclusion"] != "success":
                raise ValueError(
                    f"Build {run['id']} ended with {run['conclusion']}; no autotuning"
                )
        if complete:
            return request
        print("Waiting for all merge-and-upload builds to succeed...", flush=True)
        time.sleep(interval)
    raise TimeoutError("Timed out waiting for merge-and-upload builds")


if __name__ == "__main__":
    request = wait_for_builds(json.loads(os.environ["AUTOTUNE_REQUEST"]))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        print(f"request={json.dumps(request)}", file=output)
