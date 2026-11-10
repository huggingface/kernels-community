"""Tune published kernels and propose source config updates for maintainer review."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

# Runner groups provisioned for autotuning. Unknown GPU/group pairs are skipped.
GPU_RUNNERS = {"NVIDIA L4": "aws-g6-12xlarge-plus"}


def manifest(kernel: Path):
    config = tomllib.loads((kernel / "autotune.toml").read_text())
    output = Path(config["configs"])
    if output.is_absolute() or ".." in output.parts or output == Path("."):
        raise ValueError(f"Invalid config directory: {output}")
    if not (kernel / output).resolve().is_relative_to(kernel.resolve()):
        raise ValueError("Config directory escapes the kernel source tree")
    config_path(config)
    names = set()
    gpus = set()
    for target in config["targets"]:
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", target["name"])
            or not re.fullmatch(r"[A-Za-z0-9 _.-]+", target["gpu"])
            or target["gpu"] in {".", ".."}
        ):
            raise ValueError("Invalid target name or GPU directory")
        if target["backend"] != "cuda":
            raise ValueError("Autotuning only supports CUDA")
        if target["name"] in names or target["gpu"] in gpus:
            raise ValueError("Target names and GPUs must be unique")
        names.add(target["name"])
        gpus.add(target["gpu"])
    if not names or not config["command"]:
        raise ValueError("An autotune manifest needs a command and at least one target")
    return config


def config_path(config):
    """Strip torch-ext/<package>/ to get the path inside a built variant."""
    parts = Path(config["configs"]).parts
    source = 1 if parts[:1] == ("src",) else 0
    if len(parts) < source + 3 or parts[source] != "torch-ext":
        raise ValueError("configs must be inside torch-ext/<package>/")
    return Path(*parts[source + 2 :])


AUTOTUNE_MARKER = "<!-- kernels-community-autotune -->"


def github_api(path):
    return json.loads(subprocess.check_output(["gh", "api", path]))


def is_autotune_pr(pr, repository):
    """Recognize our own config PRs, including after squash merging."""
    return (
        pr["user"]["login"] == "github-actions[bot]"
        and pr["head"]["repo"] is not None
        and pr["head"]["repo"]["full_name"] == repository
        and pr["head"]["ref"].startswith("autotune/")
        and AUTOTUNE_MARKER in (pr.get("body") or "")
    )


def build_run(request, uploads):
    """Validate the complete successful dispatch set and download publications."""
    repository = os.environ["GITHUB_REPOSITORY"]
    base = f"repos/{repository}"
    pr = github_api(f"{base}/pulls/{int(request['pr_number'])}")
    if not pr["merged"] or pr["base"]["ref"] != "main":
        raise ValueError("Expected a merged PR targeting main")
    runs = []
    for build in request["builds"]:
        run = github_api(f"{base}/actions/runs/{int(build['run_id'])}")
        if (
            build["workflow"]
            not in {"build.yaml", "build-mac.yaml", "build-windows.yaml"}
            or run["path"] != f".github/workflows/{build['workflow']}"
            or run["event"] != "workflow_dispatch"
            or run["status"] != "completed"
            or run["conclusion"] != "success"
            or run["head_branch"] != "main"
            or run["head_repository"]["full_name"] != repository
            or not run["display_title"].endswith(f"request={build['dispatch_key']}")
            or " / mode=release / " not in run["display_title"]
        ):
            raise ValueError("Expected a successful merge-and-upload build from main")
        runs.append(run)
    sources = {run["head_sha"] for run in runs}
    if len(sources) != 1:
        raise ValueError(
            "Builds must use the same source commit; rerun merge-and-upload"
        )
    source_sha = sources.pop()
    for ancestor, descendant in [
        (pr["merge_commit_sha"], source_sha),
        (source_sha, "origin/main"),
    ]:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant], check=True
        )
    if source_sha == pr["merge_commit_sha"] and is_autotune_pr(pr, repository):
        # Do not skip if a maintainer added code changes to the generated PR.
        pages = json.loads(
            subprocess.check_output(
                [
                    "gh",
                    "api",
                    "--paginate",
                    "--slurp",
                    f"{base}/pulls/{pr['number']}/files?per_page=100",
                ]
            )
        )
        files = [file["filename"] for page in pages for file in page]
        prefixes = [
            f"{path.parent}/{manifest(path.parent)['configs']}/"
            for path in Path(".").glob("*/autotune.toml")
        ]
        if files and all(
            name.endswith(".json") and any(name.startswith(p) for p in prefixes)
            for name in files
        ):
            print("Autotuner config update was rebuilt; no further tuning needed.")
            return {"source_sha": source_sha, "skip": "true"}
    for run in runs:
        if run["path"] == ".github/workflows/build.yaml":
            subprocess.run(
                [
                    "gh",
                    "run",
                    "download",
                    str(run["id"]),
                    "--repo",
                    repository,
                    "--pattern",
                    "hub-upload-*",
                    "--dir",
                    str(uploads / str(run["id"])),
                ],
                check=True,
            )
    return {"source_sha": source_sha, "skip": "false"}


def plan(uploads: Path):
    from huggingface_hub import HfApi

    published = {
        (upload["repo_id"], upload["branch"])
        for path in uploads.glob("*/hub-upload-cuda-*/*.json")
        if (upload := json.loads(path.read_text()))["status"]
        in ("uploaded", "no_changes")
        and not upload["pull_requests"]
    }
    kernels, gpu_jobs, skipped = [], [], []
    for path in sorted(Path(".").glob("*/autotune.toml")):
        kernel = path.parent
        if not re.fullmatch(r"[A-Za-z0-9_-]+", kernel.name):
            raise ValueError(f"Invalid kernel directory: {kernel}")
        info = kernel_info(kernel)
        if (info["repo_id"], info["branch"]) not in published:
            continue
        config = manifest(kernel)
        revision = (
            HfApi()
            .repo_info(info["repo_id"], repo_type="kernel", revision=info["branch"])
            .sha
        )
        kernels.append(kernel.name)
        for target in config["targets"]:
            job = {"kernel": kernel.name, **target, "revision": revision}
            if GPU_RUNNERS.get(target["gpu"]) != target["runner"]:
                print(
                    f"Skipping {kernel.name}/{target['name']}: unsupported GPU/runner pair."
                )
                skipped.append(
                    dict(
                        job,
                        status="skipped",
                        url=None,
                        reason="Unsupported GPU/runner pair",
                    )
                )
            else:
                gpu_jobs.append(job)
        print(kernel.name)
    if not kernels:
        print("No published kernels with autotune.toml in these builds.")
    return {"kernels": kernels, "gpu_jobs": gpu_jobs, "skipped": skipped}


def probe(gpu: str):
    """Check the assigned machine before installing the tuning environment."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        supported = result.returncode == 0 and result.stdout.strip() == gpu
    except (FileNotFoundError, subprocess.TimeoutExpired):
        supported = False
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        print(f"supported={str(supported).lower()}", file=output)
    if not supported:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
            print(f"Skipped autotuning: {gpu} is not available on GPU 0.", file=summary)


def collect(output: Path, artifacts: Path):
    """Join planned targets to their actual job URLs and conclusions."""
    pages = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}/jobs?filter=all&per_page=100",
            ]
        )
    )
    jobs = {
        job["name"].split(" / ")[-1]: job
        for job in sorted(
            (job for page in pages for job in page["jobs"]), key=lambda job: job["id"]
        )
    }
    results = json.loads(os.environ["SKIPPED_TARGETS"])
    for target in json.loads(os.environ["GPU_JOBS"]):
        job = jobs.get(f"Autotune ({target['kernel']}/{target['name']})")
        if job is None:
            raise ValueError(
                f"Missing GitHub job for {target['kernel']}/{target['name']}"
            )
        status = job["conclusion"] or "unknown"
        reason = ""
        if status == "success" and any(
            step["name"] == "Tune configs"
            and step["conclusion"] == "skipped"
            for step in job["steps"]
        ):
            status, reason = "skipped", "Requested GPU unavailable on the runner"
        elif status == "success" and any(
            step["name"] == "Upload configs for collection"
            and step["conclusion"] == "skipped"
            for step in job["steps"]
        ):
            status, reason = (
                "skipped",
                "No compatible published variant with matching source provenance",
            )
        if status == "success":
            artifact = artifacts / f"autotune-{target['kernel']}.{target['name']}"
            report = json.loads((artifact / "source.json").read_text())
            status = report["status"]
            if status not in {"changed", "no_changes"}:
                raise ValueError(f"Invalid autotuner outcome: {status}")
        results.append(dict(target, status=status, url=job["html_url"], reason=reason))
    output.write_text(json.dumps(results, indent=2) + "\n")
    kernels = sorted({r["kernel"] for r in results if r["status"] == "changed"})
    with open(os.environ["GITHUB_OUTPUT"], "a") as outputs:
        print(f"kernels={json.dumps(kernels)}", file=outputs)
        print(f"has_changes={str(bool(kernels)).lower()}", file=outputs)
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        print(result_table(results), file=summary)


def result_table(results):
    lines = [
        "| Kernel | Target | GPU | Result | Job | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        job = f"[logs]({result['url']})" if result["url"] else "—"
        lines.append(
            f"| {result['kernel']} | {result['name']} | {result['gpu']} | "
            f"{result['status']} | {job} | {result['reason']} |"
        )
    return "\n".join(lines)


def tune(target_name: str, artifacts: Path, revision: str, source_sha: str):
    from huggingface_hub import HfApi
    from kernels import VariantAccepted, get_kernel, get_kernel_variants

    config = manifest(Path("."))
    target = next(t for t in config["targets"] if t["name"] == target_name)
    info = kernel_info(Path("."))
    variants = get_kernel_variants(info["repo_id"], revision=revision, backend="cuda")
    preferred = next((v for v in variants if isinstance(v, VariantAccepted)), None)
    reason = "No compatible CUDA variant"
    if preferred is not None:
        metadata = HfApi().hf_hub_download(
            info["repo_id"],
            f"build/{preferred.variant.variant_str}/metadata.json",
            repo_type="kernel",
            revision=revision,
        )
        provenance = (
            json.loads(Path(metadata).read_text()).get("provenance") or {}
        ).get("kernel") or {}
        commit = provenance.get("commit", provenance.get("sha"))
        reason = "Published variant does not have clean provenance matching the Build source commit"
        if commit == source_sha and provenance.get("dirty") is False:
            reason = ""
    if reason:
        message = f"Skipping {Path.cwd().name}/{target_name}: {reason}."
        print(message)
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as output:
                print(message, file=output)
        return
    module = get_kernel(info["repo_id"], revision=revision, backend="cuda")
    variant = Path(module.__file__).parent
    if (
        variant.parent.name != "build"
        or variant.parent.parent.name != revision
        or variant.name != preferred.variant.variant_str
    ):
        raise ValueError(
            "Autotuning for Hub publication must use the pinned Hub revision"
        )
    print(f"Tuning {info['repo_id']}@{revision} ({variant.name})")
    env = dict(
        os.environ,
        AUTOTUNE_GPU=target["gpu"],
        AUTOTUNE_CONFIGS=config["configs"],
        LOCAL_KERNELS=f"{info['repo_id']}={variant}",
    )
    output = Path(config["configs"]) / target["gpu"]
    previous = config_files(output)
    if output.exists():
        shutil.rmtree(output)
    subprocess.run(config["command"], env=env, check=True)
    current = config_files(output)
    if not current:
        raise ValueError(f"Autotuning produced no configs in {output}")
    shutil.copytree(output, artifacts / target["gpu"])
    (artifacts / "source.json").write_text(
        json.dumps(
            {
                "repo_id": info["repo_id"],
                "revision": revision,
                "variant": variant.name,
                "source_sha": source_sha,
                "status": "no_changes" if previous == current else "changed",
            }
        )
        + "\n"
    )
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as outputs:
            print("tuned=true", file=outputs)


def kernel_info(kernel: Path) -> dict[str, str]:
    build = kernel / "src/build.toml"
    if not build.exists():
        build = kernel / "build.toml"
    general = tomllib.loads(build.read_text())["general"]
    return {
        "repo_id": general["hub"]["repo-id"],
        "branch": general["hub"].get("branch", f"v{general['version']}"),
    }


def config_files(directory):
    """Compare JSON values, ignoring formatting, and reject non-config artifacts."""
    files = {}
    if directory.is_symlink():
        raise ValueError(f"Symlink config directory: {directory}")
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink config: {path}")
        if path.is_file():
            if path.suffix != ".json":
                raise ValueError(f"Non-JSON config: {path}")
            files[path.relative_to(directory).as_posix()] = json.loads(path.read_text())
    return files


def publish(
    kernel: Path, artifacts: Path, results: Path, run_url: str, dry_run: bool = False
):
    """Open a GitHub source PR; the normal build packages and signs its configs."""
    repository = os.environ["GITHUB_REPOSITORY"]
    info = kernel_info(kernel)
    config = manifest(kernel)
    results = [r for r in json.loads(results.read_text()) if r["kernel"] == kernel.name]
    changed = {r["name"] for r in results if r["status"] == "changed"}
    targets = [t for t in config["targets"] if t["name"] in changed]
    if not targets:
        print(f"{kernel}: configs unchanged")
        return
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    directories = []
    for target in targets:
        artifact = artifacts / f"autotune-{kernel.name}.{target['name']}"
        source = json.loads((artifact / "source.json").read_text())
        if (
            source["repo_id"] != info["repo_id"]
            or source["source_sha"] != source_sha
            or source["status"] != "changed"
        ):
            raise ValueError("Config artifact does not match the tuned source")
        directory = artifact / target["gpu"]
        if not config_files(directory):
            raise ValueError(f"Missing configs for {target['name']}")
        directories.append((directory, kernel / config["configs"] / target["gpu"]))
    subprocess.run(["git", "fetch", "origin", "main"], check=True)
    # Never propose stale settings over newer kernel code or newer tuning results.
    subprocess.run(
        ["git", "diff", "--exit-code", source_sha, "origin/main", "--", str(kernel)],
        check=True,
    )
    branch = f"autotune/{kernel.name}/{source_sha}"
    existing = json.loads(
        subprocess.check_output(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repository,
                "--head",
                branch,
                "--state",
                "all",
                "--json",
                "url",
            ]
        )
    )
    if existing:
        print(f"Config PR already exists: {existing[0]['url']}")
        return
    subprocess.run(["git", "switch", "-c", branch, "origin/main"], check=True)
    for directory, destination in directories:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(directory, destination)
        subprocess.run(["git", "add", "--", str(destination)], check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
    if diff.returncode == 0:
        print(f"{kernel}: configs unchanged")
        return
    if diff.returncode != 1:
        diff.check_returncode()
    if dry_run:
        subprocess.run(["git", "diff", "--cached", "--stat"], check=True)
        return
    title = f"{kernel.name}: refresh autotuning configs"
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "commit",
            "-m",
            title,
        ],
        check=True,
    )
    subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        body = Path(tmp) / "body.md"
        body.write_text(
            f"{AUTOTUNE_MARKER}\n\n"
            f"Update the source configs tuned from `{source_sha}`. The normal kernel build packages these JSON files.\n\n"
            "Failed and skipped GPU targets retain their existing configs.\n\n"
            + result_table(results)
            + f"\n\nRun: {run_url}\n\n"
            f"After review, run `/kernel-bot merge-and-upload {kernel.name}` to rebuild and upload. "
            "The autotuner recognizes this config PR and skips another tuning pass.\n"
        )
        url = subprocess.check_output(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                repository,
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                title,
                "--body-file",
                str(body),
            ],
            text=True,
        ).strip()
    print(f"Opened config PR: {url}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as output:
            print(f"Config PR: [{kernel.name}]({url})", file=output)


def report(path: Path, repo_id: str, run_url: str):
    rows = {}
    if path.exists():
        with path.open(newline="") as f:
            rows = {
                row["kernel_repo_id"]: row["autotune_runs"] for row in csv.DictReader(f)
            }
    runs = rows.get(repo_id, "").split(",") if rows.get(repo_id) else []
    if run_url not in runs:
        runs.append(run_url)
    rows[repo_id] = ",".join(runs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["kernel_repo_id", "autotune_runs"])
        writer.writerows(sorted(rows.items()))


def upload_report(dataset: str, results: Path):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

  results = [
      r for r in json.loads(results.read_text())
      if r["status"] in {"changed", "no_changes"} and r["url"]
  ]
    if not results:
        return
    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "autotune_runs.csv"
        try:
            shutil.copyfile(
                hf_hub_download(dataset, path.name, repo_type="dataset"), path
            )
        except EntryNotFoundError:
            pass
        for result in results:
            info = kernel_info(Path(result["kernel"]))
            report(path, info["repo_id"], result["url"])
        api.upload_file(
            repo_id=dataset,
            repo_type="dataset",
            path_or_fileobj=path,
            path_in_repo=path.name,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build-run")
    build_parser.add_argument("uploads", type=Path)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("uploads", type=Path)
    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("gpu")
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("output", type=Path)
    collect_parser.add_argument("artifacts", type=Path)
    tune_parser = sub.add_parser("tune")
    tune_parser.add_argument("target")
    tune_parser.add_argument("artifacts", type=Path)
    tune_parser.add_argument("--revision", required=True)
    tune_parser.add_argument("--source-sha", required=True)
    publish_parser = sub.add_parser("publish")
    publish_parser.add_argument("kernel", type=Path)
    publish_parser.add_argument("artifacts", type=Path)
    publish_parser.add_argument("results", type=Path)
    publish_parser.add_argument("run_url")
    publish_parser.add_argument("--dry-run", action="store_true")
    report_parser = sub.add_parser("report")
    report_parser.add_argument("dataset")
    report_parser.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.command == "build-run":
        result = build_run(json.loads(os.environ["AUTOTUNE_REQUEST"]), args.uploads)
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in result.items():
                print(f"{key}={value}", file=output)
    elif args.command == "plan":
        result = plan(args.uploads)
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in result.items():
                print(f"{key}={json.dumps(value)}", file=output)
            print(
                f"has_work={str(bool(result['gpu_jobs'])).lower()}",
                file=output,
            )
            print(f"has_kernels={str(bool(result['kernels'])).lower()}", file=output)
    elif args.command == "probe":
        probe(args.gpu)
    elif args.command == "collect":
        collect(args.output, args.artifacts)
    elif args.command == "tune":
        tune(args.target, args.artifacts, args.revision, args.source_sha)
    elif args.command == "publish":
        publish(args.kernel, args.artifacts, args.results, args.run_url, args.dry_run)
    else:
        upload_report(args.dataset, args.results)
