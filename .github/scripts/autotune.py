"""Discover autotuning workloads, open Hub config PRs, and report runs."""

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
    config_path(config)
    names = set()
    for target in config["targets"]:
        if target["backend"] != "cuda":
            raise ValueError("Autotuning only supports CUDA")
        if target["name"] in names:
            raise ValueError("Target names must be unique")
        names.add(target["name"])
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


def build_run(event_path: Path):
    """Resolve a successful Build run whose source is on main."""
    event = json.loads(event_path.read_text())
    run = event.get("workflow_run")
    if run is None:
        run_id = int(event["inputs"]["build_run_id"])
        run = json.loads(
            subprocess.check_output(
                [
                    "gh",
                    "api",
                    f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}",
                ]
            )
        )
    if (
        run["path"] != ".github/workflows/build.yaml"
        or run["conclusion"] != "success"
        or run["head_branch"] != "main"
        or run["head_repository"]["full_name"] != os.environ["GITHUB_REPOSITORY"]
    ):
        raise ValueError(
            "Expected a successful Build run from this repository's main branch"
        )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", run["head_sha"], "origin/main"],
        check=True,
    )
    return {"run_id": run["id"], "source_sha": run["head_sha"]}


def plan(event_path: Path, uploads: Path):
    from huggingface_hub import HfApi

    event = json.loads(event_path.read_text())
    selected = (event.get("inputs") or {}).get("kernel", "").strip()
    published = {
        (upload["repo_id"], upload["branch"])
        for path in uploads.glob("hub-upload-cuda-*/*.json")
        if (upload := json.loads(path.read_text()))["status"]
        in ("uploaded", "no_changes")
        and not upload["pull_requests"]
    }
    kernels, gpu_jobs, skipped = [], [], []
    manifests = sorted(Path(".").glob("*/autotune.toml"))
    if selected and selected not in {path.parent.name for path in manifests}:
        raise ValueError(f"Kernel has no autotune.toml: {selected}")
    for path in manifests:
        kernel = path.parent
        if not re.fullmatch(r"[A-Za-z0-9_-]+", kernel.name):
            raise ValueError(f"Invalid kernel directory: {kernel}")
        if selected and selected != kernel.name:
            continue
        info = kernel_info(kernel)
        if (info["repo_id"], info["branch"]) not in published:
            if selected:
                print(
                    f"Skipping {kernel.name}: no release publication in this Build run."
                )
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
                    f"Skipping {kernel.name}/{target['name']}: unsupported GPU/runner "
                    f"pair ({target['gpu']}, {target['runner']})."
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
        print("No published kernels with autotune.toml in this Build run.")
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


def collect(output: Path):
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
        job["name"]: job
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
            step["name"] == "Tune and verify configs"
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
        results.append(dict(target, status=status, url=job["html_url"], reason=reason))
    output.write_text(json.dumps(results, indent=2) + "\n")
    kernels = sorted({r["kernel"] for r in results if r["status"] == "success"})
    with open(os.environ["GITHUB_OUTPUT"], "a") as outputs:
        print(f"kernels={json.dumps(kernels)}", file=outputs)
        print(f"has_success={str(bool(kernels)).lower()}", file=outputs)
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
    if output.exists():
        shutil.rmtree(output)
    subprocess.run(config["command"], env=env, check=True)
    if not any(p.is_file() for p in output.rglob("*")):
        raise ValueError(f"Autotuning produced no configs in {output}")
    shutil.copytree(output, artifacts / target["gpu"])
    (artifacts / "source.json").write_text(
        json.dumps(
            {
                "repo_id": info["repo_id"],
                "revision": revision,
                "variant": variant.name,
                "source_sha": source_sha,
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


def publish(
    kernel: Path,
    artifacts: Path,
    results: Path,
    run_url: str,
    dry_run: bool = False,
):
    """Propose only configs from successful targets on the variants they tuned."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    info = kernel_info(kernel)
    repo_id, branch = info["repo_id"], info["branch"]
    config = manifest(kernel)
    results = [r for r in json.loads(results.read_text()) if r["kernel"] == kernel.name]
    successful = {r["name"] for r in results if r["status"] == "success"}
    targets = [t for t in config["targets"] if t["name"] in successful]
    if not targets:
        return
    if os.environ.get("GITHUB_ACTIONS") == "true":
        os.environ["HF_OIDC_RESOURCE"] = f"kernels/{repo_id}"
    api = HfApi()
    base = api.repo_info(repo_id, repo_type="kernel", revision=branch).sha
    remote_files = set(api.list_repo_files(repo_id, repo_type="kernel", revision=base))
    wanted, prefixes = {}, []
    for target in targets:
        artifact = artifacts / f"autotune-{kernel.name}.{target['name']}"
        source = json.loads((artifact / "source.json").read_text())
        if source["repo_id"] != repo_id or source["revision"] != base:
            raise ValueError("Hub revision changed since tuning; rerun autotuning")
        variant = source["variant"]
        if f"build/{variant}/metadata.json" not in remote_files:
            raise ValueError(f"Tuned variant is no longer published: {variant}")
        directory = artifact / target["gpu"]
        files = sorted(directory.rglob("*.json"))
        if not files:
            raise ValueError(f"Missing configs for {target['name']}")
        prefix = f"build/{variant}/{config_path(config).as_posix()}/{target['gpu']}/"
        prefixes.append(prefix)
        wanted.update((prefix + p.relative_to(directory).as_posix(), p) for p in files)
    remote = {
        name
        for name in remote_files
        if name.endswith(".json")
        and any(name.startswith(prefix) for prefix in prefixes)
    }
    operations = []
    for name, path in sorted(wanted.items()):
        if name in remote:
            previous = api.hf_hub_download(
                repo_id, name, repo_type="kernel", revision=base
            )
            if Path(previous).read_bytes() == path.read_bytes():
                continue
        operations.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=path))
    operations.extend(
        CommitOperationDelete(path_in_repo=name)
        for name in sorted(remote - wanted.keys())
    )
    if not operations:
        print(f"{repo_id}@{branch}: configs unchanged")
        return
    if dry_run:
        print(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "revision": branch,
                    "add": [
                        op.path_in_repo
                        for op in operations
                        if isinstance(op, CommitOperationAdd)
                    ],
                    "delete": [
                        op.path_in_repo
                        for op in operations
                        if isinstance(op, CommitOperationDelete)
                    ],
                },
                indent=2,
            )
        )
        return
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="kernel",
        revision=branch,
        parent_commit=base,
        create_pr=True,
        operations=operations,
        commit_message=f"Refresh autotune configs ({branch})",
        commit_description=(
            "Config-only update. Rebuild integration and metadata/signature refresh are deferred to maintainers before use.\n\n"
            + f"Tuned published revision: `{base}`.\n\n"
            + "Existing configs for failed or skipped targets are preserved.\n\n"
            + result_table(results)
            + f"\n\nRun: {run_url}"
        ),
    )
    api.change_discussion_status(repo_id, commit.pr_num, "open", repo_type="kernel")
    print(f"Opened PR for {kernel.name}: {commit.pr_url}.")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as output:
            print(f"Config PR: [{repo_id}]({commit.pr_url})", file=output)


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

    results = [r for r in json.loads(results.read_text()) if r["url"]]
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
    sub.add_parser("build-run")
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("uploads", type=Path)
    probe_parser = sub.add_parser("probe")
    probe_parser.add_argument("gpu")
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("output", type=Path)
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
        result = build_run(Path(os.environ["GITHUB_EVENT_PATH"]))
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in result.items():
                print(f"{key}={value}", file=output)
    elif args.command == "plan":
        result = plan(Path(os.environ["GITHUB_EVENT_PATH"]), args.uploads)
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
        collect(args.output)
    elif args.command == "tune":
        tune(args.target, args.artifacts, args.revision, args.source_sha)
    elif args.command == "publish":
        publish(args.kernel, args.artifacts, args.results, args.run_url, args.dry_run)
    else:
        upload_report(args.dataset, args.results)
