//! Audit the health of every kernel from the kernel-status dataset.
//!
//! The nightly snapshot (`scripts/kernel_status_snapshot.py`) decides each
//! kernel's flags and health and writes them to `latest.json`. CI writes one
//! event per build run, compacted nightly into `history.jsonl`. This CLI only
//! reads, filters and prints them. Build logs are never fetched: failures print
//! the `gh` command that shows them. `--json` prints JSON Lines for agents, and
//! `ls` and `report` exit 1 when a shown kernel has an error.

use std::collections::{BTreeMap, HashSet};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

const GITHUB_REPO: &str = "huggingface/kernels-community";
const PLATFORMS: [&str; 3] = ["linux", "mac", "windows"];
const SEVERITIES: [&str; 4] = ["error", "warn", "info", "ok"];
const SNAPSHOT_SCHEMA: u32 = 2;
const EVENT_MONTHS: i64 = 6;

// Exit quietly when stdout closes early (e.g. `| head`) instead of panicking.
macro_rules! out {
    ($($arg:tt)*) => {
        if writeln!(std::io::stdout(), $($arg)*).is_err() {
            std::process::exit(0);
        }
    };
}

// CLI

const EXAMPLES: &str = "examples:
  kernel-status                                every kernel, worst first
  kernel-status ls --severity error --platform mac
  kernel-status ls relu                        one kernel with its recent builds
  kernel-status builds --failed --since 7d
  kernel-status report                         summary grouped by flag";

#[derive(Parser)]
#[command(about, after_help = EXAMPLES)]
struct Cli {
    /// Local copy of the dataset instead of the Hub.
    #[arg(long, global = true, env = "KERNEL_STATUS_DATA")]
    data: Option<PathBuf>,
    /// Dataset repo on the Hub.
    #[arg(long, global = true, env = "KERNEL_STATUS_REPO", default_value = "kernels-community/kernel-status")]
    repo: String,
    /// JSON Lines output.
    #[arg(long, global = true)]
    json: bool,
    #[arg(long, global = true)]
    no_color: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Health of every kernel, worst first (default). A single kernel also shows its details and builds.
    Ls(LsArgs),
    /// Query build runs across kernels, newest first.
    Builds(BuildsArgs),
    /// Summary grouped by flag, ready to post to Slack.
    Report,
}

#[derive(Args)]
struct LsArgs {
    kernels: Vec<String>,
    /// Only kernels with this flag (repeatable).
    #[arg(long)]
    flag: Vec<String>,
    /// Only kernels at this severity or worse.
    #[arg(long, value_enum)]
    severity: Option<Severity>,
    /// Only kernels built on this platform.
    #[arg(long, value_parser = PLATFORMS)]
    platform: Option<String>,
    /// Builds shown for a single kernel.
    #[arg(short = 'n', long, default_value_t = 8)]
    limit: usize,
    /// Include PR builds for a single kernel.
    #[arg(long)]
    all: bool,
}

#[derive(Args)]
struct BuildsArgs {
    /// Only these kernels (repeatable).
    #[arg(long)]
    kernel: Vec<String>,
    #[arg(long, value_parser = PLATFORMS)]
    platform: Option<String>,
    #[arg(long, value_parser = ["pr", "stage", "release"])]
    mode: Option<String>,
    /// Only failed builds.
    #[arg(long)]
    failed: bool,
    /// e.g. 24h, 7d or 2026-10-01.
    #[arg(long)]
    since: Option<String>,
    #[arg(short = 'n', long, default_value_t = 30)]
    limit: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum Severity {
    Error,
    Warn,
    Info,
}

// Data

#[derive(Deserialize)]
struct Snapshot {
    generated_at: String,
    latest_builder: Rev,
    torch_versions: Vec<String>,
    flags: Vec<FlagDef>,
    kernels: BTreeMap<String, Kernel>,
}

#[derive(Deserialize)]
struct Rev {
    rev: String,
}

#[derive(Deserialize)]
struct FlagDef {
    name: String,
    severity: String,
    description: String,
}

#[derive(Deserialize)]
struct Kernel {
    version: u64,
    repo_id: String,
    backends: Vec<String>,
    platforms: Vec<String>,
    builder: Builder,
    hub: Hub,
    abi_check: Option<Abi>,
    ci_tests: u64,
    health: String,
    flags: BTreeMap<String, String>,
    release: BTreeMap<String, Release>,
    released: Option<Released>,
}

#[derive(Deserialize)]
struct Builder {
    rev: String,
    date: String,
    behind_days: i64,
}

#[derive(Deserialize)]
struct Hub {
    branch: String,
    #[serde(default)]
    variants: Vec<String>,
    #[serde(default)]
    torch_versions: Vec<String>,
    stable_abi: Option<String>,
    #[serde(default)]
    noarch: bool,
    error: Option<String>,
}

#[derive(Deserialize)]
struct Abi {
    result: String,
}

#[derive(Deserialize, Serialize)]
struct Release {
    result: String,
    finished_at: String,
    run_id: u64,
}

#[derive(Deserialize)]
struct Released {
    finished_at: String,
    builder_rev: String,
}

/// One build run, as kept in history.jsonl.
#[derive(Deserialize, Serialize)]
struct Row {
    #[serde(skip_serializing)]
    path: String,
    kernel: String,
    platform: String,
    mode: String,
    result: String,
    finished_at: String,
    run_id: u64,
    pr: Option<u64>,
    by: Option<String>,
    shipped: bool,
    builder_rev: Option<String>,
    tree_sha: Option<String>,
    failed: Vec<Failed>,
}

#[derive(Deserialize, Serialize)]
struct Failed {
    job: String,
    job_id: Option<u64>,
    backend: Option<String>,
    arch: Option<String>,
    step: Option<String>,
}

/// A build event written by `.github/scripts/emit_build_event.py`.
#[derive(Deserialize)]
struct Event {
    kernel: String,
    workflow: String,
    mode: String,
    run_id: u64,
    finished_at: String,
    trigger: Trigger,
    #[serde(default)]
    source: Source,
    result: EventResult,
}

#[derive(Deserialize)]
struct Trigger {
    pr_number: Option<u64>,
    requested_by: Option<String>,
    #[serde(default)]
    upload: bool,
    #[serde(default)]
    repo_prefix: String,
}

#[derive(Deserialize, Default)]
struct Source {
    builder_rev: Option<String>,
    kernel_tree_sha: Option<String>,
}

#[derive(Deserialize)]
struct EventResult {
    overall: String,
    jobs: Vec<Job>,
}

#[derive(Deserialize)]
struct Job {
    id: Option<u64>,
    name: String,
    backend: Option<String>,
    arch: Option<String>,
    conclusion: Option<String>,
    failed_step: Option<String>,
}

#[derive(Deserialize)]
struct TreeEntry {
    path: String,
}

/// The kernel-status dataset on the Hub, or a local copy with the same layout.
enum Dataset {
    Hub { repo: String, token: Option<String> },
    Local(PathBuf),
}

impl Dataset {
    fn read(&self, path: &str) -> Result<Option<String>, String> {
        match self {
            Dataset::Local(root) => Ok(std::fs::read_to_string(root.join(path)).ok()),
            Dataset::Hub { repo, .. } => self.get(&format!("https://huggingface.co/datasets/{repo}/resolve/main/{path}")),
        }
    }

    fn list(&self, dir: &str) -> Result<Vec<String>, String> {
        let mut paths: Vec<String> = match self {
            Dataset::Local(root) => std::fs::read_dir(root.join(dir))
                .map(|entries| {
                    let names = entries.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned());
                    names.map(|name| format!("{dir}/{name}")).collect()
                })
                .unwrap_or_default(),
            Dataset::Hub { repo, .. } => match self.get(&format!("https://huggingface.co/api/datasets/{repo}/tree/main/{dir}"))? {
                Some(body) => serde_json::from_str::<Vec<TreeEntry>>(&body)
                    .map_err(|e| format!("{dir}: {e}"))?
                    .into_iter()
                    .map(|e| e.path)
                    .collect(),
                None => vec![],
            },
        };
        paths.retain(|p| p.ends_with(".json"));
        paths.sort();
        Ok(paths)
    }

    fn get(&self, url: &str) -> Result<Option<String>, String> {
        let Dataset::Hub { token, .. } = self else { unreachable!() };
        let mut request = ureq::get(url);
        if let Some(token) = token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        match request.call() {
            Ok(response) => response.into_string().map(Some).map_err(|e| format!("{url}: {e}")),
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(format!("{url}: {e}")),
        }
    }

    fn snapshot(&self) -> Result<Snapshot, String> {
        let body = self.read("latest.json")?.ok_or("no latest.json in the dataset, the nightly snapshot has not run yet")?;
        #[derive(Deserialize)]
        struct Versioned {
            schema: u32,
        }
        let version: Versioned = serde_json::from_str(&body).map_err(|e| format!("latest.json: {e}"))?;
        if version.schema != SNAPSHOT_SCHEMA {
            return Err(format!("latest.json has schema {}, expected {SNAPSHOT_SCHEMA}", version.schema));
        }
        serde_json::from_str(&body).map_err(|e| format!("latest.json: {e}"))
    }

    /// history.jsonl plus events written since; only months at or after its newest row are listed.
    fn history(&self, now: DateTime<Utc>) -> Result<Vec<Row>, String> {
        let raw = self.read("history.jsonl")?.unwrap_or_default();
        let mut rows = raw
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).map_err(|e| format!("history.jsonl: {e}")))
            .collect::<Result<Vec<Row>, _>>()?;
        let known: HashSet<&str> = rows.iter().map(|r| r.path.as_str()).collect();
        let start = rows.iter().map(|r| parse(&r.finished_at)).max().unwrap_or(now - Duration::days(31 * EVENT_MONTHS));
        let (mut year, mut month) = (start.year(), start.month());
        let mut new = vec![];
        while (year, month) <= (now.year(), now.month()) {
            new.extend(self.list(&format!("events/{year}/{month:02}"))?.into_iter().filter(|p| !known.contains(p.as_str())));
            (year, month) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
        }
        // Fetch the events written since history.jsonl in parallel.
        let fetched: Vec<Result<Row, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = new
                .chunks(new.len().div_ceil(16).max(1))
                .map(|chunk| s.spawn(move || chunk.iter().map(|p| self.event_row(p)).collect::<Vec<_>>()))
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for row in fetched {
            rows.push(row?);
        }
        rows.sort_by(|a, b| a.finished_at.cmp(&b.finished_at));
        Ok(rows)
    }

    fn event_row(&self, path: &str) -> Result<Row, String> {
        let body = self.read(path)?.ok_or_else(|| format!("{path}: missing"))?;
        let e: Event = serde_json::from_str(&body).map_err(|err| format!("{path}: {err}"))?;
        let shipped = e.mode == "release" && e.trigger.upload && e.trigger.repo_prefix == "kernels-community";
        let failed = e.result.jobs.into_iter().filter(|j| j.conclusion.as_deref() == Some("failure"));
        Ok(Row {
            path: path.to_string(),
            kernel: e.kernel,
            platform: e.workflow,
            mode: e.mode,
            result: e.result.overall,
            finished_at: e.finished_at,
            run_id: e.run_id,
            pr: e.trigger.pr_number,
            by: e.trigger.requested_by,
            shipped,
            builder_rev: e.source.builder_rev,
            tree_sha: e.source.kernel_tree_sha,
            failed: failed
                .map(|j| Failed { job: j.name, job_id: j.id, backend: j.backend, arch: j.arch, step: j.failed_step })
                .collect(),
        })
    }
}

fn hf_token() -> Option<String> {
    if let Some(token) = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()) {
        return Some(token);
    }
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface"));
    std::fs::read_to_string(home.join("token")).ok().map(|t| t.trim().to_string())
}

// Output

struct Style(bool);

impl Style {
    fn paint(&self, text: &str, severity: &str) -> String {
        let code = match severity {
            "error" => "31",
            "warn" => "33",
            "info" => "36",
            "ok" => "32",
            _ => "2",
        };
        if self.0 && !text.is_empty() {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_string()
        }
    }
}

fn rank(severity: &str) -> usize {
    SEVERITIES.iter().position(|s| *s == severity).unwrap_or(SEVERITIES.len())
}

fn flag_severity<'a>(snapshot: &'a Snapshot, flag: &str) -> &'a str {
    snapshot.flags.iter().find(|f| f.name == flag).map_or("info", |f| f.severity.as_str())
}

/// The kernel's flags in the snapshot's order, most severe first.
fn ordered_flags<'a>(snapshot: &'a Snapshot, kernel: &Kernel) -> Vec<&'a str> {
    let mut flags: Vec<&FlagDef> = snapshot.flags.iter().filter(|f| kernel.flags.contains_key(&f.name)).collect();
    flags.sort_by_key(|f| rank(&f.severity));
    flags.into_iter().map(|f| f.name.as_str()).collect()
}

fn result_symbol(result: &str) -> (&'static str, &'static str) {
    match result {
        "success" => ("✓", "ok"),
        "failure" => ("✗", "error"),
        "cancelled" => ("⊘", "warn"),
        _ => ("?", "dim"),
    }
}

fn dig_deeper(row: &Row) -> Vec<String> {
    // `gh run view --log-failed` comes back empty for the self-hosted runners, so read the job log directly.
    let jobs: Vec<String> = row
        .failed
        .iter()
        .filter_map(|f| f.job_id)
        .map(|id| format!("gh api repos/{GITHUB_REPO}/actions/jobs/{id}/logs | grep -B 40 -m1 '##\\[error\\]'"))
        .collect();
    if jobs.is_empty() && row.result == "failure" {
        vec![format!("gh run view {} -R {GITHUB_REPO} --log-failed", row.run_id)]
    } else {
        jobs
    }
}

fn job_label(failed: &Failed) -> String {
    let parts: Vec<&str> = [&failed.backend, &failed.arch].into_iter().flatten().map(String::as_str).collect();
    if parts.is_empty() {
        failed.job.clone()
    } else {
        parts.join("/")
    }
}

fn print_builds(rows: &[&Row], style: &Style, json: bool) {
    if json {
        for row in rows {
            let mut value = serde_json::to_value(row).unwrap();
            value["run_url"] = format!("https://github.com/{GITHUB_REPO}/actions/runs/{}", row.run_id).into();
            value["next"] = dig_deeper(row).into();
            out!("{value}");
        }
        return;
    }
    let width = rows.iter().map(|r| r.kernel.len()).max().unwrap_or(0).max(6);
    for row in rows {
        let (symbol, severity) = result_symbol(&row.result);
        let who: Vec<String> = [row.pr.map(|pr| format!("#{pr}")), row.by.clone()].into_iter().flatten().collect();
        let when = row.finished_at.get(5..16).unwrap_or_default().replace('T', " ");
        let line = format!(
            "{} {when}  {:<width$}  {:<7}  {:<7}  {}",
            style.paint(symbol, severity),
            row.kernel,
            row.platform,
            row.mode,
            style.paint(&who.join(" "), "dim"),
        );
        out!("{}", line.trim_end());
        // Failures are the one pathologically long case: give each detail its own line.
        for f in &row.failed {
            let step = f.step.as_deref().unwrap_or("?");
            out!("{}", style.paint(&format!("    {} failed at `{step}`", job_label(f)), "error"));
        }
        for cmd in dig_deeper(row) {
            out!("{}", style.paint(&format!("    {cmd}"), "dim"));
        }
    }
}

fn print_table(rows: &[(&String, &Kernel)], snapshot: &Snapshot, style: &Style, now: DateTime<Utc>) {
    let width = rows.iter().map(|(name, _)| name.len()).max().unwrap_or(0).max(6);
    let columns = terminal_size::terminal_size().map_or(100, |(w, _)| w.0 as usize);
    out!("  {:<width$}  VER  BUILDER  RELEASED  L M W  FLAGS", "KERNEL");
    for (name, k) in rows {
        let platforms: Vec<String> = PLATFORMS
            .iter()
            .map(|p| match k.release.get(*p) {
                Some(r) => {
                    let (symbol, severity) = result_symbol(&r.result);
                    style.paint(symbol, severity)
                }
                None => style.paint(if k.platforms.iter().any(|x| x == p) { "?" } else { "·" }, "dim"),
            })
            .collect();
        let builder = if k.builder.behind_days == 0 { "latest".to_string() } else { format!("-{}d", k.builder.behind_days) };
        let released = ago(k.released.as_ref().map(|r| r.finished_at.as_str()), now);
        let prefix = format!("{name:<width$}  {:<4} {builder:<8} {released:<9} ", format!("v{}", k.version));
        // Flags that do not fit continue on the next lines, aligned under FLAGS.
        let indent = prefix.chars().count() + 9;
        let lines = wrap(&ordered_flags(snapshot, k), columns.saturating_sub(indent));
        let painted: Vec<String> = lines
            .iter()
            .map(|line| line.iter().map(|f| style.paint(f, flag_severity(snapshot, f))).collect::<Vec<_>>().join(" "))
            .collect();
        out!("{} {prefix}{}  {}", style.paint("●", &k.health), platforms.join(" "), painted.join(&format!("\n{}", " ".repeat(indent))));
    }
}

fn wrap<'a>(words: &[&'a str], room: usize) -> Vec<Vec<&'a str>> {
    let mut lines: Vec<Vec<&str>> = vec![vec![]];
    for word in words {
        let line = lines.last_mut().unwrap();
        let len: usize = line.iter().map(|w| w.len() + 1).sum::<usize>() + word.len();
        if !line.is_empty() && len > room.max(20) {
            lines.push(vec![]);
        }
        lines.last_mut().unwrap().push(word);
    }
    lines
}

fn ago(iso: Option<&str>, now: DateTime<Utc>) -> String {
    let Some(iso) = iso else { return "never".to_string() };
    match (now - parse(iso)).num_days() {
        days if days < 1 => "today".to_string(),
        days if days < 14 => format!("{days}d"),
        days if days < 60 => format!("{}w", days / 7),
        days => format!("{}mo", days / 30),
    }
}

fn parse(iso: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(iso).map(|d| d.with_timezone(&Utc)).unwrap_or_default()
}

fn parse_since(value: &str, now: DateTime<Utc>) -> Result<DateTime<Utc>, String> {
    if let Some(hours) = value.strip_suffix('h').and_then(|n| n.parse().ok()) {
        return Ok(now - Duration::hours(hours));
    }
    if let Some(days) = value.strip_suffix('d').and_then(|n| n.parse().ok()) {
        return Ok(now - Duration::days(days));
    }
    NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .map(|d| Utc.from_utc_datetime(&d.and_hms_opt(0, 0, 0).unwrap()))
        .map_err(|_| format!("invalid --since `{value}`, use e.g. 24h, 7d or 2026-10-01"))
}

// Commands

fn exit_code(failing: bool) -> ExitCode {
    if failing {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn cmd_ls(ds: &Dataset, args: &LsArgs, style: &Style, json: bool) -> Result<ExitCode, String> {
    let snapshot = ds.snapshot()?;
    if let Some(unknown) = args.kernels.iter().find(|k| !snapshot.kernels.contains_key(*k)) {
        return Err(format!("unknown kernel `{unknown}`"));
    }
    let worst = args.severity.map_or(SEVERITIES.len(), |s| s as usize);
    let mut rows: Vec<(&String, &Kernel)> = snapshot
        .kernels
        .iter()
        .filter(|(name, k)| {
            (args.kernels.is_empty() || args.kernels.contains(name))
                && (args.flag.is_empty() || args.flag.iter().any(|f| k.flags.contains_key(f)))
                && args.platform.as_ref().is_none_or(|p| k.platforms.contains(p))
                && rank(&k.health) <= worst
        })
        .collect();
    rows.sort_by_key(|(name, k)| (rank(&k.health), std::cmp::Reverse(k.flags.len()), name.to_string()));
    let failing = rows.iter().any(|(_, k)| k.health == "error");
    let now = Utc::now();

    if json {
        for (name, k) in &rows {
            let value = serde_json::json!({
                "kernel": name, "health": k.health, "flags": k.flags, "version": k.version,
                "platforms": k.platforms, "release": k.release,
                "released_at": k.released.as_ref().map(|r| &r.finished_at),
            });
            out!("{value}");
        }
    } else if let ([(name, k)], [_]) = (rows.as_slice(), args.kernels.as_slice()) {
        show(ds, &snapshot, name, k, args, style, now)?;
    } else {
        let counts: Vec<String> = SEVERITIES
            .iter()
            .map(|s| style.paint(&format!("{} {s}", rows.iter().filter(|(_, k)| k.health == *s).count()), s))
            .collect();
        let age = ago(Some(&snapshot.generated_at), now);
        let age = if age == "today" { age } else { format!("{age} ago") };
        let rev = snapshot.latest_builder.rev.get(..8).unwrap_or_default();
        let context = format!("   snapshot {age} · builder {rev} · torch {}", snapshot.torch_versions.join(", "));
        out!("{}{}\n", counts.join(" · "), style.paint(&context, "dim"));
        print_table(&rows, &snapshot, style, now);
    }
    Ok(exit_code(failing))
}

fn show(ds: &Dataset, snapshot: &Snapshot, name: &str, k: &Kernel, args: &LsArgs, style: &Style, now: DateTime<Utc>) -> Result<(), String> {
    let hub = match &k.hub.error {
        Some(error) => error.clone(),
        None => {
            let torch = if k.hub.torch_versions.is_empty() { "-".to_string() } else { k.hub.torch_versions.join(", ") };
            let mut line = format!("{} variants, torch {torch}", k.hub.variants.len());
            if let Some(stable) = &k.hub.stable_abi {
                line += &format!(", stable ABI {stable}+");
            }
            if k.hub.noarch {
                line += ", noarch";
            }
            line
        }
    };
    let released = match &k.released {
        Some(r) => format!("{} ago, builder {}", ago(Some(&r.finished_at), now), r.builder_rev.get(..8).unwrap_or_default()),
        None => "no record".to_string(),
    };
    let abi = k.abi_check.as_ref().map_or("?", |a| a.result.as_str());
    out!("{} {name}  v{}  {}", style.paint("●", &k.health), k.version, k.repo_id);
    out!("  backends  {} ({})", k.backends.join(", "), k.platforms.join(", "));
    out!(
        "  builder   {} ({}, {}d behind)",
        k.builder.rev.get(..8).unwrap_or_default(),
        k.builder.date.get(..10).unwrap_or_default(),
        k.builder.behind_days
    );
    out!("  released  {released}");
    out!("  hub       {}: {hub}", k.hub.branch);
    out!("  abi       {abi} · ci tests {}", k.ci_tests);
    for flag in ordered_flags(snapshot, k) {
        out!("  {}{}", style.paint(&format!("{flag:<18}"), flag_severity(snapshot, flag)), k.flags[flag]);
    }
    let history = ds.history(now)?;
    let builds: Vec<&Row> = history.iter().rev().filter(|r| r.kernel == name && (args.all || r.mode != "pr")).take(args.limit).collect();
    out!("{}", style.paint(&format!("\nrecent builds{}", if args.all { "" } else { " (--all to include PR builds)" }), "dim"));
    print_builds(&builds, style, false);
    Ok(())
}

fn cmd_builds(ds: &Dataset, args: &BuildsArgs, style: &Style, json: bool) -> Result<ExitCode, String> {
    let now = Utc::now();
    let since = args.since.as_deref().map(|s| parse_since(s, now)).transpose()?;
    let history = ds.history(now)?;
    let rows: Vec<&Row> = history
        .iter()
        .rev()
        .filter(|r| {
            (args.kernel.is_empty() || args.kernel.contains(&r.kernel))
                && args.platform.as_ref().is_none_or(|p| &r.platform == p)
                && args.mode.as_ref().is_none_or(|m| &r.mode == m)
                && (!args.failed || r.result == "failure")
                && since.is_none_or(|t| parse(&r.finished_at) >= t)
        })
        .take(args.limit)
        .collect();
    print_builds(&rows, style, json);
    Ok(ExitCode::SUCCESS)
}

fn cmd_report(ds: &Dataset) -> Result<ExitCode, String> {
    let snapshot = ds.snapshot()?;
    let counts: Vec<String> = SEVERITIES
        .iter()
        .map(|s| format!("{} {s}", snapshot.kernels.values().filter(|k| k.health == *s).count()))
        .collect();
    out!("📊 Kernel health {}: {}", snapshot.generated_at.get(..10).unwrap_or_default(), counts.join(", "));
    let mut flags: Vec<&FlagDef> = snapshot.flags.iter().collect();
    flags.sort_by_key(|f| rank(&f.severity));
    for flag in flags {
        let hits: Vec<String> = snapshot
            .kernels
            .iter()
            .filter(|(_, k)| k.flags.contains_key(&flag.name))
            .map(|(name, _)| format!("`{name}`"))
            .collect();
        if !hits.is_empty() {
            let icon = match flag.severity.as_str() {
                "error" => "❌",
                "warn" => "⚠️",
                _ => "ℹ️",
            };
            out!("\n{icon} *{}* ({}): {}\n{}", flag.name, hits.len(), flag.description, hits.join(", "));
        }
    }
    Ok(exit_code(snapshot.kernels.values().any(|k| k.health == "error")))
}

fn main() -> ExitCode {
    // `ls` is the default subcommand, so options given without one belong to it.
    let mut argv: Vec<String> = std::env::args().collect();
    if !argv.iter().skip(1).any(|a| ["ls", "builds", "report", "help", "-h", "--help", "-V", "--version"].contains(&a.as_str())) {
        argv.insert(1, "ls".to_string());
    }
    let cli = Cli::parse_from(argv);
    let style = Style(!cli.no_color && !cli.json && std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal());
    let ds = match &cli.data {
        Some(path) => Dataset::Local(path.clone()),
        None => Dataset::Hub { repo: cli.repo.clone(), token: hf_token() },
    };
    let result = match &cli.command {
        Command::Ls(args) => cmd_ls(&ds, args, &style, cli.json),
        Command::Builds(args) => cmd_builds(&ds, args, &style, cli.json),
        Command::Report => cmd_report(&ds),
    };
    result.unwrap_or_else(|e| {
        eprintln!("kernel-status: {e}");
        ExitCode::from(2)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> DateTime<Utc> {
        parse("2026-10-27T00:00:00Z")
    }

    #[test]
    fn failed_builds_point_at_the_job_log() {
        let dir = std::env::temp_dir().join(format!("kernel-status-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("events/2026/10")).unwrap();
        let event = serde_json::json!({
            "kernel": "relu", "workflow": "linux", "mode": "release", "run_id": 99,
            "finished_at": "2026-10-20T00:00:00Z",
            "trigger": {"pr_number": null, "requested_by": null, "upload": true, "repo_prefix": "kernels-community"},
            "source": {"builder_rev": "b2", "kernel_tree_sha": "t2"},
            "result": {"overall": "failure", "jobs": [
                {"id": 123, "name": "build-kernel (cuda, x86_64-linux)", "backend": "cuda", "arch": "x86_64",
                 "conclusion": "failure", "failed_step": "Build kernel"},
                {"id": 124, "name": "build-kernel (rocm, x86_64-linux)", "backend": "rocm", "arch": "x86_64",
                 "conclusion": "success", "failed_step": null}]}
        });
        std::fs::write(dir.join("events/2026/10/99-1-linux.json"), event.to_string()).unwrap();

        let rows = Dataset::Local(dir.clone()).history(now()).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].shipped);
        assert_eq!(job_label(&rows[0].failed[0]), "cuda/x86_64");
        assert_eq!(
            dig_deeper(&rows[0]),
            ["gh api repos/huggingface/kernels-community/actions/jobs/123/logs | grep -B 40 -m1 '##\\[error\\]'"]
        );
        let row = Row { failed: vec![], ..rows.into_iter().next().unwrap() };
        assert_eq!(dig_deeper(&row), ["gh run view 99 -R huggingface/kernels-community --log-failed"]);
    }

    #[test]
    fn flags_wrap_to_the_room_left() {
        assert_eq!(wrap(&["release-failed", "needs-rebuild", "no-ci-tests"], 30), [vec!["release-failed", "needs-rebuild"], vec!["no-ci-tests"]]);
        assert_eq!(wrap(&[], 30), [Vec::<&str>::new()]);
    }

    #[test]
    fn ages_and_since() {
        assert_eq!(ago(Some("2026-10-26T12:00:00Z"), now()), "today");
        assert_eq!(ago(Some("2026-10-01T00:00:00+00:00"), now()), "3w");
        assert_eq!(ago(None, now()), "never");
        assert_eq!(parse_since("2d", now()).unwrap(), parse("2026-10-25T00:00:00Z"));
        assert_eq!(parse_since("2026-10-01", now()).unwrap(), parse("2026-10-01T00:00:00Z"));
        assert!(parse_since("soon", now()).is_err());
    }
}
