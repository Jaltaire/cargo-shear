//! # cargo-shear
//!
//! A tool for detecting and removing unused dependencies from Rust projects.
//!
//! ## Overview
//!
//! `cargo-shear` analyzes your Rust codebase to identify dependencies that are declared
//! in `Cargo.toml` but never actually used in the code. It can automatically remove
//! these unused dependencies with the `--fix` flag.
//!
//! ## Architecture
//!
//! The codebase is organized into several focused modules:
//!
//! - `cargo_toml_editor` - Handles modifications to Cargo.toml files
//! - `package_analyzer` - Analyzes packages to find issues
//! - `package_processor` - Processes packages and detects unused dependencies
//! - `source_parser` - Parses Rust source to extract data
//!
//! ## Usage
//!
//! ```no_run
//! use cargo_shear::{CargoShear, CargoShearOptions};
//!
//! let options = CargoShearOptions::new(std::path::PathBuf::from("."));
//! let exit_code = CargoShear::new(std::io::stdout(), options).run();
//! ```

mod cache;
#[cfg(test)]
mod cache_tests;
mod cargo_toml_editor;
mod context;
mod diagnostics;
mod manifest;
mod output;
mod package_analyzer;
mod package_processor;
mod parse_cache;
mod resolve_cache;
mod source_parser;
#[cfg(test)]
mod tests;
pub mod util;

use std::{
    env, fs,
    io::{self, IsTerminal, Write},
    panic,
    path::{Path, PathBuf},
    process::ExitCode,
    str::FromStr,
    thread,
    time::SystemTime,
};

use anyhow::Result;
use bpaf::Bpaf;
use cargo_metadata::{CargoOpt, Metadata, MetadataCommand, Package};
use owo_colors::OwoColorize;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashSet;
use toml_edit::DocumentMut;

pub use crate::output::{ColorMode, OutputFormat};
use crate::{
    cache::{CacheDir, Digest, executable_digest},
    cargo_toml_editor::CargoTomlEditor,
    context::{PackageContext, ParseOutcome, WorkspaceContext},
    diagnostics::ShearAnalysis,
    output::Renderer,
    package_processor::{PackageAnalysis, PackageProcessor, WorkspaceAnalysis},
    parse_cache::ParseCache,
    resolve_cache::{KeyContext, ResolveKey, ResolvedDeps, Watched},
    util::read_to_string,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Command-line options for cargo-shear.
///
/// Parsed from `argv` by `bpaf`. The doc comments on individual fields below
/// become the `--help` text Cargo's users see, so keep them user-facing.
///
/// The "batteries" `cargo_helper` strips the leading `shear` argument when
/// invoked as `cargo shear`, so this struct sees the same shape either way.
/// See <https://docs.rs/bpaf/latest/bpaf/batteries/fn.cargo_helper.html>.
#[derive(Debug, Clone, Bpaf)]
#[bpaf(options("shear"), version(VERSION))]
pub struct CargoShearOptions {
    /// Remove unused dependencies.
    ///
    /// When set, cargo-shear will automatically remove detected unused
    /// dependencies from Cargo.toml files.
    #[bpaf(long)]
    fix: bool,

    /// Uses `cargo expand` to expand macros, which requires nightly and is significantly slower.
    ///
    /// This option provides more accurate detection by expanding proc macros
    /// and attribute macros, but requires a nightly Rust toolchain.
    #[bpaf(long)]
    expand: bool,

    /// Detect mismatches between `test`/`doctest` target settings and source content.
    ///
    /// When set, cargo-shear warns when `[lib] test = false` is paired with source
    /// that contains tests (or `doctest = false` with source that contains doc tests),
    /// and — within a workspace — when `test`/`doctest` are left at their default of
    /// `true` for lib targets that contain none.
    #[bpaf(long("check-test-targets"))]
    check_test_targets: bool,

    /// Treat warnings as errors.
    ///
    /// When set, warnings will cause cargo-shear to exit with a failure code.
    #[bpaf(long("deny-warnings"))]
    deny_warnings: bool,

    /// Reuse dependency resolution and parsed sources from earlier runs.
    ///
    /// Results are kept under the target directory and reused only while every
    /// input they were computed from is unchanged, so the report and any fixes
    /// are identical to a run without this flag.
    #[bpaf(long)]
    cache: bool,

    /// Assert that `Cargo.lock` will remain unchanged.
    locked: bool,

    /// Run without accessing the network
    offline: bool,

    /// Equivalent to specifying both --locked and --offline
    frozen: bool,

    /// Package(s) to check.
    ///
    /// If not specified, all packages in the workspace are checked.
    /// Can be specified multiple times to check specific packages.
    #[bpaf(long, short, argument("SPEC"))]
    package: Vec<String>,

    /// Exclude packages from the check.
    ///
    /// Can be specified multiple times to exclude multiple packages.
    exclude: Vec<String>,

    /// Output format: auto, json, github
    #[bpaf(long, fallback(OutputFormat::Auto))]
    format: OutputFormat,

    /// Color usage for output: auto, always, never
    #[bpaf(long, fallback(ColorMode::Auto))]
    color: ColorMode,

    /// Path to the project directory.
    ///
    /// Defaults to the current directory if not specified.
    #[bpaf(positional("PATH"), fallback_with(default_path))]
    path: PathBuf,
}

impl CargoShearOptions {
    /// Construct an options value with every flag at its default and the project
    /// rooted at `path`. Programmatic callers usually layer `with_*` builders on top.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            fix: false,
            expand: false,
            check_test_targets: false,
            deny_warnings: false,
            cache: false,
            locked: false,
            offline: false,
            frozen: false,
            package: vec![],
            exclude: vec![],
            format: OutputFormat::default(),
            color: ColorMode::default(),
        }
    }

    /// Enable `--fix`: rewrite manifests to apply fixable diagnostics.
    #[must_use]
    pub const fn with_fix(mut self) -> Self {
        self.fix = true;
        self
    }

    /// Enable `--expand`: run `cargo expand` first to surface macro-generated imports.
    #[must_use]
    pub const fn with_expand(mut self) -> Self {
        self.expand = true;
        self
    }

    /// Enable `--check-test-targets`: emit `test`/`doctest` flag mismatch diagnostics.
    #[must_use]
    pub const fn with_check_test_targets(mut self) -> Self {
        self.check_test_targets = true;
        self
    }

    /// Enable `--deny-warnings`: turn any warning into a non-zero exit.
    #[must_use]
    pub const fn with_deny_warnings(mut self) -> Self {
        self.deny_warnings = true;
        self
    }

    /// Enable `--cache`: reuse resolution and parsing from earlier runs.
    #[must_use]
    pub const fn with_cache(mut self) -> Self {
        self.cache = true;
        self
    }

    /// Enable `--locked`: forwarded to `cargo metadata`.
    #[must_use]
    pub const fn with_locked(mut self) -> Self {
        self.locked = true;
        self
    }

    /// Enable `--offline`: forwarded to `cargo metadata`.
    #[must_use]
    pub const fn with_offline(mut self) -> Self {
        self.offline = true;
        self
    }

    /// Enable `--frozen`: forwarded to `cargo metadata` (implies `--locked` and `--offline`).
    #[must_use]
    pub const fn with_frozen(mut self) -> Self {
        self.frozen = true;
        self
    }

    /// Restrict the run to these workspace members. Empty means "all".
    #[must_use]
    pub fn with_packages(mut self, packages: Vec<String>) -> Self {
        self.package = packages;
        self
    }

    /// Exclude these workspace members from the run.
    #[must_use]
    pub fn with_excludes(mut self, excludes: Vec<String>) -> Self {
        self.exclude = excludes;
        self
    }

    /// Pick the renderer used for diagnostics.
    #[must_use]
    pub const fn with_format(mut self, format: OutputFormat) -> Self {
        self.format = format;
        self
    }

    /// Override automatic color detection.
    #[must_use]
    pub const fn with_color(mut self, color: ColorMode) -> Self {
        self.color = color;
        self
    }

    /// Apply environment-based resolution to defaults — currently only
    /// substitutes the GitHub renderer for `format = Auto` under CI.
    /// Call this after CLI parsing but before passing options to `CargoShear`.
    #[must_use]
    pub fn resolve(mut self) -> Self {
        self.format = self.format.resolve();
        self
    }
}

pub(crate) fn default_path() -> Result<PathBuf> {
    Ok(env::current_dir()?)
}

/// Top-level entry point: orchestrates `cargo metadata`, runs each package
/// through [`PackageProcessor`], optionally rewrites manifests, and renders
/// the aggregated [`ShearAnalysis`] to `writer`.
pub struct CargoShear<W> {
    /// Sink for rendered diagnostics (typically `std::io::stdout()`).
    writer: W,

    /// Caller-supplied configuration; immutable once `run` starts.
    options: CargoShearOptions,

    /// Diagnostics accumulated as each package is processed.
    analysis: ShearAnalysis,

    /// What `--cache` reused and recorded during the run.
    cache_report: CacheReport,

    /// What the run observes of its surroundings.
    surroundings: Surroundings,
}

/// What a run observes of its surroundings, which tests replace.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Surroundings {
    /// When the run started, or `None` for the current time.
    pub started: Option<SystemTime>,
    /// Whether standard error is a terminal, or `None` to ask the system.
    pub stderr_is_terminal: Option<bool>,
}

impl Surroundings {
    const SYSTEM: Self = Self { started: None, stderr_is_terminal: None };

    fn started(self) -> SystemTime {
        self.started.unwrap_or_else(SystemTime::now)
    }

    fn stderr_is_terminal(self) -> bool {
        self.stderr_is_terminal.unwrap_or_else(|| io::stderr().is_terminal())
    }
}

/// How a run obtained dependency resolution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ResolveSource {
    /// `--cache` was not given.
    #[default]
    Uncached,
    /// `--cache` was given but this workspace's configuration cannot be cached.
    Uncacheable,
    /// Reused from the cache.
    Reused,
    /// Resolved by `cargo metadata` and recorded in the cache.
    Recorded,
    /// Resolved by `cargo metadata` and not recorded, because an input changed
    /// while it ran or too recently to be sure what cargo read, or because it
    /// printed something the `--no-deps` run did not.
    Unrecorded,
    /// Resolved by `cargo metadata` with its output going straight to a
    /// terminal, as it does without `--cache`, so what it printed could not be
    /// compared with the `--no-deps` run and nothing was recorded.
    Unobserved,
}

/// What `--cache` reused and recorded during one run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CacheReport {
    pub resolve: ResolveSource,
    pub parsed_files: usize,
    pub reused_files: usize,
    /// Which metadata the workspace's sources were parsed from.
    pub parsed_from: Option<ParsedFrom>,
}

/// Which `cargo metadata` result a run parsed the workspace's sources from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParsedFrom {
    /// The members that `cargo metadata --no-deps` listed.
    Members,
    /// The full resolution, because there was no `--no-deps` result or it
    /// listed different members.
    Resolved,
}

/// A workspace context together with what parsing it through the parse cache
/// produced.
struct ParsedWorkspace {
    workspace_ctx: WorkspaceContext,
    parse_cache: Option<(CacheDir, Digest, ParseCache)>,
    outcome: ParseOutcome,
}

/// What a run that missed the resolution cache needs to record the full
/// resolution.
struct CacheMiss<'a> {
    context: &'a KeyContext,
    before: &'a ResolveKey,
    dir: &'a CacheDir,
    warnings: &'a [u8],
    started: SystemTime,
}

impl CacheMiss<'_> {
    /// Record `resolved`, which the full `cargo metadata` run produced
    /// together with `metadata` while printing `printed`, when it is safe to.
    fn record(
        &self,
        metadata: &Metadata,
        resolved: &ResolvedDeps,
        printed: &[u8],
    ) -> ResolveSource {
        if resolve_cache::without_lock_waits(printed)
            == resolve_cache::without_lock_waits(self.warnings)
            && let Some(after) = ResolveKey::compute(self.context, metadata)
            && after.digest == self.before.digest
            && let Some(watched) = Watched::collect(metadata)
        {
            let mut newest = resolve_cache::newest_input(self.before, &watched);
            newest.merge(after.newest);
            if newest.settled_before(self.started)
                && resolve_cache::store(self.dir, self.before, &watched, resolved, metadata)
                    .is_some()
            {
                return ResolveSource::Recorded;
            }
        }
        ResolveSource::Unrecorded
    }
}

impl<W: Write> CargoShear<W> {
    /// Build a runner that will write diagnostics to `writer` using the
    /// settings in `options`.
    ///
    /// ```
    /// use cargo_shear::{CargoShear, CargoShearOptions};
    /// use std::path::PathBuf;
    ///
    /// let options = CargoShearOptions::new(PathBuf::from("."));
    /// let shear = CargoShear::new(std::io::stdout(), options);
    /// ```
    #[must_use]
    pub fn new(writer: W, options: CargoShearOptions) -> Self {
        let analysis = ShearAnalysis::new(options.clone());
        Self {
            writer,
            options,
            analysis,
            cache_report: CacheReport::default(),
            surroundings: Surroundings::SYSTEM,
        }
    }

    /// Execute the full pipeline: analyze every selected package, optionally
    /// apply `--fix` rewrites, render the report, and translate the result
    /// into a process exit code.
    ///
    /// Returns:
    /// - `0` if no issues were found, or every found issue was fixed by `--fix`.
    /// - `1` if issues were found and not all of them were fixed.
    /// - `2` if a fatal error occurred (cargo failure, IO error, ...).
    #[must_use]
    pub fn run(mut self) -> ExitCode {
        self.execute()
    }

    /// Replace what the run observes of its surroundings.
    #[cfg(test)]
    pub(crate) const fn with_surroundings(mut self, surroundings: Surroundings) -> Self {
        self.surroundings = surroundings;
        self
    }

    /// Run like [`run`](Self::run) and also return what `--cache` did.
    #[cfg(test)]
    pub(crate) fn run_with_report(mut self) -> (ExitCode, CacheReport) {
        let exit_code = self.execute();
        (exit_code, self.cache_report)
    }

    fn execute(&mut self) -> ExitCode {
        match self.shear() {
            Ok(()) => {
                let color = self.options.color.enabled();
                let mut renderer = Renderer::new(&mut self.writer, self.options.format, color);

                if let Err(err) = renderer.render(&self.analysis) {
                    let _ = writeln!(self.writer, "error rendering report: {err:?}");
                    return ExitCode::from(2);
                }

                self.determine_exit_code()
            }
            Err(err) => {
                let _ = writeln!(self.writer, "error: {err:?}");
                ExitCode::from(2)
            }
        }
    }

    /// Translate accumulated counts into the documented `0`/`1`/`2` exit code.
    /// `2` is reserved for fatal errors and is set in [`run`](Self::run).
    const fn determine_exit_code(&self) -> ExitCode {
        // `--fix` succeeded if at least one repair landed and nothing else errored.
        if self.options.fix && self.analysis.fixed > 0 && self.analysis.errors == 0 {
            return ExitCode::SUCCESS;
        }

        // Errors always fail; warnings only fail when `--deny-warnings` was passed.
        let has_errors = self.analysis.errors > 0;
        let has_warnings = self.options.deny_warnings && self.analysis.warnings > 0;

        if has_errors || has_warnings { ExitCode::FAILURE } else { ExitCode::SUCCESS }
    }

    fn shear(&mut self) -> Result<()> {
        let mut extra_opts = Vec::new();
        if self.options.locked {
            extra_opts.push("--locked".to_owned());
        }
        if self.options.offline {
            extra_opts.push("--offline".to_owned());
        }
        if self.options.frozen {
            extra_opts.push("--frozen".to_owned());
        }

        let processor = PackageProcessor::new(self.options.expand, self.options.check_test_targets);
        let (metadata, resolved, workspace_ctx) = if self.options.cache {
            self.cached_metadata(&extra_opts)?
        } else {
            self.uncached_metadata(&extra_opts)?
        };

        let packages = metadata.workspace_packages();
        let packages: Vec<_> = packages
            .into_iter()
            .filter(|package| {
                // `--exclude` always wins over `--package` (matches Cargo's behaviour).
                if self.options.exclude.iter().any(|name| name == package.name.as_str()) {
                    return false;
                }

                // When `--package` is given, restrict the run to its allowlist.
                if !self.options.package.is_empty()
                    && !self.options.package.iter().any(|name| name == package.name.as_str())
                {
                    return false;
                }

                true
            })
            .collect();

        let total = packages.len();
        let results: Vec<_> = if self.options.expand {
            // `cargo expand` shells out to `cargo build` per package, which holds
            // a global build lock — running these in parallel would just serialize
            // on the lock and lose the progress output. Stay sequential.
            packages
                .iter()
                .enumerate()
                .map(|(index, package)| {
                    eprintln!(
                        "{:>12} {} [{}/{}]",
                        "Expanding".bright_cyan().bold(),
                        package.name,
                        index + 1,
                        total
                    );

                    Self::process_package(&processor, &workspace_ctx, package, &resolved)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            // Process packages in parallel
            packages
                .par_iter()
                .map(|package| {
                    Self::process_package(&processor, &workspace_ctx, package, &resolved)
                })
                .collect::<Result<Vec<_>>>()?
        };

        let mut used_workspace_ignore_paths: FxHashSet<String> = FxHashSet::default();
        let mut used_ignores: FxHashSet<String> = FxHashSet::default();
        for (ctx, result) in results {
            let fixed = self.fix_package_issues(&ctx.manifest_path, &result)?;
            used_workspace_ignore_paths.extend(result.used_workspace_ignore_paths.iter().cloned());
            used_ignores.extend(result.used_ignores.iter().cloned());
            self.analysis.add_package_result(&ctx, &result, fixed);
        }

        // Workspace-level diagnostics need a complete view of which workspace
        // dependencies are actually used; skip them when the user has narrowed
        // the run with `--package`/`--exclude` and the picture is incomplete.
        if self.options.package.is_empty() && self.options.exclude.is_empty() {
            let workspace_result = PackageProcessor::process_workspace(
                &workspace_ctx,
                &self.analysis.packages,
                &used_workspace_ignore_paths,
                used_ignores,
            );

            let fixed =
                self.fix_workspace_issues(&workspace_ctx.manifest_path, &workspace_result)?;
            self.analysis.add_workspace_result(&workspace_ctx, &workspace_result, fixed);
        }

        Ok(())
    }

    /// Run `cargo metadata` without resolving dependencies, returning the
    /// metadata and what cargo printed. `None` when cargo fails or prints
    /// no metadata.
    fn members_metadata(path: &Path, extra_opts: &[String]) -> Option<(Metadata, Vec<u8>)> {
        let output = Self::full_metadata_command(path, extra_opts)
            .no_deps()
            .cargo_command()
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stdout = std::str::from_utf8(&output.stdout).ok()?;
        let json = stdout.lines().find(|line| line.starts_with('{'))?;
        let metadata = MetadataCommand::parse(json).ok()?;
        Some((metadata, output.stderr))
    }

    /// Run `resolve`, which runs `cargo metadata` with full dependency
    /// resolution, on another thread while the sources of the members that
    /// `members` lists are parsed. That parse is used only when its members
    /// equal the resolved metadata's; otherwise, and when there are no
    /// members to parse from, the sources are parsed from the resolved
    /// metadata exactly as before.
    fn resolve_while_parsing<T: Send>(
        &mut self,
        members: impl FnOnce() -> Option<Metadata>,
        resolve: impl FnOnce() -> Result<(Metadata, T)> + Send,
    ) -> Result<(Metadata, T, WorkspaceContext)> {
        let this = &*self;
        let (resolved, prefetched) = thread::scope(|scope| {
            let resolved = scope.spawn(resolve);
            let prefetched = members().and_then(|members| {
                let parsed = this.parse_workspace(&members).ok()?;
                Some((members, parsed))
            });
            let resolved = resolved.join().unwrap_or_else(|payload| panic::resume_unwind(payload));
            (resolved, prefetched)
        });
        let (metadata, resolution) = resolved?;

        let parsed = match prefetched {
            Some((members, parsed)) if Self::same_workspace(&members, &metadata) => {
                self.cache_report.parsed_from = Some(ParsedFrom::Members);
                parsed
            }
            _ => {
                self.cache_report.parsed_from = Some(ParsedFrom::Resolved);
                self.parse_workspace(&metadata)?
            }
        };
        let workspace_ctx = self.record_parse(parsed);
        Ok((metadata, resolution, workspace_ctx))
    }

    fn same_workspace(members: &Metadata, metadata: &Metadata) -> bool {
        let mut expected = metadata.workspace_packages();
        let mut prefetched = members.workspace_packages();
        expected.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        prefetched.sort_unstable_by(|a, b| a.id.cmp(&b.id));
        members.workspace_root == metadata.workspace_root && prefetched == expected
    }

    /// Build the workspace context for `metadata`, parsing source files
    /// through the parse cache when `--cache` is given.
    fn parse_workspace(&self, metadata: &Metadata) -> Result<ParsedWorkspace> {
        let parse_cache = if self.options.cache {
            executable_digest().map(|executable| {
                let dir = CacheDir::new(
                    metadata.target_directory.as_std_path(),
                    metadata.workspace_root.as_std_path(),
                );
                let cache = ParseCache::load(&dir, &executable);
                (dir, executable, cache)
            })
        } else {
            None
        };
        let (workspace_ctx, outcome) =
            WorkspaceContext::new(metadata, parse_cache.as_ref().map(|(_, _, cache)| cache))?;
        Ok(ParsedWorkspace { workspace_ctx, parse_cache, outcome })
    }

    /// Record what parsing through the parse cache produced, and return the
    /// workspace context.
    fn record_parse(&mut self, parsed: ParsedWorkspace) -> WorkspaceContext {
        let ParsedWorkspace { workspace_ctx, parse_cache, outcome } = parsed;
        if let Some((dir, executable, cache)) = &parse_cache {
            self.cache_report.parsed_files = outcome.entries.len();
            self.cache_report.reused_files = outcome.reused;
            if cache.is_stale(outcome.entries.iter()) {
                let _ = parse_cache::store(dir, executable, outcome.entries.iter());
            }
        }
        workspace_ctx
    }

    /// Run `cargo metadata` with full dependency resolution.
    fn full_metadata(path: &Path, extra_opts: &[String]) -> Result<Metadata> {
        Self::full_metadata_command(path, extra_opts)
            .verbose(true)
            .exec()
            .map_err(|e| anyhow::anyhow!("Metadata error: {e}"))
    }

    fn full_metadata_command(path: &Path, extra_opts: &[String]) -> MetadataCommand {
        let mut command = MetadataCommand::new();
        command
            .features(CargoOpt::AllFeatures)
            .current_dir(path)
            .other_options(extra_opts.to_vec());
        command
    }

    /// Run `cargo metadata` with full dependency resolution exactly as
    /// [`full_metadata`](Self::full_metadata) does, except that cargo's
    /// standard error is collected and then copied to this process's standard
    /// error, rather than written to it directly, and also returned. Only for
    /// a standard error that is not a terminal, which cargo writes to in the
    /// same way either way.
    fn full_metadata_capturing_stderr(
        path: &Path,
        extra_opts: &[String],
    ) -> Result<(Metadata, Vec<u8>)> {
        let output = Self::full_metadata_command(path, extra_opts)
            .cargo_command()
            .output()
            .map_err(|e| anyhow::anyhow!("Metadata error: {}", cargo_metadata::Error::from(e)))?;
        let _ = io::stderr().write_all(&output.stderr);

        let parse = || -> std::result::Result<Metadata, cargo_metadata::Error> {
            if !output.status.success() {
                return Err(cargo_metadata::Error::CargoMetadata { stderr: String::new() });
            }
            let stdout = std::str::from_utf8(&output.stdout)?;
            let json = stdout
                .lines()
                .find(|line| line.starts_with('{'))
                .ok_or(cargo_metadata::Error::NoJson)?;
            MetadataCommand::parse(json)
        };
        let metadata = parse().map_err(|e| anyhow::anyhow!("Metadata error: {e}"))?;
        Ok((metadata, output.stderr))
    }

    /// Obtain the workspace metadata, its resolved dependencies, and the
    /// workspace context without `--cache`.
    ///
    /// The full `cargo metadata` runs on another thread while the sources of
    /// the members that `cargo metadata --no-deps` lists are parsed.
    fn uncached_metadata(
        &mut self,
        extra_opts: &[String],
    ) -> Result<(Metadata, ResolvedDeps, WorkspaceContext)> {
        let path = self.options.path.clone();
        let (metadata, (), workspace_ctx) = self.resolve_while_parsing(
            || Self::members_metadata(&path, extra_opts).map(|(members, _)| members),
            || Ok((Self::full_metadata(&path, extra_opts)?, ())),
        )?;
        let resolved = ResolvedDeps::from_metadata(&metadata);
        Ok((metadata, resolved, workspace_ctx))
    }

    /// Obtain the workspace metadata, its resolved dependencies, and the
    /// workspace context through the resolution cache (see
    /// [`resolve_cache`]).
    ///
    /// `cargo metadata --no-deps` runs first and supplies the members, their
    /// targets, and the primary key. On a hit, its metadata is used together
    /// with the recorded resolution, and what it printed is passed through,
    /// which an entry guarantees is what the full resolution printed. When
    /// standard error is a terminal, cargo would have colored that output, so
    /// a hit is taken only when there is none. Otherwise the full
    /// `cargo metadata` runs as it does without `--cache`, while the sources
    /// of the members the `--no-deps` run listed are parsed, and its result is
    /// recorded when it is safe to.
    fn cached_metadata(
        &mut self,
        extra_opts: &[String],
    ) -> Result<(Metadata, ResolvedDeps, WorkspaceContext)> {
        let started = self.surroundings.started();
        let stderr_is_terminal = self.surroundings.stderr_is_terminal();
        let flags = [self.options.locked, self.options.offline, self.options.frozen];
        let path = self.options.path.clone();

        let Some(context) = KeyContext::capture(flags, &path) else {
            self.cache_report.resolve = ResolveSource::Uncacheable;
            return self.uncached_metadata(extra_opts);
        };

        let Some((mut members, warnings)) = Self::members_metadata(&path, extra_opts) else {
            self.cache_report.resolve = ResolveSource::Uncacheable;
            let (metadata, (), workspace_ctx) = self.resolve_while_parsing(
                || None,
                || Ok((Self::full_metadata(&path, extra_opts)?, ())),
            )?;
            let resolved = ResolvedDeps::from_metadata(&metadata);
            return Ok((metadata, resolved, workspace_ctx));
        };

        let Some(before) = ResolveKey::compute(&context, &members) else {
            self.cache_report.resolve = ResolveSource::Uncacheable;
            let (metadata, (), workspace_ctx) = self.resolve_while_parsing(
                || Some(members),
                || Ok((Self::full_metadata(&path, extra_opts)?, ())),
            )?;
            let resolved = ResolvedDeps::from_metadata(&metadata);
            return Ok((metadata, resolved, workspace_ctx));
        };
        let dir = CacheDir::new(
            members.target_directory.as_std_path(),
            members.workspace_root.as_std_path(),
        );

        if (warnings.is_empty() || !stderr_is_terminal)
            && let Some(resolved) = resolve_cache::load(&dir, &before, &mut members)
        {
            let _ = io::stderr().write_all(&warnings);
            self.cache_report.resolve = ResolveSource::Reused;
            self.cache_report.parsed_from = Some(ParsedFrom::Members);
            let parsed = self.parse_workspace(&members)?;
            let workspace_ctx = self.record_parse(parsed);
            return Ok((members, resolved, workspace_ctx));
        }

        if stderr_is_terminal {
            self.cache_report.resolve = ResolveSource::Unobserved;
            let (metadata, (), workspace_ctx) = self.resolve_while_parsing(
                || Some(members),
                || Ok((Self::full_metadata(&path, extra_opts)?, ())),
            )?;
            let resolved = ResolvedDeps::from_metadata(&metadata);
            return Ok((metadata, resolved, workspace_ctx));
        }

        let miss = CacheMiss {
            context: &context,
            before: &before,
            dir: &dir,
            warnings: &warnings,
            started,
        };
        let (metadata, (resolved, source), workspace_ctx) = self.resolve_while_parsing(
            || Some(members),
            || {
                let (metadata, printed) = Self::full_metadata_capturing_stderr(&path, extra_opts)?;
                let resolved = ResolvedDeps::from_metadata(&metadata);
                let source = miss.record(&metadata, &resolved, &printed);
                Ok((metadata, (resolved, source)))
            },
        )?;
        self.cache_report.resolve = source;
        Ok((metadata, resolved, workspace_ctx))
    }

    fn process_package<'a>(
        processor: &PackageProcessor,
        workspace_ctx: &'a WorkspaceContext,
        package: &Package,
        resolved: &ResolvedDeps,
    ) -> Result<(PackageContext<'a>, PackageAnalysis)> {
        let ctx = PackageContext::new(workspace_ctx, package, resolved)?;
        let result = processor.process_package(&ctx)?;
        Ok((ctx, result))
    }

    fn fix_package_issues(&self, manifest_path: &Path, result: &PackageAnalysis) -> Result<usize> {
        if !self.options.fix {
            return Ok(0);
        }

        if !result.has_fixable_issues() {
            return Ok(0);
        }

        let content = read_to_string(manifest_path)?;
        let mut manifest = DocumentMut::from_str(&content)?;

        let fixed_unused =
            CargoTomlEditor::remove_dependencies(&mut manifest, &result.unused_dependencies);
        let fixed_misplaced = CargoTomlEditor::move_to_dev_dependencies(
            &mut manifest,
            &result.misplaced_dependencies,
        );
        let mut flag_fixes = 0usize;
        if !result.test_disabled_with_tests.is_empty() {
            flag_fixes += usize::from(CargoTomlEditor::remove_lib_flag(&mut manifest, "test"));
        }
        if !result.test_enabled_without_tests.is_empty() {
            CargoTomlEditor::set_lib_flag_false(&mut manifest, "test");
            flag_fixes += 1;
        }
        if !result.doctest_disabled_with_doctests.is_empty() {
            flag_fixes += usize::from(CargoTomlEditor::remove_lib_flag(&mut manifest, "doctest"));
        }
        if !result.doctest_enabled_without_doctests.is_empty() {
            CargoTomlEditor::set_lib_flag_false(&mut manifest, "doctest");
            flag_fixes += 1;
        }

        fs::write(manifest_path, manifest.to_string())?;
        Ok(fixed_unused + fixed_misplaced + flag_fixes)
    }

    fn fix_workspace_issues(
        &self,
        manifest_path: &Path,
        result: &WorkspaceAnalysis,
    ) -> Result<usize> {
        if !self.options.fix {
            return Ok(0);
        }

        if result.unused_dependencies.is_empty() {
            return Ok(0);
        }

        let content = read_to_string(manifest_path)?;
        let mut manifest = DocumentMut::from_str(&content)?;

        let fixed =
            CargoTomlEditor::remove_workspace_deps(&mut manifest, &result.unused_dependencies);

        fs::write(manifest_path, manifest.to_string())?;
        Ok(fixed)
    }
}
