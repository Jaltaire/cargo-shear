//! Tests for `--cache`: every run with the cache must report and fix exactly
//! what a run without it does, and every input the cache depends on must
//! invalidate it.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, SystemTime},
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tempfile::TempDir;

use crate::{
    CacheReport, CargoShear, CargoShearOptions, ResolveSource, Surroundings,
    cache::{CacheDir, Newest, RECENT_CHANGE_WINDOW},
    output::{ColorMode, OutputFormat},
    resolve_cache::{
        KeyContext, ResolveKey, Watched, cargo_identity, index_cache_entry, relevant_env,
        without_lock_waits,
    },
};

/// One run's observable result.
#[derive(Debug, PartialEq)]
struct Outcome {
    exit_code: ExitCode,
    output: String,
}

#[derive(Debug, Clone, Copy, Default)]
struct Flags {
    cache: bool,
    fix: bool,
    check_test_targets: bool,
    json: bool,
    /// When the run starts. `None` starts it an hour from now, long after
    /// every change a test has made, so that the run may record what it reads.
    started: Option<SystemTime>,
    /// Whether the run's standard error is a terminal.
    terminal: bool,
}

fn run(path: &Path, flags: Flags) -> (Outcome, CacheReport) {
    let mut options = CargoShearOptions::new(path.to_path_buf()).with_color(ColorMode::Never);
    if flags.cache {
        options = options.with_cache();
    }
    if flags.fix {
        options = options.with_fix();
    }
    if flags.check_test_targets {
        options = options.with_check_test_targets();
    }
    if flags.json {
        options = options.with_format(OutputFormat::Json);
    }

    let surroundings = Surroundings {
        started: Some(
            flags.started.unwrap_or_else(|| SystemTime::now() + Duration::from_secs(3600)),
        ),
        stderr_is_terminal: Some(flags.terminal),
    };

    let mut output = Vec::new();
    let (exit_code, report) =
        CargoShear::new(&mut output, options).with_surroundings(surroundings).run_with_report();
    (Outcome { exit_code, output: String::from_utf8(output).unwrap() }, report)
}

fn uncached(path: &Path) -> Outcome {
    run(path, Flags::default()).0
}

fn cached(path: &Path) -> (Outcome, CacheReport) {
    run(path, Flags { cache: true, ..Flags::default() })
}

fn write(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

fn copy_dir(src: &Path, dst: &Path) {
    if src.is_dir() {
        fs::create_dir_all(dst).unwrap();
        for entry in fs::read_dir(src).unwrap() {
            let entry = entry.unwrap();
            copy_dir(&entry.path(), &dst.join(entry.file_name()));
        }
    } else {
        fs::copy(src, dst).unwrap();
    }
}

/// The cache directory of the workspace at `root`.
fn cache_dir(root: &Path) -> PathBuf {
    let metadata =
        cargo_metadata::MetadataCommand::new().no_deps().current_dir(root).exec().unwrap();
    CacheDir::new(metadata.target_directory.as_std_path(), metadata.workspace_root.as_std_path())
        .path()
        .to_path_buf()
}

/// When `path` last changed: the later of its modification and status-change
/// times.
fn changed_at(path: &Path) -> SystemTime {
    use std::os::unix::fs::MetadataExt;

    let metadata = fs::metadata(path).unwrap();
    let ctime = SystemTime::UNIX_EPOCH
        + Duration::new(
            metadata.ctime().try_into().unwrap(),
            metadata.ctime_nsec().try_into().unwrap(),
        );
    metadata.modified().unwrap().max(ctime)
}

/// Every file under `root` except the target directory, by relative path.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "target") {
                continue;
            }
            for entry in fs::read_dir(&path).unwrap() {
                stack.push(entry.unwrap().path());
            }
        } else {
            files.insert(path.strip_prefix(root).unwrap().to_path_buf(), fs::read(&path).unwrap());
        }
    }
    files
}

/// The fixtures every integration test runs on, except those that only make
/// sense with `--expand` (which needs a nightly toolchain).
fn fixtures() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
    let mut fixtures: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.is_dir() && !path.file_name().unwrap().to_string_lossy().starts_with("expand_")
        })
        .collect();
    fixtures.sort();
    fixtures
}

/// Check one fixture under one set of flags: a run without the cache, a run
/// that records the cache, and a run that reuses it must agree on the report,
/// the exit code, and every byte `--fix` leaves on disk.
fn check_fixture(fixture: &Path, flags: Flags) {
    let name = fixture.file_name().unwrap().to_string_lossy().into_owned();

    let reference = TempDir::new().unwrap();
    copy_dir(fixture, reference.path());
    let expected = run(reference.path(), Flags { cache: false, ..flags }).0;
    let expected_tree = snapshot(reference.path());

    let cold = TempDir::new().unwrap();
    copy_dir(fixture, cold.path());
    let (actual, _) = run(cold.path(), Flags { cache: true, ..flags });
    assert_eq!(actual, expected, "{name}: a run that fills the cache differs ({flags:?})");
    assert_eq!(snapshot(cold.path()), expected_tree, "{name}: a cold --fix differs ({flags:?})");

    let warm = TempDir::new().unwrap();
    copy_dir(fixture, warm.path());
    let resolved = uncached(warm.path());
    let (_, report) = run(warm.path(), Flags { cache: true, fix: false, ..flags });
    if resolved.exit_code == ExitCode::from(2) {
        assert_ne!(report.resolve, ResolveSource::Reused, "{name}: an error was cached");
        return;
    }
    assert_eq!(report.resolve, ResolveSource::Recorded, "{name}: nothing was recorded");

    let (actual, report) = run(warm.path(), Flags { cache: true, ..flags });
    assert_eq!(report.resolve, ResolveSource::Reused, "{name}: the cache was not reused");
    assert_eq!(report.reused_files, report.parsed_files, "{name}: files were parsed again");
    assert_eq!(actual, expected, "{name}: a run from the cache differs ({flags:?})");
    assert_eq!(
        snapshot(warm.path()),
        expected_tree,
        "{name}: a --fix from the cache differs ({flags:?})"
    );
}

#[test]
fn every_fixture_matches_uncached_runs() {
    let fixtures = fixtures();
    assert!(fixtures.len() > 40, "The fixtures were not found.");

    let combinations = [
        Flags::default(),
        Flags { json: true, ..Flags::default() },
        Flags { check_test_targets: true, ..Flags::default() },
        Flags { fix: true, ..Flags::default() },
        Flags { fix: true, check_test_targets: true, ..Flags::default() },
    ];

    let cases: Vec<_> = fixtures
        .iter()
        .flat_map(|fixture| combinations.iter().map(move |flags| (fixture, *flags)))
        .collect();
    cases.par_iter().for_each(|(fixture, flags)| check_fixture(fixture, *flags));
}

/// A workspace whose members are listed against alphabetical order (so that
/// `--no-deps` and a full resolution list them differently), with findings in
/// more than one member, and a path dependency outside the workspace whose
/// `[lib] name` differs from its package name.
struct Workspace {
    _dir: TempDir,
    outer: PathBuf,
    root: PathBuf,
}

impl Workspace {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let outer = dir.path().canonicalize().unwrap();
        let root = outer.join("ws");

        write(
            &root,
            "Cargo.toml",
            r#"[workspace]
members = ["crates/zeta", "crates/alpha"]
resolver = "2"

[workspace.dependencies]
unused-ws = { path = "crates/unused-ws" }
"#,
        );
        write(
            &root,
            "crates/zeta/Cargo.toml",
            r#"[package]
name = "zeta"
version = "0.1.0"
edition = "2021"

[dependencies]
helper = { path = "../../../outside/helper" }
alpha = { path = "../alpha" }
"#,
        );
        write(&root, "crates/zeta/src/lib.rs", "pub fn f() { helper_lib::g(); }\n");
        write(&root, "crates/zeta/src/orphan.rs", "pub fn orphan() {}\n");
        write(
            &root,
            "crates/alpha/Cargo.toml",
            r#"[package]
name = "alpha"
version = "0.1.0"
edition = "2021"

[dependencies]
helper = { path = "../../../outside/helper" }
"#,
        );
        write(&root, "crates/alpha/src/lib.rs", "pub fn a() {}\n");
        write(&root, "crates/alpha/src/stray.rs", "pub fn stray() {}\n");
        write(
            &root,
            "crates/unused-ws/Cargo.toml",
            "[package]\nname = \"unused-ws\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        );
        write(&root, "crates/unused-ws/src/lib.rs", "");
        write(
            &outer,
            "outside/helper/Cargo.toml",
            r#"[package]
name = "helper"
version = "0.1.0"
edition = "2021"

[lib]
name = "helper_lib"
"#,
        );
        write(&outer, "outside/helper/src/lib.rs", "pub fn g() {}\n");
        Self { _dir: dir, outer, root }
    }

    /// Resolve once without the cache (which writes `Cargo.lock`), record the
    /// cache, and confirm the next run reuses it.
    fn warmed() -> Self {
        let workspace = Self::new();
        let expected = uncached(&workspace.root);
        assert_eq!(expected.exit_code, ExitCode::FAILURE, "{}", expected.output);

        let (outcome, report) = cached(&workspace.root);
        assert_eq!(report.resolve, ResolveSource::Recorded);
        assert_eq!(report.reused_files, 0);
        assert_eq!(outcome, expected);

        let (outcome, report) = cached(&workspace.root);
        assert_eq!(report.resolve, ResolveSource::Reused);
        assert_eq!(report.reused_files, report.parsed_files);
        assert_eq!(outcome, expected);

        workspace
    }

    /// After `change`, the cache must not be reused, the result must match a
    /// run without the cache, and the run after it must reuse what it
    /// recorded.
    #[track_caller]
    fn assert_invalidated_by(&self, change: impl FnOnce(&Path, &Path)) -> Outcome {
        change(&self.outer, &self.root);

        let (outcome, report) = cached(&self.root);
        assert_ne!(report.resolve, ResolveSource::Reused, "The cache survived the change.");
        let expected = uncached(&self.root);
        assert_eq!(outcome, expected);

        let (_, report) = cached(&self.root);
        if expected.exit_code != ExitCode::from(2) {
            assert_ne!(report.resolve, ResolveSource::Uncacheable);
        }
        outcome
    }
}

#[test]
fn reused_runs_keep_the_full_resolution_package_order() {
    let workspace = Workspace::warmed();
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Reused);
    let zeta = outcome.output.find("crates/zeta").expect("zeta has findings");
    let alpha = outcome.output.find("crates/alpha").expect("alpha has findings");
    assert!(alpha < zeta, "{}", outcome.output);
}

#[test]
fn lib_name_of_an_outside_path_dependency_invalidates() {
    let workspace = Workspace::warmed();
    let before = cached(&workspace.root).0;
    let after = workspace.assert_invalidated_by(|outer, _| {
        let manifest = outer.join("outside/helper/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, contents.replace("helper_lib", "helper_core")).unwrap();
    });
    assert_ne!(before, after, "Renaming the library must change the findings.");
}

#[test]
fn outside_path_dependency_layout_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|outer, _| {
        write(outer, "outside/helper/src/bin/tool.rs", "fn main() {}\n");
    });
}

#[test]
fn editing_outside_sources_keeps_the_resolution() {
    let workspace = Workspace::warmed();
    write(&workspace.outer, "outside/helper/src/lib.rs", "pub fn g() {}\npub fn h() {}\n");
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Reused);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn outside_target_with_an_explicit_path_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|outer, _| {
        let manifest = outer.join("outside/helper/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(
            manifest,
            format!("{contents}\n[[bin]]\nname = \"tool\"\npath = \"tools/tool.rs\"\n"),
        )
        .unwrap();
        write(outer, "outside/helper/tools/tool.rs", "fn main() {}\n");
    });
    workspace.assert_invalidated_by(|outer, _| {
        fs::remove_file(outer.join("outside/helper/tools/tool.rs")).unwrap();
    });
}

#[test]
fn removing_an_outside_library_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|outer, _| {
        fs::remove_file(outer.join("outside/helper/src/lib.rs")).unwrap();
    });
}

#[test]
fn manifest_above_an_outside_path_dependency_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|outer, _| {
        write(
            outer,
            "outside/Cargo.toml",
            "[workspace]\nmembers = [\"helper\"]\n\n[workspace.package]\nversion = \"0.1.0\"\n",
        );
        let manifest = outer.join("outside/helper/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, contents.replace("version = \"0.1.0\"", "version.workspace = true"))
            .unwrap();
    });
    workspace.assert_invalidated_by(|outer, _| {
        let manifest = outer.join("outside/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, contents.replace("0.1.0", "0.2.0")).unwrap();
    });
}

#[test]
fn root_manifest_bytes_invalidate() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        let manifest = root.join("Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, format!("{contents}# A comment changes nothing cargo reads.\n"))
            .unwrap();
    });
}

#[test]
fn lockfile_bytes_invalidate() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        let lock = root.join("Cargo.lock");
        let contents = fs::read_to_string(&lock).unwrap();
        fs::write(lock, format!("{contents}\n")).unwrap();
    });
}

#[test]
fn removing_the_lockfile_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        fs::remove_file(root.join("Cargo.lock")).unwrap();
    });
}

#[test]
fn member_manifest_invalidates() {
    let workspace = Workspace::warmed();
    let before = cached(&workspace.root).0;
    let after = workspace.assert_invalidated_by(|_, root| {
        let manifest = root.join("crates/alpha/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, format!("{contents}unused-ws = {{ workspace = true }}\n")).unwrap();
    });
    assert_ne!(before, after);
}

#[test]
fn member_manifest_bytes_invalidate() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        let manifest = root.join("crates/zeta/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, format!("{contents}# A comment changes nothing cargo reads.\n"))
            .unwrap();
    });
}

#[test]
fn renamed_dependency_invalidates() {
    let workspace = Workspace::warmed();
    let before = cached(&workspace.root).0;
    let after = workspace.assert_invalidated_by(|_, root| {
        let manifest = root.join("crates/alpha/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(
            manifest,
            contents.replace(
                "helper = { path = \"../../../outside/helper\" }",
                "renamed = { package = \"helper\", path = \"../../../outside/helper\" }",
            ),
        )
        .unwrap();
        write(root, "crates/alpha/src/lib.rs", "pub fn a() { renamed::g(); }\n");
    });
    assert_ne!(before, after);
}

#[test]
fn new_member_matched_by_a_glob_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        let manifest = root.join("Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, contents.replace("\"crates/alpha\"]", "\"crates/alpha\", \"more/*\"]"))
            .unwrap();
    });
    workspace.assert_invalidated_by(|_, root| {
        write(
            root,
            "more/extra/Cargo.toml",
            "[package]\nname = \"extra\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nalpha = { path = \"../../crates/alpha\" }\n",
        );
        write(root, "more/extra/src/lib.rs", "");
    });
}

#[test]
fn new_inferred_target_in_a_member_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        write(root, "crates/alpha/tests/it.rs", "#[test]\nfn t() { helper_lib::g(); }\n");
    });
}

#[test]
fn workspace_config_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        write(root, ".cargo/config.toml", "[net]\nretry = 3\n");
    });
}

#[test]
fn ancestor_config_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|outer, _| {
        write(outer, ".cargo/config.toml", "[net]\nretry = 3\n");
    });
}

#[test]
fn legacy_config_name_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|_, root| {
        write(root, ".cargo/config", "[net]\nretry = 3\n");
    });
}

#[test]
fn config_paths_override_disables_the_cache() {
    let workspace = Workspace::warmed();
    write(
        &workspace.outer,
        "override/helper/Cargo.toml",
        "[package]\nname = \"helper\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    );
    write(&workspace.outer, "override/helper/src/lib.rs", "");
    write(&workspace.root, ".cargo/config.toml", "paths = [\"../override\"]\n");
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Uncacheable);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn config_source_replacement_disables_the_cache() {
    let workspace = Workspace::warmed();
    write(
        &workspace.root,
        ".cargo/config.toml",
        "[source.crates-io]\nreplace-with = \"vendored\"\n\n[source.vendored]\ndirectory = \"vendor\"\n",
    );
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Uncacheable);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn unused_patch_is_never_recorded() {
    let workspace = Workspace::warmed();
    write(
        &workspace.outer,
        "patches/patched/Cargo.toml",
        "[package]\nname = \"patched\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    );
    write(&workspace.outer, "patches/patched/src/lib.rs", "");
    let manifest = workspace.root.join("Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(
        &manifest,
        format!("{contents}\n[patch.crates-io]\npatched = {{ path = \"../patches/patched\" }}\n"),
    )
    .unwrap();

    for _ in 0..2 {
        let (outcome, report) = cached(&workspace.root);
        assert_eq!(report.resolve, ResolveSource::Unrecorded);
        assert_eq!(outcome, uncached(&workspace.root));
    }
}

#[test]
fn config_include_disables_the_cache() {
    let workspace = Workspace::warmed();
    write(&workspace.root, ".cargo/config.toml", "include = [\"other.toml\"]\n");
    write(&workspace.root, ".cargo/other.toml", "");
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Uncacheable);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn unstable_config_disables_the_cache() {
    let workspace = Workspace::warmed();
    write(&workspace.root, ".cargo/config.toml", "[unstable]\nmtime-on-use = true\n");
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Uncacheable);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn inputs_changed_during_a_run_are_not_recorded() {
    let workspace = Workspace::warmed();
    let manifest = workspace.root.join("crates/alpha/Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(&manifest, format!("{contents}# Edited while the run was reading it.\n")).unwrap();
    let changed = changed_at(&manifest);
    let at = |started| Flags { cache: true, started: Some(started), ..Flags::default() };

    let (outcome, report) = run(&workspace.root, at(changed));
    assert_eq!(report.resolve, ResolveSource::Unrecorded);
    assert_eq!(outcome, uncached(&workspace.root));
    let (_, report) = run(&workspace.root, at(changed + RECENT_CHANGE_WINDOW));
    assert_eq!(report.resolve, ResolveSource::Unrecorded);

    let (_, report) =
        run(&workspace.root, at(changed + RECENT_CHANGE_WINDOW + Duration::from_secs(1)));
    assert_eq!(report.resolve, ResolveSource::Recorded);
}

#[test]
fn inputs_with_a_modification_time_set_back_are_not_recorded() {
    let workspace = Workspace::warmed();
    let manifest = workspace.root.join("crates/alpha/Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(&manifest, format!("{contents}# Edited while the run was reading it.\n")).unwrap();
    fs::File::open(&manifest)
        .unwrap()
        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000))
        .unwrap();
    let changed = changed_at(&manifest);

    let (outcome, report) =
        run(&workspace.root, Flags { cache: true, started: Some(changed), ..Flags::default() });
    assert_eq!(report.resolve, ResolveSource::Unrecorded);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn outside_inputs_changed_during_a_run_are_not_recorded() {
    let workspace = Workspace::warmed();
    let manifest = workspace.outer.join("outside/helper/Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(&manifest, format!("{contents}# Edited while the run was reading it.\n")).unwrap();
    let changed = changed_at(&manifest);

    let (outcome, report) =
        run(&workspace.root, Flags { cache: true, started: Some(changed), ..Flags::default() });
    assert_eq!(report.resolve, ResolveSource::Unrecorded);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn settled_inputs_respect_the_recent_change_window() {
    let started = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let at = |offset: i64| {
        let mut newest = Newest::default();
        let time = if offset < 0 {
            started - Duration::from_secs(offset.unsigned_abs())
        } else {
            started + Duration::from_secs(offset.unsigned_abs())
        };
        newest.observe_time(time);
        newest
    };

    assert!(Newest::default().settled_before(started));
    assert!(at(-3).settled_before(started));
    assert!(!at(-2).settled_before(started));
    assert!(!at(-1).settled_before(started));
    assert!(!at(0).settled_before(started));
    assert!(!at(5).settled_before(started));

    let mut merged = at(-10);
    merged.merge(at(-1));
    assert!(!merged.settled_before(started));

    let mut unknown = at(-10);
    unknown.observe_unknown();
    assert!(!unknown.settled_before(started));
    let mut merged = at(-10);
    merged.merge(unknown);
    assert!(!merged.settled_before(started));
}

#[test]
fn corrupt_cache_files_are_ignored() {
    let workspace = Workspace::warmed();
    let dir = cache_dir(&workspace.root);
    for name in ["resolve.json", "parse.json"] {
        fs::write(dir.join(name), b"{ not json").unwrap();
    }

    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Recorded);
    assert_eq!(report.reused_files, 0);
    assert_eq!(outcome, uncached(&workspace.root));

    let (_, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Reused);
    assert_eq!(report.reused_files, report.parsed_files);
}

#[test]
fn changed_source_is_parsed_again() {
    let workspace = Workspace::warmed();
    let before = cached(&workspace.root).0;
    write(&workspace.root, "crates/zeta/src/lib.rs", "pub fn f() {}\n");

    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Reused);
    assert_eq!(report.reused_files + 1, report.parsed_files);
    assert_eq!(outcome, uncached(&workspace.root));
    assert_ne!(outcome, before, "Dropping the import must change the findings.");
}

#[test]
fn renamed_source_is_parsed_again() {
    let workspace = Workspace::warmed();
    let zeta = workspace.root.join("crates/zeta/src");
    fs::rename(zeta.join("orphan.rs"), zeta.join("moved.rs")).unwrap();
    write(
        &workspace.root,
        "crates/zeta/src/lib.rs",
        "pub fn f() { helper_lib::g(); }\nmod moved;\n",
    );

    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.reused_files + 2, report.parsed_files);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn source_that_becomes_an_entry_point_is_parsed_again() {
    let workspace = Workspace::warmed();
    let before = cached(&workspace.root).0;
    write(&workspace.root, "crates/alpha/src/stray.rs", "mod inner;\nfn main() {}\n");
    write(&workspace.root, "crates/alpha/src/inner.rs", "");
    let _ = cached(&workspace.root);

    let manifest = workspace.root.join("crates/alpha/Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(
        &manifest,
        format!("{contents}\n[[bin]]\nname = \"stray\"\npath = \"src/stray.rs\"\n"),
    )
    .unwrap();

    let (outcome, report) = cached(&workspace.root);
    assert_ne!(report.resolve, ResolveSource::Reused);
    assert_eq!(report.reused_files + 1, report.parsed_files);
    assert_eq!(outcome, uncached(&workspace.root));
    assert_ne!(outcome, before);
}

#[test]
fn deleted_sources_are_pruned_from_the_cache() {
    let workspace = Workspace::warmed();
    fs::remove_file(workspace.root.join("crates/zeta/src/orphan.rs")).unwrap();
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.reused_files, report.parsed_files);
    assert_eq!(outcome, uncached(&workspace.root));

    let dir = cache_dir(&workspace.root);
    let parse = fs::read_to_string(dir.join("parse.json")).unwrap();
    assert!(!parse.contains("orphan.rs"), "{parse}");
}

fn key_context(root: &Path) -> (KeyContext, cargo_metadata::Metadata) {
    let context = KeyContext::capture([false; 3], root).unwrap();
    let metadata =
        cargo_metadata::MetadataCommand::new().no_deps().current_dir(root).exec().unwrap();
    (context, metadata)
}

fn digest(context: &KeyContext, metadata: &cargo_metadata::Metadata) -> [u8; 32] {
    ResolveKey::compute(context, metadata).unwrap().digest
}

#[test]
fn key_covers_the_executable() {
    let workspace = Workspace::new();
    let (context, metadata) = key_context(&workspace.root);
    let base = digest(&context, &metadata);
    assert_eq!(base, digest(&context, &metadata));

    let mut changed = context;
    changed.executable[0] ^= 1;
    assert_ne!(base, digest(&changed, &metadata));
}

#[test]
fn key_covers_each_cargo_flag() {
    let workspace = Workspace::new();
    let (context, metadata) = key_context(&workspace.root);
    let base = digest(&context, &metadata);
    for index in 0..3 {
        let mut changed = context.clone();
        changed.flags[index] = true;
        assert_ne!(base, digest(&changed, &metadata), "flag {index}");
    }
}

#[test]
fn key_covers_the_cargo_binary() {
    let workspace = Workspace::new();
    let (context, metadata) = key_context(&workspace.root);
    let base = digest(&context, &metadata);
    let mut changed = context;
    changed.cargo.push(b'!');
    assert_ne!(base, digest(&changed, &metadata));
}

#[test]
fn key_covers_cargo_environment() {
    let workspace = Workspace::new();
    let (context, metadata) = key_context(&workspace.root);
    let base = digest(&context, &metadata);
    for name in
        ["CARGO_NET_OFFLINE", "RUSTFLAGS", "RUSTUP_TOOLCHAIN", "__CARGO_TEST", "PATH", "HOME"]
    {
        let mut changed = context.clone();
        changed.env.retain(|(existing, _)| existing != name);
        changed.env.push((name.into(), "changed".into()));
        changed.env.sort();
        assert_ne!(base, digest(&changed, &metadata), "{name}");
    }
}

#[test]
fn relevant_env_keeps_cargo_variables_only() {
    let env = relevant_env(
        [
            ("CARGO_HOME", "a"),
            ("RUSTFLAGS", "b"),
            ("__CARGO_FIX", "c"),
            ("PATH", "d"),
            ("HOME", "e"),
            ("GIT_INDEX_FILE", "f"),
            ("PWD", "g"),
        ]
        .into_iter()
        .map(|(name, value)| (name.into(), value.into())),
    );
    let names: Vec<_> = env.iter().map(|(name, _)| name.to_string_lossy().into_owned()).collect();
    assert_eq!(names, ["CARGO_HOME", "HOME", "PATH", "RUSTFLAGS", "__CARGO_FIX"]);
}

#[test]
fn key_covers_cargo_home_config() {
    let workspace = Workspace::new();
    let (mut context, metadata) = key_context(&workspace.root);
    let home = workspace.outer.join("cargo-home");
    context.env.retain(|(name, _)| name != "CARGO_HOME");
    context.env.push(("CARGO_HOME".into(), home.clone().into_os_string()));
    context.env.sort();
    let base = digest(&context, &metadata);

    write(&home, "config.toml", "[net]\nretry = 3\n");
    assert_ne!(base, digest(&context, &metadata));
}

#[test]
fn key_covers_manifests_above_patch_targets() {
    let workspace = Workspace::new();
    write(
        &workspace.outer,
        "patches/patched/Cargo.toml",
        "[package]\nname = \"patched\"\nversion.workspace = true\nedition = \"2021\"\n",
    );
    let manifest = workspace.root.join("Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(
        &manifest,
        format!("{contents}\n[patch.crates-io]\npatched = {{ path = \"../patches/patched\" }}\n"),
    )
    .unwrap();
    let (context, metadata) = key_context(&workspace.root);
    let base = digest(&context, &metadata);

    write(&workspace.outer, "patches/Cargo.toml", "[workspace.package]\nversion = \"0.2.0\"\n");
    let with_parent = digest(&context, &metadata);
    assert_ne!(base, with_parent);

    write(&workspace.outer, "patches/patched/Cargo.toml", "");
    assert_ne!(with_parent, digest(&context, &metadata));
}

#[test]
fn key_covers_patch_targets_named_in_configuration() {
    let workspace = Workspace::new();
    write(
        &workspace.root,
        ".cargo/config.toml",
        "[patch.crates-io]\npatched = { path = \"../patches/patched\" }\n",
    );
    let (context, metadata) = key_context(&workspace.root);
    let base = digest(&context, &metadata);

    write(
        &workspace.outer,
        "patches/patched/Cargo.toml",
        "[package]\nname = \"patched\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    );
    assert_ne!(base, digest(&context, &metadata));
}

#[test]
fn terminal_runs_are_never_recorded() {
    let workspace = Workspace::new();
    let expected = uncached(&workspace.root);
    let terminal = Flags { cache: true, terminal: true, ..Flags::default() };
    for _ in 0..2 {
        let (outcome, report) = run(&workspace.root, terminal);
        assert_eq!(report.resolve, ResolveSource::Unobserved);
        assert_eq!(outcome, expected);
    }
}

#[test]
fn terminal_runs_reuse_entries_for_which_cargo_printed_nothing() {
    let workspace = Workspace::warmed();
    let (outcome, report) =
        run(&workspace.root, Flags { cache: true, terminal: true, ..Flags::default() });
    assert_eq!(report.resolve, ResolveSource::Reused);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn cargo_warnings_are_reused_only_when_identical_and_not_for_a_terminal() {
    let workspace = Workspace::warmed();
    let manifest = workspace.root.join("crates/alpha/Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(&manifest, format!("{contents}\n[profile.release]\nopt-level = 1\n")).unwrap();
    let expected = uncached(&workspace.root);

    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Recorded);
    assert_eq!(outcome, expected);
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Reused);
    assert_eq!(outcome, expected);

    let (outcome, report) =
        run(&workspace.root, Flags { cache: true, terminal: true, ..Flags::default() });
    assert_eq!(report.resolve, ResolveSource::Unobserved);
    assert_eq!(outcome, expected);
}

#[test]
fn stale_lockfile_is_never_reused() {
    let workspace = Workspace::warmed();
    let lock = workspace.root.join("Cargo.lock");
    let current = fs::read_to_string(&lock).unwrap();
    let stale = current.replace("[[package]]\nname = \"helper\"\nversion = \"0.1.0\"\n\n", "");
    assert_ne!(stale, current);

    fs::write(&lock, &stale).unwrap();
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Unrecorded);
    assert_eq!(fs::read_to_string(&lock).unwrap(), current);
    assert_eq!(outcome, uncached(&workspace.root));

    fs::write(&lock, &stale).unwrap();
    let (_, report) = cached(&workspace.root);
    assert_ne!(report.resolve, ResolveSource::Reused);
    assert_eq!(fs::read_to_string(&lock).unwrap(), current);
}

#[test]
fn reused_runs_leave_a_lockfile_cargo_would_leave_alone() {
    let workspace = Workspace::warmed();
    let lock = workspace.root.join("Cargo.lock");
    let before = fs::read(&lock).unwrap();
    let (_, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Reused);
    assert_eq!(fs::read(&lock).unwrap(), before);
    let _ = uncached(&workspace.root);
    assert_eq!(fs::read(&lock).unwrap(), before);
}

#[test]
fn cargo_identity_needs_the_cargo_variable() {
    let dir = TempDir::new().unwrap();
    let cwd = dir.path().canonicalize().unwrap();
    assert!(cargo_identity(&[], &cwd).is_none());
}

/// A cargo home with a `rustup` in its `bin` directory, and the environment
/// naming it, for [`cargo_identity`].
struct Toolchains {
    _dir: TempDir,
    root: PathBuf,
    rustup: PathBuf,
}

impl Toolchains {
    fn new() -> Self {
        let dir = TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let rustup = root.join("home/bin/rustup");
        write(&root, "home/bin/rustup", "\x7fELF rustup");
        Self { _dir: dir, root, rustup }
    }

    fn identity(&self, cargo: &Path) -> Option<Vec<u8>> {
        let env: Vec<(std::ffi::OsString, std::ffi::OsString)> = vec![
            ("CARGO".into(), cargo.into()),
            ("CARGO_HOME".into(), self.root.join("home").into()),
            ("PATH".into(), self.root.join("path").into()),
        ];
        cargo_identity(&env, &self.root)
    }
}

#[test]
fn cargo_identity_covers_the_binary() {
    let toolchains = Toolchains::new();
    let cargo = toolchains.root.join("toolchain/bin/cargo");
    write(&toolchains.root, "toolchain/bin/cargo", "\x7fELF cargo 1");
    let before = toolchains.identity(&cargo).unwrap();
    assert_eq!(toolchains.identity(&cargo).unwrap(), before);

    fs::remove_file(&cargo).unwrap();
    write(&toolchains.root, "toolchain/bin/cargo", "\x7fELF cargo 2");
    assert_ne!(toolchains.identity(&cargo).unwrap(), before);
}

#[test]
fn cargo_identity_refuses_rustup_proxies() {
    let toolchains = Toolchains::new();

    let linked = toolchains.root.join("home/bin/cargo");
    std::os::unix::fs::symlink(&toolchains.rustup, &linked).unwrap();
    assert!(toolchains.identity(&linked).is_none());

    let hard = toolchains.root.join("elsewhere/cargo");
    fs::create_dir_all(hard.parent().unwrap()).unwrap();
    fs::hard_link(&toolchains.rustup, &hard).unwrap();
    assert!(toolchains.identity(&hard).is_none());

    let copy = toolchains.root.join("copy/cargo");
    write(&toolchains.root, "copy/cargo", "\x7fELF rustup");
    assert!(toolchains.identity(&copy).is_none());

    write(&toolchains.root, "path/rustup", "\x7fELF other rustup");
    let on_path = toolchains.root.join("copy2/cargo");
    write(&toolchains.root, "copy2/cargo", "\x7fELF other rustup");
    assert!(toolchains.identity(&on_path).is_none());

    let same_size = toolchains.root.join("same-size/cargo");
    write(&toolchains.root, "same-size/cargo", "\x7fELF rustuq");
    assert!(toolchains.identity(&same_size).is_some());
}

#[test]
fn cargo_identity_refuses_scripts() {
    let toolchains = Toolchains::new();
    let script = toolchains.root.join("wrapper/cargo");
    write(&toolchains.root, "wrapper/cargo", "#!/bin/sh\nexec cargo \"$@\"\n");
    assert!(toolchains.identity(&script).is_none());
}

#[test]
fn index_cache_entries_follow_cargo_layout() {
    let directory = Path::new("/r/src/index.example-0123/x-1.0.0");
    let index = Path::new("/r/index/index.example-0123/.cache");
    for (name, relative) in [
        ("a", "1/a"),
        ("ab", "2/ab"),
        ("abc", "3/a/abc"),
        ("Serde_Json", "se/rd/serde_json"),
        ("itoa", "it/oa/itoa"),
    ] {
        assert_eq!(index_cache_entry(directory, name).unwrap(), index.join(relative), "{name}");
    }
    assert!(index_cache_entry(directory, "").is_none());
}

/// A full resolution of [`Workspace`] with one more package outside it, from
/// `source`, whose manifest is `manifest`.
fn metadata_with_package(
    workspace: &Workspace,
    name: &str,
    source: &str,
    manifest: &Path,
) -> cargo_metadata::Metadata {
    let _ = uncached(&workspace.root);
    let metadata =
        cargo_metadata::MetadataCommand::new().current_dir(&workspace.root).exec().unwrap();
    let mut json = serde_json::to_value(&metadata).unwrap();
    let packages = json["packages"].as_array_mut().unwrap();
    let mut package = packages[0].clone();
    package["name"] = name.into();
    package["id"] = format!("{source}#{name}@1.0.0").into();
    package["source"] = source.into();
    package["manifest_path"] = manifest.to_str().unwrap().into();
    package["targets"] = serde_json::json!([]);
    packages.push(package);
    serde_json::from_value(json).unwrap()
}

#[track_caller]
fn assert_watched(metadata: &cargo_metadata::Metadata, change: impl FnOnce()) {
    let watched = Watched::collect(metadata).expect("The package can be watched.");
    assert!(watched.unchanged());
    change();
    assert!(!watched.unchanged(), "The change went unnoticed.");
}

#[test]
fn registry_packages_are_watched() {
    const SOURCE: &str = "registry+https://example.com/index";
    let layout = |workspace: &Workspace| {
        let package = workspace.outer.join("registry/src/index.example-0123/foo-1.0.0");
        write(&package, "Cargo.toml", "[package]\nname = \"foo\"\nversion = \"1.0.0\"\n");
        write(&package, "src/lib.rs", "");
        write(&package, ".cargo-ok", "{\"v\":1}");
        let entry = workspace.outer.join("registry/index/index.example-0123/.cache/3/f/foo");
        write(&workspace.outer, entry.to_str().unwrap(), "index entry");
        let metadata = metadata_with_package(workspace, "foo", SOURCE, &package.join("Cargo.toml"));
        (package, entry, metadata)
    };

    let workspace = Workspace::new();
    let (package, _, metadata) = layout(&workspace);
    assert_watched(&metadata, || write(&package, "Cargo.toml", "[package]\nname = \"foo\"\n"));

    let workspace = Workspace::new();
    let (package, _, metadata) = layout(&workspace);
    assert_watched(&metadata, || fs::remove_file(package.join(".cargo-ok")).unwrap());

    let workspace = Workspace::new();
    let (package, _, metadata) = layout(&workspace);
    assert_watched(&metadata, || fs::remove_file(package.join("src/lib.rs")).unwrap());

    let workspace = Workspace::new();
    let (_, entry, metadata) = layout(&workspace);
    assert_watched(&metadata, || fs::write(&entry, "a newer index entry").unwrap());

    let workspace = Workspace::new();
    let (_, entry, metadata) = layout(&workspace);
    assert_watched(&metadata, || fs::remove_file(&entry).unwrap());

    let workspace = Workspace::new();
    let (_, entry, metadata) = layout(&workspace);
    fs::remove_file(entry).unwrap();
    assert!(Watched::collect(&metadata).is_none());
}

#[test]
fn git_packages_are_watched() {
    const SOURCE: &str = "git+https://example.com/foo.git#0123456789abcdef";
    let layout = |workspace: &Workspace| {
        let checkout = workspace.outer.join("git/checkouts/foo-0123/0123456");
        let package = checkout.join("crates/foo");
        write(&package, "Cargo.toml", "[package]\nname = \"foo\"\nversion = \"1.0.0\"\n");
        write(&package, "src/lib.rs", "");
        write(&checkout, ".cargo-ok", "");
        let metadata = metadata_with_package(workspace, "foo", SOURCE, &package.join("Cargo.toml"));
        (checkout, package, metadata)
    };

    let workspace = Workspace::new();
    let (checkout, _, metadata) = layout(&workspace);
    assert_watched(&metadata, || fs::remove_file(checkout.join(".cargo-ok")).unwrap());

    let workspace = Workspace::new();
    let (checkout, _, metadata) = layout(&workspace);
    assert_watched(&metadata, || write(&checkout, "Cargo.toml", "[workspace]\n"));

    let workspace = Workspace::new();
    let (_, package, metadata) = layout(&workspace);
    assert_watched(&metadata, || fs::remove_file(package.join("src/lib.rs")).unwrap());

    let workspace = Workspace::new();
    let (_, package, metadata) = layout(&workspace);
    assert_watched(&metadata, || write(&package, "Cargo.toml", "[package]\nname = \"foo\"\n"));
}

#[test]
fn only_lock_waits_are_ignored_in_what_cargo_printed() {
    let printed = b"    Blocking waiting for file lock on package cache\nwarning: unused manifest key: package.x\n  Blocking waiting for file lock on build directory\nnote: Blocking waiting for file lock on nothing\n";
    assert_eq!(
        without_lock_waits(printed),
        b"warning: unused manifest key: package.x\nnote: Blocking waiting for file lock on nothing\n"
    );
    assert_eq!(without_lock_waits(b""), b"");
    assert_eq!(without_lock_waits(b"warning: no newline"), b"warning: no newline");
}
