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
    CacheReport, CargoShear, CargoShearOptions, ResolveSource,
    cache::Newest,
    output::{ColorMode, OutputFormat},
    resolve_cache::{KeyContext, ResolveKey, relevant_env},
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

    let mut output = Vec::new();
    let (exit_code, report) = CargoShear::new(&mut output, options).run_with_report();
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

/// The modification time [`settle`] gives every file it touches: long before
/// any run, so that a run may record what it reads.
fn settled_time() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000)
}

/// Give every file and directory under `root` that does not already have it
/// the [`settled_time`] as its modification time. Without this, the guard that
/// refuses inputs changed within moments of a run would refuse everything a
/// test has just written. Files that already have it are left alone, so that
/// an unchanged file keeps its exact metadata and only what a test changed
/// looks changed.
fn settle(root: &Path) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        if path.is_dir() {
            for entry in fs::read_dir(&path).unwrap() {
                stack.push(entry.unwrap().path());
            }
        }
        if fs::metadata(&path).unwrap().modified().unwrap() != settled_time() {
            fs::File::open(&path).unwrap().set_modified(settled_time()).unwrap();
        }
    }
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
    settle(warm.path());
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
/// more than one member, a path dependency outside the workspace whose
/// `[lib] name` differs from its package name, and an unused patch.
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

[patch.crates-io]
patched = { path = "../patches/patched" }
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
        write(
            &outer,
            "patches/patched/Cargo.toml",
            "[package]\nname = \"patched\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        );
        write(&outer, "patches/patched/src/lib.rs", "");

        Self { _dir: dir, outer, root }
    }

    /// Resolve once without the cache (which writes `Cargo.lock`), settle
    /// everything, record the cache, and confirm the next run reuses it.
    fn warmed() -> Self {
        let workspace = Self::new();
        let expected = uncached(&workspace.root);
        assert_eq!(expected.exit_code, ExitCode::FAILURE, "{}", expected.output);
        settle(&workspace.outer);

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
        settle(&self.outer);

        let (outcome, report) = cached(&self.root);
        assert_ne!(report.resolve, ResolveSource::Reused, "The cache survived the change.");
        let expected = uncached(&self.root);
        assert_eq!(outcome, expected);

        settle(&self.outer);
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
    settle(&workspace.outer);
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
fn config_path_override_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|outer, root| {
        write(
            outer,
            "override/helper/Cargo.toml",
            "[package]\nname = \"helper\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        );
        write(outer, "override/helper/src/lib.rs", "");
        write(root, ".cargo/config.toml", "paths = [\"../override/helper\"]\n");
    });
    workspace.assert_invalidated_by(|outer, _| {
        let manifest = outer.join("override/helper/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, format!("{contents}\n[lib]\nname = \"helper_lib\"\n")).unwrap();
    });
}

#[test]
fn unused_patch_target_invalidates() {
    let workspace = Workspace::warmed();
    workspace.assert_invalidated_by(|outer, _| {
        let manifest = outer.join("patches/patched/Cargo.toml");
        let contents = fs::read_to_string(&manifest).unwrap();
        fs::write(manifest, contents.replace("0.1.0", "0.2.0")).unwrap();
    });
}

#[test]
fn config_include_disables_the_cache() {
    let workspace = Workspace::warmed();
    write(&workspace.root, ".cargo/config.toml", "include = [\"other.toml\"]\n");
    write(&workspace.root, ".cargo/other.toml", "");
    settle(&workspace.outer);
    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Uncacheable);
    assert_eq!(outcome, uncached(&workspace.root));
}

#[test]
fn unstable_config_disables_the_cache() {
    let workspace = Workspace::warmed();
    write(&workspace.root, ".cargo/config.toml", "[unstable]\nmtime-on-use = true\n");
    settle(&workspace.outer);
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
    let during_the_run = SystemTime::now() + Duration::from_secs(3600);
    fs::File::open(&manifest).unwrap().set_modified(during_the_run).unwrap();

    let (outcome, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Unrecorded);
    assert_eq!(outcome, uncached(&workspace.root));
    let (_, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Unrecorded);

    settle(&workspace.outer);
    let (_, report) = cached(&workspace.root);
    assert_eq!(report.resolve, ResolveSource::Recorded);
}

#[test]
fn outside_inputs_changed_during_a_run_are_not_recorded() {
    let workspace = Workspace::warmed();
    let lib = workspace.outer.join("outside/helper/src/lib.rs");
    let during_the_run = SystemTime::now() + Duration::from_secs(3600);
    fs::File::open(&lib).unwrap().set_modified(during_the_run).unwrap();
    let manifest = workspace.outer.join("outside/helper/Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(&manifest, format!("{contents}# Edited while the run was reading it.\n")).unwrap();
    fs::File::open(&manifest).unwrap().set_modified(during_the_run).unwrap();

    let (outcome, report) = cached(&workspace.root);
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
    let cache = workspace.root.join("target").join("cargo-shear");
    let dir = fs::read_dir(&cache).unwrap().next().unwrap().unwrap().path();
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
    settle(&workspace.outer);
    let _ = cached(&workspace.root);

    let manifest = workspace.root.join("crates/alpha/Cargo.toml");
    let contents = fs::read_to_string(&manifest).unwrap();
    fs::write(
        &manifest,
        format!("{contents}\n[[bin]]\nname = \"stray\"\npath = \"src/stray.rs\"\n"),
    )
    .unwrap();
    settle(&workspace.outer);

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

    let cache = workspace.root.join("target").join("cargo-shear");
    let dir = fs::read_dir(&cache).unwrap().next().unwrap().unwrap().path();
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
