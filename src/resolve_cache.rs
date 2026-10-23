//! Cache of the dependency-resolution facts cargo-shear reads from
//! `cargo metadata`, so that a run on an unchanged workspace only needs
//! `cargo metadata --no-deps`.
//!
//! cargo-shear reads one thing from the resolved graph: for each workspace
//! member, the import name and package name of each of its dependencies
//! ([`ResolvedDeps`]). Everything else it reads (members, their targets, and
//! their manifests) comes from `--no-deps`, which runs on every call and so is
//! always current.
//!
//! An entry is reused only when two layers of inputs are unchanged:
//!
//! 1. The primary key ([`ResolveKey`]), computed before running cargo:
//!    the executable, the cargo flags and working directory, every
//!    environment variable cargo reads configuration from, the cargo binary,
//!    every cargo configuration file cargo would load, the `--no-deps` view of
//!    every member, the exact bytes of `Cargo.lock`, the root manifest, every
//!    member manifest, and every manifest named by a `[patch]`, `[replace]`, or
//!    `paths` override.
//! 2. The watched files ([`Watched`]), known only after a full resolution:
//!    the manifest of every package outside the workspace, the
//!    `Cargo.toml` files above path and git packages that cargo searches for a
//!    workspace root to inherit from, the directories cargo infers a path
//!    package's targets from, whether each other package has a `src/lib.rs`,
//!    the `.cargo-ok` markers cargo checks before trusting an unpacked
//!    registry package or git checkout, and the registry index cache entry
//!    cargo resolves each registry package from.
//!
//! An entry is written only when the primary key computed after the full
//! `cargo metadata` run equals the one computed before it (so cargo neither
//! rewrote `Cargo.lock` nor saw different members), when no input changed
//! within [`RECENT_CHANGE_WINDOW`](crate::cache::RECENT_CHANGE_WINDOW) of the
//! run's start (so what was recorded is what cargo read), and when the full
//! run printed exactly what the `--no-deps` run printed (so replaying the
//! latter on a hit prints what cargo would have).

use std::{
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Result, anyhow};
use cargo_metadata::{Metadata, Package, PackageId};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::cache::{
    CacheDir, Digest, Fingerprint, Newest, fingerprint_contents, from_hex, listing_digest,
    presence_digest, stat_digest, to_hex,
};

const FILE_NAME: &str = "resolve.json";

/// One dependency edge of a workspace member, as `cargo metadata` resolved it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDep {
    /// Name the dependency is imported under (reflects renames and `[lib] name`).
    pub import: String,
    /// Name of the dependency's package.
    pub package: String,
}

/// For each workspace member, its resolved dependency edges in the order
/// `cargo metadata` lists them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDeps {
    resolved: bool,
    members: FxHashMap<PackageId, Vec<ResolvedDep>>,
}

impl ResolvedDeps {
    /// Extract every workspace member's dependency edges from a full
    /// `cargo metadata` result.
    pub fn from_metadata(metadata: &Metadata) -> Self {
        let Some(resolve) = metadata.resolve.as_ref() else {
            return Self { resolved: false, members: FxHashMap::default() };
        };

        let mut packages: FxHashMap<&PackageId, &Package> = FxHashMap::default();
        for package in &metadata.packages {
            packages.entry(&package.id).or_insert(package);
        }

        let mut nodes = FxHashMap::default();
        for node in &resolve.nodes {
            nodes.entry(&node.id).or_insert(node);
        }

        let members = metadata
            .workspace_packages()
            .into_iter()
            .filter_map(|member| {
                let node = nodes.get(&member.id)?;
                let deps = node
                    .deps
                    .iter()
                    .filter_map(|dep| {
                        let package = packages.get(&dep.pkg)?;
                        // Artifact/bindep dependencies have an empty `dep.name` in `cargo_metadata`.
                        // Fall back to the package name so the rest of the pipeline can match them.
                        let import = if dep.name.is_empty() {
                            package.name.replace('-', "_")
                        } else {
                            dep.name.clone()
                        };
                        Some(ResolvedDep { import, package: package.name.to_string() })
                    })
                    .collect();
                Some((member.id.clone(), deps))
            })
            .collect();

        Self { resolved: true, members }
    }

    /// The dependency edges of one workspace member.
    pub fn deps(&self, package: &Package) -> Result<&[ResolvedDep]> {
        if !self.resolved {
            return Err(anyhow!(
                "`cargo_metadata::MetadataCommand::no_deps` should not be called."
            ));
        }

        self.members
            .get(&package.id)
            .map(Vec::as_slice)
            .ok_or_else(|| anyhow!("Package not found: {}", package.name))
    }
}

/// Inputs to the primary key that do not come from the file system.
#[derive(Debug, Clone)]
pub struct KeyContext {
    /// Digest of the running executable.
    pub executable: Digest,
    /// `--locked`, `--offline`, and `--frozen`, in that order.
    pub flags: [bool; 3],
    /// The directory `cargo metadata` runs in.
    pub cwd: PathBuf,
    /// Environment variables cargo reads, sorted by name.
    pub env: Vec<(OsString, OsString)>,
    /// Identity of the cargo binary that `cargo metadata` runs.
    pub cargo: Vec<u8>,
}

impl KeyContext {
    /// Capture the context of this process. `None` when the run must not be
    /// cached: the platform has no status-change time to tell whether a file
    /// was written during the run (see [`Newest`]), the executable cannot be
    /// read, the environment enables an
    /// unstable cargo feature that might make cargo read inputs this cache
    /// does not know about, or the cargo binary cannot be identified (see
    /// [`cargo_identity`]).
    pub fn capture(flags: [bool; 3], cwd: &Path) -> Option<Self> {
        if !cfg!(unix) {
            return None;
        }
        let executable = crate::cache::executable_digest()?;
        let cwd = cwd.canonicalize().ok()?;
        let env = relevant_env(env::vars_os());
        if env.iter().any(|(name, _)| unstable_env(name)) {
            return None;
        }

        let cargo = cargo_identity(&env, &cwd)?;
        Some(Self { executable, flags, cwd, env, cargo })
    }
}

/// The environment variables that can change what `cargo metadata` reports:
/// cargo's own configuration (`CARGO_*`), rustup's toolchain selection and
/// compiler settings (`RUST*`), cargo internals (`__CARGO_*`), and the
/// variables that locate the cargo binary and the cargo home directory.
pub fn relevant_env(vars: impl Iterator<Item = (OsString, OsString)>) -> Vec<(OsString, OsString)> {
    let mut env: Vec<_> = vars
        .filter(|(name, _)| {
            let name = name.as_encoded_bytes();
            name.starts_with(b"CARGO")
                || name.starts_with(b"RUST")
                || name.starts_with(b"__CARGO")
                || name == b"PATH"
                || name == b"HOME"
                || name == b"USERPROFILE"
        })
        .collect();
    env.sort();
    env
}

fn unstable_env(name: &OsString) -> bool {
    let name = name.as_encoded_bytes();
    name.starts_with(b"CARGO_UNSTABLE_") && name != b"CARGO_UNSTABLE_BINDEPS"
}

/// Identify the cargo binary `cargo metadata` will run: the one named by
/// `CARGO`, which cargo sets to its own path when it runs `cargo shear`. Its
/// path and file metadata identify it, since installing a toolchain replaces
/// the file.
///
/// `None` (do not cache) when `CARGO` is unset, so that `cargo` would be found
/// on `PATH`, or when it names a rustup proxy or a script. Those choose which
/// cargo to run each time they are called (from `rust-toolchain` files, rustup
/// settings, and the toolchains installed), and the file itself does not
/// change when that choice does.
pub fn cargo_identity(env: &[(OsString, OsString)], cwd: &Path) -> Option<Vec<u8>> {
    let cargo = env.iter().find(|(name, _)| name == "CARGO").map(|(_, value)| value)?;
    let path = cwd.join(cargo).canonicalize().ok()?;
    if is_rustup(&path, env, cwd) || is_script(&path) {
        return None;
    }

    let mut newest = Newest::default();
    let digest = stat_digest(&path, &mut newest);
    let mut identity = b"binary ".to_vec();
    identity.extend_from_slice(path.as_os_str().as_encoded_bytes());
    identity.extend_from_slice(&digest);
    Some(identity)
}

/// True when `path` (canonical) is rustup itself, which is what every rustup
/// proxy (`cargo`, `rustc`, ...) is: a link to it or a copy of it. Compared
/// against the `rustup` in the cargo home directory and every `rustup` on
/// `PATH`, by name and then by contents.
fn is_rustup(path: &Path, env: &[(OsString, OsString)], cwd: &Path) -> bool {
    let rustup = format!("rustup{}", env::consts::EXE_SUFFIX);
    if path.file_name().is_some_and(|name| name == rustup.as_str()) {
        return true;
    }

    let mut candidates: Vec<PathBuf> =
        cargo_home(env, cwd).map(|home| home.join("bin").join(&rustup)).into_iter().collect();
    if let Some((_, paths)) = env.iter().find(|(name, _)| name == "PATH") {
        candidates.extend(env::split_paths(paths).map(|directory| directory.join(&rustup)));
    }

    let Ok(metadata) = fs::metadata(path) else { return true };
    candidates.iter().any(|candidate| {
        let Ok(candidate) = candidate.canonicalize() else { return false };
        if candidate == path {
            return true;
        }
        let Ok(candidate_metadata) = fs::metadata(&candidate) else { return false };
        candidate_metadata.len() == metadata.len()
            && match (fs::read(&candidate), fs::read(path)) {
                (Ok(left), Ok(right)) => left == right,
                _ => true,
            }
    })
}

/// True when `path` is an interpreted script rather than a binary.
fn is_script(path: &Path) -> bool {
    let mut start = [0u8; 2];
    match fs::File::open(path).and_then(|mut file| {
        use std::io::Read;
        file.read_exact(&mut start)
    }) {
        Ok(()) => start == *b"#!",
        Err(_) => true,
    }
}

/// The cargo home directory cargo uses: `CARGO_HOME` (relative to `cwd`) when
/// set and not empty, otherwise `.cargo` in the home directory.
fn cargo_home(env: &[(OsString, OsString)], cwd: &Path) -> Option<PathBuf> {
    env.iter()
        .find(|(name, value)| name == "CARGO_HOME" && !value.is_empty())
        .map(|(_, value)| cwd.join(value))
        .or_else(|| env::home_dir().map(|home| home.join(".cargo")))
}

/// What cargo printed, without the status lines it prints while it waits for
/// another cargo process to release a lock, which depend on what else is
/// running rather than on anything it read.
pub fn without_lock_waits(printed: &[u8]) -> Vec<u8> {
    printed
        .split_inclusive(|byte| *byte == b'\n')
        .filter(|line| !line.trim_ascii_start().starts_with(b"Blocking waiting for file lock on "))
        .flatten()
        .copied()
        .collect()
}

/// The primary key of a cache entry, with the newest modification time among
/// the files it was computed from.
#[derive(Debug, Clone, Copy)]
pub struct ResolveKey {
    pub digest: Digest,
    pub newest: Newest,
}

impl ResolveKey {
    /// Compute the primary key for `metadata`, which may come from
    /// `--no-deps` (before deciding whether to run a full resolution) or from
    /// a full resolution (to check that nothing changed in between). Only the
    /// parts both share are read. `None` when the configuration uses a cargo
    /// feature that might read inputs this cache does not know about:
    /// `include`, unstable features other than `bindeps`, `paths` overrides
    /// (cargo searches every directory below them for packages), and source
    /// replacement (vendored and mirrored sources, which cargo verifies
    /// against checksums of every file).
    pub fn compute(context: &KeyContext, metadata: &Metadata) -> Option<Self> {
        let mut fingerprint = Fingerprint::new("cargo-shear resolve key v1");
        let mut newest = Newest::default();

        fingerprint.digest(&context.executable);
        for flag in context.flags {
            fingerprint.flag(flag);
        }
        fingerprint.path(&context.cwd);
        fingerprint.bytes(&(context.env.len() as u64).to_le_bytes());
        for (name, value) in &context.env {
            fingerprint.os_str(name).os_str(value);
        }
        fingerprint.bytes(&context.cargo);

        let mut patch_paths = Vec::new();
        for config in config_files(context) {
            fingerprint_contents(&mut fingerprint, &mut newest, &config);
            if let Some(table) = read_toml(&config) {
                if ["include", "paths", "source"].iter().any(|key| table.contains_key(*key)) {
                    return None;
                }
                if let Some(unstable) = table.get("unstable").and_then(toml::Value::as_table)
                    && unstable.keys().any(|key| key != "bindeps")
                {
                    return None;
                }
                let base = config.parent().and_then(Path::parent).unwrap_or_else(|| Path::new(""));
                collect_override_paths(&table, base, &mut patch_paths);
            }
        }

        let root = metadata.workspace_root.as_std_path();
        fingerprint.path(root);

        fingerprint.bytes(&(metadata.workspace_members.len() as u64).to_le_bytes());
        for member in &metadata.workspace_members {
            fingerprint.bytes(member.repr.as_bytes());
        }

        let mut members = metadata.workspace_packages();
        members.sort_by(|left, right| left.id.cmp(&right.id));
        fingerprint.bytes(&(members.len() as u64).to_le_bytes());
        for member in &members {
            let Ok(json) = serde_json::to_vec(member) else { return None };
            fingerprint.bytes(&json);
        }

        fingerprint_contents(&mut fingerprint, &mut newest, &root.join("Cargo.lock"));

        let root_manifest = root.join("Cargo.toml");
        fingerprint_contents(&mut fingerprint, &mut newest, &root_manifest);
        if let Some(table) = read_toml(&root_manifest) {
            collect_override_paths(&table, root, &mut patch_paths);
        }

        for member in &members {
            fingerprint_contents(&mut fingerprint, &mut newest, member.manifest_path.as_std_path());
        }

        for path in &patch_paths {
            for directory in path.ancestors() {
                fingerprint_contents(&mut fingerprint, &mut newest, &directory.join("Cargo.toml"));
            }
        }

        Some(Self { digest: fingerprint.finish(), newest })
    }
}

/// The cargo configuration files cargo loads when run in `context.cwd`:
/// `.cargo/config` and `.cargo/config.toml` in that directory and every
/// ancestor, then the same two files in the cargo home directory.
fn config_files(context: &KeyContext) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for directory in context.cwd.ancestors() {
        files.push(directory.join(".cargo").join("config"));
        files.push(directory.join(".cargo").join("config.toml"));
    }

    if let Some(cargo_home) = cargo_home(&context.env, &context.cwd) {
        files.push(cargo_home.join("config"));
        files.push(cargo_home.join("config.toml"));
    }

    files
}

fn read_toml(path: &Path) -> Option<toml::Table> {
    let contents = fs::read_to_string(path).ok()?;
    contents.parse::<toml::Table>().ok()
}

/// Collect the directories named by `[patch.<source>.<name>] path`,
/// `[replace.<spec>] path`, and `paths = [...]`, resolved against `base`.
/// A patch whose version does not match goes unused and appears nowhere else,
/// yet editing its manifest, or the workspace manifest above it that it
/// inherits its version from, can bring it into the graph.
fn collect_override_paths(table: &toml::Table, base: &Path, paths: &mut Vec<PathBuf>) {
    let mut push = |value: &toml::Value| {
        if let Some(path) = value.get("path").and_then(toml::Value::as_str) {
            paths.push(base.join(path));
        }
    };

    if let Some(patch) = table.get("patch").and_then(toml::Value::as_table) {
        for source in patch.values().filter_map(toml::Value::as_table) {
            source.values().for_each(&mut push);
        }
    }

    if let Some(replace) = table.get("replace").and_then(toml::Value::as_table) {
        replace.values().for_each(&mut push);
    }

    if let Some(overrides) = table.get("paths").and_then(toml::Value::as_array) {
        for path in overrides.iter().filter_map(toml::Value::as_str) {
            paths.push(base.join(path));
        }
    }
}

/// A file-system object whose state was recorded with a cache entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "path", rename_all = "kebab-case")]
enum WatchedPath {
    /// A file, compared by [`stat_digest`].
    File(PathBuf),
    /// A directory, compared by [`listing_digest`].
    Listing(PathBuf),
    /// A file whose existence matters but whose contents do not, compared by
    /// [`presence_digest`].
    Presence(PathBuf),
}

impl WatchedPath {
    fn digest(&self, newest: &mut Newest) -> Digest {
        match self {
            Self::File(path) => stat_digest(path, newest),
            Self::Listing(path) => listing_digest(path, newest),
            Self::Presence(path) => presence_digest(path, newest),
        }
    }
}

/// The watched files of a cache entry and their recorded digests.
#[derive(Debug, Clone)]
pub struct Watched {
    entries: Vec<(WatchedPath, Digest)>,
    newest: Newest,
}

impl Watched {
    /// Gather everything outside the workspace that the full resolution read.
    /// `None` when a registry package's index cache entry is not where cargo
    /// keeps it, so it cannot be watched.
    pub fn collect(metadata: &Metadata) -> Option<Self> {
        let members: FxHashSet<&PackageId> = metadata.workspace_members.iter().collect();
        let mut paths: FxHashSet<WatchedPath> = FxHashSet::default();

        for package in metadata.packages.iter().filter(|package| !members.contains(&package.id)) {
            let manifest = package.manifest_path.as_std_path();
            paths.insert(WatchedPath::File(manifest.to_path_buf()));

            let is_path = package.source.is_none();
            let is_git =
                package.source.as_ref().is_some_and(|source| source.repr.starts_with("git+"));
            let is_registry = package.source.as_ref().is_some_and(|source| {
                source.repr.starts_with("registry+") || source.repr.starts_with("sparse+")
            });
            let directory = manifest.parent()?;

            if is_path || is_git {
                for ancestor in directory.ancestors().skip(1) {
                    paths.insert(WatchedPath::File(ancestor.join("Cargo.toml")));
                }
            }

            if is_git {
                for ancestor in directory.ancestors() {
                    paths.insert(WatchedPath::File(ancestor.join(".cargo-ok")));
                }
            }

            if !is_path {
                paths.insert(WatchedPath::Presence(directory.join("src").join("lib.rs")));
            }

            if is_registry {
                paths.insert(WatchedPath::File(directory.join(".cargo-ok")));
                let entry = index_cache_entry(directory, package.name.as_str())?;
                if !entry.is_file() {
                    return None;
                }
                paths.insert(WatchedPath::File(entry));
            }

            if is_path {
                paths.insert(WatchedPath::Listing(directory.to_path_buf()));
                paths.insert(WatchedPath::Listing(directory.join("src")));
                for inferred in [
                    directory.join("src").join("bin"),
                    directory.join("examples"),
                    directory.join("tests"),
                    directory.join("benches"),
                ] {
                    if let Ok(entries) = fs::read_dir(&inferred) {
                        for entry in entries.flatten() {
                            if entry.path().is_dir() {
                                paths.insert(WatchedPath::Listing(entry.path()));
                            }
                        }
                    }
                    paths.insert(WatchedPath::Listing(inferred));
                }
                for target in &package.targets {
                    paths
                        .insert(WatchedPath::Presence(target.src_path.as_std_path().to_path_buf()));
                }
            }
        }

        let mut entries: Vec<_> = paths
            .into_par_iter()
            .map(|path| {
                let mut newest = Newest::default();
                let digest = path.digest(&mut newest);
                (path, digest, newest)
            })
            .collect();
        entries.sort_by(|(left, ..), (right, ..)| left.cmp_key().cmp(&right.cmp_key()));

        let mut newest = Newest::default();
        let entries = entries
            .into_iter()
            .map(|(path, digest, observed)| {
                newest.merge(observed);
                (path, digest)
            })
            .collect();

        Some(Self { entries, newest })
    }

    /// True when every watched object still has its recorded digest.
    pub fn unchanged(&self) -> bool {
        self.entries.par_iter().all(|(path, digest)| path.digest(&mut Newest::default()) == *digest)
    }
}

/// Where cargo caches the index entry of the registry package unpacked in
/// `directory` (`<registry>/src/<index>/<name>-<version>`):
/// `<registry>/index/<index>/.cache/<prefix>/<name>`, with the name in lower
/// case and the prefix cargo derives from it.
pub fn index_cache_entry(directory: &Path, name: &str) -> Option<PathBuf> {
    let index = directory.parent()?;
    let registry = index.parent()?.parent()?;
    let name = name.to_lowercase();
    let prefix = match name.len() {
        0 => return None,
        1 => PathBuf::from("1"),
        2 => PathBuf::from("2"),
        3 => Path::new("3").join(name.get(..1)?),
        _ => Path::new(name.get(..2)?).join(name.get(2..4)?),
    };
    Some(registry.join("index").join(index.file_name()?).join(".cache").join(prefix).join(name))
}

impl WatchedPath {
    fn cmp_key(&self) -> (u8, &Path) {
        match self {
            Self::File(path) => (0, path),
            Self::Listing(path) => (1, path),
            Self::Presence(path) => (2, path),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct StoredWatch {
    #[serde(flatten)]
    path: WatchedPath,
    digest: String,
}

#[derive(Serialize, Deserialize)]
struct StoredMember {
    id: String,
    deps: Vec<ResolvedDep>,
}

#[derive(Serialize, Deserialize)]
struct StoredEntry {
    key: String,
    watched: Vec<StoredWatch>,
    members: Vec<StoredMember>,
}

/// Look up the cached resolution for the workspace `metadata` (from
/// `--no-deps`) describes. `None` on any mismatch.
///
/// On a hit, `metadata.packages` is also put into the order the full
/// resolution listed the members in, which differs from `--no-deps` and
/// decides the order packages are reported in.
pub fn load(dir: &CacheDir, key: &ResolveKey, metadata: &mut Metadata) -> Option<ResolvedDeps> {
    let bytes = dir.read(FILE_NAME)?;
    let stored: StoredEntry = serde_json::from_slice(&bytes).ok()?;
    if from_hex(&stored.key)? != key.digest {
        return None;
    }

    let entries = stored
        .watched
        .into_iter()
        .map(|watch| Some((watch.path, from_hex(&watch.digest)?)))
        .collect::<Option<Vec<_>>>()?;
    let watched = Watched { entries, newest: Newest::default() };
    if !watched.unchanged() {
        return None;
    }

    let order: FxHashMap<String, usize> = stored
        .members
        .iter()
        .enumerate()
        .map(|(position, member)| (member.id.clone(), position))
        .collect();
    let positions = metadata
        .workspace_packages()
        .into_iter()
        .map(|package| order.get(&package.id.repr).copied())
        .collect::<Option<Vec<_>>>()?;
    if positions.len() != stored.members.len() {
        return None;
    }

    let mut members: FxHashMap<String, Vec<ResolvedDep>> = FxHashMap::default();
    for member in stored.members {
        members.insert(member.id, member.deps);
    }
    let members = metadata
        .workspace_packages()
        .into_iter()
        .map(|package| Some((package.id.clone(), members.remove(&package.id.repr)?)))
        .collect::<Option<FxHashMap<_, _>>>()?;

    let member_ids: FxHashSet<PackageId> = metadata.workspace_members.iter().cloned().collect();
    let (mut ordered, others): (Vec<_>, Vec<_>) = std::mem::take(&mut metadata.packages)
        .into_iter()
        .partition(|package| member_ids.contains(&package.id));
    ordered.sort_by_key(|package| order.get(&package.id.repr).copied());
    metadata.packages = ordered.into_iter().chain(others).collect();

    Some(ResolvedDeps { resolved: true, members })
}

/// Record `resolved`, which a full `cargo metadata` run produced while the
/// inputs had primary key `key`.
pub fn store(
    dir: &CacheDir,
    key: &ResolveKey,
    watched: &Watched,
    resolved: &ResolvedDeps,
    metadata: &Metadata,
) -> Option<()> {
    if !resolved.resolved {
        return None;
    }

    let members = metadata
        .workspace_packages()
        .into_iter()
        .map(|package| {
            Some(StoredMember {
                id: package.id.repr.clone(),
                deps: resolved.members.get(&package.id)?.clone(),
            })
        })
        .collect::<Option<Vec<_>>>()?;

    let watched = watched
        .entries
        .iter()
        .map(|(path, digest)| StoredWatch { path: path.clone(), digest: to_hex(digest) })
        .collect();

    let entry = StoredEntry { key: to_hex(&key.digest), watched, members };
    let bytes = serde_json::to_vec(&entry).ok()?;
    dir.write(FILE_NAME, &bytes).ok()
}

/// The newest modification time among everything an entry would record.
pub fn newest_input(key: &ResolveKey, watched: &Watched) -> Newest {
    let mut newest = key.newest;
    newest.merge(watched.newest);
    newest
}
