//! Pre-computed inputs shared across the per-package and workspace passes:
//! the parsed source files, which ones are reachable from a module tree, and
//! the dependency-name maps the analyzer needs to translate import names back
//! to package names.

use std::{
    cmp::Reverse,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

use anyhow::{Result, anyhow};
use cargo_metadata::{Metadata, Package, Target};
use ignore::{DirEntry, WalkBuilder};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{manifest::Manifest, source_parser::ParsedSource, util::read_to_string};

/// Marker that `cargo hakari` writes into a `workspace-hack` crate's `Cargo.toml`
/// to delimit its generated dependency section. Its presence uniquely identifies
/// such a crate, regardless of the name the user gave it. See
/// <https://docs.rs/cargo-hakari>.
const HAKARI_SECTION_MARKER: &str = "### BEGIN HAKARI SECTION";

/// Workspace-wide state computed once and shared by every per-package run.
pub struct WorkspaceContext {
    /// Absolute path to the workspace root directory.
    pub root: PathBuf,

    /// Absolute path to the workspace `Cargo.toml`.
    pub manifest_path: PathBuf,
    /// Raw `Cargo.toml` content (kept so renderers can show source spans).
    pub manifest_content: String,
    /// Parsed workspace manifest.
    pub manifest: Manifest,

    /// All Rust source files in the workspace, keyed by absolute path.
    pub files: FxHashMap<PathBuf, ParsedSource>,
    /// Files reachable from at least one entry point's module tree.
    pub linked: FxHashSet<PathBuf>,
    /// Absolute directories of every workspace member.
    pub packages: FxHashSet<PathBuf>,

    /// `[workspace.dependencies]` key → underlying package name (handles `package = "..."`).
    pub dep_to_pkg: FxHashMap<String, String>,
    /// `[workspace.metadata.cargo-shear].ignored` entries (raw dependency keys).
    pub ignored_deps: FxHashSet<String>,

    /// Names of workspace members that are `cargo hakari` `workspace-hack` crates,
    /// detected by the `### BEGIN HAKARI SECTION` marker in their `Cargo.toml`.
    /// They exist solely to unify Cargo features: they declare many dependencies
    /// they never import, and every member depends on them without importing them,
    /// so they're exempt from unused-dependency analysis entirely.
    pub hakari_packages: FxHashSet<String>,
}

impl WorkspaceContext {
    pub fn new(metadata: &Metadata) -> Result<Self> {
        let root = metadata.workspace_root.as_std_path().to_path_buf();

        let manifest_path = root.join("Cargo.toml");
        let manifest_content = read_to_string(&manifest_path)?;
        let manifest: Manifest = toml::from_str(&manifest_content)?;

        let packages: FxHashSet<PathBuf> = metadata
            .workspace_packages()
            .iter()
            .filter_map(|pkg| pkg.manifest_path.parent())
            .map(|path| path.as_std_path().to_path_buf())
            .collect();

        let entry_points: FxHashSet<PathBuf> = metadata
            .workspace_packages()
            .iter()
            .flat_map(|pkg| pkg.targets.iter())
            .map(|target| target.src_path.as_std_path().to_path_buf())
            .collect();

        // Walk from each entry point's parent directory, not from the package root —
        // some packages put `lib.rs` outside `src/`, and we want to follow the actual layout.
        let parents: FxHashSet<PathBuf> =
            entry_points.iter().filter_map(|path| path.parent()).map(Path::to_path_buf).collect();

        let walked: FxHashSet<PathBuf> = parents
            .into_par_iter()
            .flat_map_iter(|parent| {
                let packages = packages.clone();
                WalkBuilder::new(&parent)
                    // Don't descend into directories that are themselves Cargo packages
                    // (each member is walked from its own entry points). We *do* still
                    // visit the package root itself — packages.contains(path) lets the
                    // current member's root through.
                    .filter_entry(move |entry| {
                        if let Some(file_type) = entry.file_type()
                            && file_type.is_dir()
                        {
                            let path = entry.path();
                            if path.join("Cargo.toml").exists() {
                                return packages.contains(path);
                            }
                        }

                        true
                    })
                    .build()
                    .filter_map(Result::ok)
                    // Only `.rs` files; the walker also yields directories and other extensions.
                    .filter(|entry| {
                        entry.file_type().is_some_and(|file_type| file_type.is_file())
                            && entry.path().extension().is_some_and(|extension| extension == "rs")
                    })
                    .map(DirEntry::into_path)
            })
            .collect();

        // `WalkBuilder` was started from parent directories, which misses single-file
        // entry points whose parent isn't itself walked (e.g. `build.rs` at the package root).
        let paths: FxHashSet<PathBuf> =
            entry_points.iter().filter(|path| path.is_file()).cloned().chain(walked).collect();

        let files = Self::parse_files(paths, &entry_points);

        let linked: FxHashSet<PathBuf> = entry_points
            .into_iter()
            .chain(files.values().flat_map(|parsed| {
                parsed.paths.iter().filter_map(|path| {
                    // Only canonicalize when the path is relative (`./`, `../`); otherwise
                    // skip the syscall — the file may not exist on disk yet.
                    if path.as_os_str().as_encoded_bytes().starts_with(b".") {
                        path.canonicalize().ok()
                    } else {
                        Some(path.clone())
                    }
                })
            }))
            .collect();

        let dep_to_pkg = manifest
            .workspace
            .dependencies
            .iter()
            .map(|(dep, dependency)| {
                let dep = dep.get_ref();
                let pkg = dependency.get_ref().package().unwrap_or(dep);
                (dep.to_owned(), pkg.to_owned())
            })
            .collect();

        let ignored_deps = manifest
            .workspace
            .metadata
            .cargo_shear
            .ignored
            .iter()
            .map(|ignore| ignore.get_ref().clone())
            .collect();

        let hakari_packages = Self::detect_hakari_packages(metadata);

        Ok(Self {
            root,
            manifest_path,
            manifest_content,
            manifest,
            files,
            linked,
            packages,
            dep_to_pkg,
            ignored_deps,
            hakari_packages,
        })
    }

    fn parse_files(
        paths: FxHashSet<PathBuf>,
        entry_points: &FxHashSet<PathBuf>,
    ) -> FxHashMap<PathBuf, ParsedSource> {
        let mut sources: Vec<(PathBuf, String)> = paths
            .into_par_iter()
            .filter_map(|path| read_to_string(&path).ok().map(|source| (path, source)))
            .collect();
        sources.sort_unstable_by_key(|(_, source)| Reverse(source.len()));

        let next = AtomicUsize::new(0);
        rayon::broadcast(|_| {
            let mut parsed = Vec::new();
            while let Some((path, source)) = sources.get(next.fetch_add(1, Ordering::Relaxed)) {
                let is_entry_point = entry_points.contains(path);
                parsed
                    .push((path.clone(), ParsedSource::from_source(source, path, is_entry_point)));
            }
            parsed
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// Identify the workspace members that are `cargo hakari` `workspace-hack`
    /// crates by scanning each member's `Cargo.toml` for the hakari section
    /// marker, returning their package names. The manifests are read in parallel,
    /// like every other bulk pass in [`Self::new`].
    fn detect_hakari_packages(metadata: &Metadata) -> FxHashSet<String> {
        metadata
            .workspace_packages()
            .into_par_iter()
            .filter(|pkg| {
                read_to_string(pkg.manifest_path.as_std_path())
                    .is_ok_and(|content| content.contains(HAKARI_SECTION_MARKER))
            })
            .map(|pkg| pkg.name.to_string())
            .collect()
    }
}

/// Per-package state derived from `cargo metadata` plus the package's own manifest.
pub struct PackageContext<'a> {
    /// Shared workspace state (source file map, ignored deps, ...).
    pub workspace: &'a WorkspaceContext,

    /// Package name as declared in `[package].name`.
    pub name: String,
    /// Absolute path to the package's directory (parent of its `Cargo.toml`).
    pub directory: PathBuf,

    /// Absolute path to the package's `Cargo.toml`.
    pub manifest_path: PathBuf,
    /// Raw manifest content (kept so renderers can show source spans).
    pub manifest_content: String,
    /// Parsed package manifest.
    pub manifest: Manifest,
    /// Cargo targets declared by this package (lib, bin, test, ...).
    pub targets: Vec<Target>,

    /// Import name → package name (e.g. `tokio_util` → `tokio-util`).
    pub import_to_pkg: FxHashMap<String, String>,
    /// Package name → import name (inverse of `import_to_pkg`).
    pub pkg_to_import: FxHashMap<String, String>,

    /// Union of package- and workspace-level `ignored = [...]` entries, normalised to import names.
    pub ignored_imports: FxHashSet<String>,
}

impl<'a> PackageContext<'a> {
    pub fn new(
        workspace: &'a WorkspaceContext,
        package: &Package,
        metadata: &Metadata,
    ) -> Result<Self> {
        let manifest_path = package.manifest_path.as_std_path();
        let manifest_content = read_to_string(manifest_path)?;
        let manifest: Manifest = toml::from_str(&manifest_content)?;

        let resolved = metadata
            .resolve
            .as_ref()
            .ok_or_else(|| {
                anyhow!("`cargo_metadata::MetadataCommand::no_deps` should not be called.")
            })?
            .nodes
            .iter()
            .find(|node| node.id == package.id)
            .ok_or_else(|| anyhow!("Package not found: {}", package.name))?;

        let directory = manifest_path
            .parent()
            .ok_or_else(|| anyhow!("Package has no parent directory: {}", package.name))?
            .to_path_buf();

        let mut import_to_pkg = FxHashMap::default();
        let mut pkg_to_import = FxHashMap::default();

        for dep in &resolved.deps {
            if let Some(pkg) = metadata.packages.iter().find(|package| package.id == dep.pkg) {
                // Artifact/bindep dependencies have an empty `dep.name` in `cargo_metadata`.
                // Fall back to the package name so the rest of the pipeline can match them.
                let name =
                    if dep.name.is_empty() { pkg.name.replace('-', "_") } else { dep.name.clone() };
                import_to_pkg.insert(name.clone(), pkg.name.to_string());
                pkg_to_import.insert(pkg.name.to_string(), name);
            }
        }

        // A package ignore applies only to its own package; a workspace ignore applies
        // to every member. Both suppress unused-dependency diagnostics here, so merge them.
        let package_ignored_deps = &manifest.package.metadata.cargo_shear.ignored;
        let workspace_ignored_deps = &workspace.manifest.workspace.metadata.cargo_shear.ignored;
        let ignored_imports = package_ignored_deps
            .iter()
            .chain(workspace_ignored_deps)
            .map(|dep| dep.get_ref().replace('-', "_"))
            .collect();

        Ok(Self {
            workspace,
            name: package.name.to_string(),
            directory,
            manifest_path: manifest_path.to_path_buf(),
            manifest_content,
            manifest,
            targets: package.targets.clone(),
            import_to_pkg,
            pkg_to_import,
            ignored_imports,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use rustc_hash::{FxHashMap, FxHashSet};
    use tempfile::TempDir;

    use super::WorkspaceContext;
    use crate::source_parser::ParsedSource;

    fn parse_one_by_one(
        paths: &FxHashSet<PathBuf>,
        entry_points: &FxHashSet<PathBuf>,
    ) -> FxHashMap<PathBuf, ParsedSource> {
        paths
            .iter()
            .filter_map(|path| {
                let source = String::from_utf8(fs::read(path).ok()?).ok()?;
                let parsed = ParsedSource::from_source(&source, path, entry_points.contains(path));
                Some((path.clone(), parsed))
            })
            .collect()
    }

    fn write(paths: &mut FxHashSet<PathBuf>, root: &Path, name: &str, contents: &[u8]) -> PathBuf {
        let path = root.join(name);
        fs::write(&path, contents).unwrap();
        paths.insert(path.clone());
        path
    }

    #[test]
    fn parse_files_parses_every_readable_file_exactly_once() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let mut paths = FxHashSet::default();
        let lib = write(&mut paths, root, "lib.rs", b"mod a;\nuse serde::Serialize;\n");
        let main = write(&mut paths, root, "main.rs", b"mod a;\nfn main() {}\n");
        write(&mut paths, root, "a.rs", b"mod b;\nuse anyhow::Result;\n");
        write(&mut paths, root, "big.rs", "use regex::Regex;\n".repeat(5_000).as_bytes());
        write(&mut paths, root, "empty.rs", b"");
        write(&mut paths, root, "invalid.rs", &[0xff, 0xfe, b'\n']);
        paths.insert(root.join("missing.rs"));
        let entry_points: FxHashSet<PathBuf> = [lib, main].into_iter().collect();

        let expected = parse_one_by_one(&paths, &entry_points);
        assert_eq!(expected.len(), 5, "Every readable file should be parsed.");

        let actual = WorkspaceContext::parse_files(paths, &entry_points);
        assert_eq!(actual, expected, "Parsing by size should parse each file as before.");
    }

    #[test]
    fn parse_files_distinguishes_entry_points() {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        let mut paths = FxHashSet::default();
        let entry = write(&mut paths, root, "entry.rs", b"mod child;\n");
        let nested = write(&mut paths, root, "nested.rs", b"mod child;\n");
        let entry_points: FxHashSet<PathBuf> = std::iter::once(entry.clone()).collect();

        let actual = WorkspaceContext::parse_files(paths.clone(), &entry_points);
        assert_eq!(actual, parse_one_by_one(&paths, &entry_points));
        assert_ne!(
            actual[&entry].paths, actual[&nested].paths,
            "An entry point and a nested module should resolve `mod child;` differently."
        );
    }

    #[test]
    fn parse_files_of_nothing_is_empty() {
        let parsed = WorkspaceContext::parse_files(FxHashSet::default(), &FxHashSet::default());
        assert!(parsed.is_empty(), "No paths should parse to no files.");
    }
}
