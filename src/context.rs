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
use cargo_metadata::{Metadata, Node, Package, PackageId, Target};
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
    sorted_paths: Vec<PathBuf>,
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
            sorted_paths: Self::sort_paths(&files),
            files,
            linked,
            packages,
            dep_to_pkg,
            ignored_deps,
            hakari_packages,
        })
    }

    fn sort_paths(files: &FxHashMap<PathBuf, ParsedSource>) -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = files.keys().cloned().collect();
        paths.sort_unstable_by(|a, b| {
            a.as_os_str().as_encoded_bytes().cmp(b.as_os_str().as_encoded_bytes())
        });
        paths
    }

    pub fn file_at(&self, path: &Path) -> Option<(&PathBuf, &ParsedSource)> {
        self.files.get_key_value(path).filter(|(key, _)| {
            key.as_os_str().as_encoded_bytes() == path.as_os_str().as_encoded_bytes()
        })
    }

    pub fn files_in<'s>(
        &'s self,
        directory: &Path,
    ) -> impl Iterator<Item = (&'s PathBuf, &'s ParsedSource)> + 's {
        let directory = directory.as_os_str().as_encoded_bytes();
        (*b"/\\").map(|separator| [directory, &[separator]].concat()).into_iter().flat_map(
            move |prefix| {
                let start = self.sorted_paths.partition_point(|path| {
                    path.as_os_str().as_encoded_bytes() < prefix.as_slice()
                });
                self.sorted_paths[start..]
                    .iter()
                    .take_while(move |path| {
                        path.as_os_str().as_encoded_bytes().starts_with(&prefix)
                    })
                    .filter_map(|path| self.files.get_key_value(path))
            },
        )
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

pub struct ResolveIndex<'a> {
    nodes: Option<FxHashMap<&'a PackageId, &'a Node>>,
    packages: FxHashMap<&'a PackageId, &'a Package>,
}

impl<'a> ResolveIndex<'a> {
    pub fn new(metadata: &'a Metadata) -> Self {
        let nodes = metadata.resolve.as_ref().map(|resolve| {
            let mut nodes = FxHashMap::default();
            for node in &resolve.nodes {
                nodes.entry(&node.id).or_insert(node);
            }
            nodes
        });

        let mut packages = FxHashMap::default();
        for package in &metadata.packages {
            packages.entry(&package.id).or_insert(package);
        }

        Self { nodes, packages }
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
        resolve_index: &ResolveIndex<'_>,
    ) -> Result<Self> {
        let manifest_path = package.manifest_path.as_std_path();
        let manifest_content = read_to_string(manifest_path)?;
        let manifest: Manifest = toml::from_str(&manifest_content)?;

        let resolved = resolve_index
            .nodes
            .as_ref()
            .ok_or_else(|| {
                anyhow!("`cargo_metadata::MetadataCommand::no_deps` should not be called.")
            })?
            .get(&package.id)
            .ok_or_else(|| anyhow!("Package not found: {}", package.name))?;

        let directory = manifest_path
            .parent()
            .ok_or_else(|| anyhow!("Package has no parent directory: {}", package.name))?
            .to_path_buf();

        let mut import_to_pkg = FxHashMap::default();
        let mut pkg_to_import = FxHashMap::default();

        for dep in &resolved.deps {
            if let Some(pkg) = resolve_index.packages.get(&dep.pkg) {
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
    use std::path::{Path, PathBuf};

    use cargo_metadata::MetadataCommand;

    use super::WorkspaceContext;
    use crate::source_parser::ParsedSource;

    fn sorted<'a>(paths: impl Iterator<Item = &'a PathBuf>) -> Vec<&'a PathBuf> {
        let mut paths: Vec<_> = paths.collect();
        paths.sort();
        paths
    }

    fn synthetic_workspace(paths: &[&str]) -> WorkspaceContext {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/clean");
        let metadata = MetadataCommand::new().current_dir(fixture).no_deps().exec().unwrap();
        let mut workspace = WorkspaceContext::new(&metadata).unwrap();
        workspace.files =
            paths.iter().map(|path| (PathBuf::from(path), ParsedSource::default())).collect();
        workspace.sorted_paths = WorkspaceContext::sort_paths(&workspace.files);
        workspace
    }

    const PATHS: &[&str] = &[
        "/w/a/x.rs",
        "/w/a\\y.rs",
        "/w/a/b/c.rs",
        "/w/a/b\\d.rs",
        "/w/ab/z.rs",
        "/w/a.rs",
        "/w/a-/q.rs",
        "/w/a0/q.rs",
        "/w/a",
        "/w/b/x.rs",
        "/w/build.rs",
        "/x.rs",
        "relative/x.rs",
        "relative\\y.rs",
    ];

    #[test]
    fn files_in_matches_a_full_scan() {
        let workspace = synthetic_workspace(PATHS);
        let directories =
            ["/w/a", "/w", "/w/a/b", "/w/ab", "/w/a-", "/", "", "/w/missing", "relative", "/w/a/"];

        for directory in directories {
            let directory = Path::new(directory);
            let bytes = directory.as_os_str().as_encoded_bytes();
            let expected: Vec<_> = sorted(workspace.files.keys().filter(|path| {
                let path = path.as_os_str().as_encoded_bytes();
                path.starts_with(bytes) && matches!(path.get(bytes.len()), Some(&(b'/' | b'\\')))
            }));
            let actual: Vec<_> = sorted(workspace.files_in(directory).map(|(path, _)| path));
            assert_eq!(actual, expected, "Mismatch for {}.", directory.display());
        }

        let found = sorted(workspace.files_in(Path::new("/w/a")).map(|(path, _)| path));
        assert_eq!(found.len(), 4, "Expected both separators and nested files.");
    }

    #[test]
    fn file_at_matches_exact_bytes_only() {
        let workspace = synthetic_workspace(PATHS);

        for path in PATHS {
            let (found, _) = workspace.file_at(Path::new(path)).unwrap();
            assert_eq!(found.as_os_str(), *path);
        }

        for path in ["/w/./build.rs", "/w//build.rs", "/w/build.rs/", "/w/missing.rs"] {
            let bytes = Path::new(path).as_os_str().as_encoded_bytes();
            let expected =
                workspace.files.keys().find(|key| key.as_os_str().as_encoded_bytes() == bytes);
            assert_eq!(
                workspace.file_at(Path::new(path)).map(|(key, _)| key),
                expected,
                "Mismatch for {path}."
            );
        }
    }
}
