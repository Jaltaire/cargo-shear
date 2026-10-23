//! Cache of what the source parser extracted from each file.
//!
//! [`ParsedSource`] is a pure function of a file's text, its path (which
//! anchors `mod`, `#[path]`, and `include!` resolution), and whether it is a
//! target's entry point. An entry is reused only when all three match exactly:
//! the path and entry-point flag are compared directly and the text by its
//! BLAKE3 digest. Every later step (the walk, reachability, and every per-
//! package and workspace diagnostic) still runs on the complete set of files,
//! so only the parsing itself is skipped.
//!
//! The file is rewritten with exactly the files seen in the latest run, which
//! prunes entries for deleted or renamed files.

use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    cache::{CacheDir, Digest, from_hex, to_hex},
    source_parser::ParsedSource,
};

const FILE_NAME: &str = "parse.json";

/// A parsed file together with the key it was parsed under.
#[derive(Debug, Clone)]
pub struct ParseEntry {
    entry_point: bool,
    digest: Digest,
    parsed: ParsedSource,
}

/// Previously parsed files, loaded from the cache directory.
#[derive(Debug, Default)]
pub struct ParseCache {
    entries: FxHashMap<PathBuf, ParseEntry>,
}

/// The outcome of parsing one file through the cache.
pub struct Parsed {
    pub source: ParsedSource,
    pub entry: ParseEntry,
    pub reused: bool,
}

impl ParseCache {
    /// Load the entries recorded for `executable`. Anything unreadable, or
    /// recorded by a different executable, loads as an empty cache.
    pub fn load(dir: &CacheDir, executable: &Digest) -> Self {
        let Some(bytes) = dir.read(FILE_NAME) else { return Self::default() };
        let Ok(stored) = serde_json::from_slice::<StoredCache>(&bytes) else {
            return Self::default();
        };
        if from_hex(&stored.executable).as_ref() != Some(executable) {
            return Self::default();
        }

        let entries = stored
            .files
            .into_iter()
            .filter_map(|file| {
                let digest = from_hex(&file.digest)?;
                let parsed = ParsedSource {
                    imports: file.imports.into_iter().map(Into::into).collect(),
                    paths: file.paths.into_iter().map(PathBuf::from).collect(),
                    is_empty: file.is_empty,
                    has_tests: file.has_tests,
                    has_doctests: file.has_doctests,
                };
                Some((file.path, ParseEntry { entry_point: file.entry_point, digest, parsed }))
            })
            .collect();

        Self { entries }
    }

    /// Parse `source`, the contents of `path`, exactly as
    /// [`ParsedSource::from_source`] would, reusing the recorded result when the
    /// file's path, entry-point flag, and content are unchanged.
    pub fn parse(&self, path: &Path, source: &str, is_entry_point: bool) -> Parsed {
        let digest = *blake3::hash(source.as_bytes()).as_bytes();

        if let Some(entry) = self.entries.get(path)
            && entry.entry_point == is_entry_point
            && entry.digest == digest
        {
            return Parsed { source: entry.parsed.clone(), entry: entry.clone(), reused: true };
        }

        let parsed = ParsedSource::from_source(source, path, is_entry_point);
        let entry = ParseEntry { entry_point: is_entry_point, digest, parsed: parsed.clone() };
        Parsed { source: parsed, entry, reused: false }
    }

    /// True when `entries` differs from what was loaded, so the file needs
    /// rewriting.
    pub fn is_stale<'a>(
        &self,
        entries: impl ExactSizeIterator<Item = (&'a PathBuf, &'a ParseEntry)>,
    ) -> bool {
        entries.len() != self.entries.len()
            || entries.into_iter().any(|(path, entry)| {
                self.entries.get(path).is_none_or(|previous| {
                    previous.entry_point != entry.entry_point || previous.digest != entry.digest
                })
            })
    }
}

/// Replace the cache file with `entries`. Entries whose paths are not valid
/// UTF-8 are left out and will be parsed again next time.
pub fn store<'a>(
    dir: &CacheDir,
    executable: &Digest,
    entries: impl Iterator<Item = (&'a PathBuf, &'a ParseEntry)>,
) -> Option<()> {
    let files = entries
        .filter_map(|(path, entry)| {
            let paths = entry
                .parsed
                .paths
                .iter()
                .map(|path| path.to_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()?;
            Some(StoredFile {
                path: PathBuf::from(path.to_str()?),
                entry_point: entry.entry_point,
                digest: to_hex(&entry.digest),
                imports: entry.parsed.imports.iter().map(ToString::to_string).collect(),
                paths,
                is_empty: entry.parsed.is_empty,
                has_tests: entry.parsed.has_tests,
                has_doctests: entry.parsed.has_doctests,
            })
        })
        .collect();

    let stored = StoredCache { executable: to_hex(executable), files };
    let bytes = serde_json::to_vec(&stored).ok()?;
    dir.write(FILE_NAME, &bytes).ok()
}

#[derive(Serialize, Deserialize)]
struct StoredCache {
    executable: String,
    files: Vec<StoredFile>,
}

#[derive(Serialize, Deserialize)]
struct StoredFile {
    path: PathBuf,
    entry_point: bool,
    digest: String,
    imports: Vec<String>,
    paths: Vec<String>,
    is_empty: bool,
    has_tests: bool,
    has_doctests: bool,
}
