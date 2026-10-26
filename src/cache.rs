//! Shared plumbing for the opt-in `--cache` mode: digests, the on-disk cache
//! directory, file and directory fingerprints, and the guard that refuses to
//! record inputs that changed while they were being read.
//!
//! Two caches build on it:
//!
//! - [`crate::resolve_cache`] remembers the part of `cargo metadata`'s
//!   dependency resolution that cargo-shear reads, so an unchanged workspace
//!   only pays for `cargo metadata --no-deps`.
//! - [`crate::parse_cache`] remembers what the source parser extracted from
//!   each file, keyed by the file's exact content.
//!
//! Every cache entry is bound to the digest of the running executable, so a
//! rebuilt cargo-shear never reads results produced by different code.

use std::{
    env,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process,
    sync::OnceLock,
    time::{Duration, SystemTime},
};

use cargo_util::paths;

/// A BLAKE3 digest.
pub type Digest = [u8; 32];

/// How far before a run's start a file must have last changed for the run to
/// record it. File systems store modification times at different
/// granularities (two seconds on FAT, one second on HFS+), so a change made
/// just after the run started can carry a time stamp from just before it.
pub const RECENT_CHANGE_WINDOW: Duration = Duration::from_secs(2);

/// Incrementally builds a digest from length-prefixed fields, so that no two
/// different sequences of fields can produce the same byte stream.
pub struct Fingerprint {
    hasher: blake3::Hasher,
}

impl Fingerprint {
    /// Start a fingerprint whose purpose is named by `domain`, so that digests
    /// built for different purposes never collide.
    pub fn new(domain: &str) -> Self {
        let mut fingerprint = Self { hasher: blake3::Hasher::new() };
        fingerprint.bytes(domain.as_bytes());
        fingerprint
    }

    /// Mix in one field.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        self.hasher.update(&(bytes.len() as u64).to_le_bytes());
        self.hasher.update(bytes);
        self
    }

    /// Mix in an operating-system string exactly as the platform encodes it.
    pub fn os_str(&mut self, value: &OsStr) -> &mut Self {
        self.bytes(value.as_encoded_bytes())
    }

    /// Mix in a path exactly as the platform encodes it.
    pub fn path(&mut self, path: &Path) -> &mut Self {
        self.os_str(path.as_os_str())
    }

    /// Mix in a flag.
    pub fn flag(&mut self, value: bool) -> &mut Self {
        self.bytes(&[u8::from(value)])
    }

    /// Mix in a digest.
    pub fn digest(&mut self, digest: &Digest) -> &mut Self {
        self.bytes(digest)
    }

    pub fn finish(&self) -> Digest {
        *self.hasher.finalize().as_bytes()
    }
}

/// Digest of the running executable, computed once per process. `None` when
/// the executable cannot be read, in which case nothing is cached.
pub fn executable_digest() -> Option<Digest> {
    static DIGEST: OnceLock<Option<Digest>> = OnceLock::new();
    *DIGEST.get_or_init(|| {
        let path = env::current_exe().ok()?;
        let bytes = fs::read(path).ok()?;
        Some(*blake3::hash(&bytes).as_bytes())
    })
}

/// Encode a digest as lowercase hexadecimal.
pub fn to_hex(digest: &Digest) -> String {
    use std::fmt::Write;

    digest.iter().fold(String::with_capacity(64), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    })
}

/// Decode a digest written by [`to_hex`].
pub fn from_hex(hex: &str) -> Option<Digest> {
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return None;
    }

    let mut digest = [0u8; 32];
    for (index, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let pair = std::str::from_utf8(pair).ok()?;
        *digest.get_mut(index)? = u8::from_str_radix(pair, 16).ok()?;
    }

    Some(digest)
}

/// The directory holding one workspace's cache files:
/// `<target-dir>/cargo-shear/<digest of the workspace root>/`.
///
/// Living under the target directory means `cargo clean` discards it, and the
/// per-root subdirectory keeps workspaces that share a target directory apart.
/// A run that comes before cargo's first build creates the target directory
/// as cargo does, so that it still carries `CACHEDIR.TAG` and is excluded from
/// backups: cargo marks only a target directory it creates itself.
#[derive(Debug, Clone)]
pub struct CacheDir {
    target_directory: PathBuf,
    path: PathBuf,
}

impl CacheDir {
    pub fn new(target_directory: &Path, workspace_root: &Path) -> Self {
        let root = Fingerprint::new("cargo-shear workspace root").path(workspace_root).finish();
        let name = to_hex(&root);
        let name = name.get(..32).unwrap_or(&name);
        Self {
            target_directory: target_directory.to_path_buf(),
            path: target_directory.join("cargo-shear").join(name),
        }
    }

    #[cfg(all(test, unix))]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read a cache file. A missing or unreadable file reads as `None`.
    pub fn read(&self, name: &str) -> Option<Vec<u8>> {
        fs::read(self.path.join(name)).ok()
    }

    /// Replace a cache file atomically: write a sibling temporary file, then
    /// rename it over the old one, so a concurrent reader sees either the old
    /// contents or the new, never a mixture. A missing target directory is
    /// first created as cargo creates it.
    pub fn write(&self, name: &str, contents: &[u8]) -> io::Result<()> {
        paths::create_dir_all_excluded_from_backups_atomic(&self.target_directory)
            .map_err(io::Error::other)?;
        fs::create_dir_all(&self.path)?;
        let temporary = self.path.join(format!(".{name}.{}.tmp", process::id()));
        fs::write(&temporary, contents)?;
        fs::rename(&temporary, self.path.join(name)).inspect_err(|_| {
            let _ = fs::remove_file(&temporary);
        })
    }
}

/// Tracks the newest modification time among the inputs a cache entry is
/// built from, so the entry can be refused if any of them changed during the
/// run (see [`RECENT_CHANGE_WINDOW`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct Newest {
    time: Option<SystemTime>,
    unknown: bool,
}

impl Newest {
    /// Record one input's metadata: its modification time and, where the
    /// platform has one, its status-change time, which every write updates
    /// and which, unlike the modification time, cannot be set back.
    pub fn observe(&mut self, metadata: &fs::Metadata) {
        match metadata.modified() {
            Ok(time) => self.observe_time(time),
            Err(_) => self.unknown = true,
        }
        if cfg!(unix) {
            match status_change_time(metadata) {
                Some(time) => self.observe_time(time),
                None => self.unknown = true,
            }
        }
    }

    /// Record one input's modification time.
    pub fn observe_time(&mut self, time: SystemTime) {
        self.time = Some(self.time.map_or(time, |newest| newest.max(time)));
    }

    /// Record an input whose modification time could not be read.
    pub const fn observe_unknown(&mut self) {
        self.unknown = true;
    }

    pub fn merge(&mut self, other: Self) {
        self.unknown |= other.unknown;
        if let Some(time) = other.time {
            self.observe_time(time);
        }
    }

    /// True when every observed input last changed comfortably before
    /// `started`, so the contents read during the run are the contents that
    /// were recorded.
    pub fn settled_before(&self, started: SystemTime) -> bool {
        if self.unknown {
            return false;
        }

        self.time.is_none_or(|time| {
            time.checked_add(RECENT_CHANGE_WINDOW).is_some_and(|settled| settled < started)
        })
    }
}

#[cfg(unix)]
fn status_change_time(metadata: &fs::Metadata) -> Option<SystemTime> {
    use std::os::unix::fs::MetadataExt;

    let seconds = u64::try_from(metadata.ctime()).ok()?;
    let nanoseconds = u32::try_from(metadata.ctime_nsec()).ok()?;
    SystemTime::UNIX_EPOCH.checked_add(Duration::new(seconds, nanoseconds))
}

#[cfg(not(unix))]
const fn status_change_time(_: &fs::Metadata) -> Option<SystemTime> {
    None
}

/// Mix the exact contents of `path` (or its absence) into `fingerprint`.
pub fn fingerprint_contents(fingerprint: &mut Fingerprint, newest: &mut Newest, path: &Path) {
    fingerprint.path(path);
    match fs::metadata(path) {
        Ok(metadata) => {
            newest.observe(&metadata);
            match fs::read(path) {
                Ok(contents) => fingerprint.bytes(b"contents").bytes(&contents),
                Err(err) => {
                    fingerprint.bytes(b"unreadable").bytes(err.kind().to_string().as_bytes())
                }
            };
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            fingerprint.bytes(b"absent");
        }
        Err(err) => {
            newest.observe_unknown();
            fingerprint.bytes(b"unreadable").bytes(err.kind().to_string().as_bytes());
        }
    }
}

/// Digest of a file's metadata: identity, size, and change times. Any write
/// to the file changes its status-change time, which cannot be set back from
/// user space, so an unchanged digest means an unchanged file.
pub fn stat_digest(path: &Path, newest: &mut Newest) -> Digest {
    let mut fingerprint = Fingerprint::new("cargo-shear stat");
    match fs::metadata(path) {
        Ok(metadata) => {
            newest.observe(&metadata);
            fingerprint.bytes(b"present");
            mix_metadata(&mut fingerprint, &metadata);
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            fingerprint.bytes(b"absent");
        }
        Err(err) => {
            newest.observe_unknown();
            fingerprint.bytes(b"unreadable").bytes(err.kind().to_string().as_bytes());
        }
    }
    fingerprint.finish()
}

/// Digest of a directory's entries: each entry's name and what it resolves to
/// (file, directory, or something else). Adding, removing, renaming, or
/// retyping an entry changes it. Cargo infers targets from exactly this.
pub fn listing_digest(path: &Path, newest: &mut Newest) -> Digest {
    let mut fingerprint = Fingerprint::new("cargo-shear listing");
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_dir() => {
            newest.observe(&metadata);
            match fs::read_dir(path) {
                Ok(entries) => {
                    let mut entries: Vec<_> = entries
                        .map(|entry| {
                            entry.map(|entry| {
                                let kind = match fs::metadata(entry.path()) {
                                    Ok(metadata) if metadata.is_file() => b'f',
                                    Ok(metadata) if metadata.is_dir() => b'd',
                                    Ok(_) => b'o',
                                    Err(_) => b'x',
                                };
                                (entry.file_name(), kind)
                            })
                        })
                        .collect::<io::Result<_>>()
                        .unwrap_or_else(|_| {
                            newest.observe_unknown();
                            Vec::new()
                        });
                    entries.sort();
                    fingerprint.bytes(b"directory");
                    for (name, kind) in &entries {
                        fingerprint.os_str(name).bytes(&[*kind]);
                    }
                }
                Err(err) => {
                    newest.observe_unknown();
                    fingerprint.bytes(b"unreadable").bytes(err.kind().to_string().as_bytes());
                }
            }
        }
        Ok(_) => {
            fingerprint.bytes(b"not a directory");
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            fingerprint.bytes(b"absent");
        }
        Err(err) => {
            newest.observe_unknown();
            fingerprint.bytes(b"unreadable").bytes(err.kind().to_string().as_bytes());
        }
    }
    fingerprint.finish()
}

/// Digest of whether `path` exists and what it is (file, directory, or
/// something else). The modification time of the directory holding it is
/// observed, since creating or removing an entry changes it.
pub fn presence_digest(path: &Path, newest: &mut Newest) -> Digest {
    if let Some(parent) = path.parent() {
        match fs::metadata(parent) {
            Ok(metadata) => newest.observe(&metadata),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(_) => newest.observe_unknown(),
        }
    }

    let mut fingerprint = Fingerprint::new("cargo-shear presence");
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => fingerprint.bytes(b"file"),
        Ok(metadata) if metadata.is_dir() => fingerprint.bytes(b"directory"),
        Ok(_) => fingerprint.bytes(b"other"),
        Err(err) if err.kind() == io::ErrorKind::NotFound => fingerprint.bytes(b"absent"),
        Err(err) => {
            newest.observe_unknown();
            fingerprint.bytes(b"unreadable").bytes(err.kind().to_string().as_bytes())
        }
    };
    fingerprint.finish()
}

#[cfg(unix)]
fn mix_metadata(fingerprint: &mut Fingerprint, metadata: &fs::Metadata) {
    use std::os::unix::fs::MetadataExt;

    for value in [
        metadata.dev(),
        metadata.ino(),
        u64::from(metadata.mode()),
        metadata.size(),
        metadata.mtime().cast_unsigned(),
        metadata.mtime_nsec().cast_unsigned(),
        metadata.ctime().cast_unsigned(),
        metadata.ctime_nsec().cast_unsigned(),
    ] {
        fingerprint.bytes(&value.to_le_bytes());
    }
}

#[cfg(not(unix))]
fn mix_metadata(fingerprint: &mut Fingerprint, metadata: &fs::Metadata) {
    fingerprint.bytes(&metadata.len().to_le_bytes());
    fingerprint.flag(metadata.is_dir());
    fingerprint.flag(metadata.permissions().readonly());
    for time in [metadata.modified(), metadata.created()] {
        match time.ok().and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok()) {
            Some(duration) => fingerprint.bytes(&duration.as_nanos().to_le_bytes()),
            None => fingerprint.bytes(b"no time"),
        };
    }
}
