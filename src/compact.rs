use alloc::{sync::Arc, vec::Vec};

use core::{
    borrow::Borrow,
    cmp,
    ffi::CStr,
    fmt,
    hash::{self, Hash},
    mem,
    ops::Deref,
};

use crate::{CStr8, CStr8Error, CString8};

/// Maximum number of bytes in the inline buffer.
const INLINE_BUF: usize = 22;

/// Maximum string length (excluding NUL) that fits inline.
const MAX_INLINE_LEN: usize = INLINE_BUF - 1;

/// Compact owned string which is guaranteed UTF-8 and nul-terminated.
///
/// Uses small-string optimization: strings of up to 21 bytes are stored
/// inline without heap allocation. Longer strings are stored behind an
/// [`Arc`], making [`Clone`] a cheap reference-count bump.
///
/// This type is immutable after construction.
///
/// # Size
///
/// `CompactCStr8` is always 24 bytes on 64-bit platforms (3 machine words).
///
/// # HashMap integration
///
/// `CompactCStr8` implements [`Borrow<CStr8>`], so a
/// `HashMap<CompactCStr8, V>` can be queried with `&CStr8` keys.
pub struct CompactCStr8 {
    repr: Repr,
}

#[derive(Clone)]
enum Repr {
    /// Inline storage. `buf[..len]` is the string content (valid UTF-8,
    /// no interior NUL bytes). `buf[len]` is the NUL terminator.
    /// `len <= MAX_INLINE_LEN` (i.e. `len <= 21`).
    /// Remaining bytes in `buf` after the NUL are zero-initialized.
    Inline { buf: [u8; INLINE_BUF], len: u8 },
    /// Heap storage behind an atomically reference-counted pointer.
    Arc(Arc<CStr8>),
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(mem::size_of::<CompactCStr8>() == 24);

// ---------------------------------------------------------------------------
// Core accessor
// ---------------------------------------------------------------------------

impl CompactCStr8 {
    /// Returns a reference to the underlying [`CStr8`].
    #[inline]
    fn as_cstr8(&self) -> &CStr8 {
        match &self.repr {
            Repr::Inline { buf, len } => {
                let end = *len as usize + 1; // include NUL
                                             // SAFETY: constructors guarantee buf[..end] is valid UTF-8
                                             // with a single NUL terminator at buf[len].
                unsafe { CStr8::from_utf8_with_nul_unchecked(&buf[..end]) }
            },
            Repr::Arc(arc) => arc,
        }
    }

    /// Returns `true` if the string is stored inline (no heap allocation).
    #[inline]
    pub const fn is_inline(&self) -> bool {
        matches!(self.repr, Repr::Inline { .. })
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl CompactCStr8 {
    /// Creates a `CompactCStr8` from a [`CStr8`] reference.
    ///
    /// Strings of 21 bytes or fewer are stored inline; longer strings
    /// are placed behind an [`Arc`].
    #[inline]
    pub fn from_cstr8(s: &CStr8) -> Self {
        let str_len = s.as_bytes().len(); // without NUL
        if str_len <= MAX_INLINE_LEN {
            let mut buf = [0u8; INLINE_BUF];
            buf[..str_len + 1].copy_from_slice(s.as_bytes_with_nul());
            CompactCStr8 {
                repr: Repr::Inline {
                    buf,
                    len: str_len as u8,
                },
            }
        } else {
            CompactCStr8 {
                repr: Repr::Arc(Arc::from(s)),
            }
        }
    }

    /// Creates a `CompactCStr8` from a string literal in const context.
    ///
    /// A NUL terminator is appended automatically.
    ///
    /// # Panics
    ///
    /// Panics (at compile time if used in const context) if the string
    /// is longer than 21 bytes or contains interior NUL bytes.
    /// For fallible construction, use [`new`](Self::new) instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cstr8::CompactCStr8;
    ///
    /// const CHR1: CompactCStr8 = CompactCStr8::new_inline("chr1");
    /// assert_eq!(CHR1.as_str(), "chr1");
    /// assert!(CHR1.is_inline());
    /// ```
    #[inline]
    pub const fn new_inline(s: &str) -> Self {
        let bytes = s.as_bytes();
        assert!(
            bytes.len() <= MAX_INLINE_LEN,
            "string exceeds inline capacity (max 21 bytes)"
        );
        let mut buf = [0u8; INLINE_BUF];
        let mut i = 0;
        while i < bytes.len() {
            assert!(bytes[i] != 0, "string contains interior NUL byte");
            buf[i] = bytes[i];
            i += 1;
        }
        // buf[bytes.len()] is already 0 (NUL terminator).
        CompactCStr8 {
            repr: Repr::Inline {
                buf,
                len: bytes.len() as u8,
            },
        }
    }

    /// Creates a `CompactCStr8` from a [`CStr8`] in const context.
    ///
    /// # Panics
    ///
    /// Panics (at compile time if used in const context) if the string
    /// is longer than 21 bytes. For strings that may be longer, use
    /// [`from_cstr8`](Self::from_cstr8) instead.
    #[inline]
    pub const fn from_cstr8_inline(s: &CStr8) -> Self {
        let src = s.as_bytes_with_nul();
        let str_len = src.len() - 1; // subtract NUL
        assert!(
            str_len <= MAX_INLINE_LEN,
            "string exceeds inline capacity (max 21 bytes)"
        );
        let mut buf = [0u8; INLINE_BUF];
        let mut i = 0;
        while i < str_len + 1 {
            buf[i] = src[i];
            i += 1;
        }
        CompactCStr8 {
            repr: Repr::Inline {
                buf,
                len: str_len as u8,
            },
        }
    }

    /// An empty `CompactCStr8`, usable in const context.
    pub const EMPTY: Self = Self::new_inline("");

    /// Creates a `CompactCStr8` from a string slice.
    ///
    /// The string must not contain interior NUL bytes; a NUL terminator
    /// is appended automatically.
    ///
    /// # Errors
    ///
    /// Returns [`CStr8Error`] if the input contains interior NUL bytes.
    #[inline]
    pub fn new(s: &str) -> Result<Self, CStr8Error> {
        Self::from_utf8(s.as_bytes())
    }

    /// Creates a `CompactCStr8` from a byte slice.
    ///
    /// The bytes must be valid UTF-8 and must not contain NUL bytes.
    /// A NUL terminator is appended automatically. For short strings
    /// (≤ 21 bytes) this is entirely stack-allocated.
    ///
    /// # Errors
    ///
    /// Returns [`CStr8Error`] if the input is not valid UTF-8 or contains
    /// a NUL byte.
    #[inline]
    pub fn from_utf8(bytes: &[u8]) -> Result<Self, CStr8Error> {
        core::str::from_utf8(bytes)?;
        if bytes.len() <= MAX_INLINE_LEN {
            let mut buf = [0u8; INLINE_BUF];
            buf[..bytes.len()].copy_from_slice(bytes);
            // buf[bytes.len()] is already 0 from zero-init (NUL terminator).
            // Validate no interior NUL — CStr::from_bytes_with_nul will
            // report the actual position of any interior NUL byte.
            CStr::from_bytes_with_nul(&buf[..bytes.len() + 1])?;
            Ok(CompactCStr8 {
                repr: Repr::Inline {
                    buf,
                    len: bytes.len() as u8,
                },
            })
        } else {
            // Heap path: build bytes-with-nul once, validate, then construct.
            let mut bytes_with_nul = Vec::with_capacity(bytes.len() + 1);
            bytes_with_nul.extend_from_slice(bytes);
            bytes_with_nul.push(0);
            CStr::from_bytes_with_nul(&bytes_with_nul)?;
            // SAFETY: valid UTF-8 (checked above), exactly one NUL at end
            // (CStr check passed), no interior NULs.
            let cs = unsafe { CString8::from_vec_with_nul_unchecked(bytes_with_nul) };
            Ok(Self::from_cstr8(&cs))
        }
    }

    /// Creates a `CompactCStr8` from a byte slice that must be valid UTF-8
    /// and nul-terminated (with no interior NUL bytes).
    ///
    /// # Errors
    ///
    /// Returns [`CStr8Error`] if the input is not valid UTF-8 or does not
    /// have exactly one NUL byte at the end.
    #[inline]
    pub fn from_utf8_with_nul(bytes: &[u8]) -> Result<Self, CStr8Error> {
        let s = CStr8::from_utf8_with_nul(bytes)?;
        Ok(Self::from_cstr8(s))
    }

    /// Creates a `CompactCStr8` from a byte slice without checking invariants.
    ///
    /// # Safety
    ///
    /// The provided bytes must be valid UTF-8, nul-terminated, and not
    /// contain any interior NUL bytes.
    #[inline]
    pub unsafe fn from_utf8_with_nul_unchecked(bytes: &[u8]) -> Self {
        Self::from_cstr8(CStr8::from_utf8_with_nul_unchecked(bytes))
    }

    /// Creates a `CompactCStr8` from a raw NUL-terminated C string pointer.
    ///
    /// Reads bytes up to the first NUL, validates UTF-8, and stores inline
    /// if short enough. No interior NUL bytes are possible since
    /// [`CStr::from_ptr`] stops at the first NUL.
    ///
    /// # Safety
    ///
    /// The pointer must reference a valid NUL-terminated byte sequence,
    /// and the chosen lifetime must not outlive the allocation.
    /// (The data is copied, so the pointer need only be valid for
    /// the duration of this call.)
    ///
    /// # Errors
    ///
    /// Returns [`CStr8Error`] if the bytes before the NUL terminator are
    /// not valid UTF-8.
    #[inline]
    pub unsafe fn from_ptr(ptr: *const u8) -> Result<Self, CStr8Error> {
        let cstr = core::ffi::CStr::from_ptr(ptr.cast());
        let bytes = cstr.to_bytes(); // without NUL
                                     // Validate UTF-8.
        core::str::from_utf8(bytes)?;
        // No interior NUL possible — CStr stops at the first NUL.
        // Inline if short enough, otherwise Arc.
        if bytes.len() <= MAX_INLINE_LEN {
            let mut buf = [0u8; INLINE_BUF];
            buf[..bytes.len()].copy_from_slice(bytes);
            // buf[bytes.len()] is already 0 (NUL terminator).
            Ok(CompactCStr8 {
                repr: Repr::Inline {
                    buf,
                    len: bytes.len() as u8,
                },
            })
        } else {
            // bytes_with_nul is valid UTF-8 + NUL (validated above, no interior NULs).
            Ok(Self::from_cstr8(CStr8::from_utf8_with_nul_unchecked(
                cstr.to_bytes_with_nul(),
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// Deref
// ---------------------------------------------------------------------------

impl Deref for CompactCStr8 {
    type Target = CStr8;

    #[inline]
    fn deref(&self) -> &CStr8 {
        self.as_cstr8()
    }
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

impl Clone for CompactCStr8 {
    #[inline]
    fn clone(&self) -> Self {
        CompactCStr8 {
            repr: self.repr.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Hash / Eq / Ord — delegate to CStr8 content (Borrow contract)
// ---------------------------------------------------------------------------

impl Hash for CompactCStr8 {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.as_cstr8().hash(state);
    }
}

impl PartialEq for CompactCStr8 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_cstr8() == other.as_cstr8()
    }
}

impl Eq for CompactCStr8 {}

impl PartialOrd for CompactCStr8 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompactCStr8 {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_cstr8().cmp(other.as_cstr8())
    }
}

// ---------------------------------------------------------------------------
// Borrow — critical for HashMap<CompactCStr8, V>::get(&cstr8)
// ---------------------------------------------------------------------------

impl Borrow<CStr8> for CompactCStr8 {
    #[inline]
    fn borrow(&self) -> &CStr8 {
        self.as_cstr8()
    }
}

// ---------------------------------------------------------------------------
// AsRef
// ---------------------------------------------------------------------------

impl AsRef<CStr8> for CompactCStr8 {
    #[inline]
    fn as_ref(&self) -> &CStr8 {
        self.as_cstr8()
    }
}

impl AsRef<str> for CompactCStr8 {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<CStr> for CompactCStr8 {
    #[inline]
    fn as_ref(&self) -> &CStr {
        self.as_c_str()
    }
}

impl AsRef<[u8]> for CompactCStr8 {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

// ---------------------------------------------------------------------------
// From conversions
// ---------------------------------------------------------------------------

impl From<&CStr8> for CompactCStr8 {
    #[inline]
    fn from(s: &CStr8) -> Self {
        Self::from_cstr8(s)
    }
}

impl From<CString8> for CompactCStr8 {
    #[inline]
    fn from(s: CString8) -> Self {
        Self::from_cstr8(&s)
    }
}

impl From<CompactCStr8> for CString8 {
    #[inline]
    fn from(s: CompactCStr8) -> Self {
        // SAFETY: as_bytes_with_nul() is valid UTF-8 with exactly one
        // trailing NUL and no interior NULs (CStr8 invariant).
        unsafe { CString8::from_vec_with_nul_unchecked(s.as_cstr8().as_bytes_with_nul().to_vec()) }
    }
}

impl TryFrom<&str> for CompactCStr8 {
    type Error = CStr8Error;

    /// Converts a `&str` into a `CompactCStr8`.
    ///
    /// Equivalent to [`CompactCStr8::new`]. Fails if the string contains
    /// interior NUL bytes.
    #[inline]
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Self::from_utf8(s.as_bytes())
    }
}

impl TryFrom<&[u8]> for CompactCStr8 {
    type Error = CStr8Error;

    /// Converts a `&[u8]` into a `CompactCStr8`.
    ///
    /// Equivalent to [`CompactCStr8::from_utf8`]. Fails if the bytes
    /// are not valid UTF-8 or contain NUL bytes.
    #[inline]
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Self::from_utf8(bytes)
    }
}

impl From<CompactCStr8> for Arc<CStr8> {
    #[inline]
    fn from(s: CompactCStr8) -> Arc<CStr8> {
        match s.repr {
            Repr::Arc(arc) => arc,
            Repr::Inline { .. } => Arc::from(s.as_cstr8()),
        }
    }
}

// ---------------------------------------------------------------------------
// Display / Debug
// ---------------------------------------------------------------------------

impl fmt::Display for CompactCStr8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

impl fmt::Debug for CompactCStr8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

// ---------------------------------------------------------------------------
// Default
// ---------------------------------------------------------------------------

impl Default for CompactCStr8 {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

// ---------------------------------------------------------------------------
// Cross-type PartialEq
// ---------------------------------------------------------------------------

impl PartialEq<str> for CompactCStr8 {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<CompactCStr8> for str {
    #[inline]
    fn eq(&self, other: &CompactCStr8) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<CStr> for CompactCStr8 {
    #[inline]
    fn eq(&self, other: &CStr) -> bool {
        self.as_c_str() == other
    }
}

impl PartialEq<CompactCStr8> for CStr {
    #[inline]
    fn eq(&self, other: &CompactCStr8) -> bool {
        self == other.as_c_str()
    }
}

impl PartialEq<CStr8> for CompactCStr8 {
    #[inline]
    fn eq(&self, other: &CStr8) -> bool {
        self.as_cstr8() == other
    }
}

impl PartialEq<CompactCStr8> for CStr8 {
    #[inline]
    fn eq(&self, other: &CompactCStr8) -> bool {
        self == other.as_cstr8()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use {
        super::*,
        alloc::{borrow::ToOwned, collections::BTreeMap, vec},
    };

    extern crate std;
    use std::collections::HashMap;

    #[test]
    fn size_is_24_bytes() {
        assert_eq!(mem::size_of::<CompactCStr8>(), 24);
    }

    #[test]
    fn empty_string() {
        let s = CompactCStr8::default();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "");
        assert_eq!(s.as_bytes_with_nul(), b"\0");
        assert_eq!(s.as_c_str(), c"");
    }

    #[test]
    fn const_empty() {
        const EMPTY: CompactCStr8 = CompactCStr8::EMPTY;
        assert!(EMPTY.is_inline());
        assert_eq!(EMPTY.as_str(), "");
    }

    #[test]
    fn const_new_inline() {
        const CHR1: CompactCStr8 = CompactCStr8::new_inline("chr1");
        assert!(CHR1.is_inline());
        assert_eq!(CHR1.as_str(), "chr1");
        assert_eq!(CHR1.as_bytes_with_nul(), b"chr1\0");
    }

    #[test]
    fn const_new_inline_max() {
        const MAX: CompactCStr8 = CompactCStr8::new_inline("aaaaaaaaaaaaaaaaaaaaa"); // 21 'a's
        assert!(MAX.is_inline());
        assert_eq!(MAX.as_str().len(), MAX_INLINE_LEN);
    }

    #[test]
    fn const_from_cstr8_inline() {
        const S: CompactCStr8 = CompactCStr8::from_cstr8_inline(cstr8!("hello"));
        assert!(S.is_inline());
        assert_eq!(S.as_str(), "hello");
    }

    #[test]
    fn const_new_inline_matches_new() {
        const INLINE: CompactCStr8 = CompactCStr8::new_inline("chr1");
        let runtime = CompactCStr8::new("chr1").unwrap();
        assert_eq!(INLINE, runtime);
    }

    #[test]
    fn short_string_inline() {
        let s = CompactCStr8::new("chr1").unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "chr1");
        assert_eq!(s.as_bytes_with_nul(), b"chr1\0");
    }

    #[test]
    fn max_inline_length() {
        // 21 bytes should be inline
        let input = "aaaaaaaaaaaaaaaaaaaaa"; // 21 'a's
        assert_eq!(input.len(), MAX_INLINE_LEN);
        let s = CompactCStr8::new(input).unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), input);
    }

    #[test]
    fn just_over_inline_uses_arc() {
        // 22 bytes should use Arc
        let input = "aaaaaaaaaaaaaaaaaaaaaa"; // 22 'a's
        assert_eq!(input.len(), MAX_INLINE_LEN + 1);
        let s = CompactCStr8::new(input).unwrap();
        assert!(!s.is_inline());
        assert_eq!(s.as_str(), input);
    }

    #[test]
    fn long_string_arc() {
        let input = "a]very_long_scaffold_name_that_exceeds_inline_cap";
        assert!(input.len() > MAX_INLINE_LEN);
        let s = CompactCStr8::new(input).unwrap();
        assert!(!s.is_inline());
        assert_eq!(s.as_str(), input);
        assert_eq!(s.as_bytes_with_nul().last(), Some(&0));
    }

    #[test]
    fn clone_inline_copies() {
        let s = CompactCStr8::new("chr1").unwrap();
        let t = s.clone();
        assert!(t.is_inline());
        assert_eq!(s, t);
    }

    #[test]
    fn clone_arc_shares() {
        let input = "a_long_string_that_definitely_exceeds_inline";
        let s = CompactCStr8::new(input).unwrap();
        let t = s.clone();
        assert!(!t.is_inline());
        assert_eq!(s, t);
        // Both point to the same Arc allocation
        assert_eq!(s.as_ptr(), t.as_ptr());
    }

    #[test]
    fn from_cstr8_static() {
        let s = CompactCStr8::from_cstr8(cstr8!("hello"));
        assert!(s.is_inline());
        assert_eq!(s, *"hello");
    }

    #[test]
    fn from_cstring8() {
        let cs = CString8::new("world").unwrap();
        let s = CompactCStr8::from(cs);
        assert!(s.is_inline());
        assert_eq!(s, *"world");
    }

    #[test]
    fn into_cstring8() {
        let s = CompactCStr8::new("test").unwrap();
        let cs: CString8 = s.into();
        assert_eq!(cs.as_str(), "test");
    }

    #[test]
    fn into_arc_zero_cost_when_already_arc() {
        let input = "a_long_string_that_definitely_exceeds_inline";
        let s = CompactCStr8::new(input).unwrap();
        let ptr = s.as_ptr();
        let arc: Arc<CStr8> = s.into();
        // The Arc should reuse the same allocation
        assert_eq!(arc.as_ptr(), ptr);
    }

    #[test]
    fn borrow_hashmap_lookup() {
        let mut map = HashMap::new();
        map.insert(CompactCStr8::new("chr1").unwrap(), 0u32);
        map.insert(CompactCStr8::new("chr2").unwrap(), 1u32);

        // Lookup with &CStr8 via Borrow
        assert_eq!(map.get(cstr8!("chr1")), Some(&0));
        assert_eq!(map.get(cstr8!("chr2")), Some(&1));
        assert_eq!(map.get(cstr8!("chr3")), None);
    }

    #[test]
    fn borrow_btreemap_lookup() {
        let mut map = BTreeMap::new();
        map.insert(CompactCStr8::new("GT").unwrap(), 1i32);
        map.insert(CompactCStr8::new("AF").unwrap(), 2i32);

        assert_eq!(map.get(cstr8!("GT")), Some(&1));
        assert_eq!(map.get(cstr8!("AF")), Some(&2));
    }

    #[test]
    fn hash_consistency() {
        use core::hash::BuildHasher;
        let hasher = std::collections::hash_map::RandomState::new();

        let compact = CompactCStr8::new("chr1").unwrap();
        let cstr8_ref: &CStr8 = cstr8!("chr1");

        let h1 = hasher.hash_one(&compact);
        let h2 = hasher.hash_one(cstr8_ref);
        assert_eq!(h1, h2, "CompactCStr8 and &CStr8 must hash identically");
    }

    #[test]
    fn partial_eq_cross_type() {
        let s = CompactCStr8::new("hello").unwrap();

        // vs str
        let hello_str: &str = "hello";
        assert!(s == *hello_str);
        assert!(*hello_str == s);

        // vs CStr
        assert!(s == *c"hello");
        assert!(*c"hello" == s);

        // vs CStr8
        assert!(s == *cstr8!("hello"));
        assert!(*cstr8!("hello") == s);
    }

    #[test]
    fn from_utf8_short_inline() {
        let s = CompactCStr8::from_utf8(b"chr1").unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "chr1");
        assert_eq!(s.as_bytes_with_nul(), b"chr1\0");
    }

    #[test]
    fn from_utf8_empty() {
        let s = CompactCStr8::from_utf8(b"").unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "");
    }

    #[test]
    fn from_utf8_max_inline() {
        let input = b"aaaaaaaaaaaaaaaaaaaaa"; // 21 bytes
        assert_eq!(input.len(), MAX_INLINE_LEN);
        let s = CompactCStr8::from_utf8(input).unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "aaaaaaaaaaaaaaaaaaaaa");
    }

    #[test]
    fn from_utf8_arc_path() {
        let input = b"aaaaaaaaaaaaaaaaaaaaaa"; // 22 bytes
        assert_eq!(input.len(), MAX_INLINE_LEN + 1);
        let s = CompactCStr8::from_utf8(input).unwrap();
        assert!(!s.is_inline());
        assert_eq!(s.as_str(), "aaaaaaaaaaaaaaaaaaaaaa");
    }

    #[test]
    fn from_utf8_rejects_interior_nul() {
        assert!(CompactCStr8::from_utf8(b"has\0nul").is_err());
    }

    #[test]
    fn from_utf8_rejects_invalid_utf8() {
        assert!(CompactCStr8::from_utf8(b"\xff\xfe").is_err());
    }

    #[test]
    fn from_utf8_matches_new() {
        // from_utf8 and new should produce identical results
        for input in &["", "x", "chr1", "CHROMOSOME_I", "aaaaaaaaaaaaaaaaaaaaa"] {
            let from_new = CompactCStr8::new(input).unwrap();
            let from_utf8 = CompactCStr8::from_utf8(input.as_bytes()).unwrap();
            assert_eq!(from_new, from_utf8);
            assert_eq!(from_new.is_inline(), from_utf8.is_inline());
        }
    }

    #[test]
    fn from_ptr_short_inline() {
        let s = unsafe { CompactCStr8::from_ptr(b"chr1\0".as_ptr()) }.unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "chr1");
        assert_eq!(s.as_bytes_with_nul(), b"chr1\0");
    }

    #[test]
    fn from_ptr_empty() {
        let s = unsafe { CompactCStr8::from_ptr(b"\0".as_ptr()) }.unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "");
    }

    #[test]
    fn from_ptr_max_inline() {
        let input = b"aaaaaaaaaaaaaaaaaaaaa\0"; // 21 + NUL
        let s = unsafe { CompactCStr8::from_ptr(input.as_ptr()) }.unwrap();
        assert!(s.is_inline());
        assert_eq!(s.as_str(), "aaaaaaaaaaaaaaaaaaaaa");
    }

    #[test]
    fn from_ptr_arc_path() {
        let input = b"aaaaaaaaaaaaaaaaaaaaaa\0"; // 22 + NUL
        let s = unsafe { CompactCStr8::from_ptr(input.as_ptr()) }.unwrap();
        assert!(!s.is_inline());
        assert_eq!(s.as_str(), "aaaaaaaaaaaaaaaaaaaaaa");
    }

    #[test]
    fn from_ptr_rejects_invalid_utf8() {
        let input = b"\xff\xfe\0";
        assert!(unsafe { CompactCStr8::from_ptr(input.as_ptr()) }.is_err());
    }

    #[test]
    fn from_ptr_stops_at_first_nul() {
        // Interior NUL acts as terminator, not an error
        let input = b"hello\0world\0";
        let s = unsafe { CompactCStr8::from_ptr(input.as_ptr()) }.unwrap();
        assert_eq!(s.as_str(), "hello");
    }

    #[test]
    fn from_ptr_matches_from_utf8() {
        for input in &["", "x", "chr1", "CHROMOSOME_I", "aaaaaaaaaaaaaaaaaaaaa"] {
            let mut buf = input.as_bytes().to_vec();
            buf.push(0);
            let from_ptr = unsafe { CompactCStr8::from_ptr(buf.as_ptr()) }.unwrap();
            let from_utf8 = CompactCStr8::from_utf8(input.as_bytes()).unwrap();
            assert_eq!(from_ptr, from_utf8);
            assert_eq!(from_ptr.is_inline(), from_utf8.is_inline());
        }
    }

    #[test]
    fn error_interior_nul() {
        assert!(CompactCStr8::new("has\0nul").is_err());
    }

    #[test]
    fn error_invalid_utf8() {
        assert!(CompactCStr8::from_utf8_with_nul(b"\xff\0").is_err());
    }

    #[test]
    fn from_utf8_with_nul_valid() {
        let s = CompactCStr8::from_utf8_with_nul(b"ok\0").unwrap();
        assert_eq!(s.as_str(), "ok");
        assert!(s.is_inline());
    }

    #[test]
    fn display_and_debug() {
        let s = CompactCStr8::new("chr1").unwrap();
        assert_eq!(alloc::format!("{s}"), "chr1");
        assert_eq!(alloc::format!("{s:?}"), "\"chr1\"");
    }

    #[test]
    fn default_is_empty() {
        let s = CompactCStr8::default();
        assert_eq!(s.as_str(), "");
        assert!(s.is_inline());
    }

    #[test]
    fn deref_to_cstr8() {
        let s = CompactCStr8::new("test").unwrap();
        let r: &CStr8 = &s;
        assert_eq!(r.as_str(), "test");
    }

    #[test]
    fn as_ref_impls() {
        let s = CompactCStr8::new("test").unwrap();
        let _: &CStr8 = s.as_ref();
        let _: &str = s.as_ref();
        let _: &CStr = s.as_ref();
        let _: &[u8] = s.as_ref();
    }

    #[test]
    fn ord_consistency() {
        let a = CompactCStr8::new("aaa").unwrap();
        let b = CompactCStr8::new("bbb").unwrap();
        assert!(a < b);
        assert_eq!(a.cmp(&b), cstr8!("aaa").cmp(cstr8!("bbb")));
    }

    #[test]
    fn cstr8_to_owned_no_double_nul() {
        // Regression: ToOwned previously used from_vec_unchecked which
        // appends a NUL, but the input already had one → double NUL.
        let s = cstr8!("hello");
        let owned = s.to_owned();
        assert_eq!(owned.as_str(), "hello");
        assert_eq!(owned.as_bytes_with_nul(), b"hello\0");
    }

    #[test]
    fn from_utf8_nul_error_is_accurate() {
        // The NUL error should report the actual position of the NUL byte,
        // not a fabricated position 0.
        let err = CompactCStr8::from_utf8(b"hel\0lo").unwrap_err();
        match &err {
            CStr8Error::NulError(inner) => {
                let inner_msg = alloc::format!("{inner:?}");
                assert!(
                    inner_msg.contains("position: 3"),
                    "error should report position 3 for NUL at byte 3: {inner_msg}"
                );
            },
            other => panic!("expected NulError, got {other:?}"),
        }
    }

    #[test]
    fn from_utf8_nul_error_long_is_accurate() {
        // Same test but for the heap path (> MAX_INLINE_LEN).
        let mut input = vec![b'a'; MAX_INLINE_LEN + 5];
        let nul_pos = MAX_INLINE_LEN + 2;
        input[nul_pos] = 0;
        let err = CompactCStr8::from_utf8(&input).unwrap_err();
        match &err {
            CStr8Error::NulError(inner) => {
                let inner_msg = alloc::format!("{inner:?}");
                assert!(
                    inner_msg.contains(&alloc::format!("position: {nul_pos}")),
                    "heap-path error should report position {nul_pos}: {inner_msg}"
                );
            },
            other => panic!("expected NulError, got {other:?}"),
        }
    }

    #[test]
    fn new_returns_cstr8_error() {
        // new() should return CStr8Error, same as TryFrom<&str>.
        let err_new = CompactCStr8::new("has\0nul").unwrap_err();
        let err_try: CStr8Error = CompactCStr8::try_from("has\0nul").unwrap_err();
        assert!(
            matches!(err_new, CStr8Error::NulError(_)),
            "new() should return CStr8Error::NulError"
        );
        assert!(
            matches!(err_try, CStr8Error::NulError(_)),
            "try_from() should return CStr8Error::NulError"
        );
    }

    #[test]
    fn to_owned_roundtrip_long() {
        // Verify to_owned works for strings that go through the Arc path.
        let input = "a_long_string_that_definitely_exceeds_inline";
        let compact = CompactCStr8::new(input).unwrap();
        let owned = compact.to_owned();
        assert_eq!(owned.as_str(), input);
        assert_eq!(owned.as_bytes_with_nul().last(), Some(&0));
        // Only one NUL at the end.
        assert_eq!(
            owned
                .as_bytes_with_nul()
                .iter()
                .filter(|&&b| b == 0)
                .count(),
            1
        );
    }

    mod proptests {
        use {super::*, proptest::prelude::*};

        /// Strategy that generates valid UTF-8 strings without interior NUL bytes.
        ///
        /// Combines ASCII-heavy generation with proptest's string regex
        /// to get diverse UTF-8 coverage including multibyte characters.
        fn valid_str() -> impl Strategy<Value = std::string::String> {
            prop_oneof![
                // ASCII-heavy path: good for exercising inline storage.
                prop::collection::vec(1u8..=127, 0..=80)
                    .prop_filter_map("must be valid UTF-8", |bytes| {
                        std::string::String::from_utf8(bytes).ok()
                    }),
                // Full byte range: exercises multibyte UTF-8 sequences.
                prop::collection::vec(1u8..=255, 0..=80)
                    .prop_filter_map("must be valid UTF-8", |bytes| {
                        std::string::String::from_utf8(bytes).ok()
                    }),
                // Regex-based: ensures diverse multibyte UTF-8 (emoji, CJK, etc.)
                "[^\x00]{0,40}",
            ]
        }

        /// Strategy biased toward the inline/Arc boundary (lengths 19-25).
        fn boundary_str() -> impl Strategy<Value = std::string::String> {
            (19usize..=25).prop_flat_map(|len| {
                prop::collection::vec(b'a'..=b'z', len)
                    .prop_map(|v| unsafe { std::string::String::from_utf8_unchecked(v) })
            })
        }

        /// Strategy that always produces strings longer than MAX_INLINE_LEN.
        fn long_str() -> impl Strategy<Value = std::string::String> {
            ((MAX_INLINE_LEN + 1)..=80).prop_flat_map(|len| {
                prop::collection::vec(b'a'..=b'z', len)
                    .prop_map(|v| unsafe { std::string::String::from_utf8_unchecked(v) })
            })
        }

        proptest! {
            #[test]
            fn roundtrip_as_str(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                prop_assert_eq!(compact.as_str(), s.as_str());
            }

            #[test]
            fn roundtrip_as_bytes(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                prop_assert_eq!(compact.as_bytes(), s.as_bytes());
            }

            #[test]
            fn roundtrip_as_bytes_with_nul(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let mut expected = s.into_bytes();
                expected.push(0);
                prop_assert_eq!(compact.as_bytes_with_nul(), expected.as_slice());
            }

            #[test]
            fn roundtrip_as_c_str(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let expected = std::ffi::CString::new(s).unwrap();
                prop_assert_eq!(compact.as_c_str(), expected.as_c_str());
            }

            #[test]
            fn inline_threshold(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                if s.len() <= MAX_INLINE_LEN {
                    prop_assert!(compact.is_inline(),
                        "string of len {} should be inline", s.len());
                } else {
                    prop_assert!(!compact.is_inline(),
                        "string of len {} should be Arc", s.len());
                }
            }

            #[test]
            fn boundary_inline_threshold(s in boundary_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                if s.len() <= MAX_INLINE_LEN {
                    prop_assert!(compact.is_inline());
                } else {
                    prop_assert!(!compact.is_inline());
                }
                prop_assert_eq!(compact.as_str(), s.as_str());
            }

            #[test]
            fn clone_equals_original(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let cloned = compact.clone();
                prop_assert_eq!(&compact, &cloned);
                prop_assert_eq!(compact.as_str(), cloned.as_str());
                prop_assert_eq!(compact.is_inline(), cloned.is_inline());
            }

            #[test]
            fn clone_arc_shares_pointer(s in long_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let cloned = compact.clone();
                prop_assert!(!compact.is_inline());
                prop_assert_eq!(compact.as_ptr(), cloned.as_ptr(),
                    "Arc clones must share the same allocation");
            }

            #[test]
            fn hash_eq_consistent_with_cstr8(s in valid_str()) {
                use core::hash::BuildHasher;
                let hasher = std::collections::hash_map::RandomState::new();

                let compact = CompactCStr8::new(&s).unwrap();
                let cstring8 = CString8::new(&s).unwrap();
                let cstr8_ref: &CStr8 = &cstring8;

                let h_compact = hasher.hash_one(&compact);
                let h_cstr8 = hasher.hash_one(cstr8_ref);
                prop_assert_eq!(h_compact, h_cstr8,
                    "Borrow contract: CompactCStr8 and CStr8 must hash identically");
            }

            #[test]
            fn eq_consistent_with_cstr8(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let cstring8 = CString8::new(&s).unwrap();
                let cstr8_ref: &CStr8 = &cstring8;
                prop_assert_eq!(compact.as_cstr8(), cstr8_ref);
            }

            #[test]
            fn ord_consistent_with_cstr8(a in valid_str(), b in valid_str()) {
                let ca = CompactCStr8::new(&a).unwrap();
                let cb = CompactCStr8::new(&b).unwrap();
                let sa = CString8::new(&a).unwrap();
                let sb = CString8::new(&b).unwrap();
                let ra: &CStr8 = &sa;
                let rb: &CStr8 = &sb;
                prop_assert_eq!(ca.cmp(&cb), ra.cmp(rb),
                    "Ord must be consistent with CStr8");
            }

            #[test]
            fn hashmap_borrow_lookup(s in valid_str()) {
                let mut map = std::collections::HashMap::new();
                let compact = CompactCStr8::new(&s).unwrap();
                map.insert(compact.clone(), 42u32);

                let cstring8 = CString8::new(&s).unwrap();
                let key: &CStr8 = &cstring8;
                prop_assert_eq!(map.get(key), Some(&42u32),
                    "HashMap lookup via &CStr8 must find CompactCStr8 key");
            }

            #[test]
            fn into_cstring8_roundtrip(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let cs: CString8 = compact.into();
                prop_assert_eq!(cs.as_str(), s.as_str());
            }

            #[test]
            fn from_cstring8_roundtrip(s in valid_str()) {
                let cs = CString8::new(&s).unwrap();
                let compact = CompactCStr8::from(cs);
                prop_assert_eq!(compact.as_str(), s.as_str());
            }

            #[test]
            fn from_utf8_with_nul_roundtrip(s in valid_str()) {
                let mut bytes = s.into_bytes();
                bytes.push(0);
                let compact = CompactCStr8::from_utf8_with_nul(&bytes).unwrap();
                prop_assert_eq!(compact.as_bytes_with_nul(), bytes.as_slice());
            }

            #[test]
            fn from_utf8_roundtrip(s in valid_str()) {
                let compact = CompactCStr8::from_utf8(s.as_bytes()).unwrap();
                prop_assert_eq!(compact.as_str(), s.as_str());
            }

            #[test]
            fn from_utf8_matches_new(s in valid_str()) {
                let from_new = CompactCStr8::new(&s).unwrap();
                let from_utf8 = CompactCStr8::from_utf8(s.as_bytes()).unwrap();
                prop_assert_eq!(from_new.is_inline(), from_utf8.is_inline());
                prop_assert_eq!(from_new, from_utf8);
            }

            #[test]
            fn from_utf8_inline_threshold(s in valid_str()) {
                let compact = CompactCStr8::from_utf8(s.as_bytes()).unwrap();
                if s.len() <= MAX_INLINE_LEN {
                    prop_assert!(compact.is_inline());
                } else {
                    prop_assert!(!compact.is_inline());
                }
            }

            #[test]
            fn from_ptr_roundtrip(s in valid_str()) {
                let mut buf = s.clone().into_bytes();
                buf.push(0);
                let compact = unsafe { CompactCStr8::from_ptr(buf.as_ptr()) }.unwrap();
                prop_assert_eq!(compact.as_str(), s.as_str());
            }

            #[test]
            fn from_ptr_matches_new(s in valid_str()) {
                let mut buf = s.clone().into_bytes();
                buf.push(0);
                let from_ptr = unsafe { CompactCStr8::from_ptr(buf.as_ptr()) }.unwrap();
                let from_new = CompactCStr8::new(&s).unwrap();
                prop_assert_eq!(from_ptr.is_inline(), from_new.is_inline());
                prop_assert_eq!(from_ptr, from_new);
            }

            #[test]
            fn from_ptr_inline_threshold(s in valid_str()) {
                let mut buf = s.clone().into_bytes();
                buf.push(0);
                let compact = unsafe { CompactCStr8::from_ptr(buf.as_ptr()) }.unwrap();
                if s.len() <= MAX_INLINE_LEN {
                    prop_assert!(compact.is_inline());
                } else {
                    prop_assert!(!compact.is_inline());
                }
            }

            #[test]
            fn into_arc_roundtrip(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let arc: Arc<CStr8> = compact.into();
                prop_assert_eq!(arc.as_str(), s.as_str());
            }

            #[test]
            fn into_arc_zero_copy_when_arc(s in long_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                prop_assert!(!compact.is_inline());
                let ptr = compact.as_ptr();
                let arc: Arc<CStr8> = compact.into();
                prop_assert_eq!(arc.as_ptr(), ptr,
                    "into Arc from Arc variant must reuse allocation");
            }

            #[test]
            fn to_owned_roundtrip(s in valid_str()) {
                let compact = CompactCStr8::new(&s).unwrap();
                let owned = compact.to_owned();
                prop_assert_eq!(owned.as_str(), s.as_str());
                prop_assert_eq!(owned.as_bytes_with_nul().last(), Some(&0u8));
                // Exactly one NUL byte, at the end.
                let nul_count = owned.as_bytes_with_nul().iter().filter(|&&b| b == 0).count();
                prop_assert_eq!(nul_count, 1, "to_owned must produce exactly one NUL");
            }

            #[test]
            fn to_owned_via_cstr8_roundtrip(s in valid_str()) {
                // Test ToOwned on &CStr8 directly (the fixed impl).
                let cstring8 = CString8::new(&s).unwrap();
                let cstr8_ref: &CStr8 = &cstring8;
                let owned = cstr8_ref.to_owned();
                prop_assert_eq!(owned.as_str(), s.as_str());
                let nul_count = owned.as_bytes_with_nul().iter().filter(|&&b| b == 0).count();
                prop_assert_eq!(nul_count, 1, "CStr8::to_owned must produce exactly one NUL");
            }

            #[test]
            fn new_and_try_from_str_agree(s in valid_str()) {
                // new() and TryFrom<&str> should always agree.
                let from_new = CompactCStr8::new(&s).unwrap();
                let from_try: CompactCStr8 = CompactCStr8::try_from(s.as_str()).unwrap();
                prop_assert_eq!(from_new.is_inline(), from_try.is_inline());
                prop_assert_eq!(from_new, from_try);
            }

            #[test]
            fn from_utf8_rejects_nul_at_any_inline_position(pos in 0usize..MAX_INLINE_LEN) {
                // Place a NUL at every possible inline position.
                let mut bytes = vec![b'x'; MAX_INLINE_LEN];
                bytes[pos] = 0;
                let result = CompactCStr8::from_utf8(&bytes);
                prop_assert!(result.is_err(),
                    "NUL at position {pos} should be rejected");
            }

            #[test]
            fn from_utf8_rejects_nul_in_heap_range(
                len in (MAX_INLINE_LEN + 1)..=60usize,
                pos_frac in 0.0f64..1.0,
            ) {
                let pos = (pos_frac * (len as f64)) as usize;
                let pos = pos.min(len - 1);
                let mut bytes = vec![b'x'; len];
                bytes[pos] = 0;
                let result = CompactCStr8::from_utf8(&bytes);
                prop_assert!(result.is_err(),
                    "NUL at position {pos} in {len}-byte input should be rejected");
            }

            #[test]
            fn multibyte_utf8_roundtrip(s in "[^\x00]{0,40}") {
                // Exercises emoji, CJK, combining characters, etc.
                let compact = CompactCStr8::new(&s).unwrap();
                prop_assert_eq!(compact.as_str(), s.as_str());
                // Verify inline/arc threshold is by byte length, not char count.
                if s.len() <= MAX_INLINE_LEN {
                    prop_assert!(compact.is_inline());
                } else {
                    prop_assert!(!compact.is_inline());
                }
            }

            #[test]
            fn boundary_multibyte_utf8(s in "[^\x00]{4,8}") {
                // Multibyte chars near the boundary: 4-8 chars could be
                // 4-32 bytes depending on encoding.
                let compact = CompactCStr8::new(&s).unwrap();
                prop_assert_eq!(compact.as_str(), s.as_str());
            }
        }
    }
}
