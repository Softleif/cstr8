use {crate::CStr8, core::ffi::CStr};

// ---------------------------------------------------------------------------
// cstr8! macro — one test per input form
// ---------------------------------------------------------------------------

#[test]
fn macro_str_literal() {
    const S: &CStr8 = cstr8!("literal");
    assert_eq!(S, "literal");
    assert_eq!(S.as_bytes_with_nul(), b"literal\0");
}

#[test]
fn macro_str_constant() {
    const INPUT: &str = "constant";
    const S: &CStr8 = cstr8!(INPUT);
    assert_eq!(S, "constant");
    assert_eq!(S.as_bytes_with_nul(), b"constant\0");
}

#[test]
fn macro_byte_slice_constant() {
    const INPUT: &[u8] = b"bytes constant";
    const S: &CStr8 = cstr8!(INPUT);
    assert_eq!(S, "bytes constant");
    assert_eq!(S.as_bytes_with_nul(), b"bytes constant\0");
}

#[test]
fn macro_byte_array_ref_literal() {
    const S: &CStr8 = cstr8!(b"bytes literal");
    assert_eq!(S, "bytes literal");
    assert_eq!(S.as_bytes_with_nul(), b"bytes literal\0");
}

#[test]
fn macro_byte_array_ref_constant() {
    const INPUT: &[u8; 14] = b"bytes constant";
    const S: &CStr8 = cstr8!(INPUT);
    assert_eq!(S, "bytes constant");
    assert_eq!(S.as_bytes_with_nul(), b"bytes constant\0");
}

#[test]
fn macro_byte_array_literal() {
    const S: &CStr8 = cstr8!([b'h', b'i']);
    assert_eq!(S, "hi");
    assert_eq!(S.as_bytes_with_nul(), b"hi\0");
}

#[test]
fn macro_byte_array_constant() {
    const INPUT: [u8; 19] = *b"byte array constant";
    const S: &CStr8 = cstr8!(INPUT);
    assert_eq!(S, "byte array constant");
    assert_eq!(S.as_bytes_with_nul(), b"byte array constant\0");
}

#[test]
fn macro_cstr_literal() {
    const S: &CStr8 = cstr8!(c"cstr literal");
    assert_eq!(S, "cstr literal");
    assert_eq!(S.as_bytes_with_nul(), b"cstr literal\0");
}

#[test]
fn macro_cstr_constant() {
    const INPUT: &CStr = c"cstr constant";
    const S: &CStr8 = cstr8!(INPUT);
    assert_eq!(S, "cstr constant");
    assert_eq!(S.as_bytes_with_nul(), b"cstr constant\0");
}

// ---------------------------------------------------------------------------
// CStr8 const constructors
// ---------------------------------------------------------------------------

#[test]
fn const_from_utf8_with_nul() {
    const S: &CStr8 = match CStr8::from_utf8_with_nul(b"hello\0") {
        Ok(s) => s,
        Err(_) => panic!("invalid"),
    };
    assert_eq!(S, "hello");
    assert_eq!(S.as_c_str(), c"hello");
}

#[test]
fn const_from_utf8_with_nul_empty() {
    const S: &CStr8 = match CStr8::from_utf8_with_nul(b"\0") {
        Ok(s) => s,
        Err(_) => panic!("invalid"),
    };
    assert_eq!(S, "");
}

#[test]
fn const_from_utf8_with_nul_rejects_bad_input() {
    // Invalid UTF-8 at compile time.
    const BAD_UTF8: bool = CStr8::from_utf8_with_nul(b"\xff\0").is_err();
    assert!(BAD_UTF8);

    // Interior NUL at compile time.
    const INTERIOR_NUL: bool = CStr8::from_utf8_with_nul(b"a\0b\0").is_err();
    assert!(INTERIOR_NUL);

    // Missing NUL at compile time.
    const NO_NUL: bool = CStr8::from_utf8_with_nul(b"abc").is_err();
    assert!(NO_NUL);
}

// ---------------------------------------------------------------------------
// CStr8 cross-type comparisons
// ---------------------------------------------------------------------------

#[test]
fn cstr8_eq_str() {
    let s = cstr8!("hello");
    assert_eq!(s, "hello");
    assert_ne!(s, "world");
}

#[test]
fn cstr8_eq_cstr() {
    let s = cstr8!("hello");
    assert_eq!(s, c"hello");
    assert_ne!(s, c"world");
}

#[test]
fn cstr8_ord_str() {
    let s = cstr8!("bbb");
    assert!(s > "aaa");
    assert!(s < "ccc");
}

#[test]
fn cstr8_try_from_cstr() {
    let cs: &CStr8 = c"valid utf8".try_into().unwrap();
    assert_eq!(cs, "valid utf8");
}
