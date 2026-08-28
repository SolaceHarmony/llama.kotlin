// port-lint: source llama.cpp/src/unicode-data.h llama.cpp/src/unicode-data.cpp
package io.github.kotlinmania.llama.model

/**
 * Unicode range entry for NFD normalization.
 *
 * When a code point falls within [first]..[last], the NFD decomposition
 * maps it to [nfd].
 *
 * @param first Start of the code-point range (inclusive).
 * @param last  End of the code-point range (inclusive).
 * @param nfd   The replacement code point for NFD normalization.
 */
data class RangeNfd(
    val first: UInt,
    val last: UInt,
    val nfd: UInt,
)

/**
 * Static Unicode lookup tables used by [LlamaUnicode] for character
 * classification, case mapping, and NFD normalization.
 *
 * The C++ implementation (`unicode-data.cpp`) ships ~7 000 lines of
 * pre-computed tables that are generated from the Unicode Character Database.
 * Rather than duplicating all of that data, this Kotlin port takes a hybrid
 * approach:
 *
 * 1. **Kotlin stdlib delegation** – [Char.category], [Char.isLetter],
 *    [Char.isDigit], [Char.isWhitespace], [Char.lowercaseChar] etc. already
 *    expose the Unicode tables compiled into the Kotlin/Native runtime, so we
 *    use those for basic classification and case mapping wherever possible.
 *
 * 2. **Supplementary tables** – A small subset of the original C++ data is
 *    included verbatim (first ~50 entries) for reference and testing.  Full
 *    tables can be generated from the UCD if needed.
 *
 * 3. **NFD normalization ranges** – A representative subset is included.  The
 *    complete table has ~1 800 entries; add the remaining entries from the UCD
 *    when full NFD accuracy is required.
 */
object LlamaUnicodeData {

    /** Maximum valid Unicode code point (inclusive). */
    const val MAX_CODEPOINTS: UInt = 0x110000u

    // =========================================================================
    // Unicode ranges + flags
    // =========================================================================
    //
    // In the C++ source this is `unicode_ranges_flags` – a sorted list of
    // (codepoint, flags) pairs that defines contiguous ranges.  Each entry
    // starts a range that extends up to (but not including) the *next* entry's
    // code point.
    //
    // The full table has ~2 275 entries.  We include the first 50 as a
    // structural example; the remainder should be generated from the UCD or
    // delegated to Kotlin stdlib APIs.

    /**
     * First 50 entries of `unicode_ranges_flags`.
     *
     * Each pair is `(codepoint, flags)` where `flags` is a bitmask of
     * [UnicodeCptFlags] constants.
     *
     * LATER: Include all ~2 275 entries from unicode-data.cpp for full coverage,
     *       or rely on Kotlin stdlib's [Char.category] for runtime lookups.
     */
    val unicodeRangesFlags: List<Pair<UInt, UShort>> = listOf(
        0x000000u to 0x0080u, // CONTROL
        0x000009u to 0x0180u, // CONTROL | WHITESPACE
        0x00000Eu to 0x0080u, // CONTROL
        0x000020u to 0x0108u, // SEPARATOR | WHITESPACE
        0x000021u to 0x0020u, // PUNCTUATION
        0x000024u to 0x0040u, // SYMBOL
        0x000025u to 0x0020u, // PUNCTUATION
        0x00002Bu to 0x0040u, // SYMBOL
        0x00002Cu to 0x0020u, // PUNCTUATION
        0x000030u to 0x0002u, // NUMBER
        0x00003Au to 0x0020u, // PUNCTUATION
        0x00003Cu to 0x0040u, // SYMBOL
        0x00003Fu to 0x0020u, // PUNCTUATION
        0x000041u to 0x0404u, // LETTER | UPPERCASE
        0x00005Bu to 0x0020u, // PUNCTUATION
        0x00005Eu to 0x0040u, // SYMBOL
        0x00005Fu to 0x0020u, // PUNCTUATION
        0x000060u to 0x0040u, // SYMBOL
        0x000061u to 0x0204u, // LETTER | LOWERCASE
        0x00007Bu to 0x0020u, // PUNCTUATION
        0x00007Cu to 0x0040u, // SYMBOL
        0x00007Du to 0x0020u, // PUNCTUATION
        0x00007Eu to 0x0040u, // SYMBOL
        0x00007Fu to 0x0080u, // CONTROL
        0x000085u to 0x0180u, // CONTROL | WHITESPACE
        0x000086u to 0x0080u, // CONTROL
        0x0000A0u to 0x0108u, // SEPARATOR | WHITESPACE
        0x0000A1u to 0x0020u, // PUNCTUATION
        0x0000A2u to 0x0040u, // SYMBOL
        0x0000A7u to 0x0020u, // PUNCTUATION
        0x0000A8u to 0x0040u, // SYMBOL
        0x0000A9u to 0x0040u, // SYMBOL
        0x0000AAu to 0x0204u, // LETTER | LOWERCASE
        0x0000ABu to 0x0020u, // PUNCTUATION
        0x0000ACu to 0x0040u, // SYMBOL
        0x0000ADu to 0x0080u, // CONTROL
        0x0000AEu to 0x0040u, // SYMBOL
        0x0000B0u to 0x0040u, // SYMBOL
        0x0000B2u to 0x0002u, // NUMBER
        0x0000B4u to 0x0040u, // SYMBOL
        0x0000B5u to 0x0204u, // LETTER | LOWERCASE
        0x0000B6u to 0x0020u, // PUNCTUATION
        0x0000B8u to 0x0040u, // SYMBOL
        0x0000B9u to 0x0002u, // NUMBER
        0x0000BAu to 0x0204u, // LETTER | LOWERCASE
        0x0000BBu to 0x0020u, // PUNCTUATION
        0x0000BCu to 0x0002u, // NUMBER
        0x0000BFu to 0x0020u, // PUNCTUATION
        0x0000C0u to 0x0404u, // LETTER | UPPERCASE
        0x110000u to 0x0001u, // sentinel – UNDEFINED
    )

    // =========================================================================
    // Whitespace code points
    // =========================================================================

    /**
     * Set of all Unicode code points classified as whitespace.
     *
     * This exactly mirrors `unicode_set_whitespace` from the C++ source
     * (25 entries).
     */
    val unicodeSetWhitespace: Set<UInt> = setOf(
        0x0009u, 0x000Au, 0x000Bu, 0x000Cu, 0x000Du,
        0x0020u, 0x0085u, 0x00A0u, 0x1680u,
        0x2000u, 0x2001u, 0x2002u, 0x2003u, 0x2004u,
        0x2005u, 0x2006u, 0x2007u, 0x2008u, 0x2009u,
        0x200Au, 0x2028u, 0x2029u, 0x202Fu, 0x205Fu,
        0x3000u,
    )

    // =========================================================================
    // Lowercase mapping   (upper → lower)
    // =========================================================================
    //
    // The full table has ~1 437 entries.  We include the first 50 for reference.
    //
    // At runtime, [LlamaUnicode.unicodeToLower] prefers Kotlin stdlib's
    // [Char.lowercaseChar] for BMP code points.

    /**
     * First 50 entries of `unicode_map_lowercase`.
     *
     * Each pair is `(upper, lower)` – the uppercase code point maps to its
     * lowercase equivalent.
     *
     * LATER: Include all ~1 437 entries for full coverage, or rely on Kotlin
     *       stdlib for BMP lookups and only supply supplementary-plane data.
     */
    val unicodeMapLowercase: List<Pair<UInt, UInt>> = listOf(
        0x0041u to 0x0061u, 0x0042u to 0x0062u, 0x0043u to 0x0063u,
        0x0044u to 0x0064u, 0x0045u to 0x0065u, 0x0046u to 0x0066u,
        0x0047u to 0x0067u, 0x0048u to 0x0068u, 0x0049u to 0x0069u,
        0x004Au to 0x006Au, 0x004Bu to 0x006Bu, 0x004Cu to 0x006Cu,
        0x004Du to 0x006Du, 0x004Eu to 0x006Eu, 0x004Fu to 0x006Fu,
        0x0050u to 0x0070u, 0x0051u to 0x0071u, 0x0052u to 0x0072u,
        0x0053u to 0x0073u, 0x0054u to 0x0074u, 0x0055u to 0x0075u,
        0x0056u to 0x0076u, 0x0057u to 0x0077u, 0x0058u to 0x0078u,
        0x0059u to 0x0079u, 0x005Au to 0x007Au,
        0x00C0u to 0x00E0u, 0x00C1u to 0x00E1u, 0x00C2u to 0x00E2u,
        0x00C3u to 0x00E3u, 0x00C4u to 0x00E4u, 0x00C5u to 0x00E5u,
        0x00C6u to 0x00E6u, 0x00C7u to 0x00E7u, 0x00C8u to 0x00E8u,
        0x00C9u to 0x00E9u, 0x00CAu to 0x00EAu, 0x00CBu to 0x00EBu,
        0x00CCu to 0x00ECu, 0x00CDu to 0x00EDu, 0x00CEu to 0x00EEu,
        0x00CFu to 0x00EFu, 0x00D0u to 0x00F0u, 0x00D1u to 0x00F1u,
        0x00D2u to 0x00F2u, 0x00D3u to 0x00F3u, 0x00D4u to 0x00F4u,
        0x00D5u to 0x00F5u, 0x00D6u to 0x00F6u, 0x00D8u to 0x00F8u,
        0x00D9u to 0x00F9u, 0x00DAu to 0x00FAu,
        // LATER: ~1 387 more entries from unicode-data.cpp
    )

    // =========================================================================
    // Uppercase mapping   (lower → upper)
    // =========================================================================

    /**
     * First 50 entries of `unicode_map_uppercase`.
     *
     * Each pair is `(lower, upper)`.
     *
     * LATER: Include all ~1 453 entries for full coverage.
     */
    val unicodeMapUppercase: List<Pair<UInt, UInt>> = listOf(
        0x0061u to 0x0041u, 0x0062u to 0x0042u, 0x0063u to 0x0043u,
        0x0064u to 0x0044u, 0x0065u to 0x0045u, 0x0066u to 0x0046u,
        0x0067u to 0x0047u, 0x0068u to 0x0048u, 0x0069u to 0x0049u,
        0x006Au to 0x004Au, 0x006Bu to 0x004Bu, 0x006Cu to 0x004Cu,
        0x006Du to 0x004Du, 0x006Eu to 0x004Eu, 0x006Fu to 0x004Fu,
        0x0070u to 0x0050u, 0x0071u to 0x0051u, 0x0072u to 0x0052u,
        0x0073u to 0x0053u, 0x0074u to 0x0054u, 0x0075u to 0x0055u,
        0x0076u to 0x0056u, 0x0077u to 0x0057u, 0x0078u to 0x0058u,
        0x0079u to 0x0059u, 0x007Au to 0x005Au,
        0x00E0u to 0x00C0u, 0x00E1u to 0x00C1u, 0x00E2u to 0x00C2u,
        0x00E3u to 0x00C3u, 0x00E4u to 0x00C4u, 0x00E5u to 0x00C5u,
        0x00E6u to 0x00C6u, 0x00E7u to 0x00C7u, 0x00E8u to 0x00C8u,
        0x00E9u to 0x00C9u, 0x00EAu to 0x00CAu, 0x00EBu to 0x00CBu,
        0x00ECu to 0x00CCu, 0x00EDu to 0x00CDu, 0x00EEu to 0x00CEu,
        0x00EFu to 0x00CFu, 0x00F0u to 0x00D0u, 0x00F1u to 0x00D1u,
        0x00F2u to 0x00D2u, 0x00F3u to 0x00D3u, 0x00F4u to 0x00D4u,
        0x00F5u to 0x00D5u, 0x00F6u to 0x00D6u, 0x00F8u to 0x00D8u,
        0x00F9u to 0x00D9u, 0x00FAu to 0x00DAu,
        // LATER: ~1 403 more entries from unicode-data.cpp
    )

    // =========================================================================
    // NFD normalization ranges
    // =========================================================================
    //
    // The full table has ~1 800 entries.  We include the first 50 for reference.

    /**
     * First 50 entries of `unicode_ranges_nfd`.
     *
     * When a code point falls within [RangeNfd.first]..[RangeNfd.last], its
     * NFD decomposition is [RangeNfd.nfd].
     *
     * LATER: Include all ~1 800 entries for full NFD normalization coverage.
     */
    val unicodeRangesNfd: List<RangeNfd> = listOf(
        RangeNfd(0x00C0u, 0x00C0u, 0x0041u),
        RangeNfd(0x00C1u, 0x00C1u, 0x0041u),
        RangeNfd(0x00C2u, 0x00C2u, 0x0041u),
        RangeNfd(0x00C3u, 0x00C3u, 0x0041u),
        RangeNfd(0x00C4u, 0x00C4u, 0x0041u),
        RangeNfd(0x00C5u, 0x00C5u, 0x0041u),
        RangeNfd(0x00C7u, 0x00C7u, 0x0043u),
        RangeNfd(0x00C8u, 0x00C8u, 0x0045u),
        RangeNfd(0x00C9u, 0x00C9u, 0x0045u),
        RangeNfd(0x00CAu, 0x00CAu, 0x0045u),
        RangeNfd(0x00CBu, 0x00CBu, 0x0045u),
        RangeNfd(0x00CCu, 0x00CCu, 0x0049u),
        RangeNfd(0x00CDu, 0x00CDu, 0x0049u),
        RangeNfd(0x00CEu, 0x00CEu, 0x0049u),
        RangeNfd(0x00CFu, 0x00CFu, 0x0049u),
        RangeNfd(0x00D1u, 0x00D1u, 0x004Eu),
        RangeNfd(0x00D2u, 0x00D2u, 0x004Fu),
        RangeNfd(0x00D3u, 0x00D3u, 0x004Fu),
        RangeNfd(0x00D4u, 0x00D4u, 0x004Fu),
        RangeNfd(0x00D5u, 0x00D5u, 0x004Fu),
        RangeNfd(0x00D6u, 0x00D6u, 0x004Fu),
        RangeNfd(0x00D9u, 0x00D9u, 0x0055u),
        RangeNfd(0x00DAu, 0x00DAu, 0x0055u),
        RangeNfd(0x00DBu, 0x00DBu, 0x0055u),
        RangeNfd(0x00DCu, 0x00DCu, 0x0055u),
        RangeNfd(0x00DDu, 0x00DDu, 0x0059u),
        RangeNfd(0x00E0u, 0x00E0u, 0x0061u),
        RangeNfd(0x00E1u, 0x00E1u, 0x0061u),
        RangeNfd(0x00E2u, 0x00E2u, 0x0061u),
        RangeNfd(0x00E3u, 0x00E3u, 0x0061u),
        RangeNfd(0x00E4u, 0x00E4u, 0x0061u),
        RangeNfd(0x00E5u, 0x00E5u, 0x0061u),
        RangeNfd(0x00E7u, 0x00E7u, 0x0063u),
        RangeNfd(0x00E8u, 0x00E8u, 0x0065u),
        RangeNfd(0x00E9u, 0x00E9u, 0x0065u),
        RangeNfd(0x00EAu, 0x00EAu, 0x0065u),
        RangeNfd(0x00EBu, 0x00EBu, 0x0065u),
        RangeNfd(0x00ECu, 0x00ECu, 0x0069u),
        RangeNfd(0x00EDu, 0x00EDu, 0x0069u),
        RangeNfd(0x00EEu, 0x00EEu, 0x0069u),
        RangeNfd(0x00EFu, 0x00EFu, 0x0069u),
        RangeNfd(0x00F1u, 0x00F1u, 0x006Eu),
        RangeNfd(0x00F2u, 0x00F2u, 0x006Fu),
        RangeNfd(0x00F3u, 0x00F3u, 0x006Fu),
        RangeNfd(0x00F4u, 0x00F4u, 0x006Fu),
        RangeNfd(0x00F5u, 0x00F5u, 0x006Fu),
        RangeNfd(0x00F6u, 0x00F6u, 0x006Fu),
        RangeNfd(0x00F9u, 0x00F9u, 0x0075u),
        RangeNfd(0x00FAu, 0x00FAu, 0x0075u),
        RangeNfd(0x00FBu, 0x00FBu, 0x0075u),
        // LATER: ~1 750 more entries from unicode-data.cpp
    )
}
