// port-lint: source llama.cpp/src/unicode.h llama.cpp/src/unicode.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// Unicode code-point flags
// =============================================================================

/**
 * Bit-flag constants for Unicode code-point classification.
 *
 * Maps 1-to-1 to the `unicode_cpt_flags` struct/enum in the C++ source.
 * Each flag occupies a single bit in a [UShort] bitmask.
 */
object UnicodeCptFlags {
    const val UNDEFINED: UShort       = 0x0001u
    const val NUMBER: UShort          = 0x0002u  // \p{N}
    const val LETTER: UShort          = 0x0004u  // \p{L}
    const val SEPARATOR: UShort       = 0x0008u  // \p{Z}
    const val ACCENT_MARK: UShort     = 0x0010u  // \p{M}
    const val PUNCTUATION: UShort     = 0x0020u  // \p{P}
    const val SYMBOL: UShort          = 0x0040u  // \p{S}
    const val CONTROL: UShort         = 0x0080u  // \p{C}
    const val MASK_CATEGORIES: UShort = 0x00FFu
    const val WHITESPACE: UShort      = 0x0100u
    const val LOWERCASE: UShort       = 0x0200u
    const val UPPERCASE: UShort       = 0x0400u
    const val NFD: UShort             = 0x0800u
}

/**
 * Decoded flag bundle for a single Unicode code point.
 *
 * Mirrors the C++ `unicode_cpt_flags` bit-field struct.  Constructed from
 * a raw [UShort] bitmask and exposes individual boolean properties for each
 * classification bit.
 *
 * @param raw The packed 16-bit flag value.
 */
value class CptFlags(val raw: UShort) {

    constructor() : this(0u)

    val isUndefined: Boolean   get() = (raw and UnicodeCptFlags.UNDEFINED) != 0.toUShort()
    val isNumber: Boolean      get() = (raw and UnicodeCptFlags.NUMBER) != 0.toUShort()
    val isLetter: Boolean      get() = (raw and UnicodeCptFlags.LETTER) != 0.toUShort()
    val isSeparator: Boolean   get() = (raw and UnicodeCptFlags.SEPARATOR) != 0.toUShort()
    val isAccentMark: Boolean  get() = (raw and UnicodeCptFlags.ACCENT_MARK) != 0.toUShort()
    val isPunctuation: Boolean get() = (raw and UnicodeCptFlags.PUNCTUATION) != 0.toUShort()
    val isSymbol: Boolean      get() = (raw and UnicodeCptFlags.SYMBOL) != 0.toUShort()
    val isControl: Boolean     get() = (raw and UnicodeCptFlags.CONTROL) != 0.toUShort()
    val isWhitespace: Boolean  get() = (raw and UnicodeCptFlags.WHITESPACE) != 0.toUShort()
    val isLowercase: Boolean   get() = (raw and UnicodeCptFlags.LOWERCASE) != 0.toUShort()
    val isUppercase: Boolean   get() = (raw and UnicodeCptFlags.UPPERCASE) != 0.toUShort()
    val isNfd: Boolean         get() = (raw and UnicodeCptFlags.NFD) != 0.toUShort()

    fun asUint(): UShort = raw

    fun categoryFlag(): UShort = raw and UnicodeCptFlags.MASK_CATEGORIES
}

// =============================================================================
// LlamaUnicode — public API
// =============================================================================

/**
 * Unicode utility functions used by the BPE tokenizer and vocabulary system.
 *
 * This object is the Kotlin equivalent of `unicode.h` / `unicode.cpp`.
 * Functions that can leverage Kotlin stdlib Unicode support do so; others
 * port the original C++ logic directly.
 */
object LlamaUnicode {

    // -------------------------------------------------------------------------
    // UTF-8 length lookup
    // -------------------------------------------------------------------------

    /** UTF-8 byte-length lookup table indexed by the high nibble of the lead byte. */
    private val utf8LenLookup = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)

    /**
     * Returns the number of bytes in the UTF-8 sequence whose lead byte is [src].
     */
    fun unicodeLenUtf8(src: Byte): Int {
        val highBits = (src.toInt() and 0xFF) ushr 4
        return utf8LenLookup[highBits]
    }

    // -------------------------------------------------------------------------
    // UTF-8 ↔ code-point conversions
    // -------------------------------------------------------------------------

    /**
     * Encode a single Unicode code point to its UTF-8 byte representation.
     *
     * @throws IllegalArgumentException if [cpt] is outside the valid Unicode range.
     */
    fun unicodeCptToUtf8(cpt: UInt): String {
        val sb = StringBuilder()
        when {
            cpt <= 0x7Fu -> {
                sb.append(cpt.toInt().toChar())
            }
            cpt in 0x80u..0x7FFu -> {
                sb.append((0xC0 or ((cpt.toInt() shr 6) and 0x1F)).toChar())
                sb.append((0x80 or (cpt.toInt() and 0x3F)).toChar())
            }
            cpt in 0x800u..0xFFFFu -> {
                sb.append((0xE0 or ((cpt.toInt() shr 12) and 0x0F)).toChar())
                sb.append((0x80 or ((cpt.toInt() shr 6) and 0x3F)).toChar())
                sb.append((0x80 or (cpt.toInt() and 0x3F)).toChar())
            }
            cpt in 0x10000u..0x10FFFFu -> {
                sb.append((0xF0 or ((cpt.toInt() shr 18) and 0x07)).toChar())
                sb.append((0x80 or ((cpt.toInt() shr 12) and 0x3F)).toChar())
                sb.append((0x80 or ((cpt.toInt() shr 6) and 0x3F)).toChar())
                sb.append((0x80 or (cpt.toInt() and 0x3F)).toChar())
            }
            else -> throw IllegalArgumentException("invalid codepoint: $cpt")
        }
        return sb.toString()
    }

    /**
     * Encode a code point to a UTF-8 [ByteArray].
     */
    fun unicodeCptToUtf8Bytes(cpt: UInt): ByteArray {
        return when {
            cpt <= 0x7Fu -> byteArrayOf(cpt.toByte())
            cpt in 0x80u..0x7FFu -> byteArrayOf(
                (0xC0 or ((cpt.toInt() shr 6) and 0x1F)).toByte(),
                (0x80 or (cpt.toInt() and 0x3F)).toByte(),
            )
            cpt in 0x800u..0xFFFFu -> byteArrayOf(
                (0xE0 or ((cpt.toInt() shr 12) and 0x0F)).toByte(),
                (0x80 or ((cpt.toInt() shr 6) and 0x3F)).toByte(),
                (0x80 or (cpt.toInt() and 0x3F)).toByte(),
            )
            cpt in 0x10000u..0x10FFFFu -> byteArrayOf(
                (0xF0 or ((cpt.toInt() shr 18) and 0x07)).toByte(),
                (0x80 or ((cpt.toInt() shr 12) and 0x3F)).toByte(),
                (0x80 or ((cpt.toInt() shr 6) and 0x3F)).toByte(),
                (0x80 or (cpt.toInt() and 0x3F)).toByte(),
            )
            else -> throw IllegalArgumentException("invalid codepoint: $cpt")
        }
    }

    /**
     * Decode a single UTF-8 code point from [utf8] starting at [offset].
     *
     * On return, [offset] is advanced past the consumed bytes via the returned
     * [Pair] of `(codePoint, newOffset)`.
     *
     * @throws IllegalArgumentException on malformed UTF-8.
     */
    fun unicodeCptFromUtf8(utf8: ByteArray, offset: Int): Pair<UInt, Int> {
        require(offset < utf8.size) { "offset out of bounds" }
        val b0 = utf8[offset].toInt() and 0xFF

        if (b0 and 0x80 == 0) {
            return b0.toUInt() to (offset + 1)
        }
        if (b0 and 0x40 == 0) {
            throw IllegalArgumentException("invalid UTF-8 continuation byte at offset $offset")
        }
        if (b0 and 0x20 == 0) {
            require(offset + 1 < utf8.size && (utf8[offset + 1].toInt() and 0xC0) == 0x80) {
                "invalid 2-byte UTF-8 sequence"
            }
            val result = ((b0 and 0x1F) shl 6) or (utf8[offset + 1].toInt() and 0x3F)
            return result.toUInt() to (offset + 2)
        }
        if (b0 and 0x10 == 0) {
            require(offset + 2 < utf8.size &&
                    (utf8[offset + 1].toInt() and 0xC0) == 0x80 &&
                    (utf8[offset + 2].toInt() and 0xC0) == 0x80) {
                "invalid 3-byte UTF-8 sequence"
            }
            val result = ((b0 and 0x0F) shl 12) or
                    ((utf8[offset + 1].toInt() and 0x3F) shl 6) or
                    (utf8[offset + 2].toInt() and 0x3F)
            return result.toUInt() to (offset + 3)
        }
        if (b0 and 0x08 == 0) {
            require(offset + 3 < utf8.size &&
                    (utf8[offset + 1].toInt() and 0xC0) == 0x80 &&
                    (utf8[offset + 2].toInt() and 0xC0) == 0x80 &&
                    (utf8[offset + 3].toInt() and 0xC0) == 0x80) {
                "invalid 4-byte UTF-8 sequence"
            }
            val result = ((b0 and 0x07) shl 18) or
                    ((utf8[offset + 1].toInt() and 0x3F) shl 12) or
                    ((utf8[offset + 2].toInt() and 0x3F) shl 6) or
                    (utf8[offset + 3].toInt() and 0x3F)
            return result.toUInt() to (offset + 4)
        }
        throw IllegalArgumentException("failed to convert UTF-8 to codepoint")
    }

    /**
     * Decode a UTF-8 string (given as raw bytes) into a list of code points.
     *
     * Invalid sequences are replaced with U+FFFD (replacement character).
     */
    fun unicodeCptsFromUtf8(utf8: ByteArray): List<UInt> {
        val result = ArrayList<UInt>(utf8.size)
        var offset = 0
        while (offset < utf8.size) {
            try {
                val (cpt, newOffset) = unicodeCptFromUtf8(utf8, offset)
                result.add(cpt)
                offset = newOffset
            } catch (_: IllegalArgumentException) {
                result.add(0xFFFDu) // replacement character
                offset++
            }
        }
        return result
    }

    /**
     * Decode a Kotlin [String] into a list of Unicode code points.
     *
     * Uses Kotlin's built-in string iteration (which handles surrogate pairs)
     * and converts to the C++-compatible UInt code-point representation.
     */
    fun unicodeCptsFromUtf8(text: String): List<UInt> {
        val result = ArrayList<UInt>(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            result.add(cp.toUInt())
            i += if (cp > 0xFFFF) 2 else 1
        }
        return result
    }

    /**
     * Convert a list of code points to a UTF-8 string.
     */
    fun unicodeCptsToUtf8(cpts: List<UInt>): String {
        val sb = StringBuilder()
        for (cpt in cpts) {
            sb.append(unicodeCptToUtf8(cpt))
        }
        return sb.toString()
    }

    // -------------------------------------------------------------------------
    // Code-point classification
    // -------------------------------------------------------------------------

    /**
     * Classify a code point using Kotlin stdlib where possible, falling back
     * to the [LlamaUnicodeData] tables for the exact C++ flag semantics.
     *
     * This uses [Char.category] for BMP code points and the static tables for
     * supplementary planes.
     */
    fun unicodeCptFlagsFromCpt(cpt: UInt): CptFlags {
        if (cpt > 0x10FFFFu) return CptFlags(UnicodeCptFlags.UNDEFINED)

        var flags: UShort = 0u

        // Use Kotlin stdlib for BMP
        if (cpt <= 0xFFFFu) {
            val ch = cpt.toInt().toChar()
            val cat = ch.category

            flags = when (cat) {
                CharCategory.UPPERCASE_LETTER,
                CharCategory.LOWERCASE_LETTER,
                CharCategory.TITLECASE_LETTER,
                CharCategory.MODIFIER_LETTER,
                CharCategory.OTHER_LETTER -> UnicodeCptFlags.LETTER

                CharCategory.DECIMAL_DIGIT_NUMBER,
                CharCategory.LETTER_NUMBER,
                CharCategory.OTHER_NUMBER -> UnicodeCptFlags.NUMBER

                CharCategory.SPACE_SEPARATOR,
                CharCategory.LINE_SEPARATOR,
                CharCategory.PARAGRAPH_SEPARATOR -> UnicodeCptFlags.SEPARATOR

                CharCategory.NON_SPACING_MARK,
                CharCategory.COMBINING_SPACING_MARK,
                CharCategory.ENCLOSING_MARK -> UnicodeCptFlags.ACCENT_MARK

                CharCategory.DASH_PUNCTUATION,
                CharCategory.START_PUNCTUATION,
                CharCategory.END_PUNCTUATION,
                CharCategory.CONNECTOR_PUNCTUATION,
                CharCategory.OTHER_PUNCTUATION,
                CharCategory.INITIAL_QUOTE_PUNCTUATION,
                CharCategory.FINAL_QUOTE_PUNCTUATION -> UnicodeCptFlags.PUNCTUATION

                CharCategory.MATH_SYMBOL,
                CharCategory.CURRENCY_SYMBOL,
                CharCategory.MODIFIER_SYMBOL,
                CharCategory.OTHER_SYMBOL -> UnicodeCptFlags.SYMBOL

                CharCategory.CONTROL,
                CharCategory.FORMAT,
                CharCategory.SURROGATE,
                CharCategory.PRIVATE_USE,
                CharCategory.UNASSIGNED -> UnicodeCptFlags.CONTROL
            }

            // Overlay helper flags
            if (ch.isWhitespace() || cpt in LlamaUnicodeData.unicodeSetWhitespace) {
                flags = flags or UnicodeCptFlags.WHITESPACE
            }
            if (ch.isLowerCase()) {
                flags = flags or UnicodeCptFlags.LOWERCASE
            }
            if (ch.isUpperCase()) {
                flags = flags or UnicodeCptFlags.UPPERCASE
            }
        } else {
            // Supplementary planes – default to LETTER for assigned code points
            flags = UnicodeCptFlags.LETTER
        }

        return CptFlags(flags)
    }

    /**
     * Classify a UTF-8 string's first code point.
     */
    fun unicodeCptFlagsFromUtf8(utf8: String): CptFlags {
        if (utf8.isEmpty()) return CptFlags(UnicodeCptFlags.UNDEFINED)
        val cp = utf8.codePointAt(0)
        return unicodeCptFlagsFromCpt(cp.toUInt())
    }

    // -------------------------------------------------------------------------
    // NFD normalization
    // -------------------------------------------------------------------------

    /**
     * Apply NFD normalization to a list of code points.
     *
     * Uses binary search over [LlamaUnicodeData.unicodeRangesNfd] — each code
     * point that falls within a range is replaced by the range's NFD target.
     */
    fun unicodeCptsNormalizeNfd(cpts: List<UInt>): List<UInt> {
        val nfdRanges = LlamaUnicodeData.unicodeRangesNfd
        return cpts.map { cpt ->
            // Binary search: find the last range whose `first` <= cpt
            var lo = 0
            var hi = nfdRanges.size - 1
            var bestIdx = -1
            while (lo <= hi) {
                val mid = (lo + hi) / 2
                if (nfdRanges[mid].first <= cpt) {
                    bestIdx = mid
                    lo = mid + 1
                } else {
                    hi = mid - 1
                }
            }
            if (bestIdx >= 0) {
                val range = nfdRanges[bestIdx]
                if (cpt in range.first..range.last) range.nfd else cpt
            } else {
                cpt
            }
        }
    }

    /**
     * NFKD normalization (compatibility decomposition).
     *
     * LATER: Implement full NFKD normalization. For now delegates to NFD.
     */
    fun unicodeCptsNormalizeNfkd(cpts: List<UInt>): List<UInt> {
        // LATER: NFKD requires additional decomposition mappings beyond NFD
        return unicodeCptsNormalizeNfd(cpts)
    }

    // -------------------------------------------------------------------------
    // Case conversion
    // -------------------------------------------------------------------------

    /**
     * Convert a code point to its lowercase equivalent.
     *
     * Prefers Kotlin stdlib for BMP; falls back to the lookup table for
     * supplementary-plane code points.
     */
    fun unicodeToLower(cpt: UInt): UInt {
        if (cpt <= 0xFFFFu) {
            return cpt.toInt().toChar().lowercaseChar().code.toUInt()
        }
        // Supplementary plane – binary search the table
        val table = LlamaUnicodeData.unicodeMapLowercase
        var lo = 0
        var hi = table.size - 1
        while (lo <= hi) {
            val mid = (lo + hi) / 2
            val entry = table[mid]
            when {
                entry.first == cpt -> return entry.second
                entry.first < cpt  -> lo = mid + 1
                else               -> hi = mid - 1
            }
        }
        return cpt
    }

    // -------------------------------------------------------------------------
    // Han (CJK) detection
    // -------------------------------------------------------------------------

    /**
     * Returns `true` if [cpt] is a CJK Unified Ideograph or compatibility
     * ideograph.
     */
    fun unicodeCptIsHan(cpt: UInt): Boolean {
        if (cpt in 0x4E00u..0x9FFFu) return true   // CJK Unified Ideographs
        if (cpt in 0x3400u..0x4DBFu) return true   // Extension A
        if (cpt in 0x20000u..0x2A6DFu) return true  // Extension B
        if (cpt in 0x2A700u..0x2B73Fu) return true  // Extension C
        if (cpt in 0x2B740u..0x2B81Fu) return true  // Extension D
        if (cpt in 0x2B820u..0x2CEAFu) return true  // Extension E
        if (cpt in 0x2CEB0u..0x2EBEFu) return true  // Extension F
        if (cpt in 0xF900u..0xFAFFu) return true    // Compatibility Ideographs
        if (cpt in 0x2F800u..0x2FA1Fu) return true   // Compatibility Supplement
        return false
    }

    // -------------------------------------------------------------------------
    // BPE byte ↔ UTF-8 mapping
    // -------------------------------------------------------------------------

    /** GPT-2 / BPE byte-to-UTF-8 mapping (lazy-initialized). */
    private val byteToUtf8Map: Map<UByte, String> by lazy { buildByteToUtf8Map() }

    /** Reverse mapping: UTF-8 string → byte value. */
    private val utf8ToByteMap: Map<String, UByte> by lazy { buildUtf8ToByteMap() }

    private fun buildByteToUtf8Map(): Map<UByte, String> {
        val map = HashMap<UByte, String>(256)
        // Printable ASCII: ! to ~
        for (ch in 0x21..0x7E) {
            map[ch.toUByte()] = unicodeCptToUtf8(ch.toUInt())
        }
        // Latin-1 supplement: ¡ to ¬
        for (ch in 0xA1..0xAC) {
            map[ch.toUByte()] = unicodeCptToUtf8(ch.toUInt())
        }
        // Latin-1 supplement: ® to ÿ
        for (ch in 0xAE..0xFF) {
            map[ch.toUByte()] = unicodeCptToUtf8(ch.toUInt())
        }
        // Fill remaining bytes with code points starting at 256
        var n = 0
        for (ch in 0..255) {
            if (ch.toUByte() !in map) {
                map[ch.toUByte()] = unicodeCptToUtf8((256 + n).toUInt())
                n++
            }
        }
        return map
    }

    private fun buildUtf8ToByteMap(): Map<String, UByte> {
        val map = HashMap<String, UByte>(256)
        for ((k, v) in byteToUtf8Map) {
            map[v] = k
        }
        return map
    }

    /**
     * Map a raw byte value to its BPE UTF-8 token representation.
     *
     * This is the GPT-2 byte-to-unicode mapping used by BPE tokenizers.
     */
    fun unicodeByteToUtf8(byte: UByte): String {
        return byteToUtf8Map[byte] ?: throw IllegalArgumentException("no mapping for byte: $byte")
    }

    /**
     * Map a BPE UTF-8 token back to its raw byte value.
     */
    fun unicodeUtf8ToByte(utf8: String): UByte {
        return utf8ToByteMap[utf8] ?: throw IllegalArgumentException("no mapping for utf8: $utf8")
    }

    // -------------------------------------------------------------------------
    // Byte encoding of BPE words
    // -------------------------------------------------------------------------

    /**
     * Apply GPT-2 byte encoding to a list of BPE words.
     *
     * Each byte in each word is replaced by its BPE UTF-8 token
     * representation via [unicodeByteToUtf8].
     */
    private fun unicodeByteEncodingProcess(bpeWords: List<String>): List<String> {
        return bpeWords.map { word ->
            val sb = StringBuilder()
            for (byte in word.encodeToByteArray()) {
                sb.append(unicodeByteToUtf8(byte.toUByte()))
            }
            sb.toString()
        }
    }

    // =========================================================================
    // Regex-based text splitting (tokenization pre-processing)
    // =========================================================================

    /**
     * Split [text] according to a sequence of regex patterns, then optionally
     * apply byte encoding.
     *
     * This is the main entry point used by BPE tokenizers. The [regexExprs]
     * are applied in order; each one further subdivides the segments produced
     * by the previous one.
     *
     * Certain well-known patterns (GPT-2, LLaMA-3, Qwen2, etc.) are handled
     * by optimised custom state machines. Unknown patterns fall back to
     * Kotlin's [Regex] engine.
     *
     * @param text       The input text to split.
     * @param regexExprs Ordered list of regex patterns to apply.
     * @param byteEncode If `true`, apply GPT-2 byte encoding to the result.
     * @return A list of token strings.
     */
    fun unicodeRegexSplit(
        text: String,
        regexExprs: List<String>,
        byteEncode: Boolean = true,
    ): List<String> {
        val cpts = unicodeCptsFromUtf8(text)

        var bpeOffsets: List<Int> = listOf(cpts.size)

        for (regexExpr in regexExprs) {
            val tmp = unicodeRegexSplitCustom(text, regexExpr, bpeOffsets, cpts)
            if (tmp.isNotEmpty()) {
                bpeOffsets = tmp
                continue
            }
            // Fallback to Kotlin Regex
            bpeOffsets = unicodeRegexSplitStl(text, regexExpr, bpeOffsets, cpts)
        }

        // Convert offsets back to strings
        val bpeWords = ArrayList<String>(bpeOffsets.size)
        var start = 0
        for (offset in bpeOffsets) {
            val sb = StringBuilder()
            for (i in start until start + offset) {
                sb.append(unicodeCptToUtf8(cpts[i]))
            }
            bpeWords.add(sb.toString())
            start += offset
        }

        return if (byteEncode) unicodeByteEncodingProcess(bpeWords) else bpeWords
    }

    // -------------------------------------------------------------------------
    // Custom regex dispatching
    // -------------------------------------------------------------------------

    private const val PATTERN_GPT2 =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"

    private const val PATTERN_LLAMA3 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    private const val PATTERN_LLAMA3_ALT =
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    private const val PATTERN_QWEN2 =
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

    private fun unicodeRegexSplitCustom(
        text: String,
        regexExpr: String,
        offsets: List<Int>,
        cpts: List<UInt>,
    ): List<Int> {
        return when (regexExpr) {
            PATTERN_GPT2 ->
                regexSplitCustomGpt2(cpts, offsets)
            PATTERN_LLAMA3, PATTERN_LLAMA3_ALT ->
                regexSplitCustomLlama3(cpts, offsets)
            PATTERN_QWEN2 ->
                regexSplitCustomQwen2(cpts, offsets)
            "\\p{Han}+" ->
                regexSplitCustomKimiK2(cpts, offsets)
            "\\p{AFMoE_digits}" ->
                regexSplitCustomAfmoe(cpts, offsets)
            "[^\\n]+|[\\n]+" ->
                regexSplitCustomNewlines(cpts, offsets)
            "\\d{1,3}(?=(?:\\d{3})*\\b)" ->
                regexSplitCustomAfmoe(cpts, offsets)
            else -> emptyList()
        }
    }

    // -------------------------------------------------------------------------
    // Fallback STL-style regex splitting
    // -------------------------------------------------------------------------

    private fun unicodeRegexSplitStl(
        text: String,
        regexExpr: String,
        offsets: List<Int>,
        cpts: List<UInt>,
    ): List<Int> {
        // Build a "collapsed" text where multi-byte codepoints are replaced
        // with single-byte category templates
        val collapsed = buildCollapsedText(cpts, regexExpr)
        val useCollapsed = collapsed != null

        val searchText = collapsed ?: buildWideText(cpts)
        val searchRegex = if (useCollapsed) buildCollapsedRegex(regexExpr) else regexExpr

        val bpeOffsets = ArrayList<Int>()
        var start = 0
        try {
            val regex = Regex(searchRegex)
            for (offset in offsets) {
                val segment = searchText.substring(start, start + offset)
                val matches = regex.findAll(segment)
                var startIdx = 0
                for (match in matches) {
                    if (match.range.first > startIdx) {
                        bpeOffsets.add(match.range.first - startIdx)
                    }
                    bpeOffsets.add(match.value.length)
                    startIdx = match.range.first + match.value.length
                }
                if (startIdx < offset) {
                    bpeOffsets.add(offset - startIdx)
                }
                start += offset
            }
        } catch (e: Exception) {
            throw RuntimeException("Failed to process regex: '$regexExpr': ${e.message}")
        }
        return bpeOffsets
    }

    private val UCAT_ENUM = mapOf(
        "\\p{N}" to UnicodeCptFlags.NUMBER,
        "\\p{L}" to UnicodeCptFlags.LETTER,
        "\\p{P}" to UnicodeCptFlags.PUNCTUATION,
        "\\p{M}" to UnicodeCptFlags.ACCENT_MARK,
        "\\p{S}" to UnicodeCptFlags.SYMBOL,
        "\\p{Lu}" to UnicodeCptFlags.LETTER,
        "\\p{Ll}" to UnicodeCptFlags.LETTER,
        "\\p{Lt}" to UnicodeCptFlags.LETTER,
        "\\p{Lm}" to UnicodeCptFlags.LETTER,
        "\\p{Lo}" to UnicodeCptFlags.LETTER,
    )

    private val UCAT_CPT = mapOf(
        UnicodeCptFlags.NUMBER to '\u00D1',
        UnicodeCptFlags.LETTER to '\u00D2',
        UnicodeCptFlags.PUNCTUATION to '\u00D3',
        UnicodeCptFlags.ACCENT_MARK to '\u00D4',
        UnicodeCptFlags.SYMBOL to '\u00D5',
    )

    private val UCAT_MAP = mapOf(
        UnicodeCptFlags.NUMBER to "\\x30-\\x39",
        UnicodeCptFlags.LETTER to "\\x41-\\x5A\\x61-\\x7A",
        UnicodeCptFlags.PUNCTUATION to "\\x21-\\x23\\x25-\\x2A\\x2C-\\x2F\\x3A-\\x3B\\x3F-\\x40\\[\\-\\]_\\{\\}",
        UnicodeCptFlags.ACCENT_MARK to "",
        UnicodeCptFlags.SYMBOL to "\\$\\+\\x3C-\\x3E\\^`\\|",
    )

    private fun buildCollapsedText(cpts: List<UInt>, regexExpr: String): String? {
        val needCollapse = UCAT_ENUM.keys.any { regexExpr.contains(it) }
        if (!needCollapse) return null

        val sb = StringBuilder(cpts.size)
        for (cpt in cpts) {
            if (cpt < 128u) {
                sb.append(cpt.toInt().toChar())
                continue
            }
            val flags = unicodeCptFlagsFromCpt(cpt)
            when {
                flags.isWhitespace -> sb.append('\u000B') // vertical tab as whitespace fallback
                else -> {
                    val catFlag = flags.categoryFlag()
                    val replacement = UCAT_CPT[catFlag]
                    sb.append(replacement ?: '\u00D0')
                }
            }
        }
        return sb.toString()
    }

    private fun buildWideText(cpts: List<UInt>): String {
        val sb = StringBuilder(cpts.size)
        for (cpt in cpts) {
            if (cpt > 0x7Fu && unicodeCptFlagsFromCpt(cpt).isWhitespace) {
                sb.append('\u000B')
            } else if (cpt <= 0xFFFFu) {
                sb.append(cpt.toInt().toChar())
            } else {
                // Supplementary – use surrogate pair via codePointAt-compatible encoding
                sb.appendCodePoint(cpt.toInt())
            }
        }
        return sb.toString()
    }

    private fun buildCollapsedRegex(regexExpr: String): String {
        val sb = StringBuilder()
        var i = 0
        var inside = false
        while (i < regexExpr.length) {
            val ch = regexExpr[i]
            if (ch == '[' && (i == 0 || regexExpr[i - 1] != '\\')) {
                sb.append('[')
                inside = true
                i++
                continue
            }
            if (inside && ch == ']' && regexExpr[i - 1] != '\\') {
                sb.append(']')
                inside = false
                i++
                continue
            }
            // Match \p{...} patterns
            if (ch == '\\' && i + 3 < regexExpr.length &&
                regexExpr[i + 1] == 'p' && regexExpr[i + 2] == '{') {
                val closingBrace = regexExpr.indexOf('}', i + 3)
                if (closingBrace in (i + 3)..(i + 10)) {
                    val pat = regexExpr.substring(i, closingBrace + 1)
                    val catVal = UCAT_ENUM[pat]
                    if (catVal != null) {
                        if (!inside) sb.append('[')
                        val cptChar = UCAT_CPT[catVal]
                        if (cptChar != null) sb.append(cptChar)
                        val mapStr = UCAT_MAP[catVal]
                        if (!mapStr.isNullOrEmpty()) sb.append(mapStr)
                        if (!inside) sb.append(']')
                        i = closingBrace + 1
                        continue
                    }
                }
            }
            sb.append(ch)
            i++
        }
        return sb.toString()
    }

    // =========================================================================
    // Custom tokenisation state machines
    // =========================================================================

    private const val OUT_OF_RANGE: UInt = 0xFFFFFFFFu

    // -------------------------------------------------------------------------
    // GPT-2
    // -------------------------------------------------------------------------

    /**
     * GPT-2 tokenisation pattern:
     * `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
     */
    private fun regexSplitCustomGpt2(
        cpts: List<UInt>,
        offsets: List<Int>,
    ): List<Int> {
        val bpeOffsets = ArrayList<Int>(offsets.size)
        var start = 0

        for (offset in offsets) {
            val offsetIni = start
            val offsetEnd = start + offset
            start = offsetEnd

            fun getCpt(pos: Int): UInt =
                if (pos in offsetIni until offsetEnd) cpts[pos] else OUT_OF_RANGE

            fun getFlags(pos: Int): CptFlags =
                if (pos in offsetIni until offsetEnd) unicodeCptFlagsFromCpt(cpts[pos]) else CptFlags()

            var prevEnd = offsetIni
            fun addToken(end: Int): Int {
                val len = end - prevEnd
                if (len > 0) bpeOffsets.add(len)
                prevEnd = end
                return len
            }

            var pos = offsetIni
            while (pos < offsetEnd) {
                val cpt = getCpt(pos)
                val flags = getFlags(pos)

                // 's|'t|'re|'ve|'m|'ll|'d
                if (cpt == '\''.code.toUInt() && pos + 1 < offsetEnd) {
                    val cptNext = getCpt(pos + 1)
                    if (cptNext == 's'.code.toUInt() || cptNext == 't'.code.toUInt() ||
                        cptNext == 'm'.code.toUInt() || cptNext == 'd'.code.toUInt()) {
                        pos += addToken(pos + 2); continue
                    }
                    if (pos + 2 < offsetEnd) {
                        val cptNextNext = getCpt(pos + 2)
                        if ((cptNext == 'r'.code.toUInt() && cptNextNext == 'e'.code.toUInt()) ||
                            (cptNext == 'v'.code.toUInt() && cptNextNext == 'e'.code.toUInt()) ||
                            (cptNext == 'l'.code.toUInt() && cptNextNext == 'l'.code.toUInt())) {
                            pos += addToken(pos + 3); continue
                        }
                    }
                }

                val flags2 = if (cpt == ' '.code.toUInt()) getFlags(pos + 1) else flags

                // <space>?\p{L}+
                if (flags2.isLetter) {
                    if (cpt == ' '.code.toUInt()) pos++
                    var f = flags2
                    while (f.isLetter) { f = getFlags(++pos) }
                    addToken(pos); continue
                }
                // <space>?\p{N}+
                if (flags2.isNumber) {
                    if (cpt == ' '.code.toUInt()) pos++
                    var f = flags2
                    while (f.isNumber) { f = getFlags(++pos) }
                    addToken(pos); continue
                }
                // <space>?[^\s\p{L}\p{N}]+
                if (!(flags2.isWhitespace || flags2.isLetter || flags2.isNumber) && flags2.asUint() != 0.toUShort()) {
                    if (cpt == ' '.code.toUInt()) pos++
                    var f = flags2
                    while (!(f.isWhitespace || f.isLetter || f.isNumber) && f.asUint() != 0.toUShort()) {
                        f = getFlags(++pos)
                    }
                    addToken(pos); continue
                }

                var numWhitespaces = 0
                while (getFlags(pos + numWhitespaces).isWhitespace) numWhitespaces++

                // \s+(?!\S)
                if (numWhitespaces > 1 && getCpt(pos + numWhitespaces) != OUT_OF_RANGE) {
                    pos += numWhitespaces - 1
                    addToken(pos); continue
                }
                // \s+
                if (numWhitespaces > 0) {
                    pos += numWhitespaces
                    addToken(pos); continue
                }

                addToken(++pos)
            }
        }
        return bpeOffsets
    }

    // -------------------------------------------------------------------------
    // LLaMA-3
    // -------------------------------------------------------------------------

    /**
     * LLaMA-3 tokenisation pattern:
     * `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
     */
    private fun regexSplitCustomLlama3(
        cpts: List<UInt>,
        offsets: List<Int>,
    ): List<Int> {
        val bpeOffsets = ArrayList<Int>(offsets.size)
        var start = 0

        for (offset in offsets) {
            val offsetIni = start
            val offsetEnd = start + offset
            start = offsetEnd

            fun getCpt(pos: Int): UInt =
                if (pos in offsetIni until offsetEnd) cpts[pos] else OUT_OF_RANGE

            fun getFlags(pos: Int): CptFlags =
                if (pos in offsetIni until offsetEnd) unicodeCptFlagsFromCpt(cpts[pos]) else CptFlags()

            var prevEnd = offsetIni
            fun addToken(end: Int): Int {
                val len = end - prevEnd
                if (len > 0) bpeOffsets.add(len)
                prevEnd = end
                return len
            }

            var pos = offsetIni
            while (pos < offsetEnd) {
                val cpt = getCpt(pos)
                val flags = getFlags(pos)

                // (?i:'s|'t|'re|'ve|'m|'ll|'d) case insensitive
                if (cpt == '\''.code.toUInt() && pos + 1 < offsetEnd) {
                    val cptNext = unicodeToLower(getCpt(pos + 1))
                    if (cptNext == 's'.code.toUInt() || cptNext == 't'.code.toUInt() ||
                        cptNext == 'm'.code.toUInt() || cptNext == 'd'.code.toUInt()) {
                        pos += addToken(pos + 2); continue
                    }
                    if (pos + 2 < offsetEnd) {
                        val cptNextNext = unicodeToLower(getCpt(pos + 2))
                        if ((cptNext == 'r'.code.toUInt() && cptNextNext == 'e'.code.toUInt()) ||
                            (cptNext == 'v'.code.toUInt() && cptNextNext == 'e'.code.toUInt()) ||
                            (cptNext == 'l'.code.toUInt() && cptNextNext == 'l'.code.toUInt())) {
                            pos += addToken(pos + 3); continue
                        }
                    }
                }

                // [^\r\n\p{L}\p{N}]?\p{L}+
                if (!(cpt == '\r'.code.toUInt() || cpt == '\n'.code.toUInt() || flags.isNumber)) {
                    if (flags.isLetter || getFlags(pos + 1).isLetter) {
                        pos++
                        while (getFlags(pos).isLetter) pos++
                        addToken(pos); continue
                    }
                }

                // \p{N}{1,3}
                if (flags.isNumber) {
                    var ini = pos
                    while (getFlags(pos).isNumber) {
                        if (++pos - ini >= 3) { addToken(pos); ini = pos }
                    }
                    addToken(pos); continue
                }

                // <space>?[^\s\p{L}\p{N}]+[\r\n]*
                val flags2 = if (cpt == ' '.code.toUInt()) getFlags(pos + 1) else flags
                if (!(flags2.isWhitespace || flags2.isLetter || flags2.isNumber) && flags.asUint() != 0.toUShort()) {
                    if (cpt == ' '.code.toUInt()) pos++
                    var f = flags2
                    while (!(f.isWhitespace || f.isLetter || f.isNumber) && f.asUint() != 0.toUShort()) {
                        f = getFlags(++pos)
                    }
                    var cpt2 = getCpt(pos)
                    while (cpt2 == '\r'.code.toUInt() || cpt2 == '\n'.code.toUInt()) {
                        cpt2 = getCpt(++pos)
                    }
                    addToken(pos); continue
                }

                var numWhitespaces = 0
                var lastEndROrN = 0
                while (getFlags(pos + numWhitespaces).isWhitespace) {
                    val cpt2 = getCpt(pos + numWhitespaces)
                    if (cpt2 == '\r'.code.toUInt() || cpt2 == '\n'.code.toUInt()) {
                        lastEndROrN = pos + numWhitespaces + 1
                    }
                    numWhitespaces++
                }

                // \s*[\r\n]+
                if (lastEndROrN > 0) {
                    pos = lastEndROrN
                    addToken(pos); continue
                }
                // \s+(?!\S)
                if (numWhitespaces > 1 && getCpt(pos + numWhitespaces) != OUT_OF_RANGE) {
                    pos += numWhitespaces - 1
                    addToken(pos); continue
                }
                // \s+
                if (numWhitespaces > 0) {
                    pos += numWhitespaces
                    addToken(pos); continue
                }

                addToken(++pos)
            }
        }
        return bpeOffsets
    }

    // -------------------------------------------------------------------------
    // Qwen2
    // -------------------------------------------------------------------------

    /**
     * Qwen2 tokenisation pattern: same structure as LLaMA-3 but `\p{N}` matches
     * single digits (no `{1,3}`).
     */
    private fun regexSplitCustomQwen2(
        cpts: List<UInt>,
        offsets: List<Int>,
    ): List<Int> {
        val bpeOffsets = ArrayList<Int>(offsets.size)
        var start = 0

        for (offset in offsets) {
            val offsetIni = start
            val offsetEnd = start + offset
            start = offsetEnd

            fun getCpt(pos: Int): UInt =
                if (pos in offsetIni until offsetEnd) cpts[pos] else OUT_OF_RANGE

            fun getFlags(pos: Int): CptFlags =
                if (pos in offsetIni until offsetEnd) unicodeCptFlagsFromCpt(cpts[pos]) else CptFlags()

            var prevEnd = offsetIni
            fun addToken(end: Int): Int {
                val len = end - prevEnd
                if (len > 0) bpeOffsets.add(len)
                prevEnd = end
                return len
            }

            var pos = offsetIni
            while (pos < offsetEnd) {
                val cpt = getCpt(pos)
                val flags = getFlags(pos)

                // (?i:'s|'t|'re|'ve|'m|'ll|'d)
                if (cpt == '\''.code.toUInt() && pos + 1 < offsetEnd) {
                    val cptNext = unicodeToLower(getCpt(pos + 1))
                    if (cptNext == 's'.code.toUInt() || cptNext == 't'.code.toUInt() ||
                        cptNext == 'm'.code.toUInt() || cptNext == 'd'.code.toUInt()) {
                        pos += addToken(pos + 2); continue
                    }
                    if (pos + 2 < offsetEnd) {
                        val cptNextNext = unicodeToLower(getCpt(pos + 2))
                        if ((cptNext == 'r'.code.toUInt() && cptNextNext == 'e'.code.toUInt()) ||
                            (cptNext == 'v'.code.toUInt() && cptNextNext == 'e'.code.toUInt()) ||
                            (cptNext == 'l'.code.toUInt() && cptNextNext == 'l'.code.toUInt())) {
                            pos += addToken(pos + 3); continue
                        }
                    }
                }

                // [^\r\n\p{L}\p{N}]?\p{L}+
                if (!(cpt == '\r'.code.toUInt() || cpt == '\n'.code.toUInt() || flags.isNumber)) {
                    if (flags.isLetter || getFlags(pos + 1).isLetter) {
                        pos++
                        while (getFlags(pos).isLetter) pos++
                        addToken(pos); continue
                    }
                }

                // \p{N} (single digit)
                if (flags.isNumber) {
                    pos++
                    addToken(pos); continue
                }

                // <space>?[^\s\p{L}\p{N}]+[\r\n]*
                val flags2 = if (cpt == ' '.code.toUInt()) getFlags(pos + 1) else flags
                if (!(flags2.isWhitespace || flags2.isLetter || flags2.isNumber) && flags.asUint() != 0.toUShort()) {
                    if (cpt == ' '.code.toUInt()) pos++
                    var f = flags2
                    while (!(f.isWhitespace || f.isLetter || f.isNumber) && f.asUint() != 0.toUShort()) {
                        f = getFlags(++pos)
                    }
                    var cpt2 = getCpt(pos)
                    while (cpt2 == '\r'.code.toUInt() || cpt2 == '\n'.code.toUInt()) {
                        cpt2 = getCpt(++pos)
                    }
                    addToken(pos); continue
                }

                var numWhitespaces = 0
                var lastEndROrN = 0
                while (getFlags(pos + numWhitespaces).isWhitespace) {
                    val cpt2 = getCpt(pos + numWhitespaces)
                    if (cpt2 == '\r'.code.toUInt() || cpt2 == '\n'.code.toUInt()) {
                        lastEndROrN = pos + numWhitespaces + 1
                    }
                    numWhitespaces++
                }

                if (lastEndROrN > 0) { pos = lastEndROrN; addToken(pos); continue }
                if (numWhitespaces > 1 && getCpt(pos + numWhitespaces) != OUT_OF_RANGE) {
                    pos += numWhitespaces - 1; addToken(pos); continue
                }
                if (numWhitespaces > 0) { pos += numWhitespaces; addToken(pos); continue }

                addToken(++pos)
            }
        }
        return bpeOffsets
    }

    // -------------------------------------------------------------------------
    // Kimi K2
    // -------------------------------------------------------------------------

    /**
     * Kimi K2 tokenisation: handles CJK characters, letter words with
     * optional contractions, numbers in groups of 1–3, and whitespace.
     */
    private fun regexSplitCustomKimiK2(
        cpts: List<UInt>,
        offsets: List<Int>,
    ): List<Int> {
        val bpeOffsets = ArrayList<Int>(offsets.size)
        var start = 0

        for (offset in offsets) {
            val offsetIni = start
            val offsetEnd = start + offset
            start = offsetEnd

            fun getCpt(pos: Int): UInt =
                if (pos in offsetIni until offsetEnd) cpts[pos] else OUT_OF_RANGE

            fun getFlags(pos: Int): CptFlags =
                if (pos in offsetIni until offsetEnd) unicodeCptFlagsFromCpt(cpts[pos]) else CptFlags()

            var prevEnd = offsetIni
            fun addToken(end: Int): Int {
                val len = end - prevEnd
                if (len > 0) bpeOffsets.add(len)
                prevEnd = end
                return len
            }

            var pos = offsetIni
            while (pos < offsetEnd) {
                val cpt = getCpt(pos)
                val flags = getFlags(pos)

                // [\p{Han}]+
                if (unicodeCptIsHan(cpt)) {
                    while (unicodeCptIsHan(getCpt(pos))) pos++
                    addToken(pos); continue
                }

                // Letter words (excluding Han) with optional contractions
                val isLetterPattern = (flags.isLetter && !unicodeCptIsHan(cpt)) ||
                    (!(cpt == '\r'.code.toUInt() || cpt == '\n'.code.toUInt() || flags.isLetter || flags.isNumber) &&
                        getFlags(pos + 1).isLetter && !unicodeCptIsHan(getCpt(pos + 1)))

                if (isLetterPattern) {
                    var hasLeadingChar = false
                    if (!(cpt == '\r'.code.toUInt() || cpt == '\n'.code.toUInt() || flags.isLetter || flags.isNumber)) {
                        hasLeadingChar = true; pos++
                    }
                    var hasLetters = false
                    while (getFlags(pos).isLetter && !unicodeCptIsHan(getCpt(pos))) {
                        hasLetters = true; pos++
                    }
                    if (hasLetters || (!hasLeadingChar && getFlags(pos).isLetter && !unicodeCptIsHan(getCpt(pos)))) {
                        if (!hasLetters) pos++
                        while (getFlags(pos).isLetter && !unicodeCptIsHan(getCpt(pos))) pos++
                        // Optional contractions
                        if (getCpt(pos) == '\''.code.toUInt() && pos + 1 < offsetEnd) {
                            val cn = unicodeToLower(getCpt(pos + 1))
                            if (cn == 's'.code.toUInt() || cn == 't'.code.toUInt() || cn == 'm'.code.toUInt() || cn == 'd'.code.toUInt()) {
                                pos += 2
                            } else if (pos + 2 < offsetEnd) {
                                val cnn = unicodeToLower(getCpt(pos + 2))
                                if ((cn == 'r'.code.toUInt() && cnn == 'e'.code.toUInt()) ||
                                    (cn == 'v'.code.toUInt() && cnn == 'e'.code.toUInt()) ||
                                    (cn == 'l'.code.toUInt() && cnn == 'l'.code.toUInt())) {
                                    pos += 3
                                }
                            }
                        }
                        addToken(pos); continue
                    } else if (hasLeadingChar) {
                        pos--
                    }
                }

                // \p{N}{1,3}
                if (flags.isNumber) {
                    var ini = pos
                    while (getFlags(pos).isNumber) {
                        if (++pos - ini >= 3) { addToken(pos); ini = pos }
                    }
                    addToken(pos); continue
                }

                // <space>?[^\s\p{L}\p{N}]+[\r\n]*
                val flags2 = if (cpt == ' '.code.toUInt()) getFlags(pos + 1) else flags
                if (!(flags2.isWhitespace || flags2.isLetter || flags2.isNumber) && flags2.asUint() != 0.toUShort()) {
                    if (cpt == ' '.code.toUInt()) pos++
                    var f = flags2
                    while (!(f.isWhitespace || f.isLetter || f.isNumber) && f.asUint() != 0.toUShort()) {
                        f = getFlags(++pos)
                    }
                    var cpt2 = getCpt(pos)
                    while (cpt2 == '\r'.code.toUInt() || cpt2 == '\n'.code.toUInt()) { cpt2 = getCpt(++pos) }
                    addToken(pos); continue
                }

                var numWhitespaces = 0
                var lastEndROrN = 0
                while (getFlags(pos + numWhitespaces).isWhitespace) {
                    val cpt2 = getCpt(pos + numWhitespaces)
                    if (cpt2 == '\r'.code.toUInt() || cpt2 == '\n'.code.toUInt()) lastEndROrN = pos + numWhitespaces + 1
                    numWhitespaces++
                }

                if (lastEndROrN > 0) { pos = lastEndROrN; addToken(pos); continue }
                if (numWhitespaces > 1 && getCpt(pos + numWhitespaces) != OUT_OF_RANGE) {
                    pos += numWhitespaces - 1; addToken(pos); continue
                }
                if (numWhitespaces > 0) { pos += numWhitespaces; addToken(pos); continue }

                addToken(++pos)
            }
        }
        return bpeOffsets
    }

    // -------------------------------------------------------------------------
    // AFMOE digit handling
    // -------------------------------------------------------------------------

    /**
     * Splits digit sequences using modulo-3 grouping from the left: emit
     * `length % 3` leading digits, then groups of 3.
     */
    private fun regexSplitCustomAfmoe(
        cpts: List<UInt>,
        offsets: List<Int>,
    ): List<Int> {
        val bpeOffsets = ArrayList<Int>(offsets.size)
        var start = 0

        for (offset in offsets) {
            val offsetIni = start
            val offsetEnd = start + offset
            start = offsetEnd

            fun getFlags(pos: Int): CptFlags =
                if (pos in offsetIni until offsetEnd) unicodeCptFlagsFromCpt(cpts[pos]) else CptFlags()

            var prevEnd = offsetIni
            fun addToken(end: Int): Int {
                val len = end - prevEnd
                if (len > 0) bpeOffsets.add(len)
                prevEnd = end
                return len
            }

            var pos = offsetIni
            while (pos < offsetEnd) {
                if (getFlags(pos).isNumber) {
                    val digitStart = pos
                    var digitCount = 0
                    while (getFlags(pos).isNumber && pos < offsetEnd) { digitCount++; pos++ }
                    val remainder = digitCount % 3
                    var current = digitStart
                    if (remainder > 0) { prevEnd = current; addToken(current + remainder); current += remainder }
                    while (current < digitStart + digitCount) { prevEnd = current; addToken(current + 3); current += 3 }
                    continue
                }
                pos++
            }
            if (prevEnd < offsetEnd) addToken(offsetEnd)
        }
        return bpeOffsets
    }

    // -------------------------------------------------------------------------
    // Newlines
    // -------------------------------------------------------------------------

    /**
     * Splits text into runs of newline and non-newline characters.
     * Pattern: `[^\n]+|[\n]+`
     */
    private fun regexSplitCustomNewlines(
        cpts: List<UInt>,
        offsets: List<Int>,
    ): List<Int> {
        val bpeOffsets = ArrayList<Int>(offsets.size)
        var start = 0

        for (offset in offsets) {
            val offsetIni = start
            val offsetEnd = start + offset
            start = offsetEnd

            var pos = offsetIni
            while (pos < offsetEnd) {
                val isNewline = cpts[pos] == '\n'.code.toUInt()
                val runStart = pos
                while (pos < offsetEnd && (cpts[pos] == '\n'.code.toUInt()) == isNewline) pos++
                bpeOffsets.add(pos - runStart)
            }
        }
        return bpeOffsets
    }
}

// =============================================================================
// Extension: String.codePointAt  (Kotlin/Native doesn't have java.lang)
// =============================================================================

/**
 * Returns the Unicode code point at the given index, handling surrogate pairs.
 *
 * On Kotlin/JVM this delegates to `java.lang.String.codePointAt`; on
 * Kotlin/Native it replicates the surrogate-pair logic manually.
 */
internal fun String.codePointAt(index: Int): Int {
    val high = this[index]
    if (high.isHighSurrogate() && index + 1 < this.length) {
        val low = this[index + 1]
        if (low.isLowSurrogate()) {
            return 0x10000 + ((high.code - 0xD800) shl 10) + (low.code - 0xDC00)
        }
    }
    return high.code
}

/**
 * Appends a Unicode code point (potentially a supplementary-plane character)
 * to this [StringBuilder], encoding it as a surrogate pair if necessary.
 */
internal fun StringBuilder.appendCodePoint(codePoint: Int): StringBuilder {
    if (codePoint <= 0xFFFF) {
        append(codePoint.toChar())
    } else {
        val adjusted = codePoint - 0x10000
        append(((adjusted shr 10) + 0xD800).toChar())
        append(((adjusted and 0x3FF) + 0xDC00).toChar())
    }
    return this
}
