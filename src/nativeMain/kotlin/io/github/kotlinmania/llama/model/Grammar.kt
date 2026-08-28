// port-lint: source llama.cpp/src/llama-grammar.cpp
package io.github.kotlinmania.llama.model

/**
 * Grammar-based constrained generation for LLaMA models.
 *
 * This module implements GBNF (Grammar-Based Next-token Filtering) for structured
 * output. A GBNF grammar describes the set of valid token sequences using a
 * context-free grammar notation with character-level and token-level rules.
 *
 * Architecture overview:
 * - [LlamaGretype]: element types that make up grammar rules (chars, ranges, refs, tokens).
 * - [LlamaGrammarElement]: a single typed element with an associated code-point / rule-id value.
 * - Rules are flat lists of elements terminated by [LlamaGretype.END], with alternates
 *   separated by [LlamaGretype.ALT].
 * - [LlamaGrammar]: the runtime state machine that uses a stack-based pushdown automaton
 *   to accept or reject candidate tokens.
 * - [LlamaGrammarParser]: converts a GBNF grammar string into the rule representation
 *   consumed by [LlamaGrammar].
 *
 * Ported from llama.cpp `llama-grammar.h` / `llama-grammar.cpp`.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Safety cap on expansion of repetition operators ({m,n}, *, +, ?) to avoid
 *  unbounded rule generation. Mirrors MAX_REPETITION_THRESHOLD in the C++ source. */
private const val MAX_REPETITION_THRESHOLD = 2000

// ---------------------------------------------------------------------------
// Grammar element type enum
// ---------------------------------------------------------------------------

/**
 * Grammar element types corresponding to GBNF specification constructs.
 *
 * Each variant maps 1-to-1 to the C++ `llama_gretype` enum so that grammar
 * rules can be compared or printed in the same order as the reference implementation.
 */
enum class LlamaGretype(val value: Int) {
    /** End of a rule definition. */
    END(0),
    /** Start of an alternate definition for a rule (`|`). */
    ALT(1),
    /** Non-terminal element: reference to another rule by its id. */
    RULE_REF(2),
    /** Terminal element: exact character (Unicode code point). */
    CHAR(3),
    /** Inverse char(s): matches any character *not* in the set (`[^…]`). */
    CHAR_NOT(4),
    /** Modifies preceding CHAR or CHAR_ALT to form an inclusive range upper bound (`[a-z]`). */
    CHAR_RNG_UPPER(5),
    /** Adds an alternate character to the preceding CHAR or CHAR_RNG_UPPER (`[ab]`, `[a-zA]`). */
    CHAR_ALT(6),
    /** Matches any single character (`.`). */
    CHAR_ANY(7),
    /** Terminal element: token id (`<[token-id]>`). */
    TOKEN(8),
    /** Inverse token: matches any token *other than* the given id (`!<[token-id]>`). */
    TOKEN_NOT(9);

    companion object {
        private val byValue = entries.associateBy { it.value }
        fun fromValue(v: Int): LlamaGretype =
            byValue[v] ?: throw IllegalArgumentException("Unknown LlamaGretype value: $v")
    }
}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/**
 * A single element in a grammar rule.
 *
 * @property type The kind of grammar construct this element represents.
 * @property value Interpretation depends on [type]:
 *   - For character types: the Unicode code point.
 *   - For [LlamaGretype.RULE_REF]: the target rule index.
 *   - For [LlamaGretype.TOKEN] / [LlamaGretype.TOKEN_NOT]: the token id.
 *   - For [LlamaGretype.END] / [LlamaGretype.ALT]: unused (0).
 */
data class LlamaGrammarElement(
    val type: LlamaGretype,
    val value: UInt = 0u
)

/**
 * Tracks a partially decoded UTF-8 byte sequence so that multi-byte characters
 * split across token boundaries can be reconstructed correctly.
 *
 * @property value Accumulated bit value so far (unshifted).
 * @property nRemain Number of continuation bytes still expected; -1 means the
 *   sequence is invalid.
 */
data class LlamaPartialUtf8(
    val value: UInt = 0u,
    val nRemain: Int = 0
)

/**
 * A candidate token together with its decoded code-point representation.
 * Used internally by the rejection algorithm to walk through a token's
 * characters one by one while matching against the grammar.
 *
 * @property index Position in the original candidate array (for back-referencing).
 * @property codePoints Null-terminated list of decoded Unicode code points.
 * @property codePointOffset Current offset into [codePoints] during recursive matching.
 * @property partialUtf8 Trailing partial UTF-8 state, if any.
 * @property tokenId The original token id.
 */
data class LlamaGrammarCandidate(
    val index: Int,
    val codePoints: List<UInt>,
    val codePointOffset: Int = 0,
    val partialUtf8: LlamaPartialUtf8 = LlamaPartialUtf8(),
    val tokenId: Int = 0
)

/** A single grammar rule: a flat list of [LlamaGrammarElement]s. */
typealias LlamaGrammarRule = List<LlamaGrammarElement>
/** Mutable variant used during rule construction in the parser. */
typealias MutableGrammarRule = MutableList<LlamaGrammarElement>

/**
 * A grammar stack is a list of positions (indices) within rules.
 * Each position is represented as a (ruleIndex, elementIndex) pair.
 *
 * In the C++ source these are raw pointers into the element arrays; here we
 * use an explicit pair of indices which is safe under Kotlin's memory model.
 */
data class GrammarStackPos(val ruleIndex: Int, val elementIndex: Int)

typealias LlamaGrammarStack = List<GrammarStackPos>
typealias MutableGrammarStack = MutableList<GrammarStackPos>
typealias LlamaGrammarStacks = MutableList<LlamaGrammarStack>

/**
 * A complete grammar rule (legacy shape).
 * New code should use [LlamaGrammarRule] directly.
 */
data class GrammarRule(val elements: List<LlamaGrammarElement>)

/**
 * State in the grammar parsing state machine (legacy shape).
 * New code should use [GrammarStackPos] directly.
 */
data class GrammarState(val ruleId: Int, val position: Int)

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

/**
 * UTF-8 decoding utilities ported from the C++ helpers in llama-grammar.cpp.
 *
 * These operate on Kotlin [String]s (which are internally UTF-16) but produce
 * lists of Unicode code points ([UInt]) so that the grammar matching logic can
 * work at the code-point level, exactly as the C++ implementation does.
 */
object GrammarUtf8 {

    /** Decode a single UTF-8 code point from a [ByteArray] starting at [offset].
     *  Returns a pair of (codePoint, bytesConsumed). */
    fun decodeSingleUtf8(bytes: ByteArray, offset: Int): Pair<UInt, Int> {
        if (offset >= bytes.size) return Pair(0u, 0)
        val lookup = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4)
        val firstByte = bytes[offset].toInt() and 0xFF
        val highBits = firstByte ushr 4
        val len = lookup[highBits]
        val mask = (1 shl (8 - len)) - 1
        var value = (firstByte and mask).toUInt()
        var pos = offset + 1
        val end = (offset + len).coerceAtMost(bytes.size)
        while (pos < end) {
            value = (value shl 6) + ((bytes[pos].toInt() and 0x3F).toUInt())
            pos++
        }
        return Pair(value, pos - offset)
    }

    /**
     * Decode a [String] (or token piece) into Unicode code points, continuing
     * from a possibly incomplete previous sequence described by [partialStart].
     *
     * Returns a pair of:
     *   1. A **null-terminated** list of code points (the trailing 0u sentinel
     *      mirrors the C++ convention and simplifies the matching loops).
     *   2. A [LlamaPartialUtf8] describing any trailing incomplete sequence.
     */
    fun decodeString(
        src: String,
        partialStart: LlamaPartialUtf8 = LlamaPartialUtf8()
    ): Pair<List<UInt>, LlamaPartialUtf8> {
        val codePoints = mutableListOf<UInt>()
        val bytes = src.encodeToByteArray()
        var pos = 0
        var value = partialStart.value
        var nRemain = partialStart.nRemain

        val lookup = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4)

        // Continue previous incomplete sequence
        while (pos < bytes.size && nRemain > 0) {
            val nextByte = bytes[pos].toInt() and 0xFF
            if ((nextByte ushr 6) != 2) {
                codePoints.add(0u)
                return Pair(codePoints, LlamaPartialUtf8(0u, -1))
            }
            value = (value shl 6) + (nextByte.toUInt() and 0x3Fu)
            pos++
            nRemain--
        }
        if (partialStart.nRemain > 0 && nRemain == 0) {
            codePoints.add(value)
        }

        // Decode remaining bytes
        while (pos < bytes.size) {
            val firstByte = bytes[pos].toInt() and 0xFF
            val highBits = firstByte ushr 4
            nRemain = lookup[highBits] - 1
            if (nRemain < 0) {
                return Pair(listOf(0u), LlamaPartialUtf8(0u, nRemain))
            }
            val mask = (1 shl (7 - nRemain)) - 1
            value = (firstByte and mask).toUInt()
            pos++
            while (pos < bytes.size && nRemain > 0) {
                value = (value shl 6) + ((bytes[pos].toInt() and 0xFF).toUInt() and 0x3Fu)
                pos++
                nRemain--
            }
            if (nRemain == 0) {
                codePoints.add(value)
            }
        }
        codePoints.add(0u) // null terminator
        return Pair(codePoints, LlamaPartialUtf8(value, nRemain))
    }
}

// ---------------------------------------------------------------------------
// GBNF parser
// ---------------------------------------------------------------------------

/**
 * Parses a GBNF grammar string into a list of [LlamaGrammarRule]s together
 * with a symbol-id table.
 *
 * The parser is a faithful port of `llama_grammar_parser` from `llama-grammar.cpp`.
 * It supports:
 * - Rule definitions (`name ::= ...`)
 * - Character literals (`"..."`) and character classes (`[a-z]`, `[^0-9]`)
 * - Rule references by name
 * - Grouping with parentheses
 * - Repetition operators: `*`, `+`, `?`, `{m}`, `{m,n}`, `{m,}`
 * - Any-character wildcard (`.`)
 * - Token literals (`<[id]>`) and inverse tokens (`!<[id]>`)
 * - Alternates (`|`)
 * - Comments (`# ...`)
 * - Escape sequences (`\n`, `\t`, `\r`, `\\`, `\"`, `\xHH`, `\uHHHH`, `\UHHHHHHHH`)
 */
class LlamaGrammarParser {

    /** Maps symbol name → rule id. */
    val symbolIds: MutableMap<String, UInt> = mutableMapOf()

    /** All parsed rules, indexed by rule id. */
    val rules: MutableList<MutableGrammarRule> = mutableListOf()

    // -- symbol helpers -----------------------------------------------------

    /** Look up or create a symbol id for the given name. */
    fun getSymbolId(name: String): UInt {
        val nextId = symbolIds.size.toUInt()
        return symbolIds.getOrPut(name) { nextId }
    }

    /** Generate a unique anonymous symbol id derived from [baseName]. */
    fun generateSymbolId(baseName: String): UInt {
        val nextId = symbolIds.size.toUInt()
        symbolIds["${baseName}_$nextId"] = nextId
        return nextId
    }

    /** Append or replace the rule at [ruleId]. */
    fun addRule(ruleId: UInt, rule: List<LlamaGrammarElement>) {
        val idx = ruleId.toInt()
        while (rules.size <= idx) {
            rules.add(mutableListOf())
        }
        rules[idx] = rule.toMutableList()
    }

    // -- low-level parse helpers --------------------------------------------

    companion object {

        private fun isDigitChar(c: Char): Boolean = c in '0'..'9'

        private fun isWordChar(c: Char): Boolean =
            c in 'a'..'z' || c in 'A'..'Z' || c == '-' || isDigitChar(c)

        /** Skip whitespace (spaces, tabs, comments). If [newlineOk] also skip CR/LF. */
        fun parseSpace(src: String, pos: Int, newlineOk: Boolean): Int {
            var p = pos
            while (p < src.length) {
                val c = src[p]
                if (c == ' ' || c == '\t') {
                    p++
                } else if (c == '#') {
                    while (p < src.length && src[p] != '\r' && src[p] != '\n') p++
                } else if (newlineOk && (c == '\r' || c == '\n')) {
                    p++
                } else {
                    break
                }
            }
            return p
        }

        /** Parse an identifier (word chars). Returns end position. Throws on empty match. */
        fun parseName(src: String, pos: Int): Int {
            var p = pos
            while (p < src.length && isWordChar(src[p])) p++
            if (p == pos) throw IllegalArgumentException("expecting name at position $pos")
            return p
        }

        /** Parse a run of digit chars. Returns end position. Throws on empty match. */
        fun parseInt(src: String, pos: Int): Int {
            var p = pos
            while (p < src.length && isDigitChar(src[p])) p++
            if (p == pos) throw IllegalArgumentException("expecting integer at position $pos")
            return p
        }

        /** Parse [size] hex digits starting at [pos]. Returns (value, newPos). */
        fun parseHex(src: String, pos: Int, size: Int): Pair<UInt, Int> {
            var p = pos
            val end = pos + size
            var value = 0u
            while (p < end && p < src.length) {
                value = value shl 4
                val c = src[p]
                value += when {
                    c in 'a'..'f' -> (c - 'a' + 10).toUInt()
                    c in 'A'..'F' -> (c - 'A' + 10).toUInt()
                    c in '0'..'9' -> (c - '0').toUInt()
                    else -> throw IllegalArgumentException(
                        "expecting $size hex chars at position $pos"
                    )
                }
                p++
            }
            if (p != end) throw IllegalArgumentException("expecting $size hex chars at position $pos")
            return Pair(value, p)
        }

        /**
         * Parse a single (possibly escaped) character. Returns (codePoint, newPos).
         *
         * Supported escape sequences: `\x##`, `\u####`, `\U########`,
         * `\t`, `\r`, `\n`, `\\`, `\"`, `\[`, `\]`.
         */
        fun parseChar(src: String, pos: Int): Pair<UInt, Int> {
            if (pos >= src.length) throw IllegalArgumentException("unexpected end of input")
            if (src[pos] == '\\') {
                if (pos + 1 >= src.length) throw IllegalArgumentException("unexpected end of input after \\")
                return when (src[pos + 1]) {
                    'x' -> parseHex(src, pos + 2, 2)
                    'u' -> parseHex(src, pos + 2, 4)
                    'U' -> parseHex(src, pos + 2, 8)
                    't' -> Pair('\t'.code.toUInt(), pos + 2)
                    'r' -> Pair('\r'.code.toUInt(), pos + 2)
                    'n' -> Pair('\n'.code.toUInt(), pos + 2)
                    '\\', '"', '[', ']' -> Pair(src[pos + 1].code.toUInt(), pos + 2)
                    else -> throw IllegalArgumentException(
                        "unknown escape '\\${src[pos + 1]}' at position $pos"
                    )
                }
            }
            // Plain character – handle as a single Kotlin Char (BMP code point).
            // Full surrogate-pair handling would be needed for characters above U+FFFF.
            return Pair(src[pos].code.toUInt(), pos + 1)
        }

        /** Parse a token literal `<[id]>`. Returns (tokenId, newPos). */
        fun parseToken(src: String, pos: Int): Pair<UInt, Int> {
            if (pos >= src.length || src[pos] != '<') {
                throw IllegalArgumentException("expecting '<' at position $pos")
            }
            var p = pos + 1
            if (p < src.length && src[p] == '[') {
                p++
                val intEnd = parseInt(src, p)
                val tokenId = src.substring(p, intEnd).toUInt()
                p = intEnd
                if (p >= src.length || src[p] != ']') {
                    throw IllegalArgumentException("expecting ']' at position $p")
                }
                p++
                if (p >= src.length || src[p] != '>') {
                    throw IllegalArgumentException("expecting '>' at position $p")
                }
                return Pair(tokenId, p + 1)
            }
            // Bare <token> form requires a vocab (not supported in pure-Kotlin port)
            throw IllegalArgumentException(
                "bare <token> syntax requires a vocab; use <[id]> form instead (at position $pos)"
            )
        }
    }

    // -- sequence / alternates / rule parsers --------------------------------

    /**
     * Parse a sequence of grammar elements (one branch of an alternate).
     *
     * Reads character literals, character classes, rule references, grouping,
     * repetition operators, token literals, and the any-character wildcard until
     * the input is exhausted or an unrecognised character is found.
     *
     * @return The position in [src] after the parsed sequence.
     */
    fun parseSequence(
        src: String,
        pos: Int,
        ruleName: String,
        rule: MutableGrammarRule,
        isNested: Boolean
    ): Int {
        var p = pos
        var lastSymStart = rule.size
        var nPrevRules = 1uL

        // Repetition handler closure – mirrors handle_repetitions in C++
        fun handleRepetitions(minTimes: ULong, maxTimes: ULong) {
            val noMax = maxTimes == ULong.MAX_VALUE
            if (lastSymStart == rule.size) {
                throw IllegalArgumentException("expecting preceding item for repetition at position $p")
            }
            val prevRule = rule.subList(lastSymStart, rule.size).toList()

            val totalRules: ULong = when {
                !noMax && maxTimes > 0uL -> maxTimes
                minTimes > 0uL -> minTimes
                else -> 1uL
            }
            if (nPrevRules * totalRules >= MAX_REPETITION_THRESHOLD.toULong()) {
                throw IllegalArgumentException(
                    "repetition expansion exceeds $MAX_REPETITION_THRESHOLD rules"
                )
            }

            if (minTimes == 0uL) {
                // Remove the symbol we are repeating from the rule
                while (rule.size > lastSymStart) rule.removeAt(rule.lastIndex)
            } else {
                for (i in 1uL until minTimes) {
                    rule.addAll(prevRule)
                }
            }

            var lastRecRuleId = 0u
            val nOpt = if (noMax) 1uL else maxTimes - minTimes

            for (i in 0uL until nOpt) {
                val recRule = prevRule.toMutableList()
                val recRuleId = generateSymbolId(ruleName)
                if (i > 0uL || noMax) {
                    recRule.add(LlamaGrammarElement(
                        LlamaGretype.RULE_REF,
                        if (noMax) recRuleId else lastRecRuleId
                    ))
                }
                recRule.add(LlamaGrammarElement(LlamaGretype.ALT, 0u))
                recRule.add(LlamaGrammarElement(LlamaGretype.END, 0u))
                addRule(recRuleId, recRule)
                lastRecRuleId = recRuleId
            }
            if (nOpt > 0uL) {
                rule.add(LlamaGrammarElement(LlamaGretype.RULE_REF, lastRecRuleId))
            }
            nPrevRules *= totalRules
        }

        while (p < src.length) {
            val c = src[p]
            when {
                // --- String literal -------------------------------------------------
                c == '"' -> {
                    p++
                    lastSymStart = rule.size
                    nPrevRules = 1uL
                    while (p < src.length && src[p] != '"') {
                        val (cp, np) = parseChar(src, p)
                        p = np
                        rule.add(LlamaGrammarElement(LlamaGretype.CHAR, cp))
                    }
                    if (p >= src.length) throw IllegalArgumentException("unterminated string literal")
                    p = parseSpace(src, p + 1, isNested)
                }

                // --- Character class ------------------------------------------------
                c == '[' -> {
                    p++
                    var startType = LlamaGretype.CHAR
                    if (p < src.length && src[p] == '^') {
                        p++
                        startType = LlamaGretype.CHAR_NOT
                    }
                    lastSymStart = rule.size
                    nPrevRules = 1uL
                    while (p < src.length && src[p] != ']') {
                        val (cp, np) = parseChar(src, p)
                        p = np
                        val type = if (rule.size > lastSymStart) LlamaGretype.CHAR_ALT else startType
                        rule.add(LlamaGrammarElement(type, cp))
                        if (p < src.length && src[p] == '-' && p + 1 < src.length && src[p + 1] != ']') {
                            val (endCp, endNp) = parseChar(src, p + 1)
                            p = endNp
                            rule.add(LlamaGrammarElement(LlamaGretype.CHAR_RNG_UPPER, endCp))
                        }
                    }
                    if (p >= src.length) throw IllegalArgumentException("unterminated character class")
                    p = parseSpace(src, p + 1, isNested)
                }

                // --- Token literal / inverse token ----------------------------------
                c == '<' || c == '!' -> {
                    val type: LlamaGretype
                    if (c == '!') {
                        type = LlamaGretype.TOKEN_NOT
                        p++
                    } else {
                        type = LlamaGretype.TOKEN
                    }
                    val (tokenId, tokenEnd) = parseToken(src, p)
                    lastSymStart = rule.size
                    nPrevRules = 1uL
                    rule.add(LlamaGrammarElement(type, tokenId))
                    p = parseSpace(src, tokenEnd, isNested)
                }

                // --- Rule reference -------------------------------------------------
                isWordChar(c) -> {
                    val nameEnd = parseName(src, p)
                    val refRuleId = getSymbolId(src.substring(p, nameEnd))
                    p = parseSpace(src, nameEnd, isNested)
                    lastSymStart = rule.size
                    nPrevRules = 1uL
                    rule.add(LlamaGrammarElement(LlamaGretype.RULE_REF, refRuleId))
                }

                // --- Grouping -------------------------------------------------------
                c == '(' -> {
                    p = parseSpace(src, p + 1, true)
                    val nRulesBefore = symbolIds.size.toUInt()
                    val subRuleId = generateSymbolId(ruleName)
                    p = parseAlternates(src, p, ruleName, subRuleId, isNested = true)
                    nPrevRules = maxOf(1u, symbolIds.size.toUInt() - nRulesBefore).toULong()
                    lastSymStart = rule.size
                    rule.add(LlamaGrammarElement(LlamaGretype.RULE_REF, subRuleId))
                    if (p >= src.length || src[p] != ')') {
                        throw IllegalArgumentException("expecting ')' at position $p")
                    }
                    p = parseSpace(src, p + 1, isNested)
                }

                // --- Any character --------------------------------------------------
                c == '.' -> {
                    lastSymStart = rule.size
                    nPrevRules = 1uL
                    rule.add(LlamaGrammarElement(LlamaGretype.CHAR_ANY, 0u))
                    p = parseSpace(src, p + 1, isNested)
                }

                // --- Repetition operators -------------------------------------------
                c == '*' -> {
                    p = parseSpace(src, p + 1, isNested)
                    handleRepetitions(0uL, ULong.MAX_VALUE)
                }
                c == '+' -> {
                    p = parseSpace(src, p + 1, isNested)
                    handleRepetitions(1uL, ULong.MAX_VALUE)
                }
                c == '?' -> {
                    p = parseSpace(src, p + 1, isNested)
                    handleRepetitions(0uL, 1uL)
                }
                c == '{' -> {
                    p = parseSpace(src, p + 1, isNested)
                    if (p >= src.length || !isDigitChar(src[p])) {
                        throw IllegalArgumentException("expecting integer at position $p")
                    }
                    val intEnd = parseInt(src, p)
                    val minTimes = src.substring(p, intEnd).toULong()
                    p = parseSpace(src, intEnd, isNested)

                    var maxTimes = ULong.MAX_VALUE
                    when {
                        p < src.length && src[p] == '}' -> {
                            maxTimes = minTimes
                            p = parseSpace(src, p + 1, isNested)
                        }
                        p < src.length && src[p] == ',' -> {
                            p = parseSpace(src, p + 1, isNested)
                            if (p < src.length && isDigitChar(src[p])) {
                                val intEnd2 = parseInt(src, p)
                                maxTimes = src.substring(p, intEnd2).toULong()
                                p = parseSpace(src, intEnd2, isNested)
                            }
                            if (p >= src.length || src[p] != '}') {
                                throw IllegalArgumentException("expecting '}' at position $p")
                            }
                            p = parseSpace(src, p + 1, isNested)
                        }
                        else -> throw IllegalArgumentException("expecting ',' or '}' at position $p")
                    }
                    val hasMax = maxTimes != ULong.MAX_VALUE
                    if (minTimes > MAX_REPETITION_THRESHOLD.toULong() ||
                        (hasMax && maxTimes > MAX_REPETITION_THRESHOLD.toULong())
                    ) {
                        throw IllegalArgumentException("repetition count exceeds $MAX_REPETITION_THRESHOLD")
                    }
                    handleRepetitions(minTimes, maxTimes)
                }

                // --- End of sequence ------------------------------------------------
                else -> break
            }
        }
        return p
    }

    /**
     * Parse one or more alternate sequences separated by `|` for [ruleId].
     * @return The position in [src] after the parsed alternates.
     */
    fun parseAlternates(
        src: String,
        pos: Int,
        ruleName: String,
        ruleId: UInt,
        isNested: Boolean
    ): Int {
        val rule: MutableGrammarRule = mutableListOf()
        var p = parseSequence(src, pos, ruleName, rule, isNested)
        while (p < src.length && src[p] == '|') {
            rule.add(LlamaGrammarElement(LlamaGretype.ALT, 0u))
            p = parseSpace(src, p + 1, true)
            p = parseSequence(src, p, ruleName, rule, isNested)
        }
        rule.add(LlamaGrammarElement(LlamaGretype.END, 0u))
        addRule(ruleId, rule)
        return p
    }

    /**
     * Parse a single top-level rule definition (`name ::= ...`).
     * @return The position in [src] after the rule.
     */
    fun parseRule(src: String, pos: Int): Int {
        val nameEnd = parseName(src, pos)
        var p = parseSpace(src, nameEnd, false)
        val name = src.substring(pos, nameEnd)
        val ruleId = getSymbolId(name)

        if (p + 2 >= src.length || src[p] != ':' || src[p + 1] != ':' || src[p + 2] != '=') {
            throw IllegalArgumentException("expecting '::=' at position $p")
        }
        p = parseSpace(src, p + 3, true)
        p = parseAlternates(src, p, name, ruleId, isNested = false)

        // Consume newline
        if (p < src.length && src[p] == '\r') {
            p += if (p + 1 < src.length && src[p + 1] == '\n') 2 else 1
        } else if (p < src.length && src[p] == '\n') {
            p++
        } else if (p < src.length) {
            throw IllegalArgumentException("expecting newline or end at position $p")
        }
        return parseSpace(src, p, true)
    }

    /**
     * Parse an entire GBNF grammar string.
     *
     * @return `true` on success, `false` if a parse error occurred (errors are
     *   logged to stderr in the C++ source; here we catch and return false).
     */
    fun parse(src: String): Boolean {
        return try {
            var p = parseSpace(src, 0, true)
            while (p < src.length) {
                p = parseRule(src, p)
            }
            // Validate: every rule must be non-empty and every RULE_REF must point
            // to an existing rule.
            for (rule in rules) {
                if (rule.isEmpty()) throw IllegalStateException("undefined rule")
                for (elem in rule) {
                    if (elem.type == LlamaGretype.RULE_REF) {
                        val idx = elem.value.toInt()
                        if (idx >= rules.size || rules[idx].isEmpty()) {
                            val name = symbolIds.entries
                                .firstOrNull { it.value == elem.value }?.key ?: "?"
                            throw IllegalStateException("undefined rule identifier '$name'")
                        }
                    }
                }
            }
            true
        } catch (e: Exception) {
            // Mirror C++ behavior: log and clear rules on failure
            println("LlamaGrammarParser.parse: error: ${e.message}")
            rules.clear()
            false
        }
    }

    /**
     * Return the rules as a flat list, analogous to C++ `c_rules()`.
     *
     * In the C++ source this returns `vector<const llama_grammar_element *>`
     * (one pointer per rule). In Kotlin we just return the rules list directly
     * since callers can index into it.
     */
    fun cRules(): List<LlamaGrammarRule> = rules.toList()

    /** Pretty-print rules in GBNF notation (for debugging). */
    fun print(): String {
        val idToName = mutableMapOf<UInt, String>()
        for ((name, id) in symbolIds) idToName[id] = name

        val sb = StringBuilder()
        for (i in rules.indices) {
            val rule = rules[i]
            if (rule.isEmpty() || rule.last().type != LlamaGretype.END) continue
            sb.append("${idToName[i.toUInt()] ?: i} ::= ")
            for (j in 0 until rule.size - 1) {
                val elem = rule[j]
                when (elem.type) {
                    LlamaGretype.END -> {} // shouldn't appear mid-rule
                    LlamaGretype.ALT -> sb.append("| ")
                    LlamaGretype.RULE_REF -> sb.append("${idToName[elem.value] ?: elem.value} ")
                    LlamaGretype.CHAR -> {
                        sb.append("[")
                        appendGrammarChar(sb, elem.value)
                    }
                    LlamaGretype.CHAR_NOT -> {
                        sb.append("[^")
                        appendGrammarChar(sb, elem.value)
                    }
                    LlamaGretype.CHAR_RNG_UPPER -> {
                        sb.append("-")
                        appendGrammarChar(sb, elem.value)
                    }
                    LlamaGretype.CHAR_ALT -> appendGrammarChar(sb, elem.value)
                    LlamaGretype.CHAR_ANY -> sb.append(".")
                    LlamaGretype.TOKEN -> sb.append("<[${elem.value}]> ")
                    LlamaGretype.TOKEN_NOT -> sb.append("!<[${elem.value}]> ")
                }
                if (isCharElement(elem)) {
                    val nextType = if (j + 1 < rule.size) rule[j + 1].type else null
                    if (nextType != LlamaGretype.CHAR_ALT &&
                        nextType != LlamaGretype.CHAR_RNG_UPPER &&
                        nextType != LlamaGretype.CHAR_ANY
                    ) {
                        sb.append("] ")
                    }
                }
            }
            sb.appendLine()
        }
        return sb.toString()
    }

    private fun appendGrammarChar(sb: StringBuilder, c: UInt) {
        if (c in 0x20u..0x7Fu) sb.append(c.toInt().toChar())
        else sb.append("<U+${c.toString(16).padStart(4, '0').uppercase()}>")
    }
}

// ---------------------------------------------------------------------------
// Element classification helpers
// ---------------------------------------------------------------------------

/** Returns `true` if [elem] is a character-class element (CHAR, CHAR_NOT, CHAR_ALT, CHAR_RNG_UPPER, CHAR_ANY). */
fun isCharElement(elem: LlamaGrammarElement): Boolean = when (elem.type) {
    LlamaGretype.CHAR,
    LlamaGretype.CHAR_NOT,
    LlamaGretype.CHAR_ALT,
    LlamaGretype.CHAR_RNG_UPPER,
    LlamaGretype.CHAR_ANY -> true
    else -> false
}

/** Returns `true` if the element at the given position marks the end of an alternate (END or ALT). */
fun isEndOfSequence(rule: LlamaGrammarRule, elementIndex: Int): Boolean {
    if (elementIndex >= rule.size) return true
    return rule[elementIndex].type == LlamaGretype.END ||
           rule[elementIndex].type == LlamaGretype.ALT
}

// ---------------------------------------------------------------------------
// Debug printing utilities
// ---------------------------------------------------------------------------

/**
 * Print a grammar rule in its raw binary representation (element types and values).
 *
 * Port of `print_rule_binary()` from `llama-grammar.cpp`.
 * Useful for low-level debugging of rule construction.
 */
fun printRuleBinary(rule: LlamaGrammarRule): String {
    val sb = StringBuilder()
    for (elem in rule) {
        when (elem.type) {
            LlamaGretype.END -> sb.append("END")
            LlamaGretype.ALT -> sb.append("ALT")
            LlamaGretype.RULE_REF -> sb.append("RULE_REF")
            LlamaGretype.CHAR -> sb.append("CHAR")
            LlamaGretype.CHAR_NOT -> sb.append("CHAR_NOT")
            LlamaGretype.CHAR_RNG_UPPER -> sb.append("CHAR_RNG_UPPER")
            LlamaGretype.CHAR_ALT -> sb.append("CHAR_ALT")
            LlamaGretype.CHAR_ANY -> sb.append("CHAR_ANY")
            LlamaGretype.TOKEN -> sb.append("TOKEN")
            LlamaGretype.TOKEN_NOT -> sb.append("TOKEN_NOT")
        }
        when (elem.type) {
            LlamaGretype.END,
            LlamaGretype.ALT,
            LlamaGretype.RULE_REF -> sb.append("(${elem.value}) ")
            LlamaGretype.CHAR,
            LlamaGretype.CHAR_NOT,
            LlamaGretype.CHAR_RNG_UPPER,
            LlamaGretype.CHAR_ALT,
            LlamaGretype.CHAR_ANY -> {
                sb.append("(\"")
                appendPrintableGrammarChar(sb, elem.value)
                sb.append("\") ")
            }
            LlamaGretype.TOKEN -> sb.append("<[${elem.value}]> ")
            LlamaGretype.TOKEN_NOT -> sb.append("!<[${elem.value}]> ")
        }
    }
    return sb.toString()
}

/**
 * Print a grammar rule in human-readable GBNF notation.
 *
 * Port of the free-standing `print_rule()` function from `llama-grammar.cpp`.
 *
 * @param ruleId      Index of the rule being printed.
 * @param rule        The rule element list (must be END-terminated).
 * @param idToName    Mapping from rule id to symbol name.
 * @return A single-line GBNF representation of the rule.
 * @throws IllegalStateException if the rule is malformed.
 */
fun printRule(
    ruleId: UInt,
    rule: LlamaGrammarRule,
    idToName: Map<UInt, String>
): String {
    if (rule.isEmpty() || rule.last().type != LlamaGretype.END) {
        throw IllegalStateException("malformed rule (not END-terminated): $ruleId")
    }
    val sb = StringBuilder()
    sb.append("${idToName[ruleId] ?: ruleId} ::= ")
    for (i in 0 until rule.size - 1) {
        val elem = rule[i]
        when (elem.type) {
            LlamaGretype.END -> throw IllegalStateException(
                "unexpected END in rule $ruleId at position $i"
            )
            LlamaGretype.ALT -> sb.append("| ")
            LlamaGretype.RULE_REF -> sb.append("${idToName[elem.value] ?: elem.value} ")
            LlamaGretype.CHAR -> {
                sb.append("[")
                appendPrintableGrammarChar(sb, elem.value)
            }
            LlamaGretype.CHAR_NOT -> {
                sb.append("[^")
                appendPrintableGrammarChar(sb, elem.value)
            }
            LlamaGretype.CHAR_RNG_UPPER -> {
                sb.append("-")
                appendPrintableGrammarChar(sb, elem.value)
            }
            LlamaGretype.CHAR_ALT -> appendPrintableGrammarChar(sb, elem.value)
            LlamaGretype.CHAR_ANY -> sb.append(".")
            LlamaGretype.TOKEN -> sb.append("<[${elem.value}]> ")
            LlamaGretype.TOKEN_NOT -> sb.append("!<[${elem.value}]> ")
        }
        if (isCharElement(elem)) {
            val nextType = if (i + 1 < rule.size) rule[i + 1].type else null
            if (nextType != LlamaGretype.CHAR_ALT &&
                nextType != LlamaGretype.CHAR_RNG_UPPER &&
                nextType != LlamaGretype.CHAR_ANY
            ) {
                sb.append("] ")
            }
        }
    }
    return sb.toString()
}

/** Append a single code point as a printable character, or as `<U+XXXX>` if non-printable. */
private fun appendPrintableGrammarChar(sb: StringBuilder, c: UInt) {
    if (c in 0x20u..0x7Fu) sb.append(c.toInt().toChar())
    else sb.append("<U+${c.toString(16).padStart(4, '0').uppercase()}>")
}

// ---------------------------------------------------------------------------
// Grammar matching engine
// ---------------------------------------------------------------------------

/**
 * Runtime grammar state machine — a stack-based pushdown automaton that
 * determines which tokens are valid at any point during generation.
 *
 * The matching logic faithfully mirrors the C++ `llama_grammar` functions:
 * - [advanceStack]: expand rule references until every stack top is a terminal.
 * - [matchChar]: test whether a code point satisfies a character element.
 * - [matchPartialChar]: test whether a partial UTF-8 sequence *could* satisfy a character element.
 * - [matchToken]: test whether a token id satisfies a TOKEN / TOKEN_NOT element.
 * - [rejectCandidatesForStack]: core rejection loop for a single stack.
 * - [rejectCandidates]: rejection across all active stacks.
 * - [acceptChar]: advance stacks after a matched code point.
 * - [acceptToken]: advance stacks after a matched token (with UTF-8 decoding).
 *
 * @property rules The immutable list of grammar rules (shared, never mutated).
 * @property stacks The current set of active pushdown stacks.
 * @property partialUtf8 Trailing partial UTF-8 state from the last accepted token.
 * @property lazy When true, the grammar waits for a trigger before constraining.
 * @property awaitingTrigger True while a lazy grammar has not yet been triggered.
 * @property triggerTokens Token ids that activate a lazy grammar.
 */
/**
 * A trigger pattern for lazy grammars.
 *
 * When a lazy grammar is waiting for a trigger, incoming text is buffered and
 * tested against each pattern. If a match is found, the grammar "wakes up"
 * and starts constraining from the first capturing group onward.
 *
 * Port of `llama_grammar_trigger_pattern` from `llama-grammar.h`.
 *
 * @property pattern  The original regex string.
 */
class LlamaGrammarTriggerPattern(val pattern: String) {
    private val regex: Regex = Regex(pattern)

    /**
     * Find the start position of the first capturing-group match (or the
     * whole match if there is no group).
     *
     * @return The start offset, or -1 if no match was found.
     */
    fun find(input: String): Int {
        val isAnchored = pattern.startsWith('^') && pattern.endsWith('$')
        val matchResult = if (isAnchored) {
            regex.matchEntire(input)
        } else {
            regex.find(input)
        } ?: return -1

        // Use the first non-empty capturing group as the start position
        for (i in 1 until matchResult.groupValues.size) {
            val group = matchResult.groups[i]
            if (group != null && group.value.isNotEmpty()) {
                return group.range.first
            }
        }
        return matchResult.range.first
    }
}

/**
 * Runtime grammar state machine — a stack-based pushdown automaton that
 * determines which tokens are valid at any point during generation.
 *
 * The matching logic faithfully mirrors the C++ `llama_grammar` functions:
 * - [advanceStack]: expand rule references until every stack top is a terminal.
 * - [matchChar]: test whether a code point satisfies a character element.
 * - [matchPartialChar]: test whether a partial UTF-8 sequence *could* satisfy a character element.
 * - [matchToken]: test whether a token id satisfies a TOKEN / TOKEN_NOT element.
 * - [rejectCandidatesForStack]: core rejection loop for a single stack.
 * - [rejectCandidates]: rejection across all active stacks.
 * - [acceptChar]: advance stacks after a matched code point.
 * - [acceptToken]: advance stacks after a matched token (with UTF-8 decoding).
 *
 * @property rules The immutable list of grammar rules (shared, never mutated).
 * @property stacks The current set of active pushdown stacks.
 * @property partialUtf8 Trailing partial UTF-8 state from the last accepted token.
 * @property lazy When true, the grammar waits for a trigger before constraining.
 * @property awaitingTrigger True while a lazy grammar has not yet been triggered.
 * @property triggerTokens Token ids that activate a lazy grammar.
 * @property triggerPatterns Regex patterns that trigger a lazy grammar.
 */
class LlamaGrammar(
    val rules: List<LlamaGrammarRule>,
    var stacks: LlamaGrammarStacks,
    var partialUtf8: LlamaPartialUtf8 = LlamaPartialUtf8(),
    val lazy: Boolean = false,
    var awaitingTrigger: Boolean = false,
    val triggerTokens: List<Int> = emptyList(),
    val triggerPatterns: List<LlamaGrammarTriggerPattern> = emptyList()
) {
    /** Buffer of generated text while a lazy grammar is waiting for a trigger. */
    var triggerBuffer: String = ""

    /**
     * Tokens buffered by a lazy grammar, with their (start, end) positions in
     * [triggerBuffer]. Used to replay when a trigger is found.
     */
    val triggerBufferPositions: MutableList<Triple<Int, Int, Int>> = mutableListOf()

    // -- Static helpers (ported from C++ free functions) --------------------

    companion object {

        /**
         * Create a [LlamaGrammar] from pre-parsed rules and a start rule index.
         *
         * Builds initial stacks by expanding the alternates of the start rule,
         * and checks for left recursion.
         *
         * @return A new [LlamaGrammar], or `null` if left recursion is detected.
         */
        fun init(
            rules: List<LlamaGrammarRule>,
            startRuleIndex: Int
        ): LlamaGrammar? {
            // Left recursion check
            val rulesVisited = BooleanArray(rules.size)
            val rulesInProgress = BooleanArray(rules.size)
            val rulesMayBeEmpty = BooleanArray(rules.size)
            for (i in rules.indices) {
                if (rulesVisited[i]) continue
                if (detectLeftRecursion(rules, i, rulesVisited, rulesInProgress, rulesMayBeEmpty)) {
                    println("LlamaGrammar.init: left recursion detected at rule $i")
                    return null
                }
            }

            // Build initial stacks from alternates of start rule
            val stacks: LlamaGrammarStacks = mutableListOf()
            val startRule = rules[startRuleIndex]
            var ei = 0
            while (true) {
                val stack: MutableGrammarStack = mutableListOf()
                if (!isEndOfSequence(startRule, ei)) {
                    stack.add(GrammarStackPos(startRuleIndex, ei))
                }
                advanceStack(rules, stack, stacks)
                while (!isEndOfSequence(startRule, ei)) ei++
                if (startRule[ei].type == LlamaGretype.ALT) {
                    ei++
                } else {
                    break
                }
            }

            return LlamaGrammar(rules, stacks)
        }

        /**
         * Create a [LlamaGrammar] by parsing a GBNF string and looking up
         * the given root rule name.
         *
         * @return A new [LlamaGrammar], or `null` on parse failure or if the
         *   root rule is not found.
         */
        fun fromString(
            grammarStr: String,
            grammarRoot: String = "root",
            lazy: Boolean = false,
            triggerTokens: List<Int> = emptyList(),
            triggerPatterns: List<String> = emptyList()
        ): LlamaGrammar? {
            val parser = LlamaGrammarParser()
            if (!parser.parse(grammarStr) || parser.rules.isEmpty()) {
                println("LlamaGrammar.fromString: failed to parse grammar")
                return null
            }
            val rootId = parser.symbolIds[grammarRoot]
            if (rootId == null) {
                println("LlamaGrammar.fromString: grammar does not contain '$grammarRoot' rule")
                return null
            }
            val g = init(parser.rules, rootId.toInt()) ?: return null
            return LlamaGrammar(
                rules = g.rules,
                stacks = g.stacks,
                lazy = lazy,
                awaitingTrigger = lazy,
                triggerTokens = triggerTokens,
                triggerPatterns = triggerPatterns.map { LlamaGrammarTriggerPattern(it) }
            )
        }

        // -- Stack advancement ------------------------------------------------

        /**
         * Expand the given [stack] by resolving rule references until every
         * resulting stack has a terminal (character, token, or empty) on top.
         * New terminal stacks are appended to [newStacks].
         *
         * This is the core of the pushdown automaton: it effectively performs
         * an ε-closure over rule references.
         */
        fun advanceStack(
            rules: List<LlamaGrammarRule>,
            stack: LlamaGrammarStack,
            newStacks: LlamaGrammarStacks
        ) {
            val todo = mutableListOf(stack.toMutableList() as LlamaGrammarStack)
            val seen = mutableSetOf<LlamaGrammarStack>()

            while (todo.isNotEmpty()) {
                val currStack = todo.removeAt(todo.lastIndex)
                if (!seen.add(currStack)) continue

                if (currStack.isEmpty()) {
                    if (currStack !in newStacks) newStacks.add(currStack)
                    continue
                }

                val pos = currStack.last()
                val elem = rules[pos.ruleIndex][pos.elementIndex]

                when (elem.type) {
                    LlamaGretype.RULE_REF -> {
                        val refRuleId = elem.value.toInt()
                        val subRule = rules[refRuleId]
                        var subEi = 0
                        while (true) {
                            val nextStack = currStack.dropLast(1).toMutableList()
                            if (!isEndOfSequence(rules[pos.ruleIndex], pos.elementIndex + 1)) {
                                nextStack.add(GrammarStackPos(pos.ruleIndex, pos.elementIndex + 1))
                            }
                            if (!isEndOfSequence(subRule, subEi)) {
                                nextStack.add(GrammarStackPos(refRuleId, subEi))
                            }
                            todo.add(nextStack)
                            while (!isEndOfSequence(subRule, subEi)) subEi++
                            if (subRule[subEi].type == LlamaGretype.ALT) {
                                subEi++
                            } else {
                                break
                            }
                        }
                    }

                    LlamaGretype.CHAR,
                    LlamaGretype.CHAR_NOT,
                    LlamaGretype.CHAR_ANY,
                    LlamaGretype.TOKEN,
                    LlamaGretype.TOKEN_NOT -> {
                        if (currStack !in newStacks) newStacks.add(currStack)
                    }

                    else -> {
                        // END, ALT, CHAR_ALT, CHAR_RNG_UPPER should never be on top of stack
                        throw IllegalStateException(
                            "unexpected element type ${elem.type} on stack top"
                        )
                    }
                }
            }
        }

        // -- Character / token matching ---------------------------------------

        /**
         * Test whether [chr] satisfies the character element at position [pos]
         * within [rule].
         *
         * @return A pair of (matched, nextElementIndex) where nextElementIndex
         *   points past the entire character range specification.
         */
        fun matchChar(
            rule: LlamaGrammarRule,
            elementIndex: Int,
            chr: UInt
        ): Pair<Boolean, Int> {
            val pos0 = rule[elementIndex]
            var found = false
            val isPositive = pos0.type == LlamaGretype.CHAR || pos0.type == LlamaGretype.CHAR_ANY

            var ei = elementIndex
            do {
                val elem = rule[ei]
                if (ei + 1 < rule.size && rule[ei + 1].type == LlamaGretype.CHAR_RNG_UPPER) {
                    found = found || (elem.value <= chr && chr <= rule[ei + 1].value)
                    ei += 2
                } else if (elem.type == LlamaGretype.CHAR_ANY) {
                    found = true
                    ei += 1
                } else {
                    found = found || (elem.value == chr)
                    ei += 1
                }
            } while (ei < rule.size && rule[ei].type == LlamaGretype.CHAR_ALT)

            return Pair(found == isPositive, ei)
        }

        /**
         * Test whether *some* completion of the given partial UTF-8 sequence
         * could satisfy the character element at [elementIndex] within [rule].
         */
        fun matchPartialChar(
            rule: LlamaGrammarRule,
            elementIndex: Int,
            partial: LlamaPartialUtf8
        ): Boolean {
            val pos0 = rule[elementIndex]
            val isPositive = pos0.type == LlamaGretype.CHAR || pos0.type == LlamaGretype.CHAR_ANY

            val nRemain = partial.nRemain
            if (nRemain < 0 || (nRemain == 1 && partial.value < 2u)) return false

            val low: UInt
            val high: UInt
            val rawLow = partial.value shl (nRemain * 6)
            high = rawLow or ((1u shl (nRemain * 6)) - 1u)
            low = when {
                rawLow == 0u && nRemain == 2 -> 1u shl 11
                rawLow == 0u && nRemain == 3 -> 1u shl 16
                else -> rawLow
            }

            var ei = elementIndex
            do {
                val elem = rule[ei]
                if (ei + 1 < rule.size && rule[ei + 1].type == LlamaGretype.CHAR_RNG_UPPER) {
                    if (elem.value <= high && low <= rule[ei + 1].value) return isPositive
                    ei += 2
                } else if (elem.type == LlamaGretype.CHAR_ANY) {
                    return true
                } else {
                    if (low <= elem.value && elem.value <= high) return isPositive
                    ei += 1
                }
            } while (ei < rule.size && rule[ei].type == LlamaGretype.CHAR_ALT)

            return !isPositive
        }

        /** Test whether [token] satisfies a TOKEN or TOKEN_NOT element. */
        fun matchToken(
            rule: LlamaGrammarRule,
            elementIndex: Int,
            token: Int
        ): Boolean {
            val elem = rule[elementIndex]
            return when (elem.type) {
                LlamaGretype.TOKEN -> elem.value == token.toUInt()
                LlamaGretype.TOKEN_NOT -> elem.value != token.toUInt()
                else -> false
            }
        }

        // -- Left recursion detection -----------------------------------------

        /**
         * Detect left recursion starting from [ruleIndex].
         * Uses the standard "in-progress" DFS approach.
         */
        fun detectLeftRecursion(
            rules: List<LlamaGrammarRule>,
            ruleIndex: Int,
            rulesVisited: BooleanArray,
            rulesInProgress: BooleanArray,
            rulesMayBeEmpty: BooleanArray
        ): Boolean {
            if (rulesInProgress[ruleIndex]) return true
            rulesInProgress[ruleIndex] = true

            val rule = rules[ruleIndex]

            // Check if this rule can produce the empty string
            var atRuleStart = true
            for (elem in rule) {
                if (isEndOfSequence(listOf(elem), 0)) {
                    if (atRuleStart) {
                        rulesMayBeEmpty[ruleIndex] = true
                        break
                    }
                    atRuleStart = true
                } else {
                    atRuleStart = false
                }
            }

            // Recurse into leftmost non-terminals
            var recurseIntoNonterminal = true
            for (i in rule.indices) {
                if (rule[i].type == LlamaGretype.RULE_REF && recurseIntoNonterminal) {
                    val refIdx = rule[i].value.toInt()
                    if (detectLeftRecursion(rules, refIdx, rulesVisited, rulesInProgress, rulesMayBeEmpty)) {
                        return true
                    }
                    if (!rulesMayBeEmpty[refIdx]) {
                        recurseIntoNonterminal = false
                    }
                } else if (isEndOfSequence(rule, i)) {
                    recurseIntoNonterminal = true
                } else {
                    recurseIntoNonterminal = false
                }
            }

            rulesInProgress[ruleIndex] = false
            rulesVisited[ruleIndex] = true
            return false
        }

        // -- Candidate rejection ----------------------------------------------

        /**
         * Given a single grammar [stack] and a list of [candidates], return
         * those candidates that are **rejected** (do not match the grammar).
         *
         * Each candidate carries a list of decoded code points and a partial
         * UTF-8 trailer. The function walks through the code points one at a time,
         * matching each against the grammar, and recursively checking the
         * remainder.
         */
        fun rejectCandidatesForStack(
            rules: List<LlamaGrammarRule>,
            stack: LlamaGrammarStack,
            candidates: List<LlamaGrammarCandidate>
        ): List<LlamaGrammarCandidate> {
            val rejects = mutableListOf<LlamaGrammarCandidate>()

            if (stack.isEmpty()) {
                // Empty stack → only accept candidates at end of their code-point sequence
                for (tok in candidates) {
                    val cp = tok.codePoints.getOrElse(tok.codePointOffset) { 0u }
                    if (cp != 0u || tok.partialUtf8.nRemain != 0) {
                        rejects.add(tok)
                    }
                }
                return rejects
            }

            val stackPos = stack.last()
            val elem = rules[stackPos.ruleIndex][stackPos.elementIndex]

            // Token-level matching
            if (elem.type == LlamaGretype.TOKEN || elem.type == LlamaGretype.TOKEN_NOT) {
                for (tok in candidates) {
                    val cp = tok.codePoints.getOrElse(tok.codePointOffset) { 0u }
                    if (cp == 0u) {
                        if (tok.partialUtf8.nRemain != 0) rejects.add(tok)
                    } else if (!matchToken(rules[stackPos.ruleIndex], stackPos.elementIndex, tok.tokenId)) {
                        rejects.add(tok)
                    }
                }
                return rejects
            }

            // Character-level matching
            val nextCandidates = mutableListOf<LlamaGrammarCandidate>()

            for (tok in candidates) {
                val cp = tok.codePoints.getOrElse(tok.codePointOffset) { 0u }
                if (cp == 0u) {
                    if (tok.partialUtf8.nRemain != 0 &&
                        !matchPartialChar(
                            rules[stackPos.ruleIndex], stackPos.elementIndex, tok.partialUtf8
                        )
                    ) {
                        rejects.add(tok)
                    }
                } else {
                    val (matched, _) = matchChar(
                        rules[stackPos.ruleIndex], stackPos.elementIndex, cp
                    )
                    if (matched) {
                        nextCandidates.add(tok.copy(codePointOffset = tok.codePointOffset + 1))
                    } else {
                        rejects.add(tok)
                    }
                }
            }

            // Advance the stack past the character element
            val (_, afterEi) = matchChar(rules[stackPos.ruleIndex], stackPos.elementIndex, 0u)
            val stackAfter = stack.dropLast(1).toMutableList()
            if (!isEndOfSequence(rules[stackPos.ruleIndex], afterEi)) {
                stackAfter.add(GrammarStackPos(stackPos.ruleIndex, afterEi))
            }
            val nextStacks: LlamaGrammarStacks = mutableListOf()
            advanceStack(rules, stackAfter, nextStacks)

            val nextRejects = rejectCandidates(rules, nextStacks, nextCandidates)
            for (tok in nextRejects) {
                rejects.add(tok.copy(codePointOffset = tok.codePointOffset - 1))
            }

            return rejects
        }

        /**
         * Reject candidates across all active [stacks]. The result is the
         * intersection of rejections from each stack (a candidate must be
         * rejected by *all* stacks to be truly rejected).
         */
        fun rejectCandidates(
            rules: List<LlamaGrammarRule>,
            stacks: LlamaGrammarStacks,
            candidates: List<LlamaGrammarCandidate>
        ): List<LlamaGrammarCandidate> {
            if (stacks.isEmpty() || candidates.isEmpty()) return candidates

            var rejects = rejectCandidatesForStack(rules, stacks[0], candidates)
            for (i in 1 until stacks.size) {
                rejects = rejectCandidatesForStack(rules, stacks[i], rejects)
            }
            return rejects
        }

        /**
         * Accept a character [chr] on a single [stack], appending any resulting
         * advanced stacks to [newStacks].
         *
         * Port of `llama_grammar_accept_chr()` from `llama-grammar.cpp`.
         * Skips stacks whose top is a TOKEN/TOKEN_NOT element (those are
         * matched at the token level, not character level).
         */
        fun acceptChr(
            rules: List<LlamaGrammarRule>,
            stack: LlamaGrammarStack,
            chr: UInt,
            newStacks: LlamaGrammarStacks
        ) {
            if (stack.isEmpty()) return
            val pos = stack.last()
            val elem = rules[pos.ruleIndex][pos.elementIndex]
            if (elem.type == LlamaGretype.TOKEN || elem.type == LlamaGretype.TOKEN_NOT) return

            val (matched, afterEi) = matchChar(rules[pos.ruleIndex], pos.elementIndex, chr)
            if (matched) {
                val newStack = stack.dropLast(1).toMutableList()
                if (!isEndOfSequence(rules[pos.ruleIndex], afterEi)) {
                    newStack.add(GrammarStackPos(pos.ruleIndex, afterEi))
                }
                advanceStack(rules, newStack, newStacks)
            }
        }
    }

    // -- Instance methods (mutating grammar state) --------------------------

    /**
     * Accept a single Unicode code point, advancing all stacks that match.
     */
    fun acceptChar(chr: UInt) {
        val stacksNew: LlamaGrammarStacks = mutableListOf()

        for (stack in stacks) {
            acceptChr(rules, stack, chr, stacksNew)
        }

        stacks = stacksNew
    }

    /**
     * Accept a string by decoding it into code points and accepting each one.
     * Handles partial UTF-8 continuation from the previous call.
     */
    fun acceptString(piece: String) {
        val (codePoints, newPartial) = GrammarUtf8.decodeString(piece, partialUtf8)
        // Skip the trailing 0u sentinel
        for (i in 0 until codePoints.size - 1) {
            acceptChar(codePoints[i])
        }
        partialUtf8 = newPartial
        if (stacks.isEmpty()) {
            throw IllegalStateException("grammar stack empty after accepting: $piece")
        }
    }

    /**
     * Accept a token by its id and text piece. Handles both token-level and
     * character-level grammar elements.
     *
     * Port of `llama_grammar_accept_token()` from `llama-grammar.cpp`.
     */
    fun acceptToken(tokenId: Int, piece: String) {
        val (codePoints, newPartial) = GrammarUtf8.decodeString(piece, partialUtf8)

        val stacksNew: LlamaGrammarStacks = mutableListOf()

        for (stack in stacks) {
            if (stack.isEmpty()) continue

            val pos = stack.last()
            val elem = rules[pos.ruleIndex][pos.elementIndex]

            if (elem.type == LlamaGretype.TOKEN || elem.type == LlamaGretype.TOKEN_NOT) {
                // Token-level matching
                if (matchToken(rules[pos.ruleIndex], pos.elementIndex, tokenId)) {
                    val newStack = stack.dropLast(1).toMutableList()
                    if (!isEndOfSequence(rules[pos.ruleIndex], pos.elementIndex + 1)) {
                        newStack.add(GrammarStackPos(pos.ruleIndex, pos.elementIndex + 1))
                    }
                    advanceStack(rules, newStack, stacksNew)
                }
            } else {
                // Character-level matching: walk through code points using acceptChr
                var currentStacks: LlamaGrammarStacks = mutableListOf(stack.toList())
                for (cpIdx in 0 until codePoints.size - 1) {
                    val nextStacks: LlamaGrammarStacks = mutableListOf()
                    for (curStack in currentStacks) {
                        acceptChr(rules, curStack, codePoints[cpIdx], nextStacks)
                    }
                    currentStacks = nextStacks
                    if (currentStacks.isEmpty()) break
                }
                for (survivingStack in currentStacks) {
                    if (survivingStack !in stacksNew) stacksNew.add(survivingStack)
                }
            }
        }

        stacks = stacksNew
        partialUtf8 = newPartial

        if (stacks.isEmpty()) {
            throw IllegalStateException("grammar stack empty after accepting token $tokenId: $piece")
        }
    }

    /**
     * Apply grammar constraints to a candidate token array by setting rejected
     * tokens' logits to negative infinity.
     *
     * @param candidates Array of [TokenData]; logits of rejected tokens are set
     *   to [Float.NEGATIVE_INFINITY].
     * @param tokenPieceFn Function that maps a token id to its string piece.
     * @param isEogFn Function that returns true if a token id is end-of-generation.
     */
    fun apply(
        candidates: Array<TokenData>,
        tokenPieceFn: (Int) -> String,
        isEogFn: (Int) -> Boolean
    ) {
        if (awaitingTrigger) return

        val allowEog = stacks.any { it.isEmpty() }

        val grammarCandidates = mutableListOf<LlamaGrammarCandidate>()
        val decodedCache = mutableListOf<Pair<List<UInt>, LlamaPartialUtf8>>()

        for (i in candidates.indices) {
            val td = candidates[i]
            if (isEogFn(td.id)) {
                if (!allowEog) td.logit = Float.NEGATIVE_INFINITY
            } else {
                val piece = tokenPieceFn(td.id)
                if (piece.isEmpty()) {
                    td.logit = Float.NEGATIVE_INFINITY
                } else {
                    val decoded = GrammarUtf8.decodeString(piece, partialUtf8)
                    decodedCache.add(decoded)
                    grammarCandidates.add(
                        LlamaGrammarCandidate(
                            index = i,
                            codePoints = decoded.first,
                            partialUtf8 = decoded.second,
                            tokenId = td.id
                        )
                    )
                }
            }
        }

        val rejects = rejectCandidates(rules, stacks, grammarCandidates)
        for (reject in rejects) {
            candidates[reject.index].logit = Float.NEGATIVE_INFINITY
        }
    }

    /** Deep copy of this grammar, including independent stacks and trigger buffer. */
    fun clone(): LlamaGrammar {
        val copy = LlamaGrammar(
            rules = rules, // rules are immutable
            stacks = stacks.map { it.toList() }.toMutableList(),
            partialUtf8 = partialUtf8,
            lazy = lazy,
            awaitingTrigger = awaitingTrigger,
            triggerTokens = triggerTokens,
            triggerPatterns = triggerPatterns
        )
        copy.triggerBuffer = triggerBuffer
        copy.triggerBufferPositions.addAll(triggerBufferPositions)
        return copy
    }

    /**
     * Accept a token with full lazy-grammar trigger logic.
     *
     * Port of `llama_grammar_accept_impl()` from `llama-grammar.cpp`.
     *
     * If the grammar is awaiting a trigger:
     * 1. Check if the token matches any trigger token — if so, wake up.
     * 2. Otherwise buffer the text and test each trigger pattern.
     * 3. When a pattern matches, replay buffered tokens from the match start.
     *
     * For non-lazy grammars (or after trigger), delegates to [acceptToken].
     *
     * @param tokenId  The accepted token id.
     * @param piece    The string representation of the token.
     * @param isEogFn  Returns `true` when [tokenId] is end-of-generation.
     */
    fun acceptImpl(
        tokenId: Int,
        piece: String,
        isEogFn: (Int) -> Boolean
    ) {
        if (awaitingTrigger) {
            // Check direct trigger tokens
            if (tokenId in triggerTokens) {
                awaitingTrigger = false
                triggerBuffer = ""
                triggerBufferPositions.clear()
                acceptToken(tokenId, piece)
                return
            }

            // Buffer the piece and check trigger patterns
            val bufStart = triggerBuffer.length
            val bufEnd = bufStart + piece.length
            triggerBufferPositions.add(Triple(tokenId, bufStart, bufEnd))
            triggerBuffer += piece

            for (triggerPattern in triggerPatterns) {
                val start = triggerPattern.find(triggerBuffer)
                if (start >= 0) {
                    awaitingTrigger = false

                    // Replay tokens that overlap with [start, end)
                    for ((tok, tokStart, tokEnd) in triggerBufferPositions) {
                        if (tokEnd <= start) continue
                        val pieceStart = if (tokStart < start) start else tokStart
                        val pieceLen = tokEnd - pieceStart
                        val tokPiece = triggerBuffer.substring(pieceStart, pieceStart + pieceLen)
                        acceptToken(tok, tokPiece)
                    }

                    triggerBuffer = ""
                    triggerBufferPositions.clear()
                    return
                }
            }
            // Still awaiting trigger
            return
        }

        // Non-lazy path
        if (isEogFn(tokenId)) {
            // EOG is valid only if some stack is empty (grammar is complete)
            if (stacks.any { it.isEmpty() }) return
            throw IllegalStateException("grammar received EOG token $tokenId but no stack is complete")
        }

        acceptToken(tokenId, piece)
    }

    /**
     * Debug representation of the current grammar stacks.
     *
     * Port of `print_rule_binary` / debug helpers from `llama-grammar.cpp`.
     */
    fun printStacks(): String {
        val sb = StringBuilder()
        sb.appendLine("stacks (${stacks.size}):")
        for ((i, stack) in stacks.withIndex()) {
            sb.append("  [$i]: ")
            for (pos in stack) {
                val elem = rules[pos.ruleIndex][pos.elementIndex]
                sb.append("${elem.type}(${elem.value}) ")
            }
            sb.appendLine(if (stack.isEmpty()) "(empty)" else "")
        }
        return sb.toString()
    }
}

// ---------------------------------------------------------------------------
// Legacy GBNFGrammar wrapper (preserves the original public API)
// ---------------------------------------------------------------------------

/**
 * GBNF grammar parser and constraint engine.
 *
 * This class preserves the original simplified API from the first iteration of
 * `Grammar.kt`. Internally it delegates to [LlamaGrammar] and
 * [LlamaGrammarParser] for the heavy lifting.
 *
 * For new code, prefer [LlamaGrammar.fromString] directly.
 */
class GBNFGrammar(
    val rules: List<GrammarRule>,
    val rootRuleId: Int = 0
) {
    /** The underlying full grammar engine, lazily built from [rules]. */
    private val engine: LlamaGrammar? by lazy {
        val converted = rules.map { gr ->
            gr.elements.map { e ->
                LlamaGrammarElement(e.type, e.value)
            }
        }
        LlamaGrammar.init(converted, rootRuleId)
    }

    companion object {
        /**
         * Parse a GBNF grammar string into rules.
         *
         * This uses the full [LlamaGrammarParser] under the hood and then
         * wraps the result in the legacy [GBNFGrammar] shape.
         */
        fun parse(grammarText: String): GBNFGrammar {
            val parser = LlamaGrammarParser()
            if (!parser.parse(grammarText) || parser.rules.isEmpty()) {
                return GBNFGrammar(emptyList(), 0)
            }
            val rootId = parser.symbolIds.entries
                .minByOrNull { it.value }?.value?.toInt() ?: 0

            val grammarRules = parser.rules.map { rule ->
                GrammarRule(rule.map { elem ->
                    LlamaGrammarElement(elem.type, elem.value)
                })
            }
            return GBNFGrammar(grammarRules, rootId)
        }
    }

    /**
     * Filter tokens based on grammar constraints.
     * Returns a list of valid token IDs given the current grammar state.
     */
    fun filterTokens(
        currentStates: List<GrammarState>,
        tokenizer: Tokenizer,
        candidates: Array<TokenData>
    ): Array<TokenData> {
        val validTokens = mutableSetOf<Int>()

        for (state in currentStates) {
            val validForState = getValidTokensForState(state, tokenizer)
            validTokens.addAll(validForState)
        }

        return candidates.filter { it.id in validTokens }.toTypedArray()
    }

    private fun getValidTokensForState(state: GrammarState, tokenizer: Tokenizer): Set<Int> {
        val validTokens = mutableSetOf<Int>()
        if (state.ruleId >= rules.size) return validTokens
        val rule = rules[state.ruleId]
        if (state.position >= rule.elements.size) return validTokens
        val element = rule.elements[state.position]

        when (element.type) {
            LlamaGretype.CHAR -> {
                val char = element.value.toInt().toChar()
                for (tokenId in 0 until tokenizer.getVocabSize()) {
                    val tokenText = tokenizer.decode(tokenId)
                    if (tokenText.startsWith(char)) validTokens.add(tokenId)
                }
            }
            LlamaGretype.RULE_REF -> {
                val referencedRule = GrammarState(element.value.toInt(), 0)
                validTokens.addAll(getValidTokensForState(referencedRule, tokenizer))
            }
            LlamaGretype.ALT -> {
                val nextState = GrammarState(state.ruleId, state.position + 1)
                validTokens.addAll(getValidTokensForState(nextState, tokenizer))
            }
            LlamaGretype.END -> { /* rule complete */ }
            else -> {
                for (tokenId in 0 until tokenizer.getVocabSize()) validTokens.add(tokenId)
            }
        }
        return validTokens
    }

    /**
     * Update grammar states after consuming a token.
     */
    fun updateStates(
        currentStates: List<GrammarState>,
        tokenId: Int,
        tokenizer: Tokenizer
    ): List<GrammarState> {
        val newStates = mutableListOf<GrammarState>()
        val tokenText = tokenizer.decode(tokenId)
        for (state in currentStates) {
            newStates.addAll(processToken(state, tokenText))
        }
        return newStates.distinct()
    }

    private fun processToken(state: GrammarState, tokenText: String): List<GrammarState> {
        val nextStates = mutableListOf<GrammarState>()
        if (state.ruleId >= rules.size) return nextStates
        val rule = rules[state.ruleId]
        if (state.position >= rule.elements.size) return nextStates
        val element = rule.elements[state.position]

        when (element.type) {
            LlamaGretype.CHAR -> {
                val expectedChar = element.value.toInt().toChar()
                if (tokenText.startsWith(expectedChar)) {
                    nextStates.add(GrammarState(state.ruleId, state.position + 1))
                }
            }
            LlamaGretype.RULE_REF -> {
                nextStates.add(GrammarState(element.value.toInt(), 0))
            }
            LlamaGretype.END -> { /* completed */ }
            else -> {
                nextStates.add(GrammarState(state.ruleId, state.position + 1))
            }
        }
        return nextStates
    }

    fun getInitialStates(): List<GrammarState> =
        listOf(GrammarState(rootRuleId, 0))

    fun isComplete(states: List<GrammarState>): Boolean = states.any { state ->
        if (state.ruleId < rules.size) {
            val rule = rules[state.ruleId]
            state.position >= rule.elements.size ||
                (state.position < rule.elements.size &&
                    rule.elements[state.position].type == LlamaGretype.END)
        } else false
    }
}

// ---------------------------------------------------------------------------
// Tokenizer interface and simple implementation
// ---------------------------------------------------------------------------

/**
 * Tokenizer interface for grammar constraints.
 *
 * Implementations provide the bridge between the token-id domain and the
 * text-string domain that the grammar operates over.
 */
interface Tokenizer {
    fun encode(text: String): List<Int>
    fun decode(tokenId: Int): String
    fun decode(tokenIds: List<Int>): String
    fun getVocabSize(): Int
}

/**
 * Simple character-level tokenizer for testing.
 *
 * Each unique character or punctuation mark gets its own token id.
 * Not suitable for production use.
 */
class SimpleTokenizer : Tokenizer {
    private val vocab = mutableMapOf<String, Int>()
    private val reverseVocab = mutableMapOf<Int, String>()
    private var nextId = 0

    init {
        addToken(" ")
        addToken("\n")
        for (c in 'a'..'z') addToken(c.toString())
        for (c in 'A'..'Z') addToken(c.toString())
        for (c in '0'..'9') addToken(c.toString())
        addToken("."); addToken(","); addToken("!"); addToken("?"); addToken(":"); addToken(";")
    }

    private fun addToken(token: String) {
        if (token !in vocab) {
            vocab[token] = nextId
            reverseVocab[nextId] = token
            nextId++
        }
    }

    override fun encode(text: String): List<Int> {
        val result = mutableListOf<Int>()
        for (char in text) {
            val token = char.toString()
            addToken(token)
            result.add(vocab[token]!!)
        }
        return result
    }

    override fun decode(tokenId: Int): String = reverseVocab[tokenId] ?: "<UNK>"
    override fun decode(tokenIds: List<Int>): String = tokenIds.joinToString("") { decode(it) }
    override fun getVocabSize(): Int = nextId
}

// ---------------------------------------------------------------------------
// Grammar-constrained sampling context
// ---------------------------------------------------------------------------

/**
 * Bridges a [GBNFGrammar] (or [LlamaGrammar]) with the sampling pipeline.
 *
 * Maintains the current set of grammar states and exposes helpers for
 * filtering candidates, accepting tokens, and checking completion.
 */
class GrammarSamplingContext(
    private val grammar: GBNFGrammar,
    private val tokenizer: Tokenizer
) {
    private var currentStates = grammar.getInitialStates()

    fun filterCandidates(candidates: Array<TokenData>): Array<TokenData> =
        grammar.filterTokens(currentStates, tokenizer, candidates)

    fun acceptToken(tokenId: Int) {
        currentStates = grammar.updateStates(currentStates, tokenId, tokenizer)
    }

    fun isComplete(): Boolean = grammar.isComplete(currentStates)

    fun reset() { currentStates = grammar.getInitialStates() }

    fun getCurrentStates(): List<GrammarState> = currentStates
}
