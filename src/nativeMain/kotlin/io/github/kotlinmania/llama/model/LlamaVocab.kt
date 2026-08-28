// port-lint: source llama.cpp/src/llama-vocab.h llama.cpp/src/llama-vocab.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.max
import kotlin.math.min

// ---------------------------------------------------------------------------
// Kotlin/Native priority queue (no java.util available)
// ---------------------------------------------------------------------------

/**
 * Minimal binary-heap priority queue for use in Kotlin/Native where
 * `java.util.PriorityQueue` is unavailable. Elements are ordered by their
 * [Comparable] natural order (smallest first).
 */
private class MinPriorityQueue<T : Comparable<T>> {
    private val heap = ArrayList<T>()

    val size: Int get() = heap.size
    fun isEmpty(): Boolean = heap.isEmpty()
    fun isNotEmpty(): Boolean = heap.isNotEmpty()

    fun add(element: T) {
        heap.add(element)
        siftUp(heap.size - 1)
    }

    fun poll(): T? {
        if (heap.isEmpty()) return null
        val result = heap[0]
        val last = heap.removeAt(heap.size - 1)
        if (heap.isNotEmpty()) {
            heap[0] = last
            siftDown(0)
        }
        return result
    }

    fun clear() { heap.clear() }

    private fun siftUp(idx: Int) {
        var i = idx
        while (i > 0) {
            val parent = (i - 1) / 2
            if (heap[i] < heap[parent]) {
                val tmp = heap[i]; heap[i] = heap[parent]; heap[parent] = tmp
                i = parent
            } else break
        }
    }

    private fun siftDown(idx: Int) {
        var i = idx
        while (true) {
            var smallest = i
            val left = 2 * i + 1
            val right = 2 * i + 2
            if (left < heap.size && heap[left] < heap[smallest]) smallest = left
            if (right < heap.size && heap[right] < heap[smallest]) smallest = right
            if (smallest == i) break
            val tmp = heap[i]; heap[i] = heap[smallest]; heap[smallest] = tmp
            i = smallest
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Sentinel value for "no token". Maps to LLAMA_TOKEN_NULL in C++. */
const val LLAMA_TOKEN_NULL: LlamaToken = -1

// ---------------------------------------------------------------------------
// Vocab type enums
// ---------------------------------------------------------------------------

/**
 * Tokenizer algorithm family. Maps to `llama_vocab_type` in C++.
 *
 * - **SPM** – SentencePiece Model (unigram / BPE with ▁ space encoding)
 * - **BPE** – Byte-Pair Encoding (GPT-2 style)
 * - **WPM** – WordPiece Model (BERT style)
 * - **UGM** – Unigram Model (T5 style)
 * - **RWKV** – RWKV trie-based tokenizer
 * - **PLAMO2** – PLaMo-2 custom tokenizer
 * - **NONE** – No vocabulary (dummy model)
 */
enum class LlamaVocabType(val value: Int) {
    NONE(0),
    SPM(1),
    BPE(2),
    WPM(3),
    UGM(4),
    RWKV(5),
    PLAMO2(6);

    companion object {
        fun fromValue(v: Int): LlamaVocabType = entries.first { it.value == v }
    }
}

/**
 * Pre-tokenization strategy used before the main tokenizer runs.
 * Maps to `llama_vocab_pre_type` in C++.
 *
 * Each variant corresponds to a model family's regex-based pre-split rules.
 */
enum class LlamaVocabPreType(val value: Int) {
    DEFAULT(0),
    LLAMA3(1),
    DEEPSEEK_LLM(2),
    DEEPSEEK_CODER(3),
    FALCON(4),
    MPT(5),
    STARCODER(6),
    GPT2(7),
    REFACT(8),
    COMMAND_R(9),
    STABLELM2(10),
    QWEN2(11),
    OLMO(12),
    DBRX(13),
    SMAUG(14),
    PORO(15),
    CHATGLM3(16),
    CHATGLM4(17),
    VIKING(18),
    JAIS(19),
    TEKKEN(20),
    SMOLLM(21),
    CODESHELL(22),
    BLOOM(23),
    GPT3_FINNISH(24),
    EXAONE(25),
    CHAMELEON(26),
    MINERVA(27),
    DEEPSEEK3_LLM(28),
    GPT4O(29),
    SUPERBPE(30),
    TRILLION(31),
    BAILINGMOE(32),
    LLAMA4(33),
    PIXTRAL(34),
    SEED_CODER(35),
    HUNYUAN(36),
    KIMI_K2(37),
    HUNYUAN_DENSE(38),
    GROK_2(39),
    GRANITE_DOCLING(40),
    MINIMAX_M2(41),
    AFMOE(42),
    SOLAR_OPEN(43),
    YOUTU(44),
    EXAONE_MOE(45),
    QWEN35(46),
    TINY_AYA(47),
    JOYAI_LLM(48),
    JAIS2(49),
    GEMMA4(50);

    companion object {
        fun fromValue(v: Int): LlamaVocabPreType = entries.firstOrNull { it.value == v } ?: DEFAULT
    }
}

// ---------------------------------------------------------------------------
// Token type & attribute enums
// ---------------------------------------------------------------------------

/**
 * Token type classification stored in GGUF metadata.
 * Maps to `llama_token_type` in C++ (`LLAMA_TOKEN_TYPE_*`).
 */
enum class LlamaTokenType(val value: Int) {
    UNDEFINED(0),
    NORMAL(1),
    UNKNOWN(2),
    CONTROL(3),
    USER_DEFINED(4),
    UNUSED(5),
    BYTE(6);

    companion object {
        fun fromValue(v: Int): LlamaTokenType = entries.firstOrNull { it.value == v } ?: UNDEFINED
    }
}

/**
 * Bit-flag attributes attached to each token. Multiple flags can be OR'd
 * together. Maps to `llama_token_attr` in C++ (`LLAMA_TOKEN_ATTR_*`).
 */
object LlamaTokenAttr {
    const val UNDEFINED: Int = 0
    const val UNKNOWN: Int = 1 shl 0
    const val UNUSED: Int = 1 shl 1
    const val NORMAL: Int = 1 shl 2
    const val CONTROL: Int = 1 shl 3
    const val USER_DEFINED: Int = 1 shl 4
    const val BYTE: Int = 1 shl 5
    const val LSTRIP: Int = 1 shl 6
    const val RSTRIP: Int = 1 shl 7
    const val NORMALIZED: Int = 1 shl 6
    const val SINGLE_WORD: Int = 1 shl 9
}

// ---------------------------------------------------------------------------
// Token data
// ---------------------------------------------------------------------------

/**
 * Per-token metadata stored in the vocabulary.
 *
 * @property text  The token's surface text (e.g. "hello", "<s>", "<0x0A>").
 * @property score Log-probability score used by unigram / SPM tokenizers.
 * @property attr  Bit-field of [LlamaTokenAttr] flags.
 */
data class LlamaTokenData(
    var text: String,
    var score: Float = 0.0f,
    var attr: Int = LlamaTokenAttr.NORMAL
)

// ---------------------------------------------------------------------------
// Helper: Naive Trie
// ---------------------------------------------------------------------------

/**
 * Simple trie data structure used for prefix matching in tokenizers.
 * Ported from `naive_trie` in llama-vocab.cpp.
 */
class NaiveTrie {
    var hasValue: Boolean = false
    var value: LlamaToken = 0
    val children: HashMap<Char, NaiveTrie> = HashMap()

    fun insert(key: String, tokenValue: LlamaToken = 0) {
        insert(key, 0, tokenValue)
    }

    private fun insert(key: String, offset: Int, tokenValue: LlamaToken) {
        if (offset == key.length) {
            hasValue = true
            value = tokenValue
            return
        }
        val c = key[offset]
        val child = children.getOrPut(c) { NaiveTrie() }
        child.insert(key, offset + 1, tokenValue)
    }

    /**
     * Returns the length of the longest prefix of [key] (starting at [offset])
     * that exists in this trie.
     */
    fun getLongestPrefixLength(key: String, offset: Int = 0): Int {
        var node: NaiveTrie = this
        var longest = 0
        var pos = offset
        while (pos < key.length) {
            val child = node.children[key[pos]] ?: break
            pos++
            if (child.hasValue) longest = pos - offset
            node = child
        }
        return longest
    }

    fun traverse(c: Char): NaiveTrie? = children[c]
}

// ---------------------------------------------------------------------------
// Helper: LLM Symbol (used by SPM/BPE tokenizers)
// ---------------------------------------------------------------------------

/**
 * A lightweight symbol used during tokenization. Tracks a substring of the
 * input via [textStart] and [length] indices (or direct [text] reference)
 * plus doubly-linked-list pointers ([prev] / [next]).
 */
data class LlmSymbol(
    var prev: Int = -1,
    var next: Int = -1,
    var text: String = "",
    var textStart: Int = 0,
    var length: Int = 0
) {
    /** Convenience: get the substring this symbol represents. */
    fun piece(): String = if (length <= text.length - textStart) {
        text.substring(textStart, textStart + length)
    } else {
        text
    }
}

// ---------------------------------------------------------------------------
// Fragment buffer for special-token partitioning
// ---------------------------------------------------------------------------

/**
 * During tokenization the input is first split into fragments: either
 * already-resolved special tokens or raw text that still needs tokenization.
 * Maps to `fragment_buffer_variant` in C++.
 */
sealed class FragmentBufferVariant {
    data class Token(val token: LlamaToken) : FragmentBufferVariant()
    data class RawText(val rawText: String, val offset: Int, val length: Int) : FragmentBufferVariant()
}

// ---------------------------------------------------------------------------
// Tokenizer interface
// ---------------------------------------------------------------------------

/**
 * Abstract base for all tokenizer implementations.
 * Concrete subclasses (SPM, BPE, WPM, UGM, RWKV) live below.
 */
abstract class LlmTokenizer

/**
 * SPM (SentencePiece) tokenizer minimal. The actual merge algorithm lives in
 * [LlmTokenizerSpmSession.tokenize].
 */
class LlmTokenizerSpm : LlmTokenizer()

/**
 * BPE tokenizer. Stores the regex expressions used for pre-tokenization and
 * whether GPT-2 style byte encoding is active.
 */
class LlmTokenizerBpe(
    val regexExprs: List<String>,
    val byteEncode: Boolean = true
) : LlmTokenizer()

/** WPM (WordPiece) tokenizer minimal. */
class LlmTokenizerWpm : LlmTokenizer()

/** UGM (Unigram) tokenizer minimal. */
class LlmTokenizerUgm : LlmTokenizer()

/** RWKV trie-based tokenizer minimal. */
class LlmTokenizerRwkv(val tokenMatcher: NaiveTrie = NaiveTrie()) : LlmTokenizer()

/** PLaMo-2 tokenizer minimal. */
class LlmTokenizerPlamo2 : LlmTokenizer()

// ---------------------------------------------------------------------------
// SPM tokenizer session
// ---------------------------------------------------------------------------

/**
 * Bigram entry for the SPM priority queue. Higher [score] → higher priority;
 * ties broken by lower [left] index.
 *
 * [compareTo] returns lower values for higher-priority items (since
 * [MinPriorityQueue] is a min-heap).
 */
private data class SpmBigram(
    val left: Int,
    val right: Int,
    val score: Float,
    val size: Int
) : Comparable<SpmBigram> {
    override fun compareTo(other: SpmBigram): Int {
        // higher score = higher priority = smaller compareTo value
        val cmp = other.score.compareTo(score)
        return if (cmp != 0) cmp else left.compareTo(other.left)
    }
}

/**
 * Session that runs the SPM merge algorithm for a single tokenization call.
 * Ported from `llm_tokenizer_spm_session` in C++.
 */
class LlmTokenizerSpmSession(private val vocab: LlamaVocab) {

    private val symbols = ArrayList<LlmSymbol>()
    private val workQueue = MinPriorityQueue<SpmBigram>()
    private val revMerge = HashMap<String, Pair<Int, Int>>()

    fun tokenize(text: String, output: MutableList<LlamaToken>) {
        symbols.clear()
        workQueue.clear()
        revMerge.clear()

        // split into UTF-8 code-point symbols
        var index = 0
        var offs = 0
        while (offs < text.length) {
            val cpLen = unicodeLenUtf16(text, offs)
            val sym = LlmSymbol(
                prev = index - 1,
                next = if (offs + cpLen >= text.length) -1 else index + 1,
                text = text,
                textStart = offs,
                length = cpLen
            )
            symbols.add(sym)
            offs += cpLen
            index++
        }

        // seed work queue with all possible 2-character bigrams
        for (i in 1 until symbols.size) {
            tryAddBigram(i - 1, i)
        }

        // keep merging the highest-scoring pair
        while (workQueue.isNotEmpty()) {
            val bigram = workQueue.poll()!!
            val leftSym = symbols[bigram.left]
            val rightSym = symbols[bigram.right]

            if (leftSym.length == 0 || rightSym.length == 0 ||
                leftSym.length + rightSym.length != bigram.size
            ) continue

            // merge right into left
            leftSym.length += rightSym.length
            rightSym.length = 0

            leftSym.next = rightSym.next
            if (rightSym.next >= 0) {
                symbols[rightSym.next].prev = bigram.left
            }

            tryAddBigram(leftSym.prev, bigram.left)
            tryAddBigram(bigram.left, leftSym.next)
        }

        // collect result
        var i = 0
        while (i != -1) {
            resegment(symbols[i], output)
            i = symbols[i].next
        }
    }

    private fun resegment(symbol: LlmSymbol, output: MutableList<LlamaToken>) {
        val text = symbol.piece()
        val token = vocab.textToToken(text)

        if (token != LLAMA_TOKEN_NULL) {
            output.add(token)
            return
        }

        val p = revMerge[text]
        if (p == null) {
            // output as byte tokens
            for (byte in text.encodeToByteArray()) {
                output.add(vocab.byteToToken(byte.toUByte().toInt()))
            }
            return
        }

        resegment(symbols[p.first], output)
        resegment(symbols[p.second], output)
    }

    private fun tryAddBigram(left: Int, right: Int) {
        if (left == -1 || right == -1) return

        val text = symbols[left].piece() + symbols[right].piece()
        val token = vocab.textToToken(text)
        if (token == LLAMA_TOKEN_NULL) return
        if (token.toUInt() >= vocab.nTokens().toUInt()) return

        val tokData = vocab.getTokenData(token)

        workQueue.add(SpmBigram(left, right, tokData.score, text.length))
        revMerge[text] = Pair(left, right)
    }
}

// ---------------------------------------------------------------------------
// BPE tokenizer session
// ---------------------------------------------------------------------------

/**
 * Bigram entry for the BPE priority queue. Lower [rank] → higher priority;
 * ties broken by lower [left] index.
 */
private data class BpeBigram(
    val left: Int,
    val right: Int,
    val text: String,
    val rank: Int,
    val size: Int
) : Comparable<BpeBigram> {
    override fun compareTo(other: BpeBigram): Int {
        val cmp = rank.compareTo(other.rank) // lower rank first
        return if (cmp != 0) cmp else left.compareTo(other.left) // lower left first
    }
}

/**
 * Session for BPE tokenization. Ported from `llm_tokenizer_bpe_session`.
 *
 * The full implementation includes regex-based word splitting (via
 * [LlmTokenizerBpe.regexExprs]) and iterative merge. The regex pre-split
 * is deferred – in production it would delegate to a Unicode-aware regex engine.
 */
class LlmTokenizerBpeSession(
    private val vocab: LlamaVocab,
    private val tokenizer: LlmTokenizerBpe
) {
    private val symbols = ArrayList<LlmSymbol>()
    private val symbolsFinal = ArrayList<LlmSymbol>()
    private val workQueue = MinPriorityQueue<BpeBigram>()

    fun appendBos(output: MutableList<LlamaToken>): Boolean {
        if (vocab.addBos) {
            require(vocab.tokenBos() != LLAMA_TOKEN_NULL)
            output.add(vocab.tokenBos())
            return true
        }
        return false
    }

    fun appendEos(output: MutableList<LlamaToken>): Boolean {
        if (vocab.addEos) {
            require(vocab.tokenEos() != LLAMA_TOKEN_NULL)
            output.add(vocab.tokenEos())
            return true
        }
        return false
    }

    fun checkDoubleBosEos(output: List<LlamaToken>) {
        if (vocab.addBos && output.size >= 2 && output[1] == vocab.tokenBos()) {
            println("WARNING: prompt starts with 2 BOS tokens")
        }
        if (vocab.addEos && output.size >= 2 && output[output.size - 2] == vocab.tokenEos()) {
            println("WARNING: prompt ends with 2 EOS tokens")
        }
    }

    /**
     * Tokenize a single pre-split word using BPE merge.
     *
     * NOTE: Full regex-based word splitting is not yet implemented.
     * This method processes the entire [text] as a single word.
     */
    fun tokenize(text: String, output: MutableList<LlamaToken>) {
        // LATER: implement unicode_regex_split for proper pre-tokenization
        // For now, treat the whole text as one word
        val words = listOf(text)

        var finalPrevIndex = -1
        symbolsFinal.clear()

        for (word in words) {
            workQueue.clear()
            symbols.clear()

            // check if the entire word is a known token and merges should be ignored
            if (vocab.ignoreMerges && vocab.textToToken(word) != LLAMA_TOKEN_NULL) {
                symbols.add(LlmSymbol(prev = -1, next = -1, text = word, textStart = 0, length = word.length))
            } else {
                // split word into UTF-16 code-point symbols
                var index = 0
                var offset = 0
                while (offset < word.length) {
                    val cpLen = unicodeLenUtf16(word, offset)
                    symbols.add(LlmSymbol(
                        prev = index - 1,
                        next = if (offset + cpLen >= word.length) -1 else index + 1,
                        text = word,
                        textStart = offset,
                        length = cpLen
                    ))
                    offset += cpLen
                    index++
                }
            }

            // seed work queue
            for (i in 1 until symbols.size) {
                addNewBigram(i - 1, i)
            }

            // iterative merge
            while (workQueue.isNotEmpty()) {
                val bigram = workQueue.poll()!!
                val leftSym = symbols[bigram.left]
                val rightSym = symbols[bigram.right]

                if (leftSym.length == 0 || rightSym.length == 0) continue
                if (leftSym.piece() + rightSym.piece() != bigram.text) continue

                leftSym.length += rightSym.length
                rightSym.length = 0

                leftSym.next = rightSym.next
                if (rightSym.next >= 0) {
                    symbols[rightSym.next].prev = bigram.left
                }

                addNewBigram(leftSym.prev, bigram.left)
                addNewBigram(bigram.left, leftSym.next)
            }

            // collect into symbolsFinal preserving order
            for (sym in symbols) {
                if (sym.length > 0) {
                    val newSym = sym.copy(prev = finalPrevIndex, next = -1)
                    if (finalPrevIndex != -1) {
                        symbolsFinal[finalPrevIndex].next = symbolsFinal.size
                    }
                    symbolsFinal.add(newSym)
                    finalPrevIndex = symbolsFinal.size - 1
                }
            }
        }

        // resolve final symbols to token IDs
        if (symbolsFinal.isNotEmpty()) {
            var i = 0
            while (i != -1) {
                val sym = symbolsFinal[i]
                if (sym.length == 0) { i = sym.next; continue }

                val str = sym.piece()
                val token = vocab.textToToken(str)

                if (token == LLAMA_TOKEN_NULL) {
                    // fall back to byte tokens
                    for (byte in str.encodeToByteArray()) {
                        if (tokenizer.byteEncode) {
                            val byteStr = byte.toInt().toChar().toString()
                            val bt = vocab.textToToken(byteStr)
                            if (bt != LLAMA_TOKEN_NULL) output.add(bt)
                        } else {
                            val hex = "0123456789ABCDEF"
                            val ch = byte.toUByte().toInt()
                            val buf = "<0x${hex[ch shr 4]}${hex[ch and 0x0F]}>"
                            val bt = vocab.textToToken(buf)
                            if (bt != LLAMA_TOKEN_NULL) output.add(bt)
                        }
                    }
                } else {
                    output.add(token)
                }
                i = sym.next
            }
        }
    }

    private fun addNewBigram(left: Int, right: Int) {
        if (left == -1 || right == -1) return
        val leftToken = symbols[left].piece()
        val rightToken = symbols[right].piece()

        val rank = vocab.findBpeRank(leftToken, rightToken)
        if (rank < 0) return

        workQueue.add(BpeBigram(
            left = left,
            right = right,
            text = leftToken + rightToken,
            rank = rank,
            size = leftToken.length + rightToken.length
        ))
    }
}

// ---------------------------------------------------------------------------
// WPM tokenizer session (minimal)
// ---------------------------------------------------------------------------

class LlmTokenizerWpmSession(private val vocab: LlamaVocab) {
    fun tokenize(text: String, output: MutableList<LlamaToken>) {
    }
}

// ---------------------------------------------------------------------------
// UGM tokenizer session (minimal)
// ---------------------------------------------------------------------------

class LlmTokenizerUgmSession(private val vocab: LlamaVocab) {
    fun tokenize(text: String, output: MutableList<LlamaToken>) {
    }
}

// ---------------------------------------------------------------------------
// RWKV tokenizer session (minimal)
// ---------------------------------------------------------------------------

class LlmTokenizerRwkvSession(
    private val vocab: LlamaVocab,
    private val tokenizer: LlmTokenizerRwkv
) {
    fun tokenize(text: String, output: MutableList<LlamaToken>) {
    }
}

// ---------------------------------------------------------------------------
// PLaMo-2 tokenizer session (minimal)
// ---------------------------------------------------------------------------

class LlmTokenizerPlamo2Session {
    fun tokenize(text: String, output: MutableList<LlamaToken>) {
    }
}

// ---------------------------------------------------------------------------
// Unicode helpers
// ---------------------------------------------------------------------------

/**
 * Returns the number of Kotlin [Char]s (UTF-16 code units) consumed by the
 * Unicode code point starting at [offset] in [text]. Handles surrogate pairs.
 */
internal fun unicodeLenUtf16(text: String, offset: Int): Int {
    if (offset >= text.length) return 0
    return if (text[offset].isHighSurrogate() && offset + 1 < text.length && text[offset + 1].isLowSurrogate()) 2 else 1
}

/** Escaped space symbol – U+2581 (Lower One Eighth Block) used by SPM. */
private const val ESCAPED_SPACE = "\u2581"

private fun escapeWhitespace(text: String): String = text.replace(" ", ESCAPED_SPACE)
private fun unescapeWhitespace(text: String): String = text.replace(ESCAPED_SPACE, " ")

// ---------------------------------------------------------------------------
// Main vocabulary class
// ---------------------------------------------------------------------------

/**
 * The vocabulary and tokenizer for a llama.cpp-compatible model.
 *
 * This is the Kotlin port of `llama_vocab` from `llama-vocab.h/.cpp`. It
 * stores the token table, special-token IDs, BPE merge ranks, and drives
 * tokenization / detokenization.
 *
 * ### Lifecycle
 * 1. Construct with default constructor.
 * 2. Populate via [load] (reads GGUF metadata) **or** by manually filling
 *    [idToToken] / [tokenToId] and calling [initTokenizer].
 * 3. Call [tokenize] / [detokenize] as needed.
 */
class LlamaVocab {

    // ----- public state ----------------------------------------------------

    var type: LlamaVocabType = LlamaVocabType.SPM
    var preType: LlamaVocabPreType = LlamaVocabPreType.DEFAULT

    var tokenizerModel: String = ""
    var tokenizerPre: String = ""

    var nTokenTypes: Int = 0

    /** Maximum byte length of any token text. Used by WPM for search window. */
    var maxTokenLen: Int = 0

    // ----- token table -----------------------------------------------------

    /** Indexed by token ID → metadata. */
    val idToToken: ArrayList<LlamaTokenData> = ArrayList()

    /** Surface text → token ID reverse lookup. */
    val tokenToId: HashMap<String, LlamaToken> = HashMap()

    // ----- special tokens --------------------------------------------------

    var specialBosId: LlamaToken = 1
    var specialEosId: LlamaToken = 2
    var specialEotId: LlamaToken = LLAMA_TOKEN_NULL
    var specialEomId: LlamaToken = LLAMA_TOKEN_NULL
    var specialUnkId: LlamaToken = 0
    var specialSepId: LlamaToken = LLAMA_TOKEN_NULL
    var specialPadId: LlamaToken = LLAMA_TOKEN_NULL
    var specialMaskId: LlamaToken = LLAMA_TOKEN_NULL
    var linefeedId: LlamaToken = 13

    // FIM tokens
    var specialFimPreId: LlamaToken = LLAMA_TOKEN_NULL
    var specialFimSufId: LlamaToken = LLAMA_TOKEN_NULL
    var specialFimMidId: LlamaToken = LLAMA_TOKEN_NULL
    var specialFimPadId: LlamaToken = LLAMA_TOKEN_NULL
    var specialFimRepId: LlamaToken = LLAMA_TOKEN_NULL
    var specialFimSepId: LlamaToken = LLAMA_TOKEN_NULL

    // ----- tokenizer flags -------------------------------------------------

    var addSpacePrefix: Boolean = false
    var addBos: Boolean = false
    var addEos: Boolean = false
    var addSep: Boolean = false
    var ignoreMerges: Boolean = false
    var cleanSpaces: Boolean = false
    var removeExtraWhitespaces: Boolean = false
    var escapeWhitespaces: Boolean = true
    var treatWhitespaceAsSuffix: Boolean = false

    // ----- caches & internal state -----------------------------------------

    /** BPE merge ranks: (left, right) → rank index. */
    val bpeRanks: HashMap<Pair<String, String>, Int> = HashMap()

    /** Token IDs that signal end-of-generation. */
    val specialEogIds: MutableSet<LlamaToken> = mutableSetOf()

    /** Sorted list of special token IDs (longest text first). */
    val cacheSpecialTokens: ArrayList<LlamaToken> = ArrayList()

    /** Precomputed token → piece cache (index = token ID). */
    val cacheTokenToPiece: ArrayList<String> = ArrayList()

    /** Precompiled charsmap for UGM tokenizer. */
    val precompiledCharsmap: ArrayList<Byte> = ArrayList()

    /** The active tokenizer implementation. */
    var tokenizer: LlmTokenizer? = null
        private set

    // -----------------------------------------------------------------------
    // Query helpers
    // -----------------------------------------------------------------------

    fun nTokens(): Int = idToToken.size

    fun typeName(): String = when (type) {
        LlamaVocabType.NONE -> "no vocab"
        LlamaVocabType.SPM -> "SPM"
        LlamaVocabType.BPE -> "BPE"
        LlamaVocabType.WPM -> "WPM"
        LlamaVocabType.UGM -> "UGM"
        LlamaVocabType.RWKV -> "RWKV"
        LlamaVocabType.PLAMO2 -> "PLaMo2"
    }

    // ----- Token attribute queries -----------------------------------------

    fun isNormal(id: LlamaToken): Boolean {
        require(type != LlamaVocabType.NONE)
        return (idToToken[id].attr and LlamaTokenAttr.NORMAL) != 0
    }

    fun isUnknown(id: LlamaToken): Boolean {
        require(type != LlamaVocabType.NONE)
        return (idToToken[id].attr and LlamaTokenAttr.UNKNOWN) != 0
    }

    fun isControl(id: LlamaToken): Boolean {
        require(type != LlamaVocabType.NONE)
        return (idToToken[id].attr and LlamaTokenAttr.CONTROL) != 0
    }

    fun isByte(id: LlamaToken): Boolean {
        require(type != LlamaVocabType.NONE)
        return (idToToken[id].attr and LlamaTokenAttr.BYTE) != 0
    }

    fun isUserDefined(id: LlamaToken): Boolean {
        require(type != LlamaVocabType.NONE)
        return (idToToken[id].attr and LlamaTokenAttr.USER_DEFINED) != 0
    }

    fun isUnused(id: LlamaToken): Boolean {
        require(type != LlamaVocabType.NONE)
        return (idToToken[id].attr and LlamaTokenAttr.UNUSED) != 0
    }

    fun isEog(id: LlamaToken): Boolean = id != LLAMA_TOKEN_NULL && id in specialEogIds

    // ----- Token ↔ byte conversion -----------------------------------------

    /**
     * Decodes a byte-type token (e.g. `<0x0A>`) into its byte value.
     */
    fun tokenToByte(id: LlamaToken): Int {
        require(type != LlamaVocabType.NONE)
        require(isByte(id)) { "Token $id is not a byte token" }
        val text = idToToken[id].text
        // Format is <0xHH>
        return text.substring(3, 5).toInt(16)
    }

    /**
     * Finds the token ID for a given raw byte value. Behaviour depends on
     * vocab type (SPM uses `<0xHH>` format, BPE uses Unicode byte mapping).
     */
    fun byteToToken(ch: Int): LlamaToken {
        require(type != LlamaVocabType.NONE)
        val hex = "0123456789ABCDEF"
        return when (type) {
            LlamaVocabType.SPM, LlamaVocabType.UGM -> {
                val buf = "<0x${hex[(ch shr 4) and 0xF]}${hex[ch and 0xF]}>"
                tokenToId[buf] ?: throw IllegalStateException("Byte token '$buf' not found in vocab")
            }
            LlamaVocabType.WPM, LlamaVocabType.BPE -> {
                // GPT-2 byte-to-unicode mapping (simplified)
                // In a full implementation, this would use the unicode_byte_to_utf8 table
                val buf = "<0x${hex[(ch shr 4) and 0xF]}${hex[ch and 0xF]}>"
                tokenToId[buf]
                    ?: tokenToId[ch.toChar().toString()]
                    ?: throw IllegalStateException("Byte token for 0x${ch.toString(16)} not found")
            }
            LlamaVocabType.PLAMO2 -> {
                val buf = "<0x${hex[(ch shr 4) and 0xF]}${hex[ch and 0xF]}>"
                tokenToId[buf] ?: throw IllegalStateException("Byte token '$buf' not found")
            }
            else -> throw IllegalStateException("byteToToken not supported for $type")
        }
    }

    // ----- Token ↔ text / piece conversion ---------------------------------

    fun tokenGetAttr(id: LlamaToken): Int {
        require(type != LlamaVocabType.NONE)
        return idToToken[id].attr
    }

    fun getTokenData(id: LlamaToken): LlamaTokenData {
        require(type != LlamaVocabType.NONE)
        return idToToken[id]
    }

    fun tokenGetText(id: LlamaToken): String = idToToken[id].text
    fun tokenGetScore(id: LlamaToken): Float = idToToken[id].score

    /**
     * Look up a token ID by its surface text. Returns [LLAMA_TOKEN_NULL] if
     * not found.
     */
    fun textToToken(text: String): LlamaToken {
        require(type != LlamaVocabType.NONE)
        return tokenToId[text] ?: LLAMA_TOKEN_NULL
    }

    // ----- Special token accessors -----------------------------------------

    fun tokenBos(): LlamaToken = specialBosId
    fun tokenEos(): LlamaToken = specialEosId
    fun tokenEot(): LlamaToken = specialEotId
    fun tokenEom(): LlamaToken = specialEomId
    fun tokenUnk(): LlamaToken = specialUnkId
    fun tokenSep(): LlamaToken = specialSepId
    fun tokenNl(): LlamaToken = linefeedId
    fun tokenPad(): LlamaToken = specialPadId
    fun tokenMask(): LlamaToken = specialMaskId

    fun tokenPrefix(): LlamaToken = specialFimPreId
    fun tokenMiddle(): LlamaToken = specialFimMidId
    fun tokenSuffix(): LlamaToken = specialFimSufId

    fun tokenFimPre(): LlamaToken = specialFimPreId
    fun tokenFimSuf(): LlamaToken = specialFimSufId
    fun tokenFimMid(): LlamaToken = specialFimMidId
    fun tokenFimPad(): LlamaToken = specialFimPadId
    fun tokenFimRep(): LlamaToken = specialFimRepId
    fun tokenFimSep(): LlamaToken = specialFimSepId

    // ----- BPE rank lookup -------------------------------------------------

    /**
     * Returns the merge rank for the pair ([tokenLeft], [tokenRight]),
     * or -1 if no merge exists.
     */
    fun findBpeRank(tokenLeft: String, tokenRight: String): Int {
        return bpeRanks[Pair(tokenLeft, tokenRight)] ?: -1
    }

    // -----------------------------------------------------------------------
    // Tokenizer initialization
    // -----------------------------------------------------------------------

    /**
     * Creates the tokenizer implementation matching [type]. Call this after
     * populating the token table.
     */
    fun initTokenizer(vocabType: LlamaVocabType = type) {
        tokenizer = when (vocabType) {
            LlamaVocabType.SPM -> LlmTokenizerSpm()
            LlamaVocabType.BPE -> createBpeTokenizer()
            LlamaVocabType.WPM -> LlmTokenizerWpm()
            LlamaVocabType.UGM -> LlmTokenizerUgm()
            LlamaVocabType.RWKV -> LlmTokenizerRwkv()
            LlamaVocabType.PLAMO2 -> LlmTokenizerPlamo2()
            LlamaVocabType.NONE -> throw IllegalStateException("Cannot init tokenizer for NONE vocab")
        }
    }

    /**
     * Creates a [LlmTokenizerBpe] with regex expressions chosen by [preType].
     * Maps the large `switch` in `llm_tokenizer_bpe::llm_tokenizer_bpe()`.
     */
    private fun createBpeTokenizer(): LlmTokenizerBpe {
        val regexExprs: List<String>
        var byteEncode = true

        when (preType) {
            LlamaVocabPreType.LLAMA3,
            LlamaVocabPreType.DBRX,
            LlamaVocabPreType.SMAUG,
            LlamaVocabPreType.CHATGLM4 -> {
                regexExprs = listOf(
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                )
            }
            LlamaVocabPreType.GPT2,
            LlamaVocabPreType.MPT,
            LlamaVocabPreType.OLMO,
            LlamaVocabPreType.JAIS,
            LlamaVocabPreType.TRILLION,
            LlamaVocabPreType.GRANITE_DOCLING -> {
                regexExprs = listOf(
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
                )
            }
            LlamaVocabPreType.STABLELM2,
            LlamaVocabPreType.QWEN2,
            LlamaVocabPreType.HUNYUAN,
            LlamaVocabPreType.SOLAR_OPEN -> {
                regexExprs = listOf(
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                )
            }
            LlamaVocabPreType.STARCODER,
            LlamaVocabPreType.REFACT,
            LlamaVocabPreType.COMMAND_R,
            LlamaVocabPreType.SMOLLM,
            LlamaVocabPreType.CODESHELL,
            LlamaVocabPreType.EXAONE,
            LlamaVocabPreType.MINERVA -> {
                regexExprs = listOf(
                    "\\p{N}",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
                )
            }
            LlamaVocabPreType.FALCON -> {
                regexExprs = listOf(
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "[0-9][0-9][0-9]"
                )
            }
            LlamaVocabPreType.PORO,
            LlamaVocabPreType.BLOOM,
            LlamaVocabPreType.GPT3_FINNISH -> {
                regexExprs = listOf(" ?[^(\\s|.,!?…。，、।۔،)]+")
            }
            LlamaVocabPreType.VIKING -> {
                regexExprs = listOf(
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                    "\\p{N}"
                )
            }
            LlamaVocabPreType.GEMMA4 -> {
                regexExprs = listOf("[^\\n]+|[\\n]+")
                byteEncode = false
            }
            LlamaVocabPreType.SUPERBPE -> {
                regexExprs = listOf("\\p{N}+", "(?=(\\d{3})+(?!\\d))")
            }
            else -> {
                // default regex for BPE
                regexExprs = listOf(
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "\\p{N}+",
                    "[0-9][0-9][0-9]"
                )
            }
        }

        return LlmTokenizerBpe(regexExprs, byteEncode)
    }

    // -----------------------------------------------------------------------
    // Special-token text partitioning
    // -----------------------------------------------------------------------

    /**
     * Splits the input [buffer] by recognized special tokens. Each special
     * token is replaced by a [FragmentBufferVariant.Token] node; surrounding
     * text remains as [FragmentBufferVariant.RawText] nodes.
     */
    fun tokenizerStPartition(
        buffer: MutableList<FragmentBufferVariant>,
        parseSpecial: Boolean
    ) {
        for (specialId in cacheSpecialTokens) {
            val data = getTokenData(specialId)
            val text = data.text

            if (!parseSpecial && (data.attr and (LlamaTokenAttr.CONTROL or LlamaTokenAttr.UNKNOWN)) != 0) {
                continue
            }

            var i = 0
            while (i < buffer.size) {
                val fragment = buffer[i]
                if (fragment !is FragmentBufferVariant.RawText) {
                    i++
                    continue
                }

                val rawText = fragment.rawText
                val baseOffset = fragment.offset
                val baseLength = fragment.length
                val searchArea = rawText.substring(baseOffset, baseOffset + baseLength)
                val matchPos = searchArea.indexOf(text)

                if (matchPos == -1) {
                    i++
                    continue
                }

                val absMatch = baseOffset + matchPos

                // remove the current fragment
                buffer.removeAt(i)

                // insert: left text (if any), special token, right text (if any)
                var insertIdx = i

                // left
                if (absMatch > baseOffset) {
                    var leftLen = absMatch - baseOffset
                    if ((data.attr and LlamaTokenAttr.LSTRIP) != 0) {
                        while (leftLen > 0 && rawText[baseOffset + leftLen - 1] == ' ') leftLen--
                    }
                    if (leftLen > 0) {
                        buffer.add(insertIdx, FragmentBufferVariant.RawText(rawText, baseOffset, leftLen))
                        insertIdx++
                    }
                }

                // special token
                buffer.add(insertIdx, FragmentBufferVariant.Token(specialId))
                insertIdx++

                // right
                val rightStart = absMatch + text.length
                val rightEnd = baseOffset + baseLength
                if (rightStart < rightEnd) {
                    var rOffset = rightStart
                    var rLen = rightEnd - rightStart
                    if ((data.attr and LlamaTokenAttr.RSTRIP) != 0) {
                        while (rLen > 0 && rawText[rOffset] == ' ') {
                            rOffset++
                            rLen--
                        }
                    }
                    if (rLen > 0) {
                        buffer.add(insertIdx, FragmentBufferVariant.RawText(rawText, rOffset, rLen))
                    }
                }

                // don't advance i – re-examine the same position in case there
                // are more occurrences within the right fragment
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tokenize
    // -----------------------------------------------------------------------

    /**
     * Tokenizes [rawText] into a list of token IDs.
     *
     * @param addSpecial If true, prepends BOS / appends EOS (or SEP) tokens
     *                   as configured by the vocab metadata.
     * @param parseSpecial If true, special tokens appearing in the text are
     *                     recognized and replaced before tokenization.
     */
    fun tokenize(
        rawText: String,
        addSpecial: Boolean,
        parseSpecial: Boolean = false
    ): List<LlamaToken> {
        requireNotNull(tokenizer) { "Tokenizer not initialized. Call initTokenizer() first." }

        val output = ArrayList<LlamaToken>()
        val fragmentBuffer = ArrayList<FragmentBufferVariant>()

        if (rawText.isNotEmpty()) {
            fragmentBuffer.add(FragmentBufferVariant.RawText(rawText, 0, rawText.length))
            tokenizerStPartition(fragmentBuffer, parseSpecial)
        }

        when (type) {
            LlamaVocabType.SPM -> {
                var isPrevSpecial = true

                if (addSpecial && addBos) {
                    require(specialBosId != LLAMA_TOKEN_NULL)
                    output.add(specialBosId)
                    isPrevSpecial = true
                }

                for (fragment in fragmentBuffer) {
                    when (fragment) {
                        is FragmentBufferVariant.RawText -> {
                            var text = ""
                            if (addSpacePrefix && isPrevSpecial) text = " "
                            text += fragment.rawText.substring(fragment.offset, fragment.offset + fragment.length)
                            val escaped = escapeWhitespace(text)
                            LlmTokenizerSpmSession(this).tokenize(escaped, output)
                            isPrevSpecial = false
                        }
                        is FragmentBufferVariant.Token -> {
                            output.add(fragment.token)
                            isPrevSpecial = true
                        }
                    }
                }

                if (addSpecial && addEos) {
                    require(specialEosId != LLAMA_TOKEN_NULL)
                    output.add(specialEosId)
                }
            }

            LlamaVocabType.BPE -> {
                val bpeTok = tokenizer as LlmTokenizerBpe
                val session = LlmTokenizerBpeSession(this, bpeTok)
                if (addSpecial) session.appendBos(output)

                for (fragment in fragmentBuffer) {
                    when (fragment) {
                        is FragmentBufferVariant.RawText -> {
                            var text = fragment.rawText.substring(fragment.offset, fragment.offset + fragment.length)
                            if (escapeWhitespaces) text = escapeWhitespace(text)
                            session.tokenize(text, output)
                        }
                        is FragmentBufferVariant.Token -> output.add(fragment.token)
                    }
                }

                if (addSpecial) {
                    session.appendEos(output)
                    session.checkDoubleBosEos(output)
                }
            }

            LlamaVocabType.WPM -> {
                if (addSpecial) {
                    require(specialBosId != LLAMA_TOKEN_NULL)
                    output.add(specialBosId)
                }
                for (fragment in fragmentBuffer) {
                    when (fragment) {
                        is FragmentBufferVariant.RawText -> {
                            val text = fragment.rawText.substring(fragment.offset, fragment.offset + fragment.length)
                            LlmTokenizerWpmSession(this).tokenize(text, output)
                        }
                        is FragmentBufferVariant.Token -> output.add(fragment.token)
                    }
                }
                if (addSpecial) {
                    require(specialSepId != LLAMA_TOKEN_NULL)
                    output.add(specialSepId)
                }
            }

            LlamaVocabType.UGM -> {
                if (addSpecial && addBos) {
                    require(specialBosId != LLAMA_TOKEN_NULL)
                    output.add(specialBosId)
                }
                for (fragment in fragmentBuffer) {
                    when (fragment) {
                        is FragmentBufferVariant.RawText -> {
                            val text = fragment.rawText.substring(fragment.offset, fragment.offset + fragment.length)
                            LlmTokenizerUgmSession(this).tokenize(text, output)
                        }
                        is FragmentBufferVariant.Token -> output.add(fragment.token)
                    }
                }
                if (addSpecial && addEos) {
                    require(specialEosId != LLAMA_TOKEN_NULL)
                    output.add(specialEosId)
                }
            }

            LlamaVocabType.RWKV -> {
                val rwkvTok = tokenizer as LlmTokenizerRwkv
                for (fragment in fragmentBuffer) {
                    when (fragment) {
                        is FragmentBufferVariant.RawText -> {
                            val text = fragment.rawText.substring(fragment.offset, fragment.offset + fragment.length)
                            LlmTokenizerRwkvSession(this, rwkvTok).tokenize(text, output)
                        }
                        is FragmentBufferVariant.Token -> output.add(fragment.token)
                    }
                }
            }

            LlamaVocabType.PLAMO2 -> {
                for (fragment in fragmentBuffer) {
                    when (fragment) {
                        is FragmentBufferVariant.RawText -> {
                            val text = fragment.rawText.substring(fragment.offset, fragment.offset + fragment.length)
                            LlmTokenizerPlamo2Session().tokenize(text, output)
                        }
                        is FragmentBufferVariant.Token -> output.add(fragment.token)
                    }
                }
            }

            LlamaVocabType.NONE -> throw IllegalStateException("Cannot tokenize with NONE vocab")
        }

        return output
    }

    // -----------------------------------------------------------------------
    // Token → piece
    // -----------------------------------------------------------------------

    /**
     * Converts a single [token] ID into its UTF-8 text representation.
     *
     * @param lstrip Number of leading spaces to strip.
     * @param special If false, control / unknown tokens produce empty strings.
     * @return The piece string, or empty if the token is suppressed.
     */
    fun tokenToPiece(token: LlamaToken, lstrip: Int = 0, special: Boolean = true): String {
        val attrSpecial = LlamaTokenAttr.UNKNOWN or LlamaTokenAttr.CONTROL
        val attr = tokenGetAttr(token)
        if (!special && (attr and attrSpecial) != 0) return ""

        if (cacheTokenToPiece.isNotEmpty()) {
            var result = cacheTokenToPiece[token]
            if (lstrip > 0) {
                var stripped = 0
                while (stripped < lstrip && result.isNotEmpty() && result[0] == ' ') {
                    result = result.substring(1)
                    stripped++
                }
            }
            return result
        }

        if (token < 0 || token >= idToToken.size) return ""
        val tokenText = idToToken[token].text

        return when (type) {
            LlamaVocabType.SPM, LlamaVocabType.WPM, LlamaVocabType.UGM -> {
                when {
                    (attr and (attrSpecial or LlamaTokenAttr.USER_DEFINED)) != 0 -> tokenText
                    (attr and LlamaTokenAttr.NORMAL) != 0 -> unescapeWhitespace(tokenText)
                    (attr and LlamaTokenAttr.BYTE) != 0 -> {
                        val b = tokenToByte(token)
                        byteArrayOf(b.toByte()).decodeToString()
                    }
                    else -> ""
                }
            }
            LlamaVocabType.BPE -> {
                when {
                    (attr and (attrSpecial or LlamaTokenAttr.USER_DEFINED)) != 0 -> tokenText
                    (attr and LlamaTokenAttr.NORMAL) != 0 -> {
                        if (escapeWhitespaces) unescapeWhitespace(tokenText) else tokenText
                    }
                    (attr and LlamaTokenAttr.BYTE) != 0 -> {
                        val b = tokenToByte(token)
                        byteArrayOf(b.toByte()).decodeToString()
                    }
                    else -> ""
                }
            }
            LlamaVocabType.RWKV -> {
                // LATER: implement llama_unescape_rwkv_token
                tokenText
            }
            LlamaVocabType.PLAMO2 -> {
                if (isByte(token) && tokenText.length == 6 &&
                    tokenText.startsWith("<0x") && tokenText.endsWith(">")
                ) {
                    val hexVal = tokenText.substring(3, 5).toInt(16)
                    byteArrayOf(hexVal.toByte()).decodeToString()
                } else {
                    tokenText
                }
            }
            LlamaVocabType.NONE -> ""
        }
    }

    /**
     * Returns the cached piece string for [token].
     * The cache must have been built (e.g. during [load]).
     */
    fun tokenToPieceCached(token: LlamaToken): String = cacheTokenToPiece[token]

    // -----------------------------------------------------------------------
    // Detokenize
    // -----------------------------------------------------------------------

    /**
     * Converts a list of token IDs back into text.
     *
     * @param removeSpecial Strip BOS/EOS tokens.
     * @param unparseSpecial Render special tokens as their text form.
     */
    fun detokenize(
        tokens: List<LlamaToken>,
        removeSpecial: Boolean = false,
        unparseSpecial: Boolean = false
    ): String {
        if (type == LlamaVocabType.NONE) return ""

        val sb = StringBuilder()
        var tokenList = tokens
        var removeSpace = addSpacePrefix

        if (removeSpecial && addBos && tokenList.isNotEmpty() && tokenList[0] == specialBosId) {
            removeSpace = false
            tokenList = tokenList.subList(1, tokenList.size)
        }
        if (removeSpecial && addEos && tokenList.isNotEmpty() && tokenList.last() == specialEosId) {
            tokenList = tokenList.subList(0, tokenList.size - 1)
        }

        for (token in tokenList) {
            var piece = tokenToPiece(token, if (removeSpace) 1 else 0, unparseSpecial)
            removeSpace = false
            sb.append(piece)
        }

        var result = sb.toString()

        if (cleanSpaces) {
            result = cleanUpSpaces(result)
        }

        return result
    }

    /**
     * Applies the same whitespace cleanup as the C++ `clean_spaces` passes:
     * removes spaces before `?!.,`, strips single apostrophes between spaces,
     * and collapses apostrophe contractions.
     */
    private fun cleanUpSpaces(text: String): String {
        if (text.isEmpty()) return text
        val sb = StringBuilder()

        // pass 1: remove space before ?!.,
        sb.append(text[0])
        for (i in 1 until text.length) {
            val x = text[i]
            if (text[i - 1] == ' ' && (x == '?' || x == '!' || x == '.' || x == ',')) {
                sb.deleteAt(sb.length - 1)
            }
            sb.append(x)
        }

        // pass 2: strip single apostrophe between spaces
        val pass1 = sb.toString()
        sb.clear()
        if (pass1.isNotEmpty()) sb.append(pass1[0])
        var i = 1
        while (i < pass1.length) {
            val x = pass1[i]
            if (x == '\'' && i + 1 < pass1.length && pass1[i - 1] == ' ' && pass1[i + 1] == ' ') {
                sb.deleteAt(sb.length - 1) // remove prev space
                i++ // skip next space
                i++
                continue
            }
            sb.append(x)
            i++
        }

        // pass 3: apostrophe contractions – remove space before 's, 'm, 're, 've
        val pass2 = sb.toString()
        sb.clear()
        if (pass2.isNotEmpty()) sb.append(pass2[0])
        for (j in 1 until pass2.length) {
            val x = pass2[j]
            if (pass2[j - 1] == ' ' && x == '\'' && j + 1 < pass2.length) {
                val x1 = pass2[j + 1]
                if (x1 == 's' || x1 == 'm') {
                    sb.deleteAt(sb.length - 1) // remove space
                } else if (j + 2 < pass2.length) {
                    val x2 = pass2[j + 2]
                    if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {
                        sb.deleteAt(sb.length - 1)
                    }
                }
            }
            sb.append(x)
        }

        return sb.toString()
    }

    // -----------------------------------------------------------------------
    // Load from GGUF metadata (minimal)
    // -----------------------------------------------------------------------

    /**
     * Populates this vocabulary from GGUF metadata.
     *
     * In a full implementation this reads token lists, scores, types, merges,
     * and special token IDs from the GGUF file. The core logic is ported from
     * `llama_vocab::impl::load()` in C++.
     *
     * @param tokenizerModelStr  The `tokenizer.ggml.model` value (e.g. "llama", "gpt2", "bert").
     * @param tokenizerPreStr    The `tokenizer.ggml.pre` value (e.g. "llama3", "default").
     * @param tokenTexts         List of token surface texts indexed by token ID.
     * @param tokenScores        Parallel list of scores, or null if unavailable.
     * @param tokenTypes         Parallel list of [LlamaTokenType] ordinals, or null.
     * @param merges             BPE merge strings in "left right" format, or null.
     */
    fun load(
        tokenizerModelStr: String,
        tokenizerPreStr: String = "",
        tokenTexts: List<String>,
        tokenScores: FloatArray? = null,
        tokenTypes: IntArray? = null,
        merges: List<String>? = null
    ) {
        tokenizerModel = tokenizerModelStr
        tokenizerPre = tokenizerPreStr

        // determine vocab type from model string
        when (tokenizerModelStr) {
            "no_vocab", "none" -> {
                type = LlamaVocabType.NONE
                specialBosId = LLAMA_TOKEN_NULL
                specialEosId = LLAMA_TOKEN_NULL
                specialUnkId = LLAMA_TOKEN_NULL
                return
            }
            "llama" -> {
                type = LlamaVocabType.SPM
                specialBosId = 1; specialEosId = 2; specialUnkId = 0
            }
            "bert" -> {
                type = LlamaVocabType.WPM
                specialBosId = 101; specialEosId = LLAMA_TOKEN_NULL
                specialUnkId = 100; specialSepId = 102
                specialPadId = 0; specialMaskId = 103
                addSep = true
            }
            "gpt2" -> {
                type = LlamaVocabType.BPE
                specialBosId = 11; specialEosId = 11
                specialUnkId = LLAMA_TOKEN_NULL
            }
            "t5" -> {
                type = LlamaVocabType.UGM
                specialBosId = LLAMA_TOKEN_NULL; specialEosId = 1
                specialUnkId = 2; specialPadId = 0
            }
            "rwkv" -> {
                type = LlamaVocabType.RWKV
                specialBosId = LLAMA_TOKEN_NULL; specialEosId = LLAMA_TOKEN_NULL
                specialUnkId = LLAMA_TOKEN_NULL
            }
            "plamo2" -> {
                type = LlamaVocabType.PLAMO2
                specialBosId = 1; specialEosId = 2; specialUnkId = 0; specialPadId = 3
            }
            "gemma4" -> {
                type = LlamaVocabType.BPE
                specialBosId = LLAMA_TOKEN_NULL; specialEosId = LLAMA_TOKEN_NULL
                specialUnkId = LLAMA_TOKEN_NULL
                tokenizerPre = "gemma4"
            }
            else -> throw IllegalArgumentException("Unknown tokenizer model: '$tokenizerModelStr'")
        }

        // set default flags based on type
        when (type) {
            LlamaVocabType.BPE -> {
                addSpacePrefix = false; escapeWhitespaces = false; cleanSpaces = true
                resolvePreType(tokenizerPreStr)
            }
            LlamaVocabType.SPM -> {
                preType = LlamaVocabPreType.DEFAULT
                addSpacePrefix = true; cleanSpaces = false; addBos = true; addEos = false
            }
            LlamaVocabType.WPM -> {
                preType = LlamaVocabPreType.DEFAULT
                addSpacePrefix = false; cleanSpaces = true; addBos = true; addEos = false; addSep = true
            }
            LlamaVocabType.UGM -> {
                preType = LlamaVocabPreType.DEFAULT; addBos = false; addEos = true
            }
            LlamaVocabType.RWKV -> {
                preType = LlamaVocabPreType.DEFAULT
                addSpacePrefix = false; cleanSpaces = false; addBos = false; addEos = false
            }
            else -> { preType = LlamaVocabPreType.DEFAULT }
        }

        // populate BPE merge ranks
        merges?.forEachIndexed { idx, word ->
            val pos = word.indexOf(' ', 1)
            if (pos != -1) {
                val first = word.substring(0, pos)
                val second = word.substring(pos + 1)
                bpeRanks[Pair(first, second)] = idx
            }
        }

        // populate token table
        idToToken.ensureCapacity(tokenTexts.size)
        for (i in tokenTexts.indices) {
            var word = tokenTexts[i]
            if (word.isEmpty()) word = "[EMPTY_$i]"

            tokenToId[word] = i
            maxTokenLen = max(maxTokenLen, word.length)

            val td = LlamaTokenData(
                text = word,
                score = tokenScores?.getOrElse(i) { 0.0f } ?: 0.0f,
                attr = LlamaTokenAttr.NORMAL
            )

            if (tokenTypes != null) {
                td.attr = when (LlamaTokenType.fromValue(tokenTypes[i])) {
                    LlamaTokenType.UNKNOWN -> LlamaTokenAttr.UNKNOWN
                    LlamaTokenType.UNUSED -> LlamaTokenAttr.UNUSED
                    LlamaTokenType.NORMAL -> LlamaTokenAttr.NORMAL
                    LlamaTokenType.CONTROL -> LlamaTokenAttr.CONTROL
                    LlamaTokenType.USER_DEFINED -> LlamaTokenAttr.USER_DEFINED
                    LlamaTokenType.BYTE -> LlamaTokenAttr.BYTE
                    LlamaTokenType.UNDEFINED -> LlamaTokenAttr.UNDEFINED
                }
            }

            idToToken.add(td)
        }

        // auto-detect special tokens by text
        autoDetectSpecialTokens()

        // init tokenizer
        initTokenizer(type)

        // build special tokens cache
        buildSpecialTokensCache()

        // build token-to-piece cache
        buildTokenToPieceCache()
    }

    /**
     * Resolves [preType] from the tokenizer pre-type string. Mirrors the large
     * `if/else` chain in the C++ `load()` method.
     */
    private fun resolvePreType(pre: String) {
        preType = when {
            pre.isEmpty() || pre == "default" -> LlamaVocabPreType.DEFAULT
            pre in listOf("llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano") -> {
                ignoreMerges = true; addBos = true; LlamaVocabPreType.LLAMA3
            }
            pre == "deepseek-llm" -> { cleanSpaces = false; LlamaVocabPreType.DEEPSEEK_LLM }
            pre == "deepseek-coder" -> { cleanSpaces = false; LlamaVocabPreType.DEEPSEEK_CODER }
            pre == "deepseek-v3" -> { cleanSpaces = false; LlamaVocabPreType.DEEPSEEK3_LLM }
            pre == "falcon" -> LlamaVocabPreType.FALCON
            pre == "mpt" -> LlamaVocabPreType.MPT
            pre == "starcoder" -> LlamaVocabPreType.STARCODER
            pre in listOf("gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "a.x-4.0", "mellum", "modern-bert") ->
                LlamaVocabPreType.GPT2
            pre in listOf("qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2") -> { cleanSpaces = false; LlamaVocabPreType.QWEN2 }
            pre == "qwen35" -> { cleanSpaces = false; LlamaVocabPreType.QWEN35 }
            pre == "stablelm2" -> LlamaVocabPreType.STABLELM2
            pre == "olmo" -> LlamaVocabPreType.OLMO
            pre == "dbrx" -> LlamaVocabPreType.DBRX
            pre == "smaug-bpe" -> LlamaVocabPreType.SMAUG
            pre == "poro-chat" -> { cleanSpaces = false; LlamaVocabPreType.PORO }
            pre in listOf("glm4", "chatglm-bpe") -> { specialBosId = LLAMA_TOKEN_NULL; LlamaVocabPreType.CHATGLM4 }
            pre == "viking" -> { cleanSpaces = false; LlamaVocabPreType.VIKING }
            pre == "jais" -> LlamaVocabPreType.JAIS
            pre == "tekken" -> { cleanSpaces = false; ignoreMerges = true; addBos = true; LlamaVocabPreType.TEKKEN }
            pre == "smollm" -> { cleanSpaces = false; LlamaVocabPreType.SMOLLM }
            pre == "codeshell" -> LlamaVocabPreType.CODESHELL
            pre == "bloom" -> LlamaVocabPreType.BLOOM
            pre == "gpt3-finnish" -> LlamaVocabPreType.GPT3_FINNISH
            pre == "exaone" -> LlamaVocabPreType.EXAONE
            pre == "exaone4" -> LlamaVocabPreType.GPT2
            pre == "exaone-moe" -> LlamaVocabPreType.EXAONE_MOE
            pre == "chameleon" -> { addBos = true; cleanSpaces = false; LlamaVocabPreType.CHAMELEON }
            pre == "minerva-7b" -> LlamaVocabPreType.MINERVA
            pre == "megrez" -> LlamaVocabPreType.QWEN2
            pre in listOf("gpt-4o", "llama4", "kanana2") -> { cleanSpaces = false; LlamaVocabPreType.GPT4O }
            pre == "tiny_aya" -> { cleanSpaces = false; LlamaVocabPreType.TINY_AYA }
            pre == "superbpe" -> { cleanSpaces = false; LlamaVocabPreType.SUPERBPE }
            pre == "trillion" -> { cleanSpaces = false; LlamaVocabPreType.TRILLION }
            pre == "granite-docling" -> { cleanSpaces = false; LlamaVocabPreType.GRANITE_DOCLING }
            pre in listOf("bailingmoe", "bailingmoe2", "llada-moe") -> { cleanSpaces = false; LlamaVocabPreType.BAILINGMOE }
            pre == "seed-coder" -> { cleanSpaces = false; LlamaVocabPreType.SEED_CODER }
            pre == "hunyuan" -> { cleanSpaces = false; LlamaVocabPreType.HUNYUAN }
            pre == "hunyuan-dense" -> { cleanSpaces = false; LlamaVocabPreType.HUNYUAN_DENSE }
            pre == "joyai-llm" -> { cleanSpaces = false; LlamaVocabPreType.JOYAI_LLM }
            pre == "kimi-k2" -> { cleanSpaces = false; LlamaVocabPreType.KIMI_K2 }
            pre == "grok-2" -> { cleanSpaces = false; LlamaVocabPreType.GROK_2 }
            pre == "afmoe" -> { cleanSpaces = false; LlamaVocabPreType.AFMOE }
            pre == "minimax-m2" -> { cleanSpaces = false; LlamaVocabPreType.MINIMAX_M2 }
            pre == "solar-open" -> { cleanSpaces = false; LlamaVocabPreType.SOLAR_OPEN }
            pre == "youtu" -> { cleanSpaces = false; ignoreMerges = true; LlamaVocabPreType.YOUTU }
            pre == "jais-2" -> LlamaVocabPreType.JAIS2
            pre == "gemma4" -> { escapeWhitespaces = true; LlamaVocabPreType.GEMMA4 }
            pre in listOf("jina-v1-en", "jina-v2-code", "roberta-bpe") -> { addSep = true; LlamaVocabPreType.GPT2 }
            pre == "refact" -> LlamaVocabPreType.REFACT
            pre == "command-r" -> { cleanSpaces = false; LlamaVocabPreType.COMMAND_R }
            else -> {
                println("WARNING: unknown pre-tokenizer type: '$pre'")
                LlamaVocabPreType.DEFAULT
            }
        }
    }

    /**
     * Auto-detect EOT, EOM, FIM, and EOG tokens by scanning token texts.
     * Ported from the large auto-detection block in C++ `load()`.
     */
    private fun autoDetectSpecialTokens() {
        val eotPatterns = listOf(
            "<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
            "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]",
            "<\uFF5Cend\u2581of\u2581sentence\uFF5C>", "<end_of_utterance>"
        )
        val eomPatterns = listOf("<|eom_id|>")
        val fimPrePatterns = listOf(
            "<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>",
            "<\uFF5Cfim\u2581begin\uFF5C>", "<PRE>", "\u2581<PRE>",
            "<|code_prefix|>", "<|prefix|>"
        )
        val fimSufPatterns = listOf(
            "<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>",
            "<\uFF5Cfim\u2581hole\uFF5C>", "<SUF>", "\u2581<SUF>",
            "<|code_suffix|>", "<|suffix|>"
        )
        val fimMidPatterns = listOf(
            "<|fim_middle|>", "<fim-middle>", "<fim_middle>",
            "<\uFF5Cfim\u2581end\uFF5C>", "<MID>", "\u2581<MID>",
            "<|code_middle|>", "<|middle|>"
        )
        val fimPadPatterns = listOf("<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]")
        val fimRepPatterns = listOf("<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>")
        val fimSepPatterns = listOf("<|file_sep|>")
        val eogPatterns = listOf(
            "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>",
            "<|calls|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>",
            "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
            "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
            "<\uFF5Cend\u2581of\u2581sentence\uFF5C>"
        )

        for ((text, id) in tokenToId) {
            if (specialEotId == LLAMA_TOKEN_NULL && text in eotPatterns) {
                specialEotId = id
                ensureControlAttr(id)
            }
            if (specialEomId == LLAMA_TOKEN_NULL && text in eomPatterns) {
                specialEomId = id
                ensureControlAttr(id)
            }
            if (specialFimPreId == LLAMA_TOKEN_NULL && text in fimPrePatterns) {
                specialFimPreId = id; ensureControlAttr(id)
            }
            if (specialFimSufId == LLAMA_TOKEN_NULL && text in fimSufPatterns) {
                specialFimSufId = id; ensureControlAttr(id)
            }
            if (specialFimMidId == LLAMA_TOKEN_NULL && text in fimMidPatterns) {
                specialFimMidId = id; ensureControlAttr(id)
            }
            if (specialFimPadId == LLAMA_TOKEN_NULL && text in fimPadPatterns) {
                specialFimPadId = id; ensureControlAttr(id)
            }
            if (specialFimRepId == LLAMA_TOKEN_NULL && text in fimRepPatterns) {
                specialFimRepId = id; ensureControlAttr(id)
            }
            if (specialFimSepId == LLAMA_TOKEN_NULL && text in fimSepPatterns) {
                specialFimSepId = id; ensureControlAttr(id)
            }
            if (text in eogPatterns) {
                specialEogIds.add(id)
                ensureControlAttr(id)
            }
        }

        // ensure EOS, EOT, EOM are in EOG set
        if (specialEosId != LLAMA_TOKEN_NULL) specialEogIds.add(specialEosId)
        if (specialEotId != LLAMA_TOKEN_NULL) specialEogIds.add(specialEotId)
        if (specialEomId != LLAMA_TOKEN_NULL) specialEogIds.add(specialEomId)

        // auto-detect unused tokens
        for ((text, id) in tokenToId) {
            val attr = idToToken[id].attr
            if ((attr and LlamaTokenAttr.CONTROL) != 0 && "unused" in text) {
                idToToken[id].attr = attr or LlamaTokenAttr.UNUSED
            }
        }
    }

    private fun ensureControlAttr(id: LlamaToken) {
        val td = idToToken[id]
        if ((td.attr and LlamaTokenAttr.CONTROL) == 0) {
            td.attr = td.attr or LlamaTokenAttr.CONTROL
        }
    }

    private fun buildSpecialTokensCache() {
        cacheSpecialTokens.clear()
        for (id in 0 until nTokens()) {
            val attr = idToToken[id].attr
            if ((attr and (LlamaTokenAttr.CONTROL or LlamaTokenAttr.USER_DEFINED or LlamaTokenAttr.UNKNOWN)) != 0) {
                cacheSpecialTokens.add(id)
            }
        }
        // sort by text length descending (longest match first)
        cacheSpecialTokens.sortByDescending { idToToken[it].text.length }
    }

    private fun buildTokenToPieceCache() {
        cacheTokenToPiece.clear()
        cacheTokenToPiece.ensureCapacity(nTokens())
        for (id in 0 until nTokens()) {
            cacheTokenToPiece.add(tokenToPiece(id, special = true))
        }
    }

    // -----------------------------------------------------------------------
    // Debug / info
    // -----------------------------------------------------------------------

    /**
     * Prints a human-readable summary of the vocabulary, matching the
     * C++ `llama_vocab::impl::print_info()` output.
     */
    fun printInfo() {
        println("vocab type            = ${typeName()}")
        println("n_vocab               = ${nTokens()}")
        println("n_merges              = ${bpeRanks.size}")

        fun showToken(label: String, id: LlamaToken) {
            if (id != LLAMA_TOKEN_NULL) println("$label = $id '${idToToken[id].text}'")
        }

        showToken("BOS token            ", specialBosId)
        showToken("EOS token            ", specialEosId)
        showToken("EOT token            ", specialEotId)
        showToken("EOM token            ", specialEomId)
        showToken("UNK token            ", specialUnkId)
        showToken("SEP token            ", specialSepId)
        showToken("PAD token            ", specialPadId)
        showToken("MASK token           ", specialMaskId)
        showToken("LF token             ", linefeedId)
        showToken("FIM PRE token        ", specialFimPreId)
        showToken("FIM SUF token        ", specialFimSufId)
        showToken("FIM MID token        ", specialFimMidId)
        showToken("FIM PAD token        ", specialFimPadId)
        showToken("FIM REP token        ", specialFimRepId)
        showToken("FIM SEP token        ", specialFimSepId)

        for (id in specialEogIds) {
            println("EOG token             = $id '${idToToken[id].text}'")
        }

        println("max token length      = $maxTokenLen")
    }
}
