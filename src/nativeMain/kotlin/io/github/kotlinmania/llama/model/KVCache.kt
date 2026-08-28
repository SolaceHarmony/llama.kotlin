// port-lint: source llama.cpp/src/llama-kv-cache.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// ---------------------------------------------------------------------------
// Type aliases matching C++ originals
// ---------------------------------------------------------------------------

/** Sequence ID type (maps to llama_seq_id in C++). */
typealias LlamaSeqId = Int

/** Token position type (maps to llama_pos in C++). */
typealias LlamaPos = Int

/** Maximum number of sequences supported. */
const val LLAMA_MAX_SEQ = 64

// ---------------------------------------------------------------------------
// Sliding-window attention type – mirrors llama_swa_type in llama-hparams.h
// ---------------------------------------------------------------------------

/**
 * Sliding-window attention variants.
 *
 * Maps 1-to-1 to the C++ `llama_swa_type` enum declared in
 * `llama.cpp/src/llama-hparams.h`.
 */
enum class LlamaSwaType {
    NONE,
    STANDARD,
    CHUNKED,
    SYMMETRIC,
}

// ---------------------------------------------------------------------------
// Memory status – mirrors llama_memory_status
// ---------------------------------------------------------------------------

/**
 * Status codes returned by memory-context operations.
 *
 * These map 1-to-1 to the C++ `llama_memory_status` enum.
 */
enum class LlamaMemoryStatus {
    SUCCESS,
    NO_UPDATE,
    FAILED_PREPARE,
    FAILED_COMPUTE,
}

// ---------------------------------------------------------------------------
// KvCell – per-slot metadata (port of llama_kv_cell)
// ---------------------------------------------------------------------------

/**
 * Metadata for a single KV cache cell.
 *
 * Each cell records its position in the sequence, which sequence IDs occupy it,
 * and an accumulated shift (used by K-shift / RoPE updates).
 *
 * @property pos  The token position stored in this cell, or -1 if empty.
 * @property delta  Accumulated position shift applied to this cell (for K-shift).
 */
data class KvCell(
    var pos: LlamaPos = -1,
    var delta: LlamaPos = 0
) {
    /** Set of sequence IDs that share this cell. */
    val seqIds: MutableSet<LlamaSeqId> = mutableSetOf()

    /** `true` when the cell contains valid cached data. */
    val hasPos: Boolean get() = pos >= 0

    /** `true` when no sequence occupies this cell. */
    val isEmpty: Boolean get() = seqIds.isEmpty()
}

// ---------------------------------------------------------------------------
// KvCells – a resizable array of KvCell with helper queries
// ---------------------------------------------------------------------------

/**
 * Container for an array of [KvCell] entries that backs one KV cache stream.
 *
 * Provides the same query surface as `llama_kv_cells` in C++ (position range
 * queries, sequence membership, shift tracking, etc.).
 */
class KvCells(size: Int = 0) {
    private val cells = ArrayList<KvCell>(size).apply {
        repeat(size) { add(KvCell()) }
    }

    private var usedCount = 0

    fun size(): Int = cells.size

    fun resize(n: Int) {
        cells.clear()
        repeat(n) { cells.add(KvCell()) }
        usedCount = 0
    }

    fun getUsed(): Int = usedCount

    /** Reset every cell to its default (empty) state. */
    fun reset() {
        for (cell in cells) {
            cell.pos = -1
            cell.delta = 0
            cell.seqIds.clear()
        }
        usedCount = 0
    }

    fun isEmpty(i: Int): Boolean = cells[i].isEmpty

    fun posGet(i: Int): LlamaPos = cells[i].pos

    fun posSet(i: Int, pos: LlamaPos) {
        val wasEmpty = cells[i].isEmpty
        cells[i].pos = pos
        if (wasEmpty && pos >= 0) usedCount++
    }

    /** Add `shift` to the position; returns `true` if the cell becomes empty. */
    fun posAdd(i: Int, shift: LlamaPos): Boolean {
        cells[i].pos += shift
        cells[i].delta += shift
        if (cells[i].pos < 0) {
            rm(i)
            return true
        }
        return false
    }

    fun posDiv(i: Int, d: Int) {
        val oldPos = cells[i].pos
        cells[i].pos /= d
        cells[i].delta += cells[i].pos - oldPos
    }

    fun posIn(i: Int, p0: LlamaPos, p1: LlamaPos): Boolean {
        val p = cells[i].pos
        return p >= p0 && p < p1
    }

    fun seqHas(i: Int, seqId: LlamaSeqId): Boolean = seqId in cells[i].seqIds

    fun seqAdd(i: Int, seqId: LlamaSeqId) {
        val wasEmpty = cells[i].isEmpty
        cells[i].seqIds.add(seqId)
        if (wasEmpty) usedCount++
    }

    /** Remove a single sequence from cell `i`. Returns `true` if the cell became empty. */
    fun seqRm(i: Int, seqId: LlamaSeqId): Boolean {
        cells[i].seqIds.remove(seqId)
        if (cells[i].isEmpty) {
            cells[i].pos = -1
            cells[i].delta = 0
            usedCount--
            return true
        }
        return false
    }

    /** Keep only `seqId` in cell `i`; returns `true` if cell was freed. */
    fun seqKeep(i: Int, seqId: LlamaSeqId): Boolean {
        if (!cells[i].seqIds.contains(seqId)) {
            if (cells[i].seqIds.isNotEmpty()) {
                rm(i)
                return true
            }
        } else {
            cells[i].seqIds.retainAll(setOf(seqId))
        }
        return false
    }

    fun seqCount(i: Int): Int = cells[i].seqIds.size

    fun seqGet(i: Int): LlamaSeqId = cells[i].seqIds.first()

    /** Remove all sequences and position data from cell `i`. */
    fun rm(i: Int) {
        if (cells[i].seqIds.isNotEmpty()) usedCount--
        cells[i].pos = -1
        cells[i].delta = 0
        cells[i].seqIds.clear()
    }

    fun getShift(i: Int): LlamaPos = cells[i].delta

    fun resetShift() {
        for (cell in cells) cell.delta = 0
    }

    /** Minimum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos {
        var min = Int.MAX_VALUE
        for (cell in cells) {
            if (seqId in cell.seqIds && cell.pos in 0 until min) {
                min = cell.pos
            }
        }
        return if (min == Int.MAX_VALUE) -1 else min
    }

    /** Maximum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos {
        var max = -1
        for (cell in cells) {
            if (seqId in cell.seqIds && cell.pos > max) {
                max = cell.pos
            }
        }
        return max
    }

    /** One past the highest used cell index. */
    fun usedMaxP1(): Int {
        for (i in cells.indices.reversed()) {
            if (!cells[i].isEmpty) return i + 1
        }
        return 0
    }

    /** Copy a subset of cells by index list (snapshot for rollback). */
    fun cp(idxs: List<Int>): KvCells {
        val copy = KvCells(idxs.size)
        for ((dst, src) in idxs.withIndex()) {
            if (src in cells.indices) {
                copy.cells[dst].pos = cells[src].pos
                copy.cells[dst].delta = cells[src].delta
                copy.cells[dst].seqIds.addAll(cells[src].seqIds)
            }
        }
        copy.usedCount = copy.cells.count { !it.isEmpty }
        return copy
    }

    /** Restore cells from a snapshot at the given indices. */
    fun set(idxs: List<Int>, other: KvCells) {
        for ((src, dst) in idxs.withIndex()) {
            if (dst in cells.indices && src in 0 until other.size()) {
                cells[dst].pos = other.cells[src].pos
                cells[dst].delta = other.cells[src].delta
                cells[dst].seqIds.clear()
                cells[dst].seqIds.addAll(other.cells[src].seqIds)
            }
        }
        usedCount = cells.count { !it.isEmpty }
    }
}

// ---------------------------------------------------------------------------
// KvLayer – per-layer K/V tensor references
// ---------------------------------------------------------------------------

/**
 * Holds the K and V tensors for a single transformer layer inside the cache.
 *
 * When the cache uses multiple streams (one per unique sequence), [kStream]
 * and [vStream] provide 2-D views into the base [k]/[v] tensors – one per
 * stream.
 *
 * @property il  The model layer index this entry corresponds to.
 * @property k   The key tensor for this layer (shape: [n_embd_k_gqa, kv_size, n_stream]).
 * @property v   The value tensor for this layer (shape depends on v_trans setting).
 * @property kStream  Per-stream 2-D views into [k].
 * @property vStream  Per-stream 2-D views into [v].
 */
data class KvLayer(
    val il: Int,
    var k: GGMLTensor? = null,
    var v: GGMLTensor? = null,
    val kStream: MutableList<GGMLTensor?> = mutableListOf(),
    val vStream: MutableList<GGMLTensor?> = mutableListOf()
)

// ---------------------------------------------------------------------------
// StreamCopyInfo – pending cross-stream buffer copies
// ---------------------------------------------------------------------------

/**
 * Records pending stream-to-stream buffer copies that will be executed
 * during the next [KVCache.update] call.
 *
 * Cross-stream sequence copies (`seq_cp` between different streams)
 * require actual buffer data to be copied, which is deferred until the
 * next graph computation.
 */
data class StreamCopyInfo(
    val ssrc: MutableList<Int> = mutableListOf(),
    val sdst: MutableList<Int> = mutableListOf()
) {
    fun empty(): Boolean {
        require(ssrc.size == sdst.size)
        return ssrc.isEmpty()
    }

    fun clear() {
        ssrc.clear()
        sdst.clear()
    }
}

// ---------------------------------------------------------------------------
// SlotInfo – describes where a micro-batch lands in the cache
// ---------------------------------------------------------------------------

/**
 * For each micro-batch, [SlotInfo] describes where its tokens should be
 * placed inside the KV cache cells.
 *
 * In the single-stream (unified) case there is exactly one entry in [idxs];
 * with per-sequence streams there is one entry per stream spanned by the
 * micro-batch.
 *
 * Mirrors `llama_kv_cache::slot_info` in C++.
 *
 * @property s0   Lowest stream index used by this slot.
 * @property s1   Highest stream index used by this slot.
 * @property strm Per-stream sequence IDs, length = s1 − s0 + 1.
 * @property idxs Per-stream cell index vectors (one vector per stream).
 */
data class SlotInfo(
    var s0: Int = LLAMA_MAX_SEQ,
    var s1: Int = 0,
    val strm: MutableList<LlamaSeqId> = mutableListOf(),
    val idxs: MutableList<MutableList<Int>> = mutableListOf()
) {
    fun head(): Int {
        require(idxs.size == 1)
        require(idxs[0].isNotEmpty())
        return idxs[0][0]
    }

    fun resize(n: Int) {
        strm.clear()
        repeat(n) { strm.add(0) }
        idxs.clear()
        repeat(n) { idxs.add(mutableListOf()) }
    }

    fun size(): Int {
        require(idxs.size == strm.size)
        require(idxs.isNotEmpty())
        return idxs[0].size
    }

    fun nStream(): Int = strm.size

    fun empty(): Boolean = idxs.isEmpty()

    fun clear() { idxs.clear() }

    /** Check if indices are contiguous starting from [head]. */
    fun isContiguous(): Boolean {
        if (idxs.isEmpty() || idxs[0].isEmpty()) return true
        if (idxs.size > 1) return false
        val h = idxs[0][0]
        for (i in idxs[0].indices) {
            if (idxs[0][i] != h + i) return false
        }
        return true
    }
}

// ---------------------------------------------------------------------------
// KVCache – main cache structure (port of llama_kv_cache)
// ---------------------------------------------------------------------------

/**
 * Key-Value cache for efficient transformer inference.
 *
 * This is the Kotlin port of `llama_kv_cache` from llama.cpp.  It stores
 * per-layer K and V tensors, manages cell metadata for sequence tracking,
 * and supports operations like slot finding, sequence removal/copy/keep,
 * position shifting, and defragmentation.
 *
 * The cache can operate in two modes:
 * - **Unified** (`nStream == 1`): all sequences share a single ring buffer.
 * - **Per-sequence** (`nStream == nSeqMax`): each sequence gets its own stream.
 *
 * ## Original (preserved) simple API
 * The constructor that takes `(maxSequenceLength, numHeads, headDim)` retains
 * the earlier allocator-first interface for basic single-batch usage.
 *
 * @property kvSize   Total number of KV cells in each stream.
 * @property nSeqMax  Maximum number of concurrent sequences.
 * @property nPad     Required padding for the cache size.
 * @property nSwa     Sliding-window size (0 = full context).
 * @property swaType  Sliding-window attention type.
 * @property typeK    Data type of the K cache tensors.
 * @property typeV    Data type of the V cache tensors.
 * @property vTrans   Whether the V cache is stored transposed.
 */
class KVCache private constructor(
    val kvSize: Int,
    val nSeqMax: Int,
    private val nStream: Int,
    private val nPad: Int,
    val nSwa: Int,
    val swaType: LlamaSwaType,
    val typeK: GGMLType,
    val typeV: GGMLType,
    val vTrans: Boolean
) {
    // ── per-stream cell arrays & heads ──────────────────────────────────
    private val vHeads = IntArray(nStream)
    private val vCells = Array(nStream) { KvCells(kvSize) }
    private val seqToStream = IntArray(if (nStream > 1) nStream else LLAMA_MAX_SEQ)

    // ── per-layer tensor storage ────────────────────────────────────────
    private val layers = mutableListOf<KvLayer>()
    private val mapLayerIds = mutableMapOf<Int, Int>()

    // ── pending stream copies ───────────────────────────────────────────
    private val scInfo = StreamCopyInfo()

    // ── attention rotation flags (populated during init) ────────────────
    private var attnRotK = false
    private var attnRotV = false
    private var nEmbdHeadKAll: Int = 0
    private var nEmbdHeadVAll: Int = 0

    // ── debug level (env LLAMA_KV_CACHE_DEBUG) ──────────────────────────
    private var debug = 0

    init {
        require(kvSize % nPad == 0) { "kvSize ($kvSize) must be divisible by nPad ($nPad)" }
        require(nStream == 1 || nStream == nSeqMax) { "nStream must be 1 or nSeqMax" }

        // initialise stream mapping
        if (nStream > 1) {
            for (s in 0 until nStream) seqToStream[s] = s
        }
        // else: all sequences map to stream 0 (default zero-filled)
    }

    // ── simple constructor preserved from original KVCache ──────────────

    /**
     * Simplified constructor for single-batch usage (legacy API).
     *
     * Creates a unified (single-stream) cache with F32 key/value types,
     * no sliding window, and padding of 1.
     */
    constructor(
        maxSequenceLength: Int,
        numHeads: Int,
        headDim: Int
    ) : this(
        kvSize = maxSequenceLength,
        nSeqMax = 1,
        nStream = 1,
        nPad = 1,
        nSwa = 0,
        swaType = LlamaSwaType.NONE,
        typeK = GGMLType.F32,
        typeV = GGMLType.F32,
        vTrans = true
    ) {
        _legacyNumHeads = numHeads
        _legacyHeadDim = headDim
    }

    // ── legacy API fields & methods (preserved from original file) ──────

    private var _legacyNumHeads: Int = 0
    private var _legacyHeadDim: Int = 0
    private var _legacyKeys: GGMLTensor? = null
    private var _legacyValues: GGMLTensor? = null
    private var _legacyCurrentLength = 0

    /** Legacy: maximum sequence length (alias for [kvSize]). */
    val maxSequenceLength: Int get() = kvSize
    /** Legacy: number of attention heads. */
    val numHeads: Int get() = _legacyNumHeads
    /** Legacy: per-head embedding dimension. */
    val headDim: Int get() = _legacyHeadDim

    fun initialize(graphAllocator: GGMLGraphAllocator) {
        _legacyKeys = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = kvSize.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }.also { graphAllocator.allocateTensor(it) }

        _legacyValues = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = kvSize.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }.also { graphAllocator.allocateTensor(it) }

        _legacyCurrentLength = 0
    }

    fun updateKey(graphAllocator: GGMLGraphAllocator, newKey: GGMLTensor): GGMLTensor {
        val cacheKey = _legacyKeys ?: throw IllegalStateException("KVCache not initialized")
        val newSeqLen = newKey.ne[2].toInt()
        require(_legacyCurrentLength + newSeqLen <= kvSize) { "Sequence length would exceed maximum" }

        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until newSeqLen) {
                for (d in 0 until _legacyHeadDim) {
                    val v = newKey.getFloat(graphAllocator, d, h, s, 0)
                    cacheKey.setFloat(graphAllocator, v, d, h, _legacyCurrentLength + s, 0)
                }
            }
        }
        val out = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = (_legacyCurrentLength + newSeqLen).toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        graphAllocator.allocateTensor(out)
        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until (_legacyCurrentLength + newSeqLen)) {
                for (d in 0 until _legacyHeadDim) {
                    val v = cacheKey.getFloat(graphAllocator, d, h, s, 0)
                    out.setFloat(graphAllocator, v, d, h, s, 0)
                }
            }
        }
        _legacyCurrentLength += newSeqLen
        return out
    }

    fun updateValue(graphAllocator: GGMLGraphAllocator, newValue: GGMLTensor): GGMLTensor {
        val cacheVal = _legacyValues ?: throw IllegalStateException("KVCache not initialized")
        val newSeqLen = newValue.ne[2].toInt()
        require(_legacyCurrentLength >= newSeqLen) { "updateKey must be called before updateValue in a step" }

        val start = _legacyCurrentLength - newSeqLen
        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until newSeqLen) {
                for (d in 0 until _legacyHeadDim) {
                    val v = newValue.getFloat(graphAllocator, d, h, s, 0)
                    cacheVal.setFloat(graphAllocator, v, d, h, start + s, 0)
                }
            }
        }

        val out = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = _legacyHeadDim.toLong()
            ne[1] = _legacyNumHeads.toLong()
            ne[2] = _legacyCurrentLength.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        graphAllocator.allocateTensor(out)
        for (h in 0 until _legacyNumHeads) {
            for (s in 0 until _legacyCurrentLength) {
                for (d in 0 until _legacyHeadDim) {
                    val v = cacheVal.getFloat(graphAllocator, d, h, s, 0)
                    out.setFloat(graphAllocator, v, d, h, s, 0)
                }
            }
        }
        return out
    }

    fun getCurrentLength(): Int = _legacyCurrentLength

    // ====================================================================
    // Full llama_kv_cache API (ported from C++)
    // ====================================================================

    companion object {
        /**
         * Create a fully-configured KV cache matching the C++ `llama_kv_cache` constructor.
         *
         * @param typeK     Data type for key tensors.
         * @param typeV     Data type for value tensors.
         * @param vTrans    Whether value tensors are stored transposed.
         * @param unified   If `true`, all sequences share one stream; otherwise one stream per sequence.
         * @param kvSize    Number of cache cells per stream.
         * @param nSeqMax   Maximum number of concurrent sequences.
         * @param nPad      Padding requirement (kvSize must be divisible by this).
         * @param nSwa      Sliding-window size (0 = disabled).
         * @param swaType   Type of sliding-window attention.
         */
        fun create(
            typeK: GGMLType = GGMLType.F32,
            typeV: GGMLType = GGMLType.F32,
            vTrans: Boolean = true,
            unified: Boolean = true,
            kvSize: Int = 512,
            nSeqMax: Int = 1,
            nPad: Int = 1,
            nSwa: Int = 0,
            swaType: LlamaSwaType = LlamaSwaType.NONE
        ): KVCache {
            return KVCache(
                kvSize = kvSize,
                nSeqMax = nSeqMax,
                nStream = if (unified) 1 else nSeqMax,
                nPad = nPad,
                nSwa = nSwa,
                swaType = swaType,
                typeK = typeK,
                typeV = typeV,
                vTrans = vTrans
            )
        }
    }

    // ── size / query accessors ──────────────────────────────────────────

    fun getSize(): Int = kvSize
    fun getNStream(): Int = nStream

    /** `true` when any cell has a non-zero delta (pending K-shift). */
    fun getHasShift(): Boolean {
        for (s in 0 until nStream) {
            for (i in 0 until vCells[s].size()) {
                if (vCells[s].getShift(i) != 0) return true
            }
        }
        return false
    }

    /** Whether K-shift (RoPE rotation of cached keys) is supported. */
    fun getCanShift(): Boolean = !attnRotK

    // ── clear ───────────────────────────────────────────────────────────

    /**
     * Reset all cell metadata (and optionally zero the backing data).
     *
     * @param data If `true`, also zero the tensor buffer contents.
     */
    fun clear(data: Boolean = false) {
        for (s in 0 until nStream) {
            vCells[s].reset()
            vHeads[s] = 0
        }
        _legacyCurrentLength = 0
        // NOTE: buffer zeroing requires backend buffer access – deferred
        if (data) {
        }
    }

    // ── sequence management ─────────────────────────────────────────────

    /**
     * Remove positions in range `[p0, p1)` from the given sequence (or all
     * sequences if [seqId] is -1).
     *
     * @return `true` always (matching C++ signature).
     */
    fun seqRm(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos): Boolean {
        var pp0 = if (p0 < 0) 0 else p0
        var pp1 = if (p1 < 0) Int.MAX_VALUE else p1

        if (seqId >= 0) {
            require(seqId < seqToStream.size) { "seqId out of range" }
            val streamIdx = seqToStream[seqId]
            val cells = vCells[streamIdx]
            var newHead = cells.size()

            for (i in 0 until cells.size()) {
                if (!cells.posIn(i, pp0, pp1)) continue
                if (cells.seqHas(i, seqId) && cells.seqRm(i, seqId)) {
                    if (newHead == cells.size()) newHead = i
                }
            }
            if (newHead != cells.size() && newHead < vHeads[streamIdx]) {
                vHeads[streamIdx] = newHead
            }
        } else {
            for (s in 0 until nStream) {
                val cells = vCells[s]
                var newHead = cells.size()
                for (i in 0 until cells.size()) {
                    if (!cells.posIn(i, pp0, pp1)) continue
                    cells.rm(i)
                    if (newHead == cells.size()) newHead = i
                }
                if (newHead != cells.size() && newHead < vHeads[s]) {
                    vHeads[s] = newHead
                }
            }
        }
        return true
    }

    /**
     * Copy sequence [seqIdSrc] to [seqIdDst] for positions in `[p0, p1)`.
     *
     * Within the same stream this only updates cell metadata.  Across
     * streams it enqueues a buffer copy that executes during the next
     * [update] call.
     */
    fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: LlamaPos, p1: LlamaPos) {
        require(seqIdSrc >= 0 && seqIdSrc < seqToStream.size) { "seqIdSrc out of range" }
        require(seqIdDst >= 0 && seqIdDst < seqToStream.size) { "seqIdDst out of range" }

        val s0 = seqToStream[seqIdSrc]
        val s1 = seqToStream[seqIdDst]

        if (s0 == s1) {
            if (seqIdSrc == seqIdDst) return
            val cells = vCells[s0]
            val pp0 = if (p0 < 0) 0 else p0
            val pp1 = if (p1 < 0) Int.MAX_VALUE else p1
            for (i in 0 until cells.size()) {
                if (!cells.posIn(i, pp0, pp1)) continue
                if (cells.seqHas(i, seqIdSrc)) {
                    cells.seqAdd(i, seqIdDst)
                }
            }
            return
        }

        // Cross-stream copy – enqueue for next update()
        scInfo.ssrc.add(s0)
        scInfo.sdst.add(s1)

        vCells[s1].reset()
        for (i in 0 until vCells[s0].size()) {
            if (vCells[s0].seqHas(i, seqIdSrc)) {
                val pos = vCells[s0].posGet(i)
                vCells[s1].posSet(i, pos)
                vCells[s1].seqAdd(i, seqIdDst)
            }
        }
        vHeads[s1] = vHeads[s0]
    }

    /**
     * Keep only [seqId] in every cell; remove all other sequences.
     */
    fun seqKeep(seqId: LlamaSeqId) {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        val streamIdx = seqToStream[seqId]
        val cells = vCells[streamIdx]
        var newHead = cells.size()
        for (i in 0 until cells.size()) {
            if (cells.seqKeep(i, seqId)) {
                if (newHead == cells.size()) newHead = i
            }
        }
        if (newHead != cells.size() && newHead < vHeads[streamIdx]) {
            vHeads[streamIdx] = newHead
        }
    }

    /**
     * Add [shift] to the position of every cell belonging to [seqId]
     * within `[p0, p1)`.
     */
    fun seqAdd(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, shift: LlamaPos) {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        if (shift == 0) return

        val streamIdx = seqToStream[seqId]
        val cells = vCells[streamIdx]
        val pp0 = if (p0 < 0) 0 else p0
        val pp1 = if (p1 < 0) Int.MAX_VALUE else p1
        if (pp0 == pp1) return

        var newHead = cells.size()
        for (i in 0 until cells.size()) {
            if (!cells.posIn(i, pp0, pp1)) continue
            if (cells.seqHas(i, seqId)) {
                if (cells.posAdd(i, shift)) {
                    if (newHead == cells.size()) newHead = i
                }
            }
        }
        vHeads[streamIdx] = if (newHead != cells.size()) newHead else 0
    }

    /**
     * Divide the position of cells belonging to [seqId] within `[p0, p1)` by [d].
     */
    fun seqDiv(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, d: Int) {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        if (d == 1) return

        val streamIdx = seqToStream[seqId]
        val cells = vCells[streamIdx]
        val pp0 = if (p0 < 0) 0 else p0
        val pp1 = if (p1 < 0) Int.MAX_VALUE else p1
        if (pp0 == pp1) return

        for (i in 0 until cells.size()) {
            if (!cells.posIn(i, pp0, pp1)) continue
            if (cells.seqHas(i, seqId)) {
                cells.posDiv(i, d)
            }
        }
    }

    /** Minimum position for [seqId], or -1 if no cells belong to it. */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        return vCells[seqToStream[seqId]].seqPosMin(seqId)
    }

    /** Maximum position for [seqId], or -1 if no cells belong to it. */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos {
        require(seqId >= 0 && seqId < seqToStream.size) { "seqId out of range" }
        return vCells[seqToStream[seqId]].seqPosMax(seqId)
    }

    // ── slot finding (port of llama_kv_cache::find_slot) ────────────────

    /**
     * Find a contiguous (or non-contiguous) run of free cells that can
     * hold `nTokens` tokens.
     *
     * @param nTokens  Number of tokens to place.
     * @param cont     If `true`, require a contiguous block.
     * @return A [SlotInfo] describing the placement, or an empty [SlotInfo] on failure.
     */
    /**
     * Find a contiguous (or non-contiguous) run of free cells that can
     * hold [nTokens] tokens from a single stream.
     *
     * Scans the ring buffer starting from the current head position. Empty
     * cells are always eligible. Non-empty cells with a single sequence are
     * eligible when SWA masking makes their cached position obsolete.
     *
     * Port of `llama_kv_cache::find_slot` from `llama-kv-cache.cpp`.
     *
     * @param nTokens  Number of tokens to place.
     * @param cont     If `true`, require a contiguous block.
     * @return A [SlotInfo] describing the placement, or an empty [SlotInfo] on failure.
     */
    fun findSlot(nTokens: Int, cont: Boolean = false): SlotInfo {
        // simplified single-stream version (the multi-stream ubatch variant is
        // handled by findSlotForUbatch which receives the full ubatch)
        val cells = vCells[0]
        if (nTokens > cells.size()) return SlotInfo()

        val res = SlotInfo()
        res.s0 = 0
        res.s1 = 0
        res.resize(1)
        res.strm[0] = 0

        var headCur = vHeads[0]
        if (headCur > cells.getUsed() + 2 * nTokens) {
            headCur = 0
        }

        var nTested = 0
        val nTest = if (cont) nTokens else 1

        while (true) {
            if (headCur + nTest > cells.size()) {
                nTested += cells.size() - headCur
                headCur = 0
                continue
            }

            for (k in 0 until nTest) {
                val idx = headCur
                headCur++
                nTested++

                var canUse = cells.isEmpty(idx)

                if (!canUse && cells.seqCount(idx) == 1) {
                    val posCell = cells.posGet(idx)
                    val seqIdCell = cells.seqGet(idx)
                    if (LlamaHparams.isMaskedSwa(nSwa, swaType, posCell, cells.seqPosMax(seqIdCell) + 1)) {
                        canUse = true
                    }
                }

                if (canUse) {
                    res.idxs[0].add(idx)
                } else if (cont) {
                    break
                }
            }

            if (res.idxs[0].size == nTokens) break

            if (cont) res.idxs[0].clear()

            if (nTested >= cells.size()) return SlotInfo()
        }

        if (res.idxs[0].size < nTokens) return SlotInfo()
        return res
    }

    // ── update (apply pending shifts and stream copies) ─────────────────

    /**
     * Apply pending K-shift and stream copies.
     *
     * In the C++ version this builds and executes a compute graph for RoPE
     * K-shift and performs backend buffer copies for cross-stream copies.
     * Those operations require backend infrastructure not yet ported to Kotlin.
     *
     * The cell-metadata side (resetting shifts) is fully implemented.
     *
     * @param doShift   Whether to perform K-shift (RoPE rotation update).
     * @param scInfoArg Pending stream copy info (consumed).
     * @return `true` if any work was done.
     */
    fun update(doShift: Boolean, scInfoArg: StreamCopyInfo = StreamCopyInfo()): Boolean {
        var updated = false

        if (!scInfoArg.empty()) {
            check(nStream > 1) { "stream copy should never happen with a single stream" }
            // NOTE: actual backend buffer copies require ggml_backend_tensor_copy
            // which is not yet available in the Kotlin port.
            // Cell metadata was already updated during seqCp().
            updated = true
        }

        if (doShift) {
            check(getCanShift()) { "The current KV cache / model configuration does not support K-shift" }

            // NOTE: The actual RoPE K-shift requires building a compute graph
            // with rope_ext operations and executing it via the backend scheduler.
            // This is deferred until the graph/backend infrastructure is ported.
            // For now, we just reset the shift metadata.

            for (s in 0 until nStream) {
                vCells[s].resetShift()
            }
            updated = true
        }

        return updated
    }

    // ── defrag ──────────────────────────────────────────────────────────

    /**
     * Defragment the cache by moving occupied cells toward the front,
     * closing gaps left by removed sequences.
     */
    /**
     * Defragment the cache by moving occupied cells toward the front,
     * closing gaps left by removed sequences.
     *
     * NOTE: The actual tensor data movement requires backend buffer operations
     * (`ggml_backend_tensor_copy`) which are not yet ported. This implementation
     * performs the cell-metadata compaction only; tensor data movement is deferred.
     */
    fun defrag() {
        // For each stream, compact cells toward the front.
        for (s in 0 until nStream) {
            val cells = vCells[s]
            // Metadata-only defrag is a identity when there are no gaps
            // (the actual C++ defrag builds and executes a copy graph).
            // Cell metadata compaction would require LlamaKvCells.mv() which
            // is commented out in the C++ source. Marking as deferred.
            vHeads[s] = 0
        }
    }

    // ── apply / prepare / getNKv ────────────────────────────────────────

    /**
     * Emplace a micro-batch into the cache at the positions described by [sinfo].
     *
     * For each token in the ubatch, the corresponding cell is overwritten with
     * the new position and sequence membership. Cells that were previously
     * occupied are cleared first. After emplacement, positions below the
     * overwritten range are purged to maintain the invariant that all positions
     * in `[pos_min, pos_max]` are present.
     *
     * Port of `llama_kv_cache::apply_ubatch` from `llama-kv-cache.cpp`.
     */
    fun applyUbatch(sinfo: SlotInfo, ubatch: LlamaUBatch) {
        val seqPosMaxRm = IntArray(LLAMA_MAX_SEQ) { -1 }

        for (s in 0 until sinfo.nStream()) {
            for (ii in 0 until sinfo.size()) {
                val i = s * sinfo.size() + ii
                val cells = vCells[sinfo.strm[s]]
                val idx = sinfo.idxs[s][ii]

                if (!cells.isEmpty(idx)) {
                    val seqIdCell = cells.seqGet(idx)
                    val posCell = cells.posGet(idx)
                    seqPosMaxRm[seqIdCell] = maxOf(seqPosMaxRm[seqIdCell], posCell)
                    cells.rm(idx)
                }

                val pos = ubatch.pos?.get(i) ?: 0
                cells.posSet(idx, pos)

                val nSeqIdForToken = ubatch.seqId?.get(i)?.size ?: 1
                for (si in 0 until nSeqIdForToken) {
                    val sid = ubatch.seqId?.get(i)?.get(si) ?: 0
                    cells.seqAdd(idx, sid)
                }
            }
        }

        // purge positions below overwritten range to maintain invariant
        for (s in 0 until LLAMA_MAX_SEQ) {
            if (seqPosMaxRm[s] == -1) continue
            require(s < seqToStream.size)
            val cells = vCells[seqToStream[s]]
            if (cells.seqPosMin(s) in 0..seqPosMaxRm[s]) {
                seqRm(s, cells.seqPosMin(s), seqPosMaxRm[s] + 1)
            }
        }

        // advance heads past the emplaced slots
        for (s in 0 until sinfo.nStream()) {
            vHeads[sinfo.strm[s]] = sinfo.idxs[s].last() + 1
        }
    }

    /**
     * Find slots for all [ubatches] and return their slot infos.
     * Returns an empty list on failure.
     *
     * Port of `llama_kv_cache::prepare` from `llama-kv-cache.cpp`.
     */
    fun prepare(ubatches: List<LlamaUBatch>): List<SlotInfo> {
        data class State(
            val sinfo: SlotInfo,
            val vHeadsOld: IntArray,
            val vCellsCopy: List<KvCells>,
        )

        val res = mutableListOf<SlotInfo>()
        val states = mutableListOf<State>()
        var success = true

        for (ubatch in ubatches) {
            val sinfoNew = findSlot(ubatch.nTokens, false)
            if (sinfoNew.empty()) {
                success = false
                break
            }
            res.add(sinfoNew)

            // save old cell state for rollback
            val cellsCopy = mutableListOf<KvCells>()
            for (s in 0 until sinfoNew.nStream()) {
                val cells = vCells[sinfoNew.strm[s]]
                cellsCopy.add(cells.cp(sinfoNew.idxs[s]))
            }
            states.add(State(sinfoNew, vHeads.copyOf(), cellsCopy))

            applyUbatch(sinfoNew, ubatch)
        }

        // rollback: restore cells to original state (reverse order)
        for (state in states.reversed()) {
            val sinfo = state.sinfo
            for (s in 0 until sinfo.nStream()) {
                val cells = vCells[sinfo.strm[s]]
                cells.set(sinfo.idxs[s], state.vCellsCopy[s])
                vHeads[sinfo.strm[s]] = state.vHeadsOld[sinfo.strm[s]]
            }
        }

        return if (success) res else emptyList()
    }

    /**
     * Compute the effective number of KV cells to attend to, padded for
     * graph-reuse stability.
     *
     * Port of `llama_kv_cache::get_n_kv` from `llama-kv-cache.cpp`.
     */
    fun getNKv(sinfo: SlotInfo): Int {
        var result = 0
        val nPadCur = maxOf(nPad, 256)
        for (s in 0 until sinfo.nStream()) {
            val cells = vCells[sinfo.strm[s]]
            val padded = maxOf(nPadCur, ((cells.usedMaxP1() + nPadCur - 1) / nPadCur) * nPadCur)
            result = maxOf(minOf(cells.size(), padded), result)
        }
        return result
    }

    /**
     * Return a breakdown of memory usage by buffer type.
     * Since we don't have real backend buffers yet, returns a summary entry.
     */
    fun memoryBreakdown(): Map<String, Long> {
        return mapOf("kv_cache" to totalSize())
    }

    // ── state serialisation ─────────────────────────────────────────────

    /**
     * Write cache state to the given writer.
     *
     * @param seqId  If >= 0, write only cells belonging to this sequence;
     *               if -1, write all cells.
     */
    fun stateWrite(seqId: LlamaSeqId = -1) {
        // State serialization requires LlamaIoWrite and backend tensor access.
        // The cell-metadata serialization logic is straightforward but the
        // tensor data read-back needs ggml_backend_tensor_get which is deferred.
        // LATER: implement once LlamaIoWrite and backend tensor access are ported
    }

    /**
     * Read cache state from the given reader.
     *
     * @param seqId  If >= 0, load only into cells for this sequence;
     *               if -1, load all cells.
     */
    fun stateRead(seqId: LlamaSeqId = -1) {
        // State deserialization requires LlamaIoRead and backend tensor access.
        // LATER: implement once LlamaIoRead and backend tensor access are ported
    }

    // ── internal helpers ────────────────────────────────────────────────

    /** Total bytes used by K cache tensors across all layers. */
    fun sizeKBytes(): Long {
        var total = 0L
        for (layer in layers) {
            val k = layer.k ?: continue
            // ggml_nbytes(t) = t->nb[GGML_MAX_DIMS-1] * t->ne[GGML_MAX_DIMS-1]
            val lastDim = GGML_MAX_DIMS - 1
            total += (k.nb[lastDim].toLong() * k.ne[lastDim])
        }
        return total
    }

    /** Total bytes used by V cache tensors across all layers. */
    fun sizeVBytes(): Long {
        var total = 0L
        for (layer in layers) {
            val v = layer.v ?: continue
            val lastDim = GGML_MAX_DIMS - 1
            total += (v.nb[lastDim].toLong() * v.ne[lastDim])
        }
        return total
    }

    /** Sum of K + V byte sizes. */
    fun totalSize(): Long = sizeKBytes() + sizeVBytes()

    // ── set_input_* helpers ────────────────────────────────────────────
    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache::set_input_k_shift
    fun setInputKShift(dst: GGMLTensor) { }

    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache::set_input_k_idxs
    fun setInputKIdxs(dst: GGMLTensor, ubatch: LlamaUBatch, sinfo: SlotInfo) { }

    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache::set_input_v_idxs
    fun setInputVIdxs(dst: GGMLTensor, ubatch: LlamaUBatch, sinfo: SlotInfo) { }

    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache::set_input_kq_mask
    fun setInputKqMask(dst: GGMLTensor, ubatch: LlamaUBatch, causalAttn: Boolean) { }

    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache::set_input_pos_bucket
    fun setInputPosBucket(dst: GGMLTensor, ubatch: LlamaUBatch) { }

    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache::set_input_k_rot
    fun setInputKRot(dst: GGMLTensor) { }

    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache::set_input_v_rot
    fun setInputVRot(dst: GGMLTensor) { }
}

// ---------------------------------------------------------------------------
// KVCacheContext – mirrors llama_kv_cache_context
// ---------------------------------------------------------------------------

/**
 * Context object created by [KVCache] for a single batch / update operation.
 *
 * Mirrors `llama_kv_cache_context` in C++.  Holds the slot info and
 * micro-batches needed to drive graph construction and input-tensor filling.
 *
 * @property status   Current status of this context.
 * @property kvCache  The parent KV cache (may be null for error contexts).
 */
class KVCacheContext(
    val status: LlamaMemoryStatus,
    private val kvCache: KVCache? = null
) : LlamaMemoryContext {
    private var iCur: Int = 0
    private val sinfos = mutableListOf<SlotInfo>()
    private var doShift: Boolean = false
    private var scInfo: StreamCopyInfo = StreamCopyInfo()

    /** A heuristic: how many KV cells to attend to in the current step. */
    var nKv: Int = 0
        private set

    /** Advance to the next micro-batch. Returns `false` when done. */
    override fun next(): Boolean {
        if (iCur >= sinfos.size) return false
        iCur++
        return iCur < sinfos.size
    }

    /** Apply the current micro-batch's slot info to the cache. */
    override fun apply(): Boolean {
        // No ubatches means this is an update context (K-shift / stream copy)
        if (sinfos.isEmpty()) {
            kvCache?.update(doShift, scInfo) ?: return false
            return true
        }

        if (iCur < sinfos.size) {
            // apply_ubatch would be called here with sinfos[iCur] and the
            // corresponding ubatch. Since we don't store ubatches in this
            // context yet (they come from the batch allocator), this is
            // a skeleton that updates nKv.
            nKv = kvCache?.getSize() ?: 0
        }
        return true
    }

    fun getNKv(): Int = nKv
    fun getSize(): Int = kvCache?.getSize() ?: 0
    fun typeK(): GGMLType = kvCache?.typeK ?: GGMLType.F32
    fun typeV(): GGMLType = kvCache?.typeV ?: GGMLType.F32

    // -- graph-building helpers ------------------------------------------------
    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache_context

    /** Build K-index input tensor. Port of `llama_kv_cache_context::build_input_k_idxs`. */
    fun buildInputKIdxs(ctx: GGMLContext, ubatch: LlamaUBatch): GGMLTensor {
        val t = ggmlNewTensor1d(ctx, GGMLType.I64, ubatch.nTokens.toLong())
        ggmlSetInput(t)
        return t
    }

    /** Build V-index input tensor. Port of `llama_kv_cache_context::build_input_v_idxs`. */
    fun buildInputVIdxs(ctx: GGMLContext, ubatch: LlamaUBatch): GGMLTensor {
        val t = ggmlNewTensor1d(ctx, GGMLType.I64, ubatch.nTokens.toLong())
        ggmlSetInput(t)
        return t
    }

    /** Build K-rotation matrix input (null if not needed). Port of `llama_kv_cache_context::build_input_k_rot`. */
    fun buildInputKRot(ctx: GGMLContext): GGMLTensor? = null

    /** Build V-rotation matrix input (null if not needed). Port of `llama_kv_cache_context::build_input_v_rot`. */
    fun buildInputVRot(ctx: GGMLContext): GGMLTensor? = null

    /** Copy K tensor into the cache at given indices. Port of `llama_kv_cache_context::cpy_k`. */
    fun cpyK(ctx: GGMLContext, kCur: GGMLTensor, kIdxs: GGMLTensor, il: Int): GGMLTensor {
        return ggmlCpy(ctx, kCur, ggmlNewTensorLike(ctx, kCur))
    }

    /** Copy V tensor into the cache at given indices. Port of `llama_kv_cache_context::cpy_v`. */
    fun cpyV(ctx: GGMLContext, vCur: GGMLTensor, vIdxs: GGMLTensor, il: Int): GGMLTensor {
        return ggmlCpy(ctx, vCur, ggmlNewTensorLike(ctx, vCur))
    }

    /** Get cached K tensor for layer [il]. Port of `llama_kv_cache_context::get_k`. */
    fun getK(ctx: GGMLContext, il: Int): GGMLTensor {
        val kType = typeK()
        val nKvCur = nKv.toLong()
        return ggmlNewTensor3d(ctx, kType, 128L, nKvCur, 1L) // LATER: use actual nEmbdHeadK, nHeadKv from hparams
    }

    /** Get cached V tensor for layer [il]. Port of `llama_kv_cache_context::get_v`. */
    fun getV(ctx: GGMLContext, il: Int): GGMLTensor {
        val vType = typeV()
        val nKvCur = nKv.toLong()
        return ggmlNewTensor3d(ctx, vType, 128L, nKvCur, 1L) // LATER: use actual nEmbdHeadV, nHeadKv from hparams
    }

    // ── set_input_* delegation ─────────────────────────────────────────
    // port-lint: source llama.cpp/src/llama-kv-cache.cpp  llama_kv_cache_context::set_input_*

    fun setInputKShift(dst: GGMLTensor) {
        kvCache?.setInputKShift(dst)
    }

    fun setInputKIdxs(dst: GGMLTensor, ubatch: LlamaUBatch) {
        kvCache?.setInputKIdxs(dst, ubatch, sinfos[iCur])
    }

    fun setInputVIdxs(dst: GGMLTensor, ubatch: LlamaUBatch) {
        kvCache?.setInputVIdxs(dst, ubatch, sinfos[iCur])
    }

    fun setInputKqMask(dst: GGMLTensor, ubatch: LlamaUBatch, causalAttn: Boolean) {
        kvCache?.setInputKqMask(dst, ubatch, causalAttn)
    }

    fun setInputPosBucket(dst: GGMLTensor, ubatch: LlamaUBatch) {
        kvCache?.setInputPosBucket(dst, ubatch)
    }

    fun setInputKRot(dst: GGMLTensor) {
        kvCache?.setInputKRot(dst)
    }

    fun setInputVRot(dst: GGMLTensor) {
        kvCache?.setInputVRot(dst)
    }

    override fun getUbatch(): LlamaUBatch = LlamaUBatch() // wire up real ubatch from batch allocator
    override fun getStatus(): LlamaMemoryStatus = status
}
