// port-lint: source llama.cpp/src/llama-kv-cache-iswa.h llama.cpp/src/llama-kv-cache-iswa.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.min

// =============================================================================
// LlamaKvCacheIswa – Interleaved Sliding Window Attention KV cache
// Ported from: llama-kv-cache-iswa.h / llama-kv-cache-iswa.cpp
// =============================================================================

/**
 * A KV cache variant that wraps two [KVCache] instances:
 * - [kvBase] for non-SWA (dense / full-attention) layers
 * - [kvSwa]  for SWA (sliding-window attention) layers
 *
 * This allows different eviction and sizing strategies for each set of layers.
 *
 * Port of `llama_kv_cache_iswa` from `llama-kv-cache-iswa.h`.
 *
 * @property hparams   Model hyperparameters (needed for layer classification).
 * @property unified   Whether streams are unified (single stream for all sequences).
 * @property kvBase    KV cache for non-SWA layers.
 * @property kvSwa     KV cache for SWA layers.
 */
class LlamaKvCacheIswa(
    private val hparams: LlamaHparams,
    private val unified: Boolean,
    val kvBase: KVCache,
    val kvSwa: KVCache,
) : LlamaMemory {

    companion object {
        /**
         * Create an iSWA cache from model parameters, mirroring the C++ constructor.
         *
         * @param hparams   Model hyperparameters.
         * @param typeK     GGML type for key tensors.
         * @param typeV     GGML type for value tensors.
         * @param vTrans    Whether value tensors are stored transposed.
         * @param swaFull   When `true`, use full-size SWA cache instead of compact.
         * @param unified   Whether to use a single stream for all sequences.
         * @param kvSize    Number of cache cells for the base cache.
         * @param nSeqMax   Maximum number of concurrent sequences.
         * @param nUbatch   Maximum micro-batch size (used for SWA sizing).
         * @param nPad      Padding requirement.
         * @param filter    Optional layer filter callback.
         * @param reuse     Optional layer reuse callback.
         */
        fun create(
            hparams: LlamaHparams,
            typeK: GGMLType = GGMLType.F16,
            typeV: GGMLType = GGMLType.F16,
            vTrans: Boolean = true,
            swaFull: Boolean = false,
            unified: Boolean = true,
            kvSize: Int = 512,
            nSeqMax: Int = 1,
            nUbatch: Int = 512,
            nPad: Int = 1,
            filter: LayerFilterCallback? = null,
            reuse: LayerReuseCallback? = null,
        ): LlamaKvCacheIswa {
            val filterBase: LayerFilterCallback = { il ->
                (filter == null || filter(il)) && !hparams.isSwa(il)
            }
            val filterSwa: LayerFilterCallback = { il ->
                (filter == null || filter(il)) && hparams.isSwa(il)
            }

            val sizeBase = kvSize

            // note: the SWA cache is always padded to 256 for performance
            var sizeSwa = padTo256(
                min(sizeBase, hparams.nSwa * (if (unified) nSeqMax else 1) + nUbatch)
            )

            if (swaFull) {
                sizeSwa = sizeBase
            }

            val base = KVCache.create(
                typeK = typeK, typeV = typeV, vTrans = vTrans,
                unified = unified, kvSize = sizeBase, nSeqMax = nSeqMax,
                nPad = nPad, nSwa = 0, swaType = LlamaSwaType.NONE,
            )

            val swa = KVCache.create(
                typeK = typeK, typeV = typeV, vTrans = vTrans,
                unified = unified, kvSize = sizeSwa, nSeqMax = nSeqMax,
                nPad = nPad, nSwa = hparams.nSwa, swaType = hparams.swaType,
            )

            return LlamaKvCacheIswa(hparams, unified, base, swa)
        }

        /** Pad [n] up to the nearest multiple of 256. */
        private fun padTo256(n: Int): Int {
            val align = 256
            return ((n + align - 1) / align) * align
        }
    }

    // -- LlamaMemory implementation -------------------------------------------

    override fun initBatch(balloc: LlamaBatchAllocr, nUbatch: Int, embdAll: Boolean): LlamaMemoryContext {
        // For now, return a simple delegating context.
        // Full batch splitting logic (simple then equal) is complex and depends on
        // LlamaBatchAllocr which would be ported separately.
        return LlamaKvCacheIswaContext(LlamaMemoryStatus.FAILED_PREPARE)
    }

    override fun initFull(): LlamaMemoryContext {
        return LlamaKvCacheIswaContext(this)
    }

    override fun initUpdate(lctx: LlamaContext, optimize: Boolean): LlamaMemoryContext {
        return LlamaKvCacheIswaContext(this, optimize)
    }

    override fun getCanShift(): Boolean {
        return kvBase.getCanShift() &&
               kvSwa.getCanShift() &&
               kvBase.getSize() == kvSwa.getSize()
    }

    override fun clear(data: Boolean) {
        kvBase.clear(data)
        kvSwa.clear(data)
    }

    override fun seqRm(seqId: LlamaSeqId, p0: Int, p1: Int): Boolean {
        var res = true
        res = res and kvBase.seqRm(seqId, p0, p1)
        res = res and kvSwa.seqRm(seqId, p0, p1)
        return res
    }

    override fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: Int, p1: Int) {
        kvBase.seqCp(seqIdSrc, seqIdDst, p0, p1)
        kvSwa.seqCp(seqIdSrc, seqIdDst, p0, p1)
    }

    override fun seqKeep(seqId: LlamaSeqId) {
        kvBase.seqKeep(seqId)
        kvSwa.seqKeep(seqId)
    }

    override fun seqAdd(seqId: LlamaSeqId, p0: Int, p1: Int, shift: Int) {
        kvBase.seqAdd(seqId, p0, p1, shift)
        kvSwa.seqAdd(seqId, p0, p1, shift)
    }

    override fun seqDiv(seqId: LlamaSeqId, p0: Int, p1: Int, d: Int) {
        kvBase.seqDiv(seqId, p0, p1, d)
        kvSwa.seqDiv(seqId, p0, p1, d)
    }

    override fun seqPosMin(seqId: LlamaSeqId): Int {
        // the SWA cache is a subset — use it for min (it reflects the base too)
        return kvSwa.seqPosMin(seqId)
    }

    override fun seqPosMax(seqId: LlamaSeqId): Int {
        return kvSwa.seqPosMax(seqId)
    }

    override fun memoryBreakdown(): Map<String, Long> {
        val mb = kvBase.memoryBreakdown().toMutableMap()
        for ((k, v) in kvSwa.memoryBreakdown()) {
            mb[k] = (mb[k] ?: 0L) + v
        }
        return mb
    }

    override fun stateWrite(io: LlamaIoWrite, seqId: LlamaSeqId, flags: Int) {
        val partialOnly = (flags and LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0
        if (!partialOnly) {
            kvBase.stateWrite(seqId)
        }
        kvSwa.stateWrite(seqId)
    }

    override fun stateRead(io: LlamaIoRead, seqId: LlamaSeqId, flags: Int) {
        val partialOnly = (flags and LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != 0
        if (!partialOnly) {
            kvBase.stateRead(seqId)
        }
        kvSwa.stateRead(seqId)
    }

    // -- accessors ------------------------------------------------------------

    fun getBase(): KVCache = kvBase
    fun getSwa(): KVCache = kvSwa
}

/** Flags for partial-only state serialization. */
const val LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY = 1

// =============================================================================
// LlamaKvCacheIswaContext – memory context for iSWA cache
// Ported from: llama_kv_cache_iswa_context in llama-kv-cache-iswa.h/.cpp
// =============================================================================

/**
 * Memory context wrapping two [KVCacheContext] instances (base + SWA).
 *
 * Port of `llama_kv_cache_iswa_context` from `llama-kv-cache-iswa.cpp`.
 */
class LlamaKvCacheIswaContext : LlamaMemoryContext {

    private val status: LlamaMemoryStatus
    private var iNext: Int = 0
    private val ubatches: MutableList<LlamaUBatch> = mutableListOf()
    private var ctxBase: LlamaMemoryContext? = null
    private var ctxSwa: LlamaMemoryContext? = null

    /** Error constructor. */
    constructor(status: LlamaMemoryStatus) {
        this.status = status
    }

    /** Full-cache constructor. */
    constructor(kv: LlamaKvCacheIswa) {
        ctxBase = KVCacheContext(LlamaMemoryStatus.SUCCESS, kv.kvBase)
        ctxSwa = KVCacheContext(LlamaMemoryStatus.SUCCESS, kv.kvSwa)
        status = llamaMemoryStatusCombine(
            ctxBase!!.getStatus(), ctxSwa!!.getStatus()
        )
    }

    /** Update constructor. */
    constructor(kv: LlamaKvCacheIswa, optimize: Boolean) {
        ctxBase = KVCacheContext(LlamaMemoryStatus.SUCCESS, kv.kvBase)
        ctxSwa = KVCacheContext(LlamaMemoryStatus.SUCCESS, kv.kvSwa)
        status = llamaMemoryStatusCombine(
            ctxBase!!.getStatus(), ctxSwa!!.getStatus()
        )
    }

    /** Batch processing constructor. */
    constructor(
        kv: LlamaKvCacheIswa,
        sinfosBase: List<SlotInfo>,
        sinfosSwa: List<SlotInfo>,
        ubatches: List<LlamaUBatch>,
    ) {
        this.ubatches.addAll(ubatches)
        ctxBase = KVCacheContext(LlamaMemoryStatus.SUCCESS, kv.kvBase)
        ctxSwa = KVCacheContext(LlamaMemoryStatus.SUCCESS, kv.kvSwa)
        status = llamaMemoryStatusCombine(
            ctxBase!!.getStatus(), ctxSwa!!.getStatus()
        )
    }

    override fun next(): Boolean {
        check(status == LlamaMemoryStatus.SUCCESS)
        ctxBase?.next()
        ctxSwa?.next()
        iNext++
        return iNext < ubatches.size
    }

    override fun apply(): Boolean {
        check(!status.isFail())
        var res = true
        res = res and (ctxBase?.apply() ?: true)
        res = res and (ctxSwa?.apply() ?: true)
        return res
    }

    override fun getUbatch(): LlamaUBatch {
        check(status == LlamaMemoryStatus.SUCCESS)
        return ubatches[iNext]
    }

    override fun getStatus(): LlamaMemoryStatus = status

    fun getBase(): LlamaMemoryContext? = ctxBase
    fun getSwa(): LlamaMemoryContext? = ctxSwa
}
