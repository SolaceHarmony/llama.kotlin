// port-lint: source llama.cpp/src/llama-memory-hybrid-iswa.h llama.cpp/src/llama-memory-hybrid-iswa.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// LlamaMemoryHybridIswa — ISWA variant of hybrid memory
// Ported from: llama-memory-hybrid-iswa.h / llama-memory-hybrid-iswa.cpp
// =============================================================================

/**
 * ISWA (Interleaved Sliding-Window Attention) variant of hybrid memory.
 *
 * Combines a KV cache with full + sliding-window attention streams (ISWA)
 * and a recurrent memory for SSM/RWKV layers. Models that interleave SWA
 * attention layers with recurrent layers use this implementation.
 *
 * Port of `class llama_memory_hybrid_iswa` from `llama-memory-hybrid-iswa.h`.
 *
 * @param hparams      Model hyper-parameters.
 * @param memAttn      The ISWA KV cache (manages both base and SWA streams).
 * @param memRecr      The recurrent memory instance.
 */
class LlamaMemoryHybridIswa(
    private val hparams: LlamaHparams,
    val memAttn: KVCache,
    val memRecr: LlamaMemoryRecurrent,
) : LlamaMemory {

    // -- LlamaMemory interface ------------------------------------------------

    override fun initBatch(balloc: LlamaBatchAllocr, nUbatch: Int, embdAll: Boolean): LlamaMemoryContext {
        // Full port requires LlamaBatchAllocr integration and the ISWA
        // dual-stream prepare logic (base + SWA slot infos).
        // LATER: complete when batch splitting infrastructure is ready
        return LlamaMemoryHybridIswaContext.error(LlamaMemoryStatus.FAILED_PREPARE)
    }

    override fun initFull(): LlamaMemoryContext {
        return LlamaMemoryHybridIswaContext.full(this)
    }

    override fun initUpdate(lctx: LlamaContext, optimize: Boolean): LlamaMemoryContext {
        // LATER: requires LlamaContext integration for defrag/shift
        return LlamaMemoryHybridIswaContext.error(LlamaMemoryStatus.NO_UPDATE)
    }

    override fun getCanShift(): Boolean {
        // Shifting is trivially supported for recurrent; defer to the attention cache
        return memAttn.getCanShift()
    }

    override fun clear(data: Boolean) {
        memAttn.clear(data)
        memRecr.clear(data)
    }

    // -- sequence operations --------------------------------------------------

    override fun seqRm(seqId: LlamaSeqId, p0: Int, p1: Int): Boolean {
        // Try removing from the recurrent cache first since it may fail.
        if (!memRecr.seqRm(seqId, p0, p1)) {
            return false
        }
        return memAttn.seqRm(seqId, p0, p1)
    }

    override fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: Int, p1: Int) {
        memAttn.seqCp(seqIdSrc, seqIdDst, p0, p1)
        memRecr.seqCp(seqIdSrc, seqIdDst, p0, p1)
    }

    override fun seqKeep(seqId: LlamaSeqId) {
        memAttn.seqKeep(seqId)
        memRecr.seqKeep(seqId)
    }

    override fun seqAdd(seqId: LlamaSeqId, p0: Int, p1: Int, shift: Int) {
        memAttn.seqAdd(seqId, p0, p1, shift)
        memRecr.seqAdd(seqId, p0, p1, shift)
    }

    override fun seqDiv(seqId: LlamaSeqId, p0: Int, p1: Int, d: Int) {
        memAttn.seqDiv(seqId, p0, p1, d)
        memRecr.seqDiv(seqId, p0, p1, d)
    }

    override fun seqPosMin(seqId: LlamaSeqId): Int {
        // The min of the combined cache is the max of the two sub-caches' min values
        return maxOf(memAttn.seqPosMin(seqId), memRecr.seqPosMin(seqId))
    }

    override fun seqPosMax(seqId: LlamaSeqId): Int {
        // The max of the combined cache is the min of the two sub-caches' max values
        return minOf(memAttn.seqPosMax(seqId), memRecr.seqPosMax(seqId))
    }

    override fun memoryBreakdown(): Map<String, Long> {
        val mb = mutableMapOf<String, Long>()
        mb["kv_cache_iswa"] = memAttn.totalSize()
        mb["recurrent"] = 0L
        return mb
    }

    // -- state serialisation --------------------------------------------------

    override fun stateWrite(io: LlamaIoWrite, seqId: LlamaSeqId, flags: Int) {
        memAttn.stateWrite(seqId)
        memRecr.stateWrite(io, seqId)
    }

    override fun stateRead(io: LlamaIoRead, seqId: LlamaSeqId, flags: Int) {
        memAttn.stateRead(seqId)
        memRecr.stateRead(io, seqId)
    }
}

// =============================================================================
// LlamaMemoryHybridIswaContext — per-batch / full / update context
// Ported from: class llama_memory_hybrid_iswa_context
// =============================================================================

/**
 * Context object for [LlamaMemoryHybridIswa] that delegates to the ISWA
 * attention and recurrent sub-contexts.
 *
 * Port of `class llama_memory_hybrid_iswa_context` from
 * `llama-memory-hybrid-iswa.h`.
 */
class LlamaMemoryHybridIswaContext private constructor(
    private val status: LlamaMemoryStatus,
    private val ctxAttn: KVCacheContext?,
    private val ctxRecr: LlamaMemoryRecurrentContext?,
    private val ubatches: List<LlamaUBatch>,
) : LlamaMemoryContext {

    private var iNext: Int = 0

    companion object {
        /** Error / no-update context. */
        fun error(status: LlamaMemoryStatus) =
            LlamaMemoryHybridIswaContext(status, ctxAttn = null, ctxRecr = null, ubatches = emptyList())

        /** Full-cache context for worst-case allocation sizing. */
        fun full(mem: LlamaMemoryHybridIswa): LlamaMemoryHybridIswaContext {
            val ctxAttn = KVCacheContext(LlamaMemoryStatus.SUCCESS, null)
            val ctxRecr = LlamaMemoryRecurrentContext.full(mem.memRecr)
            val combined = llamaMemoryStatusCombine(
                ctxAttn.status, ctxRecr.getStatus()
            )
            return LlamaMemoryHybridIswaContext(combined, ctxAttn, ctxRecr, ubatches = emptyList())
        }

        /** Batch-processing context with pre-allocated slot infos and ubatches. */
        fun batch(
            mem: LlamaMemoryHybridIswa,
            ubatches: List<LlamaUBatch>,
        ): LlamaMemoryHybridIswaContext {
            val ctxAttn = KVCacheContext(LlamaMemoryStatus.SUCCESS, null)
            val ctxRecr = LlamaMemoryRecurrentContext.batch(mem.memRecr, ubatches)
            val combined = llamaMemoryStatusCombine(
                ctxAttn.status, ctxRecr.getStatus()
            )
            return LlamaMemoryHybridIswaContext(combined, ctxAttn, ctxRecr, ubatches)
        }
    }

    override fun next(): Boolean {
        ctxAttn?.next()
        ctxRecr?.next()
        iNext++
        return iNext < ubatches.size
    }

    override fun apply(): Boolean {
        require(!status.isFail()) { "Cannot apply a failed context" }
        var res = true
        if (ctxAttn != null) res = res and ctxAttn.apply()
        if (ctxRecr != null) res = res and ctxRecr.apply()
        return res
    }

    override fun getStatus(): LlamaMemoryStatus = status

    override fun getUbatch(): LlamaUBatch {
        require(status == LlamaMemoryStatus.SUCCESS) { "Context is not in SUCCESS state" }
        return ubatches[iNext]
    }

    /** The underlying ISWA attention sub-context. */
    fun getAttn(): KVCacheContext? = ctxAttn

    /** The underlying recurrent sub-context. */
    fun getRecr(): LlamaMemoryRecurrentContext? = ctxRecr
}
