// port-lint: source llama.cpp/src/llama-memory-hybrid.h llama.cpp/src/llama-memory-hybrid.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// LlamaMemoryHybrid — hybrid memory combining KV cache + recurrent state
// Ported from: llama-memory-hybrid.h / llama-memory-hybrid.cpp
// =============================================================================

/**
 * State-sequence flags used when writing/reading state.
 *
 * Port of `LLAMA_STATE_SEQ_FLAGS_*` constants.
 */
object LlamaStateSeqFlags {
    /** When set, only write/read the partial (recurrent) component. */
    const val PARTIAL_ONLY = 1
}

/**
 * Hybrid memory implementation that combines a [KVCache] for attention layers
 * with a [LlamaMemoryRecurrent] for recurrent (SSM/RWKV) layers.
 *
 * Models whose architecture mixes attention and recurrent layers (e.g.
 * Jamba, Mamba-in-LLaMA) use this to keep the right kind of state per layer.
 *
 * Port of `class llama_memory_hybrid` from `llama-memory-hybrid.h`.
 *
 * @param hparams      Model hyper-parameters (used to classify layers).
 * @param memAttn      The attention KV cache instance.
 * @param memRecr      The recurrent memory instance.
 */
class LlamaMemoryHybrid(
    private val hparams: LlamaHparams,
    val memAttn: KVCache,
    val memRecr: LlamaMemoryRecurrent,
) : LlamaMemory {

    // -- LlamaMemory interface ------------------------------------------------

    override fun initBatch(balloc: LlamaBatchAllocr, nUbatch: Int, embdAll: Boolean): LlamaMemoryContext {
        // Batch splitting follows the recurrent pattern (split_seq when embdAll).
        // The actual split logic depends on LlamaBatchAllocr which is not yet ported;
        // return a hybrid context that delegates to both sub-memories.
        // LATER: full port requires LlamaBatchAllocr integration
        return LlamaMemoryHybridContext.error(LlamaMemoryStatus.FAILED_PREPARE)
    }

    override fun initFull(): LlamaMemoryContext {
        return LlamaMemoryHybridContext.full(this)
    }

    override fun initUpdate(lctx: LlamaContext, optimize: Boolean): LlamaMemoryContext {
        // LATER: requires LlamaContext integration for defrag/shift
        return LlamaMemoryHybridContext.error(LlamaMemoryStatus.NO_UPDATE)
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
        // If it does fail, the cache will not have been mutated.
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

    // -- state serialisation --------------------------------------------------

    override fun stateWrite(io: LlamaIoWrite, seqId: LlamaSeqId, flags: Int) {
        if ((flags and LlamaStateSeqFlags.PARTIAL_ONLY) == 0) {
            memAttn.stateWrite(seqId)
        }
        memRecr.stateWrite(io, seqId)
    }

    override fun stateRead(io: LlamaIoRead, seqId: LlamaSeqId, flags: Int) {
        if ((flags and LlamaStateSeqFlags.PARTIAL_ONLY) == 0) {
            memAttn.stateRead(seqId)
        }
        memRecr.stateRead(io, seqId)
    }

    override fun memoryBreakdown(): Map<String, Long> {
        // KVCache and LlamaMemoryRecurrent do not expose a breakdown map yet.
        // Return a combined summary of byte totals.
        val mb = mutableMapOf<String, Long>()
        mb["kv_cache"] = memAttn.totalSize()
        // Recurrent sizes are behind deferred markers; report zero until ported.
        mb["recurrent"] = 0L
        return mb
    }
}

// =============================================================================
// LlamaMemoryHybridContext — per-batch / full / update context
// Ported from: class llama_memory_hybrid_context
// =============================================================================

/**
 * Context object for [LlamaMemoryHybrid] that delegates to the attention and
 * recurrent sub-contexts and iterates through micro-batches.
 *
 * Port of `class llama_memory_hybrid_context` from `llama-memory-hybrid.h`.
 */
class LlamaMemoryHybridContext private constructor(
    private val status: LlamaMemoryStatus,
    private val ctxAttn: KVCacheContext?,
    private val ctxRecr: LlamaMemoryRecurrentContext?,
    private val ubatches: List<LlamaUBatch>,
) : LlamaMemoryContext {

    private var iNext: Int = 0

    companion object {
        /** Error / no-update context. */
        fun error(status: LlamaMemoryStatus) =
            LlamaMemoryHybridContext(status, ctxAttn = null, ctxRecr = null, ubatches = emptyList())

        /** Full-cache context for worst-case allocation sizing. */
        fun full(mem: LlamaMemoryHybrid): LlamaMemoryHybridContext {
            val ctxAttn = KVCacheContext(LlamaMemoryStatus.SUCCESS, null)
            val ctxRecr = LlamaMemoryRecurrentContext.full(mem.memRecr)
            val combined = llamaMemoryStatusCombine(
                ctxAttn.status, ctxRecr.getStatus()
            )
            return LlamaMemoryHybridContext(combined, ctxAttn, ctxRecr, ubatches = emptyList())
        }

        /** Batch-processing context with pre-allocated slot infos and ubatches. */
        fun batch(
            mem: LlamaMemoryHybrid,
            ubatches: List<LlamaUBatch>,
        ): LlamaMemoryHybridContext {
            val ctxAttn = KVCacheContext(LlamaMemoryStatus.SUCCESS, null)
            val ctxRecr = LlamaMemoryRecurrentContext.batch(mem.memRecr, ubatches)
            val combined = llamaMemoryStatusCombine(
                ctxAttn.status, ctxRecr.getStatus()
            )
            return LlamaMemoryHybridContext(combined, ctxAttn, ctxRecr, ubatches)
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

    /** The underlying attention sub-context. */
    fun getAttn(): KVCacheContext? = ctxAttn

    /** The underlying recurrent sub-context. */
    fun getRecr(): LlamaMemoryRecurrentContext? = ctxRecr
}
