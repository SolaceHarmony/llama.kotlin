// port-lint: source llama.cpp/src/llama-memory.h llama.cpp/src/llama-memory.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// LlamaMemory – abstract memory interfaces for LLM inference
// Ported from: llama-memory.h / llama-memory.cpp
// =============================================================================

// ---------------------------------------------------------------------------
// LlamaMemoryParams – configuration for memory initialization
// ---------------------------------------------------------------------------

/**
 * Parameters that describe how memory (e.g. a KV cache) should be created.
 *
 * Port of `llama_memory_params` from `llama-memory.h`.
 *
 * @property typeK  GGML type used for key tensors.
 * @property typeV  GGML type used for value tensors.
 * @property swaFull  When `true`, allocate a full-size sliding-window attention cache
 *                    instead of the compact variant.
 */
data class LlamaMemoryParams(
    val typeK: GGMLType = GGMLType.F16,
    val typeV: GGMLType = GGMLType.F16,
    val swaFull: Boolean = false,
)

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

/**
 * Combine two [LlamaMemoryStatus] values into a single result.
 *
 * Useful for hybrid memory implementations (e.g. iSWA) that manage two
 * independent sub-caches and need to merge their outcomes.
 *
 * Port of `llama_memory_status_combine()` from `llama-memory.cpp`.
 *
 * Rules:
 * - If either status is a failure, that failure is returned immediately.
 * - If either status is [LlamaMemoryStatus.SUCCESS], the combined result
 *   is [LlamaMemoryStatus.SUCCESS].
 * - Otherwise both are [LlamaMemoryStatus.NO_UPDATE] and the result is
 *   [LlamaMemoryStatus.NO_UPDATE].
 */
fun llamaMemoryStatusCombine(s0: LlamaMemoryStatus, s1: LlamaMemoryStatus): LlamaMemoryStatus {
    var hasUpdate = false

    when (s0) {
        LlamaMemoryStatus.SUCCESS -> hasUpdate = true
        LlamaMemoryStatus.NO_UPDATE -> { /* no change */ }
        LlamaMemoryStatus.FAILED_PREPARE,
        LlamaMemoryStatus.FAILED_COMPUTE -> return s0
    }

    when (s1) {
        LlamaMemoryStatus.SUCCESS -> hasUpdate = true
        LlamaMemoryStatus.NO_UPDATE -> { /* no change */ }
        LlamaMemoryStatus.FAILED_PREPARE,
        LlamaMemoryStatus.FAILED_COMPUTE -> return s1
    }

    return if (hasUpdate) LlamaMemoryStatus.SUCCESS else LlamaMemoryStatus.NO_UPDATE
}

/**
 * Check whether a [LlamaMemoryStatus] indicates a failure.
 *
 * Port of `llama_memory_status_is_fail()` from `llama-memory.cpp`.
 */
fun llamaMemoryStatusIsFail(status: LlamaMemoryStatus): Boolean = when (status) {
    LlamaMemoryStatus.SUCCESS,
    LlamaMemoryStatus.NO_UPDATE -> false
    LlamaMemoryStatus.FAILED_PREPARE,
    LlamaMemoryStatus.FAILED_COMPUTE -> true
}

/**
 * Extension shorthand for [llamaMemoryStatusIsFail].
 */
fun LlamaMemoryStatus.isFail(): Boolean = llamaMemoryStatusIsFail(this)

// ---------------------------------------------------------------------------
// LlamaMemoryContext – per-batch processing context
// ---------------------------------------------------------------------------

/**
 * Interface for managing memory state during a single batch-processing session.
 *
 * Implementations are created by [LlamaMemory.initBatch], [LlamaMemory.initFull],
 * or [LlamaMemory.initUpdate] and are consumed by the inference loop. The only
 * method that should mutate the backing memory is [apply].
 *
 * Port of `llama_memory_context_i` from `llama-memory.h`.
 */
interface LlamaMemoryContext {

    /**
     * Advance to the next micro-batch.
     *
     * @return `true` while there is a ubatch to process; `false` when done.
     */
    fun next(): Boolean

    /**
     * Apply the current ubatch's state changes to the backing memory object.
     *
     * @return `true` on success, `false` on failure.
     */
    fun apply(): Boolean

    /** The current micro-batch (valid after a successful [next] call). */
    fun getUbatch(): LlamaUBatch

    /** Status of this context — used for error handling and change detection. */
    fun getStatus(): LlamaMemoryStatus
}

// ---------------------------------------------------------------------------
// LlamaMemory – general concept of LLM memory
// ---------------------------------------------------------------------------

/** Callback to filter which layers participate in memory allocation. */
typealias LayerFilterCallback = (layerIndex: Int) -> Boolean

/** Callback to specify memory reuse across layers. Return negative to skip reuse. */
typealias LayerReuseCallback = (layerIndex: Int) -> Int

/**
 * General interface for LLM memory.
 *
 * A KV cache is the most common implementation, but recurrent-state caches
 * (SSM, RWKV) and hybrid variants also implement this interface.
 *
 * Port of `llama_memory_i` from `llama-memory.h`.
 */
interface LlamaMemory {

    /**
     * Split an input batch into micro-batches, verify they fit, and return
     * a context object for processing them.
     *
     * Check [LlamaMemoryContext.getStatus] on the returned context for errors.
     *
     * @param balloc   Batch allocator used to split the input batch into micro-batches.
     * @param nUbatch  Maximum micro-batch size.
     * @param embdAll  Whether to embed all tokens.
     * @return A [LlamaMemoryContext] ready for iteration.
     */
    fun initBatch(balloc: LlamaBatchAllocr, nUbatch: Int, embdAll: Boolean): LlamaMemoryContext

    /**
     * Create a context that simulates a full cache. Used for allocating
     * worst-case compute buffers.
     */
    fun initFull(): LlamaMemoryContext

    /**
     * Prepare for any pending memory updates (shifts, defrag, etc.).
     *
     * The returned context has [LlamaMemoryStatus.NO_UPDATE] when nothing
     * needs to change.
     *
     * @param lctx      The parent [LlamaContext] (needed for graph builds during updates).
     * @param optimize  Whether to also perform optional optimisations (e.g. defrag).
     */
    fun initUpdate(lctx: LlamaContext, optimize: Boolean): LlamaMemoryContext

    /** Whether this memory type supports K-shift (position shifting). */
    fun getCanShift(): Boolean

    /**
     * Clear all stored state.
     *
     * @param data  When `true`, also zero the underlying data buffers.
     */
    fun clear(data: Boolean)

    // -- sequence operations ------------------------------------------------

    /**
     * Remove entries for [seqId] in positions `[p0, p1)`.
     *
     * @return `true` if any entries were actually removed.
     */
    fun seqRm(seqId: LlamaSeqId, p0: Int, p1: Int): Boolean

    /** Copy entries from [seqIdSrc] to [seqIdDst] in positions `[p0, p1)`. */
    fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: Int, p1: Int)

    /** Remove all entries except those belonging to [seqId]. */
    fun seqKeep(seqId: LlamaSeqId)

    /** Shift positions for [seqId] in `[p0, p1)` by [shift]. */
    fun seqAdd(seqId: LlamaSeqId, p0: Int, p1: Int, shift: Int)

    /** Divide positions for [seqId] in `[p0, p1)` by [d]. */
    fun seqDiv(seqId: LlamaSeqId, p0: Int, p1: Int, d: Int)

    /** Minimum occupied position for [seqId], or `Int.MAX_VALUE` if empty. */
    fun seqPosMin(seqId: LlamaSeqId): Int

    /** Maximum occupied position for [seqId], or `0` if empty. */
    fun seqPosMax(seqId: LlamaSeqId): Int

    /**
     * Return a breakdown of memory usage by buffer type.
     *
     * The map keys are opaque buffer-type identifiers; the values are sizes
     * in bytes.
     */
    fun memoryBreakdown(): Map<String, Long>

    // -- state serialization ------------------------------------------------

    /** Serialize memory state for [seqId] (or all if -1) via [io]. */
    fun stateWrite(io: LlamaIoWrite, seqId: LlamaSeqId = -1, flags: Int = 0)

    /** Deserialize memory state for [seqId] (or all if -1) via [io]. */
    fun stateRead(io: LlamaIoRead, seqId: LlamaSeqId = -1, flags: Int = 0)
}
