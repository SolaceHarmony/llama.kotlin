// port-lint: source llama.cpp/src/llama-context.h llama.cpp/src/llama-context.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.*
import kotlin.time.TimeSource

// ---------------------------------------------------------------------------
// Enums ported from llama.h / llama-context.h
// ---------------------------------------------------------------------------

/** Pooling strategy for output embeddings. Maps to `llama_pooling_type`. */
enum class LlamaPoolingType {
    UNSPECIFIED,
    NONE,
    MEAN,
    CLS,
    LAST,
    RANK;
}

/** Attention mode selector. Maps to `llama_attention_type`. */
enum class LlamaAttentionType {
    UNSPECIFIED,
    CAUSAL,
    NON_CAUSAL;
}

/** Flash-attention mode selector. Maps to `llama_flash_attn_type`. */
enum class LlamaFlashAttnType {
    DISABLED,
    ENABLED,
    AUTO;
}

// ---------------------------------------------------------------------------
// Context parameters  (maps to llama_context_params)
// ---------------------------------------------------------------------------

/**
 * Parameters supplied when creating a [LlamaContext].
 *
 * Every field mirrors a member of `llama_context_params` from the C++ codebase.
 * The Kotlin default values match the C++ defaults (0/false means "use model default").
 */
data class LlamaContextParams(
    val nCtx: Int = 0,
    val nBatch: Int = 2048,
    val nUbatch: Int = 512,
    val nSeqMax: Int = 1,
    val nThreads: Int = 4,
    val nThreadsBatch: Int = 4,
    val ropeScalingType: LlamaRopeScalingType = LlamaRopeScalingType.UNSPECIFIED,
    val ropeFreqBase: Float = 0.0f,
    val ropeFreqScale: Float = 0.0f,
    val yarnExtFactor: Float = -1.0f,
    val yarnAttnFactor: Float = 1.0f,
    val yarnBetaFast: Float = 32.0f,
    val yarnBetaSlow: Float = 1.0f,
    val yarnOrigCtx: Int = 0,
    val embeddings: Boolean = false,
    val offloadKqv: Boolean = true,
    val flashAttnType: LlamaFlashAttnType = LlamaFlashAttnType.DISABLED,
    val noPerf: Boolean = false,
    val attentionType: LlamaAttentionType = LlamaAttentionType.UNSPECIFIED,
    val poolingType: LlamaPoolingType = LlamaPoolingType.UNSPECIFIED,
    val typeK: GGMLType = GGMLType.F16,
    val typeV: GGMLType = GGMLType.F16,
    val swaFull: Boolean = false,
    val opOffload: Boolean = true,
    val kvUnified: Boolean = true,
)

// ---------------------------------------------------------------------------
// Performance counters  (maps to llama_perf_context_data)
// ---------------------------------------------------------------------------

/** Performance counters returned by [LlamaContext.perfGetData]. */
data class LlamaPerfContextData(
    val tStartMs: Double = 0.0,
    val tLoadMs: Double = 0.0,
    val tPEvalMs: Double = 0.0,
    val tEvalMs: Double = 0.0,
    val nPEval: Int = 1,
    val nEval: Int = 1,
    val nReused: Int = 0,
)

// ---------------------------------------------------------------------------
// Memory breakdown  (maps to llama_memory_breakdown_data)
// ---------------------------------------------------------------------------

/** Per-buffer-type memory usage breakdown. */
data class LlamaMemoryBreakdownData(
    var model: Long = 0,
    var context: Long = 0,
    var compute: Long = 0,
) {
    fun total(): Long = model + context + compute
}

// ---------------------------------------------------------------------------
// GGML status codes  (mirrors ggml_status enum)
// ---------------------------------------------------------------------------

/** Status codes returned by graph compute operations. */
object GGMLStatus {
    const val SUCCESS = 0
    const val FAILED = 1
    const val ABORTED = 2
    const val ALLOC_FAILED = 3
}

// ---------------------------------------------------------------------------
// SamplingInfo — nested data class for backend sampling state
// ---------------------------------------------------------------------------

/**
 * Per-context sampling bookkeeping.
 *
 * Port of `llama_context::sampling_info`.
 */
class SamplingInfo {
    /** Active backend samplers keyed by sequence id. */
    val samplers: MutableMap<LlamaSeqId, Any> = mutableMapOf()

    var logits: FloatArray? = null
    var logitsSize: Int = 0
    var sampled: IntArray? = null
    var sampledSize: Int = 0
    var probs: FloatArray? = null
    var probsSize: Int = 0
    var candidates: IntArray? = null
    var candidatesSize: Int = 0

    val logitsCount: MutableList<Int> = mutableListOf()
    val probsCount: MutableList<Int> = mutableListOf()
    val candidatesCount: MutableList<Int> = mutableListOf()

    /** Full vocabulary token ids for fallback candidate list. */
    var tokenIdsFullVocab: IntArray = IntArray(0)
}

// ---------------------------------------------------------------------------
// LlamaContext  (maps to struct llama_context)
// ---------------------------------------------------------------------------

/**
 * The primary inference context — owns the KV cache, backend scheduler,
 * compute buffers, and output (logits / embeddings) storage.
 *
 * This class is the Kotlin port of `struct llama_context` from
 * `llama-context.h` / `llama-context.cpp`. It is the entry-point for
 * [encode] and [decode] operations.
 *
 * Construction mirrors `llama_context::llama_context(model, params)`:
 * it resolves the computed context parameters ([cparams]) from the raw
 * [LlamaContextParams], initialises backends, creates the output buffer,
 * and reserves worst-case graphs.
 */
class LlamaContext(
    /** Reference to the loaded model (weights + hyper-parameters). */
    val model: LlamaModel,
    params: LlamaContextParams = LlamaContextParams(),
) {
    // -- resolved / computed context parameters --
    val cparams: LlamaCParams = resolveCParams(model.config, params)

    // -- memory (KV cache or equivalent abstract memory) --
    /** Per-layer KV caches; initialised lazily via [initMemory]. */
    var kvCaches: Array<KVCache>? = null
        private set

    /** Abstract memory interface (may be KV cache, recurrent, or hybrid). */
    var memory: LlamaMemory? = null
        internal set

    // -- threadpools (opaque handles, mirrors C++ ggml_threadpool_t) --
    var threadpool: Any? = null
        internal set
    var threadpoolBatch: Any? = null
        internal set

    // -- abort callback --
    var abortCallback: (() -> Boolean)? = null
        internal set

    // -- LoRA adapters: adapter → scale --
    val loras: MutableMap<Any, Float> = mutableMapOf()

    // -- batch allocator --
    var balloc: LlamaBatchAllocr? = null
        internal set

    // -- optimization context (for fine-tuning) --
    var optCtx: Any? = null
        internal set

    // -- graph results (previous and reserved) --
    var gfResPrev: Any? = null
        internal set
    var gfResReserve: Any? = null
        internal set

    // -- decode output (2-d array: [nOutputs][nVocab]) --
    var logits: FloatArray? = null
        private set
    var logitsSize: Int = 0
        private set

    // -- embeddings output (2-d array: [nOutputs][nEmbd]) --
    var embd: FloatArray? = null
        private set
    var embdSize: Int = 0
        private set

    // -- sequence-level embeddings (populated when pooling != NONE) --
    val embdSeq: MutableMap<LlamaSeqId, FloatArray> = mutableMapOf()

    // -- output bookkeeping --
    var nOutputs: Int = 0
        private set

    /** Maps batch token positions → ids inside the logits / embd buffers. */
    var outputIds: IntArray = IntArray(0)
        private set

    val outputSwaps: MutableList<SwapInfo> = mutableListOf()

    // -- scheduler state --
    var schedNeedReserve: Boolean = true
        private set

    // -- backend sampling state --
    val sampling: SamplingInfo = SamplingInfo()

    // -- cross-attention state (for encoder-decoder models) --
    val cross: LlamaCross = LlamaCross()

    // -- graph reuse control --
    var graphReuseDisable: Boolean = false
        private set

    // -- perf counters (microseconds) --
    var tStartUs: Long = 0L
        private set
    var tLoadUs: Long = 0L
        private set
    var tPEvalUs: Long = 0L
        private set
    var tEvalUs: Long = 0L
        private set
    var tComputeStartUs: Long = 0L
        private set
    var nQueuedTokens: Long = 0L
        private set
    var nPEval: Int = 0
        private set
    var nEval: Int = 0
        private set
    var nReused: Int = 0
        private set

    var hasEvaluatedOnce: Boolean = false
        private set

    // -- graph --
    /** Graph allocator for compute operations. */
    val graphAllocator: GGMLGraphAllocator = GGMLGraphAllocator()

    /** Shared GGML context for tensor metadata. */
    val ggmlContext: GGMLContext = GGMLContext()

    init {
        tStartUs = currentTimeMicros()
        initMemory()
        initSamplingVocab()
    }

    // -----------------------------------------------------------------------
    // Accessors  (mirrors C++ getter methods)
    // -----------------------------------------------------------------------

    fun nCtx(): Int = cparams.nCtx
    fun nCtxSeq(): Int = cparams.nCtxSeq
    fun nBatch(): Int = cparams.nBatch
    fun nUbatch(): Int = cparams.nUbatch
    fun nSeqMax(): Int = cparams.nSeqMax
    fun nThreads(): Int = cparams.nThreads
    fun nThreadsBatch(): Int = cparams.nThreadsBatch
    fun poolingType(): LlamaPoolingType = cparams.poolingType

    // -----------------------------------------------------------------------
    // Logits / embeddings access
    // -----------------------------------------------------------------------

    /**
     * Return the full logits buffer. Triggers output reordering first.
     * Mirrors `llama_context::get_logits()`.
     */
    fun getLogits(): FloatArray? {
        outputReorder()
        return logits
    }

    /**
     * Return a view of the logits for the i-th output token.
     * Supports negative indices (last output row).
     * Mirrors `llama_context::get_logits_ith(i)`.
     */
    fun getLogitsIth(i: Int): FloatArray? {
        outputReorder()
        val buf = logits ?: return null
        val j = outputResolveRow(i)
        val nVocab = model.config.vocabSize
        val start = (j * nVocab).toInt()
        return buf.copyOfRange(start, start + nVocab)
    }

    /**
     * Return the full embeddings buffer.
     * Mirrors `llama_context::get_embeddings()`.
     */
    fun getEmbeddings(): FloatArray? {
        outputReorder()
        return embd
    }

    /**
     * Return a view of the embeddings for the i-th output token.
     * Mirrors `llama_context::get_embeddings_ith(i)`.
     */
    fun getEmbeddingsIth(i: Int): FloatArray? {
        outputReorder()
        val buf = embd ?: return null
        val j = outputResolveRow(i)
        val nEmbd = model.config.hiddenSize
        val start = (j * nEmbd).toInt()
        return buf.copyOfRange(start, start + nEmbd)
    }

    /**
     * Return sequence-level embeddings for the given sequence id.
     * Mirrors `llama_context::get_embeddings_seq(seq_id)`.
     */
    fun getEmbeddingsSeq(seqId: LlamaSeqId): FloatArray? = embdSeq[seqId]

    /** Return the sampled tokens array. Mirrors `llama_context::get_sampled_tokens()`. */
    fun getSampledTokens(): IntArray? = sampling.sampled

    // -----------------------------------------------------------------------
    // Model / cparams accessors
    // -----------------------------------------------------------------------

    fun getModel(): LlamaModel = model
    fun getCparams(): LlamaCParams = cparams

    /** Return the abstract memory interface (KV cache). */
    fun getMemory(): LlamaMemory? = memory

    // -----------------------------------------------------------------------
    // Thread configuration
    // -----------------------------------------------------------------------

    fun setNThreads(nThreads: Int, nThreadsBatch: Int) {
        cparams.nThreads = nThreads
        cparams.nThreadsBatch = nThreadsBatch
    }

    fun attachThreadpool(threadpool: Any?, threadpoolBatch: Any?) {
        this.threadpool = threadpool
        this.threadpoolBatch = threadpoolBatch ?: threadpool
    }

    fun detachThreadpool() {
        this.threadpool = null
        this.threadpoolBatch = null
    }

    // -----------------------------------------------------------------------
    // Attention / embedding mode setters
    // -----------------------------------------------------------------------

    fun setEmbeddings(value: Boolean) {
        cparams.embeddings = value
    }

    fun setCausalAttn(value: Boolean) {
        if (cparams.causalAttn == value) return
        cparams.causalAttn = value
        schedNeedReserve = true
    }

    fun setWarmup(value: Boolean) {
        if (cparams.warmup == value) return
        cparams.warmup = value
    }

    fun setAbortCallback(callback: (() -> Boolean)?) {
        this.abortCallback = callback
    }

    // -----------------------------------------------------------------------
    // Memory (KV cache) management
    // -----------------------------------------------------------------------

    /**
     * Initialise the KV cache layers based on the resolved context parameters.
     * Called during construction; can be called again after a context reset.
     */
    private fun initMemory() {
        val nLayers = model.config.numHiddenLayers
        val nHeads = model.config.numKeyValueHeads
        val headDim = model.config.headDim
        val maxSeqLen = cparams.nCtx

        kvCaches = Array(nLayers) {
            KVCache(
                maxSequenceLength = maxSeqLen,
                numHeads = nHeads,
                headDim = headDim,
            ).also { it.initialize(graphAllocator) }
        }
    }

    /**
     * Attempt to update (compact / defragment) the memory module.
     * Returns `true` if anything was actually changed.
     * Mirrors `llama_context::memory_update(optimize)`.
     *
     * Port of `llama_context::memory_update(bool optimize)`.
     */
    fun memoryUpdate(optimize: Boolean): Boolean {
        // LATER: full implementation
        //   1. Call memory.initUpdate(this, optimize)
        //   2. Check status — return false for NO_UPDATE or failures
        //   3. Reset previous graph result
        //   4. Call mctx.apply()
        //   5. Re-reserve worst-case graph
        return false
    }

    // -----------------------------------------------------------------------
    // Scheduler
    // -----------------------------------------------------------------------

    /**
     * Reserve a new backend scheduler if needed (e.g. after changing LoRA
     * adapters, samplers, or attention type).
     *
     * Port of `llama_context::sched_reserve()`.
     */
    fun schedReserve() {
        if (!schedNeedReserve) return
        schedNeedReserve = false

        synchronize()

        val nSeqs = cparams.nSeqMax
        val nTokens = minOf(cparams.nCtx, cparams.nUbatch)
        val maxNodes = graphMaxNodes(nTokens)

        // LATER: full implementation
        //   1. Create new scheduler with backend_ptrs/buft
        //   2. Initialize memory context (full)
        //   3. Resolve auto flash-attention
        //   4. Reserve pp and tg graphs
        //   5. Log buffer sizes and graph info
    }

    /**
     * Wait for all pending backend computations to finish.
     * Mirrors `llama_context::synchronize()`.
     *
     * Port of `llama_context::synchronize()`.
     */
    fun synchronize() {
        // Add the evaluation to the stats
        if (nQueuedTokens == 1L) {
            if (!cparams.noPerf) {
                tEvalUs += currentTimeMicros() - tComputeStartUs
            }
            nEval++
        } else if (nQueuedTokens > 1L) {
            if (!cparams.noPerf) {
                tPEvalUs += currentTimeMicros() - tComputeStartUs
            }
            nPEval += nQueuedTokens.toInt()
        }

        // Get a more accurate load time upon first eval
        if (nQueuedTokens > 0 && !hasEvaluatedOnce) {
            tLoadUs = currentTimeMicros() - tStartUs
            hasEvaluatedOnce = true
        }

        nQueuedTokens = 0
        tComputeStartUs = 0
    }

    // -----------------------------------------------------------------------
    // Core inference  (encode / decode)
    // -----------------------------------------------------------------------

    /**
     * Process a single micro-batch through the graph.
     *
     * If a [memoryContext] is provided it is applied first (e.g. to set up the
     * KV cache view for the current ubatch).
     *
     * Returns a pair of (graphResult, status). The graph result is `null` when
     * the computation fails.
     *
     * Mirrors `llama_context::process_ubatch(ubatch, gtype, mctx, ret)`.
     */
    fun processUBatch(
        ubatch: LlamaUBatch,
        gtype: LlmGraphType,
        memoryContext: LlamaMemoryContext? = null,
    ): Pair<GGMLCGraph?, Int> {
        val (res, status) = processUBatchInternal(ubatch, gtype, memoryContext)
        return (res?.getGf()) to status
    }

    /**
     * Encode a batch of tokens (non-causal / encoder path).
     *
     * Returns 0 on success, negative on failure (matches C++ convention).
     * Port of `llama_context::encode(const llama_batch & batch_inp)`.
     */
    fun encode(batchInp: LlamaBatch): Int {
        require((batchInp.tokens != null) xor (batchInp.embeddings != null)) {
            "Exactly one of tokens or embeddings must be set"
        }

        if (batchInp.nTokens == 0) {
            llamaLogError("encode: nTokens == 0\n")
            return -1
        }

        val nVocab = model.config.vocabSize
        val nEmbd = model.config.hiddenSize

        // Initialize the batch allocator
        val ba = balloc
        if (ba == null) {
            llamaLogError("encode: batch allocator not initialized\n")
            return -1
        }

        val nSeqMax = if (cparams.kvUnified) LLAMA_MAX_SEQ else cparams.nSeqMax

        if (!ba.init(batchInp, nVocab, nEmbd, nSeqMax, true)) {
            llamaLogError("encode: failed to initialize batch\n")
            return -1
        }

        val nTokens = ba.getNTokens()

        // micro-batching is not possible for non-causal encoding — process full batch
        require(cparams.nUbatch >= nTokens) { "encoder requires nUbatch >= nTokens" }

        if (tComputeStartUs == 0L) {
            tComputeStartUs = currentTimeMicros()
        }

        // Clear sequence embeddings
        embdSeq.clear()

        schedReserve()

        nQueuedTokens += nTokens.toLong()

        // Reserve output buffer
        val reserved = outputReserve(nTokens)
        if (reserved < nTokens) {
            llamaLogError("encode: could not reserve space for batch with $nTokens outputs\n")
            return -2
        }

        // Map all tokens to their output indices
        for (i in 0 until nTokens) {
            outputIds[i] = i
        }
        nOutputs = nTokens

        // Save and temporarily override causal attention
        val causalAttnOrig = cparams.causalAttn
        cparams.causalAttn = false

        // Build the ubatch from the full batch via splitSimple (processes everything at once)
        val ubatchInternal = ba.splitSimple(nTokens)
        val ubatch = ubatchInternal.toLlamaUBatch()

        // Process the ubatch
        val (res, status) = processUBatchInternal(ubatch, LlmGraphType.ENCODER, null)

        // Restore causal attention
        cparams.causalAttn = causalAttnOrig

        if (res == null) {
            return when (status) {
                GGMLStatus.ABORTED -> 2
                GGMLStatus.ALLOC_FAILED -> -2
                else -> -3
            }
        }

        // Extract logits
        val tLogits = res.tLogits
        val tEmbd = res.tEmbdPooled ?: res.tEmbd

        if (logits != null && tLogits != null) {
            val logitsBuf = logits
            val srcData = tLogits.data as? ByteArray
            if (logitsBuf != null && srcData != null) {
                val count = minOf(nTokens * nVocab, logitsSize)
                copyF32FromBytes(srcData, logitsBuf, 0, count)
            }
        }

        // Extract embeddings
        if (embd != null && tEmbd != null) {
            val embdBuf = embd!!
            val nEmbdOut = nEmbd
            when (cparams.poolingType) {
                LlamaPoolingType.NONE -> {
                    val srcData = tEmbd.data as? ByteArray
                    if (srcData != null) {
                        val count = minOf(nTokens * nEmbdOut, embdSize)
                        copyF32FromBytes(srcData, embdBuf, 0, count)
                    }
                }
                LlamaPoolingType.MEAN, LlamaPoolingType.CLS, LlamaPoolingType.LAST -> {
                    val srcData = tEmbd.data as? ByteArray
                    if (srcData != null) {
                        val seqIdUnqArr = ubatch.seqIdUnq
                        val seqIdxArr = ubatch.seqIdx
                        if (seqIdUnqArr != null && seqIdxArr != null) {
                            for (s in 0 until ubatch.nSeqsUnq) {
                                val seqId = seqIdUnqArr[s]
                                val seqIdx = seqIdxArr[seqId]
                                val seqEmbd = FloatArray(nEmbdOut)
                                copyF32FromBytes(srcData, seqEmbd, nEmbdOut * seqIdx, nEmbdOut)
                                embdSeq[seqId] = seqEmbd
                            }
                        }
                    }
                }
                LlamaPoolingType.RANK -> {
                    val srcData = tEmbd.data as? ByteArray
                    if (srcData != null) {
                        val seqIdUnqArr = ubatch.seqIdUnq
                        val seqIdxArr = ubatch.seqIdx
                        if (seqIdUnqArr != null && seqIdxArr != null) {
                            for (s in 0 until ubatch.nSeqsUnq) {
                                val seqId = seqIdUnqArr[s]
                                val seqIdx = seqIdxArr[seqId]
                                val rankEmbd = FloatArray(1)
                                copyF32FromBytes(srcData, rankEmbd, seqIdx, 1)
                                embdSeq[seqId] = rankEmbd
                            }
                        }
                    }
                }
                LlamaPoolingType.UNSPECIFIED -> {
                    error("unknown pooling type")
                }
            }
        }

        // T5 cross-attention: copy encoder output for later use by decoder
        if (tEmbd != null) {
            val tEmbdNe0 = tEmbd.ne[0]
            val tEmbdNe1 = tEmbd.ne[1]
            if (tEmbdNe0 > 0 && tEmbdNe1 > 0) {
                cross.nEmbd = tEmbdNe0
                cross.nEnc = tEmbdNe1
                val nFloats = (tEmbdNe0 * tEmbdNe1).toInt()
                cross.vEmbd.clear()
                val embdBuf = embd
                if (embdBuf != null && nFloats <= embdSize) {
                    for (k in 0 until nFloats) {
                        cross.vEmbd.add(embdBuf[k])
                    }
                }
                // Remember sequence IDs used during encoding
                val batchAfterInit = ba.getBatch()
                cross.seqIdsEnc.clear()
                val batchSeqId = batchAfterInit.seqId
                val batchNSeqId = batchAfterInit.nSeqId
                if (batchSeqId != null && batchNSeqId != null) {
                    for (i in 0 until nTokens) {
                        val seqSet = mutableSetOf<LlamaSeqId>()
                        for (s in 0 until batchNSeqId[i]) {
                            seqSet.add(batchSeqId[i][s])
                        }
                        cross.seqIdsEnc.add(seqSet)
                    }
                }
            }
        }

        return 0
    }

    /**
     * Decode a batch of tokens (causal / decoder path).
     *
     * Returns 0 on success, 1 when no KV slot available, negative on failure.
     * Port of `llama_context::decode(const llama_batch & batch_inp)`.
     */
    fun decode(batchInp: LlamaBatch): Int {
        require((batchInp.tokens != null) xor (batchInp.embeddings != null)) {
            "Exactly one of tokens or embeddings must be set"
        }

        // If no memory module, fall back to encode (encoder-only model)
        if (memory == null) {
            llamaLogDebug("decode: cannot decode without memory (calling encode instead)\n")
            return encode(batchInp)
        }

        if (batchInp.nTokens == 0) {
            llamaLogError("decode: nTokens == 0\n")
            return -1
        }

        val nVocab = model.config.vocabSize
        val nEmbd = model.config.hiddenSize

        // When computing embeddings, all tokens are output
        val outputAll = cparams.embeddings
        val hasSamplers = sampling.samplers.isNotEmpty()

        val nSeqMax = if (cparams.kvUnified) LLAMA_MAX_SEQ else cparams.nSeqMax

        // Validate: backend sampling requires at most one output token per sequence
        val batchLogits = batchInp.logits
        if (hasSamplers && batchLogits != null) {
            val seqOutputCount = IntArray(nSeqMax)
            for (i in 0 until batchInp.nTokens) {
                if (!batchLogits[i]) continue
                val ns = batchInp.nSeqId?.get(i) ?: 1
                for (s in 0 until ns) {
                    val seqId = batchInp.seqId?.get(i)?.get(s) ?: 0
                    seqOutputCount[seqId]++
                    if (seqOutputCount[seqId] > 1) {
                        llamaLogError("decode: backend sampling requires at most one output token per sequence (seq_id $seqId had ${seqOutputCount[seqId]})\n")
                        return -1
                    }
                }
            }
        }

        // Initialize the batch allocator
        val ba = balloc
        if (ba == null) {
            llamaLogError("decode: batch allocator not initialized\n")
            return -1
        }

        if (!ba.init(batchInp, nVocab, nEmbd, nSeqMax, outputAll, memory)) {
            llamaLogError("decode: failed to initialize batch\n")
            return -1
        }

        val nTokensAll = ba.getNTokens()
        val nOutputsAll = ba.getNOutputs()

        if (outputAll) {
            if (nOutputsAll != nTokensAll) {
                llamaLogError("decode: pooled embedding requires that all tokens are output (nOutputsAll=$nOutputsAll, nTokensAll=$nTokensAll)\n")
                return -1
            }
        }

        check(nTokensAll <= cparams.nBatch) { "batch size $nTokensAll exceeds n_batch ${cparams.nBatch}" }
        check(cparams.causalAttn || cparams.nUbatch >= nTokensAll) {
            "non-causal attention requires nUbatch >= nTokens"
        }

        if (tComputeStartUs == 0L) {
            tComputeStartUs = currentTimeMicros()
        }
        nQueuedTokens += nTokensAll.toLong()

        // Clear sequence embeddings and output swaps
        embdSeq.clear()
        outputSwaps.clear()

        schedReserve()

        var didOptimize = false

        // Handle any pending shifts/copies
        memoryUpdate(false)

        // Initialize memory context — retry after optimization if needed
        val mem = memory!!
        var mctx: LlamaMemoryContext
        while (true) {
            mctx = mem.initBatch(ba, cparams.nUbatch, outputAll)

            when (mctx.getStatus()) {
                LlamaMemoryStatus.SUCCESS -> break
                LlamaMemoryStatus.NO_UPDATE -> {
                    llamaLogError("decode: unexpected memory context status: NO_UPDATE\n")
                    return -2
                }
                LlamaMemoryStatus.FAILED_PREPARE -> {
                    if (!didOptimize) {
                        didOptimize = true
                        if (memoryUpdate(true)) {
                            llamaLogDebug("decode: retrying batch size ${ba.getNTokens()} after cache optimization\n")
                            continue
                        }
                    }
                    llamaLogWarn("decode: failed to find a memory slot for batch of size ${ba.getNTokens()}\n")
                    return 1
                }
                LlamaMemoryStatus.FAILED_COMPUTE -> {
                    llamaLogError("decode: compute failed while preparing batch of size ${ba.getNTokens()}\n")
                    return -2
                }
            }
        }

        // Reserve output buffer
        val reserved = outputReserve(nOutputsAll)
        if (reserved < nOutputsAll) {
            llamaLogError("decode: could not reserve space for batch with $nOutputsAll outputs\n")
            return -2
        }

        var nOutputsPrev = 0

        // Process ubatches from the memory context
        do {
            val ubatch = mctx.getUbatch()

            // Count the outputs in this ubatch
            val nOutputsNew = if (nOutputsAll == nTokensAll) {
                ubatch.nTokens
            } else {
                val outArr = ubatch.output
                if (outArr != null) {
                    var count = 0
                    for (i in 0 until ubatch.nTokens) {
                        if (outArr[i]) count++
                    }
                    count
                } else {
                    ubatch.nTokens
                }
            }

            // Set before graph build
            nOutputs = nOutputsNew

            val (res, status) = processUBatchInternal(ubatch, LlmGraphType.DECODER, mctx)

            if (res == null) {
                // Failed — remove all positions of this ubatch from the memory module
                val posMin = IntArray(LLAMA_MAX_SEQ) { Int.MAX_VALUE }

                val ubatchSeqId = ubatch.seqId
                val ubatchPos = ubatch.pos
                if (ubatchSeqId != null && ubatchPos != null) {
                    for (i in 0 until ubatch.nTokens) {
                        val seqId = ubatchSeqId[i][0]
                        if (ubatchPos[i] < posMin[seqId]) {
                            posMin[seqId] = ubatchPos[i]
                        }
                    }
                }

                for (s in 0 until LLAMA_MAX_SEQ) {
                    if (posMin[s] == Int.MAX_VALUE) continue
                    llamaLogWarn("decode: removing memory entries for seq_id=$s, pos=[${posMin[s]}, +inf)\n")
                    mem.seqRm(s, posMin[s], -1)
                }

                return when (status) {
                    GGMLStatus.ABORTED -> 2
                    GGMLStatus.ALLOC_FAILED -> -2
                    else -> -3
                }
            }

            // Extract logits
            val tLogits = res.tLogits
            if (logits != null && tLogits != null && nOutputsNew > 0 &&
                needsRawLogits(ubatch, sampling.samplers)
            ) {
                val logitsBuf = logits!!
                val srcData = tLogits.data as? ByteArray
                if (srcData != null) {
                    check(nOutputsPrev + nOutputsNew <= nOutputsAll)
                    val offset = nOutputsPrev * nVocab
                    val count = minOf(nOutputsNew * nVocab, logitsSize - offset)
                    if (count > 0) {
                        copyF32FromBytes(srcData, logitsBuf, offset, count)
                    }
                }
            }

            // Extract embeddings
            val tEmbd = if (cparams.embeddings) {
                val pooled = res.tEmbdPooled
                pooled ?: res.tEmbd
            } else null

            if (embd != null && tEmbd != null && nOutputsNew > 0) {
                val embdBuf = embd!!
                val nEmbdOut = nEmbd
                val srcData = tEmbd.data as? ByteArray

                when (cparams.poolingType) {
                    LlamaPoolingType.NONE -> {
                        if (srcData != null) {
                            val offset = nOutputsPrev * nEmbdOut
                            val count = minOf(nOutputsNew * nEmbdOut, embdSize - offset)
                            if (count > 0) {
                                copyF32FromBytes(srcData, embdBuf, offset, count)
                            }
                        }
                    }
                    LlamaPoolingType.MEAN, LlamaPoolingType.CLS, LlamaPoolingType.LAST -> {
                        if (srcData != null) {
                            val seqIdUnqArr = ubatch.seqIdUnq
                            val seqIdxArr = ubatch.seqIdx
                            if (seqIdUnqArr != null && seqIdxArr != null) {
                                for (s in 0 until ubatch.nSeqsUnq) {
                                    val seqId = seqIdUnqArr[s]
                                    val seqIdx = seqIdxArr[seqId]
                                    val seqEmbd = FloatArray(nEmbdOut)
                                    copyF32FromBytes(srcData, seqEmbd, nEmbdOut * seqIdx, nEmbdOut)
                                    embdSeq[seqId] = seqEmbd
                                }
                            }
                        }
                    }
                    LlamaPoolingType.RANK -> {
                        if (srcData != null) {
                            val seqIdUnqArr = ubatch.seqIdUnq
                            val seqIdxArr = ubatch.seqIdx
                            if (seqIdUnqArr != null && seqIdxArr != null) {
                                for (s in 0 until ubatch.nSeqsUnq) {
                                    val seqId = seqIdUnqArr[s]
                                    val seqIdx = seqIdxArr[seqId]
                                    val rankEmbd = FloatArray(1)
                                    copyF32FromBytes(srcData, rankEmbd, seqIdx, 1)
                                    embdSeq[seqId] = rankEmbd
                                }
                            }
                        }
                    }
                    LlamaPoolingType.UNSPECIFIED -> error("unknown pooling type")
                }
            }

            nOutputsPrev += nOutputsNew
        } while (mctx.next())

        // Set to total number of outputs in the batch
        nOutputs = nOutputsAll

        // Set output mappings
        if (nOutputsAll > 0) {
            var sortedOutput = true
            val outIds = ba.getOutIds()

            check(outIds.size == nOutputsAll) {
                "outIds.size=${outIds.size} != nOutputsAll=$nOutputsAll"
            }

            for (i in 0 until nOutputsAll) {
                val outId = outIds[i]
                outputIds[outId] = i
                if (outId != i) {
                    sortedOutput = false
                }
            }

            // Reorder outputs if not already sorted (relevant for recurrent models)
            if (!sortedOutput && nOutputsAll > 1) {
                // Selection sort to minimize swaps
                for (i in 0 until nOutputsAll - 1) {
                    var jMin = i
                    for (j in (i + 1) until nOutputsAll) {
                        if (outIds[j] < outIds[jMin]) {
                            jMin = j
                        }
                    }
                    if (jMin == i) continue

                    // Swap in outIds
                    val tmp = outIds[i]
                    outIds[i] = outIds[jMin]
                    outIds[jMin] = tmp

                    // Remember the swap for lazy logits/embeddings reordering
                    outputSwaps.add(SwapInfo(i, jMin))
                }

                // Rebuild output_ids mapping
                outputIds.fill(-1)
                for (i in 0 until nOutputsAll) {
                    outputIds[outIds[i]] = i
                }
            }
        }

        return 0
    }

    // -----------------------------------------------------------------------
    // Output management
    // -----------------------------------------------------------------------

    /**
     * Reserve space for at least [nOutputsRequested] output rows.
     * Returns the actual number of rows reserved.
     *
     * Port of `llama_context::output_reserve(int32_t n_outputs)`.
     */
    fun outputReserve(nOutputsRequested: Int): Int {
        val nOutputsMax = maxOf(nOutputsRequested.toLong(), nSeqMax().toLong()).toInt()
        val nVocab = model.config.vocabSize
        val nEmbd = model.config.hiddenSize
        val hasLogits = true
        val hasEmbd = cparams.embeddings

        val logitsNeeded = if (hasLogits) nVocab.toLong() * nOutputsMax else 0L
        val embdNeeded = if (hasEmbd) nEmbd.toLong() * nOutputsMax else 0L

        // Allocate or grow buffers as needed
        val currentLogitsSize = logitsSize.toLong()
        if (logitsNeeded > currentLogitsSize) {
            logits = FloatArray(logitsNeeded.toInt())
            logitsSize = logitsNeeded.toInt()
        }
        val currentEmbdSize = embdSize.toLong()
        if (embdNeeded > currentEmbdSize) {
            embd = FloatArray(embdNeeded.toInt())
            embdSize = embdNeeded.toInt()
        }

        // Init output_ids if not yet done
        if (outputIds.isEmpty()) {
            outputIds = IntArray(cparams.nBatch) { -1 }
        }

        // Reset all ids as invalid (negative)
        outputIds.fill(-1)
        nOutputs = 0

        return nOutputsMax
    }

    /**
     * Reorder the output buffers according to [outputSwaps] so that
     * logits / embeddings are in the expected order.
     *
     * Port of `llama_context::output_reorder()`.
     */
    private fun outputReorder() {
        val nVocab = model.config.vocabSize
        val nEmbd = model.config.hiddenSize
        val logitsBuf = logits
        val embdBuf = embd

        for (swap in outputSwaps) {
            val i0 = swap.i0
            val i1 = swap.i1

            if (logitsBuf != null && logitsSize > 0) {
                for (k in 0 until nVocab) {
                    val a = i0 * nVocab + k
                    val b = i1 * nVocab + k
                    val tmp = logitsBuf[a]
                    logitsBuf[a] = logitsBuf[b]
                    logitsBuf[b] = tmp
                }
            }

            if (embdBuf != null && embdSize > 0) {
                for (k in 0 until nEmbd) {
                    val a = i0 * nEmbd + k
                    val b = i1 * nEmbd + k
                    val tmp = embdBuf[a]
                    embdBuf[a] = embdBuf[b]
                    embdBuf[b] = tmp
                }
            }
        }

        outputSwaps.clear()
    }

    /**
     * Map a batch-token index [i] to the actual row inside the output buffer.
     * Supports negative indices (last output row).
     * Mirrors `llama_context::output_resolve_row(i)`.
     */
    private fun outputResolveRow(i: Int): Long {
        val j: Long = if (i < 0) {
            val resolved = nOutputs + i
            require(resolved >= 0) { "Negative index out of range [0, $nOutputs)" }
            resolved.toLong()
        } else {
            require(i < outputIds.size) { "Index out of range [0, ${outputIds.size})" }
            val mapped = outputIds[i]
            require(mapped >= 0) { "batch.logits[$i] != true" }
            mapped.toLong()
        }
        require(j < nOutputs) { "Corrupt output buffer (j=$j, nOutputs=$nOutputs)" }
        return j
    }

    // -----------------------------------------------------------------------
    // Graph helpers
    // -----------------------------------------------------------------------

    /**
     * Maximum number of nodes a graph may contain for the given token count.
     *
     * Port of `llama_context::graph_max_nodes(uint32_t n_tokens)`.
     */
    fun graphMaxNodes(nTokens: Int): Int {
        // Conservative estimate: 8× number of model tensors, minimum 1024
        val nTensors = model.config.numHiddenLayers * 16L
        return maxOf(1024, (8L * nTensors).toInt())
    }

    /**
     * Submit a compute graph for asynchronous execution.
     * Returns the GGML status code.
     *
     * Port of `llama_context::graph_compute(ggml_cgraph * gf, bool batched)`.
     */
    fun graphCompute(
        graph: GGMLCGraph,
        batched: Boolean,
    ): Int {
        val nThreadsToUse = if (batched) cparams.nThreadsBatch else cparams.nThreads
        // LATER: full backend scheduler integration
        //   - Set threadpool on CPU backend
        //   - Set n_threads on all backends
        //   - Call ggml_backend_sched_graph_compute_async
        return GGMLStatus.SUCCESS
    }

    /**
     * Reserve a graph with a dummy ubatch of the specified size.
     * Used during initialisation to pre-allocate backend buffers.
     *
     * Port of `llama_context::graph_reserve(...)`.
     */
    fun graphReserve(
        nTokens: Int,
        nSeqs: Int,
        nOutputsReq: Int,
    ): GGMLCGraph? {
        require(nOutputsReq >= 1) { "nOutputs must be >= 1" }

        var adjustedTokens = nTokens
        var adjustedOutputs = nOutputsReq

        if (adjustedTokens % nSeqs != 0) {
            adjustedTokens = ((adjustedTokens + (nSeqs - 1)) / nSeqs) * nSeqs
            adjustedOutputs = maxOf(adjustedOutputs, adjustedTokens)
        }

        // LATER: full implementation
        //   - Reset scheduler
        //   - Create dummy ubatch via batch allocator
        //   - Build graph via model.buildGraph(graphParams)
        //   - Schedule / allocate the graph
        //   - Return the compute graph

        return null // minimal — full backend integration pending
    }

    // -----------------------------------------------------------------------
    // State save / load
    // -----------------------------------------------------------------------

    /**
     * Compute the total serialized state size.
     * Port of `llama_context::state_get_size()`.
     */
    fun stateGetSize(): Long {
        val io = DummyIoWriter()
        return try {
            stateWriteData(io)
        } catch (_: Exception) {
            0L
        }
    }

    /**
     * Serialize the context state into [dst].
     * Port of `llama_context::state_get_data(dst, size)`.
     */
    fun stateGetData(dst: ByteArray, size: Long): Long {
        val io = StateBufferWriter(dst)
        return try {
            stateWriteData(io)
        } catch (_: Exception) {
            0L
        }
    }

    /**
     * Deserialize context state from [src].
     * Port of `llama_context::state_set_data(src, size)`.
     */
    fun stateSetData(src: ByteArray, size: Long): Long {
        val io = ByteArrayIoReader(src)
        return try {
            stateReadData(io)
        } catch (_: Exception) {
            0L
        }
    }

    /**
     * Compute serialized size for a single sequence.
     * Port of `llama_context::state_seq_get_size(seq_id, flags)`.
     */
    fun stateSeqGetSize(seqId: LlamaSeqId, flags: Int = 0): Long {
        val io = DummyIoWriter()
        return try {
            stateSeqWriteData(io, seqId, flags)
        } catch (_: Exception) {
            0L
        }
    }

    /**
     * Serialize a single sequence's state.
     * Port of `llama_context::state_seq_get_data(seq_id, dst, size, flags)`.
     */
    fun stateSeqGetData(seqId: LlamaSeqId, dst: ByteArray, size: Long, flags: Int = 0): Long {
        val io = StateBufferWriter(dst)
        return try {
            stateSeqWriteData(io, seqId, flags)
        } catch (_: Exception) {
            0L
        }
    }

    /**
     * Deserialize a single sequence's state.
     * Port of `llama_context::state_seq_set_data(seq_id, src, size, flags)`.
     */
    fun stateSeqSetData(seqId: LlamaSeqId, src: ByteArray, size: Long, flags: Int = 0): Long {
        val io = ByteArrayIoReader(src)
        return try {
            stateSeqReadData(io, seqId, flags)
        } catch (_: Exception) {
            0L
        }
    }

    /**
     * Save full context state to a file. Mirrors `llama_context::state_save_file()`.
     * Writes magic, version, token prompt, then state data via [stateWriteData].
     */
    fun stateSaveFile(filepath: String, tokens: IntArray, nTokenCount: Int): Boolean {
        // LATER: implement file I/O when LlamaFile is available
        return false
    }

    /**
     * Load full context state from a file. Mirrors `llama_context::state_load_file()`.
     * Reads magic/version, prompt tokens, then restores state via [stateReadData].
     */
    fun stateLoadFile(
        filepath: String,
        tokensOut: IntArray,
        nTokenCapacity: Int,
    ): Pair<Boolean, Int> {
        // LATER: implement file I/O when LlamaFile is available
        return Pair(false, 0)
    }

    /**
     * Save a single sequence state to a file. Mirrors `llama_context::state_seq_save_file()`.
     */
    fun stateSeqSaveFile(
        seqId: LlamaSeqId,
        filepath: String,
        tokens: IntArray,
        nTokenCount: Int,
    ): Long {
        // LATER: implement file I/O when LlamaFile is available
        return 0L
    }

    /**
     * Load a single sequence state from a file. Mirrors `llama_context::state_seq_load_file()`.
     */
    fun stateSeqLoadFile(
        seqId: LlamaSeqId,
        filepath: String,
        tokensOut: IntArray,
        nTokenCapacity: Int,
    ): Pair<Long, Int> {
        // LATER: implement file I/O when LlamaFile is available
        return Pair(0L, 0)
    }

    // -----------------------------------------------------------------------
    // LoRA adapter management
    // -----------------------------------------------------------------------

    /**
     * Set LoRA adapters with their scales. Mirrors `llama_context::set_adapters_lora()`.
     */
    fun setAdaptersLora(adapters: List<Any>, scales: FloatArray) {
        loras.clear()
        for (i in adapters.indices) {
            if (scales[i] != 0.0f) {
                loras[adapters[i]] = scales[i]
            }
        }
        schedNeedReserve = true
    }

    /**
     * Check if the provided adapters+scales match what's currently loaded.
     * Mirrors `llama_context::adapters_lora_are_same()`.
     */
    fun adaptersLoraAreSame(adapters: List<Any>, scales: FloatArray): Boolean {
        var nNonZero = 0
        for (i in adapters.indices) {
            if (scales[i] == 0.0f) continue
            nNonZero++
            val existing = loras[adapters[i]]
            if (existing == null || existing != scales[i]) return false
        }
        return nNonZero == loras.size
    }

    // -----------------------------------------------------------------------
    // Graph callback
    // -----------------------------------------------------------------------

    /**
     * Return a naming/backend-assignment callback for graph building.
     * Mirrors `llama_context::graph_get_cb()`.
     */
    fun graphGetCb(): (GGMLTensor, String, Int) -> Unit = { cur, name, il ->
        if (il >= 0) {
            cur.name = "$name-$il"
        } else {
            cur.name = name
        }
    }

    // -----------------------------------------------------------------------
    // Performance
    // -----------------------------------------------------------------------

    /** Collect timing / throughput statistics. Mirrors `llama_context::perf_get_data()`. */
    fun perfGetData(): LlamaPerfContextData = LlamaPerfContextData(
        tStartMs = tStartUs * 1e-3,
        tLoadMs = tLoadUs * 1e-3,
        tPEvalMs = tPEvalUs * 1e-3,
        tEvalMs = tEvalUs * 1e-3,
        nPEval = maxOf(1, nPEval),
        nEval = maxOf(1, nEval),
        nReused = maxOf(0, nReused),
    )

    /** Reset the performance counters. Mirrors `llama_context::perf_reset()`. */
    fun perfReset() {
        tStartUs = currentTimeMicros()
        tEvalUs = 0L; nEval = 0
        tPEvalUs = 0L; nPEval = 0
        nReused = 0
    }

    // -----------------------------------------------------------------------
    // Sampler management
    // -----------------------------------------------------------------------

    /**
     * Register a backend sampler for a given sequence id.
     *
     * Port of `llama_context::set_sampler(seq_id, sampler)`.
     */
    fun setSampler(seqId: LlamaSeqId, sampler: Any?): Boolean {
        if (sampler == null && seqId !in sampling.samplers) {
            return true
        }

        if (sampler != null) {
            // LATER: check can_offload and call backend_init
            sampling.samplers[seqId] = sampler
            schedNeedReserve = true
            return true
        }

        // Remove sampler for this sequence
        sampling.samplers.remove(seqId)
        schedNeedReserve = true
        return true
    }

    // -----------------------------------------------------------------------
    // Sampled token / logits / probs / candidates access
    // -----------------------------------------------------------------------

    /**
     * Get the sampled token for the i-th output position.
     * Port of `llama_context::get_sampled_token_ith(idx)`.
     */
    fun getSampledTokenIth(idx: Int): LlamaToken {
        outputReorder()
        val buf = sampling.sampled ?: return LLAMA_TOKEN_NULL
        if (buf.isEmpty()) return LLAMA_TOKEN_NULL
        return try {
            val row = outputResolveRow(idx)
            if (row < 0 || row >= buf.size) LLAMA_TOKEN_NULL else buf[row.toInt()]
        } catch (_: Exception) {
            LLAMA_TOKEN_NULL
        }
    }

    /**
     * Get the sampled logits for the i-th output position.
     * Port of `llama_context::get_sampled_logits_ith(idx)`.
     */
    fun getSampledLogitsIth(idx: Int): FloatArray? {
        outputReorder()
        val buf = sampling.logits ?: return null
        if (buf.isEmpty()) return null
        return try {
            val row = outputResolveRow(idx)
            val nVocab = model.config.vocabSize
            if (row < 0 || row.toInt() >= sampling.logitsCount.size || sampling.logitsCount[row.toInt()] == 0) {
                null
            } else {
                val start = (row * nVocab).toInt()
                buf.copyOfRange(start, start + nVocab)
            }
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Count of sampled logits for the i-th output.
     * Port of `llama_context::get_sampled_logits_count(idx)`.
     */
    fun getSampledLogitsCount(idx: Int): Int {
        outputReorder()
        val buf = sampling.logits ?: return model.config.vocabSize
        if (buf.isEmpty()) return model.config.vocabSize
        return try {
            val row = outputResolveRow(idx)
            if (row.toInt() >= sampling.logitsCount.size) 0 else sampling.logitsCount[row.toInt()]
        } catch (_: Exception) {
            0
        }
    }

    /**
     * Get the sampled probabilities for the i-th output position.
     * Port of `llama_context::get_sampled_probs_ith(idx)`.
     */
    fun getSampledProbsIth(idx: Int): FloatArray? {
        outputReorder()
        val buf = sampling.probs ?: return null
        if (buf.isEmpty()) return null
        return try {
            val row = outputResolveRow(idx)
            val nVocab = model.config.vocabSize
            if (row < 0 || row.toInt() >= sampling.probsCount.size || sampling.probsCount[row.toInt()] == 0) {
                null
            } else {
                val start = (row * nVocab).toInt()
                buf.copyOfRange(start, start + nVocab)
            }
        } catch (_: Exception) {
            null
        }
    }

    /**
     * Count of sampled probs for the i-th output.
     * Port of `llama_context::get_sampled_probs_count(idx)`.
     */
    fun getSampledProbsCount(idx: Int): Int {
        outputReorder()
        val buf = sampling.probs ?: return 0
        if (buf.isEmpty()) return 0
        return try {
            val row = outputResolveRow(idx)
            if (row.toInt() >= sampling.probsCount.size) 0 else sampling.probsCount[row.toInt()]
        } catch (_: Exception) {
            0
        }
    }

    /**
     * Get the candidate tokens for the i-th output position.
     * Port of `llama_context::get_sampled_candidates_ith(idx)`.
     */
    fun getSampledCandidatesIth(idx: Int): IntArray {
        outputReorder()
        val buf = sampling.candidates
        return try {
            val row = outputResolveRow(idx)
            if (buf != null && buf.isNotEmpty() &&
                row.toInt() < sampling.candidatesCount.size &&
                sampling.candidatesCount[row.toInt()] > 0
            ) {
                val nVocab = model.config.vocabSize
                val start = (row * nVocab).toInt()
                buf.copyOfRange(start, start + nVocab)
            } else {
                sampling.tokenIdsFullVocab
            }
        } catch (_: Exception) {
            sampling.tokenIdsFullVocab
        }
    }

    /**
     * Count of candidates for the i-th output.
     * Port of `llama_context::get_sampled_candidates_count(idx)`.
     */
    fun getSampledCandidatesCount(idx: Int): Int {
        outputReorder()
        val buf = sampling.candidates ?: return 0
        if (buf.isEmpty()) return 0
        return try {
            val row = outputResolveRow(idx)
            if (row.toInt() >= sampling.candidatesCount.size) 0 else sampling.candidatesCount[row.toInt()]
        } catch (_: Exception) {
            0
        }
    }

    // -----------------------------------------------------------------------
    // Control vector adapter
    // -----------------------------------------------------------------------

    /**
     * Apply a control vector to the model.
     * Port of `llama_context::set_adapter_cvec(data, len, n_embd, il_start, il_end)`.
     */
    fun setAdapterCvec(
        data: FloatArray,
        len: Int,
        nEmbd: Int,
        ilStart: Int,
        ilEnd: Int,
    ): Boolean {
        // LATER: full implementation — apply cvec to relevant layers
        schedNeedReserve = true
        return true
    }

    // -----------------------------------------------------------------------
    // Optimization (training)
    // -----------------------------------------------------------------------

    /**
     * Initialize the optimization context for training.
     * Port of `llama_context::opt_init(model, lopt_params)`.
     */
    fun optInit() {
        // LATER: full implementation
        //   - Set up ggml_opt context
        //   - Mark model parameters for gradient computation
        //   - Configure optimizer type and hyperparameters
    }

    /**
     * Run one training epoch over a dataset.
     * Port of `llama_context::opt_epoch(...)`.
     */
    fun optEpoch(
    ) {
        // LATER: full implementation
        //   - Iterate over dataset entries
        //   - Build graphs for each batch
        //   - Compute forward + backward
        //   - Apply optimizer step
    }

    // -----------------------------------------------------------------------
    // Memory breakdown
    // -----------------------------------------------------------------------

    /**
     * Collect per-backend memory usage.
     * Port of `llama_context::memory_breakdown()`.
     */
    fun memoryBreakdown(): Map<String, LlamaMemoryBreakdownData> {
        val result = mutableMapOf<String, LlamaMemoryBreakdownData>()
        // LATER: full implementation
        //   - Aggregate model buffer sizes
        //   - Aggregate memory (KV cache) buffer sizes
        //   - Aggregate compute buffer sizes from scheduler
        result["cpu"] = LlamaMemoryBreakdownData()
        return result
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /** Initialise the full-vocabulary token id array for backend samplers. */
    private fun initSamplingVocab() {
        val nVocab = model.config.vocabSize
        sampling.tokenIdsFullVocab = IntArray(nVocab) { it }
    }

    /**
     * Construct [LlmGraphParams] for a graph build invocation.
     *
     * Port of `llama_context::graph_params(...)`.
     */
    private fun graphParams(
        res: LlmGraphResult,
        ubatch: LlamaUBatch,
        gtype: LlmGraphType,
    ): LlmGraphParams {
        return LlmGraphParams(
            arch = LlamaModelArch.LLAMA,
            hparams = LlamaHparams(
                nEmbd = model.config.hiddenSize,
                nLayer = model.config.numHiddenLayers,
            ),
            cparams = cparams,
            ubatch = ubatch,
            gtype = gtype,
            nOutputs = nOutputs,
            res = res,
        )
    }

    /**
     * Process a single micro-batch through the graph (internal version).
     *
     * Port of `llama_context::process_ubatch(ubatch, gtype, mctx, ret)`.
     *
     * @return Pair of (LlmGraphResult or null, status code)
     */
    private fun processUBatchInternal(
        ubatch: LlamaUBatch,
        gtype: LlmGraphType,
        memoryContext: LlamaMemoryContext? = null,
    ): Pair<LlmGraphResult?, Int> {
        // Apply memory context (e.g. set up KV cache view for current ubatch)
        if (memoryContext != null && !memoryContext.apply()) {
            llamaLogError("processUBatch: failed to apply memory context\n")
            return null to GGMLStatus.FAILED
        }

        val res = gfResPrev as? LlmGraphResult
        if (res == null) {
            llamaLogError("processUBatch: no previous graph result available\n")
            return null to GGMLStatus.FAILED
        }

        val gparams = graphParams(res, ubatch, gtype).copy(mctx = memoryContext)

        if (!graphReuseDisable && res.canReuse(gparams)) {
            // Graph can be reused — synchronize if pipeline parallel to avoid
            // overwriting inputs that the previous compute is still reading.
            if (cparams.pipelineParallel) {
                synchronize()
            }
            nReused++
        } else {
            // Build a new graph
            res.reset()

            // In the full implementation this would call:
            //   ggml_backend_sched_reset(sched)
            //   gf = model.build_graph(gparams)
            //   ggml_backend_sched_alloc_graph(sched, gf)
            // For now, we build a minimal graph skeleton.
            val gf = GGMLCGraph(size = graphMaxNodes(ubatch.nTokens))
            res.gf = gf

            if (res.getGf() == null) {
                llamaLogError("processUBatch: failed to build graph\n")
                return null to GGMLStatus.FAILED
            }
        }

        // Set the input data for the input tensors
        res.setInputs(ubatch)

        // Execute graph compute
        val gf = res.getGf()
        if (gf == null) {
            llamaLogError("processUBatch: graph is null after setup\n")
            return null to GGMLStatus.FAILED
        }

        val status = graphCompute(gf, ubatch.nTokens > 1)
        if (status != GGMLStatus.SUCCESS) {
            llamaLogError("processUBatch: failed to compute graph, status: $status\n")
            return null to status
        }

        return res to GGMLStatus.SUCCESS
    }

    /**
     * Write full context state to an I/O writer.
     * Port of `llama_context::state_write_data(io)`.
     */
    private fun stateWriteData(io: LlamaIoWrite): Long {
        val archStr = "llama"
        io.writeString(archStr)

        val mem = memory
        if (mem != null) {
            mem.stateWrite(io)
        }

        return io.nBytes()
    }

    /**
     * Read full context state from an I/O reader.
     * Port of `llama_context::state_read_data(io)`.
     */
    private fun stateReadData(io: LlamaIoRead): Long {
        val archStr = io.readString()
        val expectedArch = "llama"
        require(archStr == expectedArch) {
            "wrong model arch: '$archStr' instead of '$expectedArch'"
        }

        val mem = memory
        if (mem != null) {
            mem.stateRead(io)
        }

        return io.nBytes()
    }

    /**
     * Write per-sequence state.
     * Port of `llama_context::state_seq_write_data(io, seq_id, flags)`.
     */
    private fun stateSeqWriteData(io: LlamaIoWrite, seqId: LlamaSeqId, flags: Int): Long {
        val mem = memory
        if (mem != null) {
            mem.stateWrite(io, seqId, flags)
        }
        return io.nBytes()
    }

    /**
     * Read per-sequence state.
     * Port of `llama_context::state_seq_read_data(io, seq_id, flags)`.
     */
    private fun stateSeqReadData(io: LlamaIoRead, seqId: LlamaSeqId, flags: Int): Long {
        val mem = memory
        if (mem != null) {
            mem.stateRead(io, seqId, flags)
        }
        return io.nBytes()
    }

    // -----------------------------------------------------------------------
    // Companion
    // -----------------------------------------------------------------------

    companion object {
        /**
         * Resolve raw [LlamaContextParams] + model [LlamaConfig] into the
         * computed [LlamaCParams] used throughout the context.
         *
         * This mirrors the first ~200 lines of the `llama_context` constructor
         * in `llama-context.cpp`.
         */
        internal fun resolveCParams(
            config: LlamaConfig,
            params: LlamaContextParams,
        ): LlamaCParams {
            val cp = LlamaCParams()

            cp.nSeqMax = maxOf(1, params.nSeqMax)
            require(cp.nSeqMax <= LLAMA_MAX_SEQ) { "nSeqMax must be <= $LLAMA_MAX_SEQ" }

            cp.nThreads = params.nThreads
            cp.nThreadsBatch = params.nThreadsBatch

            cp.yarnExtFactor = if (params.yarnExtFactor >= 0.0f) params.yarnExtFactor else 0.0f
            cp.yarnAttnFactor = if (params.yarnAttnFactor >= 0.0f) params.yarnAttnFactor else 1.0f
            cp.yarnBetaFast = if (params.yarnBetaFast >= 0.0f) params.yarnBetaFast else 32.0f
            cp.yarnBetaSlow = if (params.yarnBetaSlow >= 0.0f) params.yarnBetaSlow else 1.0f

            cp.embeddings = params.embeddings
            cp.offloadKqv = params.offloadKqv
            cp.noPerf = params.noPerf
            cp.warmup = false

            cp.nCtx = if (params.nCtx == 0) config.maxPositionEmbeddings else params.nCtx
            cp.ropeFreqBase = if (params.ropeFreqBase == 0.0f) config.ropeTheta else params.ropeFreqBase
            cp.ropeFreqScale = if (params.ropeFreqScale == 0.0f) 1.0f else params.ropeFreqScale

            cp.nCtxOrigYarn = when {
                params.yarnOrigCtx != 0 -> params.yarnOrigCtx
                else -> config.maxPositionEmbeddings
            }

            // Resolve attention type
            when (params.attentionType) {
                LlamaAttentionType.UNSPECIFIED -> cp.causalAttn = true
                LlamaAttentionType.CAUSAL -> cp.causalAttn = true
                LlamaAttentionType.NON_CAUSAL -> cp.causalAttn = false
            }

            cp.flashAttn = params.flashAttnType != LlamaFlashAttnType.DISABLED
            cp.autoFa = params.flashAttnType == LlamaFlashAttnType.AUTO

            // With causal attention the batch size is capped by context size
            cp.nBatch = if (cp.causalAttn) minOf(cp.nCtx, params.nBatch) else params.nBatch
            cp.nUbatch = minOf(cp.nBatch, if (params.nUbatch == 0) params.nBatch else params.nUbatch)

            cp.opOffload = params.opOffload
            cp.kvUnified = params.kvUnified

            // Pad n_ctx to a multiple of 256 (matches C++ GGML_PAD)
            cp.nCtx = ((cp.nCtx + 255) / 256) * 256

            if (cp.kvUnified) {
                cp.nCtxSeq = cp.nCtx
            } else {
                cp.nCtxSeq = cp.nCtx / cp.nSeqMax
                cp.nCtxSeq = ((cp.nCtxSeq + 255) / 256) * 256
                require(cp.nCtxSeq != 0) { "nCtxSeq == 0" }
                if (cp.nCtx != cp.nCtxSeq * cp.nSeqMax) {
                    cp.nCtx = cp.nCtxSeq * cp.nSeqMax
                }
            }

            // Resolve pooling type
            cp.poolingType = when (params.poolingType) {
                LlamaPoolingType.UNSPECIFIED -> LlamaPoolingType.NONE
                else -> params.poolingType
            }

            // Resolve rope scaling
            val ropeScaling = params.ropeScalingType
            if (ropeScaling == LlamaRopeScalingType.NONE) {
                cp.ropeFreqScale = 1.0f
            }
            if (cp.yarnExtFactor < 0.0f) {
                cp.yarnExtFactor =
                    if (ropeScaling == LlamaRopeScalingType.YARN) 1.0f else 0.0f
            }

            return cp
        }

        /** Microsecond clock — uses monotonic time source. */
        private val timeOrigin = TimeSource.Monotonic.markNow()
        internal fun currentTimeMicros(): Long =
            timeOrigin.elapsedNow().inWholeMicroseconds
    }
}

// =============================================================================
// Private I/O helpers for state serialization
// =============================================================================

/**
 * Measures how much data *would* be written (for computing state size).
 * Port of `llama_io_write_dummy`.
 */
private class DummyIoWriter : LlamaIoWrite {
    private var sizeWritten: Long = 0L
    override fun write(src: ByteArray, size: Int) { sizeWritten += size }
    override fun writeTensor(tensor: GGMLTensor, offset: Int, size: Int) { sizeWritten += size }
    override fun nBytes(): Long = sizeWritten
}

/**
 * Writes state into a pre-allocated [ByteArray].
 * Port of `llama_io_write_buffer`.
 */
private class StateBufferWriter(private val dst: ByteArray) : LlamaIoWrite {
    private var pos: Int = 0
    override fun write(src: ByteArray, size: Int) {
        require(pos + size <= dst.size) { "unexpectedly reached end of buffer" }
        src.copyInto(dst, pos, 0, size)
        pos += size
    }
    override fun writeTensor(tensor: GGMLTensor, offset: Int, size: Int) {
        // LATER: full tensor read-back from backend
        require(pos + size <= dst.size) { "unexpectedly reached end of buffer" }
        pos += size
    }
    override fun nBytes(): Long = pos.toLong()
}

// =============================================================================
// Free functions (static in C++)
// =============================================================================

/**
 * Build a mapping from sequence id to output row.
 * Port of static `build_seq_to_output_row()` from `llama-context.cpp`.
 */
fun buildSeqToOutputRow(ubatch: LlamaUBatch, rowOffset: Int): Map<LlamaSeqId, Int> {
    val seqToRow = mutableMapOf<LlamaSeqId, Int>()
    var local = 0
    val output = ubatch.output ?: return seqToRow
    val seqIds = ubatch.seqId ?: return seqToRow

    for (i in 0 until ubatch.nTokens) {
        if (!output[i]) continue
        val seqId = seqIds[i][0]
        seqToRow[seqId] = rowOffset + local
        local++
    }
    return seqToRow
}

/**
 * Check whether any output token needs raw logits (no backend sampler for its sequence).
 * Port of static `needs_raw_logits()` from `llama-context.cpp`.
 */
private fun needsRawLogits(ubatch: LlamaUBatch, samplers: Map<LlamaSeqId, Any>): Boolean {
    val output = ubatch.output ?: return true
    val seqIds = ubatch.seqId ?: return true

    for (i in 0 until ubatch.nTokens) {
        if (!output[i]) continue
        for (sid in seqIds[i]) {
            if (sid !in samplers) return true
        }
    }
    return false
}

/**
 * Copy [count] floats from a little-endian [src] ByteArray into [dst] starting
 * at index [dstOffset]. Each float is 4 bytes in the source.
 *
 * Used to simulate `ggml_backend_tensor_get_async` for host-accessible tensors.
 */
internal fun copyF32FromBytes(src: ByteArray, dst: FloatArray, dstOffset: Int, count: Int) {
    val effectiveCount = minOf(count, dst.size - dstOffset)
    for (i in 0 until effectiveCount) {
        val byteIdx = i * 4
        if (byteIdx + 3 >= src.size) break
        val bits = (src[byteIdx].toInt() and 0xFF) or
            ((src[byteIdx + 1].toInt() and 0xFF) shl 8) or
            ((src[byteIdx + 2].toInt() and 0xFF) shl 16) or
            ((src[byteIdx + 3].toInt() and 0xFF) shl 24)
        dst[dstOffset + i] = Float.fromBits(bits)
    }
}
