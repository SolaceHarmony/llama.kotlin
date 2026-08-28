// port-lint: source llama.cpp/src/llama-graph.h llama.cpp/src/llama-graph.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.min

// =============================================================================
// Graph type enums from llama-graph.h
// =============================================================================

/** Top-level graph variant used during inference. */
enum class LlmGraphType {
    DEFAULT,
    ENCODER,
    DECODER,
}

/** Activation / gating function applied inside feed-forward blocks. */
enum class LlmFfnOpType {
    SILU,
    GELU,
    RELU,
    RELU_SQR,
    SWIGLU,
    GEGLU,
    REGLU,
    SWIGLU_OAI_MOE,
}

/** How the FFN gate tensor is combined with the up-projection. */
enum class LlmFfnGateType {
    /** Gate applied sequentially. */
    SEQ,
    /** Gate applied in parallel. */
    PAR,
}

/** Normalisation variant applied to hidden states. */
enum class LlmNormType {
    NORM,
    RMS,
    GROUP,
}

/**
 * Callback that allows custom logic to be applied to each tensor during graph
 * construction (e.g. naming, offloading, ggml-alloc tagging).
 *
 * Port of `llm_graph_cb`.
 */
typealias LlmGraphCb = (ubatch: LlamaUBatch, cur: GGMLTensor, name: String, il: Int) -> Unit

/**
 * A single named input that must be populated before graph evaluation.
 *
 * Each implementation owns one or more [GGMLTensor] fields that are filled by
 * [setInput] from the current micro-batch and can optionally report whether a
 * previously built graph is still valid via [canReuse].
 */
interface LlmGraphInput {
    /** Populate owned tensors from the supplied micro-batch. */
    fun setInput(ubatch: LlamaUBatch)

    /** Return `true` when the graph built with the previous params can be reused. */
    fun canReuse(params: LlmGraphParams): Boolean = false
}

// =============================================================================
// Additional graph types from llama-graph.h
//
// from the full C++ llama-graph.h header.
// =============================================================================

// ---------------------------------------------------------------------------
// LlamaCross – encoder → decoder embedding shuttle
// Ported from: llama-graph.h  struct llama_cross
// ---------------------------------------------------------------------------

/**
 * Temporary container that carries encoder output embeddings into a decoder's
 * cross-attention layers.
 *
 * Port of `struct llama_cross` from `llama-graph.h`.
 */
data class LlamaCross(
    var nEmbd: Long = 0,
    var nEnc: Long = 0,

    /** Encoder output embeddings (host-side copy). */
    val vEmbd: MutableList<Float> = mutableListOf(),

    /**
     * Sequence IDs assigned to each encoder output position.
     * Needed for constructing the cross-attention mask in the decoder.
     */
    val seqIdsEnc: MutableList<MutableSet<LlamaSeqId>> = mutableListOf(),
)

// ---------------------------------------------------------------------------
// Additional graph-input types not yet in InferencePipeline.kt
// ---------------------------------------------------------------------------

/**
 * Temperature-tuning input (used by Llama4).
 *
 * Port of `llm_graph_input_attn_temp`.
 */
class LlmGraphInputAttnTemp(
    val nAttnTempFloorScale: Int,
    val fAttnTempScale: Float,
    val fAttnTempOffset: Float,
) : LlmGraphInput {
    var attnScale: GGMLTensor? = null  // F32 [n_batch]

    override fun setInput(ubatch: LlamaUBatch) {
        val scale = attnScale ?: return
        val positions = ubatch.pos ?: return
        val data = scale.data as? ByteArray ?: return
        val nTokens = ubatch.nTokens

        require(fAttnTempScale != 0.0f) { "fAttnTempScale must not be zero" }
        require(nAttnTempFloorScale != 0) { "nAttnTempFloorScale must not be zero" }

        for (i in 0 until nTokens) {
            val pos = positions[i].toFloat()
            val value = ln(
                floor((pos + fAttnTempOffset) / nAttnTempFloorScale) + 1.0
            ).toFloat() * fAttnTempScale + 1.0f
            val bits = value.toRawBits()
            val offset = i * 4
            data[offset + 0] = (bits and 0xFF).toByte()
            data[offset + 1] = ((bits shr 8) and 0xFF).toByte()
            data[offset + 2] = ((bits shr 16) and 0xFF).toByte()
            data[offset + 3] = ((bits shr 24) and 0xFF).toByte()
        }
    }
}

/**
 * Relative-position bucket input (encoder self-attention, e.g. T5).
 *
 * Port of `llm_graph_input_pos_bucket`.
 */
class LlmGraphInputPosBucket(
    val hparams: LlamaHparams,
) : LlmGraphInput {
    var posBucket: GGMLTensor? = null  // I32 [n_batch, n_batch]

    override fun setInput(ubatch: LlamaUBatch) {
        val bucket = posBucket ?: return
        val positions = ubatch.pos ?: return
        val data = bucket.data as? ByteArray ?: return
        val nTokens = ubatch.nTokens

        require(!ubatch.equalSeqs) { "pos_bucket requires non-equal sequences" }

        for (j in 0 until nTokens) {
            for (i in 0 until nTokens) {
                val value = llamaRelativePositionBucket(
                    positions[i], positions[j],
                    hparams.nRelAttnBkts.toLong(), bidirectional = true
                )
                val offset = (j * nTokens + i) * 4
                data[offset + 0] = (value and 0xFF).toByte()
                data[offset + 1] = ((value shr 8) and 0xFF).toByte()
                data[offset + 2] = ((value shr 16) and 0xFF).toByte()
                data[offset + 3] = ((value shr 24) and 0xFF).toByte()
            }
        }
    }
}

/**
 * Relative-position bucket input for KV-cached decoder (T5-style).
 *
 * Port of `llm_graph_input_pos_bucket_kv`.
 */
class LlmGraphInputPosBucketKv(
    val hparams: LlamaHparams,
) : LlmGraphInput {
    var posBucket: GGMLTensor? = null  // I32 [n_kv, n_batch]

    override fun setInput(ubatch: LlamaUBatch) {
        // In C++, this delegates to mctx->set_input_pos_bucket(pos_bucket, ubatch).
        // The KV cache memory context computes relative-position buckets between
        // query positions (from ubatch) and cached key positions (from the KV store).
        // Requires LlamaKvCacheContext.setInputPosBucket() integration.
        val bucket = posBucket ?: return
    }
}

/**
 * Cross-attention encoder embeddings input.
 *
 * Port of `llm_graph_input_cross_embd`.
 */
class LlmGraphInputCrossEmbd(
    val cross: LlamaCross?,
) : LlmGraphInput {
    var crossEmbd: GGMLTensor? = null  // F32 [n_embd, n_outputs_enc]

    override fun setInput(ubatch: LlamaUBatch) {
        val unused = ubatch // ubatch is unused in C++ implementation

        val embd = crossEmbd ?: return
        val cr = cross ?: return
        if (cr.vEmbd.isEmpty()) return

        val data = embd.data as? ByteArray ?: return
        // Copy encoder embeddings (F32) into tensor data
        val nBytes = minOf(cr.vEmbd.size * 4, data.size)
        for (i in 0 until nBytes / 4) {
            val bits = cr.vEmbd[i].toRawBits()
            val offset = i * 4
            data[offset + 0] = (bits and 0xFF).toByte()
            data[offset + 1] = ((bits shr 8) and 0xFF).toByte()
            data[offset + 2] = ((bits shr 16) and 0xFF).toByte()
            data[offset + 3] = ((bits shr 24) and 0xFF).toByte()
        }
    }
}

/**
 * Input for self-attention backed by a KV cache.
 *
 * Holds KV-index tensors and the causal mask used during self-attention with
 * cached keys/values.  Ports `llm_graph_input_attn_kv` from `llama-graph.h`.
 */
// port-lint: source llama.cpp/src/llama-graph.h  llm_graph_input_attn_kv
class LlmGraphInputAttnKv(
    val hparams: LlamaHparams,
    val cparams: LlamaCParams,
    var mctx: KVCacheContext? = null,
) : LlmGraphInput {
    var selfKIdxs: GGMLTensor? = null       // I64 [nBatch]
    var selfVIdxs: GGMLTensor? = null       // I64 [nBatch] or [nBatch*nEmbdVGqa]
    var selfKqMask: GGMLTensor? = null      // F32 [nKv, nBatch/nStream, 1, nStream]
    var selfKqMaskCnv: GGMLTensor? = null

    // note: assumes v_rot^2 == I
    var selfKRot: GGMLTensor? = null
    var selfVRot: GGMLTensor? = null

    fun getKIdxs(): GGMLTensor? = selfKIdxs
    fun getVIdxs(): GGMLTensor? = selfVIdxs
    fun getKqMask(): GGMLTensor? = selfKqMaskCnv

    // port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_input_attn_kv::set_input
    override fun setInput(ubatch: LlamaUBatch) {
        val m = mctx ?: return
        selfKIdxs?.let { m.setInputKIdxs(it, ubatch) }
        selfVIdxs?.let { m.setInputVIdxs(it, ubatch) }
        selfKqMask?.let { m.setInputKqMask(it, ubatch, cparams.causalAttn) }
        selfKRot?.let { m.setInputKRot(it) }
        selfVRot?.let { m.setInputVRot(it) }
    }

    // port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_input_attn_kv::can_reuse
    override fun canReuse(params: LlmGraphParams): Boolean {
        val m = params.mctx as? KVCacheContext ?: return false
        this.mctx = m

        var res = true
        res = res && (selfKIdxs?.ne?.get(0) == params.ubatch.nTokens.toLong())
        // res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // need to move to unified cache
        selfKqMask?.let { mask ->
            res = res && canReuseKqMask(mask, m, params.ubatch, params.cparams)
        }
        return res
    }
}

/**
 * V-less KV cache input (stores only K, no V).
 *
 * Port of `llm_graph_input_attn_k`.
 */
// port-lint: source llama.cpp/src/llama-graph.h  llm_graph_input_attn_k
class LlmGraphInputAttnK(
    val hparams: LlamaHparams,
    val cparams: LlamaCParams,
    var mctx: KVCacheContext? = null,
) : LlmGraphInput {
    var selfKIdxs: GGMLTensor? = null       // I64 [n_batch]
    var selfKqMask: GGMLTensor? = null      // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    var selfKqMaskCnv: GGMLTensor? = null

    fun getKIdxs(): GGMLTensor? = selfKIdxs
    fun getKqMask(): GGMLTensor? = selfKqMaskCnv

    // port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_input_attn_k::set_input
    override fun setInput(ubatch: LlamaUBatch) {
        val m = mctx ?: return
        selfKIdxs?.let { m.setInputKIdxs(it, ubatch) }
        selfKqMask?.let { m.setInputKqMask(it, ubatch, cparams.causalAttn) }
    }

    // port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_input_attn_k::can_reuse
    override fun canReuse(params: LlmGraphParams): Boolean {
        val m = params.mctx as? KVCacheContext ?: return false
        this.mctx = m

        var res = true
        res = res && (selfKIdxs?.ne?.get(0) == params.ubatch.nTokens.toLong())
        selfKqMask?.let { mask ->
            res = res && canReuseKqMask(mask, m, params.ubatch, params.cparams)
        }
        return res
    }
}

/**
 * ISWA (interleaved sliding-window attention) KV cache input.
 *
 * Port of `llm_graph_input_attn_kv_iswa`.
 */
class LlmGraphInputAttnKvIswa(
    val hparams: LlamaHparams,
    val cparams: LlamaCParams,
) : LlmGraphInput {
    // Full-attention stream
    var selfKIdxs: GGMLTensor? = null
    var selfVIdxs: GGMLTensor? = null
    var selfKqMask: GGMLTensor? = null
    var selfKqMaskCnv: GGMLTensor? = null

    // SWA stream
    var selfKIdxsSwa: GGMLTensor? = null
    var selfVIdxsSwa: GGMLTensor? = null
    var selfKqMaskSwa: GGMLTensor? = null
    var selfKqMaskSwaCnv: GGMLTensor? = null

    // RoPE rotation tensors
    var selfKRot: GGMLTensor? = null
    var selfVRot: GGMLTensor? = null
    var selfKRotSwa: GGMLTensor? = null
    var selfVRotSwa: GGMLTensor? = null

    fun getKIdxs(): GGMLTensor? = selfKIdxs
    fun getVIdxs(): GGMLTensor? = selfVIdxs
    fun getKIdxsSwa(): GGMLTensor? = selfKIdxsSwa
    fun getVIdxsSwa(): GGMLTensor? = selfVIdxsSwa
    fun getKqMask(): GGMLTensor? = selfKqMaskCnv
    fun getKqMaskSwa(): GGMLTensor? = selfKqMaskSwaCnv

    override fun setInput(ubatch: LlamaUBatch) {
        // In C++, this delegates to mctx->get_base() and mctx->get_swa() for:
        //   - set_input_k_idxs / set_input_v_idxs for both base and SWA streams
        //   - set_input_kq_mask for both base and SWA streams
        //   - set_input_k_rot / set_input_v_rot (optional) for both streams
        // Requires LlamaKvCacheIswaContext integration.
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val kIdxs = selfKIdxs ?: return false
        val kIdxsSwa = selfKIdxsSwa ?: return false
        val kqMask = selfKqMask ?: return false
        val kqMaskSwa = selfKqMaskSwa ?: return false

        var res = true

        // Both base and SWA K-index tensors must match token count
        res = res && (kIdxs.ne[0] == params.ubatch.nTokens.toLong())
        res = res && (kIdxsSwa.ne[0] == params.ubatch.nTokens.toLong())

        // KQ mask dimension checks for base stream
        val nStream = if (params.cparams.kvUnified) 1L else params.ubatch.nSeqsUnq.toLong()
        res = res && (kqMask.ne[1] == params.ubatch.nTokens.toLong() / nStream)
        res = res && (kqMask.ne[2] == 1L)
        res = res && (kqMask.ne[3] == nStream)

        // KQ mask dimension checks for SWA stream
        res = res && (kqMaskSwa.ne[1] == params.ubatch.nTokens.toLong() / nStream)
        res = res && (kqMaskSwa.ne[2] == 1L)
        res = res && (kqMaskSwa.ne[3] == nStream)

        return res
    }
}

/**
 * Hybrid memory input (attention KV + recurrent state).
 *
 * Port of `llm_graph_input_mem_hybrid`.
 */
class LlmGraphInputMemHybrid(
    val cparams: LlamaCParams,
    val inpAttn: LlmGraphInputAttnKv,
    val inpRs: LlmGraphInputRs,
) : LlmGraphInput {

    fun getAttn(): LlmGraphInputAttnKv = inpAttn
    fun getRecr(): LlmGraphInputRs = inpRs

    override fun setInput(ubatch: LlamaUBatch) {
        inpAttn.setInput(ubatch)
        inpRs.setInput(ubatch)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        return inpAttn.canReuse(params) && inpRs.canReuse(params)
    }
}

/**
 * Hybrid memory input (K-only attention + recurrent state).
 *
 * Port of `llm_graph_input_mem_hybrid_k`.
 */
class LlmGraphInputMemHybridK(
    val cparams: LlamaCParams,
    val inpAttn: LlmGraphInputAttnK,
    val inpRs: LlmGraphInputRs,
) : LlmGraphInput {

    fun getAttn(): LlmGraphInputAttnK = inpAttn
    fun getRecr(): LlmGraphInputRs = inpRs

    override fun setInput(ubatch: LlamaUBatch) {
        inpAttn.setInput(ubatch)
        inpRs.setInput(ubatch)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        return inpAttn.canReuse(params) && inpRs.canReuse(params)
    }
}

/**
 * Hybrid memory input (ISWA attention + recurrent state).
 *
 * Port of `llm_graph_input_mem_hybrid_iswa`.
 */
class LlmGraphInputMemHybridIswa(
    val cparams: LlamaCParams,
    val inpAttn: LlmGraphInputAttnKvIswa,
    val inpRs: LlmGraphInputRs,
) : LlmGraphInput {

    fun getAttn(): LlmGraphInputAttnKvIswa = inpAttn
    fun getRecr(): LlmGraphInputRs = inpRs

    override fun setInput(ubatch: LlamaUBatch) {
        inpAttn.setInput(ubatch)
        inpRs.setInput(ubatch)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        return inpAttn.canReuse(params) && inpRs.canReuse(params)
    }
}

/**
 * Backend-sampling input (carries per-sequence sampler references).
 *
 * Port of `llm_graph_input_sampling`.
 */
class LlmGraphInputSampling(
    val samplers: Map<LlamaSeqId, Any>,  // will be LlamaSampler* when sampler infra is ported
) : LlmGraphInput {

    override fun setInput(ubatch: LlamaUBatch) {
        // Collect active seq_ids: those with output[i] == true
        val output = ubatch.output ?: return
        val seqIds = ubatch.seqId ?: return
        val activeSamplers = mutableSetOf<LlamaSeqId>()

        for (i in 0 until ubatch.nTokens) {
            if (output[i]) {
                val seqId = seqIds[i][0]
                activeSamplers.add(seqId)
            }
        }

        // For each active sampler, invoke its backend_set_input if available.
        // Currently samplers are typed as Any; when the sampler
        // interface is fully ported, this will call sampler.iface.backendSetInput().
        for (seqId in activeSamplers) {
            val sampler = samplers[seqId] ?: continue
            // sampler.iface.backendSetInput(sampler) — requires sampler interface port
            val unused = sampler // sampler.iface.backendSetInput(sampler) — requires sampler interface port
        }
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        // In C++, compares samplers map: sizes must match and all entries identical.
        // LlmGraphParams doesn't carry samplers yet, so conservatively return false
        // to force rebuild until the sampler infrastructure is integrated.
        return false
    }
}


// =============================================================================
// Standalone utility
// Ported from: llama-graph.cpp  llama_relative_position_bucket
// =============================================================================

/**
 * Compute a relative-position bucket index (used by T5-style position bias).
 *
 * The bucketing scheme uses exact positions for small offsets and logarithmic
 * buckets for larger distances. With `bidirectional = true` half the buckets
 * are reserved for negative (backward) relative positions.
 *
 * Port of `llama_relative_position_bucket()` from `llama-graph.cpp`.
 */
// port-lint: source llama.cpp/src/llama-graph.cpp
fun llamaRelativePositionBucket(
    x: LlamaPos,
    y: LlamaPos,
    nBuckets: Long,
    bidirectional: Boolean,
): Int {
    var relativePosition = (x - y).toLong()
    var numBuckets = nBuckets.toInt()
    var ret = 0

    if (bidirectional) {
        numBuckets /= 2
        if (relativePosition > 0) {
            ret += numBuckets
        } else {
            relativePosition = -relativePosition
        }
    } else {
        if (relativePosition > 0) {
            relativePosition = 0
        } else {
            relativePosition = -relativePosition
        }
    }

    val maxExact = numBuckets / 2
    // Small relative positions use exact bucket mapping
    if (relativePosition < maxExact) {
        return ret + relativePosition.toInt()
    }

    // Larger relative positions use logarithmic buckets
    val valIf = (maxExact +
        (kotlin.math.ln(relativePosition.toDouble() / maxExact) /
            kotlin.math.ln(((nBuckets * 4).toDouble()) / maxExact) *
            (numBuckets - maxExact)).toInt())
    val bucket = minOf(valIf, numBuckets - 1)
    return ret + bucket
}

// ---------------------------------------------------------------------------
// Static helpers (mirror C++ file-scope statics in llama-graph.cpp)
// ---------------------------------------------------------------------------

// port-lint: source llama.cpp/src/llama-graph.cpp  ggml_mul_mat_aux
/** Reshape, multiply by rotation matrix, reshape back. Used for Hadamard rotations. */
internal fun ggmlMulMatAux(
    ctx: GGMLContext,
    cur: GGMLTensor,
    rot: GGMLTensor
): GGMLTensor {
    val n = rot.ne[0]
    var res = ggmlReshape2d(ctx, cur, n, ggmlNelements(cur) / n)
    res = ggmlMulMat(ctx, rot, res)
    res = ggmlReshape4d(ctx, res, cur.ne[0], cur.ne[1], cur.ne[2], cur.ne[3])
    return res
}

// port-lint: source llama.cpp/src/llama-graph.cpp  can_reuse_kq_mask
/** Check if a previously-built KQ mask can be reused for the new params. */
internal fun canReuseKqMask(
    kqMask: GGMLTensor,
    mctx: KVCacheContext,
    ubatch: LlamaUBatch,
    cparams: LlamaCParams
): Boolean {
    val nKv = mctx.getNKv().toLong()
    val nTokens = ubatch.nTokens.toLong()
    val nStream = if (cparams.kvUnified) 1L else ubatch.nSeqsUnq.toLong()

    var res = true
    res = res && (kqMask.ne[0] == nKv)
    res = res && (kqMask.ne[1] == nTokens / nStream)
    res = res && (kqMask.ne[2] == 1L)
    res = res && (kqMask.ne[3] == nStream)
    return res
}

// port-lint: source llama.cpp/src/llama-graph.cpp  build_attn_inp_kq_mask
/** Build the KQ attention mask tensor. Port of `build_attn_inp_kq_mask`. */
internal fun buildAttnInpKqMask(
    ctx: GGMLContext,
    mctx: KVCacheContext,
    ubatch: LlamaUBatch,
    cparams: LlamaCParams
): GGMLTensor {
    val nKv = mctx.getNKv().toLong()
    val nTokens = ubatch.nTokens.toLong()
    val nStream = if (cparams.kvUnified) 1L else ubatch.nSeqsUnq.toLong()

    val res = ggmlNewTensor4d(ctx, GGMLType.F32, nKv, nTokens / nStream, 1, nStream)
    ggmlSetInput(res)
    ggmlSetName(res, "attn_inp_kq_mask")
    return res
}

// port-lint: source llama.cpp/src/llama-graph.cpp  print_mask
/** Debug-print an attention mask (rows = query tokens, cols = KV tokens). */
internal fun printMask(data: FloatArray, nTokens: Long, nKv: Long, nSwa: Long, swaType: LlamaSwaType) {
    val swaTypeStr = when (swaType) {
        LlamaSwaType.NONE      -> "LLAMA_SWA_TYPE_NONE"
        LlamaSwaType.STANDARD  -> "LLAMA_SWA_TYPE_STANDARD"
        LlamaSwaType.CHUNKED   -> "LLAMA_SWA_TYPE_CHUNKED"
        LlamaSwaType.SYMMETRIC -> "LLAMA_SWA_TYPE_SYMMETRIC"
    }
    println("print_mask: === Attention mask ===")
    println("print_mask: n_swa : $nSwa, n_kv: $nKv, swa_type: $swaTypeStr")
    println("print_mask: '0' = can attend, '∞' = masked")
    println("print_mask: Rows = query tokens, Columns = key/value tokens\n")

    val colLimit = minOf(20L, nKv)
    val rowLimit = minOf(20L, nTokens)

    print("    ")
    for (j in 0 until colLimit) print("${j.toString().padStart(2)}")
    println()

    for (i in 0 until rowLimit) {
        print(" ${i.toString().padStart(2)} ")
        for (j in 0 until colLimit) {
            val v = data[(i * nKv + j).toInt()]
            if (v == Float.NEGATIVE_INFINITY) print(" ∞") else print(" 0")
        }
        println()
    }
}

// port-lint: source llama.cpp/src/llama-graph.cpp  build_attn_inp_kv_impl
/** Build input for KV-cache attention. File-scope static. */
internal fun buildAttnInpKvImpl(
    ctx0: GGMLContext,
    ubatch: LlamaUBatch,
    hparams: LlamaHparams,
    cparams: LlamaCParams,
    mctxCur: KVCacheContext,
): LlmGraphInputAttnKv {
    val inp = LlmGraphInputAttnKv(hparams, cparams, mctxCur)

    require(hparams.swaType == LlamaSwaType.NONE) { "Use llama_kv_cache_iswa for SWA" }

    inp.selfKIdxs = mctxCur.buildInputKIdxs(ctx0, ubatch)
    inp.selfVIdxs = mctxCur.buildInputVIdxs(ctx0, ubatch)

    inp.selfKqMask = buildAttnInpKqMask(ctx0, mctxCur, ubatch, cparams)
    inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(ctx0, inp.selfKqMask!!, GGMLType.F16) else inp.selfKqMask

    inp.selfKRot = mctxCur.buildInputKRot(ctx0)
    inp.selfVRot = mctxCur.buildInputVRot(ctx0)

    return inp
}

// port-lint: source llama.cpp/src/llama-graph.cpp  build_attn_inp_k_impl
/** Build input for K-only cache attention. File-scope static. */
internal fun buildAttnInpKImpl(
    ctx0: GGMLContext,
    ubatch: LlamaUBatch,
    hparams: LlamaHparams,
    cparams: LlamaCParams,
    mctxCur: KVCacheContext,
): LlmGraphInputAttnK {
    val inp = LlmGraphInputAttnK(hparams, cparams, mctxCur)

    require(hparams.swaType == LlamaSwaType.NONE) { "Use llama_kv_cache_iswa for SWA" }

    inp.selfKIdxs = mctxCur.buildInputKIdxs(ctx0, ubatch)

    inp.selfKqMask = buildAttnInpKqMask(ctx0, mctxCur, ubatch, cparams)
    inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(ctx0, inp.selfKqMask!!, GGMLType.F16) else inp.selfKqMask

    return inp
}

// port-lint: source llama.cpp/src/llama-graph.cpp  build_rs_inp_impl
/** Build input for recurrent-state memory. File-scope static. */
internal fun buildRsInpImpl(
    ctx0: GGMLContext,
    ubatch: LlamaUBatch,
    mctxCur: LlamaMemoryRecurrentContext,
): LlmGraphInputRs {
    val inp = LlmGraphInputRs(mctxCur)

    val nRs = mctxCur.getNRs().toLong()
    val nSeqs = ubatch.nSeqs.toLong()

    inp.sCopy = ggmlNewTensor1d(ctx0, GGMLType.I32, nRs)
    ggmlSetInput(inp.sCopy!!)

    inp.sCopyMain  = ggmlView1d(ctx0, inp.sCopy!!, nSeqs, 0UL)
    inp.sCopyExtra = ggmlView1d(ctx0, inp.sCopy!!, nRs - nSeqs, nSeqs.toULong() * inp.sCopy!!.nb[0])

    inp.head = mctxCur.getHead()
    inp.rsZ  = mctxCur.getRsZ()

    return inp
}

// ---------------------------------------------------------------------------
// Tensor data-writing helpers (mirror ggml_backend_tensor_set)
// ---------------------------------------------------------------------------

/** Write an IntArray into a tensor's backing ByteArray at byte [offset]. */
internal fun tensorSetI32(tensor: GGMLTensor, src: IntArray, offset: Int, count: Int) {
    val data = tensor.data as? ByteArray ?: return
    for (i in 0 until count) {
        val v = src[i]
        val pos = offset + i * 4
        data[pos]     = (v and 0xFF).toByte()
        data[pos + 1] = ((v shr 8) and 0xFF).toByte()
        data[pos + 2] = ((v shr 16) and 0xFF).toByte()
        data[pos + 3] = ((v shr 24) and 0xFF).toByte()
    }
}

/** Write a FloatArray into a tensor's backing ByteArray at byte [offset]. */
internal fun tensorSetF32(tensor: GGMLTensor, src: FloatArray, offset: Int, count: Int) {
    val data = tensor.data as? ByteArray ?: return
    for (i in 0 until count) {
        val bits = src[i].toRawBits()
        val pos = offset + i * 4
        data[pos]     = (bits and 0xFF).toByte()
        data[pos + 1] = ((bits shr 8) and 0xFF).toByte()
        data[pos + 2] = ((bits shr 16) and 0xFF).toByte()
        data[pos + 3] = ((bits shr 24) and 0xFF).toByte()
    }
}

/** Write a single float into a tensor at byte [offset]. */
internal fun tensorSetF32Single(tensor: GGMLTensor, value: Float, offset: Int) {
    val data = tensor.data as? ByteArray ?: return
    val bits = value.toRawBits()
    data[offset]     = (bits and 0xFF).toByte()
    data[offset + 1] = ((bits shr 8) and 0xFF).toByte()
    data[offset + 2] = ((bits shr 16) and 0xFF).toByte()
    data[offset + 3] = ((bits shr 24) and 0xFF).toByte()
}

/** Write a single I32 into a tensor at byte [offset]. */
internal fun tensorSetI32Single(tensor: GGMLTensor, value: Int, offset: Int) {
    val data = tensor.data as? ByteArray ?: return
    data[offset]     = (value and 0xFF).toByte()
    data[offset + 1] = ((value shr 8) and 0xFF).toByte()
    data[offset + 2] = ((value shr 16) and 0xFF).toByte()
    data[offset + 3] = ((value shr 24) and 0xFF).toByte()
}

// ---------------------------------------------------------------------------
// Graph input implementations (moved from LlmGraphTypes.kt)
// ---------------------------------------------------------------------------

// -- embedding tokens -------------------------------------------------------

/** Input: token ids or pre-computed embeddings. */
class LlmGraphInputEmbd(val nEmbd: Long) : LlmGraphInput {
    /** I32 [nBatch] – token indices. */
    var tokens: GGMLTensor? = null
    /** F32 [nEmbd, nBatch] – pre-computed token embeddings. */
    var embd: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        // Set token ids into the tokens tensor
        val tok = ubatch.tokens
        val tokTensor = tokens
        if (tok != null && tokTensor != null) {
            tensorSetI32(tokTensor, tok, 0, ubatch.nTokens)
        }

        // Set pre-computed embeddings
        val emb = ubatch.embeddings
        val embTensor = embd
        if (emb != null && embTensor != null) {
            tensorSetF32(embTensor, emb, 0, ubatch.nTokens * nEmbd.toInt())
        }
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        var ok = true
        ok = ok && (params.ubatch.tokens == null || (tokens != null && tokens!!.ne[0] == params.ubatch.nTokens.toLong()))
        ok = ok && (params.ubatch.embeddings == null || (embd != null && embd!!.ne[1] == params.ubatch.nTokens.toLong()))
        return ok
    }
}

// -- positions --------------------------------------------------------------

/** Input: per-token position ids. */
class LlmGraphInputPos(val nPosPerEmbd: Int = 1) : LlmGraphInput {
    /** I32 [nBatch] – absolute position for each token. */
    var pos: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        val p = ubatch.pos ?: return
        val t = pos ?: return
        val nTokens = ubatch.nTokens

        if (ubatch.tokens != null && nPosPerEmbd == 4) {
            // M-RoPE with text tokens: convert 1D positions to 4D
            val posData = IntArray(nTokens * 4)
            for (i in 0 until nTokens) {
                posData[i] = p[i]
                posData[nTokens + i] = p[i]
                posData[2 * nTokens + i] = p[i]
                posData[3 * nTokens + i] = 0
            }
            tensorSetI32(t, posData, 0, posData.size)
        } else {
            tensorSetI32(t, p, 0, nTokens * nPosPerEmbd)
        }
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val p = pos ?: return false
        return p.ne[0] == params.ubatch.nTokens.toLong() * nPosPerEmbd
    }
}

// -- output ids -------------------------------------------------------------

/** Input: indices of tokens whose logits are actually needed. */
class LlmGraphInputOutIds(
    val hparams: LlamaHparams,
    val cparams: LlamaCParams,
    val nOutputs: Int,
) : LlmGraphInput {
    /** I32 [nOutputs] – indices into the batch that require output. */
    var outIds: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        val t = outIds ?: return
        val nTokens = ubatch.nTokens

        if (nOutputs == nTokens) {
            val ids = IntArray(nTokens) { it }
            tensorSetI32(t, ids, 0, nTokens)
            return
        }

        val output = ubatch.output ?: return
        val ids = mutableListOf<Int>()
        for (i in 0 until nTokens) {
            if (output[i]) ids.add(i)
        }
        tensorSetI32(t, ids.toIntArray(), 0, ids.size)
    }

    override fun canReuse(params: LlmGraphParams): Boolean = nOutputs == params.nOutputs
}

// -- mean pooling -----------------------------------------------------------

/** Input: mean-pooling weight matrix. */
class LlmGraphInputMean(val cparams: LlamaCParams) : LlmGraphInput {
    /** F32 [nBatch, nBatch] – mean-pooling weights. */
    var mean: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        if (!cparams.embeddings) return
        if (cparams.poolingType != LlamaPoolingType.MEAN &&
            cparams.poolingType != LlamaPoolingType.RANK) return

        val t = mean ?: return
        val nTokens = ubatch.nTokens
        val nSeqTokens = ubatch.nSeqTokens
        val nSeqsUnq = ubatch.nSeqsUnq

        // Zero-fill the weight matrix
        val data = FloatArray(nTokens * nSeqsUnq)

        // Sum up how many tokens belong to each unique sequence
        val sums = LongArray(nSeqsUnq)
        val seqIdArr = ubatch.seqId ?: return
        val nSeqIdArr = ubatch.nSeqId ?: return
        val seqIdxArr = ubatch.seqIdx ?: return

        var i = 0
        while (i < nTokens) {
            for (s in 0 until nSeqIdArr[i]) {
                val seqId = seqIdArr[i][s]
                val seqIdx = seqIdxArr[seqId]
                sums[seqIdx] += nSeqTokens.toLong()
            }
            i += nSeqTokens
        }

        // Compute reciprocals
        val div = FloatArray(nSeqsUnq) { s ->
            val sum = sums[s]
            if (sum > 0) 1.0f / sum.toFloat() else 0.0f
        }

        // Fill the weight matrix: each token row gets 1/count for its sequence
        i = 0
        while (i < nTokens) {
            for (s in 0 until nSeqIdArr[i]) {
                val seqId = seqIdArr[i][s]
                val seqIdx = seqIdxArr[seqId]
                for (j in 0 until nSeqTokens) {
                    data[seqIdx * nTokens + i + j] = div[seqIdx]
                }
            }
            i += nSeqTokens
        }

        tensorSetF32(t, data, 0, data.size)
    }
}

// -- CLS token selection ----------------------------------------------------

/** Input: CLS token indices for classification pooling. */
class LlmGraphInputCls(
    val cparams: LlamaCParams,
    val arch: LlamaModelArch,
) : LlmGraphInput {
    /** I32 [nBatch] – CLS token index per sequence. */
    var cls: GGMLTensor? = null

    override fun setInput(ubatch: LlamaUBatch) {
        val nTokens = ubatch.nTokens
        val nSeqsUnq = ubatch.nSeqsUnq

        if (!cparams.embeddings) return
        if (cparams.poolingType != LlamaPoolingType.CLS &&
            cparams.poolingType != LlamaPoolingType.RANK &&
            cparams.poolingType != LlamaPoolingType.LAST) return

        val t = cls ?: return
        val posArr = ubatch.pos ?: return
        val seqIdArr = ubatch.seqId ?: return
        val nSeqIdArr = ubatch.nSeqId ?: return
        val seqIdxArr = ubatch.seqIdx ?: return

        val targetPos = IntArray(nSeqsUnq) { -1 }
        val targetRow = IntArray(nSeqsUnq) { -1 }

        // Qwen3 reranking/embedding models use last token for RANK pooling
        val last = (cparams.poolingType == LlamaPoolingType.LAST ||
            (cparams.poolingType == LlamaPoolingType.RANK &&
                (arch == LlamaModelArch.QWEN3 || arch == LlamaModelArch.QWEN3VL)))

        for (i in 0 until nTokens) {
            val pos = posArr[i]

            for (s in 0 until nSeqIdArr[i]) {
                val seqId = seqIdArr[i][s]
                val seqIdx = seqIdxArr[seqId]

                if (targetPos[seqIdx] == -1 ||
                    (last && pos >= targetPos[seqIdx]) ||
                    (!last && pos < targetPos[seqIdx])) {
                    targetPos[seqIdx] = pos
                    targetRow[seqIdx] = i
                }
            }
        }

        // Write indices as I32 (uint32_t in C++)
        val data = IntArray(nSeqsUnq) { s ->
            if (targetRow[s] >= 0) targetRow[s] else 0
        }
        tensorSetI32(t, data, 0, nSeqsUnq)
    }
}

// -- recurrent state --------------------------------------------------------

/** Input: state-copy bookkeeping for recurrent (SSM / RWKV) models. */
class LlmGraphInputRs(
    var mctx: LlamaMemoryRecurrentContext?,
) : LlmGraphInput {
    /** I32 [nRs] – full state-copy source indices. */
    var sCopy: GGMLTensor? = null
    /** I32 [nSeqs] – state-copy indices for the main sequences. */
    var sCopyMain: GGMLTensor? = null
    /** I32 [nRs - nSeqs] – state-copy indices for extra sequences. */
    var sCopyExtra: GGMLTensor? = null
    var head: Int = 0
    var rsZ: Int = 0

    override fun setInput(ubatch: LlamaUBatch) {
        val ctx = mctx ?: return
        val nRs = ctx.getNRs()

        val t = sCopy ?: return
        val data = IntArray(nRs)
        // Assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (i in 0 until nRs) {
            data[i] = ctx.sCopy(i)
        }
        tensorSetI32(t, data, 0, nRs)
    }

    override fun canReuse(params: LlmGraphParams): Boolean {
        val ctx = params.mctx as? LlamaMemoryRecurrentContext ?: return false
        this.mctx = ctx

        val sc = sCopy ?: return false
        val scMain = sCopyMain ?: return false
        val scExtra = sCopyExtra ?: return false

        var res = true
        res = res && sc.ne[0] == ctx.getNRs().toLong()
        res = res && scMain.ne[0] == params.ubatch.nSeqs.toLong()
        res = res && scExtra.ne[0] == (ctx.getNRs() - params.ubatch.nSeqs).toLong()
        res = res && head == ctx.getHead()
        res = res && rsZ == ctx.getRsZ()
        return res
    }
}

// -- self-attention (no KV cache) -------------------------------------------

/** Input: KQ masks for self-attention without a persistent cache. */
class LlmGraphInputAttnNoCache(
    val hparams: LlamaHparams,
    val cparams: LlamaCParams,
) : LlmGraphInput {
    var selfKqMask: GGMLTensor? = null
    var selfKqMaskCnv: GGMLTensor? = null
    var selfKqMaskSwa: GGMLTensor? = null
    var selfKqMaskSwaCnv: GGMLTensor? = null

    /** Converted KQ mask ready for use by attention kernels. */
    fun getKqMask(): GGMLTensor? = selfKqMaskCnv

    /** Converted sliding-window KQ mask. */
    fun getKqMaskSwa(): GGMLTensor? = selfKqMaskSwaCnv

    override fun setInput(ubatch: LlamaUBatch) {
        val nKv = ubatch.nTokens
        val nTokens = ubatch.nTokens
        val posArr = ubatch.pos ?: return
        val seqIdArr = ubatch.seqId ?: return

        // Lambda to fill a mask tensor
        fun fillMask(tensor: GGMLTensor, nSwa: Int, swaType: LlamaSwaType) {
            val data = FloatArray(nTokens * nKv) { Float.NEGATIVE_INFINITY }

            for (i1 in 0 until nTokens) {
                val s1 = seqIdArr[i1][0]
                val p1 = posArr[i1]
                val idst = i1 * nKv

                for (i0 in 0 until nTokens) {
                    val s0 = seqIdArr[i0][0]
                    val p0 = posArr[i0]

                    // Mask different sequences
                    if (s0 != s1) continue
                    // Mask future tokens (causal)
                    if (cparams.causalAttn && p0 > p1) continue
                    // Apply SWA if any
                    if (LlamaHparams.isMaskedSwa(nSwa, swaType, p0, p1)) continue

                    data[idst + i0] = if (hparams.useAlibi) {
                        -(kotlin.math.abs(p0 - p1)).toFloat()
                    } else {
                        0.0f
                    }
                }
            }
            tensorSetF32(tensor, data, 0, data.size)
        }

        // Fill the main self-attention mask
        val mask = selfKqMask
        if (mask != null) {
            fillMask(mask, 0, LlamaSwaType.NONE)
        }

        // Fill the SWA mask if sliding-window attention is configured
        if (hparams.swaType != LlamaSwaType.NONE) {
            val swaMask = selfKqMaskSwa
            if (swaMask != null) {
                fillMask(swaMask, hparams.nSwa, hparams.swaType)
            }
        }
    }
}

// -- cross-attention --------------------------------------------------------

/** Input: KQ mask for encoder–decoder cross-attention. */
class LlmGraphInputAttnCross(val cross: LlamaCross?) : LlmGraphInput {
    var crossKqMask: GGMLTensor? = null
    var crossKqMaskCnv: GGMLTensor? = null

    /** Converted cross-attention KQ mask. */
    fun getKqMaskCross(): GGMLTensor? = crossKqMaskCnv

    override fun setInput(ubatch: LlamaUBatch) {
        val mask = crossKqMask ?: return
        val cr = cross ?: return

        val nEnc = mask.ne[0].toInt()
        val nTokens = ubatch.nTokens
        val seqIdArr = ubatch.seqId ?: return
        val nSeqIdArr = ubatch.nSeqId ?: return

        require(cr.seqIdsEnc.isNotEmpty()) { "llamaEncode must be called first" }
        require(!ubatch.equalSeqs) // LATER: use ubatch.nSeqs instead of failing

        val data = FloatArray(nTokens * nEnc)
        for (i in 0 until nTokens) {
            for (j in 0 until nEnc) {
                var f = Float.NEGATIVE_INFINITY

                for (s in 0 until nSeqIdArr[i]) {
                    val seqId = seqIdArr[i][s]
                    if (j < cr.seqIdsEnc.size && seqId in cr.seqIdsEnc[j]) {
                        f = 0.0f
                    }
                }

                data[i * nEnc + j] = f
            }
        }
        tensorSetF32(mask, data, 0, data.size)
    }
}

// ---------------------------------------------------------------------------
// LlmGraphQkv – packed Q/K/V triplet produced by build_qkv
// ---------------------------------------------------------------------------

/** Packed query, key, and value tensors ready for attention. */
data class LlmGraphQkv(
    /** [nEmbdHead, nHead, nTokens] */
    val q: GGMLTensor,
    /** [nEmbdHead, nHeadKv, nTokens] */
    val k: GGMLTensor,
    /** [nEmbdHead, nHeadKv, nTokens] */
    val v: GGMLTensor,
)

// ---------------------------------------------------------------------------
// LlmGraphResult – tensors produced by a single graph evaluation
// ---------------------------------------------------------------------------

/**
 * Collects every output tensor produced during graph construction and owns the
 * list of [LlmGraphInput]s that must be populated before evaluation.
 */
class LlmGraphResult(val maxNodes: Long = 8192) {
    var tInpTokens: GGMLTensor? = null
    /** [nEmbdInp, nTokens] */
    var tInpEmbd: GGMLTensor? = null
    var tLogits: GGMLTensor? = null
    var tEmbd: GGMLTensor? = null
    var tEmbdPooled: GGMLTensor? = null

    val tSampledLogits: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()
    val tCandidates: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()
    val tSampled: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()
    val tSampledProbs: MutableMap<LlamaSeqId, GGMLTensor> = mutableMapOf()

    val inputs: MutableList<LlmGraphInput> = mutableListOf()

    // port-lint: source llama-graph.h  llm_graph_result::params
    var params: LlmGraphParams = LlmGraphParams()

    /** The underlying ggml context for tensor allocation. */
    var ctxCompute: GGMLContext? = null

    /** The compute graph. */
    var gf: GGMLCGraph? = null

    // port-lint: source llama-graph.cpp  llm_graph_result::debug
    // C++ reads LLAMA_GRAPH_RESULT_DEBUG from env; cross-platform Kotlin has no getenv.
    // Overridable via constructor param or direct assignment for testing.
    var debug: Int = 0

    init {
        reset()
    }

    // port-lint: source llama-graph.cpp  llm_graph_result::get_max_nodes
    fun getMaxNodes(): Long = maxNodes

    // port-lint: source llama-graph.cpp  llm_graph_result::set_params
    fun setParams(params: LlmGraphParams) {
        this.params = params
    }

    fun addInput(input: LlmGraphInput): LlmGraphInput {
        inputs.add(input)
        return input
    }

    fun <T : LlmGraphInput> addInputTyped(input: T): T {
        inputs.add(input)
        return input
    }

    // port-lint: source llama-graph.cpp  llm_graph_result::set_inputs
    fun setInputs(ubatch: LlamaUBatch) {
        for (input in inputs) {
            input.setInput(ubatch)
        }
    }

    // port-lint: source llama-graph.cpp  llm_graph_result::set_outputs
    fun setOutputs() {
        tLogits?.let { ggmlSetOutput(it) }
        tEmbd?.let { ggmlSetOutput(it) }
        tEmbdPooled?.let { ggmlSetOutput(it) }
        for ((_, t) in tSampled) ggmlSetOutput(t)
        for ((_, t) in tSampledProbs) ggmlSetOutput(t)
        for ((_, t) in tSampledLogits) ggmlSetOutput(t)
        for ((_, t) in tCandidates) ggmlSetOutput(t)
    }

    // port-lint: source llama-graph.cpp  llm_graph_result::can_reuse
    fun canReuse(params: LlmGraphParams): Boolean {
        if (!this.params.allowReuse(params)) {
            return false
        }
        var res = true
        for (input in inputs) {
            val cur = input.canReuse(params)
            res = res && cur
        }
        return res
    }

    // port-lint: source llama-graph.cpp  llm_graph_result::reset
    fun reset() {
        tInpTokens = null
        tInpEmbd = null
        tLogits = null
        tEmbd = null
        tEmbdPooled = null
        tSampledLogits.clear()
        tCandidates.clear()
        tSampled.clear()
        tSampledProbs.clear()
        params = LlmGraphParams()
        inputs.clear()
    }

    // port-lint: source llama-graph.h  llm_graph_result accessors
    fun getInpTokens(): GGMLTensor? = tInpTokens
    fun getLogits(): GGMLTensor? = tLogits
    fun getEmbd(): GGMLTensor? = tEmbd
    fun getEmbdPooled(): GGMLTensor? = tEmbdPooled

    fun getCtx(): GGMLContext? = ctxCompute
    fun getGf(): GGMLCGraph? = gf
}

// ---------------------------------------------------------------------------
// LlmGraphParams – immutable snapshot of everything needed to build a graph
// ---------------------------------------------------------------------------

/** Parameters that fully describe a single graph-build invocation. */
data class LlmGraphParams(
    val arch: LlamaModelArch = LlamaModelArch.UNKNOWN,
    val hparams: LlamaHparams = LlamaHparams(),
    val cparams: LlamaCParams = LlamaCParams(),
    val ubatch: LlamaUBatch = LlamaUBatch(),
    val gtype: LlmGraphType = LlmGraphType.DEFAULT,
    val nOutputs: Int = 0,
    val res: LlmGraphResult? = null,

    // Adapter references
    val cvec: LlamaAdapterCvec? = null,
    val loras: LlamaAdapterLoras? = null,
    /** Memory context for KV cache / recurrent state. */
    val mctx: LlamaMemoryContext? = null,
    val cross: LlamaCross? = null,

    val samplers: Map<LlamaSeqId, LlamaSampler> = emptyMap(),

    // Graph callback for tensor naming / offloading
    val cb: LlmGraphCb? = null,
) {
    companion object {
        // port-lint: source llama-graph.h  llm_graph_params::samplers_equal
        fun samplersEqual(
            lhs: Map<LlamaSeqId, LlamaSampler>,
            rhs: Map<LlamaSeqId, LlamaSampler>,
        ): Boolean {
            if (lhs.size != rhs.size) return false
            for ((seqId, sampler) in lhs) {
                val other = rhs[seqId] ?: return false
                if (other !== sampler) return false
            }
            return true
        }
    }

    // port-lint: source llama-graph.h  llm_graph_params::allow_reuse
    fun allowReuse(other: LlmGraphParams): Boolean {
        // first check the ubatch
        var canReuseUbatch =
            ubatch.equalSeqs == other.ubatch.equalSeqs &&
            ubatch.nTokens == other.ubatch.nTokens &&
            ubatch.nSeqTokens == other.ubatch.nSeqTokens &&
            ubatch.nSeqs == other.ubatch.nSeqs &&
            ubatch.nSeqsUnq == other.ubatch.nSeqsUnq &&
            (
                (ubatch.tokens == null && other.ubatch.tokens == null) ||
                (ubatch.embeddings == null && other.ubatch.embeddings == null)
            )

        if (canReuseUbatch && ubatch.equalSeqs) {
            val seqUnq = ubatch.seqIdUnq
            val otherSeqUnq = other.ubatch.seqIdUnq
            if (seqUnq != null && otherSeqUnq != null) {
                for (s in 0 until ubatch.nSeqsUnq) {
                    canReuseUbatch = canReuseUbatch && (seqUnq[s] == otherSeqUnq[s])
                }
            } else {
                canReuseUbatch = false
            }
        }

        if (!canReuseUbatch) return false
        if (nOutputs != other.nOutputs) return false
        if (!samplersEqual(samplers, other.samplers)) return false

        if (samplers.isNotEmpty()) {
            val output = ubatch.output
            val otherOutput = other.ubatch.output
            val seqId = ubatch.seqId
            val otherSeqId = other.ubatch.seqId
            if (output != null && otherOutput != null && seqId != null && otherSeqId != null) {
                for (i in 0 until ubatch.nTokens) {
                    if (output[i] != otherOutput[i] ||
                        seqId[i][0] != otherSeqId[i][0]) {
                        return false
                    }
                }
            } else {
                return false
            }
        }

        return cparams.embeddings == other.cparams.embeddings &&
            cparams.causalAttn == other.cparams.causalAttn &&
            arch == other.arch &&
            gtype == other.gtype &&
            cvec == other.cvec &&
            loras == other.loras &&
            cross == other.cross
    }
}

// ---------------------------------------------------------------------------
// LlmGraphContext – base builder class for model-specific graph construction
// ---------------------------------------------------------------------------

/**
 * Base class for building inference graphs.
 *
 * Model-specific subclasses (e.g. `LlmBuildLlama`) override selected builder
 * methods to wire architecture-specific attention, FFN, and normalisation
 * patterns.  Shared utilities (LoRA application, KQ-mask construction, …) are
 * implemented once here.
 *
 * Port of `llm_graph_context` from `llama-graph.h` / `llama-graph.cpp`.
 */
open class LlmGraphContext(val params: LlmGraphParams) {

    // -- fields from params (mirrors C++ constructor initialiser list) -----

    val arch: LlamaModelArch get() = params.arch
    val hparams: LlamaHparams get() = params.hparams
    val cparams: LlamaCParams get() = params.cparams
    val ubatch: LlamaUBatch get() = params.ubatch

    val nEmbd: Long get() = hparams.nEmbd.toLong()
    val nLayer: Long get() = hparams.nLayer.toLong()
    val nRot: Long get() = hparams.nRot().toLong()
    val nCtx: Long get() = cparams.nCtx.toLong()
    val nHead: Long get() = hparams.nHead().toLong()
    val nHeadKv: Long get() = hparams.nHeadKv().toLong()
    val nEmbdHeadK: Long get() = hparams.nEmbdHeadK().toLong()
    val nEmbdKGqa: Long get() = hparams.nEmbdKGqa().toLong()
    val nEmbdHeadV: Long get() = hparams.nEmbdHeadV().toLong()
    val nEmbdVGqa: Long get() = hparams.nEmbdVGqa().toLong()
    val nExpert: Long get() = hparams.nExpert.toLong()
    val nExpertUsed: Long
        get() = if (cparams.warmup) hparams.nExpert.toLong() else hparams.nExpertUsed.toLong()

    val freqBase: Float get() = cparams.ropeFreqBase
    val freqScale: Float get() = cparams.ropeFreqScale
    val extFactor: Float get() = cparams.yarnExtFactor
    val attnFactor: Float get() = cparams.yarnAttnFactor
    val betaFast: Float get() = cparams.yarnBetaFast
    val betaSlow: Float get() = cparams.yarnBetaSlow
    val normEps: Float get() = hparams.fNormEps
    val normRmsEps: Float get() = hparams.fNormRmsEps

    val nTokens: Long get() = ubatch.nTokens.toLong()
    val nOutputs: Long get() = params.nOutputs.toLong()
    val nCtxOrig: Int get() = cparams.nCtxOrigYarn

    val poolingType: LlamaPoolingType get() = cparams.poolingType
    val ropeType: LlamaRopeType get() = hparams.ropeType

    val cvec: LlamaAdapterCvec? get() = params.cvec
    val loras: LlamaAdapterLoras? get() = params.loras
    val cross: LlamaCross? get() = params.cross
    val cbFunc: LlmGraphCb? get() = params.cb

    var res: LlmGraphResult? = params.res
    var ctx0: GGMLContext? = res?.getCtx()
    var gf: GGMLCGraph? = res?.getGf()

    // -- tensor callback helper ---------------------------------------------

    /** Name a tensor and invoke the user callback. */
    fun cb(cur: GGMLTensor, name: String, il: Int) {
        cbFunc?.invoke(ubatch, cur, name, il)
    }

    // -- Common builders ---------------------------------------------------------

    /** Apply control vector to the hidden state at layer [il]. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildCvec(cur: GGMLTensor, il: Int): GGMLTensor {
        val cv = cvec ?: return cur
        return cv.applyTo(cur, il)
    }

    /**
     * Matrix-multiply [cur] by weight [w] with optional LoRA deltas and
     * per-tensor scale [wS].
     *
     * Port of `llm_graph_context::build_lora_mm`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildLoraMm(w: GGMLTensor, cur: GGMLTensor, wS: GGMLTensor? = null): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var result = ggmlMulMat(c, w, cur)

        loras?.let { loraMap ->
            for ((adapter, adapterScale) in loraMap) {
                val lw = adapter.getWeight(w) ?: continue
                val scale = lw.getScale(adapter.alpha, adapterScale)

                var abCur = ggmlMulMat(c, lw.b!!, ggmlMulMat(c, lw.a!!, cur))
                abCur = ggmlScale(c, abCur, scale)
                result = ggmlAdd(c, result, abCur)
            }
        }

        if (wS != null) {
            result = ggmlMul(c, result, wS)
        }
        return result
    }

    /**
     * Expert-routed LoRA matrix-multiply.
     *
     * Port of `llm_graph_context::build_lora_mm_id`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildLoraMmId(w: GGMLTensor, cur: GGMLTensor, ids: GGMLTensor): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var result = ggmlMulMatId(c, w, cur, ids)

        loras?.let { loraMap ->
            for ((adapter, adapterScale) in loraMap) {
                val lw = adapter.getWeight(w) ?: continue
                val alpha = adapter.alpha
                val rank = lw.b!!.ne[0].toFloat()
                val scale = if (alpha != 0f) adapterScale * alpha / rank else adapterScale

                var abCur = ggmlMulMatId(c, lw.b!!, ggmlMulMatId(c, lw.a!!, cur, ids), ids)
                abCur = ggmlScale(c, abCur, scale)
                result = ggmlAdd(c, result, abCur)
            }
        }
        return result
    }

    /**
     * Apply normalisation of the given [type] with optional weight [mw] and bias [mb].
     *
     * Port of `llm_graph_context::build_norm`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildNorm(
        cur: GGMLTensor,
        mw: GGMLTensor?,
        mb: GGMLTensor?,
        type: LlmNormType,
        il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var out = cur
        when (type) {
            LlmNormType.NORM -> {
                out = ggmlNorm(c, out, hparams.fNormEps)
            }
            LlmNormType.RMS -> {
                out = ggmlRmsNorm(c, out, hparams.fNormRmsEps)
            }
            LlmNormType.GROUP -> {
                out = ggmlReshape3d(c, out, out.ne[0], 1, out.ne[1])
                out = ggmlGroupNorm(c, out, hparams.nNormGroups, hparams.fNormGroupEps)
                out = ggmlReshape2d(c, out, out.ne[0], out.ne[2])
            }
        }

        if (mw != null || mb != null) {
            cb(out, "norm", il)
        }

        if (mw != null) {
            out = ggmlMul(c, out, mw)
            if (mb != null) {
                cb(out, "norm_w", il)
            }
        }

        if (mb != null) {
            out = ggmlAdd(c, out, mb)
        }

        return out
    }

    /**
     * Compute Q, K, V projections with optional bias and reshape.
     * Supports both fused wqkv and separate wq/wk/wv paths.
     *
     * Port of `llm_graph_context::build_qkv`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildQkv(
        layer: LlamaModelLayer,
        cur: GGMLTensor,
        nEmbdHead: Long,
        nHead: Long,
        nHeadKv: Long,
        il: Int,
    ): LlmGraphQkv {
        val c = ctx0 ?: error("ctx0 not initialised")
        val nEmbdQ = nEmbdHead * nHead
        val nEmbdKv = nEmbdHead * nHeadKv

        var qCur: GGMLTensor
        var kCur: GGMLTensor
        var vCur: GGMLTensor

        if (layer.wqkv != null) {
            // fused QKV path
            var qkv = buildLoraMm(layer.wqkv!!, cur)
            cb(qkv, "wqkv", il)
            if (layer.bqkv != null) {
                qkv = ggmlAdd(c, qkv, layer.bqkv!!)
                cb(qkv, "bqkv", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                qkv = ggmlClamp(c, qkv, -hparams.fClampKqv, hparams.fClampKqv)
                cb(qkv, "wqkv_clamped", il)
            }
            val rowSize = ggmlRowSize(qkv.type, nEmbdHead)
            qCur = ggmlView3d(c, qkv, nEmbdHead, nHead, nTokens, rowSize, qkv.nb[1], 0u)
            kCur = ggmlView3d(c, qkv, nEmbdHead, nHeadKv, nTokens,
                rowSize, qkv.nb[1], ggmlRowSize(qkv.type, nEmbdQ))
            vCur = ggmlView3d(c, qkv, nEmbdHead, nHeadKv, nTokens,
                rowSize, qkv.nb[1], ggmlRowSize(qkv.type, nEmbdQ + nEmbdKv))
        } else {
            // separate Q/K/V path
            qCur = buildLoraMm(layer.wq!!, cur)
            cb(qCur, "Qcur", il)
            if (layer.bq != null) {
                qCur = ggmlAdd(c, qCur, layer.bq!!)
                cb(qCur, "Qcur", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                qCur = ggmlClamp(c, qCur, -hparams.fClampKqv, hparams.fClampKqv)
                cb(qCur, "Qcur_clamped", il)
            }

            kCur = buildLoraMm(layer.wk!!, cur)
            cb(kCur, "Kcur", il)
            if (layer.bk != null) {
                kCur = ggmlAdd(c, kCur, layer.bk!!)
                cb(kCur, "Kcur", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                kCur = ggmlClamp(c, kCur, -hparams.fClampKqv, hparams.fClampKqv)
                cb(kCur, "Kcur_clamped", il)
            }

            vCur = buildLoraMm(layer.wv!!, cur)
            cb(vCur, "Vcur", il)
            if (layer.bv != null) {
                vCur = ggmlAdd(c, vCur, layer.bv!!)
                cb(vCur, "Vcur", il)
            }
            if (hparams.fClampKqv > 0.0f) {
                vCur = ggmlClamp(c, vCur, -hparams.fClampKqv, hparams.fClampKqv)
                cb(vCur, "Vcur_clamped", il)
            }

            qCur = ggmlReshape3d(c, qCur, nEmbdHead, nHead, nTokens)
            kCur = ggmlReshape3d(c, kCur, nEmbdHead, nHeadKv, nTokens)
            vCur = ggmlReshape3d(c, vCur, nEmbdHead, nHeadKv, nTokens)
        }

        cb(qCur, "Qcur", il)
        cb(kCur, "Kcur", il)
        cb(vCur, "Vcur", il)

        return LlmGraphQkv(qCur, kCur, vCur)
    }

    /**
     * Construct a feed-forward network sub-graph at layer [il].
     *
     * Port of `llm_graph_context::build_ffn`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildFfn(
        cur: GGMLTensor,
        up: GGMLTensor?, upB: GGMLTensor?, upS: GGMLTensor?,
        gate: GGMLTensor?, gateB: GGMLTensor?, gateS: GGMLTensor?,
        down: GGMLTensor?, downB: GGMLTensor?, downS: GGMLTensor?,
        actScales: GGMLTensor?,
        typeOp: LlmFfnOpType,
        typeGate: LlmFfnGateType,
        il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        var typeGateMut = typeGate

        var tmp = if (up != null) buildLoraMm(up, cur) else cur
        cb(tmp, "ffn_up", il)

        if (upB != null) { tmp = ggmlAdd(c, tmp, upB); cb(tmp, "ffn_up_b", il) }
        if (upS != null) { tmp = ggmlMul(c, tmp, upS); cb(tmp, "ffn_up_s", il) }

        var out: GGMLTensor
        if (gate != null) {
            out = when (typeGateMut) {
                LlmFfnGateType.SEQ -> buildLoraMm(gate, tmp)
                LlmFfnGateType.PAR -> buildLoraMm(gate, cur)
            }
            cb(out, "ffn_gate", il)

            if (gateB != null) { out = ggmlAdd(c, out, gateB); cb(out, "ffn_gate_b", il) }
            if (gateS != null) { out = ggmlMul(c, out, gateS); cb(out, "ffn_gate_s", il) }
        } else {
            out = tmp
        }

        when (typeOp) {
            LlmFfnOpType.SILU -> {
                if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
                    out = ggmlSwigluSplit(c, out, tmp)
                    cb(out, "ffn_swiglu", il)
                    typeGateMut = LlmFfnGateType.SEQ
                } else {
                    out = ggmlSilu(c, out)
                    cb(out, "ffn_silu", il)
                }
            }
            LlmFfnOpType.GELU -> {
                if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
                    out = ggmlGegluSplit(c, out, tmp)
                    cb(out, "ffn_geglu", il)
                    typeGateMut = LlmFfnGateType.SEQ
                } else {
                    out = ggmlGelu(c, out)
                    cb(out, "ffn_gelu", il)
                    if (actScales != null) {
                        out = ggmlDiv(c, out, actScales)
                        cb(out, "ffn_act", il)
                    }
                }
            }
            LlmFfnOpType.RELU -> {
                if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
                    out = ggmlRegluSplit(c, out, tmp)
                    cb(out, "ffn_reglu", il)
                    typeGateMut = LlmFfnGateType.SEQ
                } else {
                    out = ggmlRelu(c, out)
                    cb(out, "ffn_relu", il)
                }
            }
            LlmFfnOpType.RELU_SQR -> {
                out = ggmlRelu(c, out)
                cb(out, "ffn_relu", il)
                out = ggmlSqr(c, out)
                cb(out, "ffn_sqr(relu)", il)
            }
            LlmFfnOpType.SWIGLU -> {
                out = ggmlSwiglu(c, out)
                cb(out, "ffn_swiglu", il)
            }
            LlmFfnOpType.GEGLU -> {
                out = ggmlGeglu(c, out)
                cb(out, "ffn_geglu", il)
            }
            LlmFfnOpType.REGLU -> {
                out = ggmlReglu(c, out)
                cb(out, "ffn_reglu", il)
            }
            else -> error("Unsupported FFN op type: $typeOp")
        }

        if (gate != null && typeGateMut == LlmFfnGateType.PAR) {
            out = ggmlMul(c, out, tmp)
            cb(out, "ffn_gate_par", il)
        }

        if (down != null) {
            out = buildLoraMm(down, out)
        }
        if (downB != null) {
            cb(out, "ffn_down", il)
            out = ggmlAdd(c, out, downB)
        }
        if (downS != null) {
            out = ggmlMul(c, out, downS)
            cb(out, "ffn_down_s", il)
        }

        return out
    }

    /**
     * Construct a Mixture-of-Experts FFN sub-graph at layer [il].
     *
     * Port of `llm_graph_context::build_moe_ffn` (no-bias overload delegates
     * to the full overload with null bias tensors).
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildMoeFfn(
        cur: GGMLTensor,
        gateInp: GGMLTensor,
        upExps: GGMLTensor,
        gateExps: GGMLTensor?,
        downExps: GGMLTensor,
        expProbsB: GGMLTensor?,
        nExpert: Long,
        nExpertUsed: Long,
        typeOp: LlmFfnOpType,
        normW: Boolean,
        wScale: Float,
        gatingOp: LlamaExpertGatingFuncType,
        il: Int,
    ): GGMLTensor {
        return buildMoeFfnFull(
            cur, gateInp, null, upExps, null, gateExps, null, downExps, null,
            expProbsB, nExpert, nExpertUsed, typeOp, normW, wScale, gatingOp, il,
        )
    }

    /**
     * Full MoE FFN with optional bias tensors on every expert weight.
     *
     * Port of the second `llm_graph_context::build_moe_ffn` overload.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildMoeFfnFull(
        cur: GGMLTensor,
        gateInp: GGMLTensor,
        gateInpB: GGMLTensor?,
        upExps: GGMLTensor,
        upExpsB: GGMLTensor?,
        gateExps: GGMLTensor?,
        gateExpsB: GGMLTensor?,
        downExps: GGMLTensor,
        downExpsB: GGMLTensor?,
        expProbsB: GGMLTensor?,
        nExpert: Long,
        nExpertUsed: Long,
        typeOp: LlmFfnOpType,
        normW: Boolean,
        wScale: Float,
        gatingOp: LlamaExpertGatingFuncType,
        il: Int,
        probsIn: GGMLTensor? = null,
        gateUpExps: GGMLTensor? = null,
        gateUpExpsB: GGMLTensor? = null,
        upExpsS: GGMLTensor? = null,
        gateExpsS: GGMLTensor? = null,
        downExpsS: GGMLTensor? = null,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        val nEmbdCur = cur.ne[0]
        val nTokensCur = cur.ne[1]

        // -- gating logits --------------------------------------------------
        var logits = if (probsIn == null) {
            val l = buildLoraMm(gateInp, cur)
            cb(l, "ffn_moe_logits", il)
            l
        } else {
            probsIn
        }

        if (gateInpB != null) {
            logits = ggmlAdd(c, logits, gateInpB)
            cb(logits, "ffn_moe_logits_biased", il)
        }

        var probs: GGMLTensor = when (gatingOp) {
            LlamaExpertGatingFuncType.SOFTMAX -> ggmlSoftMax(c, logits)
            LlamaExpertGatingFuncType.SIGMOID -> ggmlSigmoid(c, logits)
            LlamaExpertGatingFuncType.SOFTMAX_WEIGHT -> logits
            else -> error("Unsupported gating op: $gatingOp")
        }
        cb(probs, "ffn_moe_probs", il)

        var selectionProbs = probs
        if (expProbsB != null) {
            selectionProbs = ggmlAdd(c, probs, expProbsB)
            cb(selectionProbs, "ffn_moe_probs_biased", il)
        }

        if (arch == LlamaModelArch.LLAMA4) {
            selectionProbs = logits
        }

        // -- top-k expert selection -----------------------------------------
        val selectedExperts = ggmlArgsortTopK(c, selectionProbs, nExpertUsed.toInt())
        cb(selectedExperts, "ffn_moe_topk", il)

        probs = ggmlReshape3d(c, probs, 1, nExpert, nTokensCur)

        var weights = ggmlGetRows(c, probs, selectedExperts) // [1, nExpertUsed, nTokens]
        cb(weights, "ffn_moe_weights", il)

        if (gatingOp == LlamaExpertGatingFuncType.SOFTMAX_WEIGHT) {
            weights = ggmlReshape2d(c, weights, nExpertUsed, nTokensCur)
            weights = ggmlSoftMax(c, weights)
            weights = ggmlReshape3d(c, weights, 1, nExpertUsed, nTokensCur)
            cb(weights, "ffn_moe_weights_softmax", il)
        }

        if (normW) {
            weights = ggmlReshape2d(c, weights, nExpertUsed, nTokensCur)
            var weightsSum = ggmlSumRows(c, weights)
            cb(weightsSum, "ffn_moe_weights_sum", il)
            weightsSum = ggmlClamp(c, weightsSum, 6.103515625e-5f, Float.POSITIVE_INFINITY)
            cb(weightsSum, "ffn_moe_weights_sum_clamped", il)
            weights = ggmlDiv(c, weights, weightsSum)
            cb(weights, "ffn_moe_weights_norm", il)
            weights = ggmlReshape3d(c, weights, 1, nExpertUsed, nTokensCur)
        }
        if (wScale != 0.0f && wScale != 1.0f) {
            weights = ggmlScale(c, weights, wScale)
            cb(weights, "ffn_moe_weights_scaled", il)
        }

        ggmlBuildForwardExpand(graph, weights)

        var moeIn = ggmlReshape3d(c, cur, nEmbdCur, 1, nTokensCur)

        // -- up / gate projections ------------------------------------------
        var up: GGMLTensor
        if (gateUpExps != null) {
            var gateUp = buildLoraMmId(gateUpExps, moeIn, selectedExperts)
            cb(gateUp, "ffn_moe_gate_up", il)
            if (gateUpExpsB != null) {
                gateUp = ggmlAddId(c, gateUp, gateUpExpsB, selectedExperts)
                cb(gateUp, "ffn_moe_gate_up_biased", il)
            }
            val nFf = gateUp.ne[0] / 2
            moeIn = ggmlView3d(c, gateUp, nFf, gateUp.ne[1], gateUp.ne[2],
                gateUp.nb[1], gateUp.nb[2], 0u)
            cb(moeIn, "ffn_moe_gate", il)
            up = ggmlView3d(c, gateUp, nFf, gateUp.ne[1], gateUp.ne[2],
                gateUp.nb[1], gateUp.nb[2], (nFf.toULong() * gateUp.nb[0]))
            cb(up, "ffn_moe_up", il)
        } else {
            up = buildLoraMmId(upExps, moeIn, selectedExperts)
            cb(up, "ffn_moe_up", il)
            if (upExpsB != null) {
                up = ggmlAddId(c, up, upExpsB, selectedExperts)
                cb(up, "ffn_moe_up_biased", il)
            }
            if (gateExps != null) {
                moeIn = buildLoraMmId(gateExps, moeIn, selectedExperts)
                cb(moeIn, "ffn_moe_gate", il)
            } else {
                moeIn = up
            }
            if (gateExpsB != null) {
                moeIn = ggmlAddId(c, moeIn, gateExpsB, selectedExperts)
                cb(moeIn, "ffn_moe_gate_biased", il)
            }
        }

        val hasGate = gateExps != null || gateUpExps != null

        // -- activation -----------------------------------------------------
        var activated: GGMLTensor = when (typeOp) {
            LlmFfnOpType.SILU -> {
                if (hasGate) {
                    val r = ggmlSwigluSplit(c, moeIn, up); cb(r, "ffn_moe_swiglu", il); r
                } else {
                    val r = ggmlSilu(c, moeIn); cb(r, "ffn_moe_silu", il); r
                }
            }
            LlmFfnOpType.GELU -> {
                if (hasGate) {
                    val r = ggmlGegluSplit(c, moeIn, up); cb(r, "ffn_moe_geglu", il); r
                } else {
                    val r = ggmlGelu(c, moeIn); cb(r, "ffn_moe_gelu", il); r
                }
            }
            LlmFfnOpType.SWIGLU_OAI_MOE -> {
                val r = ggmlSwigluOai(c, moeIn, up, 1.702f, 7.0f)
                cb(r, "ffn_moe_swiglu_oai", il); r
            }
            LlmFfnOpType.RELU -> {
                if (hasGate) {
                    val r = ggmlRegluSplit(c, moeIn, up); cb(r, "ffn_moe_reglu", il); r
                } else {
                    val r = ggmlRelu(c, moeIn); cb(r, "ffn_moe_relu", il); r
                }
            }
            LlmFfnOpType.RELU_SQR -> {
                var r = ggmlRelu(c, moeIn); r = ggmlSqr(c, r)
                cb(r, "ffn_moe_relu_sqr", il); r
            }
            else -> error("Unsupported MoE FFN op type: $typeOp")
        }

        // -- down projection ------------------------------------------------
        var experts = buildLoraMmId(downExps, activated, selectedExperts)
        cb(experts, "ffn_moe_down", il)

        if (downExpsB != null) {
            experts = ggmlAddId(c, experts, downExpsB, selectedExperts)
            cb(experts, "ffn_moe_down_biased", il)
        }

        experts = ggmlMul(c, experts, weights)
        cb(experts, "ffn_moe_weighted", il)

        ggmlBuildForwardExpand(graph, experts)

        // aggregate expert views
        val nExpertUsedInt = hparams.nExpertUsed
        val curExperts = Array<GGMLTensor?>(nExpertUsedInt) { null }
        for (i in 0 until nExpertUsedInt) {
            curExperts[i] = ggmlView2d(c, experts, nEmbdCur, nTokensCur,
                experts.nb[2], (i.toLong().toULong() * experts.nb[1]))
            ggmlBuildForwardExpand(graph, curExperts[i]!!)
        }

        var moeOut = curExperts[0]!!
        for (i in 1 until nExpertUsedInt) {
            moeOut = ggmlAdd(c, moeOut, curExperts[i]!!)
            ggmlBuildForwardExpand(graph, moeOut)
        }

        if (nExpertUsedInt == 1) {
            moeOut = ggmlCont(c, moeOut)
        }

        cb(moeOut, "ffn_moe_out", il)
        return moeOut
    }

    // =======================================================================
    // Input builders
    // =======================================================================

    /**
     * Create input tensors for token-id / embedding injection.
     *
     * Port of `llm_graph_context::build_inp_embd`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpEmbd(tokEmbd: GGMLTensor): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val nEmbdInp = hparams.nEmbdInp().toLong()

        val inp = LlmGraphInputEmbd(nEmbdInp)
        inp.tokens = ggmlNewTensor1d(c, GGMLType.I32, nTokens)
        cb(inp.tokens!!, "inp_tokens", -1)
        ggmlSetInput(inp.tokens!!)
        r.tInpTokens = inp.tokens

        inp.embd = ggmlNewTensor2d(c, GGMLType.F32, nEmbdInp, nTokens)
        cb(inp.embd!!, "inp_embd", -1)
        ggmlSetInput(inp.embd!!)

        // Token embeddings path: look up rows in tok_embd
        var tokPath = ggmlGetRows(c, tokEmbd, inp.tokens!!)

        // For Granite architecture
        if (hparams.fEmbeddingScale != 0.0f) {
            tokPath = ggmlScale(c, tokPath, hparams.fEmbeddingScale)
        }

        cb(tokPath, "embd", -1)

        r.tInpEmbd = tokPath
        r.addInput(inp)

        val graph = gf ?: error("gf not initialised")
        ggmlBuildForwardExpand(graph, tokPath)

        return tokPath
    }

    /** Create the position-id input tensor. Port of `build_inp_pos`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpPos(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputPos(hparams.nPosPerEmbd())
        inp.pos = ggmlNewTensor1d(c, GGMLType.I32, nTokens * hparams.nPosPerEmbd())
        ggmlSetInput(inp.pos!!)
        r.addInput(inp)
        return inp.pos!!
    }

    /** Create the output-id selection tensor. Port of `build_inp_out_ids`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpOutIds(): GGMLTensor? {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputOutIds(hparams, cparams, nOutputs.toInt())
        inp.outIds = ggmlNewTensor1d(c, GGMLType.I32, nOutputs)
        ggmlSetInput(inp.outIds!!)
        r.addInput(inp)
        return inp.outIds
    }

    // =======================================================================
    // Attention builders
    // =======================================================================

    /**
     * Core multi-head attention sub-graph.
     *
     * Port of `llm_graph_context::build_attn_mha`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttnMha(
        q: GGMLTensor, k: GGMLTensor, v: GGMLTensor,
        kqB: GGMLTensor?, kqMask: GGMLTensor?, sinks: GGMLTensor?,
        vMla: GGMLTensor?, kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        val nStream = k.ne[3]

        val q4 = ggmlView4d(c, q, q.ne[0], q.ne[1], q.ne[2] / nStream, nStream,
            q.nb[1], q.nb[2], q.nb[3] / nStream.toULong(), 0u)
        val qPerm = ggmlPermute(c, q4, 0, 2, 1, 3)
        val kPerm = ggmlPermute(c, k, 0, 2, 1, 3)
        val vPerm = ggmlPermute(c, v, 0, 2, 1, 3)

        val useFlashAttn = cparams.flashAttn && kqB == null
        val cur: GGMLTensor

        if (useFlashAttn) {
            var kFa = kPerm
            var vFa = vPerm
            if (kFa.type == GGMLType.F32) kFa = ggmlCast(c, kFa, GGMLType.F16)
            if (vFa.type == GGMLType.F32) vFa = ggmlCast(c, vFa, GGMLType.F16)

            var fa = ggmlFlashAttnExt(c, qPerm, kFa, vFa, kqMask, kqScale,
                hparams.fMaxAlibiBias,
                if (hparams.attnSoftCap) hparams.fAttnLogitSoftcapping else 0.0f)
            cb(fa, "fattn", il)
            ggmlFlashAttnExtAddSinks(fa, sinks)
            ggmlFlashAttnExtSetPrec(fa, GGMLPrec.F32)

            if (vMla != null) {
                fa = ggmlPermute(c, fa, 0, 2, 1, 3)
                fa = ggmlMulMat(c, vMla, fa)
                cb(fa, "fattn_mla", il)
                fa = ggmlPermute(c, fa, 0, 2, 1, 3)
                fa = ggmlCont(c, fa)
            }

            cur = ggmlReshape2d(c, fa, fa.ne[0] * fa.ne[1], fa.ne[2] * fa.ne[3])
        } else {
            var kq = ggmlMulMat(c, kPerm, qPerm)
            cb(kq, "kq", il)
            ggmlMulMatSetPrec(kq, GGMLPrec.F32)

            if (hparams.attnSoftCap) {
                kq = ggmlScale(c, kq, 1.0f / hparams.fAttnLogitSoftcapping)
                cb(kq, "kq_scaled_1", il)
                kq = ggmlTanh(c, kq)
                cb(kq, "kq_tanh", il)
                kq = ggmlScale(c, kq, hparams.fAttnLogitSoftcapping)
                cb(kq, "kq_scaled_2", il)
            }

            if (kqB != null) {
                kq = ggmlAdd(c, kq, kqB)
                cb(kq, "kq_plus_kq_b", il)
            }

            kq = ggmlSoftMaxExt(c, kq, kqMask, kqScale, hparams.fMaxAlibiBias)
            ggmlSoftMaxAddSinks(kq, sinks)
            cb(kq, "kq_soft_max", il)

            var vT = ggmlCont(c, ggmlTranspose(c, vPerm))
            cb(vT, "v_cont", il)

            var kqv = ggmlMulMat(c, vT, kq)
            cb(kqv, "kqv", il)

            if (vMla != null) {
                kqv = ggmlMulMat(c, vMla, kqv)
                cb(kqv, "kqv_mla", il)
            }

            cur = ggmlCont2d(c, ggmlPermute(c, kqv, 0, 2, 1, 3),
                kqv.ne[0] * kqv.ne[1], kqv.ne[2] * kqv.ne[3])
        }

        ggmlBuildForwardExpand(graph, cur)
        return cur
    }

    /** Build attention inputs for the no-cache path. Port of `build_attn_inp_no_cache`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttnInpNoCache(): LlmGraphInputAttnNoCache {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputAttnNoCache(hparams, cparams)

        inp.selfKqMask = ggmlNewTensor4d(c, GGMLType.F32, nTokens, nTokens, 1, 1)
        ggmlSetInput(inp.selfKqMask!!)
        inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
            else inp.selfKqMask

        if (hparams.swaType != LlamaSwaType.NONE) {
            inp.selfKqMaskSwa = ggmlNewTensor4d(c, GGMLType.F32, nTokens, nTokens, 1, 1)
            ggmlSetInput(inp.selfKqMaskSwa!!)
            inp.selfKqMaskSwaCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMaskSwa!!, GGMLType.F16)
                else inp.selfKqMaskSwa
        }

        return r.addInputTyped(inp)
    }

    /**
     * Self-attention without a persistent KV cache.
     *
     * Port of `llm_graph_context::build_attn(llm_graph_input_attn_no_cache *)`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttn(
        inp: LlmGraphInputAttnNoCache,
        wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")

        ggmlBuildForwardExpand(graph, qCur)
        ggmlBuildForwardExpand(graph, kCur)
        ggmlBuildForwardExpand(graph, vCur)

        val isSwa = hparams.isSwa(il)
        val kqMask = if (isSwa) inp.getKqMaskSwa() else inp.getKqMask()

        var cur = buildAttnMha(qCur, kCur, vCur, kqB, kqMask, sinks, vMla, kqScale, il)
        cb(cur, "kqv_out", il)

        if (wo != null) {
            cur = buildLoraMm(wo, cur, woS)
        }
        if (woB != null) {
            cur = ggmlAdd(c, cur, woB)
        }
        return cur
    }

    // port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_context::build_attn_inp_kv

    /**
     * Build attention inputs for the KV-cache path.
     *
     * Port of `llm_graph_context::build_attn_inp_kv`.  Creates K/V index
     * tensors, the causal KQ mask, and optional rotation matrices from the
     * [KVCacheContext] stored in [params].mctx.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttnInpKv(): LlmGraphInputAttnKv {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val mctxCur = params.mctx as? KVCacheContext
            ?: error("mctx must be KVCacheContext for attn_inp_kv")

        val inp = buildAttnInpKvImpl(c, ubatch, hparams, cparams, mctxCur)

        return r.addInputTyped(inp)
    }

    // port-lint: source llama.cpp/src/llama-graph.cpp  llm_graph_context::build_attn(llm_graph_input_attn_kv *)

    /**
     * Self-attention with a persistent KV cache.
     *
     * Port of `llm_graph_context::build_attn(llm_graph_input_attn_kv *)`.
     * Applies optional rotation, stores K/V into the cache, retrieves cached
     * K/V, runs multi-head attention, and applies the output projection.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttn(
        inp: LlmGraphInputAttnKv,
        wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        val mctxCur = params.mctx as? KVCacheContext ?: error("mctx required for KV attention")

        var q = qCur
        var k = kCur
        var v = vCur

        // Apply optional rotation matrices
        if (inp.selfKRot != null) {
            q = ggmlMulMatAux(c, q, inp.selfKRot!!)
            k = ggmlMulMatAux(c, k, inp.selfKRot!!)
        }
        if (inp.selfVRot != null) {
            v = ggmlMulMatAux(c, v, inp.selfVRot!!)
        }

        // Ensure Q/K/V nodes are materialized before cache operations
        ggmlBuildForwardExpand(graph, q)
        ggmlBuildForwardExpand(graph, v)
        ggmlBuildForwardExpand(graph, k)

        // Store K and V into the cache
        val kIdxs = inp.getKIdxs() ?: error("K indices not initialised")
        val vIdxs = inp.getVIdxs() ?: error("V indices not initialised")

        ggmlBuildForwardExpand(graph, mctxCur.cpyK(c, k, kIdxs, il))
        ggmlBuildForwardExpand(graph, mctxCur.cpyV(c, v, vIdxs, il))

        // Retrieve cached K/V for the full attention window
        val kqMask = inp.getKqMask()
        val kCached = mctxCur.getK(c, il)
        val vCached = mctxCur.getV(c, il)

        var cur = buildAttnMha(q, kCached, vCached, kqB, kqMask, sinks, vMla, kqScale, il)
        cb(cur, "kqv_out", il)

        // Apply optional V rotation to the output
        if (inp.selfVRot != null) {
            cur = ggmlMulMatAux(c, cur, inp.selfVRot!!)
        }

        // Output projection
        if (wo != null) {
            if (arch == LlamaModelArch.GLM4 || arch == LlamaModelArch.GLM4_MOE || arch == LlamaModelArch.JAIS2) {
                // GLM4/JAIS2 need F32 precision accumulators
                cur = buildLoraMm(wo, cur)
                ggmlMulMatSetPrec(cur, GGMLPrec.F32)
                if (woS != null) {
                    cur = ggmlMul(c, cur, woS)
                }
            } else {
                cur = buildLoraMm(wo, cur, woS)
            }
        }
        if (woB != null) {
            cur = ggmlAdd(c, cur, woB)
        }

        return cur
    }

    // =========================================================================
    // Builder methods ported from llama-graph.cpp (lines 1801–2897)
    // =========================================================================

    /** Build attention-temperature scaling input. Port of `build_inp_attn_scale`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpAttnScale(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputAttnTemp(
            nAttnTempFloorScale = hparams.nAttnTempFloorScale,
            fAttnTempScale = hparams.fAttnTempScale,
            fAttnTempOffset = hparams.fAttnTempOffset,
        )
        // Must be 1x1xN for broadcasting across attention heads
        inp.attnScale = ggmlNewTensor3d(c, GGMLType.F32, 1, 1, nTokens)
        ggmlSetInput(inp.attnScale!!)
        ggmlSetName(inp.attnScale!!, "attn_scale")
        r.addInput(inp)
        return inp.attnScale!!
    }

    /** Build mean-pooling input. Port of `build_inp_mean`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpMean(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputMean(cparams)
        inp.mean = ggmlNewTensor2d(c, GGMLType.F32, nTokens, ubatch.nSeqsUnq.toLong())
        ggmlSetInput(inp.mean!!)
        r.addInput(inp)
        return inp.mean!!
    }

    /** Build CLS-token selector input. Port of `build_inp_cls`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpCls(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputCls(cparams, arch)
        inp.cls = ggmlNewTensor1d(c, GGMLType.I32, ubatch.nSeqsUnq.toLong())
        ggmlSetInput(inp.cls!!)
        r.addInput(inp)
        return inp.cls!!
    }

    /** Build cross-attention embedding input. Port of `build_inp_cross_embd`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpCrossEmbd(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val cr = cross
        val inp = LlmGraphInputCrossEmbd(cr)
        val nEmbdCross = if (cr != null && cr.vEmbd.isNotEmpty()) cr.nEmbd else hparams.nEmbdInp().toLong()
        val nEnc = if (cr != null && cr.vEmbd.isNotEmpty()) cr.nEnc else hparams.nCtxTrain.toLong()
        inp.crossEmbd = ggmlNewTensor2d(c, GGMLType.F32, nEmbdCross, nEnc)
        ggmlSetInput(inp.crossEmbd!!)
        r.addInput(inp)
        return inp.crossEmbd!!
    }

    /** Build encoder-side position-bucket input. Port of `build_inp_pos_bucket_enc`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpPosBucketEnc(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val inp = LlmGraphInputPosBucket(hparams)
        inp.posBucket = ggmlNewTensor2d(c, GGMLType.I32, nTokens, nTokens)
        ggmlSetInput(inp.posBucket!!)
        r.addInput(inp)
        return inp.posBucket!!
    }

    /** Build decoder-side position-bucket input. Port of `build_inp_pos_bucket_dec`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpPosBucketDec(): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val mctxCur = params.mctx as? KVCacheContext
            ?: error("mctx must be KVCacheContext for pos_bucket_dec")
        val inp = LlmGraphInputPosBucketKv(hparams)
        val nKv = mctxCur.getNKv().toLong()
        inp.posBucket = ggmlNewTensor2d(c, GGMLType.I32, nKv, nTokens)
        ggmlSetInput(inp.posBucket!!)
        r.addInput(inp)
        return inp.posBucket!!
    }

    /**
     * Compute position bias from a bucket tensor and relative-attention bias weights.
     * Port of `build_pos_bias`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildPosBias(posBucket: GGMLTensor, attnRelB: GGMLTensor): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        // Flatten the 2D bucket to 1D for ggml_get_rows
        val posBucket1d = ggmlReshape1d(c, posBucket, posBucket.ne[0] * posBucket.ne[1])
        cb(posBucket1d, "pos_bucket_1d", -1)

        var posBias = ggmlGetRows(c, attnRelB, posBucket1d)
        // Reshape to 3D and permute to [nHead, ne0, ne1] layout
        posBias = ggmlReshape3d(c, posBias, posBias.ne[0], posBucket.ne[0], posBucket.ne[1])
        posBias = ggmlPermute(c, posBias, 2, 0, 1, 3)
        posBias = ggmlCont(c, posBias)
        cb(posBias, "pos_bias", -1)
        return posBias
    }

    /** Build cross-attention inputs. Port of `build_attn_inp_cross`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttnInpCross(): LlmGraphInputAttnCross {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val cr = cross
        val inp = LlmGraphInputAttnCross(cr)
        val nEnc = if (cr != null && cr.vEmbd.isNotEmpty()) cr.nEnc else hparams.nCtxTrain.toLong()
        inp.crossKqMask = ggmlNewTensor4d(c, GGMLType.F32, nEnc, nTokens, 1, 1)
        ggmlSetInput(inp.crossKqMask!!)
        inp.crossKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.crossKqMask!!, GGMLType.F16)
            else inp.crossKqMask
        return r.addInputTyped(inp)
    }

    /** Build V-less (K-only) attention inputs. Port of `build_attn_inp_k`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttnInpK(): LlmGraphInputAttnK {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val mctxCur = params.mctx as? KVCacheContext
            ?: error("mctx must be KVCacheContext for attn_inp_k")

        val inp = buildAttnInpKImpl(c, ubatch, hparams, cparams, mctxCur)

        return r.addInputTyped(inp)
    }

    /** Build ISWA attention inputs. Port of `build_attn_inp_kv_iswa`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttnInpKvIswa(): LlmGraphInputAttnKvIswa {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val mctxCur = params.mctx as? LlamaKvCacheIswaContext
            ?: error("mctx must be LlamaKvCacheIswaContext for attn_inp_kv_iswa")
        val inp = LlmGraphInputAttnKvIswa(hparams, cparams)

        // Base stream
        val baseCtx = mctxCur.getBase() as? KVCacheContext
            ?: error("ISWA base context must be KVCacheContext")
        inp.selfKIdxs = baseCtx.buildInputKIdxs(c, ubatch)
        inp.selfVIdxs = baseCtx.buildInputVIdxs(c, ubatch)
        inp.selfKqMask = buildAttnInpKqMask(c, baseCtx, ubatch, cparams)
        inp.selfKqMaskCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMask!!, GGMLType.F16)
            else inp.selfKqMask

        // SWA stream
        val swaCtx = mctxCur.getSwa() as? KVCacheContext
            ?: error("ISWA SWA context must be KVCacheContext")
        inp.selfKIdxsSwa = swaCtx.buildInputKIdxs(c, ubatch)
        inp.selfVIdxsSwa = swaCtx.buildInputVIdxs(c, ubatch)
        inp.selfKqMaskSwa = buildAttnInpKqMask(c, swaCtx, ubatch, cparams)
        inp.selfKqMaskSwaCnv = if (cparams.flashAttn) ggmlCast(c, inp.selfKqMaskSwa!!, GGMLType.F16)
            else inp.selfKqMaskSwa

        // RoPE rotation tensors
        inp.selfKRot = baseCtx.buildInputKRot(c)
        inp.selfVRot = baseCtx.buildInputVRot(c)
        inp.selfKRotSwa = swaCtx.buildInputKRot(c)
        inp.selfVRotSwa = swaCtx.buildInputVRot(c)

        return r.addInputTyped(inp)
    }

    /** Build self-attention with V-less (K-only) cache. Port of `build_attn(attn_k)`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttn(
        inp: LlmGraphInputAttnK,
        wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        // Expand to prevent reordering
        ggmlBuildForwardExpand(graph, qCur)
        ggmlBuildForwardExpand(graph, vCur)
        ggmlBuildForwardExpand(graph, kCur)

        // Store K to cache
        val kIdxs = inp.getKIdxs()
        if (kIdxs != null) {
            val mctxCur = params.mctx as? KVCacheContext
            if (mctxCur != null) {
                ggmlBuildForwardExpand(graph, mctxCur.cpyK(c, kCur, kIdxs, il))
            }
        }

        val kqMask = inp.getKqMask()
        val q = qCur
        val k = (params.mctx as? KVCacheContext)?.getK(c, il) ?: kCur
        // V-less: derive v from k view
        val v = ggmlView4d(c, k, vCur.ne[0], k.ne[1], k.ne[2], k.ne[3],
            k.nb[1], k.nb[2], k.nb[3], 0u)

        var cur = buildAttnMha(q, k, v, kqB, kqMask, sinks, vMla, kqScale, il)
        cb(cur, "kqv_out", il)

        if (wo != null) {
            if (arch == LlamaModelArch.GLM4 || arch == LlamaModelArch.GLM4_MOE) {
                cur = buildLoraMm(wo, cur)
                ggmlMulMatSetPrec(cur, GGMLPrec.F32)
                if (woS != null) cur = ggmlMul(c, cur, woS)
            } else {
                cur = buildLoraMm(wo, cur, woS)
            }
        }
        if (woB != null) {
            cur = ggmlAdd(c, cur, woB)
        }
        return cur
    }

    /** Build self-attention with ISWA cache. Port of `build_attn(iswa)`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttn(
        inp: LlmGraphInputAttnKvIswa,
        wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor?, vCur: GGMLTensor?,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        val isSwa = hparams.isSwa(il)

        val kRot = if (isSwa) inp.selfKRotSwa else inp.selfKRot
        val vRot = if (isSwa) inp.selfVRotSwa else inp.selfVRot

        var qWork = qCur
        var kWork = kCur
        var vWork = vCur

        if (kRot != null) {
            qWork = ggmlMulMatAux(c, qWork, kRot)
            if (kWork != null) kWork = ggmlMulMatAux(c, kWork, kRot)
        }
        if (vRot != null) {
            if (vWork != null) vWork = ggmlMulMatAux(c, vWork, vRot)
        }

        ggmlBuildForwardExpand(graph, qWork)
        kWork?.let { ggmlBuildForwardExpand(graph, it) }
        vWork?.let { ggmlBuildForwardExpand(graph, it) }

        val mctxIswa = params.mctx as? LlamaKvCacheIswaContext
            ?: error("mctx must be LlamaKvCacheIswaContext for ISWA attention")
        val mctxCur = (if (isSwa) mctxIswa.getSwa() else mctxIswa.getBase()) as? KVCacheContext
            ?: error("ISWA sub-context must be KVCacheContext")

        // Optionally store K to cache
        if (kWork != null) {
            val kIdxs = if (isSwa) inp.getKIdxsSwa() else inp.getKIdxs()
            if (kIdxs != null) {
                ggmlBuildForwardExpand(graph, mctxCur.cpyK(c, kWork, kIdxs, il))
            }
        }

        // Optionally store V to cache
        if (vWork != null) {
            val vIdxs = if (isSwa) inp.getVIdxsSwa() else inp.getVIdxs()
            if (vIdxs != null) {
                ggmlBuildForwardExpand(graph, mctxCur.cpyV(c, vWork, vIdxs, il))
            }
        }

        val kqMask = if (isSwa) inp.getKqMaskSwa() else inp.getKqMask()
        val q = qWork
        val k = mctxCur.getK(c, il)
        val v = mctxCur.getV(c, il)

        var cur = buildAttnMha(q, k, v, kqB, kqMask, sinks, vMla, kqScale, il)
        cb(cur, "kqv_out", il)

        if (vRot != null) {
            cur = ggmlMulMatAux(c, cur, vRot)
        }

        if (wo != null) {
            cur = buildLoraMm(wo, cur, woS)
        }
        if (woB != null) {
            cur = ggmlAdd(c, cur, woB)
        }
        return cur
    }

    /** Build cross-attention sub-graph. Port of `build_attn(cross)`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildAttn(
        inp: LlmGraphInputAttnCross,
        wo: GGMLTensor?, woB: GGMLTensor?, woS: GGMLTensor?,
        qCur: GGMLTensor, kCur: GGMLTensor, vCur: GGMLTensor,
        kqB: GGMLTensor?, sinks: GGMLTensor?, vMla: GGMLTensor?,
        kqScale: Float, il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")
        ggmlBuildForwardExpand(graph, qCur)
        ggmlBuildForwardExpand(graph, kCur)
        ggmlBuildForwardExpand(graph, vCur)

        val kqMask = inp.getKqMaskCross()
        var cur = buildAttnMha(qCur, kCur, vCur, kqB, kqMask, sinks, vMla, kqScale, il)
        cb(cur, "kqv_out", il)

        if (wo != null) {
            cur = buildLoraMm(wo, cur, woS)
        }
        if (woB != null) {
            cur = ggmlAdd(c, cur, woB)
        }
        return cur
    }

    // -- recurrent-state builders -----------------------------------------------

    /** Build recurrent-state copy input. Port of `build_rs_inp`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildRsInp(): LlmGraphInputRs {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val mctxCur = params.mctx as? LlamaMemoryRecurrentContext
            ?: error("mctx must be LlamaMemoryRecurrentContext for rs_inp")

        val inp = buildRsInpImpl(c, ubatch, mctxCur)

        return r.addInputTyped(inp)
    }

    /**
     * Build recurrent-state (SSM / RWKV) sub-graph (low-level overload).
     * Port of `build_rs(s, state_copy_main, state_copy_extra, ...)`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildRs(
        s: GGMLTensor,
        stateCopyMain: GGMLTensor,
        stateCopyExtra: GGMLTensor,
        stateSize: Int,
        nSeqs: Int,
        nRs: Int,
        rsHead: Int,
        rsSize: Int,
        rsZero: Int,
        getStateRows: (GGMLContext, GGMLTensor, GGMLTensor) -> GGMLTensor,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val graph = gf ?: error("gf not initialised")

        val states = ggmlReshape2d(c, s, stateSize.toLong(), rsSize.toLong())

        // Clear a single state which will be copied to other cleared states
        val zeroMultiplier = if (rsZero >= 0) 1L else 0L
        val stateZero = ggmlView1d(c, states,
            stateSize.toLong() * zeroMultiplier,
            (rsZero.toLong() * states.nb[1].toLong() * zeroMultiplier).toULong())
        ggmlBuildForwardExpand(graph, ggmlScaleInplace(c, stateZero, 0.0f))

        // Copy states: {state_size, rs_size} -> {state_size, n_seqs}
        val outputStates = getStateRows(c, states, stateCopyMain)
        ggmlBuildForwardExpand(graph, outputStates)

        // Copy extra states (between n_seqs and n_rs)
        val statesExtra = ggmlGetRows(c, states, stateCopyExtra)
        ggmlBuildForwardExpand(graph,
            ggmlCpy(c, statesExtra,
                ggmlView2d(c, s, stateSize.toLong(), (nRs - nSeqs).toLong(),
                    s.nb[1], ((rsHead + nSeqs).toLong() * s.nb[1].toLong()).toULong())))

        return outputStates
    }

    /**
     * Build recurrent-state sub-graph (high-level overload using input).
     * Port of `build_rs(inp, s, state_size, n_seqs, get_state_rows)`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildRs(
        inp: LlmGraphInputRs,
        s: GGMLTensor,
        stateSize: Int,
        nSeqs: Int,
        getStateRows: (GGMLContext, GGMLTensor, GGMLTensor) -> GGMLTensor = ::ggmlGetRows,
    ): GGMLTensor {
        val kvState = inp.mctx ?: error("LlmGraphInputRs.mctx not initialised")
        val sCopyMain = inp.sCopyMain ?: error("sCopyMain not initialised")
        val sCopyExtra = inp.sCopyExtra ?: error("sCopyExtra not initialised")
        return buildRs(
            s, sCopyMain, sCopyExtra,
            stateSize, nSeqs,
            kvState.getNRs(), kvState.getHead(), kvState.getSize(), kvState.getRsZ(),
            getStateRows,
        )
    }

    /** Load RWKV token-shift state. Port of `build_rwkv_token_shift_load`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildRwkvTokenShiftLoad(
        inp: LlmGraphInputRs,
        ubatchArg: LlamaUBatch,
        il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val mctxCur = params.mctx as? LlamaMemoryRecurrentContext
            ?: error("mctx must be LlamaMemoryRecurrentContext for rwkv_token_shift_load")

        val tokenShiftCount = hparams.tokenShiftCount
        val nSeqs = ubatchArg.nSeqs

        val tokenShiftAll = mctxCur.getRL(il) ?: error("missing rL[$il]")
        var tokenShift = buildRs(inp, tokenShiftAll, hparams.nEmbdR(), nSeqs)
        tokenShift = ggmlReshape3d(c, tokenShift, hparams.nEmbd.toLong(), tokenShiftCount.toLong(), nSeqs.toLong())

        return tokenShift
    }

    /** Store RWKV token-shift state. Port of `build_rwkv_token_shift_store`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildRwkvTokenShiftStore(
        tokenShift: GGMLTensor,
        ubatchArg: LlamaUBatch,
        il: Int,
    ): GGMLTensor {
        val c = ctx0 ?: error("ctx0 not initialised")
        val mctxCur = params.mctx as? LlamaMemoryRecurrentContext
            ?: error("mctx must be LlamaMemoryRecurrentContext for rwkv_token_shift_store")

        val tokenShiftCount = hparams.tokenShiftCount
        val nEmbdVal = hparams.nEmbd
        val nSeqs = ubatchArg.nSeqs
        val kvHead = mctxCur.getHead()
        val rL = mctxCur.getRL(il) ?: error("missing rL[$il]")

        return ggmlCpy(c,
            ggmlView1d(c, tokenShift, (nEmbdVal * nSeqs * tokenShiftCount).toLong(), 0u),
            ggmlView1d(c, rL,
                hparams.nEmbdR().toLong() * nSeqs,
                (hparams.nEmbdR().toLong() * kvHead * ggmlElementSize(rL).toLong()).toULong()))
    }

    // -- hybrid memory ----------------------------------------------------------

    /** Build hybrid (KV + recurrent) memory inputs. Port of `build_inp_mem_hybrid`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpMemHybrid(): LlmGraphInputMemHybrid {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val mctxCur = params.mctx as? LlamaMemoryHybridContext
            ?: error("mctx must be LlamaMemoryHybridContext for mem_hybrid")

        val inpRs   = buildRsInpImpl(c, ubatch, mctxCur.getRecr() ?: error("recr context required"))
        val inpAttn = buildAttnInpKvImpl(c, ubatch, hparams, cparams, mctxCur.getAttn() ?: error("attn context required"))

        val inp = LlmGraphInputMemHybrid(cparams, inpAttn, inpRs)
        return r.addInputTyped(inp)
    }

    /** Build hybrid (K-only + recurrent) memory inputs. Port of `build_inp_mem_hybrid_k`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpMemHybridK(): LlmGraphInputMemHybridK {
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val mctxCur = params.mctx as? LlamaMemoryHybridContext
            ?: error("mctx must be LlamaMemoryHybridContext for mem_hybrid_k")

        val inpRs   = buildRsInpImpl(c, ubatch, mctxCur.getRecr() ?: error("recr context required"))
        val inpAttn = buildAttnInpKImpl(c, ubatch, hparams, cparams, mctxCur.getAttn() ?: error("attn context required"))

        val inp = LlmGraphInputMemHybridK(cparams, inpAttn, inpRs)
        return r.addInputTyped(inp)
    }

    /** Build hybrid (ISWA + recurrent) memory inputs. Port of `build_inp_mem_hybrid_iswa`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildInpMemHybridIswa(): LlmGraphInputMemHybridIswa {
        val r = res ?: error("res not initialised")
        val inpRs = buildRsInp()
        val inpAttn = buildAttnInpKvIswa()
        val inp = LlmGraphInputMemHybridIswa(cparams, inpAttn, inpRs)
        return r.addInputTyped(inp)
    }

    // -- pooling / sampling / dense-out -----------------------------------------

    /**
     * Build dense-out (final linear projections after pooling).
     * Port of `build_dense_out`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildDenseOut(
        dense2: GGMLTensor?, dense2B: GGMLTensor?,
        dense3: GGMLTensor?,
    ) {
        if (!cparams.embeddings || !(dense2 != null || dense2B != null || dense3 != null)) return
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val graph = gf ?: error("gf not initialised")
        var cur = r.tEmbdPooled ?: r.tEmbd ?: error("missing t_embd_pooled/t_embd")

        if (dense2 != null) {
            cur = ggmlMulMat(c, dense2, cur)
        }
        if (dense2B != null) {
            cur = ggmlAdd(c, cur, dense2B)
        }
        if (dense3 != null) {
            cur = ggmlMulMat(c, dense3, cur)
        }
        cb(cur, "result_embd_pooled", -1)
        r.tEmbdPooled = cur
        ggmlBuildForwardExpand(graph, cur)
    }

    /**
     * Build pooling sub-graph.
     * Port of `build_pooling`.
     */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildPooling(
        cls: GGMLTensor?, clsB: GGMLTensor?,
        clsOut: GGMLTensor?, clsOutB: GGMLTensor?,
        clsNorm: GGMLTensor?,
    ) {
        if (!cparams.embeddings) return
        val c = ctx0 ?: error("ctx0 not initialised")
        val r = res ?: error("res not initialised")
        val graph = gf ?: error("gf not initialised")
        val inp = r.tEmbd ?: error("missing result_norm/result_embd tensor")

        var cur: GGMLTensor

        when (poolingType) {
            LlamaPoolingType.NONE -> {
                cur = inp
            }
            LlamaPoolingType.MEAN -> {
                val inpMean = buildInpMean()
                cur = ggmlMulMat(c, ggmlCont(c, ggmlTranspose(c, inp)), inpMean)
            }
            LlamaPoolingType.CLS, LlamaPoolingType.LAST -> {
                val inpCls = buildInpCls()
                cur = ggmlGetRows(c, inp, inpCls)
            }
            LlamaPoolingType.RANK -> {
                if (arch == LlamaModelArch.MODERN_BERT) {
                    val inpMean = buildInpMean()
                    cur = ggmlMulMat(c, ggmlCont(c, ggmlTranspose(c, inp)), inpMean)
                } else {
                    val inpCls = buildInpCls()
                    cur = ggmlGetRows(c, inp, inpCls)
                }
                // Classification head
                if (cls != null) {
                    cur = ggmlMulMat(c, cls, cur)
                    if (clsB != null) cur = ggmlAdd(c, cur, clsB)
                    if (arch == LlamaModelArch.MODERN_BERT) {
                        cur = ggmlGelu(c, cur)
                    } else {
                        cur = ggmlTanh(c, cur)
                    }
                    if (clsNorm != null) {
                        cur = buildNorm(cur, clsNorm, null, LlmNormType.NORM, -1)
                    }
                }
                if (clsOut != null) {
                    cur = ggmlMulMat(c, clsOut, cur)
                    if (clsOutB != null) cur = ggmlAdd(c, cur, clsOutB)
                }
                // Softmax for Qwen3 reranker
                if (arch == LlamaModelArch.QWEN3 || arch == LlamaModelArch.QWEN3VL) {
                    cur = ggmlSoftMax(c, cur)
                }
            }
            else -> error("unknown pooling type: $poolingType")
        }

        cb(cur, "result_embd_pooled", -1)
        r.tEmbdPooled = cur
        ggmlBuildForwardExpand(graph, cur)
    }

    /** Build backend-sampling sub-graph. Port of `build_sampling`. */
    // port-lint: source llama.cpp/src/llama-graph.cpp
    fun buildSampling() {
        val r = res ?: return
        val logits = r.tLogits ?: return
        val c = ctx0 ?: return
        val graph = gf ?: return

        // Backend sampling builds a sub-graph that applies per-sequence samplers.
        // The sampler backend infrastructure is not yet ported; once available:
        //   1. Pad logits with a dummy row (ggmlPad(ctx0, logits, 0, 1, 0, 0))
        //   2. For each (seqId, sampler) pair, extract logit row and call backendApply
        //   3. Wire data.sampled/probs/logits/candidates via ggmlBuildForwardSelect
        // Sampler backend infrastructure is not yet ported; once available:
    }

}
