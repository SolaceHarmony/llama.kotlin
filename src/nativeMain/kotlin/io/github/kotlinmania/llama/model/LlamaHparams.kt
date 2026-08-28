// port-lint: source llama.cpp/src/llama-hparams.h
package io.github.kotlinmania.llama.model

// =============================================================================
// LlamaHparams – Model hyperparameters
// Ported from: llama-hparams.h  struct llama_hparams
//              llama-hparams.cpp (method implementations)
// =============================================================================

/** Maximum number of layers supported. Matches `LLAMA_MAX_LAYERS` in C++. */
const val LLAMA_MAX_LAYERS = 512

/** Maximum number of experts. Matches `LLAMA_MAX_EXPERTS` in C++. */
const val LLAMA_MAX_EXPERTS = 512

/**
 * PosNet hyperparameters (WavTokenizer).
 *
 * Port of `struct llama_hparams_posnet` from `llama-hparams.h`.
 */
data class LlamaHparamsPosnet(
    var nEmbd: Int = 0,
    var nLayer: Int = 0,
)

/**
 * ConvNext hyperparameters (WavTokenizer).
 *
 * Port of `struct llama_hparams_convnext` from `llama-hparams.h`.
 */
data class LlamaHparamsConvnext(
    var nEmbd: Int = 0,
    var nLayer: Int = 0,
)

/**
 * Model hyperparameters read from GGUF metadata.
 *
 * This is a direct port of `struct llama_hparams` in `llama-hparams.h`.
 * All fields use the same names and defaults as the C++ struct so that call
 * sites porting higher-level logic can map 1-to-1.  Per-layer arrays use
 * [IntArray] (or [FloatArray] / [BooleanArray]) of fixed size
 * [LLAMA_MAX_LAYERS].
 */
data class LlamaHparams(
    // -- basic shape --
    var vocabOnly: Boolean = false,
    var noAlloc: Boolean = false,
    var ropeFinetuned: Boolean = false,
    var useParRes: Boolean = false,
    var swinNorm: Boolean = false,

    var nCtxTrain: Int = 0,
    var nEmbd: Int = 0,
    var nLayer: Int = 0,
    var nLayerKvFromStart: Int = -1,
    var nExpert: Int = 0,
    var nExpertUsed: Int = 0,
    var nRelAttnBkts: Int = 0,

    // -- head dimensions (full-attention vs SWA) --
    var nEmbdHeadKFull: Int = 0,
    var nEmbdHeadVFull: Int = 0,
    var nEmbdHeadKSwa: Int = 0,
    var nEmbdHeadVSwa: Int = 0,

    // -- RoPE dimensions --
    var nRotFull: Int = 0,
    var nRotSwa: Int = 0,

    // -- MLA (deepseek2) --
    var nEmbdHeadKMlaImpl: Int = 0,
    var nEmbdHeadVMlaImpl: Int = 0,

    // -- WavTokenizer sub-hparams --
    val posnet: LlamaHparamsPosnet = LlamaHparamsPosnet(),
    val convnext: LlamaHparamsConvnext = LlamaHparamsConvnext(),

    var nShortconvLCache: Int = 0,

    // -- per-layer arrays --
    val nHeadArr: IntArray = IntArray(LLAMA_MAX_LAYERS),
    val nHeadKvArr: IntArray = IntArray(LLAMA_MAX_LAYERS),
    val nFfArr: IntArray = IntArray(LLAMA_MAX_LAYERS),

    // -- MoE --
    var nLayerDenseLead: Int = 0,
    var nLoraQ: Int = 0,
    var nLoraKv: Int = 0,
    var nFfExp: Int = 0,
    var nFfShexp: Int = 0,
    var nFfChexp: Int = 0,
    var nExpertShared: Int = 0,
    var nNormGroups: Int = 0,
    var nExpertGroups: Int = 0,
    var nGroupUsed: Int = 0,
    var nGroupExperts: Int = 0,

    var expertGroupScale: Float = 0.05f,
    var expertWeightsScale: Float = 0.0f,
    var expertWeightsNorm: Boolean = false,
    var expertGatingFunc: Int = LlamaExpertGatingFuncType.NONE.id,
    var moeEveryNLayers: Int = 0,
    var moeLatentSize: Int = 0,
    var nextnPredictLayers: Int = 0,

    // -- normalization --
    var fNormEps: Float = 0.0f,
    var fNormRmsEps: Float = 0.0f,
    var fNormGroupEps: Float = 0.0f,

    var fAttnLogitSoftcapping: Float = 50.0f,
    var fRouterLogitSoftcapping: Float = 30.0f,
    var fFinalLogitSoftcapping: Float = 30.0f,

    // -- RWKV --
    var rescaleEveryNLayers: Int = 0,
    var timeMixExtraDim: Int = 0,
    var timeDecayExtraDim: Int = 0,
    var wkvHeadSize: Int = 0,
    var tokenShiftCount: Int = 2,
    var nLoraDecay: Int = 0,
    var nLoraIclr: Int = 0,
    var nLoraValueResMix: Int = 0,
    var nLoraGate: Int = 0,

    // -- RoPE --
    var ropeAttnFactor: Float = 1.0f,
    var ropeFreqBaseTrain: Float = 0.0f,
    var ropeFreqBaseTrainSwa: Float = 10000.0f,
    var ropeFreqScaleTrain: Float = 0.0f,
    var ropeFreqScaleTrainSwa: Float = 1.0f,

    var nCtxOrigYarn: Int = 0,
    var ropeYarnLogMul: Float = 0.0f,
    var yarnExtFactor: Float = -1.0f,
    var yarnAttnFactor: Float = 1.0f,
    var yarnBetaFast: Float = 32.0f,
    var yarnBetaSlow: Float = 1.0f,

    val ropeSections: IntArray = IntArray(4),

    // -- Sliding Window Attention --
    var swaType: LlamaSwaType = LlamaSwaType.NONE,
    var nSwa: Int = 0,
    val swaLayers: IntArray = IntArray(LLAMA_MAX_LAYERS),

    // -- SSM (Mamba) --
    var ssmDConv: Int = 0,
    var ssmDInner: Int = 0,
    var ssmDState: Int = 0,
    var ssmDtRank: Int = 0,
    var ssmNGroup: Int = 0,

    // -- Kimi Linear KDA --
    var nEmbdHeadKda: Int = 0,

    // -- hybrid recurrent layers --
    val recurrentLayerArr: BooleanArray = BooleanArray(LLAMA_MAX_LAYERS),

    var ssmDtBcRms: Boolean = false,

    // -- clamping / alibi --
    var fClampKqv: Float = 0.0f,
    var fMaxAlibiBias: Float = 0.0f,
    var fLogitScale: Float = 0.0f,

    // -- Granite / Granite MoE --
    var fResidualScale: Float = 0.0f,
    var fEmbeddingScale: Float = 0.0f,
    var fAttentionScale: Float = 0.0f,

    // -- grok-2 --
    var fAttnOutScale: Float = 0.0f,
    var attnTempLength: Int = 0,

    var causalAttn: Boolean = true,
    var useAlibi: Boolean = false,
    var attnSoftCap: Boolean = false,
    var useKqNorm: Boolean = false,

    // -- classifier --
    var nClsOut: Int = 1,

    // -- output embedding dimension (0 = use nEmbd) --
    var nEmbdOutImpl: Int = 0,

    // -- llama4 / smallthinker --
    var nMoeLayerStep: Int = 0,
    var nNoRopeLayerStep: Int = 4,
    var nAttnTempFloorScale: Int = 0,
    var fAttnTempScale: Float = 0.0f,
    var fAttnTempOffset: Float = 0.0f,

    // -- gemma3n altup --
    var nAltup: Int = 4,
    var iAltupAct: Int = 0,
    var laurelRank: Int = 64,
    var nEmbdAltup: Int = 256,

    // -- sentence-transformers dense layers --
    var dense2FeatIn: Int = 0,
    var dense2FeatOut: Int = 0,
    var dense3FeatIn: Int = 0,
    var dense3FeatOut: Int = 0,

    // -- xIELU per-layer arrays --
    val xieluAlphaN: FloatArray = FloatArray(LLAMA_MAX_LAYERS),
    val xieluAlphaP: FloatArray = FloatArray(LLAMA_MAX_LAYERS),
    val xieluBeta: FloatArray = FloatArray(LLAMA_MAX_LAYERS),
    val xieluEps: FloatArray = FloatArray(LLAMA_MAX_LAYERS),

    // -- DSA (deepseek sparse attention) --
    var indexerNHead: Int = 0,
    var indexerHeadSize: Int = 0,
    var indexerTopK: Int = 0,

    // -- qwen3vl deepstack --
    var nDeepstackLayers: Int = 0,

    // -- gemma4 per-layer embedding --
    var nEmbdPerLayer: Int = 0,

    // -- encoder-decoder --
    var decStartTokenId: Int = LLAMA_TOKEN_NULL,
    var decNLayer: Int = 0,

    // -- pooling / rope types --
    var poolingType: LlamaPoolingType = LlamaPoolingType.NONE,
    var ropeType: LlamaRopeType = LlamaRopeType.NONE,
    var ropeScalingTypeTrain: LlamaRopeScalingType = LlamaRopeScalingType.NONE,

    // -- Step35 per-layer SwiGLU clamping --
    val swigluClampExp: FloatArray = FloatArray(LLAMA_MAX_LAYERS),
    val swigluClampShexp: FloatArray = FloatArray(LLAMA_MAX_LAYERS),
) {
    // =========================================================================
    // Methods – ported from llama-hparams.cpp
    // =========================================================================

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::set_swa_pattern
    /**
     * Populate [swaLayers] with a repeating pattern.
     *
     * @param nPattern  Repeat period – every *n*-th layer is dense. 0 means all SWA, 1 means all dense.
     * @param denseFirst Whether the pattern starts with a dense layer.
     */
    fun setSwaPattern(nPattern: Int, denseFirst: Boolean = false) {
        if (denseFirst) {
            for (il in 0 until nLayer) {
                swaLayers[il] = if (nPattern == 0 || (il % nPattern != 0)) 1 else 0
            }
        } else {
            for (il in 0 until nLayer) {
                swaLayers[il] = if (nPattern == 0 || (il % nPattern < (nPattern - 1))) 1 else 0
            }
        }
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::is_swa_any
    /** Return true if one of the layers is SWA. */
    fun isSwaAny(): Boolean {
        for (il in 0 until nLayer) {
            if (swaLayers[il] != 0) return true
        }
        return false
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_head
    /** Number of attention heads for layer [il]. */
    fun nHead(il: Int = 0): Int {
        if (il < nLayer) return nHeadArr[il]
        error("fatal error")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_head_kv
    /** Number of key-value heads for layer [il]. */
    fun nHeadKv(il: Int = 0): Int {
        if (il < nLayer) return nHeadKvArr[il]
        error("fatal error")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_ff
    /** Feed-forward size for layer [il]. */
    fun nFf(il: Int = 0): Int {
        if (il < nLayer) return nFfArr[il]
        error("fatal error")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_gqa
    /** Number of query groups per key-value head. */
    fun nGqa(il: Int = 0): Int {
        val headCount = nHead(il)
        val headKvCount = nHeadKv(il)
        if (headKvCount == 0) return 0
        return headCount / headKvCount
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_rot
    /** RoPE rotation dimension for layer [il]. */
    fun nRot(il: Int = 0): Int {
        if (il < nLayer) return if (isSwa(il)) nRotSwa else nRotFull
        error("fatal error")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_inp
    /** Dimension of main + auxiliary input embeddings. */
    fun nEmbdInp(): Int {
        var result = nEmbd
        if (nDeepstackLayers > 0) {
            result += nEmbd * nDeepstackLayers
        }
        return result
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_out
    /** Dimension of output embeddings. */
    fun nEmbdOut(): Int = if (nEmbdOutImpl > 0) nEmbdOutImpl else nEmbd

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_head_k
    /** Dimension of key embeddings per head for layer [il]. */
    fun nEmbdHeadK(il: Int = 0): Int {
        if (il < nLayer) return if (isSwa(il)) nEmbdHeadKSwa else nEmbdHeadKFull
        error("fatal error")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_head_v
    /** Dimension of value embeddings per head for layer [il]. */
    fun nEmbdHeadV(il: Int = 0): Int {
        if (il < nLayer) return if (isSwa(il)) nEmbdHeadVSwa else nEmbdHeadVFull
        error("fatal error")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_k_gqa
    /** Total key embedding size across all KV heads for layer [il]. */
    fun nEmbdKGqa(il: Int = 0): Int = nEmbdHeadK(il) * nHeadKv(il)

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_v_gqa
    /** Total value embedding size across all KV heads for layer [il]. */
    fun nEmbdVGqa(il: Int = 0): Int = nEmbdHeadV(il) * nHeadKv(il)

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::is_n_embd_k_gqa_variable
    /** Whether key GQA size varies across layers. */
    fun isNEmbdKGqaVariable(): Boolean {
        val baseline = nEmbdKGqa()
        for (il in 0 until nLayer) {
            if (nEmbdKGqa(il) != baseline) return true
        }
        return false
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::is_n_embd_v_gqa_variable
    /** Whether value GQA size varies across layers. */
    fun isNEmbdVGqaVariable(): Boolean {
        val baseline = nEmbdVGqa()
        for (il in 0 until nLayer) {
            if (nEmbdVGqa(il) != baseline) return true
        }
        return false
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_k_gqa_max
    /** Maximum key GQA size across all layers. */
    fun nEmbdKGqaMax(): Int {
        var v = nEmbdKGqa()
        for (il in 0 until nLayer) v = maxOf(v, nEmbdKGqa(il))
        return v
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_v_gqa_max
    /** Maximum value GQA size across all layers. */
    fun nEmbdVGqaMax(): Int {
        var v = nEmbdVGqa()
        for (il in 0 until nLayer) v = maxOf(v, nEmbdVGqa(il))
        return v
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_r
    /**
     * Dimension of rolling state embeddings.
     * Corresponds to Mamba's conv_states size or RWKV's token_shift states size.
     */
    fun nEmbdR(): Int {
        if (wkvHeadSize != 0) {
            // for RWKV models
            return tokenShiftCount * nEmbd
        }
        if (nShortconvLCache != 0) {
            // for LFM2 models
            return nEmbd * (nShortconvLCache - 1)
        }
        if (nEmbdHeadKda != 0) {
            // for Kimi KDA layers
            val dInner = nHead() * nEmbdHeadKda
            return 3 * (if (ssmDConv > 0) ssmDConv - 1 else 3) * dInner
        }
        // Mamba conv_states
        return (if (ssmDConv > 0) ssmDConv - 1 else 0) * (ssmDInner + 2 * ssmNGroup * ssmDState)
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_s
    /** Dimension of recurrent state embeddings. */
    fun nEmbdS(): Int {
        if (wkvHeadSize != 0) {
            // corresponds to RWKV's wkv_states size
            return nEmbd * wkvHeadSize
        }
        if (nEmbdHeadKda != 0) {
            // for Kimi KDA layers
            return nEmbdHeadKda * nEmbdHeadKda * nHead()
        }
        // corresponds to Mamba's ssm_states size
        return ssmDState * ssmDInner
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::is_recurrent
    /** Whether layer [il] is recurrent (for hybrid models). */
    fun isRecurrent(il: Int): Boolean {
        if (il < nLayer) return recurrentLayerArr[il]
        error("is_recurrent: il ($il) out of bounds (n_layer: $nLayer)")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_pos_per_embd
    /** Number of position values per embedding (4 for M-RoPE/iM-RoPE, 1 otherwise). */
    fun nPosPerEmbd(): Int =
        if (ropeType == LlamaRopeType.MROPE || ropeType == LlamaRopeType.IMROPE) 4 else 1

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::is_swa
    /** Whether layer [il] is a sliding-window attention layer. */
    fun isSwa(il: Int): Boolean {
        if (il < nLayer) return swaLayers[il] != 0
        error("fatal error")
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::is_mla
    /** Whether this model uses Multi-head Latent Attention (DeepSeek-V2 style). */
    fun isMla(): Boolean {
        check(
            (nEmbdHeadKMlaImpl == 0 && nEmbdHeadVMlaImpl == 0) ||
            (nEmbdHeadKMlaImpl != 0 && nEmbdHeadVMlaImpl != 0)
        )
        return nEmbdHeadKMlaImpl != 0 && nEmbdHeadVMlaImpl != 0
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_head_k_mla
    /** Key head dimension for MLA (falls back to full attention dimension). */
    fun nEmbdHeadKMla(): Int = if (isMla()) nEmbdHeadKMlaImpl else nEmbdHeadK()

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_embd_head_v_mla
    /** Value head dimension for MLA (falls back to full attention dimension). */
    fun nEmbdHeadVMla(): Int = if (isMla()) nEmbdHeadVMlaImpl else nEmbdHeadV()

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::has_kv
    /** Whether layer [il] has a KV cache. */
    fun hasKv(il: Int): Boolean {
        if (nLayerKvFromStart >= 0) {
            return il < nLayerKvFromStart
        }
        // by default, all layers have kv
        return true
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::n_layer_kv
    /** Number of layers for which hasKv() returns true. */
    fun nLayerKv(): Int {
        var count = 0
        for (il in 0 until nLayer) {
            if (hasKv(il)) count++
        }
        return count
    }

    // port-lint: source llama.cpp/src/llama-hparams.cpp llama_hparams::use_mrope
    /** Whether M-RoPE is active (first two rope sections are positive). */
    fun useMrope(): Boolean = ropeSections[0] > 0 && ropeSections[1] > 0

    companion object {
        // port-lint: source llama.cpp/src/llama-hparams.h llama_hparams::is_masked_swa
        /**
         * Check whether position [p0] is masked out by sliding-window attention
         * when the query position is [p1].
         */
        fun isMaskedSwa(nSwa: Int, swaType: LlamaSwaType, p0: Int, p1: Int): Boolean {
            require(p0 >= 0 && p1 >= 0)
            when (swaType) {
                LlamaSwaType.NONE -> { /* not masked */ }
                LlamaSwaType.STANDARD -> {
                    if (p1 - p0 >= nSwa) return true
                }
                LlamaSwaType.CHUNKED -> {
                    val posChunkStart = (p1 / nSwa) * nSwa
                    if (p0 < posChunkStart) return true
                }
                LlamaSwaType.SYMMETRIC -> {
                    val halfNSwa = nSwa / 2
                    val posDiff = p1 - p0
                    if (posDiff < -halfNSwa || posDiff > halfNSwa) return true
                }
            }
            return false
        }
    }
}
