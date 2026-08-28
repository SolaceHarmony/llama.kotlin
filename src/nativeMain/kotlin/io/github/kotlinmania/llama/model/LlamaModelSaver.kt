// port-lint: source llama.cpp/src/llama-model-saver.h llama.cpp/src/llama-model-saver.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.GGMLTensor
import io.github.kotlinmania.llama.platform.nativeWriteBytesToFile

// =============================================================================
// LlamaModelSaver – write a model back to GGUF
// Ported from: llama-model-saver.h / llama-model-saver.cpp
//
// The C++ implementation delegates to low-level `gguf_*` functions that
// manipulate an in-memory GGUF context and then serialise to a file.
// In the Kotlin port we use [GGUFWriteContext] (a lightweight mutable
// builder) and [GGUFFileWriter] to flush the result to disk.
// =============================================================================

// ---------------------------------------------------------------------------
// GGUFWriteContext – mutable GGUF builder (replaces gguf_context for writing)
// ---------------------------------------------------------------------------

/**
 * A single KV entry to be serialised into the GGUF metadata section.
 *
 * @param key   The GGUF metadata key string.
 * @param type  The GGUF type tag for the value.
 * @param value The payload.  Must be one of: [UInt], [Int], [Float],
 *              [Boolean], [String], [IntArray], [FloatArray], [UIntArray],
 *              [List]<[String]>, [ByteArray].
 */
data class GGUFKvEntry(
    val key: String,
    val type: GGUFType,
    val value: Any,
)

/**
 * Tensor metadata to be written into the GGUF tensor-info section.
 *
 * @param name       Tensor name (e.g. `"blk.0.attn_q.weight"`).
 * @param nDims      Number of dimensions (1..4).
 * @param ne         Element counts per dimension.
 * @param type       The GGML tensor type (F32, Q4_0, …).
 * @param dataBytes  Raw tensor data.
 */
data class GGUFTensorEntry(
    val name: String,
    val nDims: Int,
    val ne: LongArray,
    val type: io.github.kotlinmania.llama.ore.GGMLType,
    val dataBytes: ByteArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGUFTensorEntry) return false
        return name == other.name
    }

    override fun hashCode(): Int = name.hashCode()
}

/**
 * Mutable GGUF builder that collects KV metadata and tensor info.
 *
 * This is the Kotlin equivalent of the C `gguf_context` used in write mode.
 * After populating the builder you pass it to [GGUFFileWriter.write] to
 * produce a GGUF binary file.
 */
class GGUFWriteContext {
    val kvEntries: MutableList<GGUFKvEntry> = mutableListOf()
    val tensorEntries: MutableList<GGUFTensorEntry> = mutableListOf()

    // -- scalar setters -------------------------------------------------------

    fun setValU32(key: String, value: UInt) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.UINT32, value))
    }

    fun setValI32(key: String, value: Int) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.INT32, value))
    }

    fun setValF32(key: String, value: Float) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.FLOAT32, value))
    }

    fun setValBool(key: String, value: Boolean) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.BOOL, value))
    }

    fun setValStr(key: String, value: String) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.STRING, value))
    }

    // -- array setters --------------------------------------------------------

    fun setArrU32(key: String, values: IntArray, count: Int = values.size) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.UINT32, values.copyOf(count)))
    }

    fun setArrI32(key: String, values: IntArray, count: Int = values.size) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.INT32, values.copyOf(count)))
    }

    fun setArrF32(key: String, values: FloatArray, count: Int = values.size) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.FLOAT32, values.copyOf(count)))
    }

    fun setArrStr(key: String, values: List<String>) {
        kvEntries.add(GGUFKvEntry(key, GGUFType.STRING, values.toList()))
    }

    // -- tensor management ----------------------------------------------------

    /** Returns the index of a tensor with the given [name], or -1 if absent. */
    fun findTensor(name: String): Int =
        tensorEntries.indexOfFirst { it.name == name }

    /** Append a tensor entry.  Does nothing if [tensor] is null. */
    fun addTensor(entry: GGUFTensorEntry) {
        tensorEntries.add(entry)
    }
}

// ---------------------------------------------------------------------------
// Supported-arch check  (port of llama_model_saver_supports_arch)
// ---------------------------------------------------------------------------

/**
 * Returns `true` if the saver can round-trip models of the given [arch].
 *
 * Some newer or highly-custom architectures are not yet supported for
 * re-serialisation — this mirrors the C++ allowlist.
 */
fun llamaModelSaverSupportsArch(arch: LlamaModelArch): Boolean = when (arch) {
    LlamaModelArch.QWEN3NEXT,
    LlamaModelArch.QWEN35,
    LlamaModelArch.QWEN35MOE,
    LlamaModelArch.PLAMO3,
    LlamaModelArch.GEMMA3,
    LlamaModelArch.GEMMA3N,
    LlamaModelArch.COHERE2,
    LlamaModelArch.OLMO2,
    LlamaModelArch.BITNET,
    LlamaModelArch.T5,
    -> false
    else -> true
}

// ---------------------------------------------------------------------------
// LlamaModelSaver
// ---------------------------------------------------------------------------

/**
 * Serialises a [LlamaModelData] instance into GGUF format.
 *
 * Port of `struct llama_model_saver` from `llama-model-saver.h / .cpp`.
 *
 * ## Usage
 * ```
 * val saver = LlamaModelSaver(model)
 * saver.addKvFromModel()
 * saver.addTensorsFromModel()
 * saver.save("/path/to/output.gguf")
 * ```
 *
 * @param model  The loaded model whose weights and metadata will be written.
 *               May be `null` when constructing a saver for a bare
 *               [GGUFWriteContext] (used for testing or manual building).
 * @param arch   Architecture to use when [model] is `null`.
 */
class LlamaModelSaver private constructor(
    val ggufCtx: GGUFWriteContext,
    private val ggufCtxOwned: Boolean,
    val model: LlamaModelData?,
    private val llmKv: LlmKvHelper,
) {
    /**
     * Construct a saver that will serialise [model] into a fresh GGUF context.
     */
    constructor(model: LlamaModelData) : this(
        ggufCtx = GGUFWriteContext(),
        ggufCtxOwned = true,
        model = model,
        llmKv = LlmKvHelper(model.arch),
    ) {
        require(llamaModelSaverSupportsArch(model.arch)) {
            "Architecture ${model.arch} is not supported by the model saver"
        }
    }

    /**
     * Construct a saver for a specific [arch] and an optional pre-existing
     * [ggufCtx].  When [ggufCtx] is `null` a new empty context is created.
     */
    constructor(arch: LlamaModelArch, ggufCtx: GGUFWriteContext? = null) : this(
        ggufCtx = ggufCtx ?: GGUFWriteContext(),
        ggufCtxOwned = ggufCtx == null,
        model = null,
        llmKv = LlmKvHelper(arch),
    )

    // -- typed addKv overloads ------------------------------------------------

    fun addKv(key: LlmKv, value: UInt) {
        ggufCtx.setValU32(llmKv(key), value)
    }

    fun addKv(key: LlmKv, value: Int) {
        ggufCtx.setValI32(llmKv(key), value)
    }

    fun addKv(key: LlmKv, value: Float) {
        ggufCtx.setValF32(llmKv(key), value)
    }

    fun addKv(key: LlmKv, value: Boolean) {
        ggufCtx.setValBool(llmKv(key), value)
    }

    fun addKv(key: LlmKv, value: String) {
        ggufCtx.setValStr(llmKv(key), value)
    }

    // -- array addKv ----------------------------------------------------------

    /**
     * Write a per-layer [IntArray] value.
     *
     * When [perLayer] is `true` the array is trimmed to `hparams.nLayer`.
     * If all values in the trimmed range are identical the array is collapsed
     * to a single scalar — matching the C++ optimisation.
     */
    fun addKvIntArray(key: LlmKv, values: IntArray, perLayer: Boolean = false) {
        require(model != null || !perLayer) { "perLayer requires a model" }
        val n = if (perLayer) model!!.hparams.nLayer else values.size
        require(n <= values.size)
        if (n == 0) return

        if (perLayer) {
            var allSame = true
            for (i in 1 until n) {
                if (values[i] != values[0]) { allSame = false; break }
            }
            if (allSame) {
                addKv(key, values[0])
                return
            }
        }
        ggufCtx.setArrI32(llmKv(key), values, n)
    }

    /**
     * Write a per-layer [FloatArray] value, with the same collapse logic as
     * [addKvIntArray].
     */
    fun addKvFloatArray(key: LlmKv, values: FloatArray, perLayer: Boolean = false) {
        require(model != null || !perLayer) { "perLayer requires a model" }
        val n = if (perLayer) model!!.hparams.nLayer else values.size
        require(n <= values.size)
        if (n == 0) return

        if (perLayer) {
            var allSame = true
            for (i in 1 until n) {
                if (values[i] != values[0]) { allSame = false; break }
            }
            if (allSame) {
                addKv(key, values[0])
                return
            }
        }
        ggufCtx.setArrF32(llmKv(key), values, n)
    }

    fun addKv(key: LlmKv, values: List<String>) {
        ggufCtx.setArrStr(llmKv(key), values)
    }

    // -- tensor ---------------------------------------------------------------

    /**
     * Add a tensor to the GGUF output.  Skips `null` tensors and tensors
     * that are already present (by name) — matching the C++ duplicate guard
     * for rope_freqs / rope_factors tensors.
     */
    fun addTensor(tensor: GGMLTensor?) {
        if (tensor == null) return
        val name = tensor.name
        if (name.isEmpty()) return
        if (ggufCtx.findTensor(name) >= 0) {
            // Duplicate guard – same as C++ for rope_freqs / rope_factors
            return
        }
        val nDims = tensor.ne.indexOfLast { it > 1L } + 1
        val dims = maxOf(nDims, 1)
        ggufCtx.addTensor(
            GGUFTensorEntry(
                name = name,
                nDims = dims,
                ne = tensor.ne.copyOf(),
                type = tensor.type,
                dataBytes = (tensor.data as? ByteArray) ?: ByteArray(0),
            )
        )
    }

    // -- bulk operations (port of add_kv_from_model / add_tensors_from_model) -

    /**
     * Populate GGUF metadata from the model's hyperparameters and vocabulary.
     *
     * Port of `llama_model_saver::add_kv_from_model()`.
     */
    fun addKvFromModel() {
        val m = requireNotNull(model) { "addKvFromModel requires a model" }
        val hparams = m.hparams

        // -- general --
        addKv(LlmKv.GENERAL_ARCHITECTURE, m.archName())
        addKv(LlmKv.GENERAL_NAME, m.name)

        // -- hyperparameters --
        addKv(LlmKv.CONTEXT_LENGTH, hparams.nCtxTrain)
        addKv(LlmKv.EMBEDDING_LENGTH, hparams.nEmbd)
        if (hparams.nEmbdOutImpl > 0) {
            addKv(LlmKv.EMBEDDING_LENGTH_OUT, hparams.nEmbdOutImpl)
        }
        addKv(LlmKv.BLOCK_COUNT, hparams.nLayer)
        addKv(LlmKv.LEADING_DENSE_BLOCK_COUNT, hparams.nLayerDenseLead)
        addKvIntArray(LlmKv.FEED_FORWARD_LENGTH, hparams.nFfArr, perLayer = true)
        addKv(LlmKv.EXPERT_FEED_FORWARD_LENGTH, hparams.nFfExp)
        addKv(LlmKv.EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.nFfShexp)
        addKvFloatArray(LlmKv.SWIGLU_CLAMP_EXP, hparams.swigluClampExp)
        addKvFloatArray(LlmKv.SWIGLU_CLAMP_SHEXP, hparams.swigluClampShexp)
        addKv(LlmKv.USE_PARALLEL_RESIDUAL, hparams.useParRes)
        addKv(LlmKv.EXPERT_COUNT, hparams.nExpert)
        addKv(LlmKv.EXPERT_USED_COUNT, hparams.nExpertUsed)
        addKv(LlmKv.EXPERT_SHARED_COUNT, hparams.nExpertShared)
        addKv(LlmKv.EXPERT_GROUP_COUNT, hparams.nExpertGroups)
        addKv(LlmKv.EXPERT_GROUP_USED_COUNT, hparams.nGroupUsed)
        addKv(LlmKv.EXPERT_WEIGHTS_SCALE, hparams.expertWeightsScale)
        addKv(LlmKv.EXPERT_WEIGHTS_NORM, hparams.expertWeightsNorm)
        addKv(LlmKv.EXPERT_GATING_FUNC, hparams.expertGatingFunc)
        addKv(LlmKv.EXPERT_GROUP_SCALE, hparams.expertGroupScale)
        addKv(LlmKv.EXPERTS_PER_GROUP, hparams.nGroupExperts)
        addKv(LlmKv.MOE_EVERY_N_LAYERS, hparams.moeEveryNLayers)
        addKv(LlmKv.NEXTN_PREDICT_LAYERS, hparams.nextnPredictLayers)
        addKv(LlmKv.NUM_DEEPSTACK_LAYERS, hparams.nDeepstackLayers)
        addKv(LlmKv.POOLING_TYPE, hparams.poolingType.ordinal)
        addKv(LlmKv.LOGIT_SCALE, hparams.fLogitScale)
        addKv(LlmKv.DECODER_START_TOKEN_ID, hparams.decStartTokenId)
        addKv(LlmKv.DECODER_BLOCK_COUNT, hparams.decNLayer)
        addKv(LlmKv.ATTN_LOGIT_SOFTCAPPING, hparams.fAttnLogitSoftcapping)
        addKv(LlmKv.ROUTER_LOGIT_SOFTCAPPING, hparams.fRouterLogitSoftcapping)
        addKv(LlmKv.FINAL_LOGIT_SOFTCAPPING, hparams.fFinalLogitSoftcapping)
        addKv(LlmKv.SWIN_NORM, hparams.swinNorm)
        addKv(LlmKv.RESCALE_EVERY_N_LAYERS, hparams.rescaleEveryNLayers)
        addKv(LlmKv.TIME_MIX_EXTRA_DIM, hparams.timeMixExtraDim)
        addKv(LlmKv.TIME_DECAY_EXTRA_DIM, hparams.timeDecayExtraDim)
        addKv(LlmKv.RESIDUAL_SCALE, hparams.fResidualScale)
        addKv(LlmKv.EMBEDDING_SCALE, hparams.fEmbeddingScale)
        addKv(LlmKv.TOKEN_SHIFT_COUNT, hparams.tokenShiftCount)
        addKv(LlmKv.INTERLEAVE_MOE_LAYER_STEP, hparams.nMoeLayerStep)

        // -- attention --
        addKvIntArray(LlmKv.ATTENTION_HEAD_COUNT, hparams.nHeadArr, perLayer = true)
        addKvIntArray(LlmKv.ATTENTION_HEAD_COUNT_KV, hparams.nHeadKvArr, perLayer = true)
        addKv(LlmKv.ATTENTION_MAX_ALIBI_BIAS, hparams.fMaxAlibiBias)
        addKv(LlmKv.ATTENTION_CLAMP_KQV, hparams.fClampKqv)
        addKv(LlmKv.ATTENTION_KEY_LENGTH, hparams.nEmbdHeadKFull)
        addKv(LlmKv.ATTENTION_VALUE_LENGTH, hparams.nEmbdHeadVFull)
        addKv(LlmKv.ATTENTION_LAYERNORM_EPS, hparams.fNormEps)
        addKv(LlmKv.ATTENTION_LAYERNORM_RMS_EPS, hparams.fNormRmsEps)
        addKv(LlmKv.ATTENTION_GROUPNORM_EPS, hparams.fNormGroupEps)
        addKv(LlmKv.ATTENTION_GROUPNORM_GROUPS, hparams.nNormGroups)
        addKv(LlmKv.ATTENTION_CAUSAL, hparams.causalAttn)
        addKv(LlmKv.ATTENTION_Q_LORA_RANK, hparams.nLoraQ)
        addKv(LlmKv.ATTENTION_KV_LORA_RANK, hparams.nLoraKv)
        addKv(LlmKv.ATTENTION_DECAY_LORA_RANK, hparams.nLoraDecay)
        addKv(LlmKv.ATTENTION_ICLR_LORA_RANK, hparams.nLoraIclr)
        addKv(LlmKv.ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK, hparams.nLoraValueResMix)
        addKv(LlmKv.ATTENTION_GATE_LORA_RANK, hparams.nLoraGate)
        addKv(LlmKv.ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.nRelAttnBkts)
        addKv(LlmKv.ATTENTION_SLIDING_WINDOW, hparams.nSwa)
        addKv(LlmKv.ATTENTION_SCALE, hparams.fAttentionScale)
        addKv(LlmKv.ATTENTION_OUTPUT_SCALE, hparams.fAttnOutScale)
        addKv(LlmKv.ATTENTION_TEMPERATURE_LENGTH, hparams.attnTempLength)
        addKv(LlmKv.ATTENTION_TEMPERATURE_SCALE, hparams.fAttnTempScale)
        addKv(LlmKv.ATTENTION_KEY_LENGTH_MLA, hparams.nEmbdHeadKMlaImpl)
        addKv(LlmKv.ATTENTION_VALUE_LENGTH_MLA, hparams.nEmbdHeadVMlaImpl)
        addKv(LlmKv.ATTENTION_KEY_LENGTH_SWA, hparams.nEmbdHeadKSwa)
        addKv(LlmKv.ATTENTION_VALUE_LENGTH_SWA, hparams.nEmbdHeadVSwa)
        addKv(LlmKv.ATTENTION_INDEXER_HEAD_COUNT, hparams.indexerNHead)
        addKv(LlmKv.ATTENTION_INDEXER_KEY_LENGTH, hparams.indexerHeadSize)
        addKv(LlmKv.ATTENTION_INDEXER_TOP_K, hparams.indexerTopK)

        // -- RoPE --
        val ropeScalingFactor = if (hparams.ropeFreqScaleTrain == 1.0f) 0.0f
            else 1.0f / hparams.ropeFreqScaleTrain

        addKv(LlmKv.ROPE_DIMENSION_COUNT, hparams.nRotFull)
        addKv(LlmKv.ROPE_DIMENSION_COUNT_SWA, hparams.nRotSwa)
        addKvIntArray(LlmKv.ROPE_DIMENSION_SECTIONS, hparams.ropeSections)
        addKv(LlmKv.ROPE_FREQ_BASE, hparams.ropeFreqBaseTrain)
        addKv(LlmKv.ROPE_FREQ_BASE_SWA, hparams.ropeFreqBaseTrainSwa)
        addKv(LlmKv.ROPE_SCALING_TYPE, hparams.ropeScalingTypeTrain.displayName)
        addKv(LlmKv.ROPE_SCALING_FACTOR, ropeScalingFactor)
        addKv(LlmKv.ROPE_SCALING_ATTN_FACTOR, hparams.ropeAttnFactor)
        addKv(LlmKv.ROPE_SCALING_ORIG_CTX_LEN, hparams.nCtxOrigYarn)
        addKv(LlmKv.ROPE_SCALING_FINETUNED, hparams.ropeFinetuned)
        addKv(LlmKv.ROPE_SCALING_YARN_LOG_MUL, hparams.ropeYarnLogMul)
        addKv(LlmKv.ROPE_SCALING_YARN_EXT_FACTOR, hparams.yarnExtFactor)
        addKv(LlmKv.ROPE_SCALING_YARN_ATTN_FACTOR, hparams.yarnAttnFactor)
        addKv(LlmKv.ROPE_SCALING_YARN_BETA_FAST, hparams.yarnBetaFast)
        addKv(LlmKv.ROPE_SCALING_YARN_BETA_SLOW, hparams.yarnBetaSlow)

        // -- SSM --
        addKv(LlmKv.SSM_INNER_SIZE, hparams.ssmDInner)
        addKv(LlmKv.SSM_CONV_KERNEL, hparams.ssmDConv)
        addKv(LlmKv.SSM_STATE_SIZE, hparams.ssmDState)
        addKv(LlmKv.SSM_TIME_STEP_RANK, hparams.ssmDtRank)
        addKv(LlmKv.SSM_GROUP_COUNT, hparams.ssmNGroup)
        addKv(LlmKv.SSM_DT_B_C_RMS, hparams.ssmDtBcRms)

        addKv(LlmKv.KDA_HEAD_DIM, hparams.nEmbdHeadKda)
        addKv(LlmKv.WKV_HEAD_SIZE, hparams.wkvHeadSize)

        // -- PosNet / ConvNext --
        addKv(LlmKv.POSNET_EMBEDDING_LENGTH, hparams.posnet.nEmbd)
        addKv(LlmKv.POSNET_BLOCK_COUNT, hparams.posnet.nLayer)
        addKv(LlmKv.CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.nEmbd)
        addKv(LlmKv.CONVNEXT_BLOCK_COUNT, hparams.convnext.nLayer)

        // -- classifier labels --
        addKv(LlmKv.CLASSIFIER_OUTPUT_LABELS, m.classifierLabels)

        addKv(LlmKv.SHORTCONV_L_CACHE, hparams.nShortconvLCache)

        // -- xIELU --
        addKvFloatArray(LlmKv.XIELU_ALPHA_N, hparams.xieluAlphaN)
        addKvFloatArray(LlmKv.XIELU_ALPHA_P, hparams.xieluAlphaP)
        addKvFloatArray(LlmKv.XIELU_BETA, hparams.xieluBeta)
        addKvFloatArray(LlmKv.XIELU_EPS, hparams.xieluEps)

        // -- sentence-transformers dense --
        addKv(LlmKv.DENSE_2_FEAT_IN, hparams.dense2FeatIn)
        addKv(LlmKv.DENSE_2_FEAT_OUT, hparams.dense2FeatOut)
        addKv(LlmKv.DENSE_3_FEAT_IN, hparams.dense3FeatIn)
        addKv(LlmKv.DENSE_3_FEAT_OUT, hparams.dense3FeatOut)
    }

    /**
     * Add all weight tensors from the model.
     *
     * Port of `llama_model_saver::add_tensors_from_model()`.
     */
    fun addTensorsFromModel() {
        val m = requireNotNull(model) { "addTensorsFromModel requires a model" }

        // Global tensors (skip tok_embd if it aliases output)
        if (m.output != null && m.tokEmbd != null &&
            m.output!!.name != m.tokEmbd!!.name
        ) {
            addTensor(m.tokEmbd)
        }
        addTensor(m.typeEmbd)
        addTensor(m.posEmbd)
        addTensor(m.tokNorm)
        addTensor(m.tokNormB)
        addTensor(m.outputNorm)
        addTensor(m.outputNormB)
        addTensor(m.output)
        addTensor(m.outputB)
        addTensor(m.outputNormEnc)
        addTensor(m.cls)
        addTensor(m.clsB)
        addTensor(m.clsOut)
        addTensor(m.clsOutB)
        addTensor(m.clsNorm)

        // Per-layer tensors
        for (layer in m.layers) {
            addAllLayerTensors(layer)
        }
    }

    /**
     * Write the populated GGUF context to a file at [path].
     *
     * Port of `llama_model_saver::save(const std::string &)`.
     *
     * LATER: Implement actual GGUF binary serialisation via [GGUFFileWriter]
     *       once the writer utility is complete.
     */
    fun save(path: String) {
        GGUFFileWriter.write(ggufCtx, path)
    }

    // -- internal helpers -----------------------------------------------------

    /**
     * Iterate over all tensor-valued properties of a [LlamaModelLayer] and
     * add each non-null tensor to the GGUF output.
     *
     * The C++ version does a raw pointer walk (`reinterpret_cast` over the
     * struct's memory layout).  In Kotlin we list every tensor property
     * explicitly, which is safer and keeps the code readable.
     */
    private fun addAllLayerTensors(layer: LlamaModelLayer) {
        // Normalization
        addTensor(layer.attnNorm); addTensor(layer.attnNormB)
        addTensor(layer.attnNorm2); addTensor(layer.attnNorm2B)
        addTensor(layer.attnQNorm); addTensor(layer.attnQNormB)
        addTensor(layer.attnKNorm); addTensor(layer.attnKNormB)
        addTensor(layer.attnOutNorm); addTensor(layer.attnOutNormB)
        addTensor(layer.attnQANorm); addTensor(layer.attnKvANorm)
        addTensor(layer.attnSubNorm); addTensor(layer.attnPostNorm)
        addTensor(layer.ffnSubNorm)
        addTensor(layer.attnNormCross); addTensor(layer.attnNormEnc)
        addTensor(layer.ssmNorm); addTensor(layer.ssmDtNorm)
        addTensor(layer.ssmBNorm); addTensor(layer.ssmCNorm)

        // Attention
        addTensor(layer.wq); addTensor(layer.wk); addTensor(layer.wv); addTensor(layer.wo)
        addTensor(layer.wqkv)
        addTensor(layer.wqA); addTensor(layer.wqB)
        addTensor(layer.wkvAMqa); addTensor(layer.wkvB)
        addTensor(layer.wkB); addTensor(layer.wvB)
        addTensor(layer.wqCross); addTensor(layer.wkCross); addTensor(layer.wvCross); addTensor(layer.woCross)
        addTensor(layer.wqEnc); addTensor(layer.wkEnc); addTensor(layer.wvEnc); addTensor(layer.woEnc)
        addTensor(layer.wqkvGate)

        // Attention bias
        addTensor(layer.bq); addTensor(layer.bk); addTensor(layer.bv); addTensor(layer.bo)
        addTensor(layer.bqkv)

        // Relative position bias
        addTensor(layer.attnRelB); addTensor(layer.attnRelBEnc); addTensor(layer.attnRelBCross)

        // FFN normalization
        addTensor(layer.ffnNorm); addTensor(layer.ffnNormB)
        addTensor(layer.ffnPostNorm); addTensor(layer.ffnPostNorm1); addTensor(layer.ffnPostNorm2)
        addTensor(layer.ffnPreNorm2)
        addTensor(layer.layerOutNorm); addTensor(layer.layerOutNormB)
        addTensor(layer.ffnNormExps); addTensor(layer.ffnNormEnc)

        // Feed-forward
        addTensor(layer.ffnGate); addTensor(layer.ffnDown); addTensor(layer.ffnUp)
        addTensor(layer.ffnGateEnc); addTensor(layer.ffnDownEnc); addTensor(layer.ffnUpEnc)

        // MoE
        addTensor(layer.ffnGateInp); addTensor(layer.ffnGateInpS)
        addTensor(layer.ffnGateExps); addTensor(layer.ffnDownExps); addTensor(layer.ffnUpExps)
        addTensor(layer.ffnGateUpExps)
        addTensor(layer.ffnGateInpB)
        addTensor(layer.ffnGateExpsB); addTensor(layer.ffnDownExpsB); addTensor(layer.ffnUpExpsB)
        addTensor(layer.ffnGateUpExpsB)
        addTensor(layer.ffnGateExpsS); addTensor(layer.ffnDownExpsS); addTensor(layer.ffnUpExpsS)
        addTensor(layer.ffnLatentDown); addTensor(layer.ffnLatentUp)

        // Shared experts
        addTensor(layer.ffnGateInpShexp)
        addTensor(layer.ffnGateShexp); addTensor(layer.ffnDownShexp); addTensor(layer.ffnUpShexp)

        // Adjugate experts (chexps)
        addTensor(layer.ffnGateChexps); addTensor(layer.ffnDownChexps); addTensor(layer.ffnUpChexps)

        // FFN bias
        addTensor(layer.ffnGateB); addTensor(layer.ffnDownB); addTensor(layer.ffnUpB)
        addTensor(layer.ffnAct); addTensor(layer.ffnExpProbsB)

        // SSM / Mamba
        addTensor(layer.ssmIn); addTensor(layer.ssmX); addTensor(layer.ssmDt); addTensor(layer.ssmOut)
        addTensor(layer.ssmConv1d); addTensor(layer.ssmA); addTensor(layer.ssmD)
        addTensor(layer.ssmConv1dB); addTensor(layer.ssmDtB)
        addTensor(layer.ssmBetaAlpha); addTensor(layer.ssmAlpha)

        // RWKV
        addTensor(layer.timeMixW1); addTensor(layer.timeMixW2)
        addTensor(layer.timeMixLerpX); addTensor(layer.timeMixLerpW)
        addTensor(layer.timeMixLerpK); addTensor(layer.timeMixLerpV)
        addTensor(layer.timeMixLerpR); addTensor(layer.timeMixLerpG)
        addTensor(layer.timeMixLerpFused)
        addTensor(layer.timeMixFirst); addTensor(layer.timeMixDecay)
        addTensor(layer.timeMixDecayW1); addTensor(layer.timeMixDecayW2)
        addTensor(layer.timeMixKey); addTensor(layer.timeMixKeyB)
        addTensor(layer.timeMixValue); addTensor(layer.timeMixValueB)
        addTensor(layer.timeMixReceptance); addTensor(layer.timeMixReceptanceB)
        addTensor(layer.timeMixGate)

        // RWKV7
        addTensor(layer.timeMixW0)
        addTensor(layer.timeMixA0); addTensor(layer.timeMixA1); addTensor(layer.timeMixA2)
        addTensor(layer.timeMixV0); addTensor(layer.timeMixV1); addTensor(layer.timeMixV2)
        addTensor(layer.timeMixG1); addTensor(layer.timeMixG2)
        addTensor(layer.timeMixKK); addTensor(layer.timeMixKA)
        addTensor(layer.timeMixRK); addTensor(layer.timeMixLn); addTensor(layer.timeMixLnB)
        addTensor(layer.timeMixOutput)

        addTensor(layer.channelMixLerpK); addTensor(layer.channelMixLerpR)
        addTensor(layer.channelMixKey); addTensor(layer.channelMixReceptance)
        addTensor(layer.channelMixValue)

        // Long rope factors
        addTensor(layer.ropeLong); addTensor(layer.ropeShort); addTensor(layer.ropeFreqs)

        // Bitnet scale
        addTensor(layer.wqS); addTensor(layer.wkS); addTensor(layer.wvS); addTensor(layer.woS)
        addTensor(layer.wqkvS); addTensor(layer.wqkvGateS)
        addTensor(layer.ffnGateS); addTensor(layer.ffnUpS); addTensor(layer.ffnDownS)
        addTensor(layer.ffnGateShexpS); addTensor(layer.ffnUpShexpS); addTensor(layer.ffnDownShexpS)
        addTensor(layer.ssmInS); addTensor(layer.ssmOutS)
        addTensor(layer.ssmAlphaS); addTensor(layer.ssmBetaS)

        // Input scales
        addTensor(layer.wqInS); addTensor(layer.wkInS); addTensor(layer.wvInS); addTensor(layer.woInS)
        addTensor(layer.wqkvInS); addTensor(layer.wqkvGateInS)
        addTensor(layer.ffnGateInS); addTensor(layer.ffnUpInS); addTensor(layer.ffnDownInS)
        addTensor(layer.ffnGateExpsInS); addTensor(layer.ffnDownExpsInS); addTensor(layer.ffnUpExpsInS)
        addTensor(layer.ffnGateShexpInS); addTensor(layer.ffnUpShexpInS); addTensor(layer.ffnDownShexpInS)
        addTensor(layer.ssmInInS); addTensor(layer.ssmOutInS)
        addTensor(layer.ssmAlphaInS); addTensor(layer.ssmBetaInS)

        // Altup & Laurel
        addTensor(layer.perLayerInpGate); addTensor(layer.perLayerProj); addTensor(layer.perLayerPostNorm)
        addTensor(layer.altupCorrectCoef); addTensor(layer.altupCorrectScale)
        addTensor(layer.altupPredictCoef); addTensor(layer.altupRouter); addTensor(layer.altupRouterNorm)
        addTensor(layer.laurelL); addTensor(layer.laurelR); addTensor(layer.laurelPostNorm)

        // Misc
        addTensor(layer.attnSinks)
        addTensor(layer.visexpAttnWqkv); addTensor(layer.visexpAttnWo)
        addTensor(layer.visexpFfnGate); addTensor(layer.visexpFfnDown); addTensor(layer.visexpFfnUp)

        // xIELU activation
        addTensor(layer.ffnActAlphaN); addTensor(layer.ffnActAlphaP)
        addTensor(layer.ffnActBeta); addTensor(layer.ffnActEps)

        // Kimi KDA
        addTensor(layer.ssmQConv); addTensor(layer.ssmKConv); addTensor(layer.ssmVConv)
        addTensor(layer.ssmBeta); addTensor(layer.ssmGA); addTensor(layer.ssmGB)
        addTensor(layer.ssmONorm)

        // DSA indexer
        addTensor(layer.indexerKNorm); addTensor(layer.indexerKNormB)
        addTensor(layer.indexerProj)
        addTensor(layer.indexerAttnK); addTensor(layer.indexerAttnQB)

        // Gemma4
        addTensor(layer.outScale)

        // Composite sub-layers
        addPosnetTensors(layer.posnet)
        addConvnextTensors(layer.convnext)
        addShortconvTensors(layer.shortconv)
        addNextnTensors(layer.nextn)
    }

    private fun addPosnetTensors(p: LlamaLayerPosnet) {
        addTensor(p.norm1); addTensor(p.norm1B)
        addTensor(p.conv1); addTensor(p.conv1B)
        addTensor(p.norm2); addTensor(p.norm2B)
        addTensor(p.conv2); addTensor(p.conv2B)
        addTensor(p.attnNorm); addTensor(p.attnNormB)
        addTensor(p.attnQ); addTensor(p.attnQB)
        addTensor(p.attnK); addTensor(p.attnKB)
        addTensor(p.attnV); addTensor(p.attnVB)
        addTensor(p.attnO); addTensor(p.attnOB)
        addTensor(p.norm); addTensor(p.normB)
    }

    private fun addConvnextTensors(c: LlamaLayerConvnext) {
        addTensor(c.dw); addTensor(c.dwB)
        addTensor(c.norm); addTensor(c.normB)
        addTensor(c.pw1); addTensor(c.pw1B)
        addTensor(c.pw2); addTensor(c.pw2B)
        addTensor(c.gamma)
    }

    private fun addShortconvTensors(s: LlamaLayerShortconv) {
        addTensor(s.inProj); addTensor(s.conv); addTensor(s.outProj)
    }

    private fun addNextnTensors(n: LlamaLayerNextn) {
        addTensor(n.ehProj); addTensor(n.embedTokens)
        addTensor(n.enorm); addTensor(n.hnorm)
        addTensor(n.sharedHeadHead); addTensor(n.sharedHeadNorm)
    }
}

// ---------------------------------------------------------------------------
// GGUFFileWriter – serialise a GGUFWriteContext to bytes / file
// ---------------------------------------------------------------------------

/**
 * Writes a [GGUFWriteContext] to a GGUF v3 binary file.
 *
 * LATER: complete binary serialisation (header, KV section, tensor-info
 *       section, tensor data with alignment padding).  The skeleton below
 *       lays out the structure so that callers can integrate immediately.
 */
object GGUFFileWriter {

    private const val GGUF_MAGIC = 0x46554747  // "GGUF" in little-endian
    private const val GGUF_VERSION = 3
    private const val DEFAULT_ALIGNMENT = 32

    /**
     * Serialise [ctx] and write the result to a file at [path].
     */
    fun write(ctx: GGUFWriteContext, path: String) {
        val bytes = serialise(ctx)
        // Kotlin/Native file I/O via platform.posix
        writeBytesToFile(bytes, path)
    }

    /**
     * Serialise [ctx] to a [ByteArray] containing a complete GGUF v3 file.
     */
    fun serialise(ctx: GGUFWriteContext): ByteArray {
        // Step 1: build header + KV + tensor info into a growable buffer
        val buf = GrowableByteBuffer()

        // -- header --
        buf.putInt(GGUF_MAGIC)
        buf.putInt(GGUF_VERSION)
        buf.putLong(ctx.tensorEntries.size.toLong())  // n_tensors
        buf.putLong(ctx.kvEntries.size.toLong())       // n_kv

        // -- KV pairs --
        for (kv in ctx.kvEntries) {
            writeKvEntry(buf, kv)
        }

        // -- tensor info --
        // We need to compute data offsets after we know total header size
        val headerSize = buf.size
        // Align tensor data start
        val dataStartUnaligned = headerSize + ctx.tensorEntries.sumOf { tensorInfoSize(it) }
        val dataStart = align(dataStartUnaligned, DEFAULT_ALIGNMENT)

        var dataOffset = 0L
        for (entry in ctx.tensorEntries) {
            writeTensorInfo(buf, entry, dataOffset)
            val tensorBytes = entry.dataBytes.size.toLong()
            dataOffset += align(tensorBytes.toInt(), DEFAULT_ALIGNMENT)
        }

        // -- padding to alignment before tensor data --
        val paddingNeeded = dataStart - buf.size
        for (i in 0 until paddingNeeded) buf.putByte(0)

        // -- tensor data --
        for (entry in ctx.tensorEntries) {
            buf.putBytes(entry.dataBytes)
            val remainder = entry.dataBytes.size % DEFAULT_ALIGNMENT
            if (remainder != 0) {
                for (i in 0 until DEFAULT_ALIGNMENT - remainder) buf.putByte(0)
            }
        }

        return buf.toByteArray()
    }

    // -- internal helpers --

    private fun writeKvEntry(buf: GrowableByteBuffer, kv: GGUFKvEntry) {
        writeString(buf, kv.key)
        buf.putInt(kv.type.value)  // type tag
        when (kv.type) {
            GGUFType.UINT32  -> buf.putInt((kv.value as UInt).toInt())
            GGUFType.INT32   -> buf.putInt(kv.value as Int)
            GGUFType.FLOAT32 -> buf.putFloat(kv.value as Float)
            GGUFType.BOOL    -> buf.putByte(if (kv.value as Boolean) 1 else 0)
            GGUFType.STRING  -> {
                val v = kv.value
                if (v is String) {
                    writeString(buf, v)
                } else if (v is List<*>) {
                    // array of strings
                    buf.putInt(GGUFType.STRING.value)
                    buf.putLong(v.size.toLong())
                    for (s in v) {
                        writeString(buf, s as? String ?: error("Unsupported string array element: $s"))
                    }
                }
            }
            else -> {
                // Array types
                val arr = kv.value
                when (arr) {
                    is IntArray -> {
                        buf.putInt(kv.type.value)
                        buf.putLong(arr.size.toLong())
                        for (v in arr) buf.putInt(v)
                    }
                    is FloatArray -> {
                        buf.putInt(kv.type.value)
                        buf.putLong(arr.size.toLong())
                        for (v in arr) buf.putFloat(v)
                    }
                    else -> error("Unsupported KV value type: ${arr::class}")
                }
            }
        }
    }

    private fun writeTensorInfo(buf: GrowableByteBuffer, entry: GGUFTensorEntry, dataOffset: Long) {
        writeString(buf, entry.name)
        buf.putInt(entry.nDims)
        for (i in 0 until entry.nDims) {
            buf.putLong(entry.ne[i])
        }
        buf.putInt(entry.type.ordinal)
        buf.putLong(dataOffset)
    }

    private fun tensorInfoSize(entry: GGUFTensorEntry): Int {
        // string(name) + u32(n_dims) + n_dims*u64 + u32(type) + u64(offset)
        return stringSize(entry.name) + 4 + entry.nDims * 8 + 4 + 8
    }

    private fun writeString(buf: GrowableByteBuffer, str: String) {
        val bytes = str.encodeToByteArray()
        buf.putLong(bytes.size.toLong())
        buf.putBytes(bytes)
    }

    private fun stringSize(str: String): Int {
        return 8 + str.encodeToByteArray().size  // u64 len + bytes
    }

    private fun align(offset: Int, alignment: Int): Int {
        val remainder = offset % alignment
        return if (remainder == 0) offset else offset + (alignment - remainder)
    }

    private fun align(offset: Long, alignment: Int): Long {
        val remainder = (offset % alignment).toInt()
        return if (remainder == 0) offset else offset + (alignment - remainder)
    }

    private fun writeBytesToFile(bytes: ByteArray, path: String) {
        nativeWriteBytesToFile(bytes, path)
    }
}

// ---------------------------------------------------------------------------
// GrowableByteBuffer – simple resizable byte buffer
// ---------------------------------------------------------------------------

/**
 * A minimal growable byte buffer for building binary data.
 * Kotlin/Native doesn't have `java.nio.ByteBuffer`, so we roll our own.
 */
internal class GrowableByteBuffer(initialCapacity: Int = 4096) {
    private var data = ByteArray(initialCapacity)
    var size: Int = 0
        private set

    private fun ensureCapacity(needed: Int) {
        val required = size + needed
        if (required <= data.size) return
        var newCap = data.size * 2
        while (newCap < required) newCap *= 2
        data = data.copyOf(newCap)
    }

    fun putByte(b: Int) {
        ensureCapacity(1)
        data[size++] = b.toByte()
    }

    fun putInt(v: Int) {
        ensureCapacity(4)
        data[size++] = (v and 0xFF).toByte()
        data[size++] = ((v shr 8) and 0xFF).toByte()
        data[size++] = ((v shr 16) and 0xFF).toByte()
        data[size++] = ((v shr 24) and 0xFF).toByte()
    }

    fun putLong(v: Long) {
        ensureCapacity(8)
        for (i in 0 until 8) {
            data[size++] = ((v shr (i * 8)) and 0xFF).toByte()
        }
    }

    fun putFloat(v: Float) {
        putInt(v.toRawBits())
    }

    fun putBytes(bytes: ByteArray) {
        ensureCapacity(bytes.size)
        bytes.copyInto(data, size)
        size += bytes.size
    }

    fun toByteArray(): ByteArray = data.copyOf(size)
}
