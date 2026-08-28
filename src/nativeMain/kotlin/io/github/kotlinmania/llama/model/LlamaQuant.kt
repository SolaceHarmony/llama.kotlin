// port-lint: source llama.cpp/src/llama-quant.h llama.cpp/src/llama-quant.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.max

// =============================================================================
// Tensor type override  (--tensor-type CLI option)
// =============================================================================

/**
 * User-specified override mapping a tensor name pattern to a target
 * [GGMLType].
 *
 * Mirrors `tensor_type_option` in the C++ source.
 *
 * @param name The tensor name (or regex pattern) to match.
 * @param type The desired quantisation type for matching tensors.
 */
data class TensorTypeOption(
    val name: String,
    val type: GGMLType = GGMLType.COUNT,
)

// =============================================================================
// Tensor categorisation
// =============================================================================

/**
 * Broad tensor category used by the quantisation logic to decide the target
 * type.  This is intentionally coarser than the per-architecture tensor name
 * enum (`LLM_TN`).
 *
 * Mirrors `tensor_category` in the C++ source.
 */
enum class TensorCategory {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER,
}

// =============================================================================
// Per-tensor metadata (computed once, used during the main loop)
// =============================================================================

/**
 * Metadata for a single tensor in the quantisation pipeline.
 *
 * Mirrors `tensor_metadata` in the C++ source.
 */
data class QuantTensorMetadata(
    var name: String = "",
    var targetType: GGMLType = GGMLType.F32,
    var category: TensorCategory = TensorCategory.OTHER,
    var remappedImatrixName: String = "",
    var allowsQuantization: Boolean = true,
    var requiresImatrix: Boolean = false,
)

// =============================================================================
// Quantisation state
// =============================================================================

/**
 * Mutable state tracked during the quantisation pass.
 *
 * Mirrors `quantize_state_impl` in the C++ source.  The C++ version holds
 * a reference to the full model; here we accept the minimal model properties
 * needed by the type-selection logic.
 *
 * This is a richer representation than [QuantizeStateImpl] in LlamaExt.kt:
 * it encodes the counters and model metadata needed by the per-layer
 * heuristics without requiring the full model object.
 *
 * @param arch               Model architecture (e.g. [LlamaModelArch.LLAMA]).
 * @param modelType          Model size variant (e.g. [LlamaModelType.TYPE_70B]).
 * @param nLayer             Number of transformer layers.
 * @param nExpert            Number of MoE experts.
 * @param nGqa               Grouped-Query Attention head ratio.
 * @param params             Quantisation parameters.
 */
class QuantizeState(
    val arch: LlamaModelArch,
    val modelType: LlamaModelType,
    val nLayer: Int,
    val nExpert: Int,
    val nGqa: Int,
    val params: LlamaModelQuantizeParams,
) {
    var nAttentionWv: Int = 0
    var nFfnDown: Int = 0
    var nFfnGate: Int = 0
    var nFfnUp: Int = 0
    var iAttentionWv: Int = 0
    var iFfnDown: Int = 0
    var iFfnGate: Int = 0
    var iFfnUp: Int = 0

    var nFallback: Int = 0
    var hasImatrix: Boolean = false
    var hasTiedEmbeddings: Boolean = true

    // Compiled regex patterns for tensor-type overrides
    val tensorTypePatterns: List<Pair<Regex, GGMLType>> = emptyList()

    /** Reset per-computation counters (used by [LlamaQuant.quantComputeTypes]). */
    fun resetCounters() {
        nAttentionWv = 0; nFfnDown = 0; nFfnGate = 0; nFfnUp = 0
        iAttentionWv = 0; iFfnDown = 0; iFfnGate = 0; iFfnUp = 0
        nFallback = 0; hasImatrix = false; hasTiedEmbeddings = true
    }
}

// =============================================================================
// LlamaQuant — main quantisation driver
// =============================================================================

/**
 * Model quantisation utilities.
 *
 * This object ports the logic from `llama-quant.cpp`, providing:
 * - Tensor categorisation by name pattern matching
 * - Per-category type selection with layer-aware heuristics
 * - Fallback type resolution when block sizes are incompatible
 * - The main quantisation driver (deferred – requires model-loader integration)
 *
 * The actual quantise/dequantise kernels live in `GGMLComputeOps` / `GGMLQuants`
 * in the core package.
 */
object LlamaQuant {

    // -------------------------------------------------------------------------
    // Default parameters
    // -------------------------------------------------------------------------

    /**
     * Returns the default quantisation parameters.
     *
     * Mirrors `llama_model_quantize_default_params()`.
     */
    fun modelQuantizeDefaultParams(): LlamaModelQuantizeParams = LlamaModelQuantizeParams()

    // -------------------------------------------------------------------------
    // Tensor name matching
    // -------------------------------------------------------------------------

    private fun tensorNameMatchTokenEmbd(name: String): Boolean =
        name == "token_embd.weight" || name == "per_layer_token_embd.weight"

    private fun tensorNameMatchOutputWeight(name: String): Boolean =
        name == "output.weight"

    // -------------------------------------------------------------------------
    // Tensor categorisation
    // -------------------------------------------------------------------------

    /**
     * Classify a tensor into a broad [TensorCategory] based on its name.
     *
     * The classification is intentionally coarse – it groups tensors that
     * receive the same quantisation treatment.
     */
    fun tensorGetCategory(tensorName: String): TensorCategory {
        if (tensorNameMatchOutputWeight(tensorName)) return TensorCategory.OUTPUT
        if (tensorNameMatchTokenEmbd(tensorName)) return TensorCategory.TOKEN_EMBD
        if ("attn_qkv.weight" in tensorName) return TensorCategory.ATTENTION_QKV
        if ("attn_kv_b.weight" in tensorName) return TensorCategory.ATTENTION_KV_B
        if ("attn_v.weight" in tensorName) return TensorCategory.ATTENTION_V
        if ("attn_k.weight" in tensorName) return TensorCategory.ATTENTION_K
        if ("attn_q.weight" in tensorName) return TensorCategory.ATTENTION_Q
        if ("attn_output.weight" in tensorName) return TensorCategory.ATTENTION_OUTPUT
        if ("ffn_up" in tensorName) return TensorCategory.FFN_UP
        if ("ffn_gate" in tensorName) return TensorCategory.FFN_GATE
        if ("ffn_down" in tensorName) return TensorCategory.FFN_DOWN
        return TensorCategory.OTHER
    }

    /**
     * Returns `true` for attention-V-like tensor categories (more sensitive
     * to quantisation).
     */
    fun categoryIsAttnV(cat: TensorCategory): Boolean =
        cat == TensorCategory.ATTENTION_V ||
        cat == TensorCategory.ATTENTION_QKV ||
        cat == TensorCategory.ATTENTION_KV_B

    // -------------------------------------------------------------------------
    // Quantisation gating
    // -------------------------------------------------------------------------

    /**
     * Returns `true` if this tensor is eligible for quantisation.
     *
     * Mirrors `tensor_allows_quantization()` in the C++ source. The check
     * combines structural rules (dimensionality, name patterns) with user
     * overrides.
     *
     * @param params Quantisation parameters.
     * @param tensorName Name of the tensor.
     * @param nDims Number of dimensions (only 2D+ tensors are quantised).
     */
    fun tensorAllowsQuantization(
        params: LlamaModelQuantizeParams,
        tensorName: String,
        nDims: Int,
    ): Boolean {
        if (params.onlyCopy) return false
        if (nDims < 2) return false

        var quantize = tensorName.endsWith("weight")

        // Do not quantise norm tensors
        quantize = quantize && "_norm.weight" !in tensorName

        quantize = quantize && (params.quantizeOutputTensor || tensorName != "output.weight")

        // Do not quantise expert gating tensors
        quantize = quantize && "ffn_gate_inp.weight" !in tensorName

        // Small or specialised tensors that should remain unquantised
        val excludePatterns = listOf(
            "altup", "laurel",
            "per_layer_model_proj",
            "ssm_conv1d", "shortconv.conv.weight",
            "time_mix_first.weight", "time_mix_w0.weight", "time_mix_w1.weight",
            "time_mix_w2.weight", "time_mix_v0.weight", "time_mix_v1.weight",
            "time_mix_v2.weight", "time_mix_a0.weight", "time_mix_a1.weight",
            "time_mix_a2.weight", "time_mix_g1.weight", "time_mix_g2.weight",
            "time_mix_decay_w1.weight", "time_mix_decay_w2.weight",
            "time_mix_lerp_fused.weight",
            "attn_rel_b.weight",
            ".position_embd", "sam.pos_embd", "sam.neck.", "sam.net_",
            ".rel_pos", ".patch_embd", ".patch_merger",
        )
        for (pattern in excludePatterns) {
            quantize = quantize && pattern !in tensorName
        }

        return quantize
    }

    // -------------------------------------------------------------------------
    // File-type → default GGML type
    // -------------------------------------------------------------------------

    /**
     * Returns the default [GGMLType] for a given [LlamaFType].
     *
     * Mirrors `llama_ftype_get_default_type()`.
     */
    fun ftypeGetDefaultType(ftype: LlamaFType): GGMLType {
        return when (ftype) {
            LlamaFType.MOSTLY_Q4_0 -> GGMLType.Q4_0
            LlamaFType.MOSTLY_Q4_1 -> GGMLType.Q4_1
            LlamaFType.MOSTLY_Q5_0 -> GGMLType.Q5_0
            LlamaFType.MOSTLY_Q5_1 -> GGMLType.Q5_1
            LlamaFType.MOSTLY_Q8_0 -> GGMLType.Q8_0
            LlamaFType.MOSTLY_F16 -> GGMLType.F16
            LlamaFType.MOSTLY_BF16 -> GGMLType.BF16
            LlamaFType.ALL_F32 -> GGMLType.F32
            LlamaFType.MOSTLY_Q2_K -> GGMLType.Q2_K
            LlamaFType.MOSTLY_Q3_K_S,
            LlamaFType.MOSTLY_Q3_K_M,
            LlamaFType.MOSTLY_Q3_K_L -> GGMLType.Q3_K
            LlamaFType.MOSTLY_Q4_K_S,
            LlamaFType.MOSTLY_Q4_K_M -> GGMLType.Q4_K
            LlamaFType.MOSTLY_Q5_K_S,
            LlamaFType.MOSTLY_Q5_K_M -> GGMLType.Q5_K
            LlamaFType.MOSTLY_Q6_K -> GGMLType.Q6_K
            else -> GGMLType.COUNT
        }
    }

    // -------------------------------------------------------------------------
    // Type fallback
    // -------------------------------------------------------------------------

    /**
     * Returns a compatible fallback [GGMLType] when [targetType]'s block size
     * does not evenly divide [nCols].
     *
     * Mirrors `tensor_type_fallback()`.
     *
     * @param qs         Current quantisation state (for fallback counting).
     * @param nCols      Number of columns in the tensor's innermost dimension.
     * @param targetType The originally selected quantisation type.
     * @return A compatible type, potentially [GGMLType.F16] as last resort.
     */
    fun tensorTypeFallback(
        qs: QuantizeState,
        nCols: Long,
        targetType: GGMLType,
    ): GGMLType {
        val qk = ggmlBlckSize(targetType)
        if (nCols % qk == 0L) return targetType

        qs.nFallback++
        val fallback = when (targetType) {
            GGMLType.IQ1_S, GGMLType.IQ1_M,
            GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S,
            GGMLType.IQ4_XS -> GGMLType.IQ4_NL
            GGMLType.Q2_K, GGMLType.Q3_K,
            GGMLType.TQ1_0, GGMLType.TQ2_0 -> GGMLType.Q4_0
            GGMLType.Q4_K -> GGMLType.Q5_0
            GGMLType.Q5_K -> GGMLType.Q5_1
            GGMLType.Q6_K -> GGMLType.Q8_0
            else -> throw IllegalStateException(
                "no tensor type fallback defined for ${targetType.description}"
            )
        }

        // If the fallback is still incompatible, use F16
        return if (nCols % ggmlBlckSize(fallback) != 0L) GGMLType.F16 else fallback
    }

    // -------------------------------------------------------------------------
    // Internal type selection (per-category heuristics)
    // -------------------------------------------------------------------------

    /**
     * Heuristic type selection based on tensor category, ftype, and model
     * architecture.
     *
     * Mirrors `llama_tensor_get_type_impl()`. This encodes the layer-aware
     * "use more bits for early/late layers" strategy as well as the
     * per-architecture special cases.
     *
     * @param qs          Quantisation state (counters, model info).
     * @param currentType The base quantisation type from the ftype.
     * @param tensorName  Tensor name (for layer-number parsing).
     * @param ftype       Target file type.
     * @param category    Tensor category.
     * @return The refined [GGMLType].
     */
    fun tensorGetTypeImpl(
        qs: QuantizeState,
        currentType: GGMLType,
        tensorName: String,
        ftype: LlamaFType,
        category: TensorCategory,
    ): GGMLType {
        var newType = currentType

        val nExpert = max(1, qs.nExpert)

        fun useMoreBits(iLayer: Int, nLayers: Int): Boolean =
            iLayer < nLayers / 8 || iLayer >= 7 * nLayers / 8 ||
                (iLayer - nLayers / 8) % 3 == 2

        fun layerInfo(iLayerInitial: Int, nLayer: Int): Pair<Int, Int> {
            var iLayer = iLayerInitial
            if (nExpert > 1) {
                val match = Regex("""blk\.(\d+)\.""").find(tensorName)
                    ?: throw IllegalStateException("Failed to determine layer for tensor $tensorName")
                iLayer = match.groupValues[1].toInt()
                require(iLayer in 0 until nLayer) {
                    "Bad layer $iLayer for tensor $tensorName. Must be in [0, $nLayer)"
                }
            }
            return iLayer to nLayer
        }

        // --- OUTPUT / tied TOKEN_EMBD ---
        if (category == TensorCategory.OUTPUT ||
            (qs.hasTiedEmbeddings && category == TensorCategory.TOKEN_EMBD)) {
            if (qs.params.outputTensorType != GGMLType.COUNT) {
                newType = qs.params.outputTensorType
            } else {
                when (ftype) {
                    LlamaFType.MOSTLY_Q2_K,
                    LlamaFType.MOSTLY_Q3_K_S,
                    LlamaFType.MOSTLY_Q3_K_M,
                    LlamaFType.MOSTLY_Q3_K_L -> newType = GGMLType.Q5_K
                    else -> if (newType != GGMLType.Q8_0) newType = GGMLType.Q6_K
                }
            }
        }
        // --- TOKEN_EMBD (non-tied) ---
        else if (category == TensorCategory.TOKEN_EMBD) {
            if (qs.params.tokenEmbeddingType != GGMLType.COUNT) {
                newType = qs.params.tokenEmbeddingType
            }
            // Specific ftype overrides omitted for brevity — see C++ source
        }
        // --- ATTENTION V / QKV / KV_B ---
        else if (categoryIsAttnV(category)) {
            when (ftype) {
                LlamaFType.MOSTLY_Q2_K -> {
                    newType = if (qs.nGqa >= 4) GGMLType.Q4_K else GGMLType.Q3_K
                }
                LlamaFType.MOSTLY_Q3_K_M -> {
                    newType = if (qs.iAttentionWv < 2) GGMLType.Q5_K else GGMLType.Q4_K
                }
                LlamaFType.MOSTLY_Q3_K_L -> newType = GGMLType.Q5_K
                LlamaFType.MOSTLY_Q4_K_M, LlamaFType.MOSTLY_Q5_K_M -> {
                    if (useMoreBits(qs.iAttentionWv, qs.nAttentionWv)) newType = GGMLType.Q6_K
                }
                LlamaFType.MOSTLY_Q4_K_S -> {
                    if (qs.iAttentionWv < 4) newType = GGMLType.Q5_K
                }
                else -> { /* keep default */ }
            }
            if (qs.modelType == LlamaModelType.TYPE_70B) {
                if (newType == GGMLType.Q3_K || newType == GGMLType.Q4_K) newType = GGMLType.Q5_K
            }
            if (qs.nExpert == 8) newType = GGMLType.Q8_0
            qs.iAttentionWv++
        }
        // --- ATTENTION K ---
        else if (category == TensorCategory.ATTENTION_K) {
            if (qs.nExpert == 8) newType = GGMLType.Q8_0
        }
        // --- FFN DOWN ---
        else if (category == TensorCategory.FFN_DOWN) {
            val (iLayer, nLayers) = layerInfo(qs.iFfnDown, qs.nFfnDown)
            when (ftype) {
                LlamaFType.MOSTLY_Q2_K -> newType = GGMLType.Q3_K
                LlamaFType.MOSTLY_Q3_K_M -> {
                    newType = when {
                        iLayer < nLayers / 16 -> GGMLType.Q5_K
                        qs.arch != LlamaModelArch.FALCON || useMoreBits(iLayer, nLayers) -> GGMLType.Q4_K
                        else -> GGMLType.Q3_K
                    }
                }
                LlamaFType.MOSTLY_Q3_K_L -> {
                    newType = if (qs.arch == LlamaModelArch.FALCON) GGMLType.Q4_K else GGMLType.Q5_K
                }
                LlamaFType.MOSTLY_Q4_K_M -> {
                    if (qs.arch == LlamaModelArch.FALCON) {
                        newType = when {
                            iLayer < nLayers / 16 -> GGMLType.Q6_K
                            useMoreBits(iLayer, nLayers) -> GGMLType.Q5_K
                            else -> GGMLType.Q4_K
                        }
                    } else {
                        if (useMoreBits(iLayer, nLayers)) newType = GGMLType.Q6_K
                    }
                }
                LlamaFType.MOSTLY_Q5_K_M -> {
                    if (useMoreBits(iLayer, nLayers)) newType = GGMLType.Q6_K
                }
                LlamaFType.MOSTLY_Q4_K_S -> {
                    if (qs.arch != LlamaModelArch.FALCON && iLayer < nLayers / 8) {
                        newType = GGMLType.Q5_K
                    }
                }
                else -> { /* keep default */ }
            }
            qs.iFfnDown++
        }
        // --- ATTENTION OUTPUT ---
        else if (category == TensorCategory.ATTENTION_OUTPUT) {
            if (qs.arch != LlamaModelArch.FALCON) {
                if (qs.nExpert == 8) {
                    when (ftype) {
                        LlamaFType.MOSTLY_Q2_K,
                        LlamaFType.MOSTLY_Q3_K_S, LlamaFType.MOSTLY_Q3_K_M,
                        LlamaFType.MOSTLY_Q4_K_S, LlamaFType.MOSTLY_Q4_K_M ->
                            newType = GGMLType.Q5_K
                        else -> { /* keep default */ }
                    }
                } else {
                    when (ftype) {
                        LlamaFType.MOSTLY_Q2_K -> newType = GGMLType.Q3_K
                        LlamaFType.MOSTLY_Q3_K_M -> newType = GGMLType.Q4_K
                        LlamaFType.MOSTLY_Q3_K_L -> newType = GGMLType.Q5_K
                        else -> { /* keep default */ }
                    }
                }
            } else {
                if (ftype == LlamaFType.MOSTLY_Q3_K_L) newType = GGMLType.Q4_K
            }
        }
        // --- ATTENTION QKV ---
        else if (category == TensorCategory.ATTENTION_QKV) {
            when (ftype) {
                LlamaFType.MOSTLY_Q3_K_M, LlamaFType.MOSTLY_Q3_K_L ->
                    newType = GGMLType.Q4_K
                LlamaFType.MOSTLY_Q4_K_M -> newType = GGMLType.Q5_K
                LlamaFType.MOSTLY_Q5_K_M -> newType = GGMLType.Q6_K
                else -> { /* keep default */ }
            }
        }
        // --- FFN GATE ---
        else if (category == TensorCategory.FFN_GATE) {
            val (_, _) = layerInfo(qs.iFfnGate, qs.nFfnGate)
            qs.iFfnGate++
        }
        // --- FFN UP ---
        else if (category == TensorCategory.FFN_UP) {
            val (_, _) = layerInfo(qs.iFfnUp, qs.nFfnUp)
            qs.iFfnUp++
        }

        return newType
    }

    // -------------------------------------------------------------------------
    // Outer type selection wrapper
    // -------------------------------------------------------------------------

    /**
     * Determine the [GGMLType] a tensor should be quantised to.
     *
     * Mirrors `llama_tensor_get_type()`. Combines gating, overrides,
     * heuristics, and fallback.
     *
     * @param qs           Quantisation state.
     * @param params       Quantisation parameters.
     * @param tensorName   Tensor name.
     * @param nDims        Number of tensor dimensions.
     * @param nCols        Number of columns (ne[0]).
     * @param currentType  The tensor's current type (in the source file).
     * @param defaultType  The default target type from the ftype.
     * @param metadata     Computed tensor metadata.
     * @return The target [GGMLType].
     */
    fun tensorGetType(
        qs: QuantizeState,
        params: LlamaModelQuantizeParams,
        tensorName: String,
        nDims: Int,
        nCols: Long,
        currentType: GGMLType,
        defaultType: GGMLType,
        metadata: QuantTensorMetadata,
    ): GGMLType {
        if (!tensorAllowsQuantization(params, tensorName, nDims)) {
            return currentType
        }
        if (params.tokenEmbeddingType != GGMLType.COUNT &&
            metadata.category == TensorCategory.TOKEN_EMBD) {
            return params.tokenEmbeddingType
        }
        if (params.outputTensorType != GGMLType.COUNT &&
            metadata.category == TensorCategory.OUTPUT) {
            return params.outputTensorType
        }

        var newType = defaultType

        if (!params.pure && ggmlIsQuantized(defaultType)) {
            // Check manual overrides
            var manual = false
            for ((pattern, qtype) in qs.tensorTypePatterns) {
                if (pattern.containsMatchIn(tensorName)) {
                    newType = qtype
                    manual = true
                    break
                }
            }

            if (!manual) {
                newType = tensorGetTypeImpl(qs, newType, tensorName, params.ftype, metadata.category)
            }

            newType = tensorTypeFallback(qs, nCols, newType)
        }

        return newType
    }

    // -------------------------------------------------------------------------
    // Importance-matrix requirement check
    // -------------------------------------------------------------------------

    /**
     * Returns `true` if the given tensor + target type combination requires
     * an importance matrix for acceptable quality.
     */
    fun tensorRequiresImatrix(
        tensorName: String,
        dstType: GGMLType,
        ftype: LlamaFType,
    ): Boolean {
        if (tensorNameMatchTokenEmbd(tensorName) || tensorNameMatchOutputWeight(tensorName)) {
            return false
        }
        return when (dstType) {
            GGMLType.IQ3_XXS, GGMLType.IQ2_XXS, GGMLType.IQ2_XS,
            GGMLType.IQ2_S, GGMLType.IQ1_M, GGMLType.IQ1_S -> true
            GGMLType.Q2_K -> ftype == LlamaFType.MOSTLY_Q2_K
            else -> false
        }
    }

    // -------------------------------------------------------------------------
    // Metadata initialisation
    // -------------------------------------------------------------------------

    /**
     * Populate category counters in [qs] and category fields in [metadata]
     * by scanning tensor names.
     *
     * Mirrors `init_quantize_state_counters()`.
     */
    fun initQuantizeStateCounters(
        qs: QuantizeState,
        metadata: List<QuantTensorMetadata>,
    ) {
        for (tm in metadata) {
            val cat = tensorGetCategory(tm.name)
            tm.category = cat

            if (categoryIsAttnV(cat)) qs.nAttentionWv++
            if (cat == TensorCategory.OUTPUT) qs.hasTiedEmbeddings = false
        }
        qs.nFfnDown = qs.nLayer
        qs.nFfnGate = qs.nLayer
        qs.nFfnUp = qs.nLayer
    }

    // -------------------------------------------------------------------------
    // Batch type computation
    // -------------------------------------------------------------------------

    /**
     * Compute the target quantisation type for a batch of tensors.
     *
     * Mirrors `llama_quant_compute_types()`. Resets internal counters,
     * initialises metadata, and then runs the type-selection logic for
     * each tensor.
     *
     * @param qs          Quantisation state.
     * @param ftype       Target file type.
     * @param tensorNames List of tensor names.
     * @param tensorDims  Parallel list: number of dimensions per tensor.
     * @param tensorCols  Parallel list: number of columns (ne[0]) per tensor.
     * @param tensorTypes Parallel list: current types per tensor.
     * @return List of target [GGMLType] values, one per tensor.
     */
    fun quantComputeTypes(
        qs: QuantizeState,
        ftype: LlamaFType,
        tensorNames: List<String>,
        tensorDims: List<Int>,
        tensorCols: List<Long>,
        tensorTypes: List<GGMLType>,
    ): List<GGMLType> {
        qs.resetCounters()

        val metadata = tensorNames.map { QuantTensorMetadata(name = it) }
        initQuantizeStateCounters(qs, metadata)

        val localParams = qs.params.copy(ftype = ftype)
        val defaultType = ftypeGetDefaultType(ftype)

        return tensorNames.indices.map { i ->
            tensorGetType(
                qs, localParams, tensorNames[i],
                tensorDims[i], tensorCols[i], tensorTypes[i],
                defaultType, metadata[i],
            )
        }
    }

    // -------------------------------------------------------------------------
    // Main quantisation driver (minimal)
    // -------------------------------------------------------------------------

    /**
     * Quantise a model file from [fnameInp] to [fnameOut].
     *
     * LATER: Implement the full quantisation pipeline once the model-loader and
     *       GGUF writer infrastructure is available. The C++ version
     *       (`llama_model_quantize_impl`) reads tensors, dequantises if needed,
     *       quantises to the target type, and writes a new GGUF file.
     *
     * @param fnameInp Path to the source GGUF model file.
     * @param fnameOut Path for the quantised output file.
     * @param params   Quantisation parameters.
     * @return 0 on success, non-zero on error.
     */
    fun modelQuantize(
        fnameInp: String,
        fnameOut: String,
        params: LlamaModelQuantizeParams,
    ): Int {
        error("quantizeImpl not yet ported")
    }

    // -------------------------------------------------------------------------
    // Helper: block size lookup
    // -------------------------------------------------------------------------

    /**
     * Returns the block size for a [GGMLType] (number of elements per
     * quantisation block).
     *
     * For non-quantised types (F32, F16, BF16) the block size is 1.
     */
    private fun ggmlBlckSize(type: GGMLType): Long {
        return when (type) {
            GGMLType.F32, GGMLType.F16, GGMLType.BF16,
            GGMLType.I8, GGMLType.I16, GGMLType.I32, GGMLType.I64, GGMLType.F64 -> 1L
            GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
            GGMLType.Q8_0, GGMLType.Q8_1 -> 32L
            GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K,
            GGMLType.Q6_K, GGMLType.Q8_K -> 256L
            GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
            GGMLType.IQ3_XXS, GGMLType.IQ3_S,
            GGMLType.IQ1_S, GGMLType.IQ1_M,
            GGMLType.IQ4_NL, GGMLType.IQ4_XS -> 256L
            GGMLType.TQ1_0, GGMLType.TQ2_0 -> 256L
            else -> 1L
        }
    }

    /**
     * Returns `true` if the type is a quantised integer type (not F32/F16/BF16/I*).
     */
    private fun ggmlIsQuantized(type: GGMLType): Boolean {
        return when (type) {
            GGMLType.F32, GGMLType.F16, GGMLType.BF16, GGMLType.F64,
            GGMLType.I8, GGMLType.I16, GGMLType.I32, GGMLType.I64,
            GGMLType.COUNT -> false
            else -> true
        }
    }
}
