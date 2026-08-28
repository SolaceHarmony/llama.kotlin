// port-lint: source llama.cpp/src/llama-ext.h
/**
 * Extension / low-level API surface ported from `llama-ext.h`.
 *
 * These functions provide graph reservation, quantization state management,
 * and tensor-level quantization decisions used by the model quantizer.
 */
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// ---------------------------------------------------------------------------
// Graph reservation
// ---------------------------------------------------------------------------

/**
 * Reserve a new compute graph. Valid until the next call to [llamaGraphReserve].
 *
 * Port of `llama_graph_reserve()`.
 */
fun llamaGraphReserve(
    ctx: LlamaContext,
    nTokens: UInt,
    nSeqs: UInt,
    nOutputs: UInt,
): GGMLCGraph {
    error("llamaBuildGraph not yet ported")
}

// ---------------------------------------------------------------------------
// Default type for ftype
// ---------------------------------------------------------------------------

/**
 * Get the default [GGMLType] for a given file type ([LlamaFType]).
 *
 * Port of `llama_ftype_get_default_type()`.
 */
fun llamaFtypeGetDefaultType(ftype: LlamaFType): GGMLType {
    return when (ftype) {
        LlamaFType.ALL_F32          -> GGMLType.F32
        LlamaFType.MOSTLY_F16       -> GGMLType.F16
        LlamaFType.MOSTLY_Q4_0      -> GGMLType.Q4_0
        LlamaFType.MOSTLY_Q4_1      -> GGMLType.Q4_1
        LlamaFType.MOSTLY_Q8_0      -> GGMLType.Q8_0
        LlamaFType.MOSTLY_Q5_0      -> GGMLType.Q5_0
        LlamaFType.MOSTLY_Q5_1      -> GGMLType.Q5_1
        LlamaFType.MOSTLY_Q2_K      -> GGMLType.Q2_K
        LlamaFType.MOSTLY_Q3_K_S,
        LlamaFType.MOSTLY_Q3_K_M,
        LlamaFType.MOSTLY_Q3_K_L    -> GGMLType.Q3_K
        LlamaFType.MOSTLY_Q4_K_S,
        LlamaFType.MOSTLY_Q4_K_M    -> GGMLType.Q4_K
        LlamaFType.MOSTLY_Q5_K_S,
        LlamaFType.MOSTLY_Q5_K_M    -> GGMLType.Q5_K
        LlamaFType.MOSTLY_Q6_K      -> GGMLType.Q6_K
        LlamaFType.MOSTLY_BF16      -> GGMLType.BF16
        LlamaFType.MOSTLY_TQ1_0     -> GGMLType.TQ1_0
        LlamaFType.MOSTLY_TQ2_0     -> GGMLType.TQ2_0
        else -> GGMLType.Q4_0 // fallback
    }
}

// ---------------------------------------------------------------------------
// LlamaFType — model file quantization types
// ---------------------------------------------------------------------------

/**
 * File-level quantization type identifiers.
 *
 * Port of the `llama_ftype` enum from `llama.h`.
 */
enum class LlamaFType(val value: Int) {
    ALL_F32              (0),
    MOSTLY_F16           (1),
    MOSTLY_Q4_0          (2),
    MOSTLY_Q4_1          (3),
    // MOSTLY_Q4_1_SOME_F16 (4), // removed
    // MOSTLY_Q4_2          (5), // removed
    // MOSTLY_Q4_3          (6), // removed
    MOSTLY_Q8_0          (7),
    MOSTLY_Q5_0          (8),
    MOSTLY_Q5_1          (9),
    MOSTLY_Q2_K          (10),
    MOSTLY_Q3_K_S        (11),
    MOSTLY_Q3_K_M        (12),
    MOSTLY_Q3_K_L        (13),
    MOSTLY_Q4_K_S        (14),
    MOSTLY_Q4_K_M        (15),
    MOSTLY_Q5_K_S        (16),
    MOSTLY_Q5_K_M        (17),
    MOSTLY_Q6_K          (18),
    MOSTLY_IQ2_XXS       (19),
    MOSTLY_IQ2_XS        (20),
    MOSTLY_Q2_K_S        (21),
    MOSTLY_IQ3_XS        (22),
    MOSTLY_IQ3_XXS       (23),
    MOSTLY_IQ1_S         (24),
    MOSTLY_IQ4_NL        (25),
    MOSTLY_IQ3_S         (26),
    MOSTLY_IQ3_M         (27),
    MOSTLY_IQ2_S         (28),
    MOSTLY_IQ2_M         (29),
    MOSTLY_IQ4_XS        (30),
    MOSTLY_IQ1_M         (31),
    MOSTLY_BF16          (32),
    MOSTLY_Q4_0_4_4      (33),
    MOSTLY_Q4_0_4_8      (34),
    MOSTLY_Q4_0_8_8      (35),
    MOSTLY_TQ1_0         (36),
    MOSTLY_TQ2_0         (37),
    MOSTLY_MXFP4_MOE     (38),
    MOSTLY_NVFP4         (39),
    MOSTLY_Q1_0          (40),

    GUESSED              (1024);

    companion object {
        private val byValue = entries.associateBy { it.value }
        fun fromValue(v: Int): LlamaFType = byValue[v] ?: GUESSED
    }
}

// ---------------------------------------------------------------------------
// Quantization state
// ---------------------------------------------------------------------------

/**
 * Opaque quantization state — wraps per-tensor decisions and imatrix data.
 *
 * Port of `quantize_state_impl` from llama-quant.cpp.
 */
class QuantizeStateImpl(
    val model: LlamaModelData,
    val params: LlamaModelQuantizeParams,
) {
    /** Per-tensor override map: tensor name → forced GGMLType. */
    val tensorTypeOverrides: MutableMap<String, GGMLType> = mutableMapOf()
}

/**
 * Parameters for model quantization.
 *
 * Port of `llama_model_quantize_params` from `llama.h`.
 */
data class LlamaModelQuantizeParams(
    val nthread: Int = 0,
    val ftype: LlamaFType = LlamaFType.MOSTLY_Q5_1,
    val outputTensorType: GGMLType = GGMLType.COUNT, // COUNT = use default
    val tokenEmbeddingType: GGMLType = GGMLType.COUNT,
    val allowRequantize: Boolean = false,
    val quantizeOutputTensor: Boolean = true,
    val onlyCopy: Boolean = false,
    val pure: Boolean = false,
    val keepSplit: Boolean = false,
    val dryRun: Boolean = false,
    val imatrix: List<LlamaModelImatrixData>? = null,
    val kvOverrides: List<LlamaModelKvOverride>? = null,
    val ttOverrides: List<LlamaModelTensorOverride>? = null,
    val pruneLayers: IntArray? = null,
)

/**
 * Descriptor for constructing a mock model for quantization testing.
 *
 * Port of `llama_quant_model_desc` from `llama-ext.h`.
 */
data class LlamaQuantModelDesc(
    val architecture: String,
    val nEmbd: UInt,
    val nFf: UInt,
    val nLayer: UInt,
    val nHead: UInt,
    val nHeadKv: UInt,
    val nExpert: UInt,
    val nEmbdHeadK: UInt,
    val nEmbdHeadV: UInt,
)

// ---------------------------------------------------------------------------
// Quantization API functions
// ---------------------------------------------------------------------------

/**
 * Initialize quantization state.
 *
 * Port of `llama_quant_init()`.
 */
fun llamaQuantInit(
    model: LlamaModelData,
    params: LlamaModelQuantizeParams,
): QuantizeStateImpl = QuantizeStateImpl(model, params)

/** Free quantization state. Port of `llama_quant_free()`. */
fun llamaQuantFree(qs: QuantizeStateImpl) {
    // No-op in Kotlin — GC handles cleanup
}

/**
 * Returns true if this tensor should be quantized (based on name, dims, params).
 *
 * Port of `llama_quant_tensor_allows_quantization()`.
 */
fun llamaQuantTensorAllowsQuantization(
    qs: QuantizeStateImpl,
    tensor: GGMLTensor,
): Boolean {
    // 1D tensors (biases) and tensors with fewer than 32 elements are never quantized
    val nDims = tensor.ne.indexOfLast { it > 1 } + 1
    if (nDims <= 1) return false
    val totalElements = tensor.numElements()
    if (totalElements < 32) return false

    val name = tensor.name
    // Output tensors: only if explicitly enabled
    if (name.contains("output") && !qs.params.quantizeOutputTensor) return false

    return true
}

/**
 * Compute quantization type assignments for a list of tensors.
 *
 * Port of `llama_quant_compute_types()`.
 */
fun llamaQuantComputeTypes(
    qs: QuantizeStateImpl,
    ftype: LlamaFType,
    tensors: List<GGMLTensor>,
): List<GGMLType> {
    val defaultType = llamaFtypeGetDefaultType(ftype)
    return tensors.map { tensor ->
        qs.tensorTypeOverrides[tensor.name] ?: defaultType
    }
}
