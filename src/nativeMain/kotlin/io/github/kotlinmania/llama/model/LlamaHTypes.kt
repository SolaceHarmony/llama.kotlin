// port-lint: source llama.h
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// Types and constants from llama.h that are NOT already defined elsewhere.
//
// Items already ported live in:
//   LlamaApi.kt        — LlamaModelParams, LlamaSplitMode, LlamaSamplerChainParams, etc.
//   LlamaContext.kt    — LlamaContextParams, LlamaPerfContextData, LlamaMemoryBreakdownData, etc.
//   LlamaSampler.kt    — LlamaSampler interface, LlamaTokenDataArray, LlamaLogitBias, etc.
//   LlamaVocab.kt      — LlamaVocabType, LlamaTokenType, LlamaTokenAttr, LlamaTokenData, etc.
//   LlamaModel.kt      — LlamaRopeType, LlamaRopeScalingType, etc.
//   LlamaExt.kt        — LlamaFType, LlamaModelQuantizeParams, etc.
//   KVCache.kt          — LlamaPos, LlamaSeqId
//   LlamaMemoryHybrid.kt — LlamaStateSeqFlags
// =============================================================================

// ---------------------------------------------------------------------------
// Fundamental type aliases  (llama.h)
// ---------------------------------------------------------------------------

/** Token ID type. Maps to `typedef int32_t llama_token` in C++. */
typealias LlamaToken = Int

// ---------------------------------------------------------------------------
// Magic numbers & session constants  (llama.h lines 37–49)
// ---------------------------------------------------------------------------

/** File magic for LoRA adapters ('ggla'). */
const val LLAMA_FILE_MAGIC_GGLA: UInt = 0x6767_6C61u

/** File magic for session snapshots ('ggsn'). */
const val LLAMA_FILE_MAGIC_GGSN: UInt = 0x6767_736Eu

/** File magic for sequence snapshots ('ggsq'). */
const val LLAMA_FILE_MAGIC_GGSQ: UInt = 0x6767_7371u

/** Session file magic — alias for [LLAMA_FILE_MAGIC_GGSN]. */
const val LLAMA_SESSION_MAGIC: UInt = LLAMA_FILE_MAGIC_GGSN

/** Session file format version. */
const val LLAMA_SESSION_VERSION: Int = 9

/** Sequence-state file magic — alias for [LLAMA_FILE_MAGIC_GGSQ]. */
const val LLAMA_STATE_SEQ_MAGIC: UInt = LLAMA_FILE_MAGIC_GGSQ

/** Sequence-state file format version. */
const val LLAMA_STATE_SEQ_VERSION: Int = 2

// ---------------------------------------------------------------------------
// Type alias: sampler context  (llama.h line 1229)
// ---------------------------------------------------------------------------

/**
 * Opaque sampler context pointer.
 *
 * Maps to `typedef void * llama_sampler_context_t` in C++.
 */
typealias LlamaSamplerContextT = Any?

// ---------------------------------------------------------------------------
// Enum: LlamaModelKvOverrideType  (llama.h lines 246–251)
// ---------------------------------------------------------------------------

/**
 * Tag for the value stored in a [LlamaModelKvOverride].
 *
 * Maps to `enum llama_model_kv_override_type` in C++.
 */
enum class LlamaModelKvOverrideType {
    INT,
    FLOAT,
    BOOL,
    STR;
}

// ---------------------------------------------------------------------------
// Enum: LlamaModelMetaKey  (llama.h lines 253–266)
// ---------------------------------------------------------------------------

/**
 * Well-known model metadata keys for sampling defaults.
 *
 * Maps to `enum llama_model_meta_key` in C++.
 */
enum class LlamaModelMetaKey {
    SAMPLING_SEQUENCE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    SAMPLING_MIN_P,
    SAMPLING_XTC_PROBABILITY,
    SAMPLING_XTC_THRESHOLD,
    SAMPLING_TEMP,
    SAMPLING_PENALTY_LAST_N,
    SAMPLING_PENALTY_REPEAT,
    SAMPLING_MIROSTAT,
    SAMPLING_MIROSTAT_TAU,
    SAMPLING_MIROSTAT_ETA;
}

// ---------------------------------------------------------------------------
// Struct: LlamaModelKvOverride  (llama.h lines 268–279)
// ---------------------------------------------------------------------------

/**
 * Override for a single model metadata key-value pair.
 *
 * In C++ this uses a union; here we use nullable fields for each variant.
 *
 * Maps to `struct llama_model_kv_override` in C++.
 */
data class LlamaModelKvOverride(
    val tag: LlamaModelKvOverrideType,
    val key: String,
    val valI64: Long = 0L,
    val valF64: Double = 0.0,
    val valBool: Boolean = false,
    val valStr: String = "",
)

// ---------------------------------------------------------------------------
// Struct: LlamaModelTensorBuftOverride  (llama.h lines 281–284)
// ---------------------------------------------------------------------------

/**
 * Override that assigns a specific buffer type to tensors matching a pattern.
 *
 * In the Kotlin/Native CPU-only port the [buft] field is represented as a
 * string identifier rather than a native pointer.
 *
 * Maps to `struct llama_model_tensor_buft_override` in C++.
 */
data class LlamaModelTensorBuftOverride(
    val pattern: String,
    val buft: String = "cpu",
)

// ---------------------------------------------------------------------------
// Struct: LlamaSamplerSeqConfig  (llama.h lines 324–327)
// ---------------------------------------------------------------------------

/**
 * Per-sequence sampler configuration used by the backend sampling API.
 *
 * Maps to `struct llama_sampler_seq_config` in C++.
 */
data class LlamaSamplerSeqConfig(
    val seqId: LlamaSeqId,
    val sampler: LlamaSampler,
)

// ---------------------------------------------------------------------------
// Struct: LlamaModelTensorOverride  (llama.h lines 385–388)
// ---------------------------------------------------------------------------

/**
 * Override that forces a specific [GGMLType] on tensors matching [pattern].
 *
 * Maps to `struct llama_model_tensor_override` in C++.
 */
data class LlamaModelTensorOverride(
    val pattern: String,
    val type: GGMLType,
)

// ---------------------------------------------------------------------------
// Struct: LlamaModelImatrixData  (llama.h lines 390–394)
// ---------------------------------------------------------------------------

/**
 * Importance-matrix data for a single tensor.
 *
 * Maps to `struct llama_model_imatrix_data` in C++.
 */
data class LlamaModelImatrixData(
    val name: String,
    val data: FloatArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LlamaModelImatrixData) return false
        return name == other.name && data.contentEquals(other.data)
    }
    override fun hashCode(): Int = 31 * name.hashCode() + data.contentHashCode()
}

// ---------------------------------------------------------------------------
// Struct: LlamaChatMessage  (already in LlamaChat.kt — NOT duplicated)
// ---------------------------------------------------------------------------
// LlamaChatMessage is already defined in LlamaChat.kt.

// ---------------------------------------------------------------------------
// Struct: LlamaPerfSamplerData  (llama.h lines 1534–1538)
// ---------------------------------------------------------------------------

/**
 * Performance timing data for a sampler chain.
 *
 * Maps to `struct llama_perf_sampler_data` in C++.
 */
data class LlamaPerfSamplerData(
    /** Time spent sampling, in milliseconds. */
    val tSampleMs: Double = 0.0,
    /** Number of tokens sampled. */
    val nSample: Int = 0,
)

// ---------------------------------------------------------------------------
// Struct: LlamaSamplerData  (llama.h lines 1231–1236)
// ---------------------------------------------------------------------------

/**
 * Tensor references passed to backend sampler apply/accept callbacks.
 *
 * In the Kotlin port the tensor fields are nullable since the backend
 * sampling API is still experimental.
 *
 * Maps to `struct llama_sampler_data` in C++.
 */
data class LlamaSamplerData(
    var logits: GGMLTensor? = null,
    var probs: GGMLTensor? = null,
    var sampled: GGMLTensor? = null,
    var candidates: GGMLTensor? = null,
)

// ---------------------------------------------------------------------------
// Interface: LlamaSamplerI  (llama.h lines 1239–1270)
// ---------------------------------------------------------------------------

/**
 * Low-level sampler interface with all optional callbacks.
 *
 * The main [LlamaSampler] interface in `LlamaSampler.kt` covers the common
 * subset (name, accept, apply, reset, clone). This extended interface adds
 * the experimental backend-sampling hooks.
 *
 * Maps to `struct llama_sampler_i` in C++.
 */
interface LlamaSamplerI : LlamaSampler {
    /** Free any resources held by the sampler. */
    fun free() {}

    /** Return true if the backend supports all ops needed by the sampler. */
    fun backendInit(buft: String): Boolean = false

    /** Called after backendApply to accept a selected token on the graph. */
    fun backendAccept(selectedToken: GGMLTensor) {}

    /** Called after backendInit to build sampler ops into the compute graph. */
    fun backendApply(data: LlamaSamplerData) {}

    /** Called before graph execution to set inputs for the current micro-batch. */
    override fun backendSetInput() {}
}

// ---------------------------------------------------------------------------
// Struct: LlamaAdapterLora (opaque)  (llama.h line 430)
// ---------------------------------------------------------------------------

// LlamaAdapterLora is defined in LlamaAdapter.kt — removed duplicate.

// ---------------------------------------------------------------------------
// Struct: LlamaOptParams  (llama.h lines 1562–1572)
// ---------------------------------------------------------------------------

/**
 * Parameters for the training/fine-tuning optimiser.
 *
 * Maps to `struct llama_opt_params` in C++.
 */
data class LlamaOptParams(
    /** Assumed context size post-training (0 = use context params). */
    val nCtxTrain: UInt = 0u,
    /** Callback that returns true for tensors that are trainable. */
    val paramFilter: ((GGMLTensor) -> Boolean)? = null,
    /** Userdata for the param filter — not needed in Kotlin (use closure). */
    val paramFilterUd: Any? = null,
    /** Callback for calculating per-parameter optimiser hyper-parameters. */
    val getOptPars: Any? = null,
    /** Userdata for getOptPars — not needed in Kotlin (use closure). */
    val getOptParsUd: Any? = null,
)

// ---------------------------------------------------------------------------
// Memory type alias  (llama.h line 66)
// ---------------------------------------------------------------------------

/**
 * Opaque memory handle.
 *
 * In C++ this is `typedef struct llama_memory_i * llama_memory_t`.
 * In Kotlin we use an interface so implementations (KV-cache, recurrent,
 * hybrid) can be used interchangeably.
 */
interface LlamaMemoryT {
    /** Clear memory contents. If [data] is true, also clear data buffers. */
    fun clear(data: Boolean)

    /** Remove tokens for [seqId] in positions `[p0, p1)`. Returns false if partial removal fails. */
    fun seqRm(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos): Boolean

    /** Copy tokens from [seqIdSrc] to [seqIdDst] in positions `[p0, p1)`. */
    fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: LlamaPos, p1: LlamaPos)

    /** Remove all tokens that do NOT belong to [seqId]. */
    fun seqKeep(seqId: LlamaSeqId)

    /** Add [delta] to positions in `[p0, p1)` for [seqId]. */
    fun seqAdd(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, delta: LlamaPos)

    /** Integer-divide positions in `[p0, p1)` for [seqId] by [d]. */
    fun seqDiv(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, d: Int)

    /** Smallest position present for [seqId], or -1 if empty. */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos

    /** Largest position present for [seqId], or -1 if empty. */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos

    /** Whether this memory supports position shifting. */
    fun canShift(): Boolean
}
