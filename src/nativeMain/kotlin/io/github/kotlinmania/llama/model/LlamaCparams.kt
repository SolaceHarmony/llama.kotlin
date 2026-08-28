// port-lint: source llama.cpp/src/llama-cparams.h
package io.github.kotlinmania.llama.model

// =============================================================================
// LlamaCparams – computed context parameters
// Ported from: llama-cparams.h  struct llama_cparams
// =============================================================================

/**
 * Maximum number of sequences supported per context.
 *
 * This is the value used in the C++ header (`#define LLAMA_MAX_SEQ 256`).
 * The project also defines [LLAMA_MAX_SEQ] in `KVCache.kt` (currently 64)
 * for the KV-cache layer.  This constant preserves the original C++ value
 * for any code that needs the header-level cap.
 */
const val LLAMA_MAX_SEQ_CPARAMS: Int = 256

/**
 * Derived / computed context parameters that are resolved from [LlamaContextParams]
 * and the model's hyper-parameters during [LlamaContext] construction.
 *
 * Maps to `struct llama_cparams` in `llama-cparams.h`.
 */
data class LlamaCParams(
    var nCtx: Int = 0,
    var nCtxSeq: Int = 0,
    var nBatch: Int = 0,
    var nUbatch: Int = 0,
    var nSeqMax: Int = 1,
    var nThreads: Int = 4,
    var nThreadsBatch: Int = 4,
    var ropeFreqBase: Float = 10000.0f,
    var ropeFreqScale: Float = 1.0f,
    var nCtxOrigYarn: Int = 0,
    var yarnExtFactor: Float = 0.0f,
    var yarnAttnFactor: Float = 1.0f,
    var yarnBetaFast: Float = 32.0f,
    var yarnBetaSlow: Float = 1.0f,
    var embeddings: Boolean = false,
    var causalAttn: Boolean = true,
    var offloadKqv: Boolean = true,
    var flashAttn: Boolean = false,
    var autoFa: Boolean = false,
    var fusedGdnAr: Boolean = true,
    var fusedGdnCh: Boolean = true,
    var autoFgdn: Boolean = true,
    var noPerf: Boolean = false,
    var warmup: Boolean = false,
    var opOffload: Boolean = true,
    var kvUnified: Boolean = true,
    var pipelineParallel: Boolean = false,
    var poolingType: LlamaPoolingType = LlamaPoolingType.NONE,
)

/** Port of `llama_max_parallel_sequences()` from llama-cparams.cpp. */
fun llamaMaxParallelSequences(): Int = LLAMA_MAX_SEQ_CPARAMS
