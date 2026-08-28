// port-lint: source llama.cpp/src/llama-adapter.h
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// LlamaAdapter – LoRA and control-vector adapter types
// Ported from: llama-adapter.h
// =============================================================================

// ---------------------------------------------------------------------------
// Control Vectors (llama_adapter_cvec)
// ---------------------------------------------------------------------------

/**
 * A set of per-layer control vectors that can be added to the hidden state
 * to steer model behaviour without fine-tuning.
 *
 * Port of `llama_adapter_cvec` from `llama-adapter.h`.
 *
 * @property layerStart  First layer index (inclusive) that receives the vector.
 * @property layerEnd    Last layer index (exclusive).
 * @property tensors     Per-layer control-vector tensors (indexed by layer number).
 */
class LlamaAdapterCvec(
    var layerStart: Int = -1,
    var layerEnd: Int = -1,
    val tensors: MutableList<GGMLTensor?> = mutableListOf(),
) {
    /**
     * Return the control-vector tensor for layer [il], or `null` if no
     * vector is assigned to that layer.
     */
    fun tensorFor(il: Int): GGMLTensor? {
        if (il < layerStart || il >= layerEnd) return null
        val idx = il - layerStart
        return tensors.getOrNull(idx)
    }

    /**
     * Build a graph node that applies the control vector to [cur] at layer [il].
     *
     * Returns [cur] unchanged if no vector exists for that layer.
     */
    fun applyTo(cur: GGMLTensor, il: Int): GGMLTensor {
        val t = tensorFor(il) ?: return cur
        error("applyTo not yet ported")
    }

    /**
     * Load control-vector data into this adapter.
     *
     * @param data     Raw float data for all layers.
     * @param len      Total number of floats.
     * @param nEmbd    Model embedding dimension.
     * @param ilStart  First layer to apply (inclusive).
     * @param ilEnd    Last layer to apply (exclusive).
     * @return `true` on success.
     */
    fun apply(data: FloatArray, len: Int, nEmbd: Int, ilStart: Int, ilEnd: Int): Boolean {
        if (data.isEmpty()) {
            // Disable the current control vector (but leave allocated for later)
            layerStart = -1
            layerEnd = -1
            return true
        }

        layerStart = ilStart
        layerEnd = ilEnd

        // Ensure we have enough tensor slots
        while (tensors.size < ilEnd) {
            tensors.add(null)
        }

        for (il in 1 until tensors.size) {
            val t = tensors[il] ?: continue

            val off = nEmbd * (il - 1) // buffer doesn't have data for layer 0
            if (off + nEmbd <= len) {
                // Write the float data into the tensor's backing ByteArray
                val tensorData = t.data as? ByteArray ?: continue
                for (i in 0 until nEmbd) {
                    val bits = data[off + i].toRawBits()
                    val byteOff = i * 4
                    if (byteOff + 3 < tensorData.size) {
                        tensorData[byteOff] = (bits and 0xFF).toByte()
                        tensorData[byteOff + 1] = ((bits shr 8) and 0xFF).toByte()
                        tensorData[byteOff + 2] = ((bits shr 16) and 0xFF).toByte()
                        tensorData[byteOff + 3] = ((bits shr 24) and 0xFF).toByte()
                    }
                }
            }
        }

        return true
    }
}

// ---------------------------------------------------------------------------
// LoRA weight pair (llama_adapter_lora_weight)
// ---------------------------------------------------------------------------

/**
 * A pair of low-rank matrices (A and B) that form a single LoRA delta.
 *
 * The actual weight update is: `delta = (B × A) × scale`, where
 * `scale = alpha / rank * adapter_scale`.
 *
 * Port of `llama_adapter_lora_weight` from `llama-adapter.h`.
 *
 * @property a  The A (down-projection) tensor, or `null` if not yet loaded.
 * @property b  The B (up-projection) tensor, or `null` if not yet loaded.
 */
data class LlamaAdapterLoraWeight(
    var a: GGMLTensor? = null,
    var b: GGMLTensor? = null,
) {
    /**
     * Compute the effective scale for this weight.
     *
     * @param alpha         The LoRA alpha hyper-parameter (0 means use [adapterScale] directly).
     * @param adapterScale  Global adapter scaling factor.
     * @return The multiplier to apply to the `B × A` product.
     */
    fun getScale(alpha: Float, adapterScale: Float): Float {
        val bTensor = b ?: error("LoRA weight B tensor is null")
        val rank = bTensor.ne[0].toFloat()
        return if (alpha != 0f) adapterScale * alpha / rank else adapterScale
    }
}

// ---------------------------------------------------------------------------
// LoRA adapter (llama_adapter_lora)
// ---------------------------------------------------------------------------

/**
 * A complete LoRA adapter loaded from a GGUF file.
 *
 * Port of `llama_adapter_lora` from `llama-adapter.h`.
 *
 * @property alpha  The LoRA alpha hyper-parameter controlling scaling.
 * @property abMap  Map from base-model tensor name to the corresponding LoRA weight pair.
 * @property ggufKv  Metadata key-value pairs from the adapter's GGUF file.
 * @property aloraInvocationTokens  Tokens that trigger activated-LoRA (aLoRA) behaviour.
 */
class LlamaAdapterLora(
    var alpha: Float = 0f,
    val abMap: MutableMap<String, LlamaAdapterLoraWeight> = mutableMapOf(),
    val ggufKv: MutableMap<String, String> = mutableMapOf(),
    val aloraInvocationTokens: MutableList<Int> = mutableListOf(),
    /** Adapter metadata key-value pairs. */
    val metadata: MutableMap<String, String> = mutableMapOf(),
) {
    /** Number of invocation tokens for aLoRA adapters. */
    val nInvocationTokens: Long get() = aloraInvocationTokens.size.toLong()

    /** Invocation token IDs for aLoRA adapters. */
    val invocationTokens: IntArray get() = aloraInvocationTokens.toIntArray()
    /**
     * Look up the LoRA delta for a given base-model weight tensor.
     *
     * @param w  The base-model tensor (looked up by [GGMLTensor.name]).
     * @return The matching [LlamaAdapterLoraWeight], or `null` if this
     *         adapter doesn't override that weight.
     */
    fun getWeight(w: GGMLTensor): LlamaAdapterLoraWeight? = abMap[w.name]

    /**
     * Estimated number of compute-graph nodes required for this adapter.
     *
     * Each weight pair contributes ~6 nodes: A matmul, B matmul, scale,
     * add, and two intermediate results.
     */
    fun getNumNodes(): Int = abMap.size * 6
}

/**
 * Active LoRA adapters mapped to their per-adapter scaling factor.
 *
 * Port of `llama_adapter_loras` — `std::unordered_map<llama_adapter_lora *, float>`.
 */
typealias LlamaAdapterLoras = MutableMap<LlamaAdapterLora, Float>
