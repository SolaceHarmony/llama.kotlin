package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

/**
 * Key-Value cache for efficient transformer inference (per-layer, single-batch).
 * Allocator-first API; maintains internal contiguous key/value tensors and current length.
 */
class KVCache(
    val maxSequenceLength: Int,
    val numHeads: Int,
    val headDim: Int
) {
    private var keys: GGMLTensor? = null
    private var values: GGMLTensor? = null
    private var currentLength = 0

    fun initialize(graphAllocator: GGMLGraphAllocator) {
        keys = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = headDim.toLong()
            ne[1] = numHeads.toLong()
            ne[2] = maxSequenceLength.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }.also { graphAllocator.allocateTensor(it) }

        values = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = headDim.toLong()
            ne[1] = numHeads.toLong()
            ne[2] = maxSequenceLength.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }.also { graphAllocator.allocateTensor(it) }

        currentLength = 0
    }

    /**
     * Append new keys into the cache and return a tensor view (copy) of size currentLength+newLen.
     */
    fun updateKey(graphAllocator: GGMLGraphAllocator, newKey: GGMLTensor): GGMLTensor {
        val cacheKey = keys ?: throw IllegalStateException("KVCache not initialized")
        val newSeqLen = newKey.ne[2].toInt()
        require(currentLength + newSeqLen <= maxSequenceLength) { "Sequence length would exceed maximum" }

        // Copy new keys to cache at current position
        for (h in 0 until numHeads) {
            for (s in 0 until newSeqLen) {
                for (d in 0 until headDim) {
                    val v = newKey.getFloat(graphAllocator, d, h, s, 0)
                    cacheKey.setFloat(graphAllocator, v, d, h, currentLength + s, 0)
                }
            }
        }
        val out = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = headDim.toLong()
            ne[1] = numHeads.toLong()
            ne[2] = (currentLength + newSeqLen).toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        graphAllocator.allocateTensor(out)
        for (h in 0 until numHeads) {
            for (s in 0 until (currentLength + newSeqLen)) {
                for (d in 0 until headDim) {
                    val v = cacheKey.getFloat(graphAllocator, d, h, s, 0)
                    out.setFloat(graphAllocator, v, d, h, s, 0)
                }
            }
        }
        currentLength += newSeqLen
        return out
    }

    /**
     * Append new values into the cache and return a tensor view (copy) of size currentLength.
     * Note: This returns the concatenated values length-matched with keys.
     */
    fun updateValue(graphAllocator: GGMLGraphAllocator, newValue: GGMLTensor): GGMLTensor {
        val cacheVal = values ?: throw IllegalStateException("KVCache not initialized")
        val newSeqLen = newValue.ne[2].toInt()
        require(currentLength >= newSeqLen) { "updateKey must be called before updateValue in a step" }

        // Append new values at the tail region corresponding to the new keys just appended
        val start = currentLength - newSeqLen
        for (h in 0 until numHeads) {
            for (s in 0 until newSeqLen) {
                for (d in 0 until headDim) {
                    val v = newValue.getFloat(graphAllocator, d, h, s, 0)
                    cacheVal.setFloat(graphAllocator, v, d, h, start + s, 0)
                }
            }
        }

        val out = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = headDim.toLong()
            ne[1] = numHeads.toLong()
            ne[2] = currentLength.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        graphAllocator.allocateTensor(out)
        for (h in 0 until numHeads) {
            for (s in 0 until currentLength) {
                for (d in 0 until headDim) {
                    val v = cacheVal.getFloat(graphAllocator, d, h, s, 0)
                    out.setFloat(graphAllocator, v, d, h, s, 0)
                }
            }
        }
        return out
    }

    fun clear() { currentLength = 0 }
    fun getCurrentLength(): Int = currentLength
}