package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

/**
 * Key-Value cache for efficient transformer inference.
 * Manages cached keys and values to avoid recomputing attention for previous tokens.
 */
class KVCache(
    val maxSequenceLength: Int,
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
    val maxBatchSize: Int = 1
) {
    // Cache storage for keys and values
    private val keyCache = Array(numLayers) { 
        Array(maxBatchSize) { 
            GGMLTensor(type = GGMLType.F32).apply {
                ne[0] = headDim.toLong()
                ne[1] = numHeads.toLong()
                ne[2] = maxSequenceLength.toLong()
                ne[3] = 1L
                nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
            }
        }
    }
    
    private val valueCache = Array(numLayers) {
        Array(maxBatchSize) {
            GGMLTensor(type = GGMLType.F32).apply {
                ne[0] = headDim.toLong()
                ne[1] = numHeads.toLong()
                ne[2] = maxSequenceLength.toLong()
                ne[3] = 1L
                nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
            }
        }
    }
    
    // Track current sequence length for each batch
    private val sequenceLengths = IntArray(maxBatchSize) { 0 }
    
    // Sequence IDs for managing multiple conversations
    private val sequenceIds = Array(maxBatchSize) { -1 }
    
    /**
     * Initialize cache tensors with proper memory allocation.
     */
    fun initialize(graphAllocator: GGMLGraphAllocator) {
        for (layer in 0 until numLayers) {
            for (batch in 0 until maxBatchSize) {
                graphAllocator.allocateTensor(keyCache[layer][batch])
                graphAllocator.allocateTensor(valueCache[layer][batch])
                
                // Zero initialize the cache
                clearCache(graphAllocator, layer, batch)
            }
        }
    }
    
    /**
     * Update the key cache with new key tensor for a specific layer and batch.
     */
    fun updateKey(
        graphAllocator: GGMLGraphAllocator,
        layerIndex: Int,
        batchIndex: Int,
        newKey: GGMLTensor
    ): GGMLTensor {
        require(layerIndex < numLayers) { "Layer index $layerIndex out of bounds" }
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        
        val currentSeqLen = sequenceLengths[batchIndex]
        val newSeqLen = newKey.ne[2].toInt()
        
        require(currentSeqLen + newSeqLen <= maxSequenceLength) {
            "Sequence length would exceed maximum: ${currentSeqLen + newSeqLen} > $maxSequenceLength"
        }
        
        val cacheKey = keyCache[layerIndex][batchIndex]
        
        // Copy new keys to the cache at the current position
        for (head in 0 until numHeads) {
            for (seq in 0 until newSeqLen) {
                for (dim in 0 until headDim) {
                    val newValue = newKey.getFloat(graphAllocator, dim, head, seq, 0)
                    cacheKey.setFloat(graphAllocator, dim, head, currentSeqLen + seq, 0, newValue)
                }
            }
        }
        
        // Update sequence length
        sequenceLengths[batchIndex] += newSeqLen
        
        // Return a tensor that includes both cached and new keys
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = headDim.toLong()
        result.ne[1] = numHeads.toLong()
        result.ne[2] = sequenceLengths[batchIndex].toLong()
        result.ne[3] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(result)
        
        // Copy all keys (cached + new) to result
        for (head in 0 until numHeads) {
            for (seq in 0 until sequenceLengths[batchIndex]) {
                for (dim in 0 until headDim) {
                    val value = cacheKey.getFloat(graphAllocator, dim, head, seq, 0)
                    result.setFloat(graphAllocator, dim, head, seq, 0, value)
                }
            }
        }
        
        return result
    }
    
    /**
     * Update the value cache with new value tensor for a specific layer and batch.
     */
    fun updateValue(
        graphAllocator: GGMLGraphAllocator,
        layerIndex: Int,
        batchIndex: Int,
        newValue: GGMLTensor
    ): GGMLTensor {
        require(layerIndex < numLayers) { "Layer index $layerIndex out of bounds" }
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        
        val currentSeqLen = sequenceLengths[batchIndex]
        val newSeqLen = newValue.ne[2].toInt()
        
        val cacheValue = valueCache[layerIndex][batchIndex]
        
        // Copy new values to the cache at the current position
        for (head in 0 until numHeads) {
            for (seq in 0 until newSeqLen) {
                for (dim in 0 until headDim) {
                    val newVal = newValue.getFloat(graphAllocator, dim, head, seq, 0)
                    cacheValue.setFloat(graphAllocator, dim, head, currentSeqLen + seq, 0, newVal)
                }
            }
        }
        
        // Return a tensor that includes both cached and new values
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = headDim.toLong()
        result.ne[1] = numHeads.toLong()
        result.ne[2] = sequenceLengths[batchIndex].toLong()
        result.ne[3] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(result)
        
        // Copy all values (cached + new) to result
        for (head in 0 until numHeads) {
            for (seq in 0 until sequenceLengths[batchIndex]) {
                for (dim in 0 until headDim) {
                    val value = cacheValue.getFloat(graphAllocator, dim, head, seq, 0)
                    result.setFloat(graphAllocator, dim, head, seq, 0, value)
                }
            }
        }
        
        return result
    }
    
    /**
     * Get the cached key tensor for a specific layer and batch.
     */
    fun getKey(layerIndex: Int, batchIndex: Int): GGMLTensor {
        require(layerIndex < numLayers) { "Layer index $layerIndex out of bounds" }
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        
        return keyCache[layerIndex][batchIndex]
    }
    
    /**
     * Get the cached value tensor for a specific layer and batch.
     */
    fun getValue(layerIndex: Int, batchIndex: Int): GGMLTensor {
        require(layerIndex < numLayers) { "Layer index $layerIndex out of bounds" }
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        
        return valueCache[layerIndex][batchIndex]
    }
    
    /**
     * Clear the cache for a specific layer and batch.
     */
    fun clearCache(graphAllocator: GGMLGraphAllocator, layerIndex: Int, batchIndex: Int) {
        require(layerIndex < numLayers) { "Layer index $layerIndex out of bounds" }
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        
        val cacheKey = keyCache[layerIndex][batchIndex]
        val cacheValue = valueCache[layerIndex][batchIndex]
        
        // Zero out the cache tensors
        val keyElements = cacheKey.numElements().toInt()
        val valueElements = cacheValue.numElements().toInt()
        
        for (i in 0 until keyElements) {
            val indices = IntArray(GGML_MAX_DIMS) { 0 }
            var temp = i
            for (d in 0 until GGML_MAX_DIMS) {
                indices[d] = temp % cacheKey.ne[d].toInt()
                temp /= cacheKey.ne[d].toInt()
            }
            cacheKey.setFloat(graphAllocator, *indices, 0.0f)
        }
        
        for (i in 0 until valueElements) {
            val indices = IntArray(GGML_MAX_DIMS) { 0 }
            var temp = i
            for (d in 0 until GGML_MAX_DIMS) {
                indices[d] = temp % cacheValue.ne[d].toInt()
                temp /= cacheValue.ne[d].toInt()
            }
            cacheValue.setFloat(graphAllocator, *indices, 0.0f)
        }
        
        sequenceLengths[batchIndex] = 0
    }
    
    /**
     * Clear all caches.
     */
    fun clearAllCaches(graphAllocator: GGMLGraphAllocator) {
        for (layer in 0 until numLayers) {
            for (batch in 0 until maxBatchSize) {
                clearCache(graphAllocator, layer, batch)
            }
        }
    }
    
    /**
     * Get current sequence length for a batch.
     */
    fun getSequenceLength(batchIndex: Int): Int {
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        return sequenceLengths[batchIndex]
    }
    
    /**
     * Set sequence ID for tracking multiple conversations.
     */
    fun setSequenceId(batchIndex: Int, sequenceId: Int) {
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        sequenceIds[batchIndex] = sequenceId
    }
    
    /**
     * Get sequence ID for a batch.
     */
    fun getSequenceId(batchIndex: Int): Int {
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        return sequenceIds[batchIndex]
    }
    
    /**
     * Check if cache can accommodate additional tokens.
     */
    fun canAccommodate(batchIndex: Int, additionalTokens: Int): Boolean {
        require(batchIndex < maxBatchSize) { "Batch index $batchIndex out of bounds" }
        return sequenceLengths[batchIndex] + additionalTokens <= maxSequenceLength
    }
}

/**
 * Extended KVCache that integrates with LlamaAttention.
 * This version provides a simpler interface for the attention mechanism.
 */
class KVCache(
    val maxSequenceLength: Int,
    val numHeads: Int,
    val headDim: Int
) {
    private var keys: GGMLTensor? = null
    private var values: GGMLTensor? = null
    private var currentLength = 0
    
    /**
     * Initialize the cache with given dimensions.
     */
    fun initialize(graphAllocator: GGMLGraphAllocator) {
        keys = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = headDim.toLong()
            ne[1] = numHeads.toLong()
            ne[2] = maxSequenceLength.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        
        values = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = headDim.toLong()
            ne[1] = numHeads.toLong()
            ne[2] = maxSequenceLength.toLong()
            ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        
        keys?.let { graphAllocator.allocateTensor(it) }
        values?.let { graphAllocator.allocateTensor(it) }
        
        currentLength = 0
    }
    
    /**
     * Update keys and return concatenated tensor.
     */
    fun updateKey(newKey: GGMLTensor): GGMLTensor {
        val cachedKeys = keys ?: throw IllegalStateException("Cache not initialized")
        
        val newSeqLen = newKey.ne[2].toInt()
        require(currentLength + newSeqLen <= maxSequenceLength) {
            "Sequence length would exceed maximum"
        }
        
        // Create result tensor with current + new length
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = headDim.toLong()
        result.ne[1] = numHeads.toLong()
        result.ne[2] = (currentLength + newSeqLen).toLong()
        result.ne[3] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        return result
    }
    
    /**
     * Update values and return concatenated tensor.
     */
    fun updateValue(newValue: GGMLTensor): GGMLTensor {
        val cachedValues = values ?: throw IllegalStateException("Cache not initialized")
        
        val newSeqLen = newValue.ne[2].toInt()
        
        // Create result tensor with current + new length  
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = headDim.toLong()
        result.ne[1] = numHeads.toLong()
        result.ne[2] = (currentLength + newSeqLen).toLong()
        result.ne[3] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        currentLength += newSeqLen
        
        return result
    }
    
    /**
     * Clear the cache.
     */
    fun clear() {
        currentLength = 0
    }
    
    /**
     * Get current sequence length.
     */
    fun getCurrentLength(): Int = currentLength
}