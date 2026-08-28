package io.github.kotlinmania.llama.ore

/**
 * Centralized tensor utility functions implementing DRY principle.
 * 
 * This object consolidates common helper functions used across the GGML implementation
 * to avoid code duplication and provide consistent behavior. It includes:
 * 
 * - Tensor size and stride calculations
 * - Memory layout utilities
 * - Validation and bounds checking
 * - Type conversion and compatibility helpers
 * 
 * All functions are designed to work with the ByteArray-based memory model
 * used throughout the Kotlin GGML implementation.
 */
object GGMLTensorUtils {
    
    /**
     * Calculate total size (number of elements) of a tensor from its dimensions.
     * 
     * This function computes the total number of elements in a tensor by multiplying
     * all valid dimensions. It handles edge cases like zero dimensions and ensures
     * proper bounds checking.
     * 
     * Used throughout the codebase for:
     * - Memory allocation calculations
     * - Validation of tensor operations
     * - Buffer size determination
     * 
     * @param ne Array of dimension sizes (up to GGML_MAX_DIMS)
     * @return Total number of elements in the tensor
     */
    fun calculateTotalSize(ne: LongArray): Long {
        var totalSize = 1L
        for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
            if (i < ne.size && ne[i] > 0) {
                totalSize *= ne[i]
            } else if (i >= ne.size) {
                break // Only multiply valid dimensions
            }
        }
        return totalSize
    }
    
    /**
     * Legacy compatibility function that returns Int for backward compatibility.
     * 
     * @param ne Array of dimension sizes
     * @return Total number of elements as Int (may overflow for large tensors)
     * @deprecated Use calculateTotalSize() for better precision and overflow safety
     */
    @Deprecated(
        message = "Use calculateTotalSize() for better precision", 
        replaceWith = ReplaceWith("calculateTotalSize(ne).toInt()")
    )
    fun calculateTotalSizeInt(ne: LongArray): Int {
        return calculateTotalSize(ne).toInt()
    }
    
    /**
     * Calculate contiguous tensor strides for memory layout.
     * 
     * Strides define how to navigate through multi-dimensional tensor data stored
     * in a contiguous ByteArray. The stride for dimension d indicates how many bytes
     * to advance to move one position in that dimension.
     * 
     * This function consolidates the stride calculation logic from GGMLOps and 
     * other utilities to ensure consistent memory layout throughout the library.
     * 
     * @param ne Array of dimension sizes
     * @param type Tensor data type (determines element size)
     * @param maxDims Maximum dimensions to calculate (defaults to GGML_MAX_DIMS)
     * @return Array of stride values in bytes for each dimension
     */
    fun calculateContiguousStrides(ne: LongArray, type: io.github.kotlinmania.llama.ore.GGMLType, maxDims: Int = io.github.kotlinmania.llama.ore.GGML_MAX_DIMS): ULongArray {
        val nb = ULongArray(maxDims) { 0uL }
        
        if (type.byteSize == 0uL) {
            // Handle types with zero byte size (e.g., COUNT or uninitialized quantized types)
            if (type != io.github.kotlinmania.llama.ore.GGMLType.COUNT && !type.name.startsWith("Q", ignoreCase = true) && !type.name.startsWith("q", ignoreCase = true)) {
                println("Warning: GGMLType ${type.name} has byteSize 0. Strides will be all zeros.")
            }
            return nb // Return zeroed strides
        }
        
        nb[0] = type.byteSize
        if (maxDims > 1) {
            for (d in 1 until maxDims) {
                val prevDimSize = ne.getOrElse(d - 1) { 1L }
                nb[d] = nb[d - 1] * (if (prevDimSize > 0L) prevDimSize.toULong() else 1uL)
            }
        }
        return nb
    }
    
    /**
     * Calculate tensor byte size for memory allocation.
     * Handles both regular types and quantized types with block structures.
     */
    fun calculateTensorByteSize(type: io.github.kotlinmania.llama.ore.GGMLType, ne: LongArray): ULong {
        if (type.byteSize == 0uL && type != io.github.kotlinmania.llama.ore.GGMLType.COUNT && !type.name.startsWith("Q")) {
            return 0uL
        }
        
        var elements = 1UL
        var validDimFound = false
        
        for (i in ne.indices) {
            if (ne[i] > 0L) {
                elements *= ne[i].toULong()
                validDimFound = true
            } else if (ne[i] == 0L && elements != 0UL && validDimFound) {
                return 0UL
            }
        }
        
        if (!validDimFound && ne.isNotEmpty() && ne.all { it <= 1L }) {
            elements = 1UL // Scalar or effectively scalar
        } else if (!validDimFound && ne.isEmpty()) {
            elements = 1UL // Treat as scalar if ne is completely empty
        }

        // Handle quantized types with block structures
        return when (type) {
            io.github.kotlinmania.llama.ore.GGMLType.Q8_0 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % io.github.kotlinmania.llama.ore.QK8_0 != 0L) {
                        println("Warning: Total elements $elements for Q8_0 is not divisible by block size ${io.github.kotlinmania.llama.ore.QK8_0}.")
                    }
                    (elements.toLong() / io.github.kotlinmania.llama.ore.QK8_0).toULong() * type.byteSize
                } else 0uL
            }
            io.github.kotlinmania.llama.ore.GGMLType.Q4_0 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % io.github.kotlinmania.llama.ore.QK4_0 != 0L) {
                        println("Warning: Total elements $elements for Q4_0 is not divisible by block size ${io.github.kotlinmania.llama.ore.QK4_0}.")
                    }
                    (elements.toLong() / io.github.kotlinmania.llama.ore.QK4_0).toULong() * type.byteSize
                } else 0uL
            }
            io.github.kotlinmania.llama.ore.GGMLType.Q4_1 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % io.github.kotlinmania.llama.ore.QK4_1 != 0L) {
                        println("Warning: Total elements $elements for Q4_1 is not divisible by block size ${io.github.kotlinmania.llama.ore.QK4_1}.")
                    }
                    (elements.toLong() / io.github.kotlinmania.llama.ore.QK4_1).toULong() * type.byteSize
                } else 0uL
            }
            else -> elements * type.byteSize
        }
    }
    
    /**
     * Initialize a tensor's dimensions and strides based on the provided shape.
     * Consolidates the repetitive tensor setup pattern.
     */
    fun initializeTensorDimensions(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, shape: LongArray) {
        // Ensure ne array is properly sized and initialized
        tensor.ne = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 1L }
        shape.forEachIndexed { index, dimSize ->
            if (index < io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                tensor.ne[index] = dimSize
            }
        }
        
        // Calculate and set strides
        tensor.nb = GGMLTensorUtils.calculateContiguousStrides(tensor.ne, tensor.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
    }
    
    /**
     * Create a tensor with proper dimensions and memory layout.
     * Consolidates the tensor creation pattern used throughout the codebase.
     */
    fun createTensor(type: io.github.kotlinmania.llama.ore.GGMLType, name: String = "", vararg shape: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type, name = name)
        initializeTensorDimensions(tensor, shape)
        return tensor
    }
}
