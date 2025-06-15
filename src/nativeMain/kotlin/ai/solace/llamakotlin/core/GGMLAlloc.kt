package ai.solace.llamakotlin.core

/**
 * Kotlin Native port of GGML memory allocation functionality.
 * This file contains the memory allocation and management functions for GGML tensors.
 */

/**
 * Calculates the offset aligned to the specified alignment.
 *
 * @param offset The original offset
 * @param alignment The alignment requirement (must be a power of 2)
 * @return The aligned offset
 */
@OptIn(kotlin.experimental.ExperimentalNativeApi::class)
fun alignedOffset(offset: ULong, alignment: UInt): ULong {
    // Ensure alignment is a power of 2
    assert(alignment > 0u && (alignment and (alignment - 1u)) == 0u)

    val align = (alignment - (offset % alignment)) % alignment
    return offset + align
}

/**
 * Tensor allocator for managing memory allocation for individual tensors.
 */
class GGMLTensorAllocator {
    // Buffer where tensors are allocated
    var buffer: Any? = null

    // Base pointer of the buffer
    var base: Any? = null

    // Alignment requirement for tensor data
    var alignment: UInt = 16u

    // Current offset in the buffer
    var offset: ULong = 0u

    /**
     * Creates a new tensor allocator.
     *
     * @param buffer The buffer to allocate from
     * @param alignment The alignment requirement for tensor data
     */
    constructor(buffer: Any? = null, alignment: UInt = 16u) {
        this.buffer = buffer
        this.base = buffer
        this.alignment = alignment
        this.offset = 0u
    }

    private fun ensureBufferCapacity(bufferId: Int, requiredSize: ULong) {
        if (bufferId < 0 || bufferId >= buffers.size || bufferId >= tensorAllocators.size) {
            throw IllegalArgumentException("Error: Invalid bufferId $bufferId. It must be between 0 and ${buffers.size - 1}, and within the range of tensorAllocators.")
        }

        val currentBuffer = buffers[bufferId]
        if (currentBuffer == null || currentBuffer.size < requiredSize.toInt()) {
            // Ensure requiredSize is not zero if creating a new buffer,
            // though ULong to Int conversion might cap it.
            // Consider a minimum practical size or error if requiredSize is too large for Int.
            val newSize = if (requiredSize > Int.MAX_VALUE.toULong()) {
                println("Warning: requiredSize $requiredSize exceeds Int.MAX_VALUE. Clamping to Int.MAX_VALUE.")
                Int.MAX_VALUE
            } else {
                requiredSize.toInt()
            }

            if (newSize <= 0 && requiredSize > 0u) {
                throw IllegalArgumentException("Invalid buffer allocation: requiredSize $requiredSize resulted in non-positive newSize $newSize. This may indicate an overflow or logical error.")
            }

            buffers[bufferId] = ByteArray(newSize)
            tensorAllocators[bufferId].reset(newSize.toULong())
        }
    }

    /**
     * Allocates memory for a tensor.
     *
     * @param tensor The tensor to allocate memory for
     */
    fun allocate(tensor: GGMLTensor) {
        // Calculate the size of the tensor
        val size = calculateTensorSize(tensor)

        // Align the offset to the required alignment
        offset = alignedOffset(offset, alignment)

        // Allocate memory for the tensor based on its type
        when (tensor.type) {
            GGMLType.F32 -> tensor.data = FloatArray(size.toInt()) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(size.toInt()) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(size.toInt()) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(size.toInt()) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(size.toInt()) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(size.toInt()) { 0L }
            else -> tensor.data = ByteArray(size.toInt()) { 0 } // Default for quantized types
        }

        // Update the offset
        offset += size
    }

    /**
     * Calculates the size of a tensor in elements.
     *
     * @param tensor The tensor to calculate the size for
     * @return The size of the tensor in elements
     */
    private fun calculateTensorSize(tensor: GGMLTensor): ULong {
        var size = 1UL
        for (i in 0 until GGML_MAX_DIMS) {
            size *= tensor.ne[i].toULong()
        }
        return size
    }

    /**
     * Resets the allocator.
     */
    @Suppress("unused")
    fun reset() {
        offset = 0u
    }
}

/**
 * Free block for dynamic memory allocation.
 */
class FreeBlock(
    var offset: ULong = 0u,
    var size: ULong = 0u
)

/**
 * Dynamic tensor allocator for managing memory allocation with free blocks.
 */
class GGMLDynTensorAllocator {
    // Alignment requirement for tensor data
    var alignment: UInt = 16u

    // Free blocks
    var freeBlocks = mutableListOf<FreeBlock>()

    // Maximum size allocated
    var maxSize: ULong = 0u

    /**
     * Creates a new dynamic tensor allocator.
     *
     * @param alignment The alignment requirement for tensor data
     */
    constructor(alignment: UInt = 16u, bufferSize: ULong? = null) {
        this.alignment = alignment
        reset(bufferSize)
    }

    /**
     * Allocates memory for a tensor.
     *
     * @param size The size to allocate in bytes
     * @param tensor The tensor to allocate memory for (used for debugging)
     * @return The offset of the allocated memory
     */
    fun allocate(size: ULong, @Suppress("unused") tensor: GGMLTensor): ULong {
        // Align the size to the required alignment
        val alignedSize = alignedOffset(size, alignment)

        // Find the best fitting free block
        var bestFitBlock = -1
        var bestFitSize = ULong.MAX_VALUE

        for (i in 0 until freeBlocks.size - 1) {
            val block = freeBlocks[i]
            if (block.size >= alignedSize && block.size <= bestFitSize) {
                bestFitBlock = i
                bestFitSize = block.size
            }
        }

        // If no best fit found, use the last block
        if (bestFitBlock == -1) {
            val lastBlock = freeBlocks.last()
            if (lastBlock.size >= alignedSize) {
                bestFitBlock = freeBlocks.size - 1
            } else {
                throw IllegalStateException("Not enough space in the buffer to allocate $alignedSize bytes")
            }
        }

        // Allocate from the best fit block
        val block = freeBlocks[bestFitBlock]
        val offset = block.offset
        block.offset += alignedSize
        block.size -= alignedSize

        // Remove the block if it's empty
        if (block.size == 0UL) {
            freeBlocks.removeAt(bestFitBlock)
        }

        // Update the maximum size
        maxSize = maxOf(maxSize, offset + alignedSize)

        return offset
    }

    /**
     * Frees memory for a tensor.
     *
     * @param offset The offset of the memory to free
     * @param size The size of the memory to free
     * @param tensor The tensor to free memory for (used for debugging)
     */
    fun freeTensor(offset: ULong, size: ULong, @Suppress("unused") tensor: GGMLTensor) {
        // Align the size to the required alignment
        val alignedSize = alignedOffset(size, alignment)

        // Try to merge with an existing block
        for (i in freeBlocks.indices) {
            val block = freeBlocks[i]

            // Check if the memory is at the end of the block
            if (block.offset + block.size == offset) {
                block.size += alignedSize

                // Check if we can merge with the next block
                if (i < freeBlocks.size - 1 && block.offset + block.size == freeBlocks[i + 1].offset) {
                    block.size += freeBlocks[i + 1].size
                    freeBlocks.removeAt(i + 1)
                }
                return
            }

            // Check if the memory is at the beginning of the block
            if (offset + alignedSize == block.offset) {
                block.offset = offset
                block.size += alignedSize

                // Check if we can merge with the previous block
                if (i > 0 && freeBlocks[i - 1].offset + freeBlocks[i - 1].size == block.offset) {
                    freeBlocks[i - 1].size += block.size
                    freeBlocks.removeAt(i)
                }
                return
            }
        }

        // Add a new block
        val newBlock = FreeBlock(offset, alignedSize)

        // Insert the new block in the correct position to keep the array sorted by address
        var insertPos = 0
        while (insertPos < freeBlocks.size && freeBlocks[insertPos].offset < offset) {
            insertPos++
        }

        freeBlocks.add(insertPos, newBlock)
    }

    /**
     * Resets the allocator.
     */
    fun reset(bufferSize: ULong? = null) {
        freeBlocks.clear()
        freeBlocks.add(FreeBlock(0u, bufferSize ?: (ULong.MAX_VALUE / 2u))) // Restrict maximum size to half ULong.MAX_VALUE to avoid overflows
        maxSize = 0u
    }

    /**
     * Gets the maximum size allocated.
     *
     * @return The maximum size allocated
     */
    fun getMaxSize(): ULong {
        return maxSize
    }
}

/**
 * Graph allocator for managing memory allocation for computation graphs.
 */
class GGMLGraphAllocator {
    // Tensor allocator for each buffer
    var tensorAllocators = mutableListOf<GGMLDynTensorAllocator>()

    // Buffers
    var buffers = mutableListOf<ByteArray?>()

    /**
     * Creates a new graph allocator.
     */
    constructor() {
        // Create a default tensor allocator
        val defaultBufferSize = 1024 * 1024
        buffers.add(ByteArray(defaultBufferSize))
        tensorAllocators.add(GGMLDynTensorAllocator(bufferSize = defaultBufferSize.toULong()))
    }

    /**
     * Allocates memory for all tensors in a computation graph.
     *
     * @param graph The computation graph to allocate memory for
     * @return True if allocation was successful, false otherwise
     */
    fun allocateGraph(graph: GGMLCGraph): Boolean {
        // Reset the allocators
        for (allocator in tensorAllocators) {
            allocator.reset()
        }

        // Allocate memory for leaf nodes
        for (i in 0 until graph.nLeafs) {
            val leaf = graph.leafs[i] ?: continue
            if (leaf.data == null && !ggml_is_view(leaf)) {
                allocateTensor(leaf, 0)
            }
        }

        // Allocate memory for internal nodes
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue

            // Allocate memory for source tensors if needed
            for (j in 0 until GGML_MAX_SRC) {
                val src = node.src[j] ?: continue
                if (src.data == null && !ggml_is_view(src)) {
                    allocateTensor(src, 0)
                }
            }

            // Allocate memory for the node itself
            if (node.data == null && !ggml_is_view(node)) {
                allocateTensor(node, 0)
            }
        }

        val calculatedMaxSize = getBufferSize(0)
        ensureBufferCapacity(0, calculatedMaxSize)

        return true
    }

    /**
     * Allocates memory for a tensor.
     *
     * @param tensor The tensor to allocate memory for
     * @param bufferId The ID of the buffer to allocate from
     */
    private fun allocateTensor(tensor: GGMLTensor, bufferId: Int) {
        // Calculate the size of the tensor
        val size = calculateTensorSize(tensor)

        // Allocate memory from the tensor allocator
        val offset = tensorAllocators[bufferId].allocate(size, tensor)

        // Set buffer information on the tensor
        tensor.bufferId = bufferId
        tensor.dataOffset = offset
        // tensor.data remains null, actual data access will be through the buffer
    }

    /**
     * Calculates the size of a tensor in elements.
     *
     * @param tensor The tensor to calculate the size for
     * @return The size of the tensor in elements
     */
    private fun calculateTensorSize(tensor: GGMLTensor): ULong {
        var size = 1UL
        for (i in 0 until GGML_MAX_DIMS) {
            size *= tensor.ne[i].toULong()
        }
        return size
    }

    /**
     * Reserves memory for a computation graph without actually allocating it.
     *
     * @param graph The computation graph to reserve memory for
     * @return True if reservation was successful, false otherwise
     */
    @Suppress("unused")
    fun reserveGraph(graph: GGMLCGraph): Boolean {
        // This is similar to allocateGraph, but doesn't actually allocate memory
        // It just calculates the memory requirements

        // Reset the allocators
        for (allocator in tensorAllocators) {
            allocator.reset()
        }

        // Calculate memory requirements for leaf nodes
        for (i in 0 until graph.nLeafs) {
            val leaf = graph.leafs[i] ?: continue
            if (leaf.data == null && !ggml_is_view(leaf)) {
                reserveTensor(leaf, 0)
            }
        }

        // Calculate memory requirements for internal nodes
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue

            // Calculate memory requirements for source tensors if needed
            for (j in 0 until GGML_MAX_SRC) {
                val src = node.src[j] ?: continue
                if (src.data == null && !ggml_is_view(src)) {
                    reserveTensor(src, 0)
                }
            }

            // Calculate memory requirements for the node itself
            if (node.data == null && !ggml_is_view(node)) {
                reserveTensor(node, 0)
            }
        }

        return true
    }

    /**
     * Reserves memory for a tensor without actually allocating it.
     *
     * @param tensor The tensor to reserve memory for
     * @param bufferId The ID of the buffer to reserve from
     */
    private fun reserveTensor(tensor: GGMLTensor, bufferId: Int) {
        // Calculate the size of the tensor
        val size = calculateTensorSize(tensor)

        // Reserve memory from the tensor allocator
        tensorAllocators[bufferId].allocate(size, tensor)
    }

    /**
     * Gets the size of a buffer.
     *
     * @param bufferId The ID of the buffer
     * @return The size of the buffer in bytes
     */
    fun getBufferSize(bufferId: Int): ULong {
        if (bufferId < 0 || bufferId >= tensorAllocators.size) {
            return 0u
        }

        return tensorAllocators[bufferId].getMaxSize()
    }
}

/**
 * Checks if a tensor is a view.
 *
 * @param tensor The tensor to check
 * @return True if the tensor is a view, false otherwise
 */
fun ggml_is_view(tensor: GGMLTensor): Boolean {
    return tensor.viewSrc != null
}
