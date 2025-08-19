package ai.solace.llamakotlin.core

/**
 * Data class to store usage information for each tensor in the graph.
 */
internal data class TensorUsageInfo(
    var numChildren: Int = 0,       // Number of direct consumers of this tensor
    var numViews: Int = 0,          // Number of views pointing to this tensor's data
    var ownsMemory: Boolean = false, // True if this tensor is the original owner of its allocated memory segment
                                   // Becomes false if its memory is reused by an inplace child.
    var isOutputTensor: Boolean = false, // True if this tensor is marked as a graph output
    var dataOffset: ULong = 0uL,    // Actual allocated offset in the buffer
    var bufferId: Int = -1,         // Actual buffer ID where tensor is allocated
    var calculatedSize: ULong = 0uL // Calculated size of the tensor in bytes
)

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

    /**
     * Allocates memory for a tensor.
     *
     * @param tensor The tensor to allocate memory for
     */
    fun allocate(tensor: GGMLTensor) {
        // Calculate the number of elements for direct array allocation
        val numElements = tensor.numElements().toInt() // Changed from calculateTensorSize

        // Align the offset to the required alignment
        offset = alignedOffset(offset, alignment)

        // Allocate memory for the tensor based on its type
        when (tensor.type) {
            GGMLType.F32 -> tensor.data = FloatArray(numElements) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(numElements) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(numElements) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(numElements) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(numElements) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(numElements) { 0L }
            // For quantized types, GGMLTensorAllocator does not allocate raw data array this way.
            // It's more for unquantized, simple types.
            // The `else` case here might be problematic if a quantized type reaches it.
            // However, this GGMLTensorAllocator is generally for simpler, non-graph allocation scenarios.
            else -> tensor.data = ByteArray(numElements * tensor.type.byteSize.toInt()) { 0 }
        }

        // Update the offset based on byte size, not element count
        offset += numElements.toULong() * tensor.type.byteSize
    }

    // Removed private calculateTensorByteSize from GGMLTensorAllocator, will use global one from GGMLTypes.kt

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
 * Now supports backend-specific buffer allocation.
 */
class GGMLGraphAllocator {
    // Tensor allocator for each buffer
    var tensorAllocators = mutableListOf<GGMLDynTensorAllocator>()

    // Backend buffers
    var buffers = mutableListOf<ByteArray?>()
    
    // Backend buffer objects (new)
    var backendBuffers = mutableListOf<GGMLBackendBuffer?>()

    // Map to store usage information for each tensor
    private val tensorUsageMap = mutableMapOf<GGMLTensor, TensorUsageInfo>()
    
    // Backend for this allocator
    var backend: GGMLBackend? = null
    
    // Context associated with this allocator
    var context: GGMLContext = GGMLContext()

    /**
     * Creates a new graph allocator with a specific backend.
     */
    constructor(backend: GGMLBackend? = null) {
        this.backend = backend
        
        // Create a default buffer
        val defaultBufferSize = 1024 * 1024
        
        if (backend != null) {
            // Use backend buffer
            val backendBuffer = backend.allocBuffer(defaultBufferSize.toULong())
            if (backendBuffer != null) {
                backendBuffers.add(backendBuffer)
                // For CPU backend, we can still access the underlying ByteArray
                if (backendBuffer is GGMLCpuBuffer) {
                    buffers.add(backendBuffer.getBase() as ByteArray)
                } else {
                    buffers.add(null) // Non-CPU backends don't expose ByteArray
                }
            } else {
                // Fallback to regular ByteArray
                buffers.add(ByteArray(defaultBufferSize))
                backendBuffers.add(null)
            }
        } else {
            // Fallback to regular ByteArray
            buffers.add(ByteArray(defaultBufferSize))
            backendBuffers.add(null)
        }
        
        tensorAllocators.add(GGMLDynTensorAllocator(bufferSize = defaultBufferSize.toULong()))
    }

    /**
     * Creates a new graph allocator (legacy constructor).
     */
    constructor() : this(null)

    /**
     * Analyzes the computation graph to understand tensor usage patterns.
     * This information can be used for memory optimization strategies like
     * inplace operations and memory reuse.
     */
    private fun analyzeTensorUsage(graph: GGMLCGraph) {
        tensorUsageMap.clear()

        // Initialize map for all unique tensors in the graph (leafs and nodes)
        // and mark output tensors.
        val allTensors = (graph.leafs.filterNotNull() + graph.nodes.filterNotNull()).distinct()
        for (tensor in allTensors) {
            // isOutput() method was added to GGMLTensor in GGMLTypes.kt
            tensorUsageMap.getOrPut(tensor) { TensorUsageInfo() }.isOutputTensor = tensor.isOutput()
        }

        // Populate numChildren and numViews
        for (node in graph.nodes) {
            if (node == null) continue

            // Increment numViews for the source of a view tensor
            // ggml_is_view() is defined at the end of this file.
            if (ggml_is_view(node) && node.viewSrc != null) {
                tensorUsageMap[node.viewSrc!!]?.let { it.numViews++ }
            }

            // Increment numChildren for each source tensor
            for (j in 0 until GGML_MAX_SRC) {
                val srcTensor = node.src[j]
                if (srcTensor != null) {
                    tensorUsageMap[srcTensor]?.let { it.numChildren++ }
                }
            }
        }
        // ownsMemory will be determined during allocation/memory planning phase
    }

    private fun ensureBufferCapacity(bufferId: Int, requiredSize: ULong) {
        if (bufferId < 0 || bufferId >= buffers.size || bufferId >= tensorAllocators.size) {
            // Or throw an IllegalArgumentException, depending on desired error handling
            println("Error: Invalid bufferId $bufferId")
            return
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

            // Add the new check:
            if (newSize <= 0 && requiredSize > 0uL) {
                throw IllegalArgumentException(
                    "Invalid buffer size for buffer $bufferId: " +
                    "original requiredSize $requiredSize (ULong) resulted in " +
                    "non-positive effective size $newSize (Int) for ByteArray construction. " +
                    "This may indicate an overflow from ULong to Int or an invalid input."
                )
            }
            // If requiredSize is 0uL, newSize will be 0. ByteArray(0) is valid.
            // The condition above ensures that if requiredSize was > 0, newSize must also be > 0.

            buffers[bufferId] = ByteArray(newSize) // Create/resize the actual buffer. If newSize is 0, this is ByteArray(0).
            tensorAllocators[bufferId].reset(newSize.toULong()) // Reset the allocator with the new size
        }
    }


    /**
     * Allocates memory for all tensors in a computation graph.
     *
     * @param graph The computation graph to allocate memory for
     * @return True if allocation was successful, false otherwise
     */
    fun allocateGraph(graph: GGMLCGraph): Boolean {
        analyzeTensorUsage(graph) // Analyze usage first

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
                allocateTensor(node, 0) // Current node (child) gets its memory
            }

            // After node is allocated (and potentially reused parent's memory),
            // check if any of its parents can be freed.
            for (j in 0 until GGML_MAX_SRC) {
                val parentTensor = node.src[j]
                if (parentTensor != null) {
                    val parentUsageInfo = tensorUsageMap[parentTensor]
                        ?: continue // Should have been populated by analyzeTensorUsage

                    parentUsageInfo.numChildren--

                    if (parentUsageInfo.numChildren == 0 && parentUsageInfo.numViews == 0) {
                        if (ggml_is_view(parentTensor)) {
                            parentTensor.viewSrc?.let { viewSrc ->
                                tensorUsageMap[viewSrc]?.let { viewSrcUsage ->
                                    viewSrcUsage.numViews--
                                    if (viewSrcUsage.numChildren == 0 && viewSrcUsage.numViews == 0 &&
                                        viewSrcUsage.ownsMemory && !viewSrcUsage.isOutputTensor && viewSrcUsage.bufferId != -1) {
                                        tensorAllocators[viewSrcUsage.bufferId].freeTensor(
                                            viewSrcUsage.dataOffset,
                                            viewSrcUsage.calculatedSize,
                                            viewSrc
                                        )
                                        viewSrcUsage.ownsMemory = false
                                    }
                                }
                            }
                        } else if (parentUsageInfo.ownsMemory && !parentUsageInfo.isOutputTensor && parentUsageInfo.bufferId != -1) {
                            tensorAllocators[parentUsageInfo.bufferId].freeTensor(
                                parentUsageInfo.dataOffset,
                                parentUsageInfo.calculatedSize,
                                parentTensor
                            )
                            parentUsageInfo.ownsMemory = false
                        }
                    }
                }
            }
        }

        val calculatedMaxSize = getBufferSize(0)
        ensureBufferCapacity(0, calculatedMaxSize)

        return true
    }

    /**
     * Public helper: allocate a new tensor with given type and shape, owned by this allocator.
     * Useful for examples/tests to create leaf tensors without building a graph first.
     */
    fun allocateTensor(type: GGMLType, ne: LongArray): GGMLTensor {
        val t = GGMLTensor(type = type)
        // Pad/assign ne
        val shape = LongArray(GGML_MAX_DIMS) { 1L }
        val limit = if (ne.size < GGML_MAX_DIMS) ne.size else GGML_MAX_DIMS
        for (i in 0 until limit) { shape[i] = ne[i] }
        t.ne = shape
        t.nb = calculateContiguousStrides(t.ne, t.type, t.rank())

        // Register minimal usage info so internal allocateTensor() can work
        tensorUsageMap[t] = TensorUsageInfo()
        allocateTensor(t, 0)
        return t
    }

    /**
     * Allocates memory for a tensor.
     *
     * @param tensor The tensor to allocate memory for
     * @param bufferId The ID of the buffer to allocate from (default or chosen by strategy)
     */
    // Public overload to allocate an already-constructed tensor in the allocator's buffer
    fun allocateTensor(tensor: GGMLTensor, bufferId: Int = 0) {
        val tensorUsage = tensorUsageMap[tensor]
            ?: throw IllegalStateException("TensorUsageInfo not found for tensor ${tensor.name}. analyzeTensorUsage must be called first.")

        // Handle Pre-allocated/View Tensors
        if (tensor.data != null || ggml_is_view(tensor)) {
            tensorUsage.ownsMemory = false // Does not own memory from this allocator
            // For views, bufferId and dataOffset should ideally be set based on viewSrc.
            // This might require view-specific initialization logic elsewhere.
            // If it's a view and viewSrc is in tensorUsageMap, copy allocation details.
            if (ggml_is_view(tensor) && tensor.viewSrc != null) {
                tensorUsageMap[tensor.viewSrc!!]?.let { srcUsage ->
                    tensor.bufferId = srcUsage.bufferId
                    // tensor.viewOffs is the byte offset from the start of viewSrc's data region (which begins at srcUsage.dataOffset).
                    // Thus, the absolute offset of the view tensor's data is srcUsage.dataOffset + tensor.viewOffs.
                    tensor.dataOffset = srcUsage.dataOffset + tensor.viewOffs

                    tensorUsage.bufferId = srcUsage.bufferId
                    tensorUsage.dataOffset = tensor.dataOffset
                    // Ensure calculatedSize for the view tensor reflects its own dimensions and type, using byte size.
                    // Now calls the global/internal function from GGMLTypes.kt
                    tensorUsage.calculatedSize = calculateTensorByteSize(tensor)
                }
            }
            return // No new allocation needed
        }

        // tensorUsage is already fetched.

        // Use the global/internal function from GGMLTypes.kt
        val tensorCalculatedByteSize = calculateTensorByteSize(tensor)

        // New handling for zero-sized tensors, placed before inplace allocation logic
        if (tensorCalculatedByteSize == 0uL) {
            if (tensor.isValidZeroSizedTensor()) {
                // Ensure this isn't a view that has already been "allocated" by its source.
                // The initial check `if (tensor.data != null || ggml_is_view(tensor))` handles this.
                // However, if it's a view of a zero-sized tensor, its viewSrc might be what needs allocation.
                // The current logic for views seems to copy bufferId/dataOffset from srcUsage.
                // If viewSrc itself is zero-sized and gets processed here, this is fine.
                // A view of a non-zero tensor that results in a zero-sized view (e.g. ne=[0,...]) is also fine.

                println("Info: Tensor ${tensor.name} type ${tensor.type} (dims: ${tensor.ne.joinToString()}) is validly zero-sized. Allocating 0 bytes.")

                val actualBufferId = bufferId
                val offset = tensorAllocators[actualBufferId].allocate(0uL, tensor)

                tensor.bufferId = actualBufferId
                tensor.dataOffset = offset

                tensorUsage.ownsMemory = true
                tensorUsage.bufferId = actualBufferId
                tensorUsage.dataOffset = offset
                tensorUsage.calculatedSize = 0uL // It's a zero-byte allocation
            } else {
                // This case implies numElements > 0 but type.byteSize is 0 for a data type,
                // which should have been warned about during stride calculation or type definition.
                // Or, numElements became 0 in an unexpected way for a non-COUNT type.
                println("Warning: Tensor ${tensor.name} type ${tensor.type} (dims: ${tensor.ne.joinToString()}) has an unexpected calculated byte size 0. Skipping allocation.")
                tensorUsage.ownsMemory = false
                tensorUsage.calculatedSize = 0uL
                tensor.bufferId = -1
                tensor.dataOffset = 0uL
                tensorUsage.bufferId = -1
                tensorUsage.dataOffset = 0uL
            }
            return // Processed zero-sized tensor, skip further allocation logic
        }

        // Inplace allocation logic starts here
        var inplaceFound = false
        // val tensorCalculatedSize = calculateTensorSize(tensor) // This is now tensorCalculatedByteSize

        if (tensor.op.canBeInplace) {
            for (srcIdx in tensor.src.indices) {
                val parentTensor = tensor.src[srcIdx]
                if (parentTensor != null) {
                    val parentUsageInfo = tensorUsageMap[parentTensor]
                        ?: continue // Should not happen if all tensors are in map

                    val parentCalculatedSize = parentUsageInfo.calculatedSize // Relies on parent being processed already for its size

                    // Eligibility checks for inplace
                    if (parentUsageInfo.ownsMemory &&
                        parentUsageInfo.numChildren == 1 && // Current tensor is the sole effective consumer
                        parentUsageInfo.numViews == 0 &&
                        !parentUsageInfo.isOutputTensor &&
                        tensor.type == parentTensor.type &&
                        tensorCalculatedByteSize == parentCalculatedSize && // Ensure sizes match - FIXED
                        parentUsageInfo.bufferId != -1) {

                        tensor.bufferId = parentUsageInfo.bufferId
                        tensor.dataOffset = parentUsageInfo.dataOffset

                        tensorUsage.ownsMemory = true // This tensor now owns the memory segment
                        tensorUsage.bufferId = tensor.bufferId
                        tensorUsage.dataOffset = tensor.dataOffset
                        tensorUsage.calculatedSize = tensorCalculatedByteSize // FIXED

                        parentUsageInfo.ownsMemory = false // Parent relinquishes ownership

                        inplaceFound = true
                        break // Found an inplace source
                    }
                }
            }
        }

        if (!inplaceFound) {
            // Standard allocation if no inplace opportunity or tensor is not inplace eligible
            // The zero-size check is now handled above.
            // This part is only for non-zero sized tensors.

            val actualBufferId = bufferId // Use the passed-in bufferId for now
            // Pass tensorCalculatedByteSize to the allocator
            val offset = tensorAllocators[actualBufferId].allocate(tensorCalculatedByteSize, tensor)

            tensor.bufferId = actualBufferId
            tensor.dataOffset = offset

            tensorUsage.ownsMemory = true
            tensorUsage.bufferId = actualBufferId
            tensorUsage.dataOffset = offset
            tensorUsage.calculatedSize = tensorCalculatedByteSize // Store the byte size
        }
    }

    /**
     * Reserve at least the given number of bytes in the primary buffer.
     */
    fun reserve(bytes: Int) {
        ensureBufferCapacity(0, bytes.toULong())
    }

    // calculateTensorSize was renamed to calculateTensorByteSize and modified.
    // The old calculateTensorSize (which calculated num elements) is effectively replaced by tensor.numElements()

    /**
     * Reserves memory for a computation graph without actually allocating it.
     *
     * @param graph The computation graph to reserve memory for
     * @return True if reservation was successful, false otherwise
     */
    @Suppress("unused")
    fun reserveGraph(graph: GGMLCGraph): Boolean {
        analyzeTensorUsage(graph) // Analyze usage first

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
        // Calculate the byte size of the tensor using the global/internal function
        val byteSize = calculateTensorByteSize(tensor)

        // Reserve memory from the tensor allocator
        tensorAllocators[bufferId].allocate(byteSize, tensor)
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
