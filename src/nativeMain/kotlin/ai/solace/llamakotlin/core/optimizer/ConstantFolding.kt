package ai.solace.llamakotlin.core.optimizer

import ai.solace.llamakotlin.core.*

/**
 * Checks if a tensor can be considered constant for folding purposes.
 *
 * A tensor is constant if:
 * - It has no producer operation (it's a leaf node like a weight or bias).
 * - Its data is considered available and fixed.
 * - It's not marked as a graph output.
 * - It has no gradient (or gradient is not relevant for forward pass constant folding).
 *
 * @param tensor The tensor to check.
 * @param graphAllocator The graph allocator, to check for buffer-based data.
 * @return True if the tensor is constant, false otherwise.
 */
fun isTensorConstant(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): Boolean {
    // 1. Must be a leaf node in terms of graph operations (no producer op)
    if (tensor.op != GGMLOp.NONE) {
        return false
    }

    // 2. Data must be available and fixed.
    // This can mean tensor.data is populated (e.g. for standalone tensors not yet in graph buffer)
    // OR tensor.bufferId is valid and points to data in graphAllocator.
    val hasDirectData = tensor.data != null // e.g. FloatArray, ByteArray, etc.
    val hasBufferData = tensor.bufferId != -1 && graphAllocator.buffers.containsKey(tensor.bufferId)

    if (!hasDirectData && !hasBufferData) {
        // If no op, but also no data, it might be an uninitialized parameter or an issue.
        // For constant folding, we need its value.
        return false
    }

    // 3. Must not be a designated output tensor of the graph.
    if (tensor.isOutput()) {
        return false
    }

    // For op == GGMLOp.NONE (leaf/parameter tensors), the presence of a .grad field
    // (for accumulating gradients during backward pass) does not make its value
    // non-constant for the forward pass. Its data is fixed.
    // The primary check tensor.op == GGMLOp.NONE is sufficient to determine constancy
    // for the purpose of folding operations in a forward pass.
    }

    return true
}

/**
 * Defines the set of operations that are considered foldable.
 */
val FOLDABLE_OPS: Set<GGMLOp> = setOf(
    GGMLOp.ADD,
    GGMLOp.SUB,
    GGMLOp.MUL,
    GGMLOp.DIV, // Division by zero needs care if constants could lead to it.
    GGMLOp.SQR,
    GGMLOp.NEG,
    GGMLOp.RELU,
    GGMLOp.GELU,
    GGMLOp.SILU,
    GGMLOp.SCALE // SCALE is foldable if src1 (the scale) is a single constant scalar.
    // Future additions: NORM, RMS_NORM (if weights are constant), etc.
)

/**
 * Checks if a graph node (tensor) is suitable for constant folding.
 *
 * A node is foldable if:
 * - Its operation is in a predefined list of foldable operations.
 * - All its active source tensors are constant.
 * - It's not an output node itself.
 *
 * @param node The graph node (tensor) to check.
 * @param graphAllocator The graph allocator to verify source tensor data.
 * @return True if the node is foldable, false otherwise.
 */
fun isNodeFoldable(node: GGMLTensor, graphAllocator: GGMLGraphAllocator): Boolean {
    if (node.op == GGMLOp.NONE) return false // Already a constant or leaf
    if (node.isOutput()) return false      // Don't fold output nodes directly

    if (node.op !in FOLDABLE_OPS) {
        return false
    }

    // Check sources:
    // For unary ops (like NEG, RELU, GELU, SILU, SQR): src[0] must be constant.
    // For binary ops (like ADD, SUB, MUL, DIV, SCALE): src[0] and src[1] must be constant.

    if (node.src[0] == null) return false // Should not happen for valid ops in FOLDABLE_OPS
    if (!isTensorConstant(node.src[0]!!, graphAllocator)) {
        return false
    }

    // Handle ops requiring a second source
    val binaryOps = setOf(GGMLOp.ADD, GGMLOp.SUB, GGMLOp.MUL, GGMLOp.DIV, GGMLOp.SCALE)
    if (node.op in binaryOps) {
        if (node.src[1] == null) return false // Binary op needs two sources
        if (!isTensorConstant(node.src[1]!!, graphAllocator)) {
            return false
        }
        // Special check for SCALE: src1 should be a scalar constant.
        if (node.op == GGMLOp.SCALE) {
            if (node.src[1]!!.numElements() != 1L) {
                return false // Scale factor must be a scalar
            }
        }
    }

    // Add more specific checks if needed (e.g., for tensor shapes if an op requires it for folding)

    return true
}

/**
 * Performs constant folding on a computation graph.
 *
 * Iterates through the graph nodes. If a node is foldable (i.e., its operation
 * can be computed at compile/build time because its inputs are constant),
 * its operation is performed, and the node is replaced with the constant result.
 *
 * @param cgraph The computation graph to optimize.
 * @param graphAllocator The graph allocator for accessing tensor data.
 * @param context The GGML context (may be needed for some compute operations, though ideally not for simple ones).
 * @return The number of nodes folded in this pass.
 */
fun foldConstantsInGraph(
    cgraph: GGMLCGraph,
    graphAllocator: GGMLGraphAllocator,
    context: GGMLContext // Pass context for compute functions
): Int {
    var foldsPerformed = 0
    val nodes = cgraph.nodes.filterNotNull() // Process only non-null nodes

    for (node in nodes) {
        if (isNodeFoldable(node, graphAllocator)) {
            println("Node ${node.name} (op: ${node.op}) is foldable.") // Logging for debug

            val src0 = node.src[0]!!
            val src1 = node.src[1] // Null for unary ops

            // Execute the operation
            val foldedResultTensor: GGMLTensor? = when (node.op) {
                GGMLOp.ADD -> computeAdd(graphAllocator, context, src0, src1!!)
                GGMLOp.SUB -> computeSub(graphAllocator, context, src0, src1!!)
                GGMLOp.MUL -> computeMul(graphAllocator, context, src0, src1!!)
                GGMLOp.DIV -> computeDiv(graphAllocator, context, src0, src1!!)
                GGMLOp.SQR -> null // computeSqr(graphAllocator, context, src0) // Placeholder
                GGMLOp.NEG -> computeNeg(graphAllocator, context, src0)
                GGMLOp.RELU -> computeRelu(graphAllocator, context, src0)
                GGMLOp.GELU -> computeGelu(graphAllocator, context, src0)
                GGMLOp.SILU -> computeSilu(graphAllocator, context, src0)
                GGMLOp.SCALE -> computeScale(graphAllocator, context, src0, src1!!) // Placeholder for computeScale
                else -> {
                    println("Warning: Op ${node.op} was in FOLDABLE_OPS but no compute case handled.")
                    null
                }
            }

            if (foldedResultTensor != null) {
                // The computeXYZ functions now return tensors whose data is in graphAllocator.
                // We need to make 'node' look like this 'foldedResultTensor' in terms of value,
                // effectively replacing it with a constant.

                // Option 1: Copy data from foldedResultTensor's buffer to node's original buffer (if it had one and it's suitable)
                // This is complex due to buffer management.

                // Option 2: Point 'node' to foldedResultTensor's buffer and data properties.
                // This means graphAllocator now manages the data for 'node'.
                node.bufferId = foldedResultTensor.bufferId
                node.dataOffset = foldedResultTensor.dataOffset
                node.data = null // Data is now in graphAllocator's buffer, not directly in GGMLTensor.data as a raw array.
                                // Or, if compute ops set .data to a raw array, then this should be node.data = foldedResultTensor.data.
                                // The current compute ops (after recent changes) DO return tensors whose data is in graphAllocator.
                                // So, node.data should be nullified if we are relying on bufferId.

                node.type = foldedResultTensor.type // Type might change (e.g. if op promotes, though unlikely for these)
                node.op = GGMLOp.NONE             // Mark as constant
                node.src.fill(null)               // Remove dependencies
                node.opParams.fill(0)             // Clear op params

                // Important: The buffer previously held by foldedResultTensor is now "owned" by 'node'.
                // The foldedResultTensor object itself is temporary and can be discarded.
                // We must ensure its buffer isn't accidentally freed if graphAllocator tracks tensors too.
                // The current graphAllocator.buffers is a MutableMap<Int, ByteArray>, so it's just raw buffers.

                foldsPerformed++
                println("Folded ${node.name}. New type: ${node.type}, New op: ${node.op}, BufferId: ${node.bufferId}")

                // If foldedResultTensor itself allocated a new buffer that is different from src0 or src1,
                // and if src0/src1 are no longer needed by other nodes, their buffers could potentially be freed.
                // This level of memory management is more advanced (requires usage counts).
                // For now, we assume graphAllocator handles buffers, and we just re-point 'node'.
            }
        }
    }

    return foldsPerformed
}

// Placeholder for computeScale, assuming it takes (alloc, ctx, src0, scalar_src1)
// Actual computeScale might need to be more general.
fun computeScale(graphAllocator: GGMLGraphAllocator, context: GGMLContext, a: GGMLTensor, scaleTensor: GGMLTensor): GGMLTensor {
    // Assuming scaleTensor is a scalar F32 for now for simplicity in folding.
    if (scaleTensor.type != GGMLType.F32 || scaleTensor.numElements() != 1L) {
        throw IllegalArgumentException("Constant folding for SCALE requires scalar F32 scale factor.")
    }
    val scaleFactor = scaleTensor.getFloat(graphAllocator, 0) // Get the scalar value

    val result = GGMLTensor(type = a.type)
    result.ne = a.ne.copyOf()
    result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    val resultByteSize = totalSize * result.type.byteSize.toInt()
    val resultBuffer = ByteArray(resultByteSize)
    result.bufferId = graphAllocator.allocateBuffer(resultBuffer)
    result.dataOffset = 0uL

    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(a, totalSize) { _, indices ->
                result.setFloat(graphAllocator, a.getFloat(graphAllocator, *indices) * scaleFactor, *indices)
            }
        }
        GGMLType.F16 -> {
             applyNDIter(a, totalSize) { _, indices ->
                result.setHalf(graphAllocator, a.getHalf(graphAllocator, *indices) * scaleFactor, *indices)
            }
        }
        // Add other types as needed, potentially with type promotion.
        else -> throw NotImplementedError("computeScale constant folding not implemented for input type ${a.type}")
    }
    return result
}

// TODO: Implement computeSqr if it's not already in GGMLComputeOps.kt
// fun computeSqr(graphAllocator: GGMLGraphAllocator, context: GGMLContext, a: GGMLTensor): GGMLTensor { ... }
