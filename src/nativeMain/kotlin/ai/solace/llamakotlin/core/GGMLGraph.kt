package ai.solace.llamakotlin.core

/**
 * Kotlin Native port of GGML computation graph functionality.
 * This file contains the implementation of computation graph building and execution.
 */

/**
 * Flag for parameter tensors that require gradients
 */
const val GGML_TENSOR_FLAG_PARAM = 1

/**
 * Adds a tensor to another tensor, or sets the result if the first tensor is null.
 *
 * @param context The GGML context
 * @param a The first tensor (can be null)
 * @param b The second tensor
 * @param zeroTable A set of tensors that are known to be zero
 * @return The result tensor
 */
private fun addOrSet(context: GGMLContext, a: GGMLTensor?, b: GGMLTensor, zeroTable: MutableSet<GGMLTensor>): GGMLTensor {
    return if (a == null || a in zeroTable) {
        // If a is null or known to be zero, return b
        b
    } else {
        // Otherwise, add a and b
        add(context, a, b)
    }
}

/**
 * Subtracts a tensor from another tensor, or sets the negation of the second tensor if the first tensor is null.
 *
 * @param context The GGML context
 * @param a The first tensor (can be null)
 * @param b The second tensor
 * @param zeroTable A set of tensors that are known to be zero
 * @return The result tensor
 */
private fun subOrSet(context: GGMLContext, a: GGMLTensor?, b: GGMLTensor, zeroTable: MutableSet<GGMLTensor>): GGMLTensor {
    return if (a == null || a in zeroTable) {
        // If a is null or known to be zero, return -b
        // We need to implement a negate function
        val result = GGMLTensor(type = b.type)
        result.op = GGMLOp.NEG
        result.src[0] = b

        // If the context requests immediate computation, perform it now
        if (context.computeImmediately) {
            // We need to implement computeNeg
            throw NotImplementedError("NEG operation not implemented yet")
        } else {
            result
        }
    } else {
        // Otherwise, subtract b from a
        // We need to implement a subtract function
        val result = GGMLTensor(type = a.type)
        result.op = GGMLOp.SUB
        result.src[0] = a
        result.src[1] = b

        // If the context requests immediate computation, perform it now
        if (context.computeImmediately) {
            // We need to implement computeSub
            throw NotImplementedError("SUB operation not implemented yet")
        } else {
            result
        }
    }
}

/**
 * Sets a tensor as a parameter for automatic differentiation.
 *
 * @param tensor The tensor to set as a parameter
 */
fun setParam(tensor: GGMLTensor) {
    tensor.flags = tensor.flags or GGML_TENSOR_FLAG_PARAM
}

/**
 * Computes the gradients for a tensor based on its operation type.
 *
 * @param context The GGML context
 * @param tensor The tensor to compute gradients for
 * @param zeroTable A set of tensors that are known to be zero
 */
private fun computeBackward(context: GGMLContext, tensor: GGMLTensor, zeroTable: MutableSet<GGMLTensor>) {
    val src0 = tensor.src[0]
    val src1 = tensor.src[1]

    when (tensor.op) {
        GGMLOp.DUP -> {
            if (src0?.grad != null) {
                src0.grad = addOrSet(context, src0.grad, tensor.grad!!, zeroTable)
            }
        }
        GGMLOp.ADD -> {
            if (src0?.grad != null) {
                src0.grad = addOrSet(context, src0.grad, tensor.grad!!, zeroTable)
            }
            if (src1?.grad != null) {
                // TODO: Handle broadcasting case with repeat_back
                src1.grad = addOrSet(context, src1.grad, tensor.grad!!, zeroTable)
            }
        }
        GGMLOp.SUB -> {
            if (src0?.grad != null) {
                src0.grad = addOrSet(context, src0.grad, tensor.grad!!, zeroTable)
            }
            if (src1?.grad != null) {
                src1.grad = subOrSet(context, src1.grad, tensor.grad!!, zeroTable)
            }
        }
        GGMLOp.MUL -> {
            if (src0?.grad != null) {
                src0.grad = addOrSet(
                    context,
                    src0.grad,
                    mul(context, src1!!, tensor.grad!!),
                    zeroTable
                )
            }
            if (src1?.grad != null) {
                src1.grad = addOrSet(
                    context,
                    src1.grad,
                    mul(context, src0!!, tensor.grad!!),
                    zeroTable
                )
            }
        }
        GGMLOp.DIV -> {
            // TODO: Implement DIV backward pass
            throw NotImplementedError("DIV backward pass not implemented yet")
        }
        GGMLOp.SQR -> {
            // TODO: Implement SQR backward pass
            throw NotImplementedError("SQR backward pass not implemented yet")
        }
        GGMLOp.SQRT -> {
            // TODO: Implement SQRT backward pass
            throw NotImplementedError("SQRT backward pass not implemented yet")
        }
        GGMLOp.SUM -> {
            // TODO: Implement SUM backward pass
            throw NotImplementedError("SUM backward pass not implemented yet")
        }
        GGMLOp.MEAN -> {
            // TODO: Implement MEAN backward pass
            throw NotImplementedError("MEAN backward pass not implemented yet")
        }
        GGMLOp.REPEAT -> {
            // TODO: Implement REPEAT backward pass
            throw NotImplementedError("REPEAT backward pass not implemented yet")
        }
        GGMLOp.ABS -> {
            // TODO: Implement ABS backward pass
            throw NotImplementedError("ABS backward pass not implemented yet")
        }
        GGMLOp.SGN -> {
            // TODO: Implement SGN backward pass
            throw NotImplementedError("SGN backward pass not implemented yet")
        }
        GGMLOp.NEG -> {
            if (src0?.grad != null) {
                src0.grad = subOrSet(context, src0.grad, tensor.grad!!, zeroTable)
            }
        }
        GGMLOp.STEP -> {
            // TODO: Implement STEP backward pass
            throw NotImplementedError("STEP backward pass not implemented yet")
        }
        GGMLOp.RELU -> {
            if (src0?.grad != null) {
                // grad_src0 += grad_tensor * (src0 > 0)
                // TODO: Implement RELU backward pass properly
                throw NotImplementedError("RELU backward pass not implemented yet")
            }
        }
        GGMLOp.GELU -> {
            if (src0?.grad != null) {
                // TODO: Implement GELU backward pass
                throw NotImplementedError("GELU backward pass not implemented yet")
            }
        }
        GGMLOp.MUL_MAT -> {
            // TODO: Implement MUL_MAT backward pass
            throw NotImplementedError("MUL_MAT backward pass not implemented yet")
        }
        else -> {
            // For other operations, we'll implement later
            throw NotImplementedError("Backward pass for operation ${tensor.op} not implemented yet")
        }
    }
}

/**
 * Builds a computation graph from a tensor.
 *
 * @param tensor The tensor to build the graph from
 * @param cgraph The computation graph to build
 * @param visited A set of visited tensors to avoid cycles
 */
private fun buildForwardImpl(tensor: GGMLTensor, cgraph: GGMLCGraph, visited: MutableSet<GGMLTensor>) {
    // If we've already visited this tensor, return
    if (tensor in visited) {
        return
    }

    // Mark the tensor as visited
    visited.add(tensor)

    // If the tensor has no operation, it's a leaf node
    if (tensor.op == GGMLOp.NONE) {
        cgraph.leafs[cgraph.nLeafs++] = tensor
        return
    }

    // Recursively build the graph for the source tensors
    for (i in 0 until GGML_MAX_SRC) {
        val src = tensor.src[i] ?: break
        buildForwardImpl(src, cgraph, visited)
    }

    // Add the tensor to the graph
    cgraph.nodes[cgraph.nNodes++] = tensor
}

/**
 * Builds a computation graph for forward pass.
 *
 * @param cgraph The computation graph to build
 * @param tensor The output tensor
 */
fun buildForward(cgraph: GGMLCGraph, tensor: GGMLTensor) {
    // Reset the graph
    cgraph.nNodes = 0
    cgraph.nLeafs = 0

    // Build the graph
    val visited = mutableSetOf<GGMLTensor>()
    buildForwardImpl(tensor, cgraph, visited)

    // Set the order to forward
    cgraph.order = GGMLCGraphEvalOrder.FORWARD
}

/**
 * Builds a backward computation graph for automatic differentiation.
 *
 * @param context The GGML context
 * @param gf The forward computation graph
 * @param gb The backward computation graph to build
 * @param keep Whether to keep the original gradients
 */
fun buildBackward(context: GGMLContext, gf: GGMLCGraph, gb: GGMLCGraph, keep: Boolean = true) {
    // Check that the forward graph has nodes
    if (gf.nNodes <= 0) {
        throw IllegalArgumentException("Forward graph has no nodes")
    }

    // Check that the forward graph has gradients
    if (gf.grads.isEmpty()) {
        throw IllegalArgumentException("Forward graph has no gradients")
    }

    // If we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
    if (keep) {
        for (i in 0 until gf.nNodes) {
            val node = gf.nodes[i] ?: continue

            if (node.grad != null) {
                // Create a duplicate of the gradient tensor
                val gradDup = GGMLTensor(type = node.grad!!.type)
                for (j in 0 until GGML_MAX_DIMS) {
                    gradDup.ne[j] = node.grad!!.ne[j]
                    gradDup.nb[j] = node.grad!!.nb[j]
                }
                gradDup.data = node.grad!!.data

                node.grad = gradDup
                gf.grads[i] = node.grad
            }
        }
    }

    // Remember original gradients which start with zero values
    val zeroTable = mutableSetOf<GGMLTensor>()
    for (i in 0 until gf.nNodes) {
        if (gf.grads[i] != null) {
            zeroTable.add(gf.grads[i]!!)
        }
    }

    // Compute gradients for each node in reverse order
    for (i in gf.nNodes - 1 downTo 0) {
        val node = gf.nodes[i] ?: continue

        // Compute gradients for this node
        if (node.grad != null) {
            computeBackward(context, node, zeroTable)
        }
    }

    // Build forward graph for parameter gradients
    gb.nNodes = 0
    gb.nLeafs = 0

    for (i in 0 until gf.nNodes) {
        val node = gf.nodes[i] ?: continue

        if (node.flags and GGML_TENSOR_FLAG_PARAM != 0) {
            // This is a parameter tensor, add its gradient to the backward graph
            if (node.grad != null) {
                buildForward(gb, node.grad!!)
            }
        }
    }

    // Set the order to backward
    gb.order = GGMLCGraphEvalOrder.BACKWARD
}

/**
 * Executes a computation graph.
 *
 * @param context The GGML context
 * @param cgraph The computation graph to execute
 */
fun executeGraph(context: GGMLContext, cgraph: GGMLCGraph) {
    // Execute the graph based on the order
    when (cgraph.order) {
        GGMLCGraphEvalOrder.FORWARD -> executeForward(context, cgraph)
        GGMLCGraphEvalOrder.BACKWARD -> executeBackward(context, cgraph)
        else -> throw IllegalArgumentException("Invalid graph order")
    }
}

/**
 * Executes a computation graph in backward order.
 *
 * @param context The GGML context
 * @param cgraph The computation graph to execute
 */
private fun executeBackward(context: GGMLContext, cgraph: GGMLCGraph) {
    // Execute the nodes in order
    // For backward pass, we execute the nodes in the same order as they were added to the graph
    // This is because the backward graph is built by adding nodes in the correct order for backward execution
    for (i in 0 until cgraph.nNodes) {
        val node = cgraph.nodes[i] ?: continue
        executeNode(context, node)
    }
}

/**
 * Executes a computation graph in forward order.
 *
 * @param context The GGML context
 * @param cgraph The computation graph to execute
 */
private fun executeForward(context: GGMLContext, cgraph: GGMLCGraph) {
    // Execute the nodes in order
    for (i in 0 until cgraph.nNodes) {
        val node = cgraph.nodes[i] ?: continue
        executeNode(context, node)
    }
}

/**
 * Executes a single node in the computation graph.
 *
 * @param context The GGML context
 * @param node The node to execute
 */
private fun executeNode(context: GGMLContext, node: GGMLTensor) {
    // Execute the node based on its operation
    when (node.op) {
        GGMLOp.ADD -> {
            val a = node.src[0] ?: throw IllegalArgumentException("ADD operation requires two source tensors")
            val b = node.src[1] ?: throw IllegalArgumentException("ADD operation requires two source tensors")
            node.data = computeAdd(context, a, b).data
        }
        GGMLOp.MUL -> {
            val a = node.src[0] ?: throw IllegalArgumentException("MUL operation requires two source tensors")
            val b = node.src[1] ?: throw IllegalArgumentException("MUL operation requires two source tensors")
            node.data = computeMul(context, a, b).data
        }
        GGMLOp.MUL_MAT -> {
            val a = node.src[0] ?: throw IllegalArgumentException("MUL_MAT operation requires two source tensors")
            val b = node.src[1] ?: throw IllegalArgumentException("MUL_MAT operation requires two source tensors")
            node.data = computeMatMul(context, a, b).data
        }
        GGMLOp.RELU -> {
            val a = node.src[0] ?: throw IllegalArgumentException("RELU operation requires one source tensor")
            node.data = computeRelu(context, a).data
        }
        GGMLOp.GELU -> {
            val a = node.src[0] ?: throw IllegalArgumentException("GELU operation requires one source tensor")
            node.data = computeGelu(context, a).data
        }
        else -> throw NotImplementedError("Operation ${node.op} not implemented yet")
    }
}

/**
 * Creates a new computation graph.
 *
 * @param size The maximum number of nodes in the graph
 * @return The new computation graph
 */
fun createGraph(size: Int): GGMLCGraph {
    return GGMLCGraph(
        size = size,
        nNodes = 0,
        nLeafs = 0,
        nodes = Array(size) { null },
        grads = Array(size) { null },
        leafs = Array(size) { null },
        visitedHashSet = null,
        order = GGMLCGraphEvalOrder.NONE
    )
}
