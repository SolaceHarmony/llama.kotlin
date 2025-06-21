package ai.solace.llamakotlin.backends.cpu

import ai.solace.llamakotlin.backends.Backend
import ai.solace.llamakotlin.core.GGMLTensor
import ai.solace.llamakotlin.core.GGMLGraphAllocator
import ai.solace.llamakotlin.core.GGMLContext
import ai.solace.llamakotlin.core.GGMLOp

/**
 * A CPU backend implementation for executing GGML operations.
 * This backend uses the computation functions defined in `GGMLComputeOps.kt`.
 */
object CPUBackend : Backend {
    /**
     * Executes a single operation on the CPU.
     *
     * This function will dispatch to the appropriate compute function from `GGMLComputeOps.kt`
     * based on the `node.op` type.
     *
     * @param node The tensor representing the operation to execute.
     * @param graphAllocator The graph allocator for tensor data.
     * @param context The GGML context.
     * @return The tensor containing the result of the operation.
     * @throws NotImplementedError if the operation specified in `node.op` is not supported.
     */
    override fun executeOp(
        node: GGMLTensor,
        graphAllocator: GGMLGraphAllocator,
        context: GGMLContext
    ): GGMLTensor {
        val src0 = node.src[0]
        val src1 = node.src[1]

        // Most compute functions return a *new* tensor whose data is managed by graphAllocator.
        return when (node.op) {
            GGMLOp.ADD -> {
                require(src0 != null && src1 != null) { "ADD operation requires two source tensors." }
                computeAdd(graphAllocator, context, src0, src1)
            }
            GGMLOp.SUB -> {
                require(src0 != null && src1 != null) { "SUB operation requires two source tensors." }
                computeSub(graphAllocator, context, src0, src1)
            }
            GGMLOp.MUL -> {
                require(src0 != null && src1 != null) { "MUL operation requires two source tensors." }
                computeMul(graphAllocator, context, src0, src1)
            }
            GGMLOp.DIV -> {
                require(src0 != null && src1 != null) { "DIV operation requires two source tensors." }
                computeDiv(graphAllocator, context, src0, src1)
            }
            GGMLOp.MUL_MAT -> {
                require(src0 != null && src1 != null) { "MUL_MAT operation requires two source tensors." }
                computeMatMul(graphAllocator, context, src0, src1)
            }
            GGMLOp.NEG -> {
                require(src0 != null) { "NEG operation requires one source tensor." }
                computeNeg(graphAllocator, context, src0)
            }
            GGMLOp.RELU -> {
                require(src0 != null) { "RELU operation requires one source tensor." }
                computeRelu(graphAllocator, context, src0)
            }
            GGMLOp.GELU -> {
                require(src0 != null) { "GELU operation requires one source tensor." }
                computeGelu(graphAllocator, context, src0)
            }
            GGMLOp.SILU -> {
                require(src0 != null) { "SILU operation requires one source tensor." }
                computeSilu(graphAllocator, context, src0)
            }
            GGMLOp.RMS_NORM -> {
                require(src0 != null) { "RMS_NORM operation requires one source tensor." }
                // RMS_NORM also needs an epsilon parameter. This is usually stored in node.opParams
                // Assuming opParams[0] holds float epsilon bits for now.
                // This part needs careful handling of how opParams are structured and passed.
                // For now, let's assume a default epsilon or that it's handled inside computeRMSNorm if not passed.
                // The current computeRMSNorm takes eps as a direct Float argument.
                // This implies the graph node (GGMLTensor for op RMS_NORM) must store epsilon.
                // Let's assume node.opParams[0] stores Float.toRawBits(eps)
                if (node.opParams.isEmpty()) throw IllegalStateException("RMS_NORM opParams missing epsilon.")
                val epsBits = node.opParams[0]
                val eps = Float.fromBits(epsBits)
                computeRMSNorm(graphAllocator, context, src0, eps)
            }
            // Add cases for other operations as they are implemented in GGMLComputeOps
            // GGMLOp.SQR -> computeSqr(graphAllocator, context, src0!!)
            // GGMLOp.SCALE -> computeScale(graphAllocator, context, src0!!, src1!!) // Assuming computeScale exists
            // GGMLOp.CPY, GGMLOp.RESHAPE, GGMLOp.VIEW, GGMLOp.PERMUTE, GGMLOp.TRANSPOSE,
            // GGMLOp.GET_ROWS, GGMLOp.DIAG_MASK_INF, GGMLOp.SOFT_MAX, etc.
            else -> throw NotImplementedError("CPUBackend.executeOp not implemented for op: ${node.op}")
        }
    }
}
