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
                // Create a mask where src0 > 0
                val mask = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    mask.ne[i] = src0.ne[i]
                    mask.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Create mask data based on the tensor type
                when (src0.type) {
                    GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        val maskData = FloatArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0.0f) 1.0f else 0.0f
                        }
                        mask.data = maskData
                    }
                    GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        val maskData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if ((srcData[i].toFloat() / 32768.0f) > 0.0f) 32767.toShort() else 0
                        }
                        mask.data = maskData
                    }
                    GGMLType.I8 -> {
                        val srcData = src0.data as ByteArray
                        val maskData = ByteArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    GGMLType.I16 -> {
                        val srcData = src0.data as ShortArray
                        val maskData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    GGMLType.I32 -> {
                        val srcData = src0.data as IntArray
                        val maskData = IntArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    GGMLType.I64 -> {
                        val srcData = src0.data as LongArray
                        val maskData = LongArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("RELU backward pass not implemented for type ${src0.type}")
                    }
                }

                // Multiply gradient by mask: grad_tensor * (src0 > 0)
                val gradMasked = mul(context, tensor.grad!!, mask)

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradMasked, zeroTable)
            }
        }
        GGMLOp.GELU -> {
            if (src0?.grad != null) {
                // The derivative of GELU approximation
                // GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                // We need to compute the derivative of this with respect to x

                // Create a tensor for the derivative
                val derivative = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    derivative.ne[i] = src0.ne[i]
                    derivative.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Compute the derivative based on the tensor type
                when (src0.type) {
                    GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        val derivativeData = FloatArray(totalSize)

                        for (i in 0 until totalSize) {
                            val x = srcData[i]
                            val x2 = x * x
                            val x3 = x2 * x

                            // Constants from the GELU approximation
                            val sqrt2OverPi = 0.797885f // sqrt(2/π)
                            val coef = 0.044715f

                            // Inner term: sqrt(2/π) * (x + 0.044715 * x^3)
                            val inner = sqrt2OverPi * (x + coef * x3)

                            // tanh and sech^2 terms
                            val tanhInner = kotlin.math.tanh(inner)
                            val sech2 = 1.0f - tanhInner * tanhInner // sech^2(x) = 1 - tanh^2(x)

                            // Derivative of GELU
                            // 0.5 * (1 + tanh(inner) + x * sech^2(inner) * sqrt(2/π) * (1 + 3 * 0.044715 * x^2))
                            val derivTerm = sqrt2OverPi * (1.0f + 3.0f * coef * x2)
                            derivativeData[i] = 0.5f * (1.0f + tanhInner + x * sech2 * derivTerm)
                        }

                        derivative.data = derivativeData
                    }
                    GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        val derivativeData = ShortArray(totalSize)

                        for (i in 0 until totalSize) {
                            // Convert short to float
                            val x = srcData[i].toFloat() / 32768.0f
                            val x2 = x * x
                            val x3 = x2 * x

                            // Constants from the GELU approximation
                            val sqrt2OverPi = 0.797885f // sqrt(2/π)
                            val coef = 0.044715f

                            // Inner term: sqrt(2/π) * (x + 0.044715 * x^3)
                            val inner = sqrt2OverPi * (x + coef * x3)

                            // tanh and sech^2 terms
                            val tanhInner = kotlin.math.tanh(inner)
                            val sech2 = 1.0f - tanhInner * tanhInner // sech^2(x) = 1 - tanh^2(x)

                            // Derivative of GELU
                            val derivTerm = sqrt2OverPi * (1.0f + 3.0f * coef * x2)
                            val derivValue = 0.5f * (1.0f + tanhInner + x * sech2 * derivTerm)

                            // Convert back to short
                            derivativeData[i] = (derivValue * 32768.0f).toInt().toShort()
                        }

                        derivative.data = derivativeData
                    }
                    GGMLType.I8, GGMLType.I16, GGMLType.I32, GGMLType.I64 -> {
                        // For integer types, convert to float, compute derivative, then convert back

                        // Create a float array for intermediate calculations
                        val floatData = FloatArray(totalSize)

                        // Convert input data to float
                        when (src0.type) {
                            GGMLType.I8 -> {
                                val srcData = src0.data as ByteArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            GGMLType.I16 -> {
                                val srcData = src0.data as ShortArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            GGMLType.I32 -> {
                                val srcData = src0.data as IntArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            GGMLType.I64 -> {
                                val srcData = src0.data as LongArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            else -> {} // Should not happen due to the when condition
                        }

                        // Compute derivative in float
                        for (i in 0 until totalSize) {
                            val x = floatData[i]
                            val x2 = x * x
                            val x3 = x2 * x

                            // Constants from the GELU approximation
                            val sqrt2OverPi = 0.797885f // sqrt(2/π)
                            val coef = 0.044715f

                            // Inner term: sqrt(2/π) * (x + 0.044715 * x^3)
                            val inner = sqrt2OverPi * (x + coef * x3)

                            // tanh and sech^2 terms
                            val tanhInner = kotlin.math.tanh(inner)
                            val sech2 = 1.0f - tanhInner * tanhInner // sech^2(x) = 1 - tanh^2(x)

                            // Derivative of GELU
                            val derivTerm = sqrt2OverPi * (1.0f + 3.0f * coef * x2)
                            floatData[i] = 0.5f * (1.0f + tanhInner + x * sech2 * derivTerm)
                        }

                        // Convert back to the original type
                        when (src0.type) {
                            GGMLType.I8 -> {
                                val derivativeData = ByteArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toByte()
                                }
                                derivative.data = derivativeData
                            }
                            GGMLType.I16 -> {
                                val derivativeData = ShortArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toShort()
                                }
                                derivative.data = derivativeData
                            }
                            GGMLType.I32 -> {
                                val derivativeData = IntArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt()
                                }
                                derivative.data = derivativeData
                            }
                            GGMLType.I64 -> {
                                val derivativeData = LongArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toLong()
                                }
                                derivative.data = derivativeData
                            }
                            else -> {} // Should not happen due to the when condition
                        }
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("GELU backward pass not implemented for type ${src0.type}")
                    }
                }

                // Multiply gradient by derivative: grad_tensor * derivative
                val gradDerivative = mul(context, tensor.grad!!, derivative)

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradDerivative, zeroTable)
            }
        }
        GGMLOp.MUL_MAT -> {
            // For matrix multiplication C = A @ B:
            // grad_A = grad_C @ B^T
            // grad_B = A^T @ grad_C

            if (src0?.grad != null) {
                // Compute grad_A = grad_C @ B^T

                // First, we need to transpose B (src1)
                val bTransposed = GGMLTensor(type = src1!!.type)

                // Set dimensions for the transposed tensor
                // If B is (n x p), B^T will be (p x n)
                bTransposed.ne[0] = src1.ne[1]
                bTransposed.ne[1] = src1.ne[0]
                for (i in 2 until GGML_MAX_DIMS) {
                    bTransposed.ne[i] = src1.ne[i]
                }

                // Set strides for the transposed tensor
                val typeSize = when (src1.type) {
                    GGMLType.F32 -> 4u
                    GGMLType.F16 -> 2u
                    GGMLType.I8 -> 1u
                    GGMLType.I16 -> 2u
                    GGMLType.I32 -> 4u
                    GGMLType.I64 -> 8u
                    else -> 1u // Default for quantized types
                }

                bTransposed.nb[0] = src1.nb[1]
                bTransposed.nb[1] = src1.nb[0]
                for (i in 2 until GGML_MAX_DIMS) {
                    bTransposed.nb[i] = src1.nb[i]
                }

                // Calculate total size
                val totalSize = (src1.ne[0] * src1.ne[1]).toInt()

                // Create transposed data
                when (src1.type) {
                    GGMLType.F32 -> {
                        val srcData = src1.data as FloatArray
                        val transposedData = FloatArray(totalSize)

                        val rows = src1.ne[0].toInt()
                        val cols = src1.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        bTransposed.data = transposedData
                    }
                    GGMLType.F16 -> {
                        val srcData = src1.data as ShortArray
                        val transposedData = ShortArray(totalSize)

                        val rows = src1.ne[0].toInt()
                        val cols = src1.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        bTransposed.data = transposedData
                    }
                    GGMLType.I8 -> {
                        val srcData = src1.data as ByteArray
                        val transposedData = ByteArray(totalSize)

                        val rows = src1.ne[0].toInt()
                        val cols = src1.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        bTransposed.data = transposedData
                    }
                    GGMLType.I16 -> {
                        val srcData = src1.data as ShortArray
                        val transposedData = ShortArray(totalSize)

                        val rows = src1.ne[0].toInt()
                        val cols = src1.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        bTransposed.data = transposedData
                    }
                    GGMLType.I32 -> {
                        val srcData = src1.data as IntArray
                        val transposedData = IntArray(totalSize)

                        val rows = src1.ne[0].toInt()
                        val cols = src1.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        bTransposed.data = transposedData
                    }
                    GGMLType.I64 -> {
                        val srcData = src1.data as LongArray
                        val transposedData = LongArray(totalSize)

                        val rows = src1.ne[0].toInt()
                        val cols = src1.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        bTransposed.data = transposedData
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("MUL_MAT backward pass not implemented for type ${src1.type}")
                    }
                }

                // Compute grad_A = grad_C @ B^T
                val gradA = matMul(context, tensor.grad!!, bTransposed)

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradA, zeroTable)
            }

            if (src1?.grad != null) {
                // Compute grad_B = A^T @ grad_C

                // First, we need to transpose A (src0)
                val aTransposed = GGMLTensor(type = src0!!.type)

                // Set dimensions for the transposed tensor
                // If A is (m x n), A^T will be (n x m)
                aTransposed.ne[0] = src0.ne[1]
                aTransposed.ne[1] = src0.ne[0]
                for (i in 2 until GGML_MAX_DIMS) {
                    aTransposed.ne[i] = src0.ne[i]
                }

                // Set strides for the transposed tensor
                val typeSize = when (src0.type) {
                    GGMLType.F32 -> 4u
                    GGMLType.F16 -> 2u
                    GGMLType.I8 -> 1u
                    GGMLType.I16 -> 2u
                    GGMLType.I32 -> 4u
                    GGMLType.I64 -> 8u
                    else -> 1u // Default for quantized types
                }

                aTransposed.nb[0] = src0.nb[1]
                aTransposed.nb[1] = src0.nb[0]
                for (i in 2 until GGML_MAX_DIMS) {
                    aTransposed.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = (src0.ne[0] * src0.ne[1]).toInt()

                // Create transposed data
                when (src0.type) {
                    GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        val transposedData = FloatArray(totalSize)

                        val rows = src0.ne[0].toInt()
                        val cols = src0.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        aTransposed.data = transposedData
                    }
                    GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        val transposedData = ShortArray(totalSize)

                        val rows = src0.ne[0].toInt()
                        val cols = src0.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        aTransposed.data = transposedData
                    }
                    GGMLType.I8 -> {
                        val srcData = src0.data as ByteArray
                        val transposedData = ByteArray(totalSize)

                        val rows = src0.ne[0].toInt()
                        val cols = src0.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        aTransposed.data = transposedData
                    }
                    GGMLType.I16 -> {
                        val srcData = src0.data as ShortArray
                        val transposedData = ShortArray(totalSize)

                        val rows = src0.ne[0].toInt()
                        val cols = src0.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        aTransposed.data = transposedData
                    }
                    GGMLType.I32 -> {
                        val srcData = src0.data as IntArray
                        val transposedData = IntArray(totalSize)

                        val rows = src0.ne[0].toInt()
                        val cols = src0.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        aTransposed.data = transposedData
                    }
                    GGMLType.I64 -> {
                        val srcData = src0.data as LongArray
                        val transposedData = LongArray(totalSize)

                        val rows = src0.ne[0].toInt()
                        val cols = src0.ne[1].toInt()

                        for (i in 0 until rows) {
                            for (j in 0 until cols) {
                                transposedData[j * rows + i] = srcData[i * cols + j]
                            }
                        }

                        aTransposed.data = transposedData
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("MUL_MAT backward pass not implemented for type ${src0.type}")
                    }
                }

                // Compute grad_B = A^T @ grad_C
                val gradB = matMul(context, aTransposed, tensor.grad!!)

                // Add to source gradient
                src1.grad = addOrSet(context, src1.grad, gradB, zeroTable)
            }
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
