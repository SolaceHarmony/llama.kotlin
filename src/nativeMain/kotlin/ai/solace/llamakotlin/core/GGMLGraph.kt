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
            // For division C = A / B:
            // grad_A = grad_C / B
            // grad_B = -grad_C * A / (B * B)

            if (src0?.grad != null) {
                // Compute grad_A = grad_C / B
                val gradA = div(context, tensor.grad!!, src1!!)

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradA, zeroTable)
            }

            if (src1?.grad != null) {
                // Compute grad_B = -grad_C * A / (B * B)

                // First compute B * B
                val bSquared = mul(context, src1!!, src1)

                // Then compute A / (B * B)
                val aDivBSquared = div(context, src0!!, bSquared)

                // Then compute grad_C * A / (B * B)
                val gradCTimesADivBSquared = mul(context, tensor.grad!!, aDivBSquared)

                // Finally negate to get -grad_C * A / (B * B)
                val negGradCTimesADivBSquared = neg(context, gradCTimesADivBSquared)

                // Add to source gradient
                src1.grad = addOrSet(context, src1.grad, negGradCTimesADivBSquared, zeroTable)
            }
        }
        GGMLOp.SQR -> {
            // For square operation C = A^2:
            // grad_A = grad_C * 2 * A

            if (src0?.grad != null) {
                // Create a tensor with 2 * A values
                val twoTimesA = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    twoTimesA.ne[i] = src0.ne[i]
                    twoTimesA.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Create data with 2 * A values
                when (src0.type) {
                    GGMLType.F32 -> {
                        val aData = src0.data as FloatArray
                        val twoTimesAData = FloatArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = 2.0f * aData[i]
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    GGMLType.F16 -> {
                        val aData = src0.data as ShortArray
                        val twoTimesAData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            val aFloat = aData[i].toFloat() / 32768.0f
                            val resultFloat = 2.0f * aFloat
                            twoTimesAData[i] = (resultFloat * 32768.0f).toInt().toShort()
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    GGMLType.I8 -> {
                        val aData = src0.data as ByteArray
                        val twoTimesAData = ByteArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = (2 * aData[i]).toByte()
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    GGMLType.I16 -> {
                        val aData = src0.data as ShortArray
                        val twoTimesAData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = (2 * aData[i]).toShort()
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    GGMLType.I32 -> {
                        val aData = src0.data as IntArray
                        val twoTimesAData = IntArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = 2 * aData[i]
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    GGMLType.I64 -> {
                        val aData = src0.data as LongArray
                        val twoTimesAData = LongArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = 2L * aData[i]
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("SQR backward pass not implemented for type ${src0.type}")
                    }
                }

                // Compute grad_A = grad_C * (2 * A)
                val gradA = mul(context, tensor.grad!!, twoTimesA)

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradA, zeroTable)
            }
        }
        GGMLOp.SQRT -> {
            // For square root operation C = sqrt(A):
            // grad_A = grad_C * 0.5 / sqrt(A)

            if (src0?.grad != null) {
                // Create a tensor with 0.5 / sqrt(A) values
                val halfDivSqrtA = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    halfDivSqrtA.ne[i] = src0.ne[i]
                    halfDivSqrtA.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Create data with 0.5 / sqrt(A) values
                when (src0.type) {
                    GGMLType.F32 -> {
                        val aData = src0.data as FloatArray
                        val halfDivSqrtAData = FloatArray(totalSize)
                        for (i in 0 until totalSize) {
                            // Handle division by zero for very small values
                            val sqrtA = kotlin.math.sqrt(kotlin.math.max(aData[i], 1e-10f))
                            halfDivSqrtAData[i] = 0.5f / sqrtA
                        }
                        halfDivSqrtA.data = halfDivSqrtAData
                    }
                    GGMLType.F16 -> {
                        val aData = src0.data as ShortArray
                        val halfDivSqrtAData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            val aFloat = aData[i].toFloat() / 32768.0f
                            // Handle division by zero for very small values
                            val sqrtA = kotlin.math.sqrt(kotlin.math.max(aFloat, 1e-10f))
                            val resultFloat = 0.5f / sqrtA
                            halfDivSqrtAData[i] = (resultFloat * 32768.0f).toInt().toShort()
                        }
                        halfDivSqrtA.data = halfDivSqrtAData
                    }
                    GGMLType.I8, GGMLType.I16, GGMLType.I32, GGMLType.I64 -> {
                        // For integer types, we need to be careful with the square root and division
                        // We'll convert to float, compute the derivative, and convert back

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
                            // Handle division by zero for very small values
                            val sqrtA = kotlin.math.sqrt(kotlin.math.max(floatData[i], 1e-10f))
                            floatData[i] = 0.5f / sqrtA
                        }

                        // Convert back to the original type
                        when (src0.type) {
                            GGMLType.I8 -> {
                                val derivativeData = ByteArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toByte()
                                }
                                halfDivSqrtA.data = derivativeData
                            }
                            GGMLType.I16 -> {
                                val derivativeData = ShortArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toShort()
                                }
                                halfDivSqrtA.data = derivativeData
                            }
                            GGMLType.I32 -> {
                                val derivativeData = IntArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt()
                                }
                                halfDivSqrtA.data = derivativeData
                            }
                            GGMLType.I64 -> {
                                val derivativeData = LongArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toLong()
                                }
                                halfDivSqrtA.data = derivativeData
                            }
                            else -> {} // Should not happen due to the when condition
                        }
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("SQRT backward pass not implemented for type ${src0.type}")
                    }
                }

                // Compute grad_A = grad_C * (0.5 / sqrt(A))
                val gradA = mul(context, tensor.grad!!, halfDivSqrtA)

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradA, zeroTable)
            }
        }
        GGMLOp.SUM -> {
            // For sum operation C = sum(A):
            // grad_A = grad_C for each element of A

            if (src0?.grad != null) {
                // Create a tensor with the same shape as src0 but filled with the gradient value
                val gradTensor = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    gradTensor.ne[i] = src0.ne[i]
                    gradTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Get the gradient value (should be a scalar)
                val gradValue = when (tensor.grad!!.type) {
                    GGMLType.F32 -> (tensor.grad!!.data as FloatArray)[0]
                    GGMLType.F16 -> (tensor.grad!!.data as ShortArray)[0].toFloat() / 32768.0f
                    GGMLType.I8 -> (tensor.grad!!.data as ByteArray)[0].toFloat()
                    GGMLType.I16 -> (tensor.grad!!.data as ShortArray)[0].toFloat()
                    GGMLType.I32 -> (tensor.grad!!.data as IntArray)[0].toFloat()
                    GGMLType.I64 -> (tensor.grad!!.data as LongArray)[0].toFloat()
                    else -> throw NotImplementedError("SUM backward pass not implemented for type ${tensor.grad!!.type}")
                }

                // Fill the gradient tensor with the gradient value
                when (src0.type) {
                    GGMLType.F32 -> {
                        val gradData = FloatArray(totalSize) { gradValue }
                        gradTensor.data = gradData
                    }
                    GGMLType.F16 -> {
                        val gradData = ShortArray(totalSize) { (gradValue * 32768.0f).toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I8 -> {
                        val gradData = ByteArray(totalSize) { gradValue.toInt().toByte() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I16 -> {
                        val gradData = ShortArray(totalSize) { gradValue.toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I32 -> {
                        val gradData = IntArray(totalSize) { gradValue.toInt() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I64 -> {
                        val gradData = LongArray(totalSize) { gradValue.toLong() }
                        gradTensor.data = gradData
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("SUM backward pass not implemented for type ${src0.type}")
                    }
                }

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradTensor, zeroTable)
            }
        }
        GGMLOp.MEAN -> {
            // For mean operation C = mean(A):
            // grad_A = grad_C / n for each element of A, where n is the number of elements in A

            if (src0?.grad != null) {
                // Create a tensor with the same shape as src0 but filled with the gradient value / n
                val gradTensor = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    gradTensor.ne[i] = src0.ne[i]
                    gradTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Get the gradient value (should be a scalar)
                val gradValue = when (tensor.grad!!.type) {
                    GGMLType.F32 -> (tensor.grad!!.data as FloatArray)[0]
                    GGMLType.F16 -> (tensor.grad!!.data as ShortArray)[0].toFloat() / 32768.0f
                    GGMLType.I8 -> (tensor.grad!!.data as ByteArray)[0].toFloat()
                    GGMLType.I16 -> (tensor.grad!!.data as ShortArray)[0].toFloat()
                    GGMLType.I32 -> (tensor.grad!!.data as IntArray)[0].toFloat()
                    GGMLType.I64 -> (tensor.grad!!.data as LongArray)[0].toFloat()
                    else -> throw NotImplementedError("MEAN backward pass not implemented for type ${tensor.grad!!.type}")
                }

                // Calculate gradient value divided by number of elements
                val gradValueDivN = gradValue / totalSize

                // Fill the gradient tensor with the gradient value / n
                when (src0.type) {
                    GGMLType.F32 -> {
                        val gradData = FloatArray(totalSize) { gradValueDivN }
                        gradTensor.data = gradData
                    }
                    GGMLType.F16 -> {
                        val gradData = ShortArray(totalSize) { (gradValueDivN * 32768.0f).toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I8 -> {
                        val gradData = ByteArray(totalSize) { gradValueDivN.toInt().toByte() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I16 -> {
                        val gradData = ShortArray(totalSize) { gradValueDivN.toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I32 -> {
                        val gradData = IntArray(totalSize) { gradValueDivN.toInt() }
                        gradTensor.data = gradData
                    }
                    GGMLType.I64 -> {
                        val gradData = LongArray(totalSize) { gradValueDivN.toLong() }
                        gradTensor.data = gradData
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("MEAN backward pass not implemented for type ${src0.type}")
                    }
                }

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradTensor, zeroTable)
            }
        }
        GGMLOp.REPEAT -> {
            // For repeat operation C = repeat(A, dims):
            // grad_A = sum(grad_C) along the repeated dimensions

            if (src0?.grad != null) {
                // Create a tensor with the same shape as src0
                val gradTensor = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    gradTensor.ne[i] = src0.ne[i]
                    gradTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size of src0
                val totalSizeSrc0 = calculateTotalSize(src0.ne)

                // Calculate total size of tensor (output of repeat)
                val totalSizeTensor = calculateTotalSize(tensor.ne)

                // Calculate repeat factors for each dimension
                val repeatFactors = IntArray(GGML_MAX_DIMS)
                for (i in 0 until GGML_MAX_DIMS) {
                    repeatFactors[i] = if (src0.ne[i] > 0) (tensor.ne[i] / src0.ne[i]).toInt() else 1
                }

                // Initialize gradient tensor data based on type
                when (src0.type) {
                    GGMLType.F32 -> {
                        val gradData = FloatArray(totalSizeSrc0) { 0.0f }
                        gradTensor.data = gradData

                        // Get tensor gradient data
                        val tensorGradData = tensor.grad!!.data as FloatArray

                        // Sum gradients along repeated dimensions
                        // This is a simplified implementation that works for basic cases
                        // A more general implementation would handle arbitrary repeat patterns

                        // For each element in src0
                        for (i in 0 until totalSizeSrc0) {
                            // Calculate multi-dimensional indices for this element
                            var idx = i
                            val indices = IntArray(GGML_MAX_DIMS)
                            for (dim in 0 until GGML_MAX_DIMS) {
                                if (src0.ne[dim] > 0) {
                                    indices[dim] = (idx % src0.ne[dim].toInt())
                                    idx /= src0.ne[dim].toInt()
                                } else {
                                    indices[dim] = 0
                                }
                            }

                            // For each repeated instance of this element
                            var sum = 0.0f
                            for (r0 in 0 until repeatFactors[0]) {
                                for (r1 in 0 until repeatFactors[1]) {
                                    for (r2 in 0 until repeatFactors[2]) {
                                        for (r3 in 0 until repeatFactors[3]) {
                                            // Calculate the index in the repeated tensor
                                            val tensorIdx = (indices[0] + r0 * src0.ne[0].toInt()) +
                                                    (indices[1] + r1 * src0.ne[1].toInt()) * tensor.ne[0].toInt() +
                                                    (indices[2] + r2 * src0.ne[2].toInt()) * tensor.ne[0].toInt() * tensor.ne[1].toInt() +
                                                    (indices[3] + r3 * src0.ne[3].toInt()) * tensor.ne[0].toInt() * tensor.ne[1].toInt() * tensor.ne[2].toInt()

                                            if (tensorIdx < totalSizeTensor) {
                                                sum += tensorGradData[tensorIdx]
                                            }
                                        }
                                    }
                                }
                            }

                            gradData[i] = sum
                        }
                    }
                    GGMLType.F16 -> {
                        val gradData = ShortArray(totalSizeSrc0) { 0 }
                        gradTensor.data = gradData

                        // Get tensor gradient data
                        val tensorGradData = tensor.grad!!.data as ShortArray

                        // For each element in src0
                        for (i in 0 until totalSizeSrc0) {
                            // Calculate multi-dimensional indices for this element
                            var idx = i
                            val indices = IntArray(GGML_MAX_DIMS)
                            for (dim in 0 until GGML_MAX_DIMS) {
                                if (src0.ne[dim] > 0) {
                                    indices[dim] = (idx % src0.ne[dim].toInt())
                                    idx /= src0.ne[dim].toInt()
                                } else {
                                    indices[dim] = 0
                                }
                            }

                            // For each repeated instance of this element
                            var sum = 0.0f
                            for (r0 in 0 until repeatFactors[0]) {
                                for (r1 in 0 until repeatFactors[1]) {
                                    for (r2 in 0 until repeatFactors[2]) {
                                        for (r3 in 0 until repeatFactors[3]) {
                                            // Calculate the index in the repeated tensor
                                            val tensorIdx = (indices[0] + r0 * src0.ne[0].toInt()) +
                                                    (indices[1] + r1 * src0.ne[1].toInt()) * tensor.ne[0].toInt() +
                                                    (indices[2] + r2 * src0.ne[2].toInt()) * tensor.ne[0].toInt() * tensor.ne[1].toInt() +
                                                    (indices[3] + r3 * src0.ne[3].toInt()) * tensor.ne[0].toInt() * tensor.ne[1].toInt() * tensor.ne[2].toInt()

                                            if (tensorIdx < totalSizeTensor) {
                                                sum += tensorGradData[tensorIdx].toFloat() / 32768.0f
                                            }
                                        }
                                    }
                                }
                            }

                            gradData[i] = (sum * 32768.0f).toInt().toShort()
                        }
                    }
                    GGMLType.I8, GGMLType.I16, GGMLType.I32, GGMLType.I64 -> {
                        // For integer types, we'll use a float array for intermediate calculations
                        val floatGradData = FloatArray(totalSizeSrc0) { 0.0f }

                        // Get tensor gradient data and convert to float
                        val tensorGradDataFloat = FloatArray(totalSizeTensor)
                        when (tensor.grad!!.type) {
                            GGMLType.I8 -> {
                                val tensorGradData = tensor.grad!!.data as ByteArray
                                for (i in 0 until totalSizeTensor) {
                                    tensorGradDataFloat[i] = tensorGradData[i].toFloat()
                                }
                            }
                            GGMLType.I16 -> {
                                val tensorGradData = tensor.grad!!.data as ShortArray
                                for (i in 0 until totalSizeTensor) {
                                    tensorGradDataFloat[i] = tensorGradData[i].toFloat()
                                }
                            }
                            GGMLType.I32 -> {
                                val tensorGradData = tensor.grad!!.data as IntArray
                                for (i in 0 until totalSizeTensor) {
                                    tensorGradDataFloat[i] = tensorGradData[i].toFloat()
                                }
                            }
                            GGMLType.I64 -> {
                                val tensorGradData = tensor.grad!!.data as LongArray
                                for (i in 0 until totalSizeTensor) {
                                    tensorGradDataFloat[i] = tensorGradData[i].toFloat()
                                }
                            }
                            else -> {} // Should not happen due to the when condition
                        }

                        // For each element in src0
                        for (i in 0 until totalSizeSrc0) {
                            // Calculate multi-dimensional indices for this element
                            var idx = i
                            val indices = IntArray(GGML_MAX_DIMS)
                            for (dim in 0 until GGML_MAX_DIMS) {
                                if (src0.ne[dim] > 0) {
                                    indices[dim] = (idx % src0.ne[dim].toInt())
                                    idx /= src0.ne[dim].toInt()
                                } else {
                                    indices[dim] = 0
                                }
                            }

                            // For each repeated instance of this element
                            var sum = 0.0f
                            for (r0 in 0 until repeatFactors[0]) {
                                for (r1 in 0 until repeatFactors[1]) {
                                    for (r2 in 0 until repeatFactors[2]) {
                                        for (r3 in 0 until repeatFactors[3]) {
                                            // Calculate the index in the repeated tensor
                                            val tensorIdx = (indices[0] + r0 * src0.ne[0].toInt()) +
                                                    (indices[1] + r1 * src0.ne[1].toInt()) * tensor.ne[0].toInt() +
                                                    (indices[2] + r2 * src0.ne[2].toInt()) * tensor.ne[0].toInt() * tensor.ne[1].toInt() +
                                                    (indices[3] + r3 * src0.ne[3].toInt()) * tensor.ne[0].toInt() * tensor.ne[1].toInt() * tensor.ne[2].toInt()

                                            if (tensorIdx < totalSizeTensor) {
                                                sum += tensorGradDataFloat[tensorIdx]
                                            }
                                        }
                                    }
                                }
                            }

                            floatGradData[i] = sum
                        }

                        // Convert float gradient data back to the original type
                        when (src0.type) {
                            GGMLType.I8 -> {
                                val gradData = ByteArray(totalSizeSrc0)
                                for (i in 0 until totalSizeSrc0) {
                                    gradData[i] = floatGradData[i].toInt().toByte()
                                }
                                gradTensor.data = gradData
                            }
                            GGMLType.I16 -> {
                                val gradData = ShortArray(totalSizeSrc0)
                                for (i in 0 until totalSizeSrc0) {
                                    gradData[i] = floatGradData[i].toInt().toShort()
                                }
                                gradTensor.data = gradData
                            }
                            GGMLType.I32 -> {
                                val gradData = IntArray(totalSizeSrc0)
                                for (i in 0 until totalSizeSrc0) {
                                    gradData[i] = floatGradData[i].toInt()
                                }
                                gradTensor.data = gradData
                            }
                            GGMLType.I64 -> {
                                val gradData = LongArray(totalSizeSrc0)
                                for (i in 0 until totalSizeSrc0) {
                                    gradData[i] = floatGradData[i].toLong()
                                }
                                gradTensor.data = gradData
                            }
                            else -> {} // Should not happen due to the when condition
                        }
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("REPEAT backward pass not implemented for type ${src0.type}")
                    }
                }

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradTensor, zeroTable)
            }
        }
        GGMLOp.ABS -> {
            // For absolute value operation C = abs(A):
            // grad_A = grad_C * sign(A), where sign(A) is 1 if A > 0, -1 if A < 0, 0 if A = 0

            if (src0?.grad != null) {
                // Create a tensor with the sign of src0
                val signTensor = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    signTensor.ne[i] = src0.ne[i]
                    signTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Create data with sign values
                when (src0.type) {
                    GGMLType.F32 -> {
                        val aData = src0.data as FloatArray
                        val signData = FloatArray(totalSize)
                        for (i in 0 until totalSize) {
                            signData[i] = when {
                                aData[i] > 0.0f -> 1.0f
                                aData[i] < 0.0f -> -1.0f
                                else -> 0.0f
                            }
                        }
                        signTensor.data = signData
                    }
                    GGMLType.F16 -> {
                        val aData = src0.data as ShortArray
                        val signData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            val aFloat = aData[i].toFloat() / 32768.0f
                            val signFloat = when {
                                aFloat > 0.0f -> 1.0f
                                aFloat < 0.0f -> -1.0f
                                else -> 0.0f
                            }
                            signData[i] = (signFloat * 32768.0f).toInt().toShort()
                        }
                        signTensor.data = signData
                    }
                    GGMLType.I8 -> {
                        val aData = src0.data as ByteArray
                        val signData = ByteArray(totalSize)
                        for (i in 0 until totalSize) {
                            signData[i] = when {
                                aData[i] > 0 -> 1
                                aData[i] < 0 -> -1
                                else -> 0
                            }.toByte()
                        }
                        signTensor.data = signData
                    }
                    GGMLType.I16 -> {
                        val aData = src0.data as ShortArray
                        val signData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            signData[i] = when {
                                aData[i] > 0 -> 1
                                aData[i] < 0 -> -1
                                else -> 0
                            }.toShort()
                        }
                        signTensor.data = signData
                    }
                    GGMLType.I32 -> {
                        val aData = src0.data as IntArray
                        val signData = IntArray(totalSize)
                        for (i in 0 until totalSize) {
                            signData[i] = when {
                                aData[i] > 0 -> 1
                                aData[i] < 0 -> -1
                                else -> 0
                            }
                        }
                        signTensor.data = signData
                    }
                    GGMLType.I64 -> {
                        val aData = src0.data as LongArray
                        val signData = LongArray(totalSize)
                        for (i in 0 until totalSize) {
                            signData[i] = when {
                                aData[i] > 0 -> 1L
                                aData[i] < 0 -> -1L
                                else -> 0L
                            }
                        }
                        signTensor.data = signData
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("ABS backward pass not implemented for type ${src0.type}")
                    }
                }

                // Compute grad_A = grad_C * sign(A)
                val gradA = mul(context, tensor.grad!!, signTensor)

                // Add to source gradient
                src0.grad = addOrSet(context, src0.grad, gradA, zeroTable)
            }
        }
        GGMLOp.SGN -> {
            // For sign operation C = sgn(A):
            // grad_A = 0 (since the derivative of sgn is 0 everywhere except at 0, where it's undefined)
            // In practice, we treat it as 0 everywhere for backpropagation

            if (src0?.grad != null) {
                // Create a tensor with zeros (same shape as src0)
                val zeroTensor = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    zeroTensor.ne[i] = src0.ne[i]
                    zeroTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Create data with zeros
                when (src0.type) {
                    GGMLType.F32 -> {
                        zeroTensor.data = FloatArray(totalSize) { 0.0f }
                    }
                    GGMLType.F16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    GGMLType.I8 -> {
                        zeroTensor.data = ByteArray(totalSize) { 0 }
                    }
                    GGMLType.I16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    GGMLType.I32 -> {
                        zeroTensor.data = IntArray(totalSize) { 0 }
                    }
                    GGMLType.I64 -> {
                        zeroTensor.data = LongArray(totalSize) { 0L }
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("SGN backward pass not implemented for type ${src0.type}")
                    }
                }

                // Add to source gradient (which is effectively just keeping the existing gradient since we're adding zeros)
                src0.grad = addOrSet(context, src0.grad, zeroTensor, zeroTable)
            }
        }
        GGMLOp.NEG -> {
            if (src0?.grad != null) {
                src0.grad = subOrSet(context, src0.grad, tensor.grad!!, zeroTable)
            }
        }
        GGMLOp.STEP -> {
            // For step operation C = step(A):
            // grad_A = 0 (since the derivative of step is 0 everywhere except at the threshold, where it's undefined)
            // In practice, we treat it as 0 everywhere for backpropagation

            if (src0?.grad != null) {
                // Create a tensor with zeros (same shape as src0)
                val zeroTensor = GGMLTensor(type = src0.type)
                for (i in 0 until GGML_MAX_DIMS) {
                    zeroTensor.ne[i] = src0.ne[i]
                    zeroTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = calculateTotalSize(src0.ne)

                // Create data with zeros
                when (src0.type) {
                    GGMLType.F32 -> {
                        zeroTensor.data = FloatArray(totalSize) { 0.0f }
                    }
                    GGMLType.F16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    GGMLType.I8 -> {
                        zeroTensor.data = ByteArray(totalSize) { 0 }
                    }
                    GGMLType.I16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    GGMLType.I32 -> {
                        zeroTensor.data = IntArray(totalSize) { 0 }
                    }
                    GGMLType.I64 -> {
                        zeroTensor.data = LongArray(totalSize) { 0L }
                    }
                    else -> {
                        // For other types, we'll implement later
                        throw NotImplementedError("STEP backward pass not implemented for type ${src0.type}")
                    }
                }

                // Add to source gradient (which is effectively just keeping the existing gradient since we're adding zeros)
                src0.grad = addOrSet(context, src0.grad, zeroTensor, zeroTable)
            }
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
        GGMLOp.SUB -> {
            val a = node.src[0] ?: throw IllegalArgumentException("SUB operation requires two source tensors")
            val b = node.src[1] ?: throw IllegalArgumentException("SUB operation requires two source tensors")
            node.data = computeSub(context, a, b).data
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
        GGMLOp.NEG -> {
            val a = node.src[0] ?: throw IllegalArgumentException("NEG operation requires one source tensor")
            node.data = computeNeg(context, a).data
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
