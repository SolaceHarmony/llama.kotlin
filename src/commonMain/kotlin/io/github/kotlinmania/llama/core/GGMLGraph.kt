// port-lint: source ggml/src/ggml.c
@file:OptIn(kotlin.concurrent.atomics.ExperimentalAtomicApi::class)

package io.github.kotlinmania.llama.ore

import kotlin.concurrent.atomics.AtomicLong
import kotlin.time.TimeSource

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
private fun addOrSet(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor?, b: io.github.kotlinmania.llama.ore.GGMLTensor, zeroTable: MutableSet<io.github.kotlinmania.llama.ore.GGMLTensor>): io.github.kotlinmania.llama.ore.GGMLTensor {
    return if (a == null || a in zeroTable) {
        // If a is null or known to be zero, return b
        b
    } else {
        // Otherwise, add a and b
        io.github.kotlinmania.llama.ore.ggmlAdd(context, a, b)
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
private fun subOrSet(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor?, b: io.github.kotlinmania.llama.ore.GGMLTensor, zeroTable: MutableSet<io.github.kotlinmania.llama.ore.GGMLTensor>): io.github.kotlinmania.llama.ore.GGMLTensor {
    return if (a == null || a in zeroTable) {
        // If a is null or known to be zero, return -b
        // We need to implement a negate function
        val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = b.type)
        result.op = io.github.kotlinmania.llama.ore.GGMLOp.NEG
        result.src[0] = b

        // If the context requests immediate computation, perform it now
        if (context.computeImmediately) {
            // We need to implement computeNeg
            error("fatal error")
        } else {
            result
        }
    } else {
        // Otherwise, subtract b from a
        // We need to implement a subtract function
        val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
        result.op = io.github.kotlinmania.llama.ore.GGMLOp.SUB
        result.src[0] = a
        result.src[1] = b

        // If the context requests immediate computation, perform it now
        if (context.computeImmediately) {
            // We need to implement computeSub
            error("fatal error")
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
fun setParam(tensor: io.github.kotlinmania.llama.ore.GGMLTensor) {
    tensor.flags = tensor.flags or io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_PARAM
}

/**
 * Computes the gradients for a tensor based on its operation type.
 *
 * @param context The GGML context
 * @param tensor The tensor to compute gradients for
 * @param zeroTable A set of tensors that are known to be zero
 */
private fun computeBackward(context: io.github.kotlinmania.llama.ore.GGMLContext, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, zeroTable: MutableSet<io.github.kotlinmania.llama.ore.GGMLTensor>) {
    val src0 = tensor.src[0]
    val src1 = tensor.src[1]

    when (tensor.op) {
        io.github.kotlinmania.llama.ore.GGMLOp.DUP -> {
            if (src0?.grad != null) {
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    tensor.grad!!,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.CPY -> {
            // C = A (tensor = tensor.src[0])
            // grad_A = grad_C
            // tensor.src[0].grad += tensor.grad
            if (src0?.grad != null) {
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    tensor.grad!!,
                    zeroTable
                )
            }
            // src1 is not used by CPY, so no gradient for it
        }
        io.github.kotlinmania.llama.ore.GGMLOp.ADD -> {
            if (src0?.grad != null) {
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    tensor.grad!!,
                    zeroTable
                )
            }
            if (src1?.grad != null) {
                var needsRepeatBack = false
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    val srcDim = src1.ne[i]
                    val outDim = tensor.ne[i]
                    if (srcDim == outDim) continue
                    if (srcDim == 1L && outDim >= 1L) {
                        needsRepeatBack = true
                    } else if (srcDim == 0L && outDim == 0L) {
                        continue
                    } else {
                        throw IllegalArgumentException(
                            "ADD backward does not support broadcasting mismatch: src dim ${srcDim} vs result ${outDim} on axis $i"
                        )
                    }
                }

                val gradForSrc1 = if (needsRepeatBack) {
                    io.github.kotlinmania.llama.ore.ggmlRepeatBack(context, tensor.grad!!, src1)
                } else {
                    tensor.grad!!
                }
                src1.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src1.grad,
                    gradForSrc1,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SUB -> {
            if (src0?.grad != null) {
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    tensor.grad!!,
                    zeroTable
                )
            }
            if (src1?.grad != null) {
                src1.grad = io.github.kotlinmania.llama.ore.subOrSet(
                    context,
                    src1.grad,
                    tensor.grad!!,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.MUL -> {
            if (src0?.grad != null) {
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    io.github.kotlinmania.llama.ore.ggmlMul(context, src1!!, tensor.grad!!),
                    zeroTable
                )
            }
            if (src1?.grad != null) {
                src1.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src1.grad,
                    io.github.kotlinmania.llama.ore.ggmlMul(context, src0!!, tensor.grad!!),
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.DIV -> {
            // For division C = A / B:
            // grad_A = grad_C / B
            // grad_B = -grad_C * A / (B * B)

            if (src0?.grad != null) {
                // Compute grad_A = grad_C / B
                val gradA = io.github.kotlinmania.llama.ore.ggmlDiv(context, tensor.grad!!, src1!!)

                // Add to source gradient
                src0.grad =
                    io.github.kotlinmania.llama.ore.addOrSet(context, src0.grad, gradA, zeroTable)
            }

            if (src1?.grad != null) {
                // Compute grad_B = -grad_C * A / (B * B)

                // First compute B * B
                val bSquared = io.github.kotlinmania.llama.ore.ggmlMul(context, src1, src1)

                // Then compute A / (B * B)
                val aDivBSquared =
                    io.github.kotlinmania.llama.ore.ggmlDiv(context, src0!!, bSquared)

                // Then compute grad_C * A / (B * B)
                val gradCTimesADivBSquared =
                    io.github.kotlinmania.llama.ore.ggmlMul(context, tensor.grad!!, aDivBSquared)

                // Finally negate to get -grad_C * A / (B * B)
                val negGradCTimesADivBSquared =
                    io.github.kotlinmania.llama.ore.ggmlNeg(context, gradCTimesADivBSquared)

                // Add to source gradient
                src1.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src1.grad,
                    negGradCTimesADivBSquared,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SQR -> {
            // For square operation C = A^2:
            // grad_A = grad_C * 2 * A

            if (src0?.grad != null) {
                // Create a tensor with 2 * A values
                val twoTimesA = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    twoTimesA.ne[i] = src0.ne[i]
                    twoTimesA.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Create data with 2 * A values
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val aData = src0.data as FloatArray
                        val twoTimesAData = FloatArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = 2.0f * aData[i]
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val aData = src0.data as ShortArray
                        val twoTimesAData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            val aFloat = aData[i].toFloat() / 32768.0f
                            val resultFloat = 2.0f * aFloat
                            twoTimesAData[i] = (resultFloat * 32768.0f).toInt().toShort()
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        val aData = src0.data as ByteArray
                        val twoTimesAData = ByteArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = (2 * aData[i]).toByte()
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        val aData = src0.data as ShortArray
                        val twoTimesAData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = (2 * aData[i]).toShort()
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        val aData = src0.data as IntArray
                        val twoTimesAData = IntArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = 2 * aData[i]
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        val aData = src0.data as LongArray
                        val twoTimesAData = LongArray(totalSize)
                        for (i in 0 until totalSize) {
                            twoTimesAData[i] = 2L * aData[i]
                        }
                        twoTimesA.data = twoTimesAData
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Compute grad_A = grad_C * (2 * A)
                val gradA =
                    io.github.kotlinmania.llama.ore.ggmlMul(context, tensor.grad!!, twoTimesA)

                // Add to source gradient
                src0.grad =
                    io.github.kotlinmania.llama.ore.addOrSet(context, src0.grad, gradA, zeroTable)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SQRT -> {
            // For square root operation C = sqrt(A):
            // grad_A = grad_C * 0.5 / sqrt(A)

            if (src0?.grad != null) {
                // Create a tensor with 0.5 / sqrt(A) values
                val halfDivSqrtA = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    halfDivSqrtA.ne[i] = src0.ne[i]
                    halfDivSqrtA.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Create data with 0.5 / sqrt(A) values
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val aData = src0.data as FloatArray
                        val halfDivSqrtAData = FloatArray(totalSize)
                        for (i in 0 until totalSize) {
                            // Handle division by zero for very small values
                            val sqrtA = kotlin.math.sqrt(kotlin.math.max(aData[i], 1e-10f))
                            halfDivSqrtAData[i] = 0.5f / sqrtA
                        }
                        halfDivSqrtA.data = halfDivSqrtAData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I8, io.github.kotlinmania.llama.ore.GGMLType.I16, io.github.kotlinmania.llama.ore.GGMLType.I32, io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        // For integer types, we need to be careful with the square root and division
                        // We'll convert to float, compute the derivative, and convert back

                        // Create a float array for intermediate calculations
                        val floatData = FloatArray(totalSize)

                        // Convert input data to float
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val srcData = src0.data as ByteArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val srcData = src0.data as ShortArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val srcData = src0.data as IntArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val derivativeData = ByteArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toByte()
                                }
                                halfDivSqrtA.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val derivativeData = ShortArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toShort()
                                }
                                halfDivSqrtA.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val derivativeData = IntArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt()
                                }
                                halfDivSqrtA.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                        error("fatal error")
                    }
                }

                // Compute grad_A = grad_C * (0.5 / sqrt(A))
                val gradA =
                    io.github.kotlinmania.llama.ore.ggmlMul(context, tensor.grad!!, halfDivSqrtA)

                // Add to source gradient
                src0.grad =
                    io.github.kotlinmania.llama.ore.addOrSet(context, src0.grad, gradA, zeroTable)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SUM -> {
            // For sum operation C = sum(A):
            // grad_A = grad_C for each element of A

            if (src0?.grad != null) {
                // Create a tensor with the same shape as src0 but filled with the gradient value
                val gradTensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    gradTensor.ne[i] = src0.ne[i]
                    gradTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Get the gradient value (should be a scalar)
                val gradValue = when (tensor.grad!!.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> (tensor.grad!!.data as FloatArray)[0]
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> (tensor.grad!!.data as ShortArray)[0].toFloat() / 32768.0f
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> (tensor.grad!!.data as ByteArray)[0].toFloat()
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> (tensor.grad!!.data as ShortArray)[0].toFloat()
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> (tensor.grad!!.data as IntArray)[0].toFloat()
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> (tensor.grad!!.data as LongArray)[0].toFloat()
                    else -> error("fatal error")
                }

                // Fill the gradient tensor with the gradient value
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val gradData = FloatArray(totalSize) { gradValue }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val gradData = ShortArray(totalSize) { (gradValue * 32768.0f).toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        val gradData = ByteArray(totalSize) { gradValue.toInt().toByte() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        val gradData = ShortArray(totalSize) { gradValue.toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        val gradData = IntArray(totalSize) { gradValue.toInt() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        val gradData = LongArray(totalSize) { gradValue.toLong() }
                        gradTensor.data = gradData
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradTensor,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.MEAN -> {
            // For mean operation C = mean(A):
            // grad_A = grad_C / n for each element of A, where n is the number of elements in A

            if (src0?.grad != null) {
                // Create a tensor with the same shape as src0 but filled with the gradient value / n
                val gradTensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    gradTensor.ne[i] = src0.ne[i]
                    gradTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Get the gradient value (should be a scalar)
                val gradValue = when (tensor.grad!!.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> (tensor.grad!!.data as FloatArray)[0]
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> (tensor.grad!!.data as ShortArray)[0].toFloat() / 32768.0f
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> (tensor.grad!!.data as ByteArray)[0].toFloat()
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> (tensor.grad!!.data as ShortArray)[0].toFloat()
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> (tensor.grad!!.data as IntArray)[0].toFloat()
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> (tensor.grad!!.data as LongArray)[0].toFloat()
                    else -> error("fatal error")
                }

                // Calculate gradient value divided by number of elements
                val gradValueDivN = gradValue / totalSize

                // Fill the gradient tensor with the gradient value / n
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val gradData = FloatArray(totalSize) { gradValueDivN }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val gradData = ShortArray(totalSize) { (gradValueDivN * 32768.0f).toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        val gradData = ByteArray(totalSize) { gradValueDivN.toInt().toByte() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        val gradData = ShortArray(totalSize) { gradValueDivN.toInt().toShort() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        val gradData = IntArray(totalSize) { gradValueDivN.toInt() }
                        gradTensor.data = gradData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        val gradData = LongArray(totalSize) { gradValueDivN.toLong() }
                        gradTensor.data = gradData
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradTensor,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.REPEAT -> {
            // For repeat operation C = repeat(A, dims):
            // grad_A = sum(grad_C) along the repeated dimensions

            if (src0?.grad != null) {
                // Create a tensor with the same shape as src0
                val gradTensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    gradTensor.ne[i] = src0.ne[i]
                    gradTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size of src0
                val totalSizeSrc0 = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Calculate total size of tensor (output of repeat)
                val totalSizeTensor = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(tensor.ne).toInt()

                // Calculate repeat factors for each dimension
                val repeatFactors = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    repeatFactors[i] = if (src0.ne[i] > 0) (tensor.ne[i] / src0.ne[i]).toInt() else 1
                }

                // Initialize gradient tensor data based on type
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
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
                            val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                            for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
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
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val gradData = ShortArray(totalSizeSrc0) { 0 }
                        gradTensor.data = gradData

                        // Get tensor gradient data
                        val tensorGradData = tensor.grad!!.data as ShortArray

                        // For each element in src0
                        for (i in 0 until totalSizeSrc0) {
                            // Calculate multi-dimensional indices for this element
                            var idx = i
                            val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                            for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I8, io.github.kotlinmania.llama.ore.GGMLType.I16, io.github.kotlinmania.llama.ore.GGMLType.I32, io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        // For integer types, we'll use a float array for intermediate calculations
                        val floatGradData = FloatArray(totalSizeSrc0) { 0.0f }

                        // Get tensor gradient data and convert to float
                        val tensorGradDataFloat = FloatArray(totalSizeTensor)
                        when (tensor.grad!!.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val tensorGradData = tensor.grad!!.data as ByteArray
                                for (i in 0 until totalSizeTensor) {
                                    tensorGradDataFloat[i] = tensorGradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val tensorGradData = tensor.grad!!.data as ShortArray
                                for (i in 0 until totalSizeTensor) {
                                    tensorGradDataFloat[i] = tensorGradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val tensorGradData = tensor.grad!!.data as IntArray
                                for (i in 0 until totalSizeTensor) {
                                    tensorGradDataFloat[i] = tensorGradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                            val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                            for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
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
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val gradData = ByteArray(totalSizeSrc0)
                                for (i in 0 until totalSizeSrc0) {
                                    gradData[i] = floatGradData[i].toInt().toByte()
                                }
                                gradTensor.data = gradData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val gradData = ShortArray(totalSizeSrc0)
                                for (i in 0 until totalSizeSrc0) {
                                    gradData[i] = floatGradData[i].toInt().toShort()
                                }
                                gradTensor.data = gradData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val gradData = IntArray(totalSizeSrc0)
                                for (i in 0 until totalSizeSrc0) {
                                    gradData[i] = floatGradData[i].toInt()
                                }
                                gradTensor.data = gradData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                        error("fatal error")
                    }
                }

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradTensor,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.ABS -> {
            // For absolute value operation C = abs(A):
            // grad_A = grad_C * sign(A), where sign(A) is 1 if A > 0, -1 if A < 0, 0 if A = 0

            if (src0?.grad != null) {
                // Create a tensor with the sign of src0
                val signTensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    signTensor.ne[i] = src0.ne[i]
                    signTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Create data with sign values
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                        error("fatal error")
                    }
                }

                // Compute grad_A = grad_C * sign(A)
                val gradA =
                    io.github.kotlinmania.llama.ore.ggmlMul(context, tensor.grad!!, signTensor)

                // Add to source gradient
                src0.grad =
                    io.github.kotlinmania.llama.ore.addOrSet(context, src0.grad, gradA, zeroTable)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SGN -> {
            // For sign operation C = sgn(A):
            // grad_A = 0 (since the derivative of sgn is 0 everywhere except at 0, where it's undefined)
            // In practice, we treat it as 0 everywhere for backpropagation

            if (src0?.grad != null) {
                // Create a tensor with zeros (same shape as src0)
                val zeroTensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    zeroTensor.ne[i] = src0.ne[i]
                    zeroTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Create data with zeros
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        zeroTensor.data = FloatArray(totalSize) { 0.0f }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        zeroTensor.data = ByteArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        zeroTensor.data = IntArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        zeroTensor.data = LongArray(totalSize) { 0L }
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Add to source gradient (which is effectively just keeping the existing gradient since we're adding zeros)
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    zeroTensor,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.NEG -> {
            if (src0?.grad != null) {
                src0.grad = io.github.kotlinmania.llama.ore.subOrSet(
                    context,
                    src0.grad,
                    tensor.grad!!,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.STEP -> {
            // For step operation C = step(A):
            // grad_A = 0 (since the derivative of step is 0 everywhere except at the threshold, where it's undefined)
            // In practice, we treat it as 0 everywhere for backpropagation

            if (src0?.grad != null) {
                // Create a tensor with zeros (same shape as src0)
                val zeroTensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    zeroTensor.ne[i] = src0.ne[i]
                    zeroTensor.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Create data with zeros
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        zeroTensor.data = FloatArray(totalSize) { 0.0f }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        zeroTensor.data = ByteArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        zeroTensor.data = ShortArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        zeroTensor.data = IntArray(totalSize) { 0 }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        zeroTensor.data = LongArray(totalSize) { 0L }
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Add to source gradient (which is effectively just keeping the existing gradient since we're adding zeros)
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    zeroTensor,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.RELU -> {
            if (src0?.grad != null) {
                // grad_src0 += grad_tensor * (src0 > 0)
                // Create a mask where src0 > 0
                val mask = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    mask.ne[i] = src0.ne[i]
                    mask.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Create mask data based on the tensor type
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        val maskData = FloatArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0.0f) 1.0f else 0.0f
                        }
                        mask.data = maskData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        val maskData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if ((srcData[i].toFloat() / 32768.0f) > 0.0f) 32767.toShort() else 0
                        }
                        mask.data = maskData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        val srcData = src0.data as ByteArray
                        val maskData = ByteArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        val srcData = src0.data as ShortArray
                        val maskData = ShortArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        val srcData = src0.data as IntArray
                        val maskData = IntArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        val srcData = src0.data as LongArray
                        val maskData = LongArray(totalSize)
                        for (i in 0 until totalSize) {
                            maskData[i] = if (srcData[i] > 0) 1 else 0
                        }
                        mask.data = maskData
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Multiply gradient by mask: grad_tensor * (src0 > 0)
                val gradMasked =
                    io.github.kotlinmania.llama.ore.ggmlMul(context, tensor.grad!!, mask)

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradMasked,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.GELU -> {
            if (src0?.grad != null) {
                // The derivative of GELU approximation
                // GELU(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                // We need to compute the derivative of this with respect to x

                // Create a tensor for the derivative
                val derivative = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    derivative.ne[i] = src0.ne[i]
                    derivative.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Compute the derivative based on the tensor type
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I8, io.github.kotlinmania.llama.ore.GGMLType.I16, io.github.kotlinmania.llama.ore.GGMLType.I32, io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        // For integer types, convert to float, compute derivative, then convert back

                        // Create a float array for intermediate calculations
                        val floatData = FloatArray(totalSize)

                        // Convert input data to float
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val srcData = src0.data as ByteArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val srcData = src0.data as ShortArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val srcData = src0.data as IntArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val derivativeData = ByteArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toByte()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val derivativeData = ShortArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toShort()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val derivativeData = IntArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                        error("fatal error")
                    }
                }

                // Multiply gradient by derivative: grad_tensor * derivative
                val gradDerivative =
                    io.github.kotlinmania.llama.ore.ggmlMul(context, tensor.grad!!, derivative)

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradDerivative,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SCALE -> {
            // C = A * scale_factor
            // grad_A = grad_C * scale_factor
            // src0 is A
            // src1 is scale_factor (scalar)
            // tensor.grad is grad_C
            if (src0?.grad != null) {
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    io.github.kotlinmania.llama.ore.ggmlMul(
                        context,
                        tensor.grad!!,
                        src1!!
                    ), // grad_C * scale_factor
                    zeroTable
                )
            }
            // No gradient for src1 (scale_factor) as it's a constant
        }
        io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE -> {
            // C = reshape(A, new_shape)
            // grad_A = reshape(grad_C, shape_of_A)
            // tensor.src[0] is A
            // tensor.grad is grad_C
            val src0 = tensor.src[0]
            if (src0?.grad != null && tensor.grad != null) {
                val gradC = tensor.grad!!
                val shapeOfA = src0.ne // LongArray

                // grad_A = reshape(grad_C, shape_of_A)
                val gradCReshapedToAShape = io.github.kotlinmania.llama.ore.reshape(
                    context,
                    gradC,
                    *shapeOfA // Spread operator for LongArray into vararg Long
                )
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradCReshapedToAShape,
                    zeroTable
                )
            }
            // src1 is not used by RESHAPE (new shape is usually immediate, not a tensor src)
        }
        io.github.kotlinmania.llama.ore.GGMLOp.VIEW -> {
            // C = view(A) (tensor is C, tensor.src[0] is A)
            // As per instructions, grad_A += reshape(grad_C, shape_of_A).
            // This implies grad_C corresponds to the entirety of A, just potentially in a different layout.
            // Note: This is a simplification. A true view backward pass might need to write grad_C
            // into a *view* of grad_A, especially if the view C is only a part of A.
            // However, following the provided instructions for this subtask.
            val src0 = tensor.src[0]
            if (src0?.grad != null && tensor.grad != null) {
                val gradC = tensor.grad!!
                val shapeOfA = src0.ne // LongArray

                // grad_A_contribution = reshape(grad_C, shape_of_A)
                val gradCReshapedToAShape = io.github.kotlinmania.llama.ore.reshape(
                    context,
                    gradC,
                    *shapeOfA // Spread operator for LongArray into vararg Long
                )
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradCReshapedToAShape,
                    zeroTable
                )
            }
            // View parameters (like offset or new shape for view) are typically not differentiable.
        }
        io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE -> {
            // C = permute(A, ax0, ax1, ax2, ax3)
            // grad_A = permute(grad_C, inv_ax0, inv_ax1, inv_ax2, inv_ax3)
            // tensor.src[0] is A
            // tensor.grad is grad_C
            // tensor.opParams is assumed to be IntArray([ax0, ax1, ax2, ax3])

            val src0 = tensor.src[0]
            // Ensure opParams is an IntArray of expected size (4 for axes)
            val opParams = tensor.opParams
            if (src0?.grad != null && tensor.grad != null && opParams.size == 4) {

                val gradC = tensor.grad!!
                val originalAxes = opParams

                // Calculate inverse permutation axes
                // If P = [ax0, ax1, ax2, ax3], then P_inv is such that P_inv[ax_i] = i.
                val inverseAxes = IntArray(4)
                for (i in 0..3) {
                    // originalAxes[i] gives the new position of original axis i.
                    // So, inverseAxes[originalAxes[i]] should be i.
                    inverseAxes[originalAxes[i]] = i
                }

                val gradCPermutedBack = io.github.kotlinmania.llama.ore.permute(
                    context,
                    gradC,
                    inverseAxes[0],
                    inverseAxes[1],
                    inverseAxes[2],
                    inverseAxes[3]
                )
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradCPermutedBack,
                    zeroTable
                )
            }
            // Parameters for permute (axes) are not differentiable.
        }
        io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE -> {
            // C = transpose(A, ax0, ax1)
            // grad_A = transpose(grad_C, ax0, ax1) (because transpose(transpose(X)) = X with same axes)
            // tensor.src[0] is A
            // tensor.grad is grad_C
            // tensor.opParams is assumed to be IntArray([ax0, ax1])

            val src0 = tensor.src[0]
            // Ensure opParams is an IntArray of expected size (2 for axes)
            val opParams = tensor.opParams
            if (src0?.grad != null && tensor.grad != null && opParams.size == 2) {

                val gradC = tensor.grad!!
                val axes = opParams

                // Transpose grad_C back using the same axes
                val gradCTransposedBack = io.github.kotlinmania.llama.ore.transpose(
                    context,
                    gradC,
                    axes[0], // ax0
                    axes[1]  // ax1
                )
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradCTransposedBack,
                    zeroTable
                )
            }
            // Parameters for transpose (axes) are not differentiable.
        }
        io.github.kotlinmania.llama.ore.GGMLOp.GET_ROWS -> {
            // C = get_rows(A, B) -> tensor = get_rows(src0, src1)
            // grad_A[B[i]] += grad_C[i] (row-wise)
            val src0 = tensor.src[0] // A: source tensor
            val src1 = tensor.src[1] // B: indices tensor

            // Check if src0 needs gradient and other inputs are valid
            if (src0 != null && src0.grad != null && // src0.grad being non-null implies it's a parameter and grad accumulation is expected
                tensor.grad != null && src1 != null && src1.type == io.github.kotlinmania.llama.ore.GGMLType.I32) {

                // Create a temporary tensor for the gradient contribution to src0, initialized to zeros.
                // This tensor will have the same shape and type as src0.
                val gradAContribution = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                gradAContribution.ne = src0.ne.copyOf()
                gradAContribution.nb = src0.nb.copyOf()
                val totalElementsSrc0 = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(gradAContribution.ne).toInt()
                gradAContribution.data = when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> FloatArray(totalElementsSrc0) { 0.0f }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> ShortArray(totalElementsSrc0) { 0 } // Represents 0.0f for F16
                    io.github.kotlinmania.llama.ore.GGMLType.I8  -> ByteArray(totalElementsSrc0) { 0 }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> ShortArray(totalElementsSrc0) { 0 }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> IntArray(totalElementsSrc0) { 0 }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> LongArray(totalElementsSrc0) { 0L }
                    else -> error("fatal error")
                }

                val gradC = tensor.grad!!             // Gradient of the output tensor C
                val indicesB = src1.data as IntArray  // Indices from tensor B

                // Calculate number of elements per row/slice in src0 (A)
                // This assumes ne[0] is the dimension being indexed.
                val elementsPerRowInt = if (src0.ne[0] != 0L) (totalElementsSrc0 / src0.ne[0]).toInt() else 0

                // If elementsPerRowInt is 0 but totalElementsSrc0 is not, it implies an issue or 1D src0.
                // For a 1D src0 (e.g. shape [N]), ne[0]=N, elementsPerRowInt should be 1.
                // (totalElementsSrc0 / src0.ne[0]) = (N / N) = 1. This holds.

                val numRowsC = tensor.ne[0].toInt() // Number of rows in C, also number of indices in B

                // Accumulate gradients from gradC into gradAContribution based on indicesB
                when (gradAContribution.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val gradAData = gradAContribution.data as FloatArray
                        val gradCData = gradC.data as FloatArray
                        for (i in 0 until numRowsC) { // For each row/index
                            val rowIndexInA = indicesB[i]
                            if (rowIndexInA < 0 || rowIndexInA >= src0.ne[0]) continue // Bounds check for safety
                            for (j in 0 until elementsPerRowInt) { // For each element in the row
                                gradAData[rowIndexInA * elementsPerRowInt + j] += gradCData[i * elementsPerRowInt + j]
                            }
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val gradAData = gradAContribution.data as ShortArray
                        val gradCData = gradC.data as ShortArray
                        for (i in 0 until numRowsC) {
                            val rowIndexInA = indicesB[i]
                            if (rowIndexInA < 0 || rowIndexInA >= src0.ne[0]) continue
                            for (j in 0 until elementsPerRowInt) {
                                val currentAValFloat = (gradAData[rowIndexInA * elementsPerRowInt + j].toFloat() / 32768.0f)
                                val valCFloat = (gradCData[i * elementsPerRowInt + j].toFloat() / 32768.0f)
                                val sumFloat = currentAValFloat + valCFloat
                                gradAData[rowIndexInA * elementsPerRowInt + j] = (sumFloat * 32768.0f).toInt().toShort()
                            }
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        val gradAData = gradAContribution.data as ByteArray
                        val gradCData = gradC.data as ByteArray
                        for (i in 0 until numRowsC) {
                            val rowIndexInA = indicesB[i]
                            if (rowIndexInA < 0 || rowIndexInA >= src0.ne[0]) continue
                            for (j in 0 until elementsPerRowInt) {
                                gradAData[rowIndexInA * elementsPerRowInt + j] =
                                    (gradAData[rowIndexInA * elementsPerRowInt + j] + gradCData[i * elementsPerRowInt + j]).toByte()
                            }
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        val gradAData = gradAContribution.data as ShortArray
                        val gradCData = gradC.data as ShortArray
                        for (i in 0 until numRowsC) {
                            val rowIndexInA = indicesB[i]
                            if (rowIndexInA < 0 || rowIndexInA >= src0.ne[0]) continue
                            for (j in 0 until elementsPerRowInt) {
                                gradAData[rowIndexInA * elementsPerRowInt + j] =
                                    (gradAData[rowIndexInA * elementsPerRowInt + j] + gradCData[i * elementsPerRowInt + j]).toShort()
                            }
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        val gradAData = gradAContribution.data as IntArray
                        val gradCData = gradC.data as IntArray
                        for (i in 0 until numRowsC) {
                            val rowIndexInA = indicesB[i]
                            if (rowIndexInA < 0 || rowIndexInA >= src0.ne[0]) continue
                            for (j in 0 until elementsPerRowInt) {
                                gradAData[rowIndexInA * elementsPerRowInt + j] += gradCData[i * elementsPerRowInt + j]
                            }
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        val gradAData = gradAContribution.data as LongArray
                        val gradCData = gradC.data as LongArray
                        for (i in 0 until numRowsC) {
                            val rowIndexInA = indicesB[i]
                            if (rowIndexInA < 0 || rowIndexInA >= src0.ne[0]) continue
                            for (j in 0 until elementsPerRowInt) {
                                gradAData[rowIndexInA * elementsPerRowInt + j] += gradCData[i * elementsPerRowInt + j]
                            }
                        }
                    }
                    else -> error("fatal error")
                }

                // Add the accumulated contributions from gradAContribution to src0.grad
                // src0.grad is assumed to be non-null here due to the initial check.
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad!!,
                    gradAContribution,
                    zeroTable
                )
            }
        }
    io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_INF -> {
        // C = diag_mask_inf(A, n_past)
        // grad_A = grad_C * unmasked_mask, where unmasked_mask is 1 if A[i]==C[i], 0 otherwise.
        // tensor is C, src0 is A.
        // tensor.grad is grad_C.
        // It's assumed that src0.type == tensor.type == tensor.grad.type.
        val src0 = tensor.src[0]

        if (src0 != null && src0.grad != null && tensor.grad != null) {
            val type = src0.type // Assuming this type is consistent for A, C, and grad_C
            val gradC = tensor.grad!!

            // Create unmasked_mask tensor of the same type and shape as src0
            val unmaskedMask = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)
            unmaskedMask.ne = src0.ne.copyOf()
            unmaskedMask.nb = src0.nb.copyOf()
            val totalElements = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(unmaskedMask.ne).toInt()

            val s0Data = src0.data
            val cData = tensor.data // Data of the output tensor C

            // Allocate and populate unmaskedMask.data
            // Mask value is 1 if src0.data[i] == tensor.data[i], else 0
            when (type) {
                io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                    val maskData = FloatArray(totalElements)
                    val s0Array = s0Data as FloatArray
                    val cArray = cData as FloatArray
                    for (i in 0 until totalElements) {
                        maskData[i] = if (s0Array[i] == cArray[i]) 1.0f else 0.0f
                    }
                    unmaskedMask.data = maskData
                }
                io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                    val maskData = ShortArray(totalElements)
                    val s0Array = s0Data as ShortArray
                    val cArray = cData as ShortArray
                    val oneF16 = (1.0f * 32768.0f).toInt().toShort()
                    val zeroF16 = 0.toShort()
                    for (i in 0 until totalElements) {
                        maskData[i] = if (s0Array[i] == cArray[i]) oneF16 else zeroF16
                    }
                    unmaskedMask.data = maskData
                }
                io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                    val maskData = ByteArray(totalElements)
                    val s0Array = s0Data as ByteArray
                    val cArray = cData as ByteArray
                    for (i in 0 until totalElements) {
                        maskData[i] = if (s0Array[i] == cArray[i]) 1.toByte() else 0.toByte()
                    }
                    unmaskedMask.data = maskData
                }
                io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                    val maskData = ShortArray(totalElements)
                    val s0Array = s0Data as ShortArray
                    val cArray = cData as ShortArray
                    for (i in 0 until totalElements) {
                        maskData[i] = if (s0Array[i] == cArray[i]) 1.toShort() else 0.toShort()
                    }
                    unmaskedMask.data = maskData
                }
                io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                    val maskData = IntArray(totalElements)
                    val s0Array = s0Data as IntArray
                    val cArray = cData as IntArray
                    for (i in 0 until totalElements) {
                        maskData[i] = if (s0Array[i] == cArray[i]) 1 else 0
                    }
                    unmaskedMask.data = maskData
                }
                io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                    val maskData = LongArray(totalElements)
                    val s0Array = s0Data as LongArray
                    val cArray = cData as LongArray
                    for (i in 0 until totalElements) {
                        maskData[i] = if (s0Array[i] == cArray[i]) 1L else 0L
                    }
                    unmaskedMask.data = maskData
                }
                else -> error("fatal error")
            }

            // Calculate contribution: grad_C * unmasked_mask
            val contribution = io.github.kotlinmania.llama.ore.ggmlMul(context, gradC, unmaskedMask)

            // Add contribution to src0's gradient
            src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                context,
                src0.grad!!,
                contribution,
                zeroTable
            )
        }
    }
        io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT -> {
            // For matrix multiplication C = A @ B:
            // grad_A = grad_C @ B^T
            // grad_B = A^T @ grad_C

            if (src0?.grad != null) {
                // Compute grad_A = grad_C @ B^T

                // First, we need to transpose B (src1)
                val bTransposed = io.github.kotlinmania.llama.ore.GGMLTensor(type = src1!!.type)

                // Set dimensions for the transposed tensor
                // If B is (n x p), B^T will be (p x n)
                bTransposed.ne[0] = src1.ne[1]
                bTransposed.ne[1] = src1.ne[0]
                for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    bTransposed.ne[i] = src1.ne[i]
                }

                // Set strides for the transposed tensor
                val typeSize = when (src1.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> 4u
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> 2u
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> 1u
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> 2u
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> 4u
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> 8u
                    else -> 1u // Default for quantized types
                }

                bTransposed.nb[0] = src1.nb[1]
                bTransposed.nb[1] = src1.nb[0]
                for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    bTransposed.nb[i] = src1.nb[i]
                }

                // Calculate total size
                val totalSize = (src1.ne[0] * src1.ne[1]).toInt()

                // Create transposed data
                when (src1.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                        error("fatal error")
                    }
                }

                // Compute grad_A = grad_C @ B^T
                val gradA =
                    io.github.kotlinmania.llama.ore.ggmlMulMat(context, tensor.grad!!, bTransposed)

                // Add to source gradient
                src0.grad =
                    io.github.kotlinmania.llama.ore.addOrSet(context, src0.grad, gradA, zeroTable)
            }

            if (src1?.grad != null) {
                // Compute grad_B = A^T @ grad_C

                // First, we need to transpose A (src0)
                val aTransposed = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0!!.type)

                // Set dimensions for the transposed tensor
                // If A is (m x n), A^T will be (n x m)
                aTransposed.ne[0] = src0.ne[1]
                aTransposed.ne[1] = src0.ne[0]
                for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    aTransposed.ne[i] = src0.ne[i]
                }

                // Set strides for the transposed tensor
                val typeSize = when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> 4u
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> 2u
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> 1u
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> 2u
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> 4u
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> 8u
                    else -> 1u // Default for quantized types
                }

                aTransposed.nb[0] = src0.nb[1]
                aTransposed.nb[1] = src0.nb[0]
                for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    aTransposed.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = (src0.ne[0] * src0.ne[1]).toInt()

                // Create transposed data
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
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
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                        error("fatal error")
                    }
                }

                // Compute grad_B = A^T @ grad_C
                val gradB =
                    io.github.kotlinmania.llama.ore.ggmlMulMat(context, aTransposed, tensor.grad!!)

                // Add to source gradient
                src1.grad =
                    io.github.kotlinmania.llama.ore.addOrSet(context, src1.grad, gradB, zeroTable)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SILU -> {
            // For SILU operation C = x * sigmoid(x):
            // grad_x = grad_C * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))

            if (src0?.grad != null) {
                // Create a tensor for the derivative
                val derivative = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    derivative.ne[i] = src0.ne[i]
                    derivative.nb[i] = src0.nb[i]
                }

                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // Compute the derivative based on the tensor type
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        val derivativeData = FloatArray(totalSize)

                        for (i in 0 until totalSize) {
                            val x = srcData[i]
                            val sigmoid = 1.0f / (1.0f + kotlin.math.exp(-x))
                            // Derivative of SILU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                            derivativeData[i] = sigmoid + x * sigmoid * (1.0f - sigmoid)
                        }

                        derivative.data = derivativeData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        val derivativeData = ShortArray(totalSize)

                        for (i in 0 until totalSize) {
                            // Convert short to float
                            val x = srcData[i].toFloat() / 32768.0f
                            val sigmoid = 1.0f / (1.0f + kotlin.math.exp(-x))
                            // Derivative of SILU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                            val derivValue = sigmoid + x * sigmoid * (1.0f - sigmoid)

                            // Convert back to short
                            derivativeData[i] = (derivValue * 32768.0f).toInt().toShort()
                        }

                        derivative.data = derivativeData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8, io.github.kotlinmania.llama.ore.GGMLType.I16, io.github.kotlinmania.llama.ore.GGMLType.I32, io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        // For integer types, convert to float, compute derivative, then convert back
                        val floatData = FloatArray(totalSize)

                        // Convert input data to float
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val srcData = src0.data as ByteArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val srcData = src0.data as ShortArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val srcData = src0.data as IntArray
                                for (i in 0 until totalSize) {
                                    floatData[i] = srcData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                            val sigmoid = 1.0f / (1.0f + kotlin.math.exp(-x))
                            // Derivative of SILU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                            floatData[i] = sigmoid + x * sigmoid * (1.0f - sigmoid)
                        }

                        // Convert back to the original type
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val derivativeData = ByteArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toByte()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val derivativeData = ShortArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt().toShort()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val derivativeData = IntArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = floatData[i].toInt()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
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
                        error("fatal error")
                    }
                }

                // Multiply gradient by derivative: grad_tensor * derivative
                val gradDerivative =
                    io.github.kotlinmania.llama.ore.ggmlMul(context, tensor.grad!!, derivative)

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    gradDerivative,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.NORM -> {
            // where ⊗ is the outer product and I is the identity matrix

            if (src0?.grad != null) {
                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()

                // First, we need to compute the L2 norm of src0
                var norm = 0.0f
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        for (i in 0 until totalSize) {
                            norm += srcData[i] * srcData[i]
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat() / 32768.0f
                            norm += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        val srcData = src0.data as ByteArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            norm += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        val srcData = src0.data as ShortArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            norm += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        val srcData = src0.data as IntArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            norm += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        val srcData = src0.data as LongArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            norm += value * value
                        }
                    }
                    else -> {
                        error("fatal error")
                    }
                }
                norm = kotlin.math.sqrt(norm)

                // Create a tensor for the derivative
                val derivative = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    derivative.ne[i] = src0.ne[i]
                    derivative.nb[i] = src0.nb[i]
                }

                // Compute the derivative based on the tensor type
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        val derivativeData = FloatArray(totalSize)
                        val normalizedData = FloatArray(totalSize)

                        for (i in 0 until totalSize) {
                            normalizedData[i] = srcData[i] / norm
                        }

                        // Now compute the derivative for each element
                        for (i in 0 until totalSize) {
                            var outerProductSum = 0.0f
                            for (j in 0 until totalSize) {
                                // Compute the outer product term: C_i * C_j * grad_C_j
                                outerProductSum += normalizedData[i] * normalizedData[j] * (tensor.grad!!.data as FloatArray)[j]
                            }
                            derivativeData[i] = ((tensor.grad!!.data as FloatArray)[i] - outerProductSum) / norm
                        }

                        derivative.data = derivativeData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        val derivativeData = ShortArray(totalSize)
                        val normalizedData = FloatArray(totalSize)
                        val gradData = FloatArray(totalSize)

                        // Convert gradient data to float
                        for (i in 0 until totalSize) {
                            gradData[i] = (tensor.grad!!.data as ShortArray)[i].toFloat() / 32768.0f
                        }

                        for (i in 0 until totalSize) {
                            normalizedData[i] = (srcData[i].toFloat() / 32768.0f) / norm
                        }

                        // Now compute the derivative for each element
                        for (i in 0 until totalSize) {
                            var outerProductSum = 0.0f
                            for (j in 0 until totalSize) {
                                // Compute the outer product term: C_i * C_j * grad_C_j
                                outerProductSum += normalizedData[i] * normalizedData[j] * gradData[j]
                            }
                            val derivValue = (gradData[i] - outerProductSum) / norm
                            derivativeData[i] = (derivValue * 32768.0f).toInt().toShort()
                        }

                        derivative.data = derivativeData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8, io.github.kotlinmania.llama.ore.GGMLType.I16, io.github.kotlinmania.llama.ore.GGMLType.I32, io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        // For integer types, convert to float, compute derivative, then convert back
                        val srcFloatData = FloatArray(totalSize)
                        val gradFloatData = FloatArray(totalSize)
                        val normalizedData = FloatArray(totalSize)
                        val derivativeFloatData = FloatArray(totalSize)

                        // Convert source data to float
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val srcData = src0.data as ByteArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as ByteArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val srcData = src0.data as ShortArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as ShortArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val srcData = src0.data as IntArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as IntArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                                val srcData = src0.data as LongArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as LongArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            else -> {} // Should not happen due to the when condition
                        }

                        for (i in 0 until totalSize) {
                            normalizedData[i] = srcFloatData[i] / norm
                        }

                        // Now compute the derivative for each element
                        for (i in 0 until totalSize) {
                            var outerProductSum = 0.0f
                            for (j in 0 until totalSize) {
                                // Compute the outer product term: C_i * C_j * grad_C_j
                                outerProductSum += normalizedData[i] * normalizedData[j] * gradFloatData[j]
                            }
                            derivativeFloatData[i] = (gradFloatData[i] - outerProductSum) / norm
                        }

                        // Convert back to the original type
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val derivativeData = ByteArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toInt().toByte()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val derivativeData = ShortArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toInt().toShort()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val derivativeData = IntArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toInt()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                                val derivativeData = LongArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toLong()
                                }
                                derivative.data = derivativeData
                            }
                            else -> {} // Should not happen due to the when condition
                        }
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    derivative,
                    zeroTable
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM -> {
            // For RMS_NORM operation C = x / sqrt(mean(x^2) + eps):
            // grad_x = grad_C * (1 / sqrt(mean(x^2) + eps) - x^2 / (mean(x^2) + eps)^(3/2) / n)
            // where n is the number of elements in x and eps is a small constant for numerical stability

            if (src0?.grad != null) {
                // Calculate total size
                val totalSize = io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(src0.ne).toInt()
                val eps = 1e-5f // Small constant for numerical stability

                // First, we need to compute the mean of squared values
                var meanSquared = 0.0f
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        for (i in 0 until totalSize) {
                            meanSquared += srcData[i] * srcData[i]
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat() / 32768.0f
                            meanSquared += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                        val srcData = src0.data as ByteArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            meanSquared += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                        val srcData = src0.data as ShortArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            meanSquared += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                        val srcData = src0.data as IntArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            meanSquared += value * value
                        }
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        val srcData = src0.data as LongArray
                        for (i in 0 until totalSize) {
                            val value = srcData[i].toFloat()
                            meanSquared += value * value
                        }
                    }
                    else -> {
                        error("fatal error")
                    }
                }
                meanSquared /= totalSize
                meanSquared += eps // Add epsilon for numerical stability

                val rmsFactor = 1.0f / kotlin.math.sqrt(meanSquared)
                val rmsFactorCubed = rmsFactor * rmsFactor * rmsFactor

                // Create a tensor for the derivative
                val derivative = io.github.kotlinmania.llama.ore.GGMLTensor(type = src0.type)
                for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    derivative.ne[i] = src0.ne[i]
                    derivative.nb[i] = src0.nb[i]
                }

                // Compute the derivative based on the tensor type
                when (src0.type) {
                    io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                        val srcData = src0.data as FloatArray
                        val derivativeData = FloatArray(totalSize)
                        val gradData = tensor.grad!!.data as FloatArray

                        // Compute the derivative for each element
                        for (i in 0 until totalSize) {
                            // Derivative: grad_C * (rmsFactor - x^2 * rmsFactorCubed / n)
                            val term1 = rmsFactor
                            val term2 = srcData[i] * srcData[i] * rmsFactorCubed / totalSize
                            derivativeData[i] = gradData[i] * (term1 - term2)
                        }

                        derivative.data = derivativeData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                        val srcData = src0.data as ShortArray
                        val derivativeData = ShortArray(totalSize)
                        val gradData = FloatArray(totalSize)

                        // Convert gradient data to float
                        for (i in 0 until totalSize) {
                            gradData[i] = (tensor.grad!!.data as ShortArray)[i].toFloat() / 32768.0f
                        }

                        // Compute the derivative for each element
                        for (i in 0 until totalSize) {
                            val x = srcData[i].toFloat() / 32768.0f
                            // Derivative: grad_C * (rmsFactor - x^2 * rmsFactorCubed / n)
                            val term1 = rmsFactor
                            val term2 = x * x * rmsFactorCubed / totalSize
                            val derivValue = gradData[i] * (term1 - term2)
                            derivativeData[i] = (derivValue * 32768.0f).toInt().toShort()
                        }

                        derivative.data = derivativeData
                    }
                    io.github.kotlinmania.llama.ore.GGMLType.I8, io.github.kotlinmania.llama.ore.GGMLType.I16, io.github.kotlinmania.llama.ore.GGMLType.I32, io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                        // For integer types, convert to float, compute derivative, then convert back
                        val srcFloatData = FloatArray(totalSize)
                        val gradFloatData = FloatArray(totalSize)
                        val derivativeFloatData = FloatArray(totalSize)

                        // Convert source and gradient data to float
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val srcData = src0.data as ByteArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as ByteArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val srcData = src0.data as ShortArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as ShortArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val srcData = src0.data as IntArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as IntArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                                val srcData = src0.data as LongArray
                                for (i in 0 until totalSize) {
                                    srcFloatData[i] = srcData[i].toFloat()
                                }
                                val gradData = tensor.grad!!.data as LongArray
                                for (i in 0 until totalSize) {
                                    gradFloatData[i] = gradData[i].toFloat()
                                }
                            }
                            else -> {} // Should not happen due to the when condition
                        }

                        // Compute the derivative for each element
                        for (i in 0 until totalSize) {
                            // Derivative: grad_C * (rmsFactor - x^2 * rmsFactorCubed / n)
                            val term1 = rmsFactor
                            val term2 = srcFloatData[i] * srcFloatData[i] * rmsFactorCubed / totalSize
                            derivativeFloatData[i] = gradFloatData[i] * (term1 - term2)
                        }

                        // Convert back to the original type
                        when (src0.type) {
                            io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
                                val derivativeData = ByteArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toInt().toByte()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
                                val derivativeData = ShortArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toInt().toShort()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                                val derivativeData = IntArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toInt()
                                }
                                derivative.data = derivativeData
                            }
                            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                                val derivativeData = LongArray(totalSize)
                                for (i in 0 until totalSize) {
                                    derivativeData[i] = derivativeFloatData[i].toLong()
                                }
                                derivative.data = derivativeData
                            }
                            else -> {} // Should not happen due to the when condition
                        }
                    }
                    else -> {
                        // For other types, we'll implement later
                        error("fatal error")
                    }
                }

                // Add to source gradient
                src0.grad = io.github.kotlinmania.llama.ore.addOrSet(
                    context,
                    src0.grad,
                    derivative,
                    zeroTable
                )
            }
        }
        else -> {
            // For other operations, we'll implement later
            error("fatal error")
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
private fun buildForwardImpl(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, visited: MutableSet<io.github.kotlinmania.llama.ore.GGMLTensor>) {
    // If we've already visited this tensor, return
    if (tensor in visited) {
        return
    }

    // Mark the tensor as visited
    visited.add(tensor)

    // If the tensor has no operation, it's a leaf node
    if (tensor.op == io.github.kotlinmania.llama.ore.GGMLOp.NONE) {
        cgraph.leafs[cgraph.nLeafs++] = tensor
        return
    }

    // Recursively build the graph for the source tensors
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
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
fun buildForward(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, tensor: io.github.kotlinmania.llama.ore.GGMLTensor) {
    // Reset the graph
    cgraph.nNodes = 0
    cgraph.nLeafs = 0

    // Build the graph
    val visited = mutableSetOf<io.github.kotlinmania.llama.ore.GGMLTensor>()
    io.github.kotlinmania.llama.ore.buildForwardImpl(tensor, cgraph, visited)

    // Set the order to forward
    cgraph.order = io.github.kotlinmania.llama.ore.GGMLCGraphEvalOrder.FORWARD
}

/**
 * Builds a backward computation graph for automatic differentiation.
 *
 * @param context The GGML context
 * @param gf The forward computation graph
 * @param gb The backward computation graph to build
 * @param keep Whether to keep the original gradients
 */
fun buildBackward(context: io.github.kotlinmania.llama.ore.GGMLContext, gf: io.github.kotlinmania.llama.ore.GGMLCGraph, gb: io.github.kotlinmania.llama.ore.GGMLCGraph, keep: Boolean = true) {
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
                val gradDup = io.github.kotlinmania.llama.ore.GGMLTensor(type = node.grad!!.type)
                for (j in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
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
    val zeroTable = mutableSetOf<io.github.kotlinmania.llama.ore.GGMLTensor>()
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
            io.github.kotlinmania.llama.ore.computeBackward(context, node, zeroTable)
        }
    }

    // Build forward graph for parameter gradients
    gb.nNodes = 0
    gb.nLeafs = 0

    for (i in 0 until gf.nNodes) {
        val node = gf.nodes[i] ?: continue

        if (node.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_PARAM != 0) {
            // This is a parameter tensor, add its gradient to the backward graph
            if (node.grad != null) {
                io.github.kotlinmania.llama.ore.buildForward(gb, node.grad!!)
            }
        }
    }

    // Set the order to backward
    gb.order = io.github.kotlinmania.llama.ore.GGMLCGraphEvalOrder.BACKWARD
}

/**
 * Executes a computation graph.
 *
 * @param context The GGML context
 * @param cgraph The computation graph to execute
 */
fun executeGraph(context: io.github.kotlinmania.llama.ore.GGMLContext, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    // Unified execution path: ensure allocation, then compute via backend or CPU compute ops
    val allocator = cgraph.allocator ?: io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
    // Attach allocator if missing
    if (cgraph.allocator == null) {
        cgraph.allocator = allocator
    }

    // Allocate buffers for the graph
    val allocated = allocator.allocateGraph(cgraph)
    if (!allocated) throw IllegalStateException("Failed to allocate graph buffers")

    // Prefer backend if available, otherwise use CPU compute ops
    val backend = allocator.backend
    if (backend != null) {
        val status = backend.graphCompute(cgraph)
        if (status != io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS) throw IllegalStateException("Backend graph compute failed: $status")
    } else {
        io.github.kotlinmania.llama.ore.GGMLComputeOps.computeGraph(cgraph)
    }
}

/**
 * Executes a computation graph in backward order.
 *
 * @param context The GGML context
 * @param cgraph The computation graph to execute
 */
private fun executeBackward(context: io.github.kotlinmania.llama.ore.GGMLContext, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    // Legacy path replaced: ensure allocation and compute via backend/CPU
    io.github.kotlinmania.llama.ore.computeGraphWithBackend(cgraph)
}

/**
 * Executes a computation graph in forward order.
 *
 * @param context The GGML context
 * @param cgraph The computation graph to execute
 */
private fun executeForward(context: io.github.kotlinmania.llama.ore.GGMLContext, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    // Legacy path replaced: ensure allocation and compute via backend/CPU
    io.github.kotlinmania.llama.ore.computeGraphWithBackend(cgraph)
}

/**
 * Executes a single node in the computation graph.
 *
 * @param context The GGML context
 * @param node The node to execute
 */
private fun executeNode(context: io.github.kotlinmania.llama.ore.GGMLContext, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
    // Legacy per-node execution replaced by backend/CPU graph compute.
    // Build a temporary single-node graph and compute it via unified path.
    val tempGraph = io.github.kotlinmania.llama.ore.createGraph(1)
    tempGraph.nodes[0] = node
    tempGraph.nNodes = 1
    io.github.kotlinmania.llama.ore.computeGraphWithBackend(tempGraph)
}

/**
 * Creates a new computation graph.
 *
 * @param size The maximum number of nodes in the graph
 * @return The new computation graph
 */
fun createGraph(size: Int, backend: io.github.kotlinmania.llama.ore.GGMLBackend? = null): io.github.kotlinmania.llama.ore.GGMLCGraph {
    val allocator = if (backend != null) {
        io.github.kotlinmania.llama.ore.GGMLGraphAllocator(backend)
    } else {
        io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
    }
    
    return io.github.kotlinmania.llama.ore.GGMLCGraph(
        size = size,
        nNodes = 0,
        nLeafs = 0,
        nodes = Array(size) { null },
        grads = Array(size) { null },
        leafs = Array(size) { null },
        visitedHashSet = null,
        order = io.github.kotlinmania.llama.ore.GGMLCGraphEvalOrder.NONE,
        allocator = allocator
    )
}

/**
 * Compute a graph using its associated backend.
 * Falls back to the existing executeGraph function if no backend is available.
 * 
 * @param graph The graph to compute
 * @param context Optional context for legacy execution
 * @return The computation status
 */
fun computeGraphWithBackend(graph: io.github.kotlinmania.llama.ore.GGMLCGraph, context: io.github.kotlinmania.llama.ore.GGMLContext? = null): io.github.kotlinmania.llama.ore.GGMLStatus {
    val allocator = graph.allocator ?: io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
        .also { graph.allocator = it }
    val backend = allocator.backend

    // Ensure graph is allocated before compute
    return try {
        val ok = allocator.allocateGraph(graph)
        if (!ok) return io.github.kotlinmania.llama.ore.GGMLStatus.FAILED

        if (backend != null) {
            backend.graphCompute(graph)
        } else {
            io.github.kotlinmania.llama.ore.GGMLComputeOps.computeGraph(graph)
            io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS
        }
    } catch (e: Exception) {
        println("Graph computation failed: ${e.message}")
        io.github.kotlinmania.llama.ore.GGMLStatus.FAILED
    }
}

// ============================================================================
// Ported from ggml/src/ggml.c — global state, timing, graph management
// ============================================================================

/** Atomic counter for unique graph IDs. Ported from ggml_graph_next_uid. */
private val graphUidCounter = AtomicLong(1L)

/** Returns the next unique graph identifier. Thread-safe via atomic increment. */
fun ggmlGraphNextUid(): Long = io.github.kotlinmania.llama.ore.graphUidCounter.fetchAndAdd(1L)

// ---------------------------------------------------------------------------
// Timing (ported from ggml_time_init / ggml_time_ms / ggml_time_us)
// ---------------------------------------------------------------------------

/** No-op on JVM/Native — monotonic clocks need no initialisation. */
fun ggmlTimeInit() { /* identity on Kotlin targets */ }

private val timeOrigin = TimeSource.Monotonic.markNow()

/** Current wall-clock time in milliseconds (monotonic). */
fun ggmlTimeMs(): Long = io.github.kotlinmania.llama.ore.timeOrigin.elapsedNow().inWholeMilliseconds

/** Current wall-clock time in microseconds (monotonic). */
fun ggmlTimeUs(): Long = io.github.kotlinmania.llama.ore.timeOrigin.elapsedNow().inWholeMicroseconds

// ---------------------------------------------------------------------------
// Status helpers (ported from ggml_status_to_string)
// ---------------------------------------------------------------------------

/** Converts a [io.github.kotlinmania.llama.ore.GGMLStatus] to a human-readable description. */
fun ggmlStatusToString(status: io.github.kotlinmania.llama.ore.GGMLStatus): String = when (status) {
    io.github.kotlinmania.llama.ore.GGMLStatus.ALLOC_FAILED -> "GGML status: error (failed to allocate memory)"
    io.github.kotlinmania.llama.ore.GGMLStatus.FAILED       -> "GGML status: error (operation failed)"
    io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS      -> "GGML status: success"
    io.github.kotlinmania.llama.ore.GGMLStatus.ABORTED      -> "GGML status: warning (operation aborted)"
}

// ---------------------------------------------------------------------------
// Logging (ported from ggml_log_internal / ggml_log_set / ggml_log_callback_default)
// ---------------------------------------------------------------------------

// Logging functions (ggmlLogInternal, ggmlLogSet, GGMLLogCallback, ggmlLogCallbackDefault)
// moved to NumericConversions.kt (ggml-impl.h)

// ---------------------------------------------------------------------------
// Tensor shape predicates (supplements already in GGMLOps.kt)
// ---------------------------------------------------------------------------

/** True when stride layout indicates channel-last ordering. */
fun ggmlIsContiguousChannels(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.nb[0] > tensor.nb[2] &&
    tensor.nb[1] > tensor.nb[0] &&
    tensor.nb[2] == io.github.kotlinmania.llama.ore.ggmlTypeSize(tensor.type)

/** True when rows are contiguous in memory. */
fun ggmlIsContiguousRows(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.ne[0] == io.github.kotlinmania.llama.ore.ggmlBlckSize(tensor.type) ||
    tensor.nb[0] == io.github.kotlinmania.llama.ore.ggmlTypeSize(tensor.type)

/** True when `nb[0]` is type-sized and higher dims pack tightly. */
fun ggmlIsPadded1d(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.nb[0] == io.github.kotlinmania.llama.ore.ggmlTypeSize(tensor.type) &&
    tensor.nb[2] == tensor.nb[1] * tensor.ne[1].toULong() &&
    tensor.nb[3] == tensor.nb[2] * tensor.ne[2].toULong()

/** True when t0 can be broadcast row-wise onto t1. */
fun ggmlCanRepeatRows(t0: io.github.kotlinmania.llama.ore.GGMLTensor, t1: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    t0.ne[0] == t1.ne[0] && io.github.kotlinmania.llama.ore.ggmlCanRepeat(t0, t1)

// ---------------------------------------------------------------------------
// Tensor utilities ported from ggml.c
// ---------------------------------------------------------------------------

/**
 * Converts a flat element index into per-dimension indices for [tensor].
 * Ported from `ggml_unravel_index`.
 */
fun ggmlUnravelIndex(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, i: Long): LongArray {
    val ne0 = tensor.ne[0]; val ne1 = tensor.ne[1]; val ne2 = tensor.ne[2]
    val i3 = i / (ne2 * ne1 * ne0)
    val i2 = (i - i3 * ne2 * ne1 * ne0) / (ne1 * ne0)
    val i1 = (i - i3 * ne2 * ne1 * ne0 - i2 * ne1 * ne0) / ne0
    val i0 = i - i3 * ne2 * ne1 * ne0 - i2 * ne1 * ne0 - i1 * ne0
    return longArrayOf(i0, i1, i2, i3)
}

// Note: ggmlSetZero is already defined in GGMLOps.kt.

// Note: ggmlSetLoss, GGML_TENSOR_FLAG_LOSS, GGML_TENSOR_FLAG_INPUT,
// and GGML_TENSOR_FLAG_COMPUTE are defined in GGMLOps.kt.

// ---------------------------------------------------------------------------
// Graph view / copy / dup / reset / print (ported from ggml.c ~7127-7337)
// ---------------------------------------------------------------------------

/**
 * Creates a lightweight view into [cgraph] spanning nodes `[i0, i1)`.
 * The returned graph shares the underlying node array — no deep copy.
 * Ported from `ggml_graph_view`.
 */
fun ggmlGraphView(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, i0: Int, i1: Int): io.github.kotlinmania.llama.ore.GGMLCGraph {
    require(i0 in 0..cgraph.nNodes && i1 in i0..cgraph.nNodes)
    return io.github.kotlinmania.llama.ore.GGMLCGraph(
        size = 0,
        nNodes = i1 - i0,
        nLeafs = 0,
        nodes = cgraph.nodes.copyOfRange(i0, i1),
        grads = emptyArray(),
        gradAccs = emptyArray(),
        leafs = emptyArray(),
        useCounts = cgraph.useCounts,
        visitedHashSet = cgraph.visitedHashSet,
        order = cgraph.order,
        uid = 0L
    )
}

/**
 * Deep-copies [src] graph data into [dst]. [dst] must be large enough.
 * Ported from `ggml_graph_cpy`.
 */
fun ggmlGraphCpy(src: io.github.kotlinmania.llama.ore.GGMLCGraph, dst: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    require(dst.size >= src.nLeafs) { "dst too small for leafs" }
    require(dst.size >= src.nNodes) { "dst too small for nodes" }
    dst.nLeafs = src.nLeafs
    dst.nNodes = src.nNodes
    dst.order  = src.order
    for (i in 0 until src.nLeafs) dst.leafs[i] = src.leafs[i]
    for (i in 0 until src.nNodes) dst.nodes[i] = src.nodes[i]
}

/**
 * Duplicates [cgraph] into a new graph allocated from [ctx].
 * Ported from `ggml_graph_dup`.
 */
fun ggmlGraphDup(ctx: io.github.kotlinmania.llama.ore.GGMLContext, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, forceGrads: Boolean = false): io.github.kotlinmania.llama.ore.GGMLCGraph {
    val result = io.github.kotlinmania.llama.ore.ggmlNewGraphCustom(
        ctx,
        cgraph.size.toULong(),
        cgraph.grads.isNotEmpty() || forceGrads
    )
    io.github.kotlinmania.llama.ore.ggmlGraphCpy(cgraph, result)
    return result
}

/**
 * Resets gradient accumulators in [cgraph].
 * Loss nodes get gradient = 1.0f; all others are zeroed.
 * Ported from `ggml_graph_reset`.
 */
fun ggmlGraphReset(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    for (i in 0 until cgraph.nNodes) {
        val node = cgraph.nodes[i] ?: continue
        // If the node has a gradient accumulator, zero it (or set to 1.0 for loss)
        val grad = cgraph.grads.getOrNull(i)
        if (grad != null) {
            if (node.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_LOSS != 0) {
                require(grad.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
                require(io.github.kotlinmania.llama.ore.ggmlIsScalar(grad))
                val data = grad.data
                if (data is FloatArray && data.isNotEmpty()) data[0] = 1.0f
            } else {
                io.github.kotlinmania.llama.ore.ggmlSetZero(grad)
            }
        }
    }
}

/**
 * Prints a human-readable summary of [cgraph] to the log.
 * Ported from `ggml_graph_print`.
 */
fun ggmlGraphPrint(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    io.github.kotlinmania.llama.ore.ggmlLogInternal(
        io.github.kotlinmania.llama.ore.GGMLLogLevel.INFO,
        "=== GRAPH ===\n"
    )
    io.github.kotlinmania.llama.ore.ggmlLogInternal(
        io.github.kotlinmania.llama.ore.GGMLLogLevel.INFO,
        "n_nodes = ${cgraph.nNodes}\n"
    )
    for (i in 0 until cgraph.nNodes) {
        val node = cgraph.nodes[i] ?: continue
        val flag = if (node.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_PARAM != 0) "x"
                   else if (cgraph.grads.getOrNull(i) != null) "g"
                   else " "
        io.github.kotlinmania.llama.ore.ggmlLogInternal(
            io.github.kotlinmania.llama.ore.GGMLLogLevel.INFO,
            " - ${i.toString().padStart(3)}: [${node.ne[0].toString().padStart(5)}, ${
                node.ne[1].toString().padStart(5)
            }, ${node.ne[2].toString().padStart(5)}] ${
                io.github.kotlinmania.llama.ore.ggmlOpName(
                    node.op
                ).padStart(16)
            } $flag\n"
        )
    }
    io.github.kotlinmania.llama.ore.ggmlLogInternal(
        io.github.kotlinmania.llama.ore.GGMLLogLevel.INFO,
        "n_leafs = ${cgraph.nLeafs}\n"
    )
    for (i in 0 until cgraph.nLeafs) {
        val leaf = cgraph.leafs[i] ?: continue
        io.github.kotlinmania.llama.ore.ggmlLogInternal(
            io.github.kotlinmania.llama.ore.GGMLLogLevel.INFO,
            " - ${i.toString().padStart(3)}: [${leaf.ne[0].toString().padStart(5)}, ${
                leaf.ne[1].toString().padStart(5)
            }] ${
                io.github.kotlinmania.llama.ore.ggmlOpName(leaf.op).padStart(8)
            } ${leaf.name.padStart(16)}\n"
        )
    }
    io.github.kotlinmania.llama.ore.ggmlLogInternal(
        io.github.kotlinmania.llama.ore.GGMLLogLevel.INFO,
        "========================================\n"
    )
}

// Note: ggmlGraphGetTensor is already defined in GGMLOps.kt.

/**
 * Returns the gradient tensor for [node] inside [cgraph], or null.
 * Ported from `ggml_graph_get_grad` (simplified — uses array index).
 */
fun ggmlGraphGetGrad(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, node: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor? {
    if (cgraph.grads.isEmpty()) return null
    for (i in 0 until cgraph.nNodes) {
        if (cgraph.nodes[i] === node) return cgraph.grads.getOrNull(i)
    }
    return null
}

/**
 * Returns the gradient accumulator tensor for [node], or null.
 * Ported from `ggml_graph_get_grad_acc`.
 */
fun ggmlGraphGetGradAcc(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, node: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor? {
    if (cgraph.gradAccs.isEmpty()) return null
    for (i in 0 until cgraph.nNodes) {
        if (cgraph.nodes[i] === node) return cgraph.gradAccs.getOrNull(i)
    }
    return null
}

/**
 * True when [node] appears in [cgraph] nodes.
 * Ported from `ggml_graph_find`.
 */
fun ggmlGraphFind(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph?, node: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    if (cgraph == null) return true
    for (i in 0 until cgraph.nNodes) {
        if (cgraph.nodes[i] === node) return true
    }
    return false
}

// ggmlBuildBackwardExpand moved to GGMLOps.kt (ggml.h)

// ---------------------------------------------------------------------------
// Graph compute plan (minimal — actual scheduling is backend-specific)
// ---------------------------------------------------------------------------

/**
 * Parameters describing how a graph computation should be executed.
 * Mirrors `struct ggml_cplan` from ggml.c.
 */
data class GGMLComputePlan(
    /** Size of the required work buffer in bytes. */
    val workSize: ULong = 0u,
    /** Work buffer (allocated by caller if workSize > 0). */
    var workData: Any? = null,
    /** Number of threads to use. */
    var nThreads: Int = 1,
    /** Optional threadpool handle. */
    var threadpool: Any? = null,
    /** Optional abort callback. */
    var abortCallback: (() -> Boolean)? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGMLComputePlan) return false
        return workSize == other.workSize && nThreads == other.nThreads
    }
    override fun hashCode(): Int = workSize.hashCode() * 31 + nThreads
}

/**
 * Creates a compute plan for [cgraph]. Currently returns a minimal plan;
 * work-buffer sizing will be filled in when the CPU backend is formalised.
 * Ported from `ggml_graph_plan` (minimal).
 */
fun ggmlGraphPlan(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, nThreads: Int = 1): io.github.kotlinmania.llama.ore.GGMLComputePlan {
    return io.github.kotlinmania.llama.ore.GGMLComputePlan(nThreads = nThreads)
}

/**
 * Executes [cgraph] according to [plan].
 * Ported from `ggml_graph_compute` (delegates to backend infrastructure).
 */
fun ggmlGraphCompute(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, plan: io.github.kotlinmania.llama.ore.GGMLComputePlan): io.github.kotlinmania.llama.ore.GGMLStatus {
    return io.github.kotlinmania.llama.ore.computeGraphWithBackend(cgraph)
}

// ---------------------------------------------------------------------------
// Version / commit info
// ---------------------------------------------------------------------------

/** GGML version string. */
fun ggmlVersion(): String = "0.0.0-kotlin"

/** Placeholder for the build commit hash. */
fun ggmlCommit(): String = "unknown"

// ---------------------------------------------------------------------------
// FType → GGMLType conversion (ported from ggml_ftype_to_ggml_type)
// ---------------------------------------------------------------------------

// GGMLFType enum is defined in GGMLOps.kt.

/** Converts a file-level type hint to the corresponding tensor data type. */
fun ggmlFtypeToGgmlType(ftype: io.github.kotlinmania.llama.ore.GGMLFType): io.github.kotlinmania.llama.ore.GGMLType = when (ftype) {
    io.github.kotlinmania.llama.ore.GGMLFType.ALL_F32     -> io.github.kotlinmania.llama.ore.GGMLType.F32
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_F16  -> io.github.kotlinmania.llama.ore.GGMLType.F16
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q4_0 -> io.github.kotlinmania.llama.ore.GGMLType.Q4_0
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q4_1 -> io.github.kotlinmania.llama.ore.GGMLType.Q4_1
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q5_0 -> io.github.kotlinmania.llama.ore.GGMLType.Q5_0
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q5_1 -> io.github.kotlinmania.llama.ore.GGMLType.Q5_1
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q8_0 -> io.github.kotlinmania.llama.ore.GGMLType.Q8_0
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q2_K -> io.github.kotlinmania.llama.ore.GGMLType.Q2_K
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q3_K -> io.github.kotlinmania.llama.ore.GGMLType.Q3_K
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q4_K -> io.github.kotlinmania.llama.ore.GGMLType.Q4_K
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q5_K -> io.github.kotlinmania.llama.ore.GGMLType.Q5_K
    io.github.kotlinmania.llama.ore.GGMLFType.MOSTLY_Q6_K -> io.github.kotlinmania.llama.ore.GGMLType.Q6_K
    else -> error("Cannot convert ftype ${ftype.name} to ggml_type")
}
