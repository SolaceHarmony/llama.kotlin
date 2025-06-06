package ggml

import org.llama.core.*

/** Utility functions and compute operations implemented per GGML_COMPUTE_OPS_DESIGN.md */

/**
 * Calculates the total size of a tensor based on its dimensions.
 */
fun calculateTotalSize(ne: LongArray): Int {
    var totalSize = 1
    for (i in 0 until GGML_MAX_DIMS) {
        totalSize *= ne[i].toInt()
    }
    return totalSize
}

/**
 * Allocates memory for a tensor based on its type and size.
 */
fun allocateMemory(type: GGMLType, size: Int): Any {
    return when (type) {
        GGMLType.F32 -> FloatArray(size) { 0.0f }
        GGMLType.F16 -> ShortArray(size) { 0 }
        GGMLType.I8  -> ByteArray(size) { 0 }
        GGMLType.I16 -> ShortArray(size) { 0 }
        GGMLType.I32 -> IntArray(size) { 0 }
        GGMLType.I64 -> LongArray(size) { 0L }
        else -> ByteArray(size) { 0 } // TODO: support quantized types
    }
}

/**
 * Adds two tensors element-wise.
 */
fun computeAdd(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) {
            throw IllegalArgumentException("Incompatible dimensions for addition")
        }
    }

    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    val totalSize = calculateTotalSize(a.ne)

    when (a.type) {
        GGMLType.F32 -> {
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val res = FloatArray(totalSize)
            for (i in 0 until totalSize) {
                res[i] = aData[i] + bData[i]
            }
            result.data = res
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val res = IntArray(totalSize)
            for (i in 0 until totalSize) {
                res[i] = aData[i] + bData[i]
            }
            result.data = res
        }
        else -> {
            // TODO: implement for remaining types
            result.data = null
        }
    }
    return result
}

/**
 * Multiplies two tensors element-wise.
 */
fun computeMul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) {
            throw IllegalArgumentException("Incompatible dimensions for multiplication")
        }
    }

    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    val totalSize = calculateTotalSize(a.ne)

    when (a.type) {
        GGMLType.F32 -> {
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val res = FloatArray(totalSize)
            for (i in 0 until totalSize) {
                res[i] = aData[i] * bData[i]
            }
            result.data = res
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val res = IntArray(totalSize)
            for (i in 0 until totalSize) {
                res[i] = aData[i] * bData[i]
            }
            result.data = res
        }
        else -> {
            // TODO: implement for remaining types
            result.data = null
        }
    }
    return result
}

/**
 * Performs matrix multiplication of two tensors.
 */
fun computeMatMul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val m = a.ne[0]
    val n = a.ne[1]
    val p = b.ne[1]
    if (b.ne[0] != n) {
        throw IllegalArgumentException("Incompatible dimensions for matrix multiplication")
    }

    val result = GGMLTensor(type = a.type)
    result.ne[0] = m
    result.ne[1] = p
    for (i in 2 until GGML_MAX_DIMS) {
        result.ne[i] = 1
    }

    val typeSize = when (a.type) {
        GGMLType.F32 -> 4u
        GGMLType.F16 -> 2u
        GGMLType.I8  -> 1u
        GGMLType.I16 -> 2u
        GGMLType.I32 -> 4u
        GGMLType.I64 -> 8u
        else -> 1u // TODO: handle quantized types
    }

    result.nb[0] = typeSize
    result.nb[1] = result.nb[0] * result.ne[0].toULong()
    for (i in 2 until GGML_MAX_DIMS) {
        result.nb[i] = result.nb[i - 1] * result.ne[i - 1].toULong()
    }

    val totalSize = (m * p).toInt()

    when (a.type) {
        GGMLType.F32 -> {
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val res = FloatArray(totalSize)
            for (i in 0 until m.toInt()) {
                for (j in 0 until p.toInt()) {
                    var sum = 0.0f
                    for (k in 0 until n.toInt()) {
                        sum += aData[i * n.toInt() + k] * bData[k * p.toInt() + j]
                    }
                    res[i * p.toInt() + j] = sum
                }
            }
            result.data = res
        }
        else -> {
            // TODO: implement for remaining types
            result.data = null
        }
    }
    return result
}

/**
 * Applies the ReLU activation function to a tensor.
 */
fun computeRelu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    val totalSize = calculateTotalSize(a.ne)

    when (a.type) {
        GGMLType.F32 -> {
            val input = a.data as FloatArray
            val res = FloatArray(totalSize)
            for (i in 0 until totalSize) {
                res[i] = if (input[i] > 0.0f) input[i] else 0.0f
            }
            result.data = res
        }
        else -> {
            // TODO: implement for remaining types
            result.data = null
        }
    }
    return result
}

/**
 * Applies the GELU activation function to a tensor.
 */
fun computeGelu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    val totalSize = calculateTotalSize(a.ne)

    when (a.type) {
        GGMLType.F32 -> {
            val input = a.data as FloatArray
            val res = FloatArray(totalSize)
            for (i in 0 until totalSize) {
                val x = input[i]
                res[i] = x * 0.5f * (1.0f + kotlin.math.tanh(0.797885f * (x + 0.044715f * x * x * x)))
            }
            result.data = res
        }
        else -> {
            // TODO: implement for remaining types
            result.data = null
        }
    }
    return result
}

// TODO: SIMD implementations and multi-threaded variants for performance
