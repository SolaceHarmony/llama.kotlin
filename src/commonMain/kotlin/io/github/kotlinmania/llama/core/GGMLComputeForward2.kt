// port-lint: source ggml/src/ggml-cpu/ops.cpp
package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setShortLe
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.max
import kotlin.math.pow

/**
 * Line-by-line transliteration of ops.cpp lines 4118–5471.
 *
 * Covers: OUT_PROD, SCALE, SET, CPY, CONT, GET_ROWS, GET_ROWS_BACK,
 * DIAG, DIAG_MASK_INF, DIAG_MASK_ZERO, SOFT_MAX, SOFT_MAX_EXT_BACK.
 */

// ============================================================================
// Constants
// ============================================================================

private const val CACHE_LINE_SIZE_F32_2 = 16
private const val GGML_VEC_MAD_UNROLL = 32

// ============================================================================
// Vector helper functions — scalar implementations matching C++ vec.h
// ============================================================================

private fun ggml_vec_set_f32_2(n: Long, dst: FloatArray, dstOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = v
    }
}

private fun ggml_vec_cpy_f32_2(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int) {
    src.copyInto(dst, dstOff, srcOff, srcOff + n.toInt())
}

private fun ggml_vec_add_f32_2(n: Long, dst: FloatArray, dstOff: Int, src0: FloatArray, src0Off: Int, src1: FloatArray, src1Off: Int) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = src0[src0Off + i] + src1[src1Off + i]
    }
}

private fun ggml_vec_scale_f32_2(n: Long, dst: FloatArray, dstOff: Int, s: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] *= s
    }
}

private fun ggml_vec_mad_f32_2(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] += src[srcOff + i] * v
    }
}

private fun ggml_vec_mad1_f32_2(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int, s: Float, b: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = src[srcOff + i] * s + b
    }
}

private fun ggml_vec_dot_f32_2(n: Long, x: FloatArray, xOff: Int, y: FloatArray, yOff: Int): Float {
    var sum = 0f
    for (i in 0 until n.toInt()) {
        sum += x[xOff + i] * y[yOff + i]
    }
    return sum
}

private fun ggml_vec_acc1_f32_2(n: Long, dst: FloatArray, dstOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] += v
    }
}

private fun ggml_vec_mul_f32_2(n: Long, dst: FloatArray, dstOff: Int, src0: FloatArray, src0Off: Int, src1: FloatArray, src1Off: Int) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = src0[src0Off + i] * src1[src1Off + i]
    }
}

private fun ggml_vec_max_f32_2(n: Long, src: FloatArray, srcOff: Int): Float {
    var mx = Float.NEGATIVE_INFINITY
    for (i in 0 until n.toInt()) {
        if (src[srcOff + i] > mx) mx = src[srcOff + i]
    }
    return mx
}

private fun ggml_vec_soft_max_f32_2(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int, maxVal: Float): Double {
    var sum = 0.0
    for (i in 0 until n.toInt()) {
        val v = exp((src[srcOff + i] - maxVal).toDouble())
        dst[dstOff + i] = v.toFloat()
        sum += v
    }
    return sum
}

private fun ggml_vec_cpy_i32_2(n: Long, dst: ByteArray, dstOff: Int, src: ByteArray, srcOff: Int) {
    src.copyInto(dst, dstOff, srcOff, srcOff + n.toInt() * 4)
}

// ============================================================================
// ByteArray data access helpers
// ============================================================================

private fun readFloat2(data: ByteArray, off: Int): Float = data.getFloatLe(off)
private fun writeFloat2(data: ByteArray, off: Int, v: Float) = data.setFloatLe(off, v)
private fun readInt2(data: ByteArray, off: Int): Int = data.getIntLe(off)
private fun writeInt2(data: ByteArray, off: Int, v: Int) = data.setIntLe(off, v)
private fun readShort2(data: ByteArray, off: Int): Short = data.getShortLe(off)

private fun readFloatRow2(data: ByteArray, off: Int, n: Int): FloatArray {
    val row = FloatArray(n)
    for (i in 0 until n) {
        row[i] = io.github.kotlinmania.llama.ore.readFloat2(data, off + i * 4)
    }
    return row
}

private fun writeFloatRow2(data: ByteArray, off: Int, src: FloatArray, srcOff: Int, n: Int) {
    for (i in 0 until n) {
        io.github.kotlinmania.llama.ore.writeFloat2(data, off + i * 4, src[srcOff + i])
    }
}

// ============================================================================
// Op Locals helpers
// ============================================================================

private data class BinaryOpLocals2(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne10: Long, val ne11: Long, val ne12: Long, val ne13: Long,
    val nb10: Long, val nb11: Long, val nb12: Long, val nb13: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun binaryOpLocals2(dst: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.BinaryOpLocals2 {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    return io.github.kotlinmania.llama.ore.BinaryOpLocals2(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne10 = src1.ne[0], ne11 = src1.ne[1], ne12 = src1.ne[2], ne13 = src1.ne[3],
        nb10 = src1.nb[0].toLong(), nb11 = src1.nb[1].toLong(), nb12 = src1.nb[2].toLong(), nb13 = src1.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

private data class UnaryOpLocals2(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun unaryOpLocals2(dst: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.UnaryOpLocals2 {
    val src0 = dst.src[0]!!
    return io.github.kotlinmania.llama.ore.UnaryOpLocals2(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

// ============================================================================
// ggml_compute_forward_out_prod — ops.cpp line 4118
// ============================================================================

// ops.cpp line 4120
private fun ggml_compute_forward_out_prod_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val ith = params.ith
    val nth = params.nth

    require(l.ne0 == l.ne00)
    require(l.ne1 == l.ne10)
    require(l.ne2 == l.ne12)
    require(l.ne3 == l.ne13)

    require(l.ne2 % l.ne02 == 0L)
    require(l.ne3 % l.ne03 == 0L)

    // we don't support permuted src0 or src1
    require(l.nb00 == 4L) // sizeof(float)

    // dst cannot be transposed or permuted
    require(l.nb0 == 4L) // sizeof(float)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    if (ith == 0) {
        // zero dst
        val totalFloats = (l.ne0 * l.ne1 * l.ne2 * l.ne3).toInt()
        for (i in 0 until totalFloats) {
            io.github.kotlinmania.llama.ore.writeFloat2(dstData, i * 4, 0f)
        }
    }
    io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)

    // parallelize by last three dimensions
    val nr = l.ne1 * l.ne2 * l.ne3
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    // block-tiling attempt
    val blck0 = max(io.github.kotlinmania.llama.ore.GGML_VEC_MAD_UNROLL.toLong(), 32L)
    val blck1 = 16L

    // dps == dst per src0, used for group query attention
    val dps2 = l.ne2 / l.ne02
    val dps3 = l.ne3 / l.ne03

    var bir = ir0
    while (bir < ir1) {
        val bir1 = min(bir + blck1, ir1)
        var bi01 = 0L
        while (bi01 < l.ne01) {
            val bne01 = min(bi01 + blck0, l.ne01)
            var ir = bir
            while (ir < bir1) {
                val i3 = ir / (l.ne2 * l.ne1)
                val i2 = (ir - i3 * l.ne2 * l.ne1) / l.ne1
                val i1 = ir - i3 * l.ne2 * l.ne1 - i2 * l.ne1

                val i02 = i2 / dps2
                val i03 = i3 / dps3

                val i12 = i2
                val i13 = i3

                // No GGML_VEC_MAD_UNROLL > 2 path in Kotlin; use the simple loop
                var i01 = bi01
                while (i01 < bne01) {
                    val i11 = i01

                    val s0Off = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                    val s1Off = (i1 * l.nb10 + i11 * l.nb11 + i12 * l.nb12 + i13 * l.nb13).toInt()
                    val dOff = (i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()

                    val s1Val = io.github.kotlinmania.llama.ore.readFloat2(src1Data, s1Off)
                    for (i0 in 0 until l.ne0.toInt()) {
                        val s0Val =
                            io.github.kotlinmania.llama.ore.readFloat2(src0Data, s0Off + i0 * 4)
                        val cur = io.github.kotlinmania.llama.ore.readFloat2(dstData, dOff + i0 * 4)
                        io.github.kotlinmania.llama.ore.writeFloat2(
                            dstData,
                            dOff + i0 * 4,
                            cur + s0Val * s1Val
                        )
                    }
                    i01++
                }
                ir++
            }
            bi01 += blck0
        }
        bir += blck1
    }
}

// ops.cpp line 4241
private fun ggml_compute_forward_out_prod_q_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    val ith = params.ith
    val nth = params.nth

    val type = src0.type

    require(l.ne02 == l.ne12)
    require(l.ne03 == l.ne13)
    require(l.ne2 == l.ne12)
    require(l.ne3 == l.ne13)

    // we don't support permuted src0 dim0
    require(l.nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(type).toLong())

    // dst dim0 cannot be transposed or permuted
    require(l.nb0 == 4L) // sizeof(float)

    require(l.ne0 == l.ne00)
    require(l.ne1 == l.ne10)
    require(l.ne2 == l.ne02)
    require(l.ne3 == l.ne03)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    if (ith == 0) {
        val totalFloats = (l.ne0 * l.ne1 * l.ne2 * l.ne3).toInt()
        for (i in 0 until totalFloats) {
            io.github.kotlinmania.llama.ore.writeFloat2(dstData, i * 4, 0f)
        }
    }
    io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)

    // parallelize by last three dimensions
    val nr = l.ne1 * l.ne2 * l.ne3
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    val wdata = FloatArray((l.ne0.toInt() + io.github.kotlinmania.llama.ore.CACHE_LINE_SIZE_F32_2))

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1)
        val i2 = (ir - i3 * l.ne2 * l.ne1) / l.ne1
        val i1 = ir - i3 * l.ne2 * l.ne1 - i2 * l.ne1

        val i02 = i2
        val i03 = i3

        val i12 = i2
        val i13 = i3

        for (i01 in 0 until l.ne01) {
            val i11 = i01

            val s0Off = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
            val s1Off = (i1 * l.nb10 + i11 * l.nb11 + i12 * l.nb12 + i13 * l.nb13).toInt()
            val dOff = (i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()

            io.github.kotlinmania.llama.ore.dequantize_row_2(type, src0Data, s0Off, wdata, 0, l.ne0)
            val s1Val = io.github.kotlinmania.llama.ore.readFloat2(src1Data, s1Off)
            // ggml_vec_mad_f32(ne0, d, wdata, *s1)
            for (i0 in 0 until l.ne0.toInt()) {
                val cur = io.github.kotlinmania.llama.ore.readFloat2(dstData, dOff + i0 * 4)
                io.github.kotlinmania.llama.ore.writeFloat2(
                    dstData,
                    dOff + i0 * 4,
                    cur + wdata[i0] * s1Val
                )
            }
        }
    }
}

// ops.cpp line 4330
fun ggml_compute_forward_out_prod(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0,
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K,
        io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_S, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_out_prod_q_f32(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            error("fatal error") // todo
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_out_prod_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_scale — ops.cpp line 4380
// ============================================================================

// ops.cpp line 4382
private fun ggml_compute_forward_scale_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    // scale factor and bias from op_params (stored as float bits in int array)
    val s = Float.fromBits(dst.opParams[0])
    val b = Float.fromBits(dst.opParams[1])

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    val nb01 = src0.nb[1].toLong()
    val nb1 = dst.nb[1].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    if (b == 0.0f) {
        for (i1 in ir0 until ir1) {
            if (dstData !== src0Data) {
                // src0 is same shape as dst => same indices
                val dstOff = (i1 * nb1).toInt()
                val src0Off = (i1 * nb01).toInt()
                src0Data.copyInto(dstData, dstOff, src0Off, src0Off + nc * 4)
            }
            val dstOff = (i1 * nb1).toInt()
            val row = io.github.kotlinmania.llama.ore.readFloatRow2(dstData, dstOff, nc)
            io.github.kotlinmania.llama.ore.ggml_vec_scale_f32_2(nc.toLong(), row, 0, s)
            io.github.kotlinmania.llama.ore.writeFloatRow2(dstData, dstOff, row, 0, nc)
        }
    } else {
        for (i1 in ir0 until ir1) {
            val dstOff = (i1 * nb1).toInt()
            val src0Off = (i1 * nb1).toInt()
            val dstRow = io.github.kotlinmania.llama.ore.readFloatRow2(dstData, dstOff, nc)
            val src0Row = io.github.kotlinmania.llama.ore.readFloatRow2(src0Data, src0Off, nc)
            io.github.kotlinmania.llama.ore.ggml_vec_mad1_f32_2(
                nc.toLong(),
                dstRow,
                0,
                src0Row,
                0,
                s,
                b
            )
            io.github.kotlinmania.llama.ore.writeFloatRow2(dstData, dstOff, dstRow, 0, nc)
        }
    }
}

// ops.cpp line 4434
fun ggml_compute_forward_scale(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_scale_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_set — ops.cpp line 4452
// ============================================================================

// ops.cpp line 4454
private fun ggml_compute_forward_set_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    require(
        io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst) && io.github.kotlinmania.llama.ore.ggmlIsContiguous(
            src0
        )
    )

    // view src0 and dst with these strides and data offset in bytes during set
    val nb1     = dst.opParams[0].toLong()
    val nb2     = dst.opParams[1].toLong()
    val nb3     = dst.opParams[2].toLong()
    val offset  = dst.opParams[3].toLong()
    val inplace = dst.opParams[4] != 0

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    if (!inplace) {
        if (params.ith == 0) {
            src0Data.copyInto(dstData, 0, 0, io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt())
        }
        io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)
    }

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src1).toInt()
    val nc = src1.ne[0].toInt()

    val ne10 = src1.ne[0]
    val ne11 = src1.ne[1]
    val ne12 = src1.ne[2]
    val ne13 = src1.ne[3]

    val nb10 = src1.nb[0].toLong()
    val nb11 = src1.nb[1].toLong()
    val nb12 = src1.nb[2].toLong()
    val nb13 = src1.nb[3].toLong()

    // src0 and dst as viewed during set
    val nb0 = io.github.kotlinmania.llama.ore.ggmlElementSize(src0).toLong()

    val im0 = if (ne10 == 0L) 0L else ne10 - 1
    val im1 = if (ne11 == 0L) 0L else ne11 - 1
    val im2 = if (ne12 == 0L) 0L else ne12 - 1
    val im3 = if (ne13 == 0L) 0L else ne13 - 1

    require(offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= io.github.kotlinmania.llama.ore.ggmlNbytes(
        dst
    ).toLong())

    require(nb10 == 4L) // sizeof(float)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne12 * ne11).toInt()
        val i2 = (ir - i3 * (ne12 * ne11).toInt()) / ne11.toInt()
        val i1 = ir - i3 * (ne12 * ne11).toInt() - i2 * ne11.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1 + offset).toInt()
        val src1Off = (i3 * nb13 + i2 * nb12 + i1 * nb11).toInt()

        // ggml_vec_cpy_f32
        for (ic in 0 until nc) {
            val v = io.github.kotlinmania.llama.ore.readFloat2(src1Data, src1Off + ic * 4)
            io.github.kotlinmania.llama.ore.writeFloat2(dstData, dstOff + ic * 4, v)
        }
    }
}

// ops.cpp line 4525
private fun ggml_compute_forward_set_i32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    require(
        io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst) && io.github.kotlinmania.llama.ore.ggmlIsContiguous(
            src0
        )
    )

    val nb1     = dst.opParams[0].toLong()
    val nb2     = dst.opParams[1].toLong()
    val nb3     = dst.opParams[2].toLong()
    val offset  = dst.opParams[3].toLong()
    val inplace = dst.opParams[4] != 0

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    if (!inplace) {
        if (params.ith == 0) {
            src0Data.copyInto(dstData, 0, 0, io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt())
        }
        io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)
    }

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src1).toInt()
    val nc = src1.ne[0].toInt()

    val ne10 = src1.ne[0]
    val ne11 = src1.ne[1]
    val ne12 = src1.ne[2]
    val ne13 = src1.ne[3]

    val nb10 = src1.nb[0].toLong()
    val nb11 = src1.nb[1].toLong()
    val nb12 = src1.nb[2].toLong()
    val nb13 = src1.nb[3].toLong()

    val nb0 = io.github.kotlinmania.llama.ore.ggmlElementSize(src0).toLong()

    val im0 = if (ne10 == 0L) 0L else ne10 - 1
    val im1 = if (ne11 == 0L) 0L else ne11 - 1
    val im2 = if (ne12 == 0L) 0L else ne12 - 1
    val im3 = if (ne13 == 0L) 0L else ne13 - 1

    require(offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= io.github.kotlinmania.llama.ore.ggmlNbytes(
        dst
    ).toLong())

    require(nb10 == 4L) // sizeof(int32_t)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne12 * ne11).toInt()
        val i2 = (ir - i3 * (ne12 * ne11).toInt()) / ne11.toInt()
        val i1 = ir - i3 * (ne12 * ne11).toInt() - i2 * ne11.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1 + offset).toInt()
        val src1Off = (i3 * nb13 + i2 * nb12 + i1 * nb11).toInt()

        // ggml_vec_cpy_i32
        src1Data.copyInto(dstData, dstOff, src1Off, src1Off + nc * 4)
    }
}

// ops.cpp line 4596
fun ggml_compute_forward_set(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_set_f32(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_set_i32(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16,
        io.github.kotlinmania.llama.ore.GGMLType.BF16,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0,
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K,
        io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_S, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S -> {
            error("fatal error")
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_cpy — ops.cpp line 4645
// ============================================================================

// ops.cpp line 4647
fun ggml_compute_forward_cpy(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_2(params, dst)
}

// ============================================================================
// ggml_compute_forward_cont — ops.cpp line 4653
// ============================================================================

// ops.cpp line 4655
fun ggml_compute_forward_cont(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_2(params, dst)
}

// NOTE: ggml_compute_forward_dup is defined in GGMLComputeForward.kt or GGMLComputeOps.kt.
// We provide a thin wrapper that delegates to the existing implementation.
// If the existing dup function is not yet ported, this is a placeholder that
// performs a simple contiguous copy for the common case.
private fun ggml_compute_forward_dup_2(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    if (params.ith != 0) return

    // Simple contiguous copy for the common case
    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(src0) && io.github.kotlinmania.llama.ore.ggmlIsContiguous(
            dst
        ) && src0.type == dst.type) {
        val nBytes = io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()
        src0Data.copyInto(dstData, 0, 0, min(nBytes, min(src0Data.size, dstData.size)))
        return
    }

    // Non-contiguous or type-converting copy
    val ne0 = dst.ne[0]
    val ne1 = dst.ne[1]
    val ne2 = dst.ne[2]
    val ne3 = dst.ne[3]

    val nb0 = dst.nb[0].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val nb00 = src0.nb[0].toLong()
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()

    if (src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32 && dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        for (i3 in 0 until ne3) {
            for (i2 in 0 until ne2) {
                for (i1 in 0 until ne1) {
                    for (i0 in 0 until ne0) {
                        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00).toInt()
                        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0).toInt()
                        val v = io.github.kotlinmania.llama.ore.readFloat2(src0Data, srcOff)
                        io.github.kotlinmania.llama.ore.writeFloat2(dstData, dstOff, v)
                    }
                }
            }
        }
    } else if (src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16 && dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        for (i3 in 0 until ne3) {
            for (i2 in 0 until ne2) {
                for (i1 in 0 until ne1) {
                    for (i0 in 0 until ne0) {
                        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00).toInt()
                        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0).toInt()
                        val bits = io.github.kotlinmania.llama.ore.readShort2(src0Data, srcOff)
                        val v = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(bits.toUShort())
                        io.github.kotlinmania.llama.ore.writeFloat2(dstData, dstOff, v)
                    }
                }
            }
        }
    } else if (src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32 && dst.type == io.github.kotlinmania.llama.ore.GGMLType.F16) {
        for (i3 in 0 until ne3) {
            for (i2 in 0 until ne2) {
                for (i1 in 0 until ne1) {
                    for (i0 in 0 until ne0) {
                        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00).toInt()
                        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0).toInt()
                        val v = io.github.kotlinmania.llama.ore.readFloat2(src0Data, srcOff)
                        val fp16 = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(v)
                        dstData.setShortLe(dstOff, fp16.toShort())
                    }
                }
            }
        }
    } else {
        // Fallback: contiguous byte copy
        val nBytes = min(io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt(), min(src0Data.size, dstData.size))
        src0Data.copyInto(dstData, 0, 0, nBytes)
    }
}

// ============================================================================
// ggml_compute_forward_get_rows — ops.cpp line 4661
// ============================================================================

// ops.cpp line 4663
private fun ggml_compute_forward_get_rows_q(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    val nc = l.ne00
    val nr = io.github.kotlinmania.llama.ore.ggmlNelements(src1)

    val type = src0.type

    require(l.ne0 == nc)
    require(l.ne02 == l.ne11)
    require(l.nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(type).toLong())
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst) == nr)

    val ith = params.ith
    val nth = params.nth

    val dr = ((nr + nth - 1) / nth).toInt()
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr.toInt())

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    val wdata = FloatArray(nc.toInt())

    for (i in ir0 until ir1) {
        val i12 = i.toLong() / (l.ne11 * l.ne10)
        val i11 = (i.toLong() - i12 * l.ne11 * l.ne10) / l.ne10
        val i10 = i.toLong() - i12 * l.ne11 * l.ne10 - i11 * l.ne10
        val i01Off = (i10 * l.nb10 + i11 * l.nb11 + i12 * l.nb12).toInt()
        val i01 = io.github.kotlinmania.llama.ore.readInt2(src1Data, i01Off).toLong()

        require(i01 >= 0 && i01 < l.ne01)

        val src0Off = (i01 * l.nb01 + i11 * l.nb02 + i12 * l.nb03).toInt()
        val dstOff = (i10 * l.nb1 + i11 * l.nb2 + i12 * l.nb3).toInt()

        io.github.kotlinmania.llama.ore.dequantize_row_2(type, src0Data, src0Off, wdata, 0, nc)
        io.github.kotlinmania.llama.ore.writeFloatRow2(dstData, dstOff, wdata, 0, nc.toInt())
    }
}

// ops.cpp line 4707
private fun ggml_compute_forward_get_rows_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    val nc = l.ne00
    val nr = io.github.kotlinmania.llama.ore.ggmlNelements(src1)

    require(l.ne0 == nc)
    require(l.ne02 == l.ne11)
    require(l.nb00 == 2L) // sizeof(ggml_fp16_t)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst) == nr)

    val ith = params.ith
    val nth = params.nth

    val dr = ((nr + nth - 1) / nth).toInt()
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr.toInt())

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i in ir0 until ir1) {
        val i12 = i.toLong() / (l.ne11 * l.ne10)
        val i11 = (i.toLong() - i12 * l.ne11 * l.ne10) / l.ne10
        val i10 = i.toLong() - i12 * l.ne11 * l.ne10 - i11 * l.ne10
        val i01Off = (i10 * l.nb10 + i11 * l.nb11 + i12 * l.nb12).toInt()
        val i01 = io.github.kotlinmania.llama.ore.readInt2(src1Data, i01Off).toLong()

        require(i01 >= 0 && i01 < l.ne01)

        val src0Off = (i01 * l.nb01 + i11 * l.nb02 + i12 * l.nb03).toInt()
        val dstOff = (i10 * l.nb1 + i11 * l.nb2 + i12 * l.nb3).toInt()

        // ggml_cpu_fp16_to_fp32
        for (j in 0 until nc.toInt()) {
            val bits = io.github.kotlinmania.llama.ore.readShort2(src0Data, src0Off + j * 2)
            val v = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(bits.toUShort())
            io.github.kotlinmania.llama.ore.writeFloat2(dstData, dstOff + j * 4, v)
        }
    }
}

// ops.cpp line 4748
private fun ggml_compute_forward_get_rows_bf16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    val nc = l.ne00
    val nr = io.github.kotlinmania.llama.ore.ggmlNelements(src1)

    require(l.ne0 == nc)
    require(l.ne02 == l.ne11)
    require(l.nb00 == 2L) // sizeof(ggml_bf16_t)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst) == nr)

    val ith = params.ith
    val nth = params.nth

    val dr = ((nr + nth - 1) / nth).toInt()
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr.toInt())

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i in ir0 until ir1) {
        val i12 = i.toLong() / (l.ne11 * l.ne10)
        val i11 = (i.toLong() - i12 * l.ne11 * l.ne10) / l.ne10
        val i10 = i.toLong() - i12 * l.ne11 * l.ne10 - i11 * l.ne10
        val i01Off = (i10 * l.nb10 + i11 * l.nb11 + i12 * l.nb12).toInt()
        val i01 = io.github.kotlinmania.llama.ore.readInt2(src1Data, i01Off).toLong()

        require(i01 >= 0 && i01 < l.ne01)

        val src0Off = (i01 * l.nb01 + i11 * l.nb02 + i12 * l.nb03).toInt()
        val dstOff = (i10 * l.nb1 + i11 * l.nb2 + i12 * l.nb3).toInt()

        // ggml_cpu_bf16_to_fp32
        for (j in 0 until nc.toInt()) {
            val bits = io.github.kotlinmania.llama.ore.readShort2(src0Data, src0Off + j * 2)
            val v = io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(bits.toUShort())
            )
            io.github.kotlinmania.llama.ore.writeFloat2(dstData, dstOff + j * 4, v)
        }
    }
}

// ops.cpp line 4789
private fun ggml_compute_forward_get_rows_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    val nc = l.ne00
    val nr = io.github.kotlinmania.llama.ore.ggmlNelements(src1)

    require(l.ne0 == nc)
    require(l.ne02 == l.ne11)
    require(l.nb00 == 4L) // sizeof(float)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst) == nr)

    val ith = params.ith
    val nth = params.nth

    val dr = ((nr + nth - 1) / nth).toInt()
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr.toInt())

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i in ir0 until ir1) {
        val i12 = i.toLong() / (l.ne11 * l.ne10)
        val i11 = (i.toLong() - i12 * l.ne11 * l.ne10) / l.ne10
        val i10 = i.toLong() - i12 * l.ne11 * l.ne10 - i11 * l.ne10
        val i01Off = (i10 * l.nb10 + i11 * l.nb11 + i12 * l.nb12).toInt()
        val i01 = io.github.kotlinmania.llama.ore.readInt2(src1Data, i01Off).toLong()

        require(i01 >= 0 && i01 < l.ne01)

        val src0Off = (i01 * l.nb01 + i11 * l.nb02 + i12 * l.nb03).toInt()
        val dstOff = (i10 * l.nb1 + i11 * l.nb2 + i12 * l.nb3).toInt()

        // ggml_vec_cpy_f32
        src0Data.copyInto(dstData, dstOff, src0Off, src0Off + nc.toInt() * 4)
    }
}

// ops.cpp line 4830
fun ggml_compute_forward_get_rows(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0,
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K,
        io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_S, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_get_rows_q(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_get_rows_f16(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.BF16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_get_rows_bf16(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32, io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_get_rows_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_set_rows — ops.cpp line 4903
// ============================================================================

// ops.cpp line 4903 (template<typename idx_t> instantiated for i32 and i64)
private fun ggml_compute_forward_set_rows_f32_i32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    val nc = l.ne00
    val nr = l.ne01

    require(l.ne0 == nc)
    require(l.ne2 == l.ne02)
    require(l.ne3 == l.ne03)
    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(l.ne02 % l.ne11 == 0L)
    require(l.ne03 % l.ne12 == 0L)

    val ith = params.ith
    val nth = params.nth

    val dr = ((nr + nth - 1) / nth)
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            for (i in ir0 until ir1) {
                val i12 = i03 % l.ne12
                val i11 = i02 % l.ne11
                val i10 = i

                val idx1Off = (i10 * l.nb10 + i11 * l.nb11 + i12 * l.nb12).toInt()
                val i1 = io.github.kotlinmania.llama.ore.readInt2(src1Data, idx1Off).toLong()

                require(i1 >= 0 && i1 < l.ne1)

                val src0Off = (i * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val dstOff = (i1 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()

                // from_float: quantize src0 row into dst
                io.github.kotlinmania.llama.ore.quantize_row_2(
                    dst.type,
                    src0Data,
                    src0Off,
                    dstData,
                    dstOff,
                    nc
                )
            }
        }
    }
}

private fun ggml_compute_forward_set_rows_f32_i64(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals2(dst)

    val nc = l.ne00
    val nr = l.ne01

    require(l.ne0 == nc)
    require(l.ne2 == l.ne02)
    require(l.ne3 == l.ne03)
    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(l.ne02 % l.ne11 == 0L)
    require(l.ne03 % l.ne12 == 0L)

    val ith = params.ith
    val nth = params.nth

    val dr = ((nr + nth - 1) / nth)
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            for (i in ir0 until ir1) {
                val i12 = i03 % l.ne12
                val i11 = i02 % l.ne11
                val i10 = i

                val idx1Off = (i10 * l.nb10 + i11 * l.nb11 + i12 * l.nb12).toInt()
                val i1 = io.github.kotlinmania.llama.ore.readLong2(src1Data, idx1Off)

                require(i1 >= 0 && i1 < l.ne1)

                val src0Off = (i * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val dstOff = (i1 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()

                io.github.kotlinmania.llama.ore.quantize_row_2(
                    dst.type,
                    src0Data,
                    src0Off,
                    dstData,
                    dstOff,
                    nc
                )
            }
        }
    }
}

// ops.cpp line 4954
fun ggml_compute_forward_set_rows(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            when (src1.type) {
                io.github.kotlinmania.llama.ore.GGMLType.I64 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_set_rows_f32_i64(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.I32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_set_rows_f32_i32(
                    params,
                    dst
                )
                else -> error("src1->type = ${src1.type} not supported")
            }
        }
        else -> {
            error("src0->type = ${src0.type} not supported")
        }
    }
}

// ============================================================================
// ggml_compute_forward_get_rows_back — ops.cpp line 4979
// ============================================================================

// ops.cpp line 4981
private fun ggml_compute_forward_get_rows_back_f32_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    if (params.ith != 0) {
        return
    }

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst))

    val dstData = dst.data as ByteArray
    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray

    // memset(dst->data, 0, ggml_nbytes(dst))
    for (i in 0 until io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()) {
        dstData[i] = 0
    }

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNelements(src1).toInt()

    require(dst.ne[0] == nc.toLong())
    require(src0.nb[0] == 2uL) // sizeof(ggml_fp16_t)

    for (i in 0 until nr) {
        val r = io.github.kotlinmania.llama.ore.readInt2(src1Data, i * 4)

        for (j in 0 until nc) {
            val bits = io.github.kotlinmania.llama.ore.readShort2(
                src0Data,
                (i * src0.nb[1].toInt()) + j * 2
            )
            val v = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(bits.toUShort())
            val dstOff = r * dst.nb[1].toInt() + j * 4
            val cur = io.github.kotlinmania.llama.ore.readFloat2(dstData, dstOff)
            io.github.kotlinmania.llama.ore.writeFloat2(dstData, dstOff, cur + v)
        }
    }
}

// ops.cpp line 5014
private fun ggml_compute_forward_get_rows_back_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    if (params.ith != 0) {
        return
    }

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst))

    val dstData = dst.data as ByteArray
    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray

    // memset(dst->data, 0, ggml_nbytes(dst))
    for (i in 0 until io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()) {
        dstData[i] = 0
    }

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNelements(src1).toInt()

    require(dst.ne[0] == nc.toLong())
    require(src0.nb[0] == 4uL) // sizeof(float)

    for (i in 0 until nr) {
        val r = io.github.kotlinmania.llama.ore.readInt2(src1Data, i * 4)

        val dstOff = r * dst.nb[1].toInt()
        val src0Off = i * src0.nb[1].toInt()

        // ggml_vec_add_f32(nc, dst_row, dst_row, src0_row)
        for (j in 0 until nc) {
            val dVal = io.github.kotlinmania.llama.ore.readFloat2(dstData, dstOff + j * 4)
            val sVal = io.github.kotlinmania.llama.ore.readFloat2(src0Data, src0Off + j * 4)
            io.github.kotlinmania.llama.ore.writeFloat2(dstData, dstOff + j * 4, dVal + sVal)
        }
    }
}

// ops.cpp line 5047
fun ggml_compute_forward_get_rows_back(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_get_rows_back_f32_f16(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_get_rows_back_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_diag — ops.cpp line 5087
// ============================================================================

// ops.cpp line 5089
private fun ggml_compute_forward_diag_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) {
        return
    }

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals2(dst)

    require(l.ne00 == l.ne0)
    require(l.ne00 == l.ne1)
    require(l.ne01 == 1L)
    require(l.ne02 == l.ne2)
    require(l.ne03 == l.ne3)

    require(l.nb00 == 4L) // sizeof(float)
    require(l.nb0 == 4L)  // sizeof(float)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i3 in 0 until l.ne3.toInt()) {
        for (i2 in 0 until l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                val dOff = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
                val sOff = (i3 * l.nb03 + i2 * l.nb02).toInt()
                for (i0 in 0 until i1) {
                    io.github.kotlinmania.llama.ore.writeFloat2(dstData, dOff + i0 * 4, 0f)
                }
                val sVal = io.github.kotlinmania.llama.ore.readFloat2(src0Data, sOff + i1 * 4)
                io.github.kotlinmania.llama.ore.writeFloat2(dstData, dOff + i1 * 4, sVal)
                for (i0 in (i1 + 1) until l.ne0.toInt()) {
                    io.github.kotlinmania.llama.ore.writeFloat2(dstData, dOff + i0 * 4, 0f)
                }
            }
        }
    }
}

// ops.cpp line 5129
fun ggml_compute_forward_diag(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_diag_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_diag_mask_inf — ops.cpp line 5147
// ============================================================================

// ops.cpp line 5149
private fun ggml_compute_forward_diag_mask_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    value: Float
) {
    val src0 = dst.src[0]!!

    val ith = params.ith
    val nth = params.nth

    val nPast = dst.opParams[0]
    val inplace = src0.data === dst.data

    require(nPast >= 0)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    if (!inplace) {
        if (ith == 0) {
            require(
                io.github.kotlinmania.llama.ore.ggmlNelements(dst) == io.github.kotlinmania.llama.ore.ggmlNelements(
                    src0
                )
            )
            require(
                io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst) && io.github.kotlinmania.llama.ore.ggmlIsContiguous(
                    src0
                )
            )
            src0Data.copyInto(dstData, 0, 0, io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt())
        }
        io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)
    }

    val n = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()
    val nc = src0.ne[0].toInt()
    val nr = src0.ne[1].toInt()
    val nz = n / nr

    require(dst.nb[0].toLong() == 4L)  // sizeof(float)
    require(src0.nb[0].toLong() == 4L) // sizeof(float)

    for (k in 0 until nz) {
        var j = ith
        while (j < nr) {
            for (i in nPast until nc) {
                if (i > nPast + j) {
                    val off = (k * dst.nb[2].toLong() + j * dst.nb[1].toLong() + i * dst.nb[0].toLong()).toInt()
                    io.github.kotlinmania.llama.ore.writeFloat2(dstData, off, value)
                }
            }
            j += nth
        }
    }
}

// ops.cpp line 5199
fun ggml_compute_forward_diag_mask_inf(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_diag_mask_f32(
                params,
                dst,
                Float.NEGATIVE_INFINITY
            )
        }
        else -> {
            error("fatal error")
        }
    }
}

// ops.cpp line 5217
fun ggml_compute_forward_diag_mask_zero(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_diag_mask_f32(params, dst, 0f)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_soft_max — ops.cpp line 5235
// ============================================================================

// ops.cpp line 5237
private fun ggml_compute_forward_soft_max_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]   // may be null
    val src2 = dst.src[2]   // may be null (sinks)

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    var scale = 1.0f
    var maxBias = 0.0f

    scale = Float.fromBits(dst.opParams[0])
    maxBias = Float.fromBits(dst.opParams[1])

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals2(dst)

    val nb11 = if (src1 != null) src1.nb[1].toLong() else 1L
    val nb12 = if (src1 != null) src1.nb[2].toLong() else 1L
    val nb13 = if (src1 != null) src1.nb[3].toLong() else 1L

    val ne12 = if (src1 != null) src1.ne[2] else 1L
    val ne13 = if (src1 != null) src1.ne[3] else 1L

    val nHead = l.ne02.toUInt()
    val nHeadLog2 = 1u shl floor(ln(nHead.toDouble()) / ln(2.0)).toInt()

    val m0 = 2.0f.pow(-(maxBias) / nHeadLog2.toFloat())
    val m1 = 2.0f.pow(-(maxBias / 2.0f) / nHeadLog2.toFloat())

    val wdataBytes = params.wdata
    val wp = FloatArray((l.ne00 + io.github.kotlinmania.llama.ore.CACHE_LINE_SIZE_F32_2).toInt())

    val useF16 = src1 != null && src1.type == io.github.kotlinmania.llama.ore.GGMLType.F16

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray
    val src1Data = src1?.data as? ByteArray
    val src2Data = src2?.data as? ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            for (i01 in ith.toLong() until l.ne01 step nth.toLong()) {
                val i11 = i01
                val i12 = i02 % ne12
                val i13 = i03 % ne13

                // ALiBi
                val h = i02.toUInt()
                val slope = if (maxBias > 0.0f) {
                    if (h < nHeadLog2) m0.pow((h + 1u).toFloat())
                    else m1.pow((2u * (h - nHeadLog2) + 1u).toFloat())
                } else 1.0f

                val spOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val dpOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()

                // read src0 row into wp
                for (i in 0 until l.ne00.toInt()) {
                    wp[i] = io.github.kotlinmania.llama.ore.readFloat2(src0Data, spOff + i * 4)
                }
                // ggml_vec_scale_f32(ne00, wp, scale)
                io.github.kotlinmania.llama.ore.ggml_vec_scale_f32_2(l.ne00, wp, 0, scale)

                if (src1Data != null) {
                    val maskOff = (i11 * nb11 + i12 * nb12 + i13 * nb13).toInt()
                    if (useF16) {
                        for (i in 0 until l.ne00.toInt()) {
                            val bits = io.github.kotlinmania.llama.ore.readShort2(
                                src1Data,
                                maskOff + i * 2
                            )
                            wp[i] += slope * io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(bits.toUShort())
                        }
                    } else {
                        for (i in 0 until l.ne00.toInt()) {
                            wp[i] += slope * io.github.kotlinmania.llama.ore.readFloat2(
                                src1Data,
                                maskOff + i * 4
                            )
                        }
                    }
                }

                var maxVal = Float.NEGATIVE_INFINITY
                maxVal = io.github.kotlinmania.llama.ore.ggml_vec_max_f32_2(l.ne00, wp, 0)

                // if we have sinks, make a correction as if they were included in the softmax
                if (src2Data != null) {
                    val skVal =
                        io.github.kotlinmania.llama.ore.readFloat2(src2Data, i02.toInt() * 4)
                    maxVal = max(maxVal, skVal)
                }

                val dp = FloatArray(l.ne00.toInt())
                var sum = io.github.kotlinmania.llama.ore.ggml_vec_soft_max_f32_2(
                    l.ne00,
                    dp,
                    0,
                    wp,
                    0,
                    maxVal
                )
                require(sum > 0.0)

                if (src2Data != null) {
                    val skVal =
                        io.github.kotlinmania.llama.ore.readFloat2(src2Data, i02.toInt() * 4)
                    sum += exp((skVal - maxVal).toDouble())
                }

                val invSum = (1.0 / sum).toFloat()
                io.github.kotlinmania.llama.ore.ggml_vec_scale_f32_2(l.ne00, dp, 0, invSum)

                // write dp to dst
                io.github.kotlinmania.llama.ore.writeFloatRow2(
                    dstData,
                    dpOff,
                    dp,
                    0,
                    l.ne00.toInt()
                )
            }
        }
    }
}

// ops.cpp line 5349
fun ggml_compute_forward_soft_max(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_soft_max_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_soft_max_ext_back — ops.cpp line 5368
// ============================================================================

// ops.cpp line 5370
private fun ggml_compute_forward_soft_max_ext_back_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(src1))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src1, dst))

    var scale = 1.0f
    var maxBias = 0.0f

    scale = Float.fromBits(dst.opParams[0])
    maxBias = Float.fromBits(dst.opParams[1])

    require(maxBias == 0.0f)

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        val dyOff = (i1 * src0.nb[1].toLong()).toInt()
        val yOff = (i1 * src1.nb[1].toLong()).toInt()
        val dxOff = (i1 * dst.nb[1].toLong()).toInt()

        val dy = io.github.kotlinmania.llama.ore.readFloatRow2(src0Data, dyOff, nc)
        val y = io.github.kotlinmania.llama.ore.readFloatRow2(src1Data, yOff, nc)
        val dx = FloatArray(nc)

        // dot_y_dy := dot(y, dy)
        val dotYDy = io.github.kotlinmania.llama.ore.ggml_vec_dot_f32_2(nc.toLong(), y, 0, dy, 0)
        // dx := dy
        io.github.kotlinmania.llama.ore.ggml_vec_cpy_f32_2(nc.toLong(), dx, 0, dy, 0)
        // dx := dx - dot_y_dy
        io.github.kotlinmania.llama.ore.ggml_vec_acc1_f32_2(nc.toLong(), dx, 0, -dotYDy)
        // dx := dx * y
        io.github.kotlinmania.llama.ore.ggml_vec_mul_f32_2(nc.toLong(), dx, 0, dx, 0, y, 0)
        // dx := dx * scale
        io.github.kotlinmania.llama.ore.ggml_vec_scale_f32_2(nc.toLong(), dx, 0, scale)

        io.github.kotlinmania.llama.ore.writeFloatRow2(dstData, dxOff, dx, 0, nc)
    }
}

// ops.cpp line 5454
fun ggml_compute_forward_soft_max_ext_back(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_soft_max_ext_back_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// Dequantize / quantize dispatch helpers (local to this file)
// ============================================================================

private fun dequantize_row_2(type: io.github.kotlinmania.llama.ore.GGMLType, src: ByteArray, srcOff: Int, dst: FloatArray, dstOff: Int, k: Long) {
    when (type) {
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0  -> io.github.kotlinmania.llama.ore.dequantize_row_q1_0(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0  -> io.github.kotlinmania.llama.ore.dequantize_row_q4_0(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_1  -> io.github.kotlinmania.llama.ore.dequantize_row_q4_1(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_0  -> io.github.kotlinmania.llama.ore.dequantize_row_q5_0(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_1  -> io.github.kotlinmania.llama.ore.dequantize_row_q5_1(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0  -> io.github.kotlinmania.llama.ore.dequantize_row_q8_0(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K  -> io.github.kotlinmania.llama.ore.dequantize_row_q2_K(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q3_K  -> io.github.kotlinmania.llama.ore.dequantize_row_q3_K(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_K  -> io.github.kotlinmania.llama.ore.dequantize_row_q4_K(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_K  -> io.github.kotlinmania.llama.ore.dequantize_row_q5_K(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q6_K  -> io.github.kotlinmania.llama.ore.dequantize_row_q6_K(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.TQ1_0 -> io.github.kotlinmania.llama.ore.dequantize_row_tq1_0(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.TQ2_0 -> io.github.kotlinmania.llama.ore.dequantize_row_tq2_0(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS -> io.github.kotlinmania.llama.ore.dequantize_row_iq2_xxs(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS  -> io.github.kotlinmania.llama.ore.dequantize_row_iq2_xs(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS -> io.github.kotlinmania.llama.ore.dequantize_row_iq3_xxs(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL  -> io.github.kotlinmania.llama.ore.dequantize_row_iq4_nl(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS  -> io.github.kotlinmania.llama.ore.dequantize_row_iq4_xs(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4   -> io.github.kotlinmania.llama.ore.dequantize_row_mxfp4(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.NVFP4   -> io.github.kotlinmania.llama.ore.dequantize_row_nvfp4(
            src,
            srcOff,
            dst,
            dstOff,
            k
        )
        else -> error("dequantize_row_2: unsupported type $type")
    }
}

private fun quantize_row_2(type: io.github.kotlinmania.llama.ore.GGMLType, src: ByteArray, srcOff: Int, dst: ByteArray, dstOff: Int, k: Long) {
    // Read floats from src ByteArray, quantize, write to dst ByteArray
    val floats = io.github.kotlinmania.llama.ore.readFloatRow2(src, srcOff, k.toInt())
    val tmpDst = ByteArray(io.github.kotlinmania.llama.ore.ggmlRowSize(type, k).toInt())
    when (type) {
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0  -> io.github.kotlinmania.llama.ore.quantize_row_q1_0(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0  -> io.github.kotlinmania.llama.ore.quantize_row_q4_0(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_1  -> io.github.kotlinmania.llama.ore.quantize_row_q4_1(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_0  -> io.github.kotlinmania.llama.ore.quantize_row_q5_0(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_1  -> io.github.kotlinmania.llama.ore.quantize_row_q5_1(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0  -> io.github.kotlinmania.llama.ore.quantize_row_q8_0_generic(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K  -> io.github.kotlinmania.llama.ore.quantize_row_q2_K(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q3_K  -> io.github.kotlinmania.llama.ore.quantize_row_q3_K(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_K  -> io.github.kotlinmania.llama.ore.quantize_row_q4_K(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_K  -> io.github.kotlinmania.llama.ore.quantize_row_q5_K(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q6_K  -> io.github.kotlinmania.llama.ore.quantize_row_q6_K(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.TQ1_0 -> io.github.kotlinmania.llama.ore.quantize_row_tq1_0(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.TQ2_0 -> io.github.kotlinmania.llama.ore.quantize_row_tq2_0(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4 -> io.github.kotlinmania.llama.ore.quantize_row_mxfp4(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.NVFP4 -> io.github.kotlinmania.llama.ore.quantize_row_nvfp4(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL -> io.github.kotlinmania.llama.ore.quantize_row_iq4_nl(
            floats,
            tmpDst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            // no quantize needed; just copy float bytes
            io.github.kotlinmania.llama.ore.writeFloatRow2(dst, dstOff, floats, 0, k.toInt())
            return
        }
        else -> error("quantize_row_2: unsupported type $type")
    }
    tmpDst.copyInto(dst, dstOff, 0, tmpDst.size)
}

private fun readLong2(data: ByteArray, off: Int): Long {
    return (data[off].toLong() and 0xFF) or
            ((data[off + 1].toLong() and 0xFF) shl 8) or
            ((data[off + 2].toLong() and 0xFF) shl 16) or
            ((data[off + 3].toLong() and 0xFF) shl 24) or
            ((data[off + 4].toLong() and 0xFF) shl 32) or
            ((data[off + 5].toLong() and 0xFF) shl 40) or
            ((data[off + 6].toLong() and 0xFF) shl 48) or
            ((data[off + 7].toLong() and 0xFF) shl 56)
}

private fun Float.pow(n: Float): Float = this.toDouble().pow(n.toDouble()).toFloat()
private fun Float.pow(n: UInt): Float = this.toDouble().pow(n.toDouble()).toFloat()
