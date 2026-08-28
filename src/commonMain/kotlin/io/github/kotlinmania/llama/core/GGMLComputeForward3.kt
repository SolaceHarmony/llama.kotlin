// port-lint: source ggml/src/ggml-cpu/ops.cpp
package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setShortLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setIntLe
import kotlin.math.*

/**
 * Line-by-line transliteration of ops.cpp lines 5472–7528.
 *
 * Covers: CLAMP, ROPE, CONV_TRANSPOSE_1D, IM2COL, IM2COL_BACK,
 * CONV_2D, CONV_3D, CONV_TRANSPOSE_2D, CONV_2D_DW, POOL_1D, POOL_2D,
 * POOL_2D_BACK.
 */

// ============================================================================
// Constants
// ============================================================================

private const val CACHE_LINE_SIZE_F32_3 = 16

// ============================================================================
// ByteArray data access helpers
// ============================================================================

private fun readFloat3(data: ByteArray, off: Int): Float = data.getFloatLe(off)
private fun writeFloat3(data: ByteArray, off: Int, v: Float) = data.setFloatLe(off, v)
private fun readInt3(data: ByteArray, off: Int): Int = data.getIntLe(off)
private fun readShort3(data: ByteArray, off: Int): Short = data.getShortLe(off)
private fun writeShort3(data: ByteArray, off: Int, v: Short) = data.setShortLe(off, v)

// ============================================================================
// Op Locals helpers
// ============================================================================

private data class UnaryOpLocals3(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun unaryOpLocals3(dst: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.UnaryOpLocals3 {
    val src0 = dst.src[0]!!
    return io.github.kotlinmania.llama.ore.UnaryOpLocals3(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

private data class BinaryOpLocals3(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne10: Long, val ne11: Long, val ne12: Long, val ne13: Long,
    val nb10: Long, val nb11: Long, val nb12: Long, val nb13: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun binaryOpLocals3(dst: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.BinaryOpLocals3 {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    return io.github.kotlinmania.llama.ore.BinaryOpLocals3(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne10 = src1.ne[0], ne11 = src1.ne[1], ne12 = src1.ne[2], ne13 = src1.ne[3],
        nb10 = src1.nb[0].toLong(), nb11 = src1.nb[1].toLong(), nb12 = src1.nb[2].toLong(), nb13 = src1.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

// ============================================================================
// ggml_compute_forward_clamp — ops.cpp line 5472
// ============================================================================

private fun ggmlComputeForwardClampF32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    val minVal = Float.fromBits(dst.opParams[0])
    val maxVal = Float.fromBits(dst.opParams[1])

    val ith = params.ith
    val nth = params.nth

    val n = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()
    val nc = src0.ne[0].toInt()

    val nb00 = src0.nb[0].toLong()
    val nb01 = src0.nb[1].toLong()

    val nb0 = dst.nb[0].toLong()
    val nb1 = dst.nb[1].toLong()

    require(nb0 == 4L)
    require(nb00 == 4L)

    val dstData = dst.data as ByteArray
    val src0Data = src0.data as ByteArray

    var j = ith
    while (j < n) {
        val dstOff = (j * nb1).toInt()
        val src0Off = (j * nb01).toInt()

        for (i in 0 until nc) {
            val v = io.github.kotlinmania.llama.ore.readFloat3(src0Data, src0Off + i * 4)
            io.github.kotlinmania.llama.ore.writeFloat3(
                dstData,
                dstOff + i * 4,
                max(min(v, maxVal), minVal)
            )
        }
        j += nth
    }
}

private fun ggmlComputeForwardClampF16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    val minVal = Float.fromBits(dst.opParams[0])
    val maxVal = Float.fromBits(dst.opParams[1])

    val ith = params.ith
    val nth = params.nth

    val n = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()
    val nc = src0.ne[0].toInt()

    val nb00 = src0.nb[0].toLong()
    val nb01 = src0.nb[1].toLong()

    val nb0 = dst.nb[0].toLong()
    val nb1 = dst.nb[1].toLong()

    require(nb0 == 2L)
    require(nb00 == 2L)

    val dstData = dst.data as ByteArray
    val src0Data = src0.data as ByteArray

    var j = ith
    while (j < n) {
        val dstOff = (j * nb1).toInt()
        val src0Off = (j * nb01).toInt()

        for (i in 0 until nc) {
            val bits = io.github.kotlinmania.llama.ore.readShort3(src0Data, src0Off + i * 2)
            val v = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(bits.toUShort())
            val clamped = max(min(v, maxVal), minVal)
            io.github.kotlinmania.llama.ore.writeShort3(
                dstData,
                dstOff + i * 2,
                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(clamped).toShort()
            )
        }
        j += nth
    }
}

fun ggmlComputeForwardClamp(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardClampF32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardClampF16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_rope — ops.cpp line 5601
// ============================================================================

private fun ropeYarnRamp(low: Float, high: Float, i0: Int): Float {
    val y = (i0 / 2 - low) / max(0.001f, high - low)
    return 1.0f - min(1.0f, max(0.0f, y))
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
private fun ropeYarn(
    thetaExtrap: Float, freqScale: Float, corrDims: FloatArray, i0: Long, extFactor: Float, mscale: Float
): Pair<Float, Float> {
    var mscaleLocal = mscale
    val thetaInterp = freqScale * thetaExtrap
    var theta = thetaInterp
    if (extFactor != 0.0f) {
        val rampMix = io.github.kotlinmania.llama.ore.ropeYarnRamp(
            corrDims[0],
            corrDims[1],
            i0.toInt()
        ) * extFactor
        theta = thetaInterp * (1 - rampMix) + thetaExtrap * rampMix
        mscaleLocal *= 1.0f + 0.1f * ln(1.0f / freqScale)
    }
    val cosTheta = cos(theta) * mscaleLocal
    val sinTheta = sin(theta) * mscaleLocal
    return Pair(cosTheta, sinTheta)
}

private fun ggmlRopeCacheInit(
    thetaBase: Float, freqScale: Float, freqFactors: FloatArray?, corrDims: FloatArray,
    ne0: Long, extFactor: Float, mscale: Float,
    cache: FloatArray, sinSign: Float, thetaScale: Float
) {
    var theta = thetaBase
    var i0 = 0L
    while (i0 < ne0) {
        val ff = if (freqFactors != null) freqFactors[(i0 / 2).toInt()] else 1.0f
        val (cosVal, sinVal) = io.github.kotlinmania.llama.ore.ropeYarn(
            theta / ff, freqScale, corrDims, i0, extFactor, mscale
        )
        cache[i0.toInt() + 0] = cosVal
        cache[i0.toInt() + 1] = sinVal * sinSign
        theta *= thetaScale
        i0 += 2
    }
}

private fun ggmlMropeCacheInit(
    thetaBaseT: Float, thetaBaseH: Float, thetaBaseW: Float, thetaBaseE: Float,
    sections: IntArray, isImrope: Boolean, indepSects: Boolean,
    freqScale: Float, freqFactors: FloatArray?, corrDims: FloatArray,
    ne0: Long, extFactor: Float, mscale: Float,
    cache: FloatArray, sinSign: Float, thetaScale: Float
) {
    var thetaT = thetaBaseT
    var thetaH = thetaBaseH
    var thetaW = thetaBaseW
    var thetaE = thetaBaseE
    val sectDims = sections[0] + sections[1] + sections[2] + sections[3]
    val secW = sections[1] + sections[0]
    val secE = sections[2] + secW
    require(sectDims <= ne0.toInt())

    var i0 = 0L
    while (i0 < ne0) {
        val ff = if (freqFactors != null) freqFactors[(i0 / 2).toInt()] else 1.0f

        val sector = ((i0 / 2) % sectDims).toInt()
        if (indepSects) {
            if (sector == 0) {
                thetaT = thetaBaseT
            } else if (sector == sections[0]) {
                thetaH = thetaBaseH
            } else if (sector == secW) {
                thetaW = thetaBaseW
            } else if (sector == secE) {
                thetaE = thetaBaseE
            }
        }

        var theta = thetaT
        if (isImrope) {
            if (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = thetaH
            } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = thetaW
            } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = thetaT
            } else {
                theta = thetaE
            }
        } else {
            if (sector >= sections[0] && sector < secW) {
                theta = thetaH
            } else if (sector >= secW && sector < secW + sections[2]) {
                theta = thetaW
            } else if (sector >= secW + sections[2]) {
                theta = thetaE
            }
        }

        val (cosVal, sinVal) = io.github.kotlinmania.llama.ore.ropeYarn(
            theta / ff, freqScale, corrDims, i0, extFactor, mscale
        )
        cache[i0.toInt() + 0] = cosVal
        cache[i0.toInt() + 1] = sinVal * sinSign

        thetaT *= thetaScale
        thetaW *= thetaScale
        thetaH *= thetaScale
        thetaE *= thetaScale
        i0 += 2
    }
}

private fun rotatePairsF32(
    n: Long, nOffset: Long, cache: FloatArray, srcData: ByteArray, srcOff: Int,
    dstData: ByteArray, dstOff: Int, scale: Int = 2
) {
    var i0 = 0L
    while (i0 < n) {
        val ic = (i0 / scale).toInt()

        val cosTheta = cache[i0.toInt() + 0]
        val sinTheta = cache[i0.toInt() + 1]

        val x0 = io.github.kotlinmania.llama.ore.readFloat3(srcData, srcOff + ic * 4)
        val x1 =
            io.github.kotlinmania.llama.ore.readFloat3(srcData, srcOff + (ic + nOffset.toInt()) * 4)

        io.github.kotlinmania.llama.ore.writeFloat3(
            dstData,
            dstOff + ic * 4,
            x0 * cosTheta - x1 * sinTheta
        )
        io.github.kotlinmania.llama.ore.writeFloat3(
            dstData,
            dstOff + (ic + nOffset.toInt()) * 4,
            x0 * sinTheta + x1 * cosTheta
        )
        i0 += 2
    }
}

private fun rotatePairsF16(
    n: Long, nOffset: Long, cache: FloatArray, srcData: ByteArray, srcOff: Int,
    dstData: ByteArray, dstOff: Int, scale: Int = 2
) {
    var i0 = 0L
    while (i0 < n) {
        val ic = (i0 / scale).toInt()

        val cosTheta = cache[i0.toInt() + 0]
        val sinTheta = cache[i0.toInt() + 1]

        val x0 = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
            io.github.kotlinmania.llama.ore.readShort3(
                srcData,
                srcOff + ic * 2
            ).toUShort()
        )
        val x1 = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
            io.github.kotlinmania.llama.ore.readShort3(
                srcData,
                srcOff + (ic + nOffset.toInt()) * 2
            ).toUShort()
        )

        io.github.kotlinmania.llama.ore.writeShort3(
            dstData,
            dstOff + ic * 2,
            io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(x0 * cosTheta - x1 * sinTheta).toShort()
        )
        io.github.kotlinmania.llama.ore.writeShort3(
            dstData,
            dstOff + (ic + nOffset.toInt()) * 2,
            io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(x0 * sinTheta + x1 * cosTheta).toShort()
        )
        i0 += 2
    }
}

fun ggmlComputeForwardRopeFltF32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    forward: Boolean
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    val src2 = dst.src.getOrNull(2)

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.I32)

    val nDims = dst.opParams[1]
    val mode = dst.opParams[2]
    val nCtxOrig = dst.opParams[4]

    val freqBase = Float.fromBits(dst.opParams[5])
    val freqScale = Float.fromBits(dst.opParams[6])
    val extFactor = Float.fromBits(dst.opParams[7])
    val attnFactor = Float.fromBits(dst.opParams[8])
    val betaFast = Float.fromBits(dst.opParams[9])
    val betaSlow = Float.fromBits(dst.opParams[10])
    val sections = intArrayOf(dst.opParams[11], dst.opParams[12], dst.opParams[13], dst.opParams[14])

    val u = io.github.kotlinmania.llama.ore.unaryOpLocals3(dst)
    val ne0 = u.ne0; val ne1 = u.ne1; val ne2 = u.ne2; val ne3 = u.ne3
    val nb0 = u.nb0; val nb1 = u.nb1; val nb2 = u.nb2; val nb3 = u.nb3
    val nb00 = u.nb00; val nb01 = u.nb01; val nb02 = u.nb02; val nb03 = u.nb03

    require(nb0 == nb00)
    require(nb0 == 4L) // sizeof(float)

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt()

    require(nDims <= ne0.toInt())
    require(nDims % 2 == 0)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    var ir = 0

    val thetaScale = freqBase.pow(-2.0f / nDims)

    val corrDims = io.github.kotlinmania.llama.ore.ggmlRopeYarnCorrDims(
        nDims,
        nCtxOrig,
        freqBase,
        betaFast,
        betaSlow
    )

    val isImrope = mode == io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_IMROPE
    val mropeUsed = (mode and io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_MROPE) != 0
    val isVision = mode == io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_VISION

    if (mropeUsed) {
        require(sections[0] > 0 || sections[1] > 0 || sections[2] > 0)
    }

    if (isVision) {
        require(nDims.toLong() == ne0 / 2)
    }

    val freqFactors: FloatArray? = if (src2 != null) {
        require(src2.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
        require(src2.ne[0] >= nDims / 2)
        val src2Data = src2.data as ByteArray
        FloatArray(nDims / 2) { i ->
            io.github.kotlinmania.llama.ore.readFloat3(
                src2Data,
                i * 4
            )
        }
    } else null

    val sinSign = if (forward) 1.0f else -1.0f

    val posData = src1.data as ByteArray

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var lastI2 = -1L

    val cache = FloatArray((ne0 + io.github.kotlinmania.llama.ore.CACHE_LINE_SIZE_F32_3).toInt())

    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            for (i1 in 0 until ne1) {
                if (ir++ < ir0) continue
                if (ir > ir1) break

                if (lastI2 != i2) {
                    if (!mropeUsed) {
                        val p = io.github.kotlinmania.llama.ore.readInt3(posData, i2.toInt() * 4)
                        io.github.kotlinmania.llama.ore.ggmlRopeCacheInit(
                            p.toFloat(), freqScale, freqFactors, corrDims,
                            ne0, extFactor, attnFactor, cache, sinSign, thetaScale
                        )
                    } else {
                        val pT = io.github.kotlinmania.llama.ore.readInt3(posData, i2.toInt() * 4)
                        val pH = io.github.kotlinmania.llama.ore.readInt3(
                            posData,
                            (i2 + ne2).toInt() * 4
                        )
                        val pW = io.github.kotlinmania.llama.ore.readInt3(
                            posData,
                            (i2 + ne2 * 2).toInt() * 4
                        )
                        val pE = io.github.kotlinmania.llama.ore.readInt3(
                            posData,
                            (i2 + ne2 * 3).toInt() * 4
                        )
                        io.github.kotlinmania.llama.ore.ggmlMropeCacheInit(
                            pT.toFloat(), pH.toFloat(), pW.toFloat(), pE.toFloat(),
                            sections, isImrope, isVision,
                            freqScale, freqFactors, corrDims, ne0, extFactor, attnFactor,
                            cache, sinSign, thetaScale
                        )
                    }
                    lastI2 = i2
                }

                val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()
                val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()

                when (mode) {
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_NORMAL ->
                        io.github.kotlinmania.llama.ore.rotatePairsF32(
                            nDims.toLong(),
                            1L,
                            cache,
                            src0Data,
                            srcOff,
                            dstData,
                            dstOff,
                            1
                        )
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_NEOX, io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_MROPE, io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_IMROPE ->
                        io.github.kotlinmania.llama.ore.rotatePairsF32(
                            nDims.toLong(),
                            (nDims / 2).toLong(),
                            cache,
                            src0Data,
                            srcOff,
                            dstData,
                            dstOff
                        )
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_VISION ->
                        io.github.kotlinmania.llama.ore.rotatePairsF32(
                            ne0,
                            nDims.toLong(),
                            cache,
                            src0Data,
                            srcOff,
                            dstData,
                            dstOff
                        )
                    else -> error("rope type not supported")
                }

                if (!isVision) {
                    var i0 = nDims.toLong()
                    while (i0 < ne0) {
                        val srcI0Off = (i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00).toInt()
                        val dstI0Off = (i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0).toInt()
                        io.github.kotlinmania.llama.ore.writeFloat3(
                            dstData,
                            dstI0Off,
                            io.github.kotlinmania.llama.ore.readFloat3(src0Data, srcI0Off)
                        )
                        io.github.kotlinmania.llama.ore.writeFloat3(
                            dstData,
                            dstI0Off + 4,
                            io.github.kotlinmania.llama.ore.readFloat3(src0Data, srcI0Off + 4)
                        )
                        i0 += 2
                    }
                }
            }
        }
    }
}

fun ggmlComputeForwardRopeFltF16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    forward: Boolean
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    val src2 = dst.src.getOrNull(2)

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.I32)

    val nDims = dst.opParams[1]
    val mode = dst.opParams[2]
    val nCtxOrig = dst.opParams[4]

    val freqBase = Float.fromBits(dst.opParams[5])
    val freqScale = Float.fromBits(dst.opParams[6])
    val extFactor = Float.fromBits(dst.opParams[7])
    val attnFactor = Float.fromBits(dst.opParams[8])
    val betaFast = Float.fromBits(dst.opParams[9])
    val betaSlow = Float.fromBits(dst.opParams[10])
    val sections = intArrayOf(dst.opParams[11], dst.opParams[12], dst.opParams[13], dst.opParams[14])

    val u = io.github.kotlinmania.llama.ore.unaryOpLocals3(dst)
    val ne0 = u.ne0; val ne1 = u.ne1; val ne2 = u.ne2; val ne3 = u.ne3
    val nb0 = u.nb0; val nb1 = u.nb1; val nb2 = u.nb2; val nb3 = u.nb3
    val nb00 = u.nb00; val nb01 = u.nb01; val nb02 = u.nb02; val nb03 = u.nb03

    require(nb0 == nb00)
    require(nb0 == 2L) // sizeof(ggml_fp16_t)

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt()

    require(nDims <= ne0.toInt())
    require(nDims % 2 == 0)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    var ir = 0

    val thetaScale = freqBase.pow(-2.0f / nDims)

    val corrDims = io.github.kotlinmania.llama.ore.ggmlRopeYarnCorrDims(
        nDims,
        nCtxOrig,
        freqBase,
        betaFast,
        betaSlow
    )

    val isImrope = mode == io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_IMROPE
    val mropeUsed = (mode and io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_MROPE) != 0
    val isVision = mode == io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_VISION

    if (mropeUsed) {
        require(sections[0] > 0 || sections[1] > 0 || sections[2] > 0)
    }

    if (isVision) {
        require(nDims.toLong() == ne0 / 2)
    }

    val freqFactors: FloatArray? = if (src2 != null) {
        require(src2.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
        require(src2.ne[0] >= nDims / 2)
        val src2Data = src2.data as ByteArray
        FloatArray(nDims / 2) { i ->
            io.github.kotlinmania.llama.ore.readFloat3(
                src2Data,
                i * 4
            )
        }
    } else null

    val sinSign = if (forward) 1.0f else -1.0f

    val posData = src1.data as ByteArray

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var lastI2 = -1L

    val cache = FloatArray((ne0 + io.github.kotlinmania.llama.ore.CACHE_LINE_SIZE_F32_3).toInt())

    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            for (i1 in 0 until ne1) {
                if (ir++ < ir0) continue
                if (ir > ir1) break

                if (lastI2 != i2) {
                    if (!mropeUsed) {
                        val p = io.github.kotlinmania.llama.ore.readInt3(posData, i2.toInt() * 4)
                        io.github.kotlinmania.llama.ore.ggmlRopeCacheInit(
                            p.toFloat(), freqScale, freqFactors, corrDims,
                            ne0, extFactor, attnFactor, cache, sinSign, thetaScale
                        )
                    } else {
                        val pT = io.github.kotlinmania.llama.ore.readInt3(posData, i2.toInt() * 4)
                        val pH = io.github.kotlinmania.llama.ore.readInt3(
                            posData,
                            (i2 + ne2).toInt() * 4
                        )
                        val pW = io.github.kotlinmania.llama.ore.readInt3(
                            posData,
                            (i2 + ne2 * 2).toInt() * 4
                        )
                        val pE = io.github.kotlinmania.llama.ore.readInt3(
                            posData,
                            (i2 + ne2 * 3).toInt() * 4
                        )
                        io.github.kotlinmania.llama.ore.ggmlMropeCacheInit(
                            pT.toFloat(), pH.toFloat(), pW.toFloat(), pE.toFloat(),
                            sections, isImrope, isVision,
                            freqScale, freqFactors, corrDims, ne0, extFactor, attnFactor,
                            cache, sinSign, thetaScale
                        )
                    }
                    lastI2 = i2
                }

                val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()
                val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()

                when (mode) {
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_NORMAL ->
                        io.github.kotlinmania.llama.ore.rotatePairsF16(
                            nDims.toLong(),
                            1L,
                            cache,
                            src0Data,
                            srcOff,
                            dstData,
                            dstOff,
                            1
                        )
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_NEOX, io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_MROPE, io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_IMROPE ->
                        io.github.kotlinmania.llama.ore.rotatePairsF16(
                            nDims.toLong(),
                            (nDims / 2).toLong(),
                            cache,
                            src0Data,
                            srcOff,
                            dstData,
                            dstOff
                        )
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_VISION ->
                        io.github.kotlinmania.llama.ore.rotatePairsF16(
                            ne0,
                            nDims.toLong(),
                            cache,
                            src0Data,
                            srcOff,
                            dstData,
                            dstOff
                        )
                    else -> error("rope type not supported")
                }

                if (!isVision) {
                    var i0 = nDims.toLong()
                    while (i0 < ne0) {
                        val srcI0Off = (i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00).toInt()
                        val dstI0Off = (i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0).toInt()
                        io.github.kotlinmania.llama.ore.writeShort3(
                            dstData,
                            dstI0Off,
                            io.github.kotlinmania.llama.ore.readShort3(src0Data, srcI0Off)
                        )
                        io.github.kotlinmania.llama.ore.writeShort3(
                            dstData,
                            dstI0Off + 2,
                            io.github.kotlinmania.llama.ore.readShort3(src0Data, srcI0Off + 2)
                        )
                        i0 += 2
                    }
                }
            }
        }
    }
}

fun ggmlComputeForwardRope(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardRopeFltF16(
            params,
            dst,
            true
        )
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardRopeFltF32(
            params,
            dst,
            true
        )
        else -> error("fatal error")
    }
}

// ggml_compute_forward_rope_back

fun ggmlComputeForwardRopeBack(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardRopeFltF16(
            params,
            dst,
            false
        )
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardRopeFltF32(
            params,
            dst,
            false
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_conv_transpose_1d — ops.cpp line 5927
// ============================================================================

private fun ggmlVecDotF16(n: Long, src0: ByteArray, src0Off: Int, src1: ByteArray, src1Off: Int): Float {
    var v = 0.0f
    for (i in 0 until n.toInt()) {
        val a = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
            io.github.kotlinmania.llama.ore.readShort3(
                src0,
                src0Off + i * 2
            ).toUShort()
        )
        val b = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
            io.github.kotlinmania.llama.ore.readShort3(
                src1,
                src1Off + i * 2
            ).toUShort()
        )
        v += a * b
    }
    return v
}

private fun ggmlVecDotF32Simple(n: Long, src0: FloatArray, src0Off: Int, src1: FloatArray, src1Off: Int): Float {
    var v = 0.0f
    for (i in 0 until n.toInt()) {
        v += src0[src0Off + i] * src1[src1Off + i]
    }
    return v
}

private fun ggmlComputeForwardConvTranspose1dF16F32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01; val ne02 = b.ne02
    val nb00 = b.nb00; val nb01 = b.nb01; val nb02 = b.nb02
    val ne10 = b.ne10; val ne11 = b.ne11
    val nb10 = b.nb10; val nb11 = b.nb11
    val ne1 = b.ne1
    val nb1 = b.nb1

    val ith = params.ith
    val nth = params.nth

    val nk = (ne00 * ne01 * ne02).toInt()

    require(nb00 == 2L) // sizeof(ggml_fp16_t)
    require(nb10 == 4L) // sizeof(float)

    val wdata = params.wdata as ByteArray

    if (ith == 0) {
        // zero wdata
        for (i in 0 until (params.wsize.toInt())) {
            wdata[i] = 0
        }

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        val src0Data = src0.data as ByteArray
        for (i02 in 0 until ne02.toInt()) {
            for (i01 in 0 until ne01.toInt()) {
                val srcOff = (i02 * nb02 + i01 * nb01).toInt()
                val dstOff = (i01 * ne00 * ne02).toInt() * 2
                for (i00 in 0 until ne00.toInt()) {
                    val srcVal =
                        io.github.kotlinmania.llama.ore.readShort3(src0Data, srcOff + i00 * 2)
                    io.github.kotlinmania.llama.ore.writeShort3(
                        wdata,
                        dstOff + (i00 * ne02.toInt() + i02) * 2,
                        srcVal
                    )
                }
            }
        }

        // permute source data (src1) from (L x Cin) to (Cin x L)
        val src1Data = src1.data as ByteArray
        val wdataOff = nk * 2
        for (i11 in 0 until ne11.toInt()) {
            val srcRowOff = (i11 * nb11).toInt()
            for (i10 in 0 until ne10.toInt()) {
                val srcVal =
                    io.github.kotlinmania.llama.ore.readFloat3(src1Data, srcRowOff + i10 * 4)
                val fp16 = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(srcVal).toShort()
                io.github.kotlinmania.llama.ore.writeShort3(
                    wdata,
                    wdataOff + (i10 * ne11.toInt() + i11) * 2,
                    fp16
                )
            }
        }

        // zero dst
        val dstData = dst.data as ByteArray
        for (i in 0 until io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()) {
            dstData[i] = 0
        }
    }
    io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)

    val s0 = dst.opParams[0]

    val nr = ne1.toInt()
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    val dstData = dst.data as ByteArray
    val wdataSrcOff = nk * 2

    for (i1 in ir0 until ir1) {
        val dstRowOff = (i1 * nb1).toInt()
        val wdataKernelOff = i1 * ne02.toInt() * ne00.toInt() * 2
        for (i10 in 0 until ne10.toInt()) {
            val i1n = i10 * ne11.toInt()
            for (i00 in 0 until ne00.toInt()) {
                val v = io.github.kotlinmania.llama.ore.ggmlVecDotF16(
                    ne02,
                    wdata, wdataSrcOff + i1n * 2,
                    wdata, wdataKernelOff + i00 * ne02.toInt() * 2
                )
                val idx = (i10 * s0 + i00) * 4
                val prev = io.github.kotlinmania.llama.ore.readFloat3(dstData, dstRowOff + idx)
                io.github.kotlinmania.llama.ore.writeFloat3(dstData, dstRowOff + idx, prev + v)
            }
        }
    }
}

private fun ggmlComputeForwardConvTranspose1dF32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01; val ne02 = b.ne02
    val nb01 = b.nb01; val nb02 = b.nb02
    val ne10 = b.ne10; val ne11 = b.ne11
    val nb11 = b.nb11
    val ne1 = b.ne1
    val nb1 = b.nb1

    val ith = params.ith
    val nth = params.nth

    val nk = (ne00 * ne01 * ne02).toInt()

    require(b.nb00 == 4L) // sizeof(float)
    require(b.nb10 == 4L) // sizeof(float)

    // Use FloatArrays for wdata since all values are f32
    val wdataBytes = params.wdata as ByteArray

    if (ith == 0) {
        for (i in 0 until params.wsize.toInt()) {
            wdataBytes[i] = 0
        }

        // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
        val src0Data = src0.data as ByteArray
        for (i02 in 0 until ne02.toInt()) {
            for (i01 in 0 until ne01.toInt()) {
                val srcOff = (i02 * nb02 + i01 * nb01).toInt()
                val dstOff = (i01 * ne00 * ne02).toInt() * 4
                for (i00 in 0 until ne00.toInt()) {
                    val srcVal =
                        io.github.kotlinmania.llama.ore.readFloat3(src0Data, srcOff + i00 * 4)
                    io.github.kotlinmania.llama.ore.writeFloat3(
                        wdataBytes,
                        dstOff + (i00 * ne02.toInt() + i02) * 4,
                        srcVal
                    )
                }
            }
        }

        // permute source data (src1)
        val src1Data = src1.data as ByteArray
        val wdataOff = nk * 4
        for (i11 in 0 until ne11.toInt()) {
            val srcRowOff = (i11 * nb11).toInt()
            for (i10 in 0 until ne10.toInt()) {
                val srcVal =
                    io.github.kotlinmania.llama.ore.readFloat3(src1Data, srcRowOff + i10 * 4)
                io.github.kotlinmania.llama.ore.writeFloat3(
                    wdataBytes,
                    wdataOff + (i10 * ne11.toInt() + i11) * 4,
                    srcVal
                )
            }
        }

        // zero dst
        val dstData = dst.data as ByteArray
        for (i in 0 until io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()) {
            dstData[i] = 0
        }
    }
    io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)

    val s0 = dst.opParams[0]

    val nr = ne1.toInt()
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    val dstData = dst.data as ByteArray
    val wdataSrcOff = nk * 4

    for (i1 in ir0 until ir1) {
        val dstRowOff = (i1 * nb1).toInt()
        val wdataKernelOff = i1 * ne02.toInt() * ne00.toInt() * 4
        for (i10 in 0 until ne10.toInt()) {
            val i1n = i10 * ne11.toInt()
            for (i00 in 0 until ne00.toInt()) {
                var v = 0.0f
                for (k in 0 until ne02.toInt()) {
                    val a = io.github.kotlinmania.llama.ore.readFloat3(
                        wdataBytes,
                        wdataSrcOff + (i1n + k) * 4
                    )
                    val bVal = io.github.kotlinmania.llama.ore.readFloat3(
                        wdataBytes,
                        wdataKernelOff + (i00 * ne02.toInt() + k) * 4
                    )
                    v += a * bVal
                }
                val idx = (i10 * s0 + i00) * 4
                val prev = io.github.kotlinmania.llama.ore.readFloat3(dstData, dstRowOff + idx)
                io.github.kotlinmania.llama.ore.writeFloat3(dstData, dstRowOff + idx, prev + v)
            }
        }
    }
}

fun ggmlComputeForwardConvTranspose1d(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardConvTranspose1dF16F32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardConvTranspose1dF32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_im2col — ops.cpp line 6127
// ============================================================================

private fun ggmlComputeForwardIm2colF32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01
    val ne10 = b.ne10; val ne11 = b.ne11; val ne12 = b.ne12; val ne13 = b.ne13
    val nb10 = b.nb10; val nb11 = b.nb11; val nb12 = b.nb12; val nb13 = b.nb13
    val ne1 = b.ne1; val ne2 = b.ne2

    val s0 = dst.opParams[0]
    val s1 = dst.opParams[1]
    val p0 = dst.opParams[2]
    val p1 = dst.opParams[3]
    val d0 = dst.opParams[4]
    val d1 = dst.opParams[5]
    val is2D = dst.opParams[6] == 1

    val ith = params.ith
    val nth = params.nth

    val N = if (is2D) ne13 else ne12
    val IC = if (is2D) ne12 else ne11
    val IH = if (is2D) ne11 else 1L
    val IW = ne10

    val KH = if (is2D) ne01 else 1L
    val KW = ne00

    val OH = if (is2D) ne2 else 1L
    val OW = ne1

    val ofs0 = if (is2D) nb13 else nb12
    val ofs1 = if (is2D) nb12 else nb11

    require(nb10 == 4L)

    val wdata = dst.data as ByteArray
    val src1Data = src1.data as ByteArray

    for (inn in 0 until N) {
        for (ioh in 0 until OH) {
            for (iow in 0 until OW) {
                var iic = ith.toLong()
                while (iic < IC) {
                    val dstBase = ((inn * OH * OW + ioh * OW + iow) * (IC * KH * KW)).toInt()
                    val srcBase = (inn * ofs0 + iic * ofs1).toInt()

                    for (ikh in 0 until KH) {
                        for (ikw in 0 until KW) {
                            val iiw = iow * s0 + ikw * d0 - p0
                            val iih = ioh * s1 + ikh * d1 - p1

                            val dstIdx = (iic * (KH * KW) + ikh * KW + ikw).toInt()

                            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                io.github.kotlinmania.llama.ore.writeFloat3(
                                    wdata,
                                    (dstBase + dstIdx) * 4,
                                    0.0f
                                )
                            } else {
                                val srcVal = io.github.kotlinmania.llama.ore.readFloat3(
                                    src1Data,
                                    srcBase + (iih * IW + iiw).toInt() * 4
                                )
                                io.github.kotlinmania.llama.ore.writeFloat3(
                                    wdata,
                                    (dstBase + dstIdx) * 4,
                                    srcVal
                                )
                            }
                        }
                    }
                    iic += nth
                }
            }
        }
    }
}

private fun ggmlComputeForwardIm2colF16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F16 || src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F16)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01
    val nb00 = b.nb00
    val ne10 = b.ne10; val ne11 = b.ne11; val ne12 = b.ne12; val ne13 = b.ne13
    val nb10 = b.nb10; val nb11 = b.nb11; val nb12 = b.nb12; val nb13 = b.nb13
    val ne1 = b.ne1; val ne2 = b.ne2

    val s0 = dst.opParams[0]
    val s1 = dst.opParams[1]
    val p0 = dst.opParams[2]
    val p1 = dst.opParams[3]
    val d0 = dst.opParams[4]
    val d1 = dst.opParams[5]
    val is2D = dst.opParams[6] == 1

    val ith = params.ith
    val nth = params.nth

    val N = if (is2D) ne13 else ne12
    val IC = if (is2D) ne12 else ne11
    val IH = if (is2D) ne11 else 1L
    val IW = ne10

    val KH = if (is2D) ne01 else 1L
    val KW = ne00

    val OH = if (is2D) ne2 else 1L
    val OW = ne1

    val ofs0 = if (is2D) nb13 else nb12
    val ofs1 = if (is2D) nb12 else nb11

    require(nb00 == 2L)
    require(nb10 == io.github.kotlinmania.llama.ore.ggmlTypeSize(src1.type).toLong())

    val wdata = dst.data as ByteArray
    val src1Data = src1.data as ByteArray
    val isSrc1F32 = src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32

    for (inn in 0 until N) {
        for (ioh in 0 until OH) {
            for (iow in 0 until OW) {
                var iic = ith.toLong()
                while (iic < IC) {
                    val dstBase = ((inn * OH * OW + ioh * OW + iow) * (IC * KH * KW)).toInt()
                    val srcBase = (inn * ofs0 + iic * ofs1).toInt()

                    for (ikh in 0 until KH) {
                        for (ikw in 0 until KW) {
                            val iiw = iow * s0 + ikw * d0 - p0
                            val iih = ioh * s1 + ikh * d1 - p1

                            val dstIdx = (iic * (KH * KW) + ikh * KW + ikw).toInt()

                            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                io.github.kotlinmania.llama.ore.writeShort3(
                                    wdata,
                                    (dstBase + dstIdx) * 2,
                                    0
                                )
                            } else {
                                if (isSrc1F32) {
                                    val srcVal = io.github.kotlinmania.llama.ore.readFloat3(
                                        src1Data,
                                        srcBase + (iih * IW + iiw).toInt() * 4
                                    )
                                    io.github.kotlinmania.llama.ore.writeShort3(
                                        wdata,
                                        (dstBase + dstIdx) * 2,
                                        io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(srcVal)
                                            .toShort()
                                    )
                                } else {
                                    val srcVal = io.github.kotlinmania.llama.ore.readShort3(
                                        src1Data,
                                        srcBase + (iih * IW + iiw).toInt() * 2
                                    )
                                    io.github.kotlinmania.llama.ore.writeShort3(
                                        wdata,
                                        (dstBase + dstIdx) * 2,
                                        srcVal
                                    )
                                }
                            }
                        }
                    }
                    iic += nth
                }
            }
        }
    }
}

fun ggmlComputeForwardIm2col(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    when (dst.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardIm2colF16(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardIm2colF32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_im2col_back_f32 — ops.cpp line 6308
// ============================================================================

fun ggmlComputeForwardIm2colBackF32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne0 = b.ne0; val ne1 = b.ne1; val ne2 = b.ne2; val ne3 = b.ne3
    val nb0 = b.nb0; val nb1 = b.nb1; val nb2 = b.nb2; val nb3 = b.nb3
    val ne01 = b.ne01; val ne02 = b.ne02
    val ne10 = b.ne10; val ne11 = b.ne11

    val s0 = dst.opParams[0]
    val s1 = dst.opParams[1]
    val p0 = dst.opParams[2]
    val p1 = dst.opParams[3]
    val d0 = dst.opParams[4]
    val d1 = dst.opParams[5]
    val is2D = dst.opParams[6] == 1

    val ith = params.ith
    val nth = params.nth

    val N = if (is2D) ne3 else ne2
    val IC = if (is2D) ne2 else ne1
    val IH = if (is2D) ne1 else 1L
    val IW = ne0

    val KH = if (is2D) ne11 else 1L
    val KW = ne10

    val OH = if (is2D) ne02 else 1L
    val OW = ne01

    val ofs0 = if (is2D) nb3 else nb2
    val ofs1 = if (is2D) nb2 else nb1

    require(nb0 == 4L)

    val wdata = dst.data as ByteArray
    val src0Data = src0.data as ByteArray

    for (inn in 0 until N) {
        var iic = ith.toLong()
        while (iic < IC) {
            for (iih in 0 until IH) {
                for (iiw in 0 until IW) {
                    var grad = 0.0f
                    for (ikh in 0 until KH) {
                        for (ikw in 0 until KW) {
                            val tmpw = (iiw + p0 - ikw * d0)
                            if (tmpw % s0 != 0L) continue
                            val iow = tmpw / s0

                            val ioh: Long
                            if (is2D) {
                                val tmph = iih + p1 - ikh * d1
                                if (tmph % s1 != 0L) continue
                                ioh = tmph / s1
                            } else {
                                ioh = 0L
                            }

                            if (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) continue

                            val gradInOff = ((inn * OH * OW + ioh * OW + iow) * (IC * KH * KW)).toInt()
                            val gradIdx = (iic * (KH * KW) + ikh * KW + ikw).toInt()
                            grad += io.github.kotlinmania.llama.ore.readFloat3(
                                src0Data,
                                (gradInOff + gradIdx) * 4
                            )
                        }
                    }
                    val dstOff = (inn * ofs0 + iic * ofs1).toInt()
                    io.github.kotlinmania.llama.ore.writeFloat3(
                        wdata,
                        dstOff + (iih * IW + iiw).toInt() * 4,
                        grad
                    )
                }
            }
            iic += nth
        }
    }
}

// ============================================================================
// ggml_compute_forward_im2col_3d — ops.cpp line 6404
// ============================================================================

private fun ggmlComputeForwardIm2col3dF16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F16)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01; val ne02 = b.ne02; val ne03 = b.ne03
    val ne10 = b.ne10; val ne11 = b.ne11; val ne12 = b.ne12; val ne13 = b.ne13
    val nb10 = b.nb10; val nb11 = b.nb11; val nb12 = b.nb12; val nb13 = b.nb13
    val ne1 = b.ne1; val ne2 = b.ne2; val ne3 = b.ne3

    val s0 = dst.opParams[0]; val s1 = dst.opParams[1]; val s2 = dst.opParams[2]
    val p0 = dst.opParams[3]; val p1 = dst.opParams[4]; val p2 = dst.opParams[5]
    val d0 = dst.opParams[6]; val d1 = dst.opParams[7]; val d2 = dst.opParams[8]
    val IC = dst.opParams[9].toLong()

    val ith = params.ith
    val nth = params.nth

    val N = ne13 / IC
    val ID = ne12
    val IH = ne11
    val IW = ne10

    val KD = ne02
    val KH = ne01
    val KW = ne00

    val OD = ne3 / N
    val OH = ne2
    val OW = ne1
    val OH_OW = OH * OW
    val KD_KH_KW = KD * KH * KW
    val KH_KW = KH * KW
    val IC_KD_KH_KW = IC * KD * KH * KW

    require(nb10 == 4L)

    val wdata = dst.data as ByteArray
    val src1Data = src1.data as ByteArray

    for (inn in 0 until N) {
        for (iod in 0 until OD) {
            for (ioh in 0 until OH) {
                for (iow in 0 until OW) {
                    var iic = ith.toLong()
                    while (iic < IC) {
                        val dstBase = ((inn * OD * OH_OW + iod * OH_OW + ioh * OW + iow) * IC_KD_KH_KW).toInt()
                        val srcDataOff = ((inn * IC + iic) * nb13).toInt()

                        for (ikd in 0 until KD) {
                            for (ikh in 0 until KH) {
                                for (ikw in 0 until KW) {
                                    val iiw = iow * s0 + ikw * d0 - p0
                                    val iih = ioh * s1 + ikh * d1 - p1
                                    val iid = iod * s2 + ikd * d2 - p2

                                    val dstIdx = (iic * KD_KH_KW + ikd * KH_KW + ikh * KW + ikw).toInt()

                                    if (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                        io.github.kotlinmania.llama.ore.writeShort3(
                                            wdata,
                                            (dstBase + dstIdx) * 2,
                                            0
                                        )
                                    } else {
                                        val sOff = srcDataOff + (iid * nb12 + iih * nb11 + iiw * nb10).toInt()
                                        val srcVal = io.github.kotlinmania.llama.ore.readFloat3(
                                            src1Data,
                                            sOff
                                        )
                                        io.github.kotlinmania.llama.ore.writeShort3(
                                            wdata,
                                            (dstBase + dstIdx) * 2,
                                            io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(srcVal)
                                                .toShort()
                                        )
                                    }
                                }
                            }
                        }
                        iic += nth
                    }
                }
            }
        }
    }
}

private fun ggmlComputeForwardIm2col3dF32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01; val ne02 = b.ne02; val ne03 = b.ne03
    val ne10 = b.ne10; val ne11 = b.ne11; val ne12 = b.ne12; val ne13 = b.ne13
    val nb10 = b.nb10; val nb11 = b.nb11; val nb12 = b.nb12; val nb13 = b.nb13
    val ne1 = b.ne1; val ne2 = b.ne2; val ne3 = b.ne3

    val s0 = dst.opParams[0]; val s1 = dst.opParams[1]; val s2 = dst.opParams[2]
    val p0 = dst.opParams[3]; val p1 = dst.opParams[4]; val p2 = dst.opParams[5]
    val d0 = dst.opParams[6]; val d1 = dst.opParams[7]; val d2 = dst.opParams[8]
    val IC = dst.opParams[9].toLong()

    val ith = params.ith
    val nth = params.nth

    val N = ne13 / IC
    val ID = ne12
    val IH = ne11
    val IW = ne10

    val KD = ne02
    val KH = ne01
    val KW = ne00

    val OD = ne3 / N
    val OH = ne2
    val OW = ne1
    val OH_OW = OH * OW
    val KD_KH_KW = KD * KH * KW
    val KH_KW = KH * KW
    val IC_KD_KH_KW = IC * KD * KH * KW

    require(nb10 == 4L)

    val wdata = dst.data as ByteArray
    val src1Data = src1.data as ByteArray

    for (inn in 0 until N) {
        for (iod in 0 until OD) {
            for (ioh in 0 until OH) {
                for (iow in 0 until OW) {
                    var iic = ith.toLong()
                    while (iic < IC) {
                        val dstBase = ((inn * OD * OH_OW + iod * OH_OW + ioh * OW + iow) * IC_KD_KH_KW).toInt()
                        val srcDataOff = ((inn * IC + iic) * nb13).toInt()

                        for (ikd in 0 until KD) {
                            for (ikh in 0 until KH) {
                                for (ikw in 0 until KW) {
                                    val iiw = iow * s0 + ikw * d0 - p0
                                    val iih = ioh * s1 + ikh * d1 - p1
                                    val iid = iod * s2 + ikd * d2 - p2

                                    val dstIdx = (iic * KD_KH_KW + ikd * KH_KW + ikh * KW + ikw).toInt()

                                    if (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                        io.github.kotlinmania.llama.ore.writeFloat3(
                                            wdata,
                                            (dstBase + dstIdx) * 4,
                                            0.0f
                                        )
                                    } else {
                                        val sOff = srcDataOff + (iid * nb12 + iih * nb11 + iiw * nb10).toInt()
                                        val srcVal = io.github.kotlinmania.llama.ore.readFloat3(
                                            src1Data,
                                            sOff
                                        )
                                        io.github.kotlinmania.llama.ore.writeFloat3(
                                            wdata,
                                            (dstBase + dstIdx) * 4,
                                            srcVal
                                        )
                                    }
                                }
                            }
                        }
                        iic += nth
                    }
                }
            }
        }
    }
}

fun ggmlComputeForwardIm2col3d(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    when (dst.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardIm2col3dF16(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardIm2col3dF32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_conv_2d — ops.cpp line 6653
// ============================================================================

private fun ggmlWrapAround(coord: Long, size: Long): Long {
    return (coord + size) % size
}

private fun ggmlComputeForwardConv2dImpl(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    kernel: io.github.kotlinmania.llama.ore.GGMLTensor,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    kernelType: io.github.kotlinmania.llama.ore.GGMLType
) {
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(kernel))
    require(kernelType == io.github.kotlinmania.llama.ore.GGMLType.F16 || kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(kernel.type == kernelType)

    val traits = io.github.kotlinmania.llama.ore.ggmlGetTypeTraits(kernelType)

    val strideX = dst.opParams[0]
    val strideY = dst.opParams[1]
    val padX = dst.opParams[2]
    val padY = dst.opParams[3]
    val dilationX = dst.opParams[4]
    val dilationY = dst.opParams[5]

    val cIn = src.ne[2]
    val cOut = kernel.ne[3]
    require(cIn == kernel.ne[2])

    val srcW = src.ne[0]
    val srcH = src.ne[1]
    val knlW = kernel.ne[0]
    val knlH = kernel.ne[1]
    val dstW = dst.ne[0]
    val dstH = dst.ne[1]

    val srcData = src.data as ByteArray
    val knlData = kernel.data as ByteArray
    val dstData = dst.data as ByteArray

    val knlN = knlW * knlH * cIn
    val patchTotal = dst.ne[3] * dstW * dstH

    val typeSizeBytes = traits.typeSize.toInt()
    val spacePP = knlN * typeSizeBytes + cOut * 4
    val batchSize = params.wsize.toLong() / spacePP
    val patchesPerBatch = if (batchSize > 8) (batchSize / 8) * 8 else batchSize
    val batchN = (patchTotal + patchesPerBatch - 1) / patchesPerBatch

    require(patchesPerBatch > 0 && batchSize >= 1)

    val tmp = params.wdata as ByteArray
    val threadpool = params.threadpool ?: return

    for (batchI in 0 until batchN) {
        val patchStartBatch = batchI * patchesPerBatch
        val patchEndBatch = min(patchStartBatch + patchesPerBatch, patchTotal)
        val patchN = patchEndBatch - patchStartBatch

        val patchPerThread = (patchN + params.nth - 1) / params.nth
        val patchStart = patchStartBatch + params.ith * patchPerThread
        val patchEnd = min(patchStart + patchPerThread, patchEndBatch)

        // im2col for a patch
        for (p in patchStart until patchEnd) {
            val batchIdx = p / (dstW * dstH)
            val srcX = (p / dstW) % dstH
            val srcY = p % dstW

            val srcBaseOff = (batchIdx * src.nb[3].toLong()).toInt()
            val dstRowOff = ((p % patchesPerBatch) * knlN * typeSizeBytes).toInt()

            for (ic in 0 until cIn) {
                for (ky in 0 until knlH) {
                    for (kx in 0 until knlW) {
                        val sy = srcX * strideY + ky * dilationY - padY
                        val sx = srcY * strideX + kx * dilationX - padX

                        val dstIdx = (ic * (knlH * knlW) + ky * knlW + kx).toInt()

                        val srcVal: Float
                        if (sy < 0 || sy >= srcH || sx < 0 || sx >= srcW) {
                            srcVal = 0.0f
                        } else {
                            val srcOff = srcBaseOff + (sx * src.nb[0].toLong() + sy * src.nb[1].toLong() + ic * src.nb[2].toLong()).toInt()
                            srcVal = io.github.kotlinmania.llama.ore.readFloat3(srcData, srcOff)
                        }

                        val elemOff = dstRowOff + dstIdx * typeSizeBytes
                        if (kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                            io.github.kotlinmania.llama.ore.writeFloat3(tmp, elemOff, srcVal)
                        } else {
                            io.github.kotlinmania.llama.ore.writeShort3(
                                tmp,
                                elemOff,
                                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(srcVal).toShort()
                            )
                        }
                    }
                }
            }
        }

        io.github.kotlinmania.llama.ore.ggmlBarrier(threadpool)

        val gemmOutputOff = (patchesPerBatch * knlN * typeSizeBytes).toInt()

        // NOTE: GEMM step simplified — in the C++ this calls ggml_call_mul_mat.
        // Here we do a direct scalar matrix multiply: patches[patchN, knlN] × kernel[knlN, cOut] = output[patchN, cOut]
        val gemmPerThread = (patchN + params.nth - 1) / params.nth
        val gemmStart = params.ith * gemmPerThread
        val gemmEnd = min(gemmStart + gemmPerThread, patchN)

        for (row in gemmStart until gemmEnd) {
            for (oc in 0 until cOut) {
                var sum = 0.0f
                for (k in 0 until knlN) {
                    val aVal: Float
                    val aOff = (row * knlN * typeSizeBytes + k * typeSizeBytes).toInt()
                    if (kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                        aVal = io.github.kotlinmania.llama.ore.readFloat3(tmp, aOff)
                    } else {
                        aVal = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                            io.github.kotlinmania.llama.ore.readShort3(
                                tmp,
                                aOff
                            ).toUShort()
                        )
                    }
                    val bOff = (oc * knlN * typeSizeBytes + k * typeSizeBytes).toInt()
                    val bVal: Float
                    if (kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                        bVal = io.github.kotlinmania.llama.ore.readFloat3(knlData, bOff)
                    } else {
                        bVal = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                            io.github.kotlinmania.llama.ore.readShort3(
                                knlData,
                                bOff
                            ).toUShort()
                        )
                    }
                    sum += aVal * bVal
                }
                io.github.kotlinmania.llama.ore.writeFloat3(
                    tmp,
                    gemmOutputOff + (row * cOut + oc).toInt() * 4,
                    sum
                )
            }
        }

        io.github.kotlinmania.llama.ore.ggmlBarrier(threadpool)

        // permute back
        val permutePerThread = (patchN + params.nth - 1) / params.nth
        val permuteStart = params.ith * permutePerThread
        val permuteEnd = min(permuteStart + permutePerThread, patchN)

        for (i in permuteStart until permuteEnd) {
            val p = patchStartBatch + i
            val batchIdx = p / (dstW * dstH)
            val dstY = (p / dstW) % dstH
            val dstX = p % dstW

            for (oc in 0 until cOut) {
                val value = io.github.kotlinmania.llama.ore.readFloat3(
                    tmp,
                    gemmOutputOff + (i * cOut + oc).toInt() * 4
                )
                val dstOff = (dstX * dst.nb[0].toLong() + dstY * dst.nb[1].toLong() + oc * dst.nb[2].toLong() + batchIdx * dst.nb[3].toLong()).toInt()
                io.github.kotlinmania.llama.ore.writeFloat3(dstData, dstOff, value)
            }
        }
    }
}

fun ggmlComputeForwardConv2d(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    io.github.kotlinmania.llama.ore.ggmlComputeForwardConv2dImpl(params, src0, src1, dst, src0.type)
}

// ============================================================================
// ggml_compute_forward_conv_3d — ops.cpp line 6791
// ============================================================================

private fun ggmlComputeForwardConv3dImpl(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    kernel: io.github.kotlinmania.llama.ore.GGMLTensor,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    kernelType: io.github.kotlinmania.llama.ore.GGMLType
) {
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(kernel))
    require(kernelType == io.github.kotlinmania.llama.ore.GGMLType.F16 || kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(kernel.type == kernelType)

    val traits = io.github.kotlinmania.llama.ore.ggmlGetTypeTraits(kernelType)

    val s0 = dst.opParams[0]; val s1 = dst.opParams[1]; val s2 = dst.opParams[2]
    val p0 = dst.opParams[3]; val p1 = dst.opParams[4]; val p2 = dst.opParams[5]
    val d0 = dst.opParams[6]; val d1 = dst.opParams[7]; val d2 = dst.opParams[8]
    val c = dst.opParams[9].toLong()
    val n = dst.opParams[10].toLong()
    val oc = dst.opParams[11].toLong()

    val srcW = src.ne[0]; val srcH = src.ne[1]; val srcD = src.ne[2]
    val knlW = kernel.ne[0]; val knlH = kernel.ne[1]; val knlD = kernel.ne[2]
    val dstW = dst.ne[0]; val dstH = dst.ne[1]; val dstD = dst.ne[2]

    val srcData = src.data as ByteArray
    val knlData = kernel.data as ByteArray
    val dstData = dst.data as ByteArray

    val typeSizeBytes = traits.typeSize.toInt()
    val knlNPerChannel = knlW * knlH * knlD
    val knlNTotal = knlNPerChannel * c
    val patchTotal = n * dstW * dstH * dstD

    val spacePP = knlNTotal * typeSizeBytes + oc * 4
    val batchSize = params.wsize.toLong() / spacePP
    val patchesPerBatch = if (batchSize > 8) (batchSize / 8) * 8 else batchSize
    val batchN = (patchTotal + patchesPerBatch - 1) / patchesPerBatch

    require(patchesPerBatch > 0 && batchSize >= 1)

    val tmp = params.wdata as ByteArray

    for (batchI in 0 until batchN) {
        val patchStartBatch = batchI * patchesPerBatch
        val patchEndBatch = min(patchStartBatch + patchesPerBatch, patchTotal)
        val patchNInBatch = patchEndBatch - patchStartBatch

        val patchPerThread = (patchNInBatch + params.nth - 1) / params.nth
        val patchStart = patchStartBatch + params.ith * patchPerThread
        val patchEnd = min(patchStart + patchPerThread, patchEndBatch)

        for (p in patchStart until patchEnd) {
            val pInBatch = p % (dstW * dstH * dstD)
            val pInDepth = pInBatch % (dstW * dstH)
            val batchIdx = p / (dstW * dstH * dstD)
            val dstZ = pInBatch / (dstW * dstH)
            val dstY = pInDepth / dstW
            val dstX = pInDepth % dstW

            val dstRowOff = ((p % patchesPerBatch) * knlNTotal * typeSizeBytes).toInt()

            for (ic in 0 until c) {
                for (kz in 0 until knlD) {
                    for (ky in 0 until knlH) {
                        for (kx in 0 until knlW) {
                            val sz = dstZ * s2 + kz * d2 - p2
                            val sy = dstY * s1 + ky * d1 - p1
                            val sx = dstX * s0 + kx * d0 - p0

                            val dstIdx = (ic * knlNPerChannel + kz * (knlH * knlW) + ky * knlW + kx).toInt()

                            val srcVal: Float
                            if (sz < 0 || sz >= srcD || sy < 0 || sy >= srcH || sx < 0 || sx >= srcW) {
                                srcVal = 0.0f
                            } else {
                                val cnIdx = batchIdx * c + ic
                                val srcOff = (sx * src.nb[0].toLong() + sy * src.nb[1].toLong() + sz * src.nb[2].toLong() + cnIdx * src.nb[3].toLong()).toInt()
                                srcVal = io.github.kotlinmania.llama.ore.readFloat3(srcData, srcOff)
                            }

                            val elemOff = dstRowOff + dstIdx * typeSizeBytes
                            if (kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                                io.github.kotlinmania.llama.ore.writeFloat3(tmp, elemOff, srcVal)
                            } else {
                                io.github.kotlinmania.llama.ore.writeShort3(
                                    tmp,
                                    elemOff,
                                    io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(srcVal).toShort()
                                )
                            }
                        }
                    }
                }
            }
        }

        val threadpool = params.threadpool ?: return
        io.github.kotlinmania.llama.ore.ggmlBarrier(threadpool)

        val gemmOutputOff = (patchesPerBatch * knlNTotal * typeSizeBytes).toInt()

        // GEMM: patches[patchNInBatch, knlNTotal] × kernel[knlNTotal, oc] = output[patchNInBatch, oc]
        val gemmPerThread = (patchNInBatch + params.nth - 1) / params.nth
        val gemmStart = params.ith * gemmPerThread
        val gemmEnd = min(gemmStart + gemmPerThread, patchNInBatch)

        for (row in gemmStart until gemmEnd) {
            for (ioc in 0 until oc) {
                var sum = 0.0f
                for (k in 0 until knlNTotal) {
                    val aOff = (row * knlNTotal * typeSizeBytes + k * typeSizeBytes).toInt()
                    val aVal: Float = if (kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                        io.github.kotlinmania.llama.ore.readFloat3(tmp, aOff)
                    } else {
                        io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                            io.github.kotlinmania.llama.ore.readShort3(
                                tmp,
                                aOff
                            ).toUShort()
                        )
                    }
                    val bOff = (ioc * knlNTotal * typeSizeBytes + k * typeSizeBytes).toInt()
                    val bVal: Float = if (kernelType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                        io.github.kotlinmania.llama.ore.readFloat3(knlData, bOff)
                    } else {
                        io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                            io.github.kotlinmania.llama.ore.readShort3(
                                knlData,
                                bOff
                            ).toUShort()
                        )
                    }
                    sum += aVal * bVal
                }
                io.github.kotlinmania.llama.ore.writeFloat3(
                    tmp,
                    gemmOutputOff + (row * oc + ioc).toInt() * 4,
                    sum
                )
            }
        }

        io.github.kotlinmania.llama.ore.ggmlBarrier(threadpool)

        val permutePerThread = (patchNInBatch + params.nth - 1) / params.nth
        val permuteStart = params.ith * permutePerThread
        val permuteEnd = min(permuteStart + permutePerThread, patchNInBatch)

        for (i in permuteStart until permuteEnd) {
            val p = patchStartBatch + i
            val pInBatch2 = p % (dstW * dstH * dstD)
            val pInDepth2 = pInBatch2 % (dstW * dstH)
            val batchIdx2 = p / (dstW * dstH * dstD)
            val dstZ2 = pInBatch2 / (dstW * dstH)
            val dstY2 = pInDepth2 / dstW
            val dstX2 = pInDepth2 % dstW

            for (ioc in 0 until oc) {
                val value = io.github.kotlinmania.llama.ore.readFloat3(
                    tmp,
                    gemmOutputOff + (i * oc + ioc).toInt() * 4
                )
                val ocnIdx = batchIdx2 * oc + ioc
                val dstOff = (dstX2 * dst.nb[0].toLong() + dstY2 * dst.nb[1].toLong() + dstZ2 * dst.nb[2].toLong() + ocnIdx * dst.nb[3].toLong()).toInt()
                io.github.kotlinmania.llama.ore.writeFloat3(dstData, dstOff, value)
            }
        }
    }
}

fun ggmlComputeForwardConv3d(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    io.github.kotlinmania.llama.ore.ggmlComputeForwardConv3dImpl(params, src0, src1, dst, src0.type)
}

// ============================================================================
// ggml_compute_forward_conv_transpose_2d — ops.cpp line 6933
// ============================================================================

private fun ggmlComputeForwardConvTranspose2dImplF16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01; val ne02 = b.ne02; val ne03 = b.ne03
    val nb00 = b.nb00; val nb01 = b.nb01; val nb02 = b.nb02; val nb03 = b.nb03
    val ne10 = b.ne10; val ne11 = b.ne11; val ne12 = b.ne12
    val nb10 = b.nb10; val nb11 = b.nb11; val nb12 = b.nb12
    val ne0 = b.ne0; val ne2 = b.ne2
    val nb2 = b.nb2

    val ith = params.ith
    val nth = params.nth

    val nk = (ne00 * ne01 * ne02 * ne03).toInt()

    require(nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type).toLong())
    require(nb10 == 4L)

    val wdata = params.wdata as ByteArray

    if (ith == 0) {
        for (i in 0 until params.wsize.toInt()) {
            wdata[i] = 0
        }

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        val src0Data = src0.data as ByteArray
        for (i03 in 0 until ne03.toInt()) {
            for (i02 in 0 until ne02.toInt()) {
                val srcOff = (i03 * nb03 + i02 * nb02).toInt()
                val dstOff = (i02 * ne01 * ne00 * ne03).toInt() * 2
                for (i01 in 0 until ne01.toInt()) {
                    for (i00 in 0 until ne00.toInt()) {
                        val srcVal = io.github.kotlinmania.llama.ore.readShort3(
                            src0Data,
                            srcOff + (i01 * ne00.toInt() + i00) * 2
                        )
                        io.github.kotlinmania.llama.ore.writeShort3(
                            wdata,
                            dstOff + (i01 * ne00.toInt() * ne03.toInt() + i00 * ne03.toInt() + i03) * 2,
                            srcVal
                        )
                    }
                }
            }
        }

        // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
        val src1Data = src1.data as ByteArray
        val wdataOff = nk * 2
        for (i12 in 0 until ne12.toInt()) {
            for (i11 in 0 until ne11.toInt()) {
                val srcRowOff = (i12 * nb12 + i11 * nb11).toInt()
                val dstRowOff = wdataOff + (i11 * ne10.toInt() * ne12.toInt()) * 2
                for (i10 in 0 until ne10.toInt()) {
                    val srcVal =
                        io.github.kotlinmania.llama.ore.readFloat3(src1Data, srcRowOff + i10 * 4)
                    io.github.kotlinmania.llama.ore.writeShort3(
                        wdata,
                        dstRowOff + (i10 * ne12.toInt() + i12) * 2,
                        io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(srcVal).toShort()
                    )
                }
            }
        }

        // zero dst
        val dstData = dst.data as ByteArray
        for (i in 0 until io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()) {
            dstData[i] = 0
        }
    }
    io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)

    val stride = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    val np = ne2.toInt()
    val dp = (np + nth - 1) / nth
    val ip0 = dp * ith
    val ip1 = min(ip0 + dp, np)

    val wdataSrcOff = nk * 2
    val dstData = dst.data as ByteArray

    for (i2 in ip0 until ip1) {
        val dstRowOff = (i2 * nb2).toInt()
        val wdataKernelOff = i2 * ne01.toInt() * ne00.toInt() * ne03.toInt() * 2
        for (i11 in 0 until ne11.toInt()) {
            for (i10 in 0 until ne10.toInt()) {
                val i1n = (i11 * ne10.toInt() * ne12.toInt() + i10 * ne12.toInt()) * 2
                for (i01 in 0 until ne01.toInt()) {
                    for (i00 in 0 until ne00.toInt()) {
                        val v = io.github.kotlinmania.llama.ore.ggmlVecDotF16(
                            ne03,
                            wdata, wdataSrcOff + i1n,
                            wdata, wdataKernelOff + (i01 * ne00.toInt() * ne03.toInt() + i00 * ne03.toInt()) * 2
                        )
                        val dstIdx = ((i11 * stride + i01) * ne0.toInt() + i10 * stride + i00) * 4
                        val prev =
                            io.github.kotlinmania.llama.ore.readFloat3(dstData, dstRowOff + dstIdx)
                        io.github.kotlinmania.llama.ore.writeFloat3(
                            dstData,
                            dstRowOff + dstIdx,
                            prev + v
                        )
                    }
                }
            }
        }
    }
}

private fun ggmlComputeForwardConvTranspose2dImplF32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals3(dst)
    val ne00 = b.ne00; val ne01 = b.ne01; val ne02 = b.ne02; val ne03 = b.ne03
    val nb00 = b.nb00; val nb01 = b.nb01; val nb02 = b.nb02; val nb03 = b.nb03
    val ne10 = b.ne10; val ne11 = b.ne11; val ne12 = b.ne12
    val nb10 = b.nb10; val nb11 = b.nb11; val nb12 = b.nb12
    val ne0 = b.ne0; val ne2 = b.ne2
    val nb2 = b.nb2

    val ith = params.ith
    val nth = params.nth

    val nk = (ne00 * ne01 * ne02 * ne03).toInt()

    require(nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type).toLong())
    require(nb10 == 4L)

    val wdata = params.wdata as ByteArray

    if (ith == 0) {
        for (i in 0 until params.wsize.toInt()) {
            wdata[i] = 0
        }

        // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
        val src0Data = src0.data as ByteArray
        for (i03 in 0 until ne03.toInt()) {
            for (i02 in 0 until ne02.toInt()) {
                val srcOff = (i03 * nb03 + i02 * nb02).toInt()
                val dstOff = (i02 * ne01 * ne00 * ne03).toInt() * 4
                for (i01 in 0 until ne01.toInt()) {
                    for (i00 in 0 until ne00.toInt()) {
                        val srcVal = io.github.kotlinmania.llama.ore.readFloat3(
                            src0Data,
                            srcOff + (i01 * ne00.toInt() + i00) * 4
                        )
                        io.github.kotlinmania.llama.ore.writeFloat3(
                            wdata,
                            dstOff + (i01 * ne00.toInt() * ne03.toInt() + i00 * ne03.toInt() + i03) * 4,
                            srcVal
                        )
                    }
                }
            }
        }

        // permute source data (src1)
        val src1Data = src1.data as ByteArray
        val wdataOff = nk * 4
        for (i12 in 0 until ne12.toInt()) {
            for (i11 in 0 until ne11.toInt()) {
                val srcRowOff = (i12 * nb12 + i11 * nb11).toInt()
                val dstRowOff = wdataOff + (i11 * ne10.toInt() * ne12.toInt()) * 4
                for (i10 in 0 until ne10.toInt()) {
                    val srcVal =
                        io.github.kotlinmania.llama.ore.readFloat3(src1Data, srcRowOff + i10 * 4)
                    io.github.kotlinmania.llama.ore.writeFloat3(
                        wdata,
                        dstRowOff + (i10 * ne12.toInt() + i12) * 4,
                        srcVal
                    )
                }
            }
        }

        // zero dst
        val dstData = dst.data as ByteArray
        for (i in 0 until io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()) {
            dstData[i] = 0
        }
    }
    io.github.kotlinmania.llama.ore.ggmlBarrier(params.threadpool!!)

    val stride = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    val np = ne2.toInt()
    val dp = (np + nth - 1) / nth
    val ip0 = dp * ith
    val ip1 = min(ip0 + dp, np)

    val wdataSrcOff = nk * 4
    val dstData = dst.data as ByteArray

    for (i2 in ip0 until ip1) {
        val dstRowOff = (i2 * nb2).toInt()
        val wdataKernelOff = i2 * ne01.toInt() * ne00.toInt() * ne03.toInt() * 4
        for (i11 in 0 until ne11.toInt()) {
            for (i10 in 0 until ne10.toInt()) {
                val i1n = (i11 * ne10.toInt() * ne12.toInt() + i10 * ne12.toInt()) * 4
                for (i01 in 0 until ne01.toInt()) {
                    for (i00 in 0 until ne00.toInt()) {
                        var v = 0.0f
                        for (k in 0 until ne03.toInt()) {
                            val a = io.github.kotlinmania.llama.ore.readFloat3(
                                wdata,
                                wdataSrcOff + i1n + k * 4
                            )
                            val bVal = io.github.kotlinmania.llama.ore.readFloat3(
                                wdata,
                                wdataKernelOff + (i01 * ne00.toInt() * ne03.toInt() + i00 * ne03.toInt() + k) * 4
                            )
                            v += a * bVal
                        }
                        val dstIdx = ((i11 * stride + i01) * ne0.toInt() + i10 * stride + i00) * 4
                        val prev =
                            io.github.kotlinmania.llama.ore.readFloat3(dstData, dstRowOff + dstIdx)
                        io.github.kotlinmania.llama.ore.writeFloat3(
                            dstData,
                            dstRowOff + dstIdx,
                            prev + v
                        )
                    }
                }
            }
        }
    }
}

fun ggmlComputeForwardConvTranspose2d(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardConvTranspose2dImplF16(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardConvTranspose2dImplF32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_conv_2d_dw — ops.cpp line 7060
// ============================================================================

data class GGMLConv2dDwParams(
    val channels: Long,
    val batch: Long,
    val srcW: Long,
    val srcH: Long,
    val dstW: Long,
    val dstH: Long,
    val knlW: Long,
    val knlH: Long,
    val strideX: Int,
    val strideY: Int,
    val padX: Int,
    val padY: Int,
    val dilationX: Int,
    val dilationY: Int
)

private fun ggmlComputeForwardConv2dDwCwhn(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    kernel: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    p: io.github.kotlinmania.llama.ore.GGMLConv2dDwParams
) {
    val c = p.channels
    val knlData = kernel.data as ByteArray
    val srcDataBytes = src.data as ByteArray
    val dstDataBytes = dst.data as ByteArray

    val rowsTotal = p.dstH * p.batch
    val rowsPerThread = (rowsTotal + params.nth - 1) / params.nth
    val rowStart = params.ith * rowsPerThread
    val rowEnd = min(rowStart + rowsPerThread, rowsTotal)

    // Scalar only (no SIMD in Kotlin)
    for (row in rowStart until rowEnd) {
        val dstY = row % p.dstH
        val srcBase = ((row / p.dstH) * p.srcW * p.srcH * c).toInt() * 4
        for (dstX in 0 until p.dstW) {
            val dstBase = ((row * p.dstW + dstX) * c).toInt() * 4
            val srcYBase = dstY * p.strideY - p.padY
            val srcXBase = dstX * p.strideX - p.padX

            for (cI in 0 until c) {
                var sum = 0.0f
                for (knlY in 0 until p.knlH) {
                    val srcY = srcYBase + knlY * p.dilationY
                    if (srcY < 0 || srcY >= p.srcH) continue
                    for (knlX in 0 until p.knlW) {
                        val srcX = srcXBase + knlX * p.dilationX
                        if (srcX < 0 || srcX >= p.srcW) continue
                        val kOff = ((knlY * p.knlW + knlX) * c + cI).toInt() * 4
                        val sOff = srcBase + ((srcY * p.srcW + srcX) * c + cI).toInt() * 4
                        sum += io.github.kotlinmania.llama.ore.readFloat3(
                            knlData,
                            kOff
                        ) * io.github.kotlinmania.llama.ore.readFloat3(srcDataBytes, sOff)
                    }
                }
                io.github.kotlinmania.llama.ore.writeFloat3(
                    dstDataBytes,
                    dstBase + cI.toInt() * 4,
                    sum
                )
            }
        }
    }
}

private fun ggmlComputeForwardConv2dDwWhcn(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    kernel: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    p: io.github.kotlinmania.llama.ore.GGMLConv2dDwParams
) {
    val n = p.channels * p.batch
    val perThread = (n + params.nth - 1) / params.nth
    val start = params.ith * perThread
    val end = min(start + perThread, n)

    val knlData = kernel.data as ByteArray
    val srcData = src.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i in start until end) {
        val knlBase = ((i % p.channels) * p.knlW * p.knlH).toInt() * 4
        val srcBase = (i * p.srcW * p.srcH).toInt() * 4
        val dstBase = (i * p.dstW * p.dstH).toInt() * 4

        for (dstY in 0 until p.dstH) {
            for (dstX in 0 until p.dstW) {
                var sum = 0.0f
                for (knlY in 0 until p.knlH) {
                    val srcY = dstY * p.strideY + knlY * p.dilationY - p.padY
                    if (srcY < 0 || srcY >= p.srcH) continue
                    for (knlX in 0 until p.knlW) {
                        val srcX = dstX * p.strideX + knlX * p.dilationX - p.padX
                        if (srcX < 0 || srcX >= p.srcW) continue
                        val kOff = knlBase + (knlY * p.knlW + knlX).toInt() * 4
                        val sOff = srcBase + (srcY * p.srcW + srcX).toInt() * 4
                        sum += io.github.kotlinmania.llama.ore.readFloat3(
                            knlData,
                            kOff
                        ) * io.github.kotlinmania.llama.ore.readFloat3(srcData, sOff)
                    }
                }
                io.github.kotlinmania.llama.ore.writeFloat3(
                    dstData,
                    dstBase + (dstY * p.dstW + dstX).toInt() * 4,
                    sum
                )
            }
        }
    }
}

fun ggmlComputeForwardConv2dDw(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val kernel = dst.src[0]!!
    val src = dst.src[1]!!

    val p = io.github.kotlinmania.llama.ore.GGMLConv2dDwParams(
        channels = src.ne[2],
        batch = src.ne[3],
        srcW = src.ne[0],
        srcH = src.ne[1],
        dstW = dst.ne[0],
        dstH = dst.ne[1],
        knlW = kernel.ne[0],
        knlH = kernel.ne[1],
        strideX = dst.opParams[0],
        strideY = dst.opParams[1],
        padX = dst.opParams[2],
        padY = dst.opParams[3],
        dilationX = dst.opParams[4],
        dilationY = dst.opParams[5]
    )

    require(kernel.ne[3] == p.channels)
    require(dst.ne[3] == p.batch)

    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(src)) {
        io.github.kotlinmania.llama.ore.ggmlComputeForwardConv2dDwWhcn(params, src, kernel, dst, p)
    } else if (io.github.kotlinmania.llama.ore.ggmlIsContiguousChannels(src)) {
        require(kernel.nb[0] >= kernel.nb[2] && kernel.nb[1] >= kernel.nb[0])
        io.github.kotlinmania.llama.ore.ggmlComputeForwardConv2dDwCwhn(params, src, kernel, dst, p)
    } else {
        error("non-contiguous memory layout not supported")
    }
}

// ============================================================================
// ggml_compute_forward_pool_1d — ops.cpp line 7236
// ============================================================================

private fun ggmlComputeForwardPool1dKsp(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    op: io.github.kotlinmania.llama.ore.GGMLOpPool,
    k: Int,
    s: Int,
    p: Int,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src = dst.src[0]!!

    require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || src.type == io.github.kotlinmania.llama.ore.GGMLType.F16)

    if (params.ith != 0) return

    val IW = src.ne[0]
    val OW = dst.ne[0]

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src)

    val srcData = src.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in 0 until nr) {
        val srowOff = (ir * src.nb[1].toLong()).toInt()
        val drowOff = (ir * dst.nb[1].toLong()).toInt()

        for (ow in 0 until OW) {
            var res: Float = when (op) {
                io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> 0.0f
                io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> -Float.MAX_VALUE
                io.github.kotlinmania.llama.ore.GGMLOpPool.COUNT -> error("fatal error")
            }

            var count = 0
            val base = (ow * s - p).toInt()

            for (ki in 0 until k) {
                val j = base + ki
                if (j < 0 || j >= IW.toInt()) continue

                val v: Float = if (src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                    io.github.kotlinmania.llama.ore.readFloat3(srcData, srowOff + j * 4)
                } else {
                    io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                        io.github.kotlinmania.llama.ore.readShort3(
                            srcData,
                            srowOff + j * 2
                        ).toUShort()
                    )
                }

                when (op) {
                    io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> res += v
                    io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> res = max(v, res)
                    io.github.kotlinmania.llama.ore.GGMLOpPool.COUNT -> error("fatal error")
                }

                ++count
            }

            when (op) {
                io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> res = if (count > 0) res / count else 0.0f
                io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> {}
                io.github.kotlinmania.llama.ore.GGMLOpPool.COUNT -> error("fatal error")
            }

            io.github.kotlinmania.llama.ore.writeFloat3(dstData, drowOff + ow.toInt() * 4, res)
        }
    }
}

fun ggmlComputeForwardPool1d(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val op = io.github.kotlinmania.llama.ore.GGMLOpPool.entries[dst.opParams[0]]
    val k0 = dst.opParams[1]
    val s0 = dst.opParams[2]
    val p0 = dst.opParams[3]

    io.github.kotlinmania.llama.ore.ggmlComputeForwardPool1dKsp(params, op, k0, s0, p0, dst)
}

// ============================================================================
// ggml_compute_forward_pool_2d — ops.cpp line 7321
// ============================================================================

fun ggmlComputeForwardPool2d(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src = dst.src[0]!!

    require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || src.type == io.github.kotlinmania.llama.ore.GGMLType.F16)

    if (params.ith != 0) return

    val op = io.github.kotlinmania.llama.ore.GGMLOpPool.entries[dst.opParams[0]]
    val k0 = dst.opParams[1]
    val k1 = dst.opParams[2]
    val s0 = dst.opParams[3]
    val s1 = dst.opParams[4]
    val p0 = dst.opParams[5]
    val p1 = dst.opParams[6]

    val srcData = src.data as ByteArray
    val srcNb1 = src.nb[1].toLong()
    val srcNb2 = src.nb[2].toLong()
    val totalBytes = io.github.kotlinmania.llama.ore.ggmlNbytes(src).toLong()

    val px = dst.ne[0]
    val py = dst.ne[1]
    val pa = px * py

    val dstData = dst.data as ByteArray
    var dplaneOff = 0

    val ka = k0 * k1
    val offset0 = -p0
    val offset1 = -p1

    var cdataOff = 0
    while (cdataOff < totalBytes) {
        for (oy in 0 until py.toInt()) {
            for (ox in 0 until px.toInt()) {
                var res: Float = when (op) {
                    io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> 0.0f
                    io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> -Float.MAX_VALUE
                    io.github.kotlinmania.llama.ore.GGMLOpPool.COUNT -> error("fatal error")
                }

                val ix = offset0 + ox * s0
                val iy = offset1 + oy * s1

                for (ky in 0 until k1) {
                    if (iy + ky < 0 || iy + ky >= src.ne[1].toInt()) continue

                    val srowOff = cdataOff + (srcNb1 * (iy + ky)).toInt()
                    for (kx in 0 until k0) {
                        val j = ix + kx
                        if (j < 0 || j >= src.ne[0].toInt()) continue

                        val srowJ: Float = if (src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                            io.github.kotlinmania.llama.ore.readFloat3(srcData, srowOff + j * 4)
                        } else {
                            io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                                io.github.kotlinmania.llama.ore.readShort3(
                                    srcData,
                                    srowOff + j * 2
                                ).toUShort()
                            )
                        }

                        when (op) {
                            io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> res += srowJ
                            io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> res = max(srowJ, res)
                            io.github.kotlinmania.llama.ore.GGMLOpPool.COUNT -> error("fatal error")
                        }
                    }
                }

                when (op) {
                    io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> res /= ka
                    io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> {}
                    io.github.kotlinmania.llama.ore.GGMLOpPool.COUNT -> error("fatal error")
                }

                io.github.kotlinmania.llama.ore.writeFloat3(
                    dstData,
                    (dplaneOff + oy * px.toInt() + ox) * 4,
                    res
                )
            }
        }

        cdataOff += srcNb2.toInt()
        dplaneOff += pa.toInt()
    }
}

// ============================================================================
// ggml_compute_forward_pool_2d_back — ops.cpp line 7408
// ============================================================================

fun ggmlComputeForwardPool2dBack(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src = dst.src[0]!!
    val dstf = dst.src[1]!!

    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || dst.type == io.github.kotlinmania.llama.ore.GGMLType.F16)

    if (params.ith != 0) return

    val op = io.github.kotlinmania.llama.ore.GGMLOpPool.entries[dst.opParams[0]]
    val k0 = dst.opParams[1]
    val k1 = dst.opParams[2]
    val s0 = dst.opParams[3]
    val s1 = dst.opParams[4]
    val p0 = dst.opParams[5]
    val p1 = dst.opParams[6]

    val cdata = dst.data as ByteArray
    val cdataf = dstf.data as ByteArray
    val totalBytes = io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt()

    require(params.ith == 0)
    // memset dst to 0
    for (i in 0 until totalBytes) {
        cdata[i] = 0
    }

    val px = src.ne[0]
    val py = src.ne[1]
    val pa = px * py

    val srcData = src.data as ByteArray
    var splaneOff = 0

    val ka = k0 * k1
    val offset0 = -p0
    val offset1 = -p1

    val dstNb1 = dst.nb[1].toLong()
    val dstNb2 = dst.nb[2].toLong()

    var cdataOff = 0
    var cdatafOff = 0
    while (cdataOff < totalBytes) {
        for (oy in 0 until py.toInt()) {
            for (ox in 0 until px.toInt()) {
                val grad0 = io.github.kotlinmania.llama.ore.readFloat3(
                    srcData,
                    (splaneOff + oy * px.toInt() + ox) * 4
                )

                val ix = offset0 + ox * s0
                val iy = offset1 + oy * s1

                if (op == io.github.kotlinmania.llama.ore.GGMLOpPool.MAX) {
                    var maxval = -Float.MAX_VALUE
                    var kxmax = -1
                    var kymax = -1

                    for (ky in 0 until k1) {
                        if (iy + ky < 0 || iy + ky >= dst.ne[1].toInt()) continue
                        val drowfOff = cdatafOff + (dstNb1 * (iy + ky)).toInt()
                        for (kx in 0 until k0) {
                            val j = ix + kx
                            if (j < 0 || j >= dst.ne[0].toInt()) continue

                            val v = if (dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                                io.github.kotlinmania.llama.ore.readFloat3(cdataf, drowfOff + j * 4)
                            } else {
                                io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                                    io.github.kotlinmania.llama.ore.readShort3(
                                        cdataf,
                                        drowfOff + j * 2
                                    ).toUShort()
                                )
                            }
                            if (v <= maxval) continue

                            maxval = v
                            kxmax = kx
                            kymax = ky
                        }
                    }

                    if (kxmax == -1 || kymax == -1) continue

                    val drowOff = cdataOff + (dstNb1 * (iy + kymax)).toInt()
                    val j = ix + kxmax
                    if (dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                        val prev =
                            io.github.kotlinmania.llama.ore.readFloat3(cdata, drowOff + j * 4)
                        io.github.kotlinmania.llama.ore.writeFloat3(
                            cdata,
                            drowOff + j * 4,
                            prev + grad0
                        )
                    } else {
                        val prevFp16 =
                            io.github.kotlinmania.llama.ore.readShort3(cdata, drowOff + j * 2)
                        val prevF32 =
                            io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(prevFp16.toUShort())
                        io.github.kotlinmania.llama.ore.writeShort3(
                            cdata,
                            drowOff + j * 2,
                            io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(grad0 + prevF32)
                                .toShort()
                        )
                    }
                } else if (op == io.github.kotlinmania.llama.ore.GGMLOpPool.AVG) {
                    val grad = grad0 / ka

                    for (ky in 0 until k1) {
                        if (iy + ky < 0 || iy + ky >= dst.ne[1].toInt()) continue
                        val drowOff = cdataOff + (dstNb1 * (iy + ky)).toInt()
                        for (kx in 0 until k0) {
                            val j = ix + kx
                            if (j < 0 || j >= dst.ne[0].toInt()) continue

                            if (dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                                val prev = io.github.kotlinmania.llama.ore.readFloat3(
                                    cdata,
                                    drowOff + j * 4
                                )
                                io.github.kotlinmania.llama.ore.writeFloat3(
                                    cdata,
                                    drowOff + j * 4,
                                    prev + grad
                                )
                            } else {
                                val prevFp16 = io.github.kotlinmania.llama.ore.readShort3(
                                    cdata,
                                    drowOff + j * 2
                                )
                                val sum = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(grad).toShort() + prevFp16
                                io.github.kotlinmania.llama.ore.writeShort3(
                                    cdata,
                                    drowOff + j * 2,
                                    sum.toShort()
                                )
                            }
                        }
                    }
                } else {
                    require(false)
                }
            }
        }

        cdataOff += dstNb2.toInt()
        cdatafOff += dstNb2.toInt()
        splaneOff += pa.toInt()
    }
}
