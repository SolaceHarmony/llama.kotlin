// port-lint: source ggml/src/ggml-cpu/quants.c
package io.github.kotlinmania.llama.ore


import io.github.kotlinmania.llama.ore.readIntLE
import io.github.kotlinmania.llama.ore.readShortLE
import io.github.kotlinmania.llama.ore.readUIntLE
import io.github.kotlinmania.llama.ore.weightAt
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

// ---------------------------------------------------------------------------
// Lookup tables from ggml-common.h
// ---------------------------------------------------------------------------

internal val kmask_iq2xs = ubyteArrayOf(1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u)

internal val ksigns_iq2xs = ubyteArrayOf(
      0u, 129u, 130u,   3u, 132u,   5u,   6u, 135u, 136u,   9u,  10u, 139u,  12u, 141u, 142u,  15u,
    144u,  17u,  18u, 147u,  20u, 149u, 150u,  23u,  24u, 153u, 154u,  27u, 156u,  29u,  30u, 159u,
    160u,  33u,  34u, 163u,  36u, 165u, 166u,  39u,  40u, 169u, 170u,  43u, 172u,  45u,  46u, 175u,
     48u, 177u, 178u,  51u, 180u,  53u,  54u, 183u, 184u,  57u,  58u, 187u,  60u, 189u, 190u,  63u,
    192u,  65u,  66u, 195u,  68u, 197u, 198u,  71u,  72u, 201u, 202u,  75u, 204u,  77u,  78u, 207u,
     80u, 209u, 210u,  83u, 212u,  85u,  86u, 215u, 216u,  89u,  90u, 219u,  92u, 221u, 222u,  95u,
     96u, 225u, 226u,  99u, 228u, 101u, 102u, 231u, 232u, 105u, 106u, 235u, 108u, 237u, 238u, 111u,
    240u, 113u, 114u, 243u, 116u, 245u, 246u, 119u, 120u, 249u, 250u, 123u, 252u, 125u, 126u, 255u,
)

internal val kvalues_iq4nl = byteArrayOf(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
)

internal val kvalues_mxfp4 = byteArrayOf(
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
)

// ---------------------------------------------------------------------------
// ByteArray block-access helpers (read fields from packed quantized data)
// ---------------------------------------------------------------------------

private fun ByteArray.readShortLE(offset: Int): Short =
    ((this[offset].toInt() and 0xFF) or ((this[offset + 1].toInt() and 0xFF) shl 8)).toShort()

private fun ByteArray.readIntLE(offset: Int): Int =
    (this[offset].toInt() and 0xFF) or
    ((this[offset + 1].toInt() and 0xFF) shl 8) or
    ((this[offset + 2].toInt() and 0xFF) shl 16) or
    ((this[offset + 3].toInt() and 0xFF) shl 24)

private fun ByteArray.readUIntLE(offset: Int): UInt = readIntLE(offset).toUInt()

private fun fp16ToF32(bits: Short): Float =
    io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(bits)

// ---------------------------------------------------------------------------
// quantize_row_* delegates (from ggml-cpu/quants.c)
// These delegate to _ref implementations which live in ggml-quants.c
// ---------------------------------------------------------------------------

fun quantize_row_q1_0(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q1_0_ref(x, y, k)
}
fun quantize_row_q4_0(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q4_0_ref(x, y, k)
}
fun quantize_row_q4_1(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q4_1_ref(x, y, k)
}
fun quantize_row_q5_0(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q5_0_ref(x, y, k)
}
fun quantize_row_q5_1(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q5_1_ref(x, y, k)
}
fun quantize_row_q8_0_generic(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_0_ref(x, y, k)
}
fun quantize_row_q8_1_generic(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_1_ref(x, y, k)
}
fun quantize_row_mxfp4(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_mxfp4_ref(x, y, k)
}
fun quantize_row_nvfp4(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_nvfp4_ref(x, y, k)
}
fun quantize_row_q2_K(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q2_K_ref(x, y, k)
}
fun quantize_row_q3_K(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q3_K_ref(x, y, k)
}
fun quantize_row_q4_K(x: FloatArray, y: ByteArray, k: Long) { require(k % io.github.kotlinmania.llama.ore.QK_K == 0L); io.github.kotlinmania.llama.ore.quantize_row_q4_K_ref(
    x,
    y,
    k
)
}
fun quantize_row_q5_K(x: FloatArray, y: ByteArray, k: Long) { require(k % io.github.kotlinmania.llama.ore.QK_K == 0L); io.github.kotlinmania.llama.ore.quantize_row_q5_K_ref(
    x,
    y,
    k
)
}
fun quantize_row_q6_K(x: FloatArray, y: ByteArray, k: Long) { require(k % io.github.kotlinmania.llama.ore.QK_K == 0L); io.github.kotlinmania.llama.ore.quantize_row_q6_K_ref(
    x,
    y,
    k
)
}
fun quantize_row_tq1_0(x: FloatArray, y: ByteArray, k: Long) { require(k % io.github.kotlinmania.llama.ore.QK_K == 0L); io.github.kotlinmania.llama.ore.quantize_row_tq1_0_ref(
    x,
    y,
    k
)
}
fun quantize_row_tq2_0(x: FloatArray, y: ByteArray, k: Long) { require(k % io.github.kotlinmania.llama.ore.QK_K == 0L); io.github.kotlinmania.llama.ore.quantize_row_tq2_0_ref(
    x,
    y,
    k
)
}
fun quantize_row_q8_K_generic(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_K_ref(x, y, k)
}
fun quantize_row_iq4_nl(x: FloatArray, y: ByteArray, k: Long) { require(k % io.github.kotlinmania.llama.ore.QK4_NL == 0L); io.github.kotlinmania.llama.ore.quantize_row_iq4_nl_ref(
    x,
    y,
    k
)
}

// _ref implementations — delegate to GGMLQuantsRef top-level functions with offset 0
private fun quantize_row_q1_0_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q1_0_ref(x, 0, y, 0, k)
}
private fun quantize_row_q4_0_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q4_0_ref(x, 0, y, 0, k)
}
private fun quantize_row_q4_1_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q4_1_ref(x, 0, y, 0, k)
}
private fun quantize_row_q5_0_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q5_0_ref(x, 0, y, 0, k)
}
private fun quantize_row_q5_1_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q5_1_ref(x, 0, y, 0, k)
}
private fun quantize_row_q8_0_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_0_ref(x, 0, y, 0, k)
}
private fun quantize_row_q8_1_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_1_ref(x, 0, y, 0, k)
}
private fun quantize_row_mxfp4_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_mxfp4_ref(x, 0, y, 0, k)
}
private fun quantize_row_nvfp4_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_nvfp4_ref(x, 0, y, 0, k)
}
private fun quantize_row_q2_K_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q2_K_ref(x, 0, y, 0, k)
}
private fun quantize_row_q3_K_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q3_K_ref(x, 0, y, 0, k)
}
private fun quantize_row_q4_K_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q4_K_ref(x, 0, y, 0, k)
}
private fun quantize_row_q5_K_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q5_K_ref(x, 0, y, 0, k)
}
private fun quantize_row_q6_K_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q6_K_ref(x, 0, y, 0, k)
}
private fun quantize_row_tq1_0_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_tq1_0_ref(x, 0, y, 0, k)
}
private fun quantize_row_tq2_0_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_tq2_0_ref(x, 0, y, 0, k)
}
private fun quantize_row_q8_K_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_K_ref(x, 0, y, 0, k)
}
private fun quantize_row_iq4_nl_ref(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_iq4_nl_ref(x, 0, y, 0, k)
}
private fun quantize_iq4_xs(x: FloatArray, y: ByteArray, nrow: Int, k: Long, imatrix: FloatArray?) {
    io.github.kotlinmania.llama.ore.quantize_iq4_xs(x, y, nrow.toLong(), k, imatrix)
}
fun quantize_row_iq4_xs(x: FloatArray, y: ByteArray, k: Long) { require(k % io.github.kotlinmania.llama.ore.QK_K == 0L); io.github.kotlinmania.llama.ore.quantize_iq4_xs(
    x,
    y,
    1,
    k,
    null
)
}

// ---------------------------------------------------------------------------
// Dot products (ggml-cpu/quants.c generic scalar fallbacks)
//
// All functions operate on packed block data stored as ByteArray.
// Block layout: array of blocks, each block is SIZE_BYTES bytes.
// The 'd' field (fp16 scale) is always the first 2 bytes of a block.
// ---------------------------------------------------------------------------

/**
 * Port of `ggml_vec_dot_q1_0_q8_0_generic` from ggml-cpu/quants.c.
 * Binary (1-bit) quantized × Q8_0 dot product.
 */
fun ggml_vec_dot_q1_0_q8_0_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    val qk = io.github.kotlinmania.llama.ore.QK1_0
    val nb = n / qk
    require(n % qk == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ1_0.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES

    var sumf = 0.0f

    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val d0 = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff))

        var sumi = 0.0f

        for (k in 0 until 4) {
            val yOff = vyOffset + (i * 4 + k) * yBlockSize
            val d1 = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff))

            var sumiBlock = 0

            for (j in 0 until io.github.kotlinmania.llama.ore.QK8_0) {
                val bitIndex = k * io.github.kotlinmania.llama.ore.QK8_0 + j
                val byteIndex = bitIndex / 8
                val bitOffset = bitIndex % 8

                val xi = if ((vx[xOff + 2 + byteIndex].toInt() ushr bitOffset) and 1 == 1) 1 else -1
                sumiBlock += xi * vy[yOff + 2 + j].toInt()
            }

            sumi += d1 * sumiBlock
        }

        sumf += d0 * sumi
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q4_0_q8_0_generic` from ggml-cpu/quants.c.
 * 4-bit symmetric quantized × Q8_0 dot product.
 */
fun ggml_vec_dot_q4_0_q8_0_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    val nb = n / qk
    require(n % qk == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ4_0.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * xBlockSize
        val yOff = vyOffset + ib * yBlockSize
        var sumi0 = 0
        var sumi1 = 0

        for (j in 0 until qk / 2) {
            val v0 = (vx[xOff + 2 + j].toInt() and 0x0F) - 8
            val v1 = ((vx[xOff + 2 + j].toInt() and 0xFF) ushr 4) - 8

            sumi0 += v0 * vy[yOff + 2 + j].toInt()
            sumi1 += v1 * vy[yOff + 2 + j + qk / 2].toInt()
        }

        val sumi = sumi0 + sumi1
        sumf += sumi * io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * io.github.kotlinmania.llama.ore.fp16ToF32(
            vy.readShortLE(yOff)
        )
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q4_1_q8_1_generic` from ggml-cpu/quants.c.
 * 4-bit asymmetric quantized × Q8_1 dot product.
 */
fun ggml_vec_dot_q4_1_q8_1_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    val qk = io.github.kotlinmania.llama.ore.QK8_1
    val nb = n / qk
    require(n % qk == 0)
    require(nrc == 1)

    // BlockQ4_1: d(2) + m(2) + qs(QK4_1/2)
    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ4_1.SIZE_BYTES
    // BlockQ8_1: d(2) + s(2) + qs(QK8_1)
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_1.SIZE_BYTES

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * xBlockSize
        val yOff = vyOffset + ib * yBlockSize
        var sumi0 = 0
        var sumi1 = 0

        for (j in 0 until qk / 2) {
            val v0 = vx[xOff + 4 + j].toInt() and 0x0F
            val v1 = (vx[xOff + 4 + j].toInt() and 0xFF) ushr 4

            sumi0 += v0 * vy[yOff + 4 + j].toInt()
            sumi1 += v1 * vy[yOff + 4 + j + qk / 2].toInt()
        }

        val sumi = sumi0 + sumi1
        val xd = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff))
        val yd = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff))
        val xm = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + 2))
        val ys = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff + 2))
        sumf += (xd * yd) * sumi + xm * ys
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_mxfp4_q8_0_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_mxfp4_q8_0_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(nrc == 1)
    require(n % io.github.kotlinmania.llama.ore.QK_MXFP4 == 0)

    // BlockMXFP4: e(1) + qs(QK_MXFP4/2)
    val xBlockSize = io.github.kotlinmania.llama.ore.BlockMXFP4.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_MXFP4

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * xBlockSize
        val yOff = vyOffset + ib * yBlockSize
        val e = vx[xOff].toInt() and 0xFF
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff)) * io.github.kotlinmania.llama.ore.ggml_e8m0_to_fp32_half(
            e.toUByte()
        )

        var sumi1 = 0
        var sumi2 = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_MXFP4 / 2) {
            val qsByte = vx[xOff + 1 + j].toInt() and 0xFF
            sumi1 += vy[yOff + 2 + j].toInt() * io.github.kotlinmania.llama.ore.kvalues_mxfp4[qsByte and 0xf].toInt()
            sumi2 += vy[yOff + 2 + j + io.github.kotlinmania.llama.ore.QK_MXFP4 / 2].toInt() * io.github.kotlinmania.llama.ore.kvalues_mxfp4[qsByte ushr 4].toInt()
        }
        sumf += d * (sumi1 + sumi2)
    }
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_nvfp4_q8_0_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_nvfp4_q8_0_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(nrc == 1)
    require(n % io.github.kotlinmania.llama.ore.QK_NVFP4 == 0)

    // BlockNVFP4: d(QK_NVFP4/QK_NVFP4_SUB bytes) + qs(QK_NVFP4/2 bytes)
    val xBlockSize = io.github.kotlinmania.llama.ore.BlockNVFP4.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_NVFP4

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * xBlockSize
        val dFieldSize = io.github.kotlinmania.llama.ore.QK_NVFP4 / io.github.kotlinmania.llama.ore.QK_NVFP4_SUB

        for (sIdx in 0 until 4) {
            val d =
                io.github.kotlinmania.llama.ore.ggml_ue4m3_to_fp32((vx[xOff + sIdx].toInt() and 0xFF).toUByte())
            val q8Block = sIdx / 2
            val q8Off = (sIdx % 2) * io.github.kotlinmania.llama.ore.QK_NVFP4_SUB
            val yOff = vyOffset + (2 * ib + q8Block) * yBlockSize
            val dy = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff))

            var sumiLo = 0
            var sumiHi = 0
            for (j in 0 until io.github.kotlinmania.llama.ore.QK_NVFP4_SUB / 2) {
                val qv = vx[xOff + dFieldSize + sIdx * (io.github.kotlinmania.llama.ore.QK_NVFP4_SUB / 2) + j].toInt() and 0xFF
                sumiLo += vy[yOff + 2 + q8Off + j].toInt() * io.github.kotlinmania.llama.ore.kvalues_mxfp4[qv and 0xf].toInt()
                sumiHi += vy[yOff + 2 + q8Off + j + io.github.kotlinmania.llama.ore.QK_NVFP4_SUB / 2].toInt() * io.github.kotlinmania.llama.ore.kvalues_mxfp4[qv ushr 4].toInt()
            }

            sumf += dy * d * (sumiLo + sumiHi)
        }
    }
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q5_0_q8_0_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_q5_0_q8_0_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    val nb = n / qk
    require(n % qk == 0)
    require(nrc == 1)

    // BlockQ5_0: d(2) + qh(4) + qs(QK5_0/2)
    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ5_0.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * xBlockSize
        val yOff = vyOffset + ib * yBlockSize

        val qh = vx.readUIntLE(xOff + 2)

        var sumi0 = 0
        var sumi1 = 0

        for (j in 0 until qk / 2) {
            val xh0 = (((qh and (1u shl (j + 0))) shr (j + 0)) shl 4).toInt() and 0xFF
            val xh1 = (((qh and (1u shl (j + 16))) shr (j + 12))).toInt() and 0xFF

            val x0 = ((((vx[xOff + 6 + j].toInt() and 0x0F) or xh0) - 16).toByte()).toInt()
            val x1 = (((((vx[xOff + 6 + j].toInt() and 0xFF) ushr 4) or xh1) - 16).toByte()).toInt()

            sumi0 += x0 * vy[yOff + 2 + j].toInt()
            sumi1 += x1 * vy[yOff + 2 + j + qk / 2].toInt()
        }

        val sumi = sumi0 + sumi1
        sumf += (io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * io.github.kotlinmania.llama.ore.fp16ToF32(
            vy.readShortLE(yOff)
        )) * sumi
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q5_1_q8_1_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_q5_1_q8_1_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    val qk = io.github.kotlinmania.llama.ore.QK8_1
    val nb = n / qk
    require(n % qk == 0)
    require(nrc == 1)

    // BlockQ5_1: d(2) + m(2) + qh(4) + qs(QK5_1/2)
    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ5_1.SIZE_BYTES
    // BlockQ8_1: d(2) + s(2) + qs(QK8_1)
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_1.SIZE_BYTES

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * xBlockSize
        val yOff = vyOffset + ib * yBlockSize

        val qh = vx.readUIntLE(xOff + 4)

        var sumi0 = 0
        var sumi1 = 0

        for (j in 0 until qk / 2) {
            val xh0 = (((qh shr (j + 0)) shl 4) and 0x10u).toInt()
            val xh1 = (((qh shr (j + 12))) and 0x10u).toInt()

            val x0 = (vx[xOff + 8 + j].toInt() and 0xF) or xh0
            val x1 = ((vx[xOff + 8 + j].toInt() and 0xFF) ushr 4) or xh1

            sumi0 += x0 * vy[yOff + 4 + j].toInt()
            sumi1 += x1 * vy[yOff + 4 + j + qk / 2].toInt()
        }

        val sumi = sumi0 + sumi1
        val xd = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff))
        val yd = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff))
        val xm = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + 2))
        val ys = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff + 2))
        sumf += (xd * yd) * sumi + xm * ys
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q8_0_q8_0_generic` from ggml-cpu/quants.c.
 * Q8_0 × Q8_0 dot product.
 */
fun ggml_vec_dot_q8_0_q8_0_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    val nb = n / qk
    require(n % qk == 0)
    require(nrc == 1)

    val blockSize = io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * blockSize
        val yOff = vyOffset + ib * blockSize
        var sumi = 0

        for (j in 0 until qk) {
            sumi += vx[xOff + 2 + j].toInt() * vy[yOff + 2 + j].toInt()
        }

        sumf += sumi * (io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * io.github.kotlinmania.llama.ore.fp16ToF32(
            vy.readShortLE(yOff)
        ))
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_tq1_0_q8_K_generic` from ggml-cpu/quants.c.
 * Ternary 1-bit × Q8_K dot product.
 */
fun ggml_vec_dot_tq1_0_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockTQ1_0.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val pow3 = intArrayOf(1, 3, 9, 27, 81, 243)

    // BlockTQ1_0: d(2) + qs[QK_K/64 + (QK_K - 4*QK_K/64)/5] + qh[QK_K/64]
    val qsSize = io.github.kotlinmania.llama.ore.QK_K / 64 + (io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5
    val qhSize = io.github.kotlinmania.llama.ore.QK_K / 64

    // BlockQ8K: d(4 float) + qs[QK_K] + bsums[QK_K/16]
    // Layout: d(4 bytes float) + qs(QK_K bytes) + bsums(QK_K/16 * 2 bytes)

    var sumf = 0.0f

    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize

        var sum = 0

        val qsAlignedEnd = qsSize - qsSize % 32
        for (j in 0 until qsAlignedEnd step 32) {
            for (l in 0 until 5) {
                for (m in 0 until 32) {
                    val q = (vx[xOff + 2 + j + m].toInt() and 0xFF) * pow3[l]
                    val xi = ((q.toUShort().toInt() * 3) ushr 8)
                    sum += (xi - 1) * vy[yOff + 4 + j * 5 + l * 32 + m].toInt()
                }
            }
        }
        for (j in qsAlignedEnd until qsSize step 16) {
            val chunkSize = minOf(16, qsSize - j)
            for (l in 0 until 5) {
                for (m in 0 until chunkSize) {
                    val q = (vx[xOff + 2 + j + m].toInt() and 0xFF) * pow3[l]
                    val xi = ((q.toUShort().toInt() * 3) ushr 8)
                    sum += (xi - 1) * vy[yOff + 4 + j * 5 + l * chunkSize + m].toInt()
                }
            }
        }

        for (l in 0 until 4) {
            for (j in 0 until qhSize) {
                val q = (vx[xOff + 2 + qsSize + j].toInt() and 0xFF) * pow3[l]
                val xi = ((q.toUShort().toInt() * 3) ushr 8)
                sum += (xi - 1) * vy[yOff + 4 + qsSize * 5 + l * qhSize + j].toInt()
            }
        }

        val xd = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff))
        val yd = Float.fromBits(vy.readIntLE(yOff))
        sumf += sum.toFloat() * (xd * yd)
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_tq2_0_q8_K_generic` from ggml-cpu/quants.c.
 * Ternary 2-bit × Q8_K dot product.
 */
fun ggml_vec_dot_tq2_0_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockTQ2_0.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K
    // BlockTQ2_0: d(2) + qs(QK_K/4)
    val qsSize = io.github.kotlinmania.llama.ore.QK_K / 4

    var sumf = 0.0f

    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize

        var sumi = 0

        for (j in 0 until qsSize step 32) {
            for (l in 0 until 4) {
                for (k in 0 until 32) {
                    sumi += vy[yOff + 4 + j * 4 + l * 32 + k].toInt() *
                        ((((vx[xOff + 2 + j + k].toInt() and 0xFF) ushr (l * 2)) and 3) - 1)
                }
            }
        }

        val yd = Float.fromBits(vy.readIntLE(yOff))
        val xd = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff))
        sumf += sumi.toFloat() * (yd * xd)
    }

    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q2_K_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_q2_K_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ2K.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockQ2K: scales(QK_K/16=16) + qs(QK_K/4=64) + d(2) + dmin(2) = 84
    val q2ScalesOff = 0
    val q2QsOff = io.github.kotlinmania.llama.ore.QK_K / 16
    val q2dOff = io.github.kotlinmania.llama.ore.QK_K / 16 + io.github.kotlinmania.llama.ore.QK_K / 4
    val q2dminOff = io.github.kotlinmania.llama.ore.QK_K / 16 + io.github.kotlinmania.llama.ore.QK_K / 4 + 2

    // BlockQ8K: d(4 float) + qs(QK_K) + bsums(QK_K/16 * 2)
    val q8dOff = 0
    val q8QsOff = 4
    val q8BsumsOff = 4 + io.github.kotlinmania.llama.ore.QK_K

    var sumf = 0.0f

    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize

        var q2Pos = 0
        var q8Pos = 0

        var summs = 0
        for (j in 0 until 16) {
            val bsum = vy.readShortLE(yOff + q8BsumsOff + j * 2).toInt()
            summs += bsum * ((vx[xOff + q2ScalesOff + j].toInt() and 0xFF) ushr 4)
        }

        val dall = Float.fromBits(vy.readIntLE(yOff + q8dOff)) * io.github.kotlinmania.llama.ore.fp16ToF32(
            vx.readShortLE(xOff + q2dOff)
        )
        val dmin = Float.fromBits(vy.readIntLE(yOff + q8dOff)) * io.github.kotlinmania.llama.ore.fp16ToF32(
            vx.readShortLE(xOff + q2dminOff)
        )

        var isum = 0
        var isIdx = 0
        for (k in 0 until io.github.kotlinmania.llama.ore.QK_K / 128) {
            var shift = 0
            for (j in 0 until 4) {
                val d1 = (vx[xOff + q2ScalesOff + isIdx].toInt() and 0xFF) and 0xF
                isIdx++
                var isuml = 0
                for (l in 0 until 16) {
                    isuml += vy[yOff + q8QsOff + q8Pos + l].toInt() *
                        (((vx[xOff + q2QsOff + q2Pos + l].toInt() and 0xFF) ushr shift) and 3)
                }
                isum += d1 * isuml

                val d2 = (vx[xOff + q2ScalesOff + isIdx].toInt() and 0xFF) and 0xF
                isIdx++
                isuml = 0
                for (l in 16 until 32) {
                    isuml += vy[yOff + q8QsOff + q8Pos + l].toInt() *
                        (((vx[xOff + q2QsOff + q2Pos + l].toInt() and 0xFF) ushr shift) and 3)
                }
                isum += d2 * isuml

                shift += 2
                q8Pos += 32
            }
            q2Pos += 32
        }
        sumf += dall * isum - dmin * summs
    }
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q3_K_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_q3_K_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val kmask1 = 0x03030303u
    val kmask2 = 0x0f0f0f0fu

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ3K.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockQ3K: hmask(QK_K/8=32) + qs(QK_K/4=64) + scales(12) + d(2) = 110
    val hmaskOff = 0
    val q3QsOff = io.github.kotlinmania.llama.ore.QK_K / 8
    val q3ScalesOff = io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 4
    val q3dOff = io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 4 + 12

    val q8dOff = 0
    val q8QsOff = 4

    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)
    val auxs = IntArray(4)

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        for (l in 0 until 8) aux32[l] = 0

        var aIdx = 0
        var q3Pos = 0
        var m = 1
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) aux8[aIdx + l] = (vx[xOff + q3QsOff + q3Pos + l].toInt() and 0xFF) and 3
            for (l in 0 until 32) aux8[aIdx + l] -= if ((vx[xOff + hmaskOff + l].toInt() and 0xFF) and m != 0) 0 else 4
            aIdx += 32; m = m shl 1
            for (l in 0 until 32) aux8[aIdx + l] = ((vx[xOff + q3QsOff + q3Pos + l].toInt() and 0xFF) ushr 2) and 3
            for (l in 0 until 32) aux8[aIdx + l] -= if ((vx[xOff + hmaskOff + l].toInt() and 0xFF) and m != 0) 0 else 4
            aIdx += 32; m = m shl 1
            for (l in 0 until 32) aux8[aIdx + l] = ((vx[xOff + q3QsOff + q3Pos + l].toInt() and 0xFF) ushr 4) and 3
            for (l in 0 until 32) aux8[aIdx + l] -= if ((vx[xOff + hmaskOff + l].toInt() and 0xFF) and m != 0) 0 else 4
            aIdx += 32; m = m shl 1
            for (l in 0 until 32) aux8[aIdx + l] = ((vx[xOff + q3QsOff + q3Pos + l].toInt() and 0xFF) ushr 6) and 3
            for (l in 0 until 32) aux8[aIdx + l] -= if ((vx[xOff + hmaskOff + l].toInt() and 0xFF) and m != 0) 0 else 4
            aIdx += 32; m = m shl 1
            q3Pos += 32
        }
        aIdx = 0

        // Unpack scales from 12 bytes
        auxs[0] = vx.readIntLE(xOff + q3ScalesOff)
        auxs[1] = vx.readIntLE(xOff + q3ScalesOff + 4)
        auxs[2] = vx.readIntLE(xOff + q3ScalesOff + 8)
        val tmp = auxs[2]
        auxs[2] = ((auxs[0].toUInt() shr 4) and kmask2).toInt() or ((((tmp.toUInt() shr 4) and kmask1) shl 4).toInt())
        auxs[3] = ((auxs[1].toUInt() shr 4) and kmask2).toInt() or ((((tmp.toUInt() shr 6) and kmask1) shl 4).toInt())
        auxs[0] = (auxs[0].toUInt() and kmask2).toInt() or ((((tmp.toUInt() shr 0) and kmask1) shl 4).toInt())
        auxs[1] = (auxs[1].toUInt() and kmask2).toInt() or ((((tmp.toUInt() shr 2) and kmask1) shl 4).toInt())

        // Interpret auxs as byte array of scales
        var q8Pos = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val scaleIdx = j
            val scaleWord = auxs[scaleIdx / 4]
            val scaleByte = ((scaleWord ushr ((scaleIdx % 4) * 8)) and 0xFF).toByte().toInt()

            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += (scaleByte - 32) * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += (scaleByte - 32) * aux16[l]
            q8Pos += 8; aIdx += 8
        }
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + q3dOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))
        for (l in 0 until 8) sums[l] += d * aux32[l]
    }
    for (l in 0 until 8) sumf += sums[l]
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q4_K_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_q4_K_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val kmask1 = 0x3f3f3f3fu
    val kmask2 = 0x0f0f0f0fu
    val kmask3 = 0x03030303u

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ4K.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockQ4K: d(2) + dmin(2) + scales(12) + qs(QK_K/2)
    val q4dOff = 0
    val q4dminOff = 2
    val q4ScalesOff = 4
    val q4QsOff = 16

    val q8dOff = 0
    val q8QsOff = 4
    val q8BsumsOff = 4 + io.github.kotlinmania.llama.ore.QK_K

    val utmp = UIntArray(4)
    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        for (l in 0 until 8) aux32[l] = 0

        var aIdx = 0
        var q4Pos = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 64) {
            for (l in 0 until 32) aux8[aIdx + l] = (vx[xOff + q4QsOff + q4Pos + l].toInt() and 0xFF) and 0xF
            aIdx += 32
            for (l in 0 until 32) aux8[aIdx + l] = ((vx[xOff + q4QsOff + q4Pos + l].toInt() and 0xFF) ushr 4)
            aIdx += 32; q4Pos += 32
        }

        utmp[0] = vx.readUIntLE(xOff + q4ScalesOff)
        utmp[1] = vx.readUIntLE(xOff + q4ScalesOff + 4)
        utmp[2] = vx.readUIntLE(xOff + q4ScalesOff + 8)
        utmp[3] = ((utmp[2] shr 4) and kmask2) or (((utmp[1] shr 6) and kmask3) shl 4)
        val uaux = utmp[1] and kmask1
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] shr 6) and kmask3) shl 4)
        utmp[2] = uaux
        utmp[0] = utmp[0] and kmask1

        // scales = bytes of utmp[0..1], mins = bytes of utmp[2..3]
        var sumi = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val bsum = vy.readShortLE(yOff + q8BsumsOff + j * 2).toInt()
            val minByte = ((utmp[2 + j / 4] shr ((j % 4) * 8)) and 0xFFu).toInt()
            sumi += bsum * minByte
        }

        aIdx = 0
        var isIdx = 0
        var q8Pos = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val scale = ((utmp[isIdx / 4] shr ((isIdx % 4) * 8)) and 0xFFu).toInt()
            isIdx++
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
        }
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + q4dOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))
        for (l in 0 until 8) sums[l] += d * aux32[l]
        val dmin = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + q4dminOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))
        sumf -= dmin * sumi
    }
    for (l in 0 until 8) sumf += sums[l]
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q5_K_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_q5_K_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val kmask1 = 0x3f3f3f3fu
    val kmask2 = 0x0f0f0f0fu
    val kmask3 = 0x03030303u

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ5K.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockQ5K: d(2) + dmin(2) + scales(12) + qh(QK_K/8) + qs(QK_K/2)
    val q5dOff = 0
    val q5dminOff = 2
    val q5ScalesOff = 4
    val q5QhOff = 16
    val q5QsOff = 16 + io.github.kotlinmania.llama.ore.QK_K / 8

    val q8dOff = 0
    val q8QsOff = 4
    val q8BsumsOff = 4 + io.github.kotlinmania.llama.ore.QK_K

    val utmp = UIntArray(4)
    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        for (l in 0 until 8) aux32[l] = 0

        var aIdx = 0
        var q4Pos = 0
        var m = 1
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 64) {
            for (l in 0 until 32) aux8[aIdx + l] = (vx[xOff + q5QsOff + q4Pos + l].toInt() and 0xFF) and 0xF
            for (l in 0 until 32) aux8[aIdx + l] += if ((vx[xOff + q5QhOff + l].toInt() and 0xFF) and m != 0) 16 else 0
            aIdx += 32; m = m shl 1
            for (l in 0 until 32) aux8[aIdx + l] = ((vx[xOff + q5QsOff + q4Pos + l].toInt() and 0xFF) ushr 4)
            for (l in 0 until 32) aux8[aIdx + l] += if ((vx[xOff + q5QhOff + l].toInt() and 0xFF) and m != 0) 16 else 0
            aIdx += 32; m = m shl 1
            q4Pos += 32
        }

        utmp[0] = vx.readUIntLE(xOff + q5ScalesOff)
        utmp[1] = vx.readUIntLE(xOff + q5ScalesOff + 4)
        utmp[2] = vx.readUIntLE(xOff + q5ScalesOff + 8)
        utmp[3] = ((utmp[2] shr 4) and kmask2) or (((utmp[1] shr 6) and kmask3) shl 4)
        val uaux = utmp[1] and kmask1
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] shr 6) and kmask3) shl 4)
        utmp[2] = uaux
        utmp[0] = utmp[0] and kmask1

        var sumi = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val bsum = vy.readShortLE(yOff + q8BsumsOff + j * 2).toInt()
            val minByte = ((utmp[2 + j / 4] shr ((j % 4) * 8)) and 0xFFu).toInt()
            sumi += bsum * minByte
        }

        aIdx = 0
        var isIdx = 0
        var q8Pos = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val scale = ((utmp[isIdx / 4] shr ((isIdx % 4) * 8)) and 0xFFu).toInt()
            isIdx++
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
        }
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + q5dOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))
        for (l in 0 until 8) sums[l] += d * aux32[l]
        val dmin = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + q5dminOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))
        sumf -= dmin * sumi
    }
    for (l in 0 until 8) sumf += sums[l]
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_q6_K_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_q6_K_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockQ6K.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockQ6K: ql(QK_K/2) + qh(QK_K/4) + scales(QK_K/16) + d(2)
    val qlOff = 0
    val qhOff = io.github.kotlinmania.llama.ore.QK_K / 2
    val scalesOff = io.github.kotlinmania.llama.ore.QK_K / 2 + io.github.kotlinmania.llama.ore.QK_K / 4
    val q6dOff = io.github.kotlinmania.llama.ore.QK_K / 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 16

    val q8dOff = 0
    val q8QsOff = 4

    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        for (l in 0 until 8) aux32[l] = 0

        var aIdx = 0
        var q4Pos = 0
        var qhPos = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                aux8[aIdx + l + 0] = ((vx[xOff + qlOff + q4Pos + l].toInt() and 0xFF) and 0xF) or
                    ((((vx[xOff + qhOff + qhPos + l].toInt() and 0xFF) ushr 0) and 3) shl 4) - 32
                aux8[aIdx + l + 32] = ((vx[xOff + qlOff + q4Pos + l + 32].toInt() and 0xFF) and 0xF) or
                    ((((vx[xOff + qhOff + qhPos + l].toInt() and 0xFF) ushr 2) and 3) shl 4) - 32
                aux8[aIdx + l + 64] = (((vx[xOff + qlOff + q4Pos + l].toInt() and 0xFF) ushr 4)) or
                    ((((vx[xOff + qhOff + qhPos + l].toInt() and 0xFF) ushr 4) and 3) shl 4) - 32
                aux8[aIdx + l + 96] = (((vx[xOff + qlOff + q4Pos + l + 32].toInt() and 0xFF) ushr 4)) or
                    ((((vx[xOff + qhOff + qhPos + l].toInt() and 0xFF) ushr 6) and 3) shl 4) - 32
            }
            aIdx += 128
            q4Pos += 64
            qhPos += 32
        }
        aIdx = 0
        var isIdx = 0
        var q8Pos = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val scale = vx[xOff + scalesOff + isIdx].toInt()
            isIdx++
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
            for (l in 0 until 8) aux16[l] = vy[yOff + q8QsOff + q8Pos + l].toInt() * aux8[aIdx + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Pos += 8; aIdx += 8
        }
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff + q6dOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))
        for (l in 0 until 8) sums[l] += d * aux32[l]
    }
    for (l in 0 until 8) sumf += sums[l]
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_iq4_nl_q8_0_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq4_nl_q8_0_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(nrc == 1)
    require(n % io.github.kotlinmania.llama.ore.QK4_NL == 0)

    // BlockIQ4NL: d(2) + qs(QK4_NL/2)
    val xBlockSize = 2 + io.github.kotlinmania.llama.ore.QK4_NL / 2
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK4_NL

    var sumf = 0.0f

    for (ib in 0 until nb) {
        val xOff = vxOffset + ib * xBlockSize
        val yOff = vyOffset + ib * yBlockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vy.readShortLE(yOff)) * io.github.kotlinmania.llama.ore.fp16ToF32(
            vx.readShortLE(xOff)
        )
        var sumi1 = 0
        var sumi2 = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK4_NL / 2) {
            val qsByte = vx[xOff + 2 + j].toInt() and 0xFF
            sumi1 += vy[yOff + 2 + j].toInt() * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qsByte and 0xf].toInt()
            sumi2 += vy[yOff + 2 + j + io.github.kotlinmania.llama.ore.QK4_NL / 2].toInt() * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qsByte ushr 4].toInt()
        }
        sumf += d * (sumi1 + sumi2)
    }
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_iq4_xs_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq4_xs_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(nrc == 1)
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    // BlockIQ4XS: d(2) + scales_h(2) + scales_l(QK_K/64) + qs(QK_K/2)
    val xScalesHOff = 2
    val xScalesLOff = 4
    val xQsOff = 4 + io.github.kotlinmania.llama.ore.QK_K / 64
    val xBlockSize = 4 + io.github.kotlinmania.llama.ore.QK_K / 64 + io.github.kotlinmania.llama.ore.QK_K / 2
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val q8dOff = 0
    val q8QsOff = 4

    var sumf = 0.0f
    for (ibl in 0 until nb) {
        val xOff = vxOffset + ibl * xBlockSize
        val yOff = vyOffset + ibl * yBlockSize
        val d4d8 = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))
        var h = vx.readShortLE(xOff + xScalesHOff).toInt() and 0xFFFF
        var qsPos = 0
        var q8Pos = 0
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32 step 2) {
            val ls1byte = vx[xOff + xScalesLOff + ib / 2].toInt() and 0xFF
            val ls1 = (ls1byte and 0xf) or ((h shl 4) and 0x30)
            val ls2 = (ls1byte ushr 4) or ((h shl 2) and 0x30)
            h = h ushr 4
            val d1 = d4d8 * (ls1 - 32)
            val d2 = d4d8 * (ls2 - 32)
            var sumi1 = 0; var sumi2 = 0
            for (j in 0 until 16) {
                val qsByte = vx[xOff + xQsOff + qsPos + j].toInt() and 0xFF
                sumi1 += vy[yOff + q8QsOff + q8Pos + j + 0].toInt() * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qsByte and 0xf].toInt()
                sumi2 += vy[yOff + q8QsOff + q8Pos + j + 16].toInt() * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qsByte ushr 4].toInt()
            }
            sumf += d1 * (sumi1 + sumi2)
            qsPos += 16; q8Pos += 32
            sumi1 = 0; sumi2 = 0
            for (j in 0 until 16) {
                val qsByte = vx[xOff + xQsOff + qsPos + j].toInt() and 0xFF
                sumi1 += vy[yOff + q8QsOff + q8Pos + j + 0].toInt() * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qsByte and 0xf].toInt()
                sumi2 += vy[yOff + q8QsOff + q8Pos + j + 16].toInt() * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qsByte ushr 4].toInt()
            }
            sumf += d2 * (sumi1 + sumi2)
            qsPos += 16; q8Pos += 32
        }
    }
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_iq2_xxs_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq2_xxs_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockIQ2XXS.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val q8dOff = 0
    val q8QsOff = 4

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))

        // qs starts at xOff + 2 (uint16_t[QK_K/8], read as raw bytes)
        var q2Pos = xOff + 2
        var q8Pos = 0

        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            // memcpy(aux32, q2, 2*sizeof(uint32_t))
            val aux32_0 = vx.readUIntLE(q2Pos)
            val aux32_1 = vx.readUIntLE(q2Pos + 4)
            q2Pos += 8

            val ls = (2u * (aux32_1 shr 28) + 1u).toInt()
            var sumi = 0
            for (l in 0 until 4) {
                val gridIdx = ((aux32_0 shr (l * 8)) and 0xFFu).toInt()
                val gridVal = io.github.kotlinmania.llama.ore.iq2xxs_grid[gridIdx]
                val signIdx = ((aux32_1 shr (7 * l)) and 127u).toInt()
                val signs = io.github.kotlinmania.llama.ore.ksigns_iq2xs[signIdx]
                for (j in 0 until 8) {
                    val gridByte = ((gridVal shr (j * 8)) and 0xFFu).toInt()
                    val q8val = vy[yOff + q8QsOff + q8Pos + j].toInt()
                    val sign = if ((signs and io.github.kotlinmania.llama.ore.kmask_iq2xs[j]) != 0.toUByte()) -1 else 1
                    sumi += gridByte * q8val * sign
                }
                q8Pos += 8
            }
            bsum += sumi * ls
        }
        sumf += d * bsum
    }
    s[sOffset] = 0.125f * sumf
}

/**
 * Port of `ggml_vec_dot_iq2_xs_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq2_xs_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockIQ2XS.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockIQ2XS: d(2) + qs(uint16_t[32]=64) + scales(8)
    val xQsOff = 2
    val xScOff = 2 + (io.github.kotlinmania.llama.ore.QK_K / 8) * 2  // 66

    val q8dOff = 0
    val q8QsOff = 4

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))

        var q2Pos = xOff + xQsOff
        var q8Pos = 0

        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val scByte = vx[xOff + xScOff + ib32].toInt() and 0xFF
            val ls1 = 2 * (scByte and 0xf) + 1
            val ls2 = 2 * (scByte ushr 4) + 1

            var sumi = 0
            for (l in 0 until 2) {
                val q2val = (vx[q2Pos + l * 2].toInt() and 0xFF) or
                    ((vx[q2Pos + l * 2 + 1].toInt() and 0xFF) shl 8)
                val gridVal = io.github.kotlinmania.llama.ore.iq2xs_grid[q2val and 511]
                val signs = io.github.kotlinmania.llama.ore.ksigns_iq2xs[q2val ushr 9]
                for (j in 0 until 8) {
                    val gridByte = ((gridVal shr (j * 8)) and 0xFFu).toInt()
                    val q8val = vy[yOff + q8QsOff + q8Pos + j].toInt()
                    val sign = if ((signs and io.github.kotlinmania.llama.ore.kmask_iq2xs[j]) != 0.toUByte()) -1 else 1
                    sumi += gridByte * q8val * sign
                }
                q8Pos += 8
            }
            bsum += sumi * ls1

            sumi = 0
            for (l in 2 until 4) {
                val q2val = (vx[q2Pos + l * 2].toInt() and 0xFF) or
                    ((vx[q2Pos + l * 2 + 1].toInt() and 0xFF) shl 8)
                val gridVal = io.github.kotlinmania.llama.ore.iq2xs_grid[q2val and 511]
                val signs = io.github.kotlinmania.llama.ore.ksigns_iq2xs[q2val ushr 9]
                for (j in 0 until 8) {
                    val gridByte = ((gridVal shr (j * 8)) and 0xFFu).toInt()
                    val q8val = vy[yOff + q8QsOff + q8Pos + j].toInt()
                    val sign = if ((signs and io.github.kotlinmania.llama.ore.kmask_iq2xs[j]) != 0.toUByte()) -1 else 1
                    sumi += gridByte * q8val * sign
                }
                q8Pos += 8
            }
            bsum += sumi * ls2
            q2Pos += 8  // q2 += 4 uint16_t = 8 bytes
        }
        sumf += d * bsum
    }
    s[sOffset] = 0.125f * sumf
}

/**
 * Port of `ggml_vec_dot_iq2_s_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq2_s_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockIQ2S.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockIQ2S: d(2) + qs[QK_K/4](64) + qh[QK_K/32](8) + scales[QK_K/32](8)
    val xQsOff = 2
    val xQhOff = 2 + io.github.kotlinmania.llama.ore.QK_K / 4          // 66
    val xScalesOff = 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 32  // 74

    val q8dOff = 0
    val q8QsOff = 4

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))

        var qsPos = xOff + xQsOff
        var signsPos = xOff + xQsOff + io.github.kotlinmania.llama.ore.QK_K / 8  // signs = qs + QK_K/8
        var q8Pos = 0

        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val scByte = vx[xOff + xScalesOff + ib32].toInt() and 0xFF
            val ls1 = 1 + 2 * (scByte and 0xf)
            val ls2 = 1 + 2 * (scByte ushr 4)
            val qhVal = vx[xOff + xQhOff + ib32].toInt() and 0xFF

            var sumi1 = 0
            var sumi2 = 0
            for (l in 0 until 2) {
                val gridIdx = (vx[qsPos + l].toInt() and 0xFF) or (((qhVal shl (8 - 2 * l)) and 0x300))
                val gridVal = io.github.kotlinmania.llama.ore.iq2s_grid[gridIdx]
                val signByte = vx[signsPos + l].toInt() and 0xFF
                for (j in 0 until 8) {
                    val gridByte = ((gridVal shr (j * 8)) and 0xFFu).toInt()
                    val q8val = vy[yOff + q8QsOff + q8Pos + j].toInt()
                    val sign = if ((signByte and io.github.kotlinmania.llama.ore.kmask_iq2xs[j].toInt()) != 0) -1 else 1
                    sumi1 += q8val * gridByte * sign
                }
                q8Pos += 8
            }
            for (l in 2 until 4) {
                val gridIdx = (vx[qsPos + l].toInt() and 0xFF) or (((qhVal shl (8 - 2 * l)) and 0x300))
                val gridVal = io.github.kotlinmania.llama.ore.iq2s_grid[gridIdx]
                val signByte = vx[signsPos + l].toInt() and 0xFF
                for (j in 0 until 8) {
                    val gridByte = ((gridVal shr (j * 8)) and 0xFFu).toInt()
                    val q8val = vy[yOff + q8QsOff + q8Pos + j].toInt()
                    val sign = if ((signByte and io.github.kotlinmania.llama.ore.kmask_iq2xs[j].toInt()) != 0) -1 else 1
                    sumi2 += q8val * gridByte * sign
                }
                q8Pos += 8
            }
            bsum += ls1 * sumi1 + ls2 * sumi2
            qsPos += 4
            signsPos += 4
        }
        sumf += d * bsum
    }
    s[sOffset] = 0.125f * sumf
}

/**
 * Port of `ggml_vec_dot_iq3_xxs_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq3_xxs_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockIQ3XXS.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockIQ3XXS: d(2) + qs[3*QK_K/8](96)
    // First QK_K/4 = 64 bytes are quant indices (q3), rest is gas (scale+sign aux)
    val xQsOff = 2
    val xGasOff = 2 + io.github.kotlinmania.llama.ore.QK_K / 4  // 66

    val q8dOff = 0
    val q8QsOff = 4

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))

        var q3Pos = xOff + xQsOff
        var gasPos = xOff + xGasOff
        var q8Pos = 0

        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val aux32 = vx.readUIntLE(gasPos)
            gasPos += 4

            val ls = (2u * (aux32 shr 28) + 1u).toInt()
            var sumi = 0
            for (l in 0 until 4) {
                val grid1Val = io.github.kotlinmania.llama.ore.iq3xxs_grid[vx[q3Pos + 2 * l + 0].toInt() and 0xFF]
                val grid2Val = io.github.kotlinmania.llama.ore.iq3xxs_grid[vx[q3Pos + 2 * l + 1].toInt() and 0xFF]
                val signs = io.github.kotlinmania.llama.ore.ksigns_iq2xs[((aux32 shr (7 * l)) and 127u).toInt()]
                for (j in 0 until 4) {
                    val g1byte = ((grid1Val shr (j * 8)) and 0xFFu).toInt()
                    val g2byte = ((grid2Val shr (j * 8)) and 0xFFu).toInt()
                    val sign0 = if ((signs and io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 0]) != 0.toUByte()) -1 else 1
                    val sign1 = if ((signs and io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 4]) != 0.toUByte()) -1 else 1
                    sumi += g1byte * vy[yOff + q8QsOff + q8Pos + j + 0].toInt() * sign0
                    sumi += g2byte * vy[yOff + q8QsOff + q8Pos + j + 4].toInt() * sign1
                }
                q8Pos += 8
            }
            q3Pos += 8
            bsum += sumi * ls
        }
        sumf += d * bsum
    }
    s[sOffset] = 0.25f * sumf
}

/**
 * Port of `ggml_vec_dot_iq3_s_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq3_s_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockIQ3S.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockIQ3S: d(2) + qs[QK_K/4](64) + qh[QK_K/32](8) + signs[QK_K/8](32) + scales[IQ3S_N_SCALE](4)
    val xQsOff = 2
    val xQhOff = 2 + io.github.kotlinmania.llama.ore.QK_K / 4                       // 66
    val xSignsOff = 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 32        // 74
    val xScalesOff = 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 32 + io.github.kotlinmania.llama.ore.QK_K / 8  // 106

    val q8dOff = 0
    val q8QsOff = 4

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff))

        var qsPos = xOff + xQsOff
        var signsPos = xOff + xSignsOff
        var q8Pos = 0

        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32 step 2) {
            val scByte = vx[xOff + xScalesOff + ib32 / 2].toInt() and 0xFF
            val ls1 = 2 * (scByte and 0xf) + 1
            val ls2 = 2 * (scByte ushr 4) + 1

            val qh0 = vx[xOff + xQhOff + ib32 + 0].toInt() and 0xFF

            var sumi = 0
            for (l in 0 until 4) {
                val gridIdx1 = (vx[qsPos + 2 * l + 0].toInt() and 0xFF) or (((qh0 shl (8 - 2 * l)) and 256))
                val gridIdx2 = (vx[qsPos + 2 * l + 1].toInt() and 0xFF) or (((qh0 shl (7 - 2 * l)) and 256))
                val grid1Val = io.github.kotlinmania.llama.ore.iq3s_grid[gridIdx1]
                val grid2Val = io.github.kotlinmania.llama.ore.iq3s_grid[gridIdx2]
                val signByte = vx[signsPos + l].toInt() and 0xFF
                for (j in 0 until 4) {
                    val g1byte = ((grid1Val shr (j * 8)) and 0xFFu).toInt()
                    val g2byte = ((grid2Val shr (j * 8)) and 0xFFu).toInt()
                    val sign0 = if ((signByte and io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 0].toInt()) != 0) -1 else 1
                    val sign1 = if ((signByte and io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 4].toInt()) != 0) -1 else 1
                    sumi += g1byte * vy[yOff + q8QsOff + q8Pos + j + 0].toInt() * sign0
                    sumi += g2byte * vy[yOff + q8QsOff + q8Pos + j + 4].toInt() * sign1
                }
                q8Pos += 8
            }
            qsPos += 8
            signsPos += 4
            bsum += sumi * ls1

            val qh1 = vx[xOff + xQhOff + ib32 + 1].toInt() and 0xFF

            sumi = 0
            for (l in 0 until 4) {
                val gridIdx1 = (vx[qsPos + 2 * l + 0].toInt() and 0xFF) or (((qh1 shl (8 - 2 * l)) and 256))
                val gridIdx2 = (vx[qsPos + 2 * l + 1].toInt() and 0xFF) or (((qh1 shl (7 - 2 * l)) and 256))
                val grid1Val = io.github.kotlinmania.llama.ore.iq3s_grid[gridIdx1]
                val grid2Val = io.github.kotlinmania.llama.ore.iq3s_grid[gridIdx2]
                val signByte = vx[signsPos + l].toInt() and 0xFF
                for (j in 0 until 4) {
                    val g1byte = ((grid1Val shr (j * 8)) and 0xFFu).toInt()
                    val g2byte = ((grid2Val shr (j * 8)) and 0xFFu).toInt()
                    val sign0 = if ((signByte and io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 0].toInt()) != 0) -1 else 1
                    val sign1 = if ((signByte and io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 4].toInt()) != 0) -1 else 1
                    sumi += g1byte * vy[yOff + q8QsOff + q8Pos + j + 0].toInt() * sign0
                    sumi += g2byte * vy[yOff + q8QsOff + q8Pos + j + 4].toInt() * sign1
                }
                q8Pos += 8
            }
            qsPos += 8
            signsPos += 4
            bsum += sumi * ls2
        }
        sumf += d * bsum
    }
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_iq1_s_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq1_s_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockIQ1S.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockIQ1S: d(2) + qs[QK_K/8](32) + qh[QK_K/32] as uint16_t[8](16)
    val xQsOff = 2
    val xQhOff = 2 + io.github.kotlinmania.llama.ore.QK_K / 8  // 34

    val q8dOff = 0
    val q8QsOff = 4
    val q8BsumsOff = 4 + io.github.kotlinmania.llama.ore.QK_K  // 260

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize

        var qsPos = xOff + xQsOff
        var q8Pos = 0

        var sumi = 0
        var sumi1 = 0
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val qhVal = (vx[xOff + xQhOff + ib * 2].toInt() and 0xFF) or
                ((vx[xOff + xQhOff + ib * 2 + 1].toInt() and 0xFF) shl 8)
            val ls = 2 * ((qhVal ushr 12) and 7) + 1
            val delta = if (qhVal and 0x8000 != 0) -1 else 1

            var lsum = 0
            for (l in 0 until 4) {
                val gridIdx = (vx[qsPos + l].toInt() and 0xFF) or ((((qhVal ushr (3 * l)) and 7) shl 8))
                val gridVal = io.github.kotlinmania.llama.ore.iq1s_grid[gridIdx]
                for (j in 0 until 8) {
                    // iq1s_grid is accessed as int8_t — signed byte extraction
                    val gridByte = ((gridVal shr (j * 8)) and 0xFFu).toByte().toInt()
                    lsum += vy[yOff + q8QsOff + q8Pos + j].toInt() * gridByte
                }
                q8Pos += 8
            }
            sumi += ls * lsum
            // bsums are int16_t LE
            val bsum0 = vy.readShortLE(yOff + q8BsumsOff + (2 * ib) * 2).toInt()
            val bsum1 = vy.readShortLE(yOff + q8BsumsOff + (2 * ib + 1) * 2).toInt()
            sumi1 += ls * delta * (bsum0 + bsum1)
            qsPos += 4
        }

        sumf += io.github.kotlinmania.llama.ore.fp16ToF32(vx.readShortLE(xOff)) * Float.fromBits(vy.readIntLE(yOff + q8dOff)) *
            (sumi + io.github.kotlinmania.llama.ore.IQ1S_DELTA * sumi1)
    }
    s[sOffset] = sumf
}

/**
 * Port of `ggml_vec_dot_iq1_m_q8_K_generic` from ggml-cpu/quants.c.
 */
fun ggml_vec_dot_iq1_m_q8_K_generic(
    n: Int, s: FloatArray, sOffset: Int,
    vx: ByteArray, vxOffset: Int,
    vy: ByteArray, vyOffset: Int, nrc: Int
) {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)
    require(nrc == 1)

    val xBlockSize = io.github.kotlinmania.llama.ore.BlockIQ1M.SIZE_BYTES
    val yBlockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES
    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    // BlockIQ1M: qs[QK_K/8](32) + qh[QK_K/16](16) + scales[QK_K/32](8)
    val xQsOff = 0
    val xQhOff = io.github.kotlinmania.llama.ore.QK_K / 8              // 32
    val xScalesOff = io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 16  // 48

    val q8dOff = 0
    val q8QsOff = 4

    val sum1 = IntArray(2)
    val sum2 = IntArray(2)
    val delta = IntArray(4)

    var sumf = 0.0f
    for (i in 0 until nb) {
        val xOff = vxOffset + i * xBlockSize
        val yOff = vyOffset + i * yBlockSize

        var qsPos = xOff + xQsOff
        var qhPos = xOff + xQhOff

        // Read scales as uint16_t[4]
        val sc0 = (vx[xOff + xScalesOff + 0].toInt() and 0xFF) or
            ((vx[xOff + xScalesOff + 1].toInt() and 0xFF) shl 8)
        val sc1 = (vx[xOff + xScalesOff + 2].toInt() and 0xFF) or
            ((vx[xOff + xScalesOff + 3].toInt() and 0xFF) shl 8)
        val sc2 = (vx[xOff + xScalesOff + 4].toInt() and 0xFF) or
            ((vx[xOff + xScalesOff + 5].toInt() and 0xFF) shl 8)
        val sc3 = (vx[xOff + xScalesOff + 6].toInt() and 0xFF) or
            ((vx[xOff + xScalesOff + 7].toInt() and 0xFF) shl 8)
        val sc = intArrayOf(sc0, sc1, sc2, sc3)

        // scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)
        val scaleU16 = (sc[0] ushr 12) or ((sc[1] ushr 8) and 0x00f0) or
            ((sc[2] ushr 4) and 0x0f00) or (sc[3] and 0xf000)

        var q8Pos = 0
        var sumi1 = 0
        var sumi2 = 0

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val qh0 = vx[qhPos + 0].toInt() and 0xFF
            val qh1 = vx[qhPos + 1].toInt() and 0xFF
            delta[0] = if (qh0 and 0x08 != 0) -1 else 1
            delta[1] = if (qh0 and 0x80 != 0) -1 else 1
            delta[2] = if (qh1 and 0x08 != 0) -1 else 1
            delta[3] = if (qh1 and 0x80 != 0) -1 else 1

            sum1[0] = 0; sum1[1] = 0; sum2[0] = 0; sum2[1] = 0
            for (l in 0 until 4) {
                val qhByte = vx[qhPos + l / 2].toInt() and 0xFF
                val gridIdx = (vx[qsPos + l].toInt() and 0xFF) or
                    (((qhByte shl (8 - 4 * (l % 2))) and 0x700))
                val gridVal = io.github.kotlinmania.llama.ore.iq1s_grid[gridIdx]
                var lsum1 = 0
                var lsum2 = 0
                for (j in 0 until 8) {
                    val gridByte = ((gridVal shr (j * 8)) and 0xFFu).toByte().toInt()
                    val q8val = vy[yOff + q8QsOff + q8Pos + j].toInt()
                    lsum1 += q8val * gridByte
                    lsum2 += q8val
                }
                q8Pos += 8
                sum1[l / 2] += lsum1
                sum2[l / 2] += lsum2 * delta[l]
            }

            val ls1 = 2 * ((sc[ib / 2] ushr (6 * (ib % 2) + 0)) and 0x7) + 1
            val ls2 = 2 * ((sc[ib / 2] ushr (6 * (ib % 2) + 3)) and 0x7) + 1

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2
            qsPos += 4
            qhPos += 2
        }

        sumf += io.github.kotlinmania.llama.ore.fp16ToF32(scaleU16.toShort()) * Float.fromBits(vy.readIntLE(yOff + q8dOff)) *
            (sumi1 + io.github.kotlinmania.llama.ore.IQ1M_DELTA * sumi2)
    }
    s[sOffset] = sumf
}

// ---------------------------------------------------------------------------
// Original quantization helper functions (from ggml-quants.c ref implementations)
// ---------------------------------------------------------------------------

internal data class QuantizationStats(val scale: Float, val min: Float)

private const val GROUP_MAX_EPS = 1e-15f

internal fun nearestIntFloat(value: Float): Int {
    if (!value.isFinite()) return 0
    if (value > 4_194_303.0f || value < -4_194_303.0f) {
        return value.roundToInt()
    }
    val adjusted = value + 12_582_912.0f
    val bits = adjusted.toRawBits()
    return (bits and 0x007F_FFFF) - 0x0040_0000
}

private fun FloatArray.weightAt(offset: Int, index: Int): Float = this[offset + index]

private fun qkx3Weight(
    values: FloatArray,
    valuesOffset: Int,
    weights: FloatArray?,
    weightsOffset: Int,
    index: Int
): Float {
    return weights?.weightAt(weightsOffset, index) ?: run {
        val xi = values[valuesOffset + index]
        xi * xi
    }
}

private fun computeRmseWeight(
    values: FloatArray,
    valuesOffset: Int,
    quantWeights: FloatArray?,
    weightsOffset: Int,
    index: Int,
    rmseType: Int
): Float {
    quantWeights?.let { return it[weightsOffset + index] }
    val xi = values[valuesOffset + index]
    return when (rmseType) {
        1 -> xi * xi
        2 -> 1f
        3 -> abs(xi)
        else -> sqrt(abs(xi))
    }
}

internal fun makeQ3Quants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    doRmse: Boolean
): Float {
    var maxValue = 0f
    var absMax = 0f
    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        val ax = abs(xi)
        if (ax > absMax) {
            absMax = ax
            maxValue = xi
        }
    }

    if (absMax < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
        for (i in 0 until n) dest[destOffset + i] = 0
        return 0f
    }

    val codes = IntArray(n)
    var iscale = -nmax.toFloat() / maxValue

    if (doRmse) {
        var sumLx = 0f
        var sumL2 = 0f
        for (i in 0 until n) {
            val xi = values[valuesOffset + i]
            var l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * xi)
            l = l.coerceIn(-nmax, nmax - 1)
            codes[i] = l
            val w = xi * xi
            sumLx += w * xi * l
            sumL2 += w * l * l
        }

        repeat(5) {
            var changed = 0
            for (i in 0 until n) {
                val xi = values[valuesOffset + i]
                val w = xi * xi
                val current = codes[i]
                val slx = sumLx - w * xi * current
                if (slx > 0f) {
                    val sl2 = sumL2 - w * current * current
                    var newCode = io.github.kotlinmania.llama.ore.nearestIntFloat(xi * sl2 / slx)
                    newCode = newCode.coerceIn(-nmax, nmax - 1)
                    if (newCode != current) {
                        val candidateSlx = slx + w * xi * newCode
                        val candidateSl2 = sl2 + w * newCode * newCode
                        if (candidateSl2 > 0f && candidateSlx * candidateSlx * sumL2 > sumLx * sumLx * candidateSl2) {
                            codes[i] = newCode
                            sumLx = candidateSlx
                            sumL2 = candidateSl2
                            changed++
                        }
                    }
                }
            }
            if (changed == 0) return@repeat
        }

        for (i in 0 until n) {
            dest[destOffset + i] = (codes[i] + nmax).toByte()
        }
        return if (sumL2 > 0f) sumLx / sumL2 else 0f
    }

    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        var l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * xi)
        l = l.coerceIn(-nmax, nmax - 1)
        dest[destOffset + i] = (l + nmax).toByte()
    }
    return 1f / iscale
}

internal fun makeQKX2Quants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    weights: FloatArray,
    weightsOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    mins: FloatArray,
    minsIndex: Int,
    aux: ByteArray,
    auxOffset: Int,
    rmin: Float,
    rdelta: Float,
    nstep: Int,
    useMad: Boolean
): io.github.kotlinmania.llama.ore.QuantizationStats {
    var minVal = values[valuesOffset]
    var maxVal = minVal
    var sumW = weights.weightAt(weightsOffset, 0)
    var sumX = sumW * values[valuesOffset]
    for (i in 1 until n) {
        val xi = values[valuesOffset + i]
        if (xi < minVal) minVal = xi
        if (xi > maxVal) maxVal = xi
        val w = weights.weightAt(weightsOffset, i)
        sumW = sumW + w
        sumX = sumX + w * xi
    }
    if (minVal > 0f) minVal = 0f
    if (maxVal == minVal) {
        for (i in 0 until n) dest[destOffset + i] = 0
        mins[minsIndex] = -minVal
        return io.github.kotlinmania.llama.ore.QuantizationStats(scale = 0f, min = -minVal)
    }

    val maxFixed = maxVal
    var currentMin = minVal
    var iscale = nmax.toFloat() / (maxFixed - currentMin)
    var scale = 1f / iscale
    var bestMetric = 0f
    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        val l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * (xi - currentMin))
            .coerceIn(0, nmax)
        dest[destOffset + i] = l.toByte()
        var diff = scale * l + currentMin - xi
        diff = if (useMad) abs(diff) else diff * diff
        val w = weights.weightAt(weightsOffset, i)
        bestMetric += w * diff
    }
    if (nstep < 1) {
        mins[minsIndex] = -currentMin
        return io.github.kotlinmania.llama.ore.QuantizationStats(scale = scale, min = -currentMin)
    }

    for (step in 0..nstep) {
        val stepIscale = (rmin + rdelta * step + nmax) / (maxFixed - currentMin)
        var sumL = 0f
        var sumL2 = 0f
        var sumXL = 0f
        for (i in 0 until n) {
            val xi = values[valuesOffset + i]
            var l = io.github.kotlinmania.llama.ore.nearestIntFloat(stepIscale * (xi - currentMin))
            l = l.coerceIn(0, nmax)
            aux[auxOffset + i] = l.toByte()
            val w = weights.weightAt(weightsOffset, i)
            val lf = l.toFloat()
            sumL = sumL + (w * lf)
            sumL2 = sumL2 + (w * lf * lf)
            sumXL = sumXL + (w * lf * xi)
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D > 0f) {
            var scaleCand = (sumW * sumXL - sumX * sumL) / D
            var minCand = (sumL2 * sumX - sumL * sumXL) / D
            if (minCand > 0f) {
                minCand = 0f
                scaleCand = sumXL / sumL2
            }
            var metric = 0f
            for (i in 0 until n) {
                val xi = values[valuesOffset + i]
                val li = (aux[auxOffset + i].toInt() and 0xFF).toFloat()
                var diff = scaleCand * li + minCand - xi
                diff = if (useMad) abs(diff) else diff * diff
                val w = weights.weightAt(weightsOffset, i)
                metric = metric + w * diff
            }
            if (metric < bestMetric) {
                for (i in 0 until n) dest[destOffset + i] = aux[auxOffset + i]
                bestMetric = metric
                scale = scaleCand
                currentMin = minCand
            }
        }
    }

    mins[minsIndex] = -currentMin
    return io.github.kotlinmania.llama.ore.QuantizationStats(scale = scale, min = -currentMin)
}

internal fun makeQKX3Quants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    weights: FloatArray?,
    weightsOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    mins: FloatArray,
    minsIndex: Int,
    aux: ByteArray,
    auxOffset: Int,
    rmin: Float,
    rdelta: Float,
    nstep: Int,
    useMad: Boolean
): io.github.kotlinmania.llama.ore.QuantizationStats {
    var minVal = values[valuesOffset]
    var maxVal = minVal
    var sumW =
        io.github.kotlinmania.llama.ore.qkx3Weight(values, valuesOffset, weights, weightsOffset, 0)
    var sumX = sumW * values[valuesOffset]
    for (i in 1 until n) {
        val xi = values[valuesOffset + i]
        if (xi < minVal) minVal = xi
        if (xi > maxVal) maxVal = xi
        val w = io.github.kotlinmania.llama.ore.qkx3Weight(
            values,
            valuesOffset,
            weights,
            weightsOffset,
            i
        )
        sumW = sumW + w
        sumX = sumX + w * xi
    }
    if (minVal > 0f) minVal = 0f
    if (maxVal <= minVal) {
        for (i in 0 until n) dest[destOffset + i] = 0
        mins[minsIndex] = -minVal
        return io.github.kotlinmania.llama.ore.QuantizationStats(scale = 0f, min = -minVal)
    }

    val maxFixed = maxVal
    var currentMin = minVal
    var iscale = nmax.toFloat() / (maxFixed - currentMin)
    var scale = 1f / iscale
    var bestMetric = 0f
    for (i in 0 until n) {
        val xi = values[valuesOffset + i]
        val l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * (xi - currentMin))
            .coerceIn(0, nmax)
        dest[destOffset + i] = l.toByte()
        var diff = scale * l + currentMin - xi
        diff = if (useMad) abs(diff) else diff * diff
        val w = io.github.kotlinmania.llama.ore.qkx3Weight(
            values,
            valuesOffset,
            weights,
            weightsOffset,
            i
        )
        bestMetric += w * diff
    }
    if (nstep < 1) {
        mins[minsIndex] = -currentMin
        return io.github.kotlinmania.llama.ore.QuantizationStats(scale = scale, min = -currentMin)
    }

    for (step in 0..nstep) {
        val stepIscale = (rmin + rdelta * step + nmax) / (maxFixed - currentMin)
        var sumL = 0f
        var sumL2 = 0f
        var sumXL = 0f
        for (i in 0 until n) {
            val xi = values[valuesOffset + i]
            var l = io.github.kotlinmania.llama.ore.nearestIntFloat(stepIscale * (xi - currentMin))
            l = l.coerceIn(0, nmax)
            aux[auxOffset + i] = l.toByte()
            val w = io.github.kotlinmania.llama.ore.qkx3Weight(
                values,
                valuesOffset,
                weights,
                weightsOffset,
                i
            )
            val lf = l.toFloat()
            sumL = sumL + (w * lf)
            sumL2 = sumL2 + (w * lf * lf)
            sumXL = sumXL + (w * lf * xi)
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D > 0f) {
            var scaleCand = (sumW * sumXL - sumX * sumL) / D
            var minCand = (sumL2 * sumX - sumL * sumXL) / D
            if (minCand > 0f) {
                minCand = 0f
                scaleCand = sumXL / sumL2
            }
            var metric = 0f
            for (i in 0 until n) {
                val xi = values[valuesOffset + i]
                val li = (aux[auxOffset + i].toInt() and 0xFF).toFloat()
                var diff = scaleCand * li + minCand - xi
                diff = if (useMad) abs(diff) else diff * diff
                val w = io.github.kotlinmania.llama.ore.qkx3Weight(
                    values,
                    valuesOffset,
                    weights,
                    weightsOffset,
                    i
                )
                metric = metric + w * diff
            }
            if (metric < bestMetric) {
                for (i in 0 until n) dest[destOffset + i] = aux[auxOffset + i]
                bestMetric = metric
                scale = scaleCand
                currentMin = minCand
            }
        }
    }

    mins[minsIndex] = -currentMin
    return io.github.kotlinmania.llama.ore.QuantizationStats(scale = scale, min = -currentMin)
}

internal fun makeQPQuants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    quantWeights: FloatArray,
    weightsOffset: Int
): Float {
    var maxVal = 0f
    for (i in 0 until n) {
        maxVal = max(maxVal, values[valuesOffset + i])
    }
    if (maxVal == 0f) {
        for (i in 0 until n) dest[destOffset + i] = 0
        return 0f
    }
    var iscale = nmax.toFloat() / maxVal
    val tempCodes = IntArray(n)
    for (i in 0 until n) {
        val l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * values[valuesOffset + i])
            .coerceIn(0, nmax)
        dest[destOffset + i] = l.toByte()
        tempCodes[i] = l
    }
    var scale = 1f / iscale
    var bestMse = 0f
    for (i in 0 until n) {
        val diff = values[valuesOffset + i] - scale * tempCodes[i]
        bestMse += quantWeights[weightsOffset + i] * diff * diff
    }
    for (adjust in -4..4) {
        if (adjust == 0) continue
        val candidateIscale = (0.1f * adjust + nmax) / maxVal
        val candidateScale = 1f / candidateIscale
        var mse = 0f
        for (i in 0 until n) {
            var l =
                io.github.kotlinmania.llama.ore.nearestIntFloat(candidateIscale * values[valuesOffset + i])
            l = min(nmax, l)
            val diff = values[valuesOffset + i] - candidateScale * l
            mse += quantWeights[weightsOffset + i] * diff * diff
        }
        if (mse < bestMse) {
            bestMse = mse
            iscale = candidateIscale
        }
    }

    var sumLx = 0f
    var sumL2 = 0f
    for (i in 0 until n) {
        var l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * values[valuesOffset + i])
        l = min(nmax, l)
        dest[destOffset + i] = l.toByte()
        val w = quantWeights[weightsOffset + i]
        sumLx += w * values[valuesOffset + i] * l
        sumL2 += w * l * l
    }
    for (iteration in 0 until 5) {
        var changed = 0
        for (i in 0 until n) {
            val w = quantWeights[weightsOffset + i]
            val xi = values[valuesOffset + i]
            val current = dest[destOffset + i].toInt() and 0xFF
            var slx = sumLx - w * xi * current
            var sl2 = sumL2 - w * current * current
            if (slx > 0f && sl2 > 0f) {
                var newL = io.github.kotlinmania.llama.ore.nearestIntFloat(xi * sl2 / slx)
                newL = min(nmax, newL)
                if (newL != current) {
                    slx += w * xi * newL
                    sl2 += w * newL * newL
                    if (slx * slx * sumL2 > sumLx * sumLx * sl2) {
                        dest[destOffset + i] = newL.toByte()
                        sumLx = slx
                        sumL2 = sl2
                        changed++
                    }
                }
            }
        }
        if (changed == 0) break
    }
    return if (sumL2 > 0f) sumLx / sumL2 else 0f
}

internal fun makeQXQuants(
    n: Int,
    nmax: Int,
    values: FloatArray,
    valuesOffset: Int,
    dest: ByteArray,
    destOffset: Int,
    rmseTypeInput: Int,
    quantWeights: FloatArray?,
    weightsOffset: Int
): Float {
    var maxValue = 0f
    var absMax = 0f
    for (i in 0 until n) {
        val ax = abs(values[valuesOffset + i])
        if (ax > absMax) {
            absMax = ax
            maxValue = values[valuesOffset + i]
        }
    }
    if (absMax < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
        for (i in 0 until n) dest[destOffset + i] = 0
        return 0f
    }
    var iscale = -nmax.toFloat() / maxValue
    var rmseType = rmseTypeInput
    if (rmseType == 0) {
        for (i in 0 until n) {
            var l =
                io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * values[valuesOffset + i])
            l = max(-nmax, min(nmax - 1, l))
            dest[destOffset + i] = (nmax + l).toByte()
        }
        return 1f / iscale
    }
    var returnEarly = false
    if (rmseType < 0) {
        rmseType = -rmseType
        returnEarly = true
    }

    var sumLx = 0f
    var sumL2 = 0f
    for (i in 0 until n) {
        var l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * values[valuesOffset + i])
        l = max(-nmax, min(nmax - 1, l))
        dest[destOffset + i] = (nmax + l).toByte()
        val w = io.github.kotlinmania.llama.ore.computeRmseWeight(
            values,
            valuesOffset,
            quantWeights,
            weightsOffset,
            i,
            rmseType
        )
        sumLx += w * values[valuesOffset + i] * l
        sumL2 += w * l * l
    }
    var scale = if (sumL2 > 0f) sumLx / sumL2 else 0f
    if (returnEarly) {
        return if (sumL2 > 0f) 0.5f * (scale + 1f / iscale) else 1f / iscale
    }
    var best = scale * sumLx

    for (adjust in -9..9) {
        if (adjust == 0) continue
        val candidateIscale = -(nmax + 0.1f * adjust) / maxValue
        var candidateSumLx = 0f
        var candidateSumL2 = 0f
        for (i in 0 until n) {
            val l = io.github.kotlinmania.llama.ore.nearestIntFloat(candidateIscale * values[valuesOffset + i])
                .coerceIn(-nmax, nmax - 1)
            val w = io.github.kotlinmania.llama.ore.computeRmseWeight(
                values,
                valuesOffset,
                quantWeights,
                weightsOffset,
                i,
                rmseType
            )
            candidateSumLx += w * values[valuesOffset + i] * l
            candidateSumL2 += w * l * l
        }
        if (candidateSumL2 > 0f && candidateSumLx * candidateSumLx > best * candidateSumL2) {
            iscale = candidateIscale
            scale = candidateSumLx / candidateSumL2
            best = scale * candidateSumLx
            for (i in 0 until n) {
                val l = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * values[valuesOffset + i])
                    .coerceIn(-nmax, nmax - 1)
                dest[destOffset + i] = (nmax + l).toByte()
            }
        }
    }

    return scale
}
