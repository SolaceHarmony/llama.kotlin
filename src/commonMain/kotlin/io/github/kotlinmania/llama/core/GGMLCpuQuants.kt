// port-lint: source ggml/src/ggml-cpu/quants.c

package io.github.kotlinmania.llama.ore

/**
 * CPU-specific quantization / dequantization kernels ported from `ggml-cpu/quants.c`.
 *
 * This file contains:
 * - `quantizeRow*` — row-level quantization for every quant type
 * - `vecDot*` — quantized dot-product routines (scalar fallbacks)
 * - Helper conversions (E8M0, UE4M3, etc.)
 *
 * SIMD-optimised paths are **not** ported; every function here corresponds to the
 * `*_generic` (scalar) path in the C source. Block type definitions live in
 * [GGMLCommon.kt]; fp16↔fp32 helpers live in [NumericConversions.kt].
 *
 * Numeric helpers (E8M0, UE4M3 conversions) are at the bottom of the file.
 */

private const val GROUP_MAX_EPS = 1e-15f

// ════════════════════════════════════════════════════════════════════════════════
//  Quantize-row wrappers (delegate to _ref once those are ported in GGMLQuantsRef.kt)
// ════════════════════════════════════════════════════════════════════════════════

fun quantizeRowQ10(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q1_0_ref(x, 0, y, 0, k)
}

fun quantizeRowQ40(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q4_0_ref(x, 0, y, 0, k)
}

fun quantizeRowQ41(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q4_1_ref(x, 0, y, 0, k)
}

fun quantizeRowQ50(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q5_0_ref(x, 0, y, 0, k)
}

fun quantizeRowQ51(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q5_1_ref(x, 0, y, 0, k)
}

fun quantizeRowQ80Generic(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_0_ref(x, 0, y, 0, k)
}

fun quantizeRowQ81Generic(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_1_ref(x, 0, y, 0, k)
}

fun quantizeRowMxfp4(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_mxfp4_ref(x, 0, y, 0, k)
}

fun quantizeRowNvfp4(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_nvfp4_ref(x, 0, y, 0, k)
}

fun quantizeRowQ2K(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q2_K_ref(x, 0, y, 0, k)
}

fun quantizeRowQ3K(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q3_K_ref(x, 0, y, 0, k)
}

fun quantizeRowQ4K(x: FloatArray, y: ByteArray, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L) { "k ($k) must be a multiple of QK_K (${io.github.kotlinmania.llama.ore.QK_K})" }
    io.github.kotlinmania.llama.ore.quantize_row_q4_K_ref(x, 0, y, 0, k)
}

fun quantizeRowQ5K(x: FloatArray, y: ByteArray, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L) { "k ($k) must be a multiple of QK_K (${io.github.kotlinmania.llama.ore.QK_K})" }
    io.github.kotlinmania.llama.ore.quantize_row_q5_K_ref(x, 0, y, 0, k)
}

fun quantizeRowQ6K(x: FloatArray, y: ByteArray, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L) { "k ($k) must be a multiple of QK_K (${io.github.kotlinmania.llama.ore.QK_K})" }
    io.github.kotlinmania.llama.ore.quantize_row_q6_K_ref(x, 0, y, 0, k)
}

fun quantizeRowQ8KGeneric(x: FloatArray, y: ByteArray, k: Long) {
    io.github.kotlinmania.llama.ore.quantize_row_q8_K_ref(x, 0, y, 0, k)
}

fun quantizeRowTq10(x: FloatArray, y: ByteArray, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L) { "k ($k) must be a multiple of QK_K (${io.github.kotlinmania.llama.ore.QK_K})" }
    io.github.kotlinmania.llama.ore.quantize_row_tq1_0_ref(x, 0, y, 0, k)
}

fun quantizeRowTq20(x: FloatArray, y: ByteArray, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L) { "k ($k) must be a multiple of QK_K (${io.github.kotlinmania.llama.ore.QK_K})" }
    io.github.kotlinmania.llama.ore.quantize_row_tq2_0_ref(x, 0, y, 0, k)
}

fun quantizeRowIq4Nl(x: FloatArray, y: ByteArray, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK4_NL == 0L) { "k ($k) must be a multiple of QK4_NL (${io.github.kotlinmania.llama.ore.QK4_NL})" }
    io.github.kotlinmania.llama.ore.quantize_row_iq4_nl_ref(x, 0, y, 0, k)
}

fun quantizeRowIq4Xs(x: FloatArray, y: ByteArray, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L) { "k ($k) must be a multiple of QK_K (${io.github.kotlinmania.llama.ore.QK_K})" }
    io.github.kotlinmania.llama.ore.quantize_iq4_xs(x, y, 1L, k, null)
}

// ════════════════════════════════════════════════════════════════════════════════
//  Scalar dot-product kernels (generic / fallback paths)
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Q1_0 × Q8_0 dot product (scalar).
 *
 * Each Q1_0 block covers [io.github.kotlinmania.llama.ore.QK1_0] = 128 elements, which spans 4 Q8_0 blocks
 * of 32 elements each. Each bit in `x[i].qs` maps to ±1.
 *
 * @param n   number of elements (must be divisible by [io.github.kotlinmania.llama.ore.QK1_0])
 * @param vx  Q1_0 blocks
 * @param vy  Q8_0 blocks (4× as many as vx)
 * @return    the dot product
 */
fun ggmlVecDotQ10Q80Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ1_0>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>): Float {
    val qk = io.github.kotlinmania.llama.ore.QK1_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (i in 0 until nb) {
        val d0 = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d)
        var sumi = 0.0f

        for (k in 0 until 4) {
            val d1 = io.github.kotlinmania.llama.ore.halfToFloat(vy[i * 4 + k].d)
            var sumiBlock = 0

            for (j in 0 until io.github.kotlinmania.llama.ore.QK8_0) {
                val bitIndex = k * io.github.kotlinmania.llama.ore.QK8_0 + j
                val byteIndex = bitIndex / 8
                val bitOffset = bitIndex % 8

                val xi = if (((vx[i].qs[byteIndex].toInt() ushr bitOffset) and 1) != 0) 1 else -1
                sumiBlock += xi * vy[i * 4 + k].qs[j].toInt()
            }
            sumi += d1 * sumiBlock
        }
        sumf += d0 * sumi
    }
    return sumf
}

/**
 * Q4_0 × Q8_0 dot product (scalar).
 *
 * Each Q4_0 block stores 32 four-bit weights packed into 16 bytes (two per byte,
 * offset by 8). The companion Q8_0 block has 32 signed-byte weights.
 *
 * @param n   number of elements (must be divisible by [io.github.kotlinmania.llama.ore.QK8_0])
 * @param vx  Q4_0 blocks
 * @param vy  Q8_0 blocks
 * @return    the dot product
 */
fun ggmlVecDotQ40Q80Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ4_0>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>): Float {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val v0 = (vx[ib].qs[j].toInt() and 0x0F) - 8
            val v1 = ((vx[ib].qs[j].toInt() and 0xFF) ushr 4) - 8

            sumi0 += v0 * vy[ib].qs[j].toInt()
            sumi1 += v1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += sumi * io.github.kotlinmania.llama.ore.halfToFloat(vx[ib].d) * io.github.kotlinmania.llama.ore.halfToFloat(
            vy[ib].d
        )
    }
    return sumf
}

/**
 * Q4_1 × Q8_1 dot product (scalar).
 *
 * Like Q4_0 but with an additional minimum (`m`) and sum (`s`) per block.
 * Nibbles are unsigned (no −8 offset).
 */
fun ggmlVecDotQ41Q81Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ4_1>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_1>): Float {
    val qk = io.github.kotlinmania.llama.ore.QK8_1
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val v0 = vx[ib].qs[j].toInt() and 0x0F
            val v1 = (vx[ib].qs[j].toInt() and 0xFF) ushr 4

            sumi0 += v0 * vy[ib].qs[j].toInt()
            sumi1 += v1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += (io.github.kotlinmania.llama.ore.halfToFloat(vx[ib].d) * io.github.kotlinmania.llama.ore.halfToFloat(
            vy[ib].d
        )) * sumi +
                io.github.kotlinmania.llama.ore.halfToFloat(vx[ib].m) * io.github.kotlinmania.llama.ore.halfToFloat(
            vy[ib].s
        )
    }
    return sumf
}

/**
 * Q5_0 × Q8_0 dot product (scalar).
 *
 * Q5_0 uses 4 low bits in `qs` and a 5th (high) bit in `qh` (packed as a uint32).
 */
fun ggmlVecDotQ50Q80Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ5_0>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>): Float {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        // Reconstruct uint32 qh from 4 bytes (little-endian)
        val qh: Int = (vx[ib].qh[0].toInt() and 0xFF) or
                ((vx[ib].qh[1].toInt() and 0xFF) shl 8) or
                ((vx[ib].qh[2].toInt() and 0xFF) shl 16) or
                ((vx[ib].qh[3].toInt() and 0xFF) shl 24)

        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val xh0 = ((qh and (1 shl (j + 0))) ushr (j + 0)) shl 4
            val xh1 = ((qh and (1 shl (j + 16))) ushr (j + 12))

            val x0 = (((vx[ib].qs[j].toInt() and 0x0F) or xh0) - 16).toByte().toInt()
            val x1 = ((((vx[ib].qs[j].toInt() and 0xFF) ushr 4) or xh1) - 16).toByte().toInt()

            sumi0 += x0 * vy[ib].qs[j].toInt()
            sumi1 += x1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += (io.github.kotlinmania.llama.ore.halfToFloat(vx[ib].d) * io.github.kotlinmania.llama.ore.halfToFloat(
            vy[ib].d
        )) * sumi
    }
    return sumf
}

/**
 * Q5_1 × Q8_1 dot product (scalar).
 *
 * Like Q5_0 but with separate min/sum fields and unsigned nibbles.
 */
fun ggmlVecDotQ51Q81Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ5_1>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_1>): Float {
    val qk = io.github.kotlinmania.llama.ore.QK8_1
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        val qh: Int = (vx[ib].qh[0].toInt() and 0xFF) or
                ((vx[ib].qh[1].toInt() and 0xFF) shl 8) or
                ((vx[ib].qh[2].toInt() and 0xFF) shl 16) or
                ((vx[ib].qh[3].toInt() and 0xFF) shl 24)

        var sumi0 = 0
        var sumi1 = 0
        for (j in 0 until qk / 2) {
            val xh0 = ((qh ushr (j + 0)) shl 4) and 0x10
            val xh1 = ((qh ushr (j + 12))) and 0x10

            val x0 = (vx[ib].qs[j].toInt() and 0x0F) or xh0
            val x1 = ((vx[ib].qs[j].toInt() and 0xFF) ushr 4) or xh1

            sumi0 += x0 * vy[ib].qs[j].toInt()
            sumi1 += x1 * vy[ib].qs[j + qk / 2].toInt()
        }
        val sumi = sumi0 + sumi1
        sumf += (io.github.kotlinmania.llama.ore.halfToFloat(vx[ib].d) * io.github.kotlinmania.llama.ore.halfToFloat(
            vy[ib].d
        )) * sumi +
                io.github.kotlinmania.llama.ore.halfToFloat(vx[ib].m) * io.github.kotlinmania.llama.ore.halfToFloat(
            vy[ib].s
        )
    }
    return sumf
}

/**
 * Q8_0 × Q8_0 dot product (scalar).
 *
 * Straightforward: multiply corresponding 8-bit quants and accumulate,
 * then scale by the product of both block deltas.
 */
fun ggmlVecDotQ80Q80Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>): Float {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    val nb = n / qk
    require(n % qk == 0) { "n ($n) must be divisible by $qk" }

    var sumf = 0.0f
    for (ib in 0 until nb) {
        var sumi = 0
        for (j in 0 until qk) {
            sumi += vx[ib].qs[j].toInt() * vy[ib].qs[j].toInt()
        }
        sumf += sumi * (io.github.kotlinmania.llama.ore.halfToFloat(vx[ib].d) * io.github.kotlinmania.llama.ore.halfToFloat(
            vy[ib].d
        ))
    }
    return sumf
}

/**
 * MXFP4 × Q8_0 dot product (scalar).
 *
 * Uses the [io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesMXFP4] lookup to dequantize 4-bit mantissa values.
 */
fun ggmlVecDotMxfp4Q80Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockMXFP4>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_MXFP4 == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_MXFP4}" }
    val nb = n / io.github.kotlinmania.llama.ore.QK_MXFP4

    var sumf = 0.0f
    for (ib in 0 until nb) {
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vy[ib].d) * io.github.kotlinmania.llama.ore.ggmlE8M0ToFp32Half(
            vx[ib].e
        )
        var sumi1 = 0
        var sumi2 = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_MXFP4 / 2) {
            sumi1 += vy[ib].qs[j].toInt() *
                    io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesMXFP4[vx[ib].qs[j].toInt() and 0xF].toInt()
            sumi2 += vy[ib].qs[j + io.github.kotlinmania.llama.ore.QK_MXFP4 / 2].toInt() *
                    io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesMXFP4[(vx[ib].qs[j].toInt() and 0xFF) ushr 4].toInt()
        }
        sumf += d * (sumi1 + sumi2)
    }
    return sumf
}

/**
 * NVFP4 × Q8_0 dot product (scalar).
 *
 * NVFP4 super-blocks have 64 elements = 4 sub-blocks of 16, spanning 2 Q8_0 blocks.
 */
fun ggmlVecDotNvfp4Q80Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockNVFP4>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_NVFP4 == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_NVFP4}" }
    val nb = n / io.github.kotlinmania.llama.ore.QK_NVFP4

    var sumf = 0.0f
    for (ib in 0 until nb) {
        for (sIdx in 0 until 4) {
            val d = io.github.kotlinmania.llama.ore.ggmlUE4M3ToFp32(vx[ib].d[sIdx])
            val q8Block = sIdx / 2
            val q8Off = (sIdx % 2) * io.github.kotlinmania.llama.ore.QK_NVFP4_SUB
            val dy = io.github.kotlinmania.llama.ore.halfToFloat(vy[2 * ib + q8Block].d)

            var sumiLo = 0
            var sumiHi = 0
            for (j in 0 until io.github.kotlinmania.llama.ore.QK_NVFP4_SUB / 2) {
                val qv = vx[ib].qs[sIdx * (io.github.kotlinmania.llama.ore.QK_NVFP4_SUB / 2) + j].toInt() and 0xFF
                sumiLo += vy[2 * ib + q8Block].qs[q8Off + j].toInt() *
                        io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesMXFP4[qv and 0xF].toInt()
                sumiHi += vy[2 * ib + q8Block].qs[q8Off + j + io.github.kotlinmania.llama.ore.QK_NVFP4_SUB / 2].toInt() *
                        io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesMXFP4[qv ushr 4].toInt()
            }
            sumf += dy * d * (sumiLo + sumiHi)
        }
    }
    return sumf
}

// ── Ternary dot products ────────────────────────────────────────────────────────

/**
 * TQ1_0 × Q8_K dot product (scalar).
 *
 * Ternary base-3 packed quants. 5 ternary values per byte in `qs`,
 * 4 values per byte in `qh`.
 */
fun ggmlVecDotTq10Q8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockTQ1_0>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    val nb = n / io.github.kotlinmania.llama.ore.QK_K
    val pow3 = intArrayOf(1, 3, 9, 27, 81, 243)

    var sumf = 0.0f
    for (i in 0 until nb) {
        var sum = 0
        val qsSize = vx[i].qs.size

        // Main loop over qs in chunks of 32
        val qsAligned = qsSize - (qsSize % 32)
        var j = 0
        while (j < qsAligned) {
            for (l in 0 until 5) {
                for (m in 0 until 32) {
                    val q = ((vx[i].qs[j + m].toInt() and 0xFF) * pow3[l]) and 0xFF
                    val xi = ((q * 3) ushr 8)
                    sum += (xi - 1) * vy[i].qs[j * 5 + l * 32 + m].toInt()
                }
            }
            j += 32
        }
        // Remainder (chunks of 16)
        while (j < qsSize) {
            for (l in 0 until 5) {
                for (m in 0 until 16) {
                    val q = ((vx[i].qs[j + m].toInt() and 0xFF) * pow3[l]) and 0xFF
                    val xi = ((q * 3) ushr 8)
                    sum += (xi - 1) * vy[i].qs[j * 5 + l * 16 + m].toInt()
                }
            }
            j += 16
        }

        // qh part
        val qhSize = vx[i].qh.size
        for (l in 0 until 4) {
            for (jj in 0 until qhSize) {
                val q = ((vx[i].qh[jj].toInt() and 0xFF) * pow3[l]) and 0xFF
                val xi = ((q * 3) ushr 8)
                sum += (xi - 1) * vy[i].qs[qsSize * 5 + l * qhSize + jj].toInt()
            }
        }

        sumf += sum.toFloat() * (io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d)
    }
    return sumf
}

/**
 * TQ2_0 × Q8_K dot product (scalar).
 *
 * 2-bit ternary packed: each byte holds 4 ternary values in 2-bit pairs.
 */
fun ggmlVecDotTq20Q8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockTQ2_0>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    val nb = n / io.github.kotlinmania.llama.ore.QK_K
    var sumf = 0.0f

    for (i in 0 until nb) {
        var sumi = 0
        val qsSize = vx[i].qs.size

        var j = 0
        while (j < qsSize) {
            val chunkSize = minOf(32, qsSize - j)
            for (l in 0 until 4) {
                for (k in 0 until chunkSize) {
                    sumi += vy[i].qs[j * 4 + l * 32 + k].toInt() *
                            ((((vx[i].qs[j + k].toInt() and 0xFF) ushr (l * 2)) and 3) - 1)
                }
            }
            j += 32
        }

        val d = vy[i].d * io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d)
        sumf += sumi.toFloat() * d
    }
    return sumf
}

// ── K-quant dot products ────────────────────────────────────────────────────────

/**
 * Q2_K × Q8_K dot product (scalar).
 *
 * Super-block with 4-bit scales/mins in `scales`, 2-bit quants in `qs`.
 */
fun ggmlVecDotQ2KQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ2K>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    val nb = n / io.github.kotlinmania.llama.ore.QK_K
    var sumf = 0.0f

    for (i in 0 until nb) {
        val sc = vx[i].scales

        var summs = 0
        for (j in 0 until 16) {
            summs += vy[i].bsums[j].toInt() * ((sc[j].toInt() and 0xFF) ushr 4)
        }

        val dall = vy[i].d * io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d)
        val dmin = vy[i].d * io.github.kotlinmania.llama.ore.halfToFloat(vx[i].dmin)

        var isum = 0
        var isIdx = 0
        var q2Off = 0
        var q8Off = 0
        for (k in 0 until io.github.kotlinmania.llama.ore.QK_K / 128) {
            var shift = 0
            for (j in 0 until 4) {
                val d1 = sc[isIdx++].toInt() and 0xF
                var isuml = 0
                for (l in 0 until 16) {
                    isuml += vy[i].qs[q8Off + l].toInt() *
                            (((vx[i].qs[q2Off + l].toInt() and 0xFF) ushr shift) and 3)
                }
                isum += d1 * isuml

                val d2 = sc[isIdx++].toInt() and 0xF
                isuml = 0
                for (l in 16 until 32) {
                    isuml += vy[i].qs[q8Off + l].toInt() *
                            (((vx[i].qs[q2Off + l].toInt() and 0xFF) ushr shift) and 3)
                }
                isum += d2 * isuml
                shift += 2
                q8Off += 32
            }
            q2Off += 32
        }
        sumf += dall * isum - dmin * summs
    }
    return sumf
}

/** Q3_K × Q8_K dot product (scalar). */
fun ggmlVecDotQ3KQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ3K>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val kmask1 = 0x03030303
    val kmask2 = 0x0f0f0f0f

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)

    var sumf = 0.0f
    for (i in 0 until nb) {
        var q3Off = 0
        var q8Off = 0
        for (l in 0 until 8) aux32[l] = 0
        var aOff = 0
        var m = 1
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) aux8[aOff + l] = (vx[i].qs[q3Off + l].toInt() and 0xFF) and 3
            for (l in 0 until 32) aux8[aOff + l] -= if ((vx[i].hmask[l].toInt() and 0xFF) and m != 0) 0 else 4
            aOff += 32; m = m shl 1
            for (l in 0 until 32) aux8[aOff + l] = ((vx[i].qs[q3Off + l].toInt() and 0xFF) ushr 2) and 3
            for (l in 0 until 32) aux8[aOff + l] -= if ((vx[i].hmask[l].toInt() and 0xFF) and m != 0) 0 else 4
            aOff += 32; m = m shl 1
            for (l in 0 until 32) aux8[aOff + l] = ((vx[i].qs[q3Off + l].toInt() and 0xFF) ushr 4) and 3
            for (l in 0 until 32) aux8[aOff + l] -= if ((vx[i].hmask[l].toInt() and 0xFF) and m != 0) 0 else 4
            aOff += 32; m = m shl 1
            for (l in 0 until 32) aux8[aOff + l] = ((vx[i].qs[q3Off + l].toInt() and 0xFF) ushr 6) and 3
            for (l in 0 until 32) aux8[aOff + l] -= if ((vx[i].hmask[l].toInt() and 0xFF) and m != 0) 0 else 4
            aOff += 32; m = m shl 1
            q3Off += 32
        }
        aOff = 0

        // memcpy(auxs, x[i].scales, 12) → assemble 3 uint32s from 12 bytes
        val sc = vx[i].scales
        var auxs0 = (sc[0].toInt() and 0xFF) or ((sc[1].toInt() and 0xFF) shl 8) or
                ((sc[2].toInt() and 0xFF) shl 16) or ((sc[3].toInt() and 0xFF) shl 24)
        var auxs1 = (sc[4].toInt() and 0xFF) or ((sc[5].toInt() and 0xFF) shl 8) or
                ((sc[6].toInt() and 0xFF) shl 16) or ((sc[7].toInt() and 0xFF) shl 24)
        val tmp = (sc[8].toInt() and 0xFF) or ((sc[9].toInt() and 0xFF) shl 8) or
                ((sc[10].toInt() and 0xFF) shl 16) or ((sc[11].toInt() and 0xFF) shl 24)
        var auxs2 = ((auxs0 ushr 4) and kmask2) or ((((tmp ushr 4) and kmask1)) shl 4)
        var auxs3 = ((auxs1 ushr 4) and kmask2) or ((((tmp ushr 6) and kmask1)) shl 4)
        auxs0 = (auxs0 and kmask2) or ((((tmp ushr 0) and kmask1)) shl 4)
        auxs1 = (auxs1 and kmask2) or ((((tmp ushr 2) and kmask1)) shl 4)
        val auxsArr = intArrayOf(auxs0, auxs1, auxs2, auxs3)

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            // scales[j] = (int8_t) from auxs reinterpreted as bytes
            val scaleVal = (auxsArr[j / 4] ushr ((j % 4) * 8)).toByte().toInt()
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += (scaleVal - 32) * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += (scaleVal - 32) * aux16[l]
            q8Off += 8; aOff += 8
        }
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        for (l in 0 until 8) sums[l] += d * aux32[l]
    }
    for (l in 0 until 8) sumf += sums[l]
    return sumf
}

/** Q4_K × Q8_K dot product (scalar). */
fun ggmlVecDotQ4KQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ4K>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)

    var sumf = 0.0f
    for (i in 0 until nb) {
        var q4Off = 0
        var q8Off = 0
        for (l in 0 until 8) aux32[l] = 0
        var aOff = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 64) {
            for (l in 0 until 32) aux8[aOff + l] = ((vx[i].qs[q4Off + l].toInt() and 0xFF) and 0xF).toByte().toInt()
            aOff += 32
            for (l in 0 until 32) aux8[aOff + l] = ((vx[i].qs[q4Off + l].toInt() and 0xFF) ushr 4).toByte().toInt()
            aOff += 32; q4Off += 32
        }

        // memcpy(utmp, x[i].scales, 12) → assemble 3 uint32s
        val sc = vx[i].scales
        var utmp0 = (sc[0].toInt() and 0xFF) or ((sc[1].toInt() and 0xFF) shl 8) or
                ((sc[2].toInt() and 0xFF) shl 16) or ((sc[3].toInt() and 0xFF) shl 24)
        var utmp1 = (sc[4].toInt() and 0xFF) or ((sc[5].toInt() and 0xFF) shl 8) or
                ((sc[6].toInt() and 0xFF) shl 16) or ((sc[7].toInt() and 0xFF) shl 24)
        var utmp2 = (sc[8].toInt() and 0xFF) or ((sc[9].toInt() and 0xFF) shl 8) or
                ((sc[10].toInt() and 0xFF) shl 16) or ((sc[11].toInt() and 0xFF) shl 24)
        var utmp3 = ((utmp2 ushr 4) and kmask2) or (((utmp1 ushr 6) and kmask3) shl 4)
        val uaux = utmp1 and kmask1
        utmp1 = (utmp2 and kmask2) or (((utmp0 ushr 6) and kmask3) shl 4)
        utmp2 = uaux
        utmp0 = utmp0 and kmask1
        val utmpArr = intArrayOf(utmp0, utmp1, utmp2, utmp3)

        var sumi = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            // mins[j/2] = byte from utmp[2 + (j/2)/4] at position (j/2)%4
            val minsIdx = j / 2
            val minsVal = (utmpArr[2 + minsIdx / 4] ushr ((minsIdx % 4) * 8)) and 0xFF
            sumi += vy[i].bsums[j].toInt() * minsVal
        }

        aOff = 0
        var isIdx = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            // scales[is] = byte from utmp[isIdx/4] at position isIdx%4
            val scale = (utmpArr[isIdx / 4] ushr ((isIdx % 4) * 8)) and 0xFF
            isIdx++
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
        }
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        for (l in 0 until 8) sums[l] += d * aux32[l]
        val dmin = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].dmin) * vy[i].d
        sumf -= dmin * sumi
    }
    for (l in 0 until 8) sumf += sums[l]
    return sumf
}

/** Q5_K × Q8_K dot product (scalar). */
fun ggmlVecDotQ5KQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ5K>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)

    var sumf = 0.0f
    for (i in 0 until nb) {
        var q4Off = 0
        var q8Off = 0
        for (l in 0 until 8) aux32[l] = 0
        var aOff = 0
        var m = 1
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 64) {
            for (l in 0 until 32) aux8[aOff + l] = ((vx[i].qs[q4Off + l].toInt() and 0xFF) and 0xF).toByte().toInt()
            for (l in 0 until 32) aux8[aOff + l] += if ((vx[i].qh[l].toInt() and 0xFF) and m != 0) 16 else 0
            aOff += 32; m = m shl 1
            for (l in 0 until 32) aux8[aOff + l] = ((vx[i].qs[q4Off + l].toInt() and 0xFF) ushr 4).toByte().toInt()
            for (l in 0 until 32) aux8[aOff + l] += if ((vx[i].qh[l].toInt() and 0xFF) and m != 0) 16 else 0
            aOff += 32; m = m shl 1
            q4Off += 32
        }

        // memcpy(utmp, x[i].scales, 12)
        val sc = vx[i].scales
        var utmp0 = (sc[0].toInt() and 0xFF) or ((sc[1].toInt() and 0xFF) shl 8) or
                ((sc[2].toInt() and 0xFF) shl 16) or ((sc[3].toInt() and 0xFF) shl 24)
        var utmp1 = (sc[4].toInt() and 0xFF) or ((sc[5].toInt() and 0xFF) shl 8) or
                ((sc[6].toInt() and 0xFF) shl 16) or ((sc[7].toInt() and 0xFF) shl 24)
        var utmp2 = (sc[8].toInt() and 0xFF) or ((sc[9].toInt() and 0xFF) shl 8) or
                ((sc[10].toInt() and 0xFF) shl 16) or ((sc[11].toInt() and 0xFF) shl 24)
        var utmp3 = ((utmp2 ushr 4) and kmask2) or (((utmp1 ushr 6) and kmask3) shl 4)
        val uaux = utmp1 and kmask1
        utmp1 = (utmp2 and kmask2) or (((utmp0 ushr 6) and kmask3) shl 4)
        utmp2 = uaux
        utmp0 = utmp0 and kmask1
        val utmpArr = intArrayOf(utmp0, utmp1, utmp2, utmp3)

        var sumi = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val minsIdx = j / 2
            val minsVal = (utmpArr[2 + minsIdx / 4] ushr ((minsIdx % 4) * 8)) and 0xFF
            sumi += vy[i].bsums[j].toInt() * minsVal
        }

        aOff = 0
        var isIdx = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val scale = (utmpArr[isIdx / 4] ushr ((isIdx % 4) * 8)) and 0xFF
            isIdx++
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
        }
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        for (l in 0 until 8) sums[l] += d * aux32[l]
        val dmin = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].dmin) * vy[i].d
        sumf -= dmin * sumi
    }
    for (l in 0 until 8) sumf += sums[l]
    return sumf
}

/** Q6_K × Q8_K dot product (scalar). */
fun ggmlVecDotQ6KQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockQ6K>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val aux8 = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val aux16 = IntArray(8)
    val sums = FloatArray(8)
    val aux32 = IntArray(8)

    var sumf = 0.0f
    for (i in 0 until nb) {
        var q4Off = 0
        var qhOff = 0
        var q8Off = 0
        for (l in 0 until 8) aux32[l] = 0
        var aOff = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                aux8[aOff + l + 0] = (((vx[i].ql[q4Off + l + 0].toInt() and 0xFF) and 0xF) or
                        ((((vx[i].qh[qhOff + l].toInt() and 0xFF) ushr 0) and 3) shl 4)).toByte().toInt() - 32
                aux8[aOff + l + 32] = (((vx[i].ql[q4Off + l + 32].toInt() and 0xFF) and 0xF) or
                        ((((vx[i].qh[qhOff + l].toInt() and 0xFF) ushr 2) and 3) shl 4)).toByte().toInt() - 32
                aux8[aOff + l + 64] = (((vx[i].ql[q4Off + l + 0].toInt() and 0xFF) ushr 4) or
                        ((((vx[i].qh[qhOff + l].toInt() and 0xFF) ushr 4) and 3) shl 4)).toByte().toInt() - 32
                aux8[aOff + l + 96] = (((vx[i].ql[q4Off + l + 32].toInt() and 0xFF) ushr 4) or
                        ((((vx[i].qh[qhOff + l].toInt() and 0xFF) ushr 6) and 3) shl 4)).toByte().toInt() - 32
            }
            aOff += 128
            q4Off += 64
            qhOff += 32
        }
        aOff = 0
        var isIdx = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val scale = vx[i].scales[isIdx++].toInt()
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
            for (l in 0 until 8) aux16[l] = vy[i].qs[q8Off + l].toInt() * aux8[aOff + l]
            for (l in 0 until 8) aux32[l] += scale * aux16[l]
            q8Off += 8; aOff += 8
        }
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        for (l in 0 until 8) sums[l] += d * aux32[l]
    }
    for (l in 0 until 8) sumf += sums[l]
    return sumf
}

// ── IQ dot products ─────────────────────────────────────────────────────────────

fun ggmlVecDotIq2XxsQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ2XXS>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    var sumf = 0.0f
    for (i in 0 until nb) {
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        var q2Off = 0
        var q8Off = 0
        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            // memcpy(aux32, q2, 2*sizeof(uint32_t)) — 4 uint16s → 2 uint32s (little-endian)
            val a32_0 = (vx[i].qs[q2Off + 0].toInt() and 0xFFFF) or ((vx[i].qs[q2Off + 1].toInt() and 0xFFFF) shl 16)
            val a32_1 = (vx[i].qs[q2Off + 2].toInt() and 0xFFFF) or ((vx[i].qs[q2Off + 3].toInt() and 0xFFFF) shl 16)
            q2Off += 4
            val ls = 2 * ((a32_1 ushr 28) and 0xF) + 1
            var sumi = 0
            // aux8 = bytes of a32_0/a32_1: aux8[0..3] from a32_0, aux8[4..7] from a32_1
            for (l in 0 until 4) {
                val aux8val = (a32_0 ushr (l * 8)) and 0xFF
                val gridVal = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq2xxsGrid[aux8val]
                val signs = io.github.kotlinmania.llama.ore.GGMLCommonTables.ksignsIQ2XS[((a32_1 ushr (7 * l)) and 127).toByte().toInt() and 0xFF]
                for (j in 0 until 8) {
                    val gridByte = ((gridVal ushr (j * 8)) and 0xFFL).toInt()
                    val sign = if ((signs.toInt() and 0xFF) and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j].toInt() and 0xFF) != 0) -1 else 1
                    sumi += gridByte * vy[i].qs[q8Off + j].toInt() * sign
                }
                q8Off += 8
            }
            bsum += sumi * ls
        }
        sumf += d * bsum
    }
    return 0.125f * sumf
}

fun ggmlVecDotIq2XsQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ2XS>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    var sumf = 0.0f
    for (i in 0 until nb) {
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        var q8Off = 0
        var bsum = 0
        var q2Off = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val ls1 = 2 * (vx[i].scales[ib32].toInt() and 0xF) + 1
            val ls2 = 2 * ((vx[i].scales[ib32].toInt() and 0xFF) ushr 4) + 1
            var sumi = 0
            for (l in 0 until 2) {
                val q2val = vx[i].qs[q2Off + l].toInt() and 0xFFFF
                val gridVal = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq2xsGrid[q2val and 511]
                val signs = io.github.kotlinmania.llama.ore.GGMLCommonTables.ksignsIQ2XS[(q2val ushr 9) and 0xFF]
                for (j in 0 until 8) {
                    val gridByte = ((gridVal ushr (j * 8)) and 0xFFL).toInt()
                    val sign = if ((signs.toInt() and 0xFF) and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j].toInt() and 0xFF) != 0) -1 else 1
                    sumi += gridByte * vy[i].qs[q8Off + j].toInt() * sign
                }
                q8Off += 8
            }
            bsum += sumi * ls1
            sumi = 0
            for (l in 2 until 4) {
                val q2val = vx[i].qs[q2Off + l].toInt() and 0xFFFF
                val gridVal = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq2xsGrid[q2val and 511]
                val signs = io.github.kotlinmania.llama.ore.GGMLCommonTables.ksignsIQ2XS[(q2val ushr 9) and 0xFF]
                for (j in 0 until 8) {
                    val gridByte = ((gridVal ushr (j * 8)) and 0xFFL).toInt()
                    val sign = if ((signs.toInt() and 0xFF) and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j].toInt() and 0xFF) != 0) -1 else 1
                    sumi += gridByte * vy[i].qs[q8Off + j].toInt() * sign
                }
                q8Off += 8
            }
            bsum += sumi * ls2
            q2Off += 4
        }
        sumf += d * bsum
    }
    return 0.125f * sumf
}

fun ggmlVecDotIq2SQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ2S>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    var sumf = 0.0f
    for (i in 0 until nb) {
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        var q8Off = 0
        var qsOff = 0
        val signsBase = io.github.kotlinmania.llama.ore.QK_K / 8
        var signsOff = 0

        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val ls1 = 1 + 2 * (vx[i].scales[ib32].toInt() and 0xF)
            val ls2 = 1 + 2 * ((vx[i].scales[ib32].toInt() and 0xFF) ushr 4)
            var sumi1 = 0
            var sumi2 = 0
            for (l in 0 until 2) {
                val qsVal = vx[i].qs[qsOff + l].toInt() and 0xFF
                val qhVal = vx[i].qh[ib32].toInt() and 0xFF
                val gridIdx = qsVal or ((qhVal shl (8 - 2 * l)) and 0x300)
                val gridVal = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq2sGrid[gridIdx]
                val signsVal = vx[i].qs[signsBase + signsOff + l].toInt() and 0xFF
                for (j in 0 until 8) {
                    val gridByte = ((gridVal ushr (j * 8)) and 0xFFL).toInt()
                    val sign = if (signsVal and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j].toInt() and 0xFF) != 0) -1 else 1
                    sumi1 += vy[i].qs[q8Off + j].toInt() * gridByte * sign
                }
                q8Off += 8
            }
            for (l in 2 until 4) {
                val qsVal = vx[i].qs[qsOff + l].toInt() and 0xFF
                val qhVal = vx[i].qh[ib32].toInt() and 0xFF
                val gridIdx = qsVal or ((qhVal shl (8 - 2 * l)) and 0x300)
                val gridVal = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq2sGrid[gridIdx]
                val signsVal = vx[i].qs[signsBase + signsOff + l].toInt() and 0xFF
                for (j in 0 until 8) {
                    val gridByte = ((gridVal ushr (j * 8)) and 0xFFL).toInt()
                    val sign = if (signsVal and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j].toInt() and 0xFF) != 0) -1 else 1
                    sumi2 += vy[i].qs[q8Off + j].toInt() * gridByte * sign
                }
                q8Off += 8
            }
            bsum += ls1 * sumi1 + ls2 * sumi2
            qsOff += 4
            signsOff += 4
        }

        sumf += d * bsum
    }

    return 0.125f * sumf
}

fun ggmlVecDotIq3XxsQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ3XXS>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    var sumf = 0.0f
    for (i in 0 until nb) {
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        var q3Off = 0
        var gasOff = io.github.kotlinmania.llama.ore.QK_K / 4
        var q8Off = 0
        var bsum = 0
        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            // memcpy(&aux32, gas, sizeof(uint32_t))
            val aux32 = (vx[i].qs[gasOff + 0].toInt() and 0xFF) or
                    ((vx[i].qs[gasOff + 1].toInt() and 0xFF) shl 8) or
                    ((vx[i].qs[gasOff + 2].toInt() and 0xFF) shl 16) or
                    ((vx[i].qs[gasOff + 3].toInt() and 0xFF) shl 24)
            gasOff += 4
            val ls = 2 * ((aux32 ushr 28) and 0xF) + 1
            var sumi = 0
            for (l in 0 until 4) {
                val grid1Val = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq3xxsGrid[vx[i].qs[q3Off + 2 * l + 0].toInt() and 0xFF]
                val grid2Val = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq3xxsGrid[vx[i].qs[q3Off + 2 * l + 1].toInt() and 0xFF]
                val signs = io.github.kotlinmania.llama.ore.GGMLCommonTables.ksignsIQ2XS[((aux32 ushr (7 * l)) and 127).toByte().toInt() and 0xFF]
                for (j in 0 until 4) {
                    val g1byte = ((grid1Val ushr (j * 8)) and 0xFF)
                    val g2byte = ((grid2Val ushr (j * 8)) and 0xFF)
                    val sign1 = if ((signs.toInt() and 0xFF) and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j + 0].toInt() and 0xFF) != 0) -1 else 1
                    val sign2 = if ((signs.toInt() and 0xFF) and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j + 4].toInt() and 0xFF) != 0) -1 else 1
                    sumi += g1byte * vy[i].qs[q8Off + j + 0].toInt() * sign1
                    sumi += g2byte * vy[i].qs[q8Off + j + 4].toInt() * sign2
                }
                q8Off += 8
            }
            q3Off += 8
            bsum += sumi * ls
        }
        sumf += d * bsum
    }
    return 0.25f * sumf
}

fun ggmlVecDotIq3SQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ3S>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    var sumf = 0.0f
    for (i in 0 until nb) {
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d
        var qsOff = 0
        var q8Off = 0
        var signsOff = 0
        var bsum = 0
        var ib32 = 0
        while (ib32 < io.github.kotlinmania.llama.ore.QK_K / 32) {
            val ls1 = 2 * (vx[i].scales[ib32 / 2].toInt() and 0xF) + 1
            val ls2 = 2 * ((vx[i].scales[ib32 / 2].toInt() and 0xFF) ushr 4) + 1
            var sumi = 0
            for (l in 0 until 4) {
                val qhVal0 = vx[i].qh[ib32 + 0].toInt() and 0xFF
                val grid1Idx = (vx[i].qs[qsOff + 2 * l + 0].toInt() and 0xFF) or (((qhVal0 shl (8 - 2 * l)) and 256))
                val grid2Idx = (vx[i].qs[qsOff + 2 * l + 1].toInt() and 0xFF) or (((qhVal0 shl (7 - 2 * l)) and 256))
                val grid1Val = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq3sGrid[grid1Idx]
                val grid2Val = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq3sGrid[grid2Idx]
                val signsVal = vx[i].signs[signsOff + l].toInt() and 0xFF
                for (j in 0 until 4) {
                    val g1byte = ((grid1Val ushr (j * 8)) and 0xFF)
                    val g2byte = ((grid2Val ushr (j * 8)) and 0xFF)
                    val sign1 = if (signsVal and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j + 0].toInt() and 0xFF) != 0) -1 else 1
                    val sign2 = if (signsVal and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j + 4].toInt() and 0xFF) != 0) -1 else 1
                    sumi += g1byte * vy[i].qs[q8Off + j + 0].toInt() * sign1
                    sumi += g2byte * vy[i].qs[q8Off + j + 4].toInt() * sign2
                }
                q8Off += 8
            }
            qsOff += 8
            signsOff += 4
            bsum += sumi * ls1
            sumi = 0
            for (l in 0 until 4) {
                val qhVal1 = vx[i].qh[ib32 + 1].toInt() and 0xFF
                val grid1Idx = (vx[i].qs[qsOff + 2 * l + 0].toInt() and 0xFF) or (((qhVal1 shl (8 - 2 * l)) and 256))
                val grid2Idx = (vx[i].qs[qsOff + 2 * l + 1].toInt() and 0xFF) or (((qhVal1 shl (7 - 2 * l)) and 256))
                val grid1Val = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq3sGrid[grid1Idx]
                val grid2Val = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq3sGrid[grid2Idx]
                val signsVal = vx[i].signs[signsOff + l].toInt() and 0xFF
                for (j in 0 until 4) {
                    val g1byte = ((grid1Val ushr (j * 8)) and 0xFF)
                    val g2byte = ((grid2Val ushr (j * 8)) and 0xFF)
                    val sign1 = if (signsVal and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j + 0].toInt() and 0xFF) != 0) -1 else 1
                    val sign2 = if (signsVal and (io.github.kotlinmania.llama.ore.GGMLCommonTables.kmaskIQ2XS[j + 4].toInt() and 0xFF) != 0) -1 else 1
                    sumi += g1byte * vy[i].qs[q8Off + j + 0].toInt() * sign1
                    sumi += g2byte * vy[i].qs[q8Off + j + 4].toInt() * sign2
                }
                q8Off += 8
            }
            qsOff += 8
            signsOff += 4
            bsum += sumi * ls2
            ib32 += 2
        }
        sumf += d * bsum
    }
    return sumf
}

fun ggmlVecDotIq1SQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ1S>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K
    val IQ1S_DELTA = 0.125f

    var sumf = 0.0f
    for (i in 0 until nb) {
        var q8Off = 0
        var qsOff = 0

        var sumi = 0
        var sumi1 = 0
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val qhVal = vx[i].qh[ib].toInt() and 0xFFFF
            val ls = 2 * ((qhVal ushr 12) and 7) + 1
            val delta = if (qhVal and 0x8000 != 0) -1 else 1
            var lsum = 0
            for (l in 0 until 4) {
                val gridIdx = (vx[i].qs[qsOff + l].toInt() and 0xFF) or (((qhVal ushr (3 * l)) and 7) shl 8)
                val gridVal = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq1sGrid[gridIdx]
                for (j in 0 until 8) {
                    val gridByte = ((gridVal ushr (j * 8)) and 0xFFL).toByte().toInt()
                    lsum += vy[i].qs[q8Off + j].toInt() * gridByte
                }
                q8Off += 8
            }
            sumi += ls * lsum
            sumi1 += ls * delta * (vy[i].bsums[2 * ib + 0].toInt() + vy[i].bsums[2 * ib + 1].toInt())
            qsOff += 4
        }

        sumf += io.github.kotlinmania.llama.ore.halfToFloat(vx[i].d) * vy[i].d * (sumi + IQ1S_DELTA * sumi1)
    }

    return sumf
}

fun ggmlVecDotIq1MQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ1M>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    val sum1 = IntArray(2)
    val sum2 = IntArray(2)
    val delta = IntArray(4)

    var sumf = 0.0f
    for (i in 0 until nb) {
        var q8Off = 0
        var qsOff = 0
        var qhOff = 0

        // sc = (uint16_t*)x[i].scales — reinterpret scales bytes as uint16 little-endian
        val scBytes = vx[i].scales
        val sc = IntArray(scBytes.size / 2) { idx ->
            (scBytes[idx * 2].toInt() and 0xFF) or ((scBytes[idx * 2 + 1].toInt() and 0xFF) shl 8)
        }

        val scaleBits = ((sc[0] ushr 12) and 0xF) or
                (((sc[1] ushr 8) and 0x00F0)) or
                (((sc[2] ushr 4) and 0x0F00)) or
                ((sc[3] and 0xF000))
        val scale = io.github.kotlinmania.llama.ore.IQ1MScale(scaleBits.toShort())

        var sumi1 = 0
        var sumi2 = 0
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val qh0 = vx[i].qh[qhOff + 0].toInt() and 0xFF
            val qh1 = vx[i].qh[qhOff + 1].toInt() and 0xFF
            delta[0] = if (qh0 and 0x08 != 0) -1 else 1
            delta[1] = if (qh0 and 0x80 != 0) -1 else 1
            delta[2] = if (qh1 and 0x08 != 0) -1 else 1
            delta[3] = if (qh1 and 0x80 != 0) -1 else 1
            sum1[0] = 0; sum1[1] = 0; sum2[0] = 0; sum2[1] = 0
            for (l in 0 until 4) {
                val qhByte = vx[i].qh[qhOff + l / 2].toInt() and 0xFF
                val gridIdx = (vx[i].qs[qsOff + l].toInt() and 0xFF) or
                        (((qhByte shl (8 - 4 * (l % 2))) and 0x700))
                val gridVal = io.github.kotlinmania.llama.ore.GGMLCommonTables.iq1sGrid[gridIdx]
                var lsum1 = 0
                var lsum2 = 0
                for (j in 0 until 8) {
                    val gridByte = ((gridVal ushr (j * 8)) and 0xFFL).toByte().toInt()
                    lsum1 += vy[i].qs[q8Off + j].toInt() * gridByte
                    lsum2 += vy[i].qs[q8Off + j].toInt()
                }
                q8Off += 8
                sum1[l / 2] += lsum1
                sum2[l / 2] += lsum2 * delta[l]
            }

            val ls1 = 2 * ((sc[ib / 2] ushr (6 * (ib % 2) + 0)) and 0x7) + 1
            val ls2 = 2 * ((sc[ib / 2] ushr (6 * (ib % 2) + 3)) and 0x7) + 1

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2
            qsOff += 4
            qhOff += 2
        }

        sumf += io.github.kotlinmania.llama.ore.halfToFloat(scale.f16) * vy[i].d * (sumi1 + io.github.kotlinmania.llama.ore.IQ1M_DELTA * sumi2)
    }

    return sumf
}

/**
 * IQ4_NL × Q8_0 dot product (scalar).
 *
 * Non-linear 4-bit quants use [io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesIQ4NL] for dequantization.
 */
fun ggmlVecDotIq4NlQ80Generic(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ4NL>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8_0>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK4_NL == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK4_NL}" }
    val nb = n / io.github.kotlinmania.llama.ore.QK4_NL

    var sumf = 0.0f
    for (ib in 0 until nb) {
        val d = io.github.kotlinmania.llama.ore.halfToFloat(vy[ib].d) * io.github.kotlinmania.llama.ore.halfToFloat(
            vx[ib].d
        )
        var sumi1 = 0
        var sumi2 = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK4_NL / 2) {
            sumi1 += vy[ib].qs[j].toInt() *
                    io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesIQ4NL[vx[ib].qs[j].toInt() and 0xF].toInt()
            sumi2 += vy[ib].qs[j + io.github.kotlinmania.llama.ore.QK4_NL / 2].toInt() *
                    io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesIQ4NL[(vx[ib].qs[j].toInt() and 0xFF) ushr 4].toInt()
        }
        sumf += d * (sumi1 + sumi2)
    }
    return sumf
}

fun ggmlVecDotIq4XsQ8KGeneric(n: Int, vx: Array<io.github.kotlinmania.llama.ore.BlockIQ4XS>, vy: Array<io.github.kotlinmania.llama.ore.BlockQ8K>): Float {
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0) { "n ($n) must be divisible by ${io.github.kotlinmania.llama.ore.QK_K}" }

    val nb = n / io.github.kotlinmania.llama.ore.QK_K

    var sumf = 0.0f
    for (ibl in 0 until nb) {
        val d4d8 = io.github.kotlinmania.llama.ore.halfToFloat(vx[ibl].d) * vy[ibl].d
        var h = vx[ibl].scalesH.toInt() and 0xFFFF
        var qsOff = 0
        var q8Off = 0
        var ib = 0
        while (ib < io.github.kotlinmania.llama.ore.QK_K / 32) {
            val ls1 = ((vx[ibl].scalesL[ib / 2].toInt() and 0xFF) and 0xF) or ((h shl 4) and 0x30)
            val ls2 = ((vx[ibl].scalesL[ib / 2].toInt() and 0xFF) ushr 4) or ((h shl 2) and 0x30)
            h = h ushr 4
            val d1 = d4d8 * (ls1 - 32)
            val d2 = d4d8 * (ls2 - 32)
            var sumi1 = 0
            var sumi2 = 0
            for (j in 0 until 16) {
                sumi1 += vy[ibl].qs[q8Off + j + 0].toInt() *
                        io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesIQ4NL[vx[ibl].qs[qsOff + j].toInt() and 0xF].toInt()
                sumi2 += vy[ibl].qs[q8Off + j + 16].toInt() *
                        io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesIQ4NL[(vx[ibl].qs[qsOff + j].toInt() and 0xFF) ushr 4].toInt()
            }
            sumf += d1 * (sumi1 + sumi2)
            qsOff += 16
            q8Off += 32
            sumi1 = 0; sumi2 = 0
            for (j in 0 until 16) {
                sumi1 += vy[ibl].qs[q8Off + j + 0].toInt() *
                        io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesIQ4NL[vx[ibl].qs[qsOff + j].toInt() and 0xF].toInt()
                sumi2 += vy[ibl].qs[q8Off + j + 16].toInt() *
                        io.github.kotlinmania.llama.ore.GGMLCommonTables.kvaluesIQ4NL[(vx[ibl].qs[qsOff + j].toInt() and 0xFF) ushr 4].toInt()
            }
            sumf += d2 * (sumi1 + sumi2)
            qsOff += 16
            q8Off += 32
            ib += 2
        }
    }
    return sumf
}

// ════════════════════════════════════════════════════════════════════════════════
//  Numeric helpers (E8M0, UE4M3)
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Convert an E8M0 exponent byte to a half-precision scale (returned as Float).
 *
 * For x < 2 the result is a subnormal float (IEEE 754 denormalized).
 * For x >= 2 the result is `2^(x - 128)`.
 * Mirrors `ggml_e8m0_to_fp32_half` from the C source.
 */
internal fun ggmlE8M0ToFp32Half(e: Byte): Float {
    val x = e.toInt() and 0xFF
    val bits = if (x < 2) {
        0x00200000 shl x
    } else {
        (x - 1) shl 23
    }
    return Float.fromBits(bits)
}

/**
 * Convert a UE4M3 byte (unsigned 4-bit exponent, 3-bit mantissa) to Float.
 *
 * Mirrors `ggml_ue4m3_to_fp32` from the C source.
 */
internal fun ggmlUE4M3ToFp32(b: Byte): Float {
    val x = b.toInt() and 0xFF
    if (x == 0 || x == 0x7F) return 0.0f
    val exp = (x ushr 3) and 0xF
    val man = x and 0x7
    val raw = if (exp == 0) {
        man.toFloat() * (1.0f / 512.0f)
    } else {
        (1.0f + man.toFloat() / 8.0f) * Float.fromBits((exp - 7 + 127) shl 23)
    }
    return raw * 0.5f
}
