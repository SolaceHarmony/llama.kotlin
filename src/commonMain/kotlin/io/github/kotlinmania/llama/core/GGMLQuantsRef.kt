// port-lint: source ggml/src/ggml-quants.c

package io.github.kotlinmania.llama.ore

// QK constants are top-level in GGMLTypes.kt and GGMLCommon.kt — same package, no import needed.

/**
 * Port of `ggml-quants.c` — dequantization, quantization-row-ref, and
 * importance-matrix quantization functions.
 *
 * Block data is stored as packed [ByteArray]. Each block type has a known
 * SIZE_BYTES constant in [GGMLCommon]. Functions here operate on raw byte
 * buffers with explicit block offsets, matching the C memory layout.
 *
 * Dequantize functions read packed blocks and write F32 output.
 * Quantize-row-ref functions read F32 input and write packed blocks.
 *
 * IQ dequantize functions use grid lookup tables from GGMLIQGrids.kt
 * (iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid, iq1s_grid)
 * and sign/mask tables from GGMLQuants.kt (ksigns_iq2xs, kmask_iq2xs).
 */

private const val GROUP_MAX_EPS = 1e-15f
private const val GROUP_MAX_EPS_IQ3_XXS = 1e-8f
private const val GROUP_MAX_EPS_IQ2_S = 1e-8f
private const val GROUP_MAX_EPS_IQ1_M = 1e-7f
private const val GROUP_MAX_EPS_IQ1_S = 1e-12f

// C line 24
private fun best_index_int8(n: Int, `val`: ByteArray, valOff: Int, x: Float): Int {
    if (x <= `val`[valOff].toInt()) return 0
    if (x >= `val`[valOff + n - 1].toInt()) return n - 1
    var ml = 0; var mu = n - 1
    while (mu - ml > 1) {
        val mav = (ml + mu) / 2
        if (x < `val`[valOff + mav].toInt()) mu = mav else ml = mav
    }
    return if (x - `val`[valOff + mu - 1].toInt() < `val`[valOff + mu].toInt() - x) mu - 1 else mu
}

// C line 295
private fun best_index_mxfp4(x: Float, e: Float): Int {
    var bestIndex = 0
    var bestErr = kotlin.math.abs(io.github.kotlinmania.llama.ore.kvalues_mxfp4[0].toInt() * e - x)
    for (i in 1 until 16) {
        val err = kotlin.math.abs(io.github.kotlinmania.llama.ore.kvalues_mxfp4[i].toInt() * e - x)
        if (err < bestErr) {
            bestIndex = i
            bestErr = err
        }
    }
    return bestIndex
}

// ════════════════════════════════════════════════════════════════════════════════
//  Helper: get_scale_min_k4 (used by Q4_K, Q5_K dequantize)
// ════════════════════════════════════════════════════════════════════════════════

private data class ScaleMin(val d: Int, val m: Int)

/** Port of `get_scale_min_k4` from ggml-quants.c. */
private fun get_scale_min_k4(j: Int, q: ByteArray, qOff: Int): io.github.kotlinmania.llama.ore.ScaleMin {
    return if (j < 4) {
        io.github.kotlinmania.llama.ore.ScaleMin(
            d = (q[qOff + j].toInt() and 0xFF) and 63,
            m = (q[qOff + j + 4].toInt() and 0xFF) and 63
        )
    } else {
        io.github.kotlinmania.llama.ore.ScaleMin(
            d = ((q[qOff + j + 4].toInt() and 0xFF) and 0xF) or ((((q[qOff + j - 4].toInt() and 0xFF) shr 6) and 3) shl 4),
            m = ((q[qOff + j + 4].toInt() and 0xFF) shr 4) or ((((q[qOff + j].toInt() and 0xFF) shr 6) and 3) shl 4)
        )
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Quantization helper functions (ggml-quants.c lines 559–1080)
// ════════════════════════════════════════════════════════════════════════════════

/** Port of `nearest_int` (C line 559). Uses the float bit trick for fast rounding. */
private fun nearest_int(fval: Float): Int {
    val v = fval + 12582912f
    val i = v.toRawBits()
    return (i and 0x007fffff) - 0x00400000
}

/** Port of `make_qx_quants` (C line 566). Signed quantization helper. */
private fun make_qx_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: ByteArray, lOff: Int, rmseType: Int,
    qw: FloatArray?, qwOff: Int
): Float {
    var max = 0f
    var amax = 0f
    for (i in 0 until n) {
        val ax = kotlin.math.abs(x[xOff + i])
        if (ax > amax) { amax = ax; max = x[xOff + i] }
    }
    if (amax < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
        for (i in 0 until n) L[lOff + i] = 0
        return 0f
    }
    var iscale = -nmax.toFloat() / max
    var rmseTypeVar = rmseType
    if (rmseTypeVar == 0) {
        for (i in 0 until n) {
            val l = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i])
            L[lOff + i] = (nmax + maxOf(-nmax, minOf(nmax - 1, l))).toByte()
        }
        return 1f / iscale
    }
    var returnEarly = false
    if (rmseTypeVar < 0) {
        rmseTypeVar = -rmseTypeVar
        returnEarly = true
    }
    var sumlx = 0f
    var suml2 = 0f
    for (i in 0 until n) {
        var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i])
        l = maxOf(-nmax, minOf(nmax - 1, l))
        L[lOff + i] = (l + nmax).toByte()
        val w = if (qw != null) qw[qwOff + i]
                else if (rmseTypeVar == 1) x[xOff + i] * x[xOff + i]
                else if (rmseTypeVar == 2) 1f
                else if (rmseTypeVar == 3) kotlin.math.abs(x[xOff + i])
                else kotlin.math.sqrt(kotlin.math.abs(x[xOff + i]))
        sumlx += w * x[xOff + i] * l
        suml2 += w * l * l
    }
    var scale = if (suml2 != 0f) sumlx / suml2 else 0f
    if (returnEarly) return if (suml2 > 0f) 0.5f * (scale + 1f / iscale) else 1f / iscale
    var best = scale * sumlx
    for (is_ in -9..9) {
        if (is_ == 0) continue
        iscale = -(nmax + 0.1f * is_) / max
        sumlx = 0f; suml2 = 0f
        for (i in 0 until n) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i])
            l = maxOf(-nmax, minOf(nmax - 1, l))
            val w = if (qw != null) qw[qwOff + i]
                    else if (rmseTypeVar == 1) x[xOff + i] * x[xOff + i]
                    else if (rmseTypeVar == 2) 1f
                    else if (rmseTypeVar == 3) kotlin.math.abs(x[xOff + i])
                    else kotlin.math.sqrt(kotlin.math.abs(x[xOff + i]))
            sumlx += w * x[xOff + i] * l
            suml2 += w * l * l
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (i in 0 until n) {
                val l = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i])
                L[lOff + i] = (nmax + maxOf(-nmax, minOf(nmax - 1, l))).toByte()
            }
            scale = sumlx / suml2; best = scale * sumlx
        }
    }
    return scale
}

/** Port of `make_q3_quants` (C line 635). Signed quantization with optional RMSE refinement. */
private fun make_q3_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: ByteArray, lOff: Int, doRmse: Boolean
): Float {
    var max = 0f
    var amax = 0f
    for (i in 0 until n) {
        val ax = kotlin.math.abs(x[xOff + i])
        if (ax > amax) { amax = ax; max = x[xOff + i] }
    }
    if (amax < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
        for (i in 0 until n) L[lOff + i] = 0
        return 0f
    }
    val iscale = -nmax.toFloat() / max
    if (doRmse) {
        var sumlx = 0f
        var suml2 = 0f
        for (i in 0 until n) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i])
            l = maxOf(-nmax, minOf(nmax - 1, l))
            L[lOff + i] = l.toByte()
            val w = x[xOff + i] * x[xOff + i]
            sumlx += w * x[xOff + i] * l
            suml2 += w * l * l
        }
        for (itry in 0 until 5) {
            var nChanged = 0
            for (i in 0 until n) {
                val w = x[xOff + i] * x[xOff + i]
                val slx0 = sumlx - w * x[xOff + i] * L[lOff + i]
                if (slx0 > 0) {
                    var sl2 = suml2 - w * L[lOff + i] * L[lOff + i]
                    var newL = io.github.kotlinmania.llama.ore.nearest_int(x[xOff + i] * sl2 / slx0)
                    newL = maxOf(-nmax, minOf(nmax - 1, newL))
                    if (newL != L[lOff + i].toInt()) {
                        val slx = slx0 + w * x[xOff + i] * newL
                        sl2 += w * newL * newL
                        if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            L[lOff + i] = newL.toByte(); sumlx = slx; suml2 = sl2
                            ++nChanged
                        }
                    }
                }
            }
            if (nChanged == 0) break
        }
        for (i in 0 until n) {
            L[lOff + i] = (L[lOff + i] + nmax).toByte()
        }
        return if (suml2 > 0f) sumlx / suml2 else 0f
    }
    for (i in 0 until n) {
        var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i])
        l = maxOf(-nmax, minOf(nmax - 1, l))
        L[lOff + i] = (l + nmax).toByte()
    }
    return 1f / iscale
}

/** Port of `make_qkx1_quants` (C line 694). Unsigned quantization with min subtraction. */
private fun make_qkx1_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: UByteArray, lOff: Int, theMin: FloatArray, minIdx: Int,
    ntry: Int, alpha: Float
): Float {
    var min = x[xOff]
    var max = x[xOff]
    for (i in 1 until n) {
        if (x[xOff + i] < min) min = x[xOff + i]
        if (x[xOff + i] > max) max = x[xOff + i]
    }
    if (max == min) {
        for (i in 0 until n) L[lOff + i] = 0u
        theMin[minIdx] = 0f
        return 0f
    }
    if (min > 0) min = 0f
    var iscale = nmax.toFloat() / (max - min)
    var scale = 1f / iscale
    for (itry in 0 until ntry) {
        var sumlx = 0f; var suml2 = 0
        var didChange = false
        for (i in 0 until n) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * (x[xOff + i] - min))
            l = maxOf(0, minOf(nmax, l))
            if (l.toUByte() != L[lOff + i]) {
                L[lOff + i] = l.toUByte()
                didChange = true
            }
            sumlx += (x[xOff + i] - min) * l
            suml2 += l * l
        }
        scale = sumlx / suml2
        var sum = 0f
        for (i in 0 until n) {
            sum += x[xOff + i] - scale * L[lOff + i].toInt()
        }
        min = alpha * min + (1 - alpha) * sum / n
        if (min > 0) min = 0f
        iscale = 1f / scale
        if (!didChange) break
    }
    theMin[minIdx] = -min
    return scale
}

/** Port of `make_qkx2_quants` (C line 737). Weighted quantization with search. */
private fun make_qkx2_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    weights: FloatArray, wOff: Int, L: UByteArray, lOff: Int,
    theMin: FloatArray, minIdx: Int, Laux: UByteArray, lauxOff: Int,
    rmin: Float, rdelta: Float, nstep: Int, useMad: Boolean
): Float {
    var min = x[xOff]
    var max = x[xOff]
    var sumW = weights[wOff]
    var sumX = sumW * x[xOff]
    for (i in 1 until n) {
        if (x[xOff + i] < min) min = x[xOff + i]
        if (x[xOff + i] > max) max = x[xOff + i]
        val w = weights[wOff + i]
        sumW += w
        sumX += w * x[xOff + i]
    }
    if (min > 0) min = 0f
    if (max == min) {
        for (i in 0 until n) L[lOff + i] = 0u
        theMin[minIdx] = -min
        return 0f
    }
    var iscale = nmax.toFloat() / (max - min)
    var scale = 1f / iscale
    var bestError = 0f
    for (i in 0 until n) {
        var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * (x[xOff + i] - min))
        L[lOff + i] = maxOf(0, minOf(nmax, l)).toUByte()
        var diff = scale * L[lOff + i].toInt() + min - x[xOff + i]
        diff = if (useMad) kotlin.math.abs(diff) else diff * diff
        val w = weights[wOff + i]
        bestError += w * diff
    }
    if (nstep < 1) {
        theMin[minIdx] = -min
        return scale
    }
    for (is_ in 0..nstep) {
        iscale = (rmin + rdelta * is_ + nmax) / (max - min)
        var sumL = 0f; var sumL2 = 0f; var sumXL = 0f
        for (i in 0 until n) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * (x[xOff + i] - min))
            l = maxOf(0, minOf(nmax, l))
            Laux[lauxOff + i] = l.toUByte()
            val w = weights[wOff + i]
            sumL += w * l
            sumL2 += w * l * l
            sumXL += w * l * x[xOff + i]
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D > 0) {
            var thisScale = (sumW * sumXL - sumX * sumL) / D
            var thisMin = (sumL2 * sumX - sumL * sumXL) / D
            if (thisMin > 0) {
                thisMin = 0f
                thisScale = sumXL / sumL2
            }
            var curError = 0f
            for (i in 0 until n) {
                var diff = thisScale * Laux[lauxOff + i].toInt() + thisMin - x[xOff + i]
                diff = if (useMad) kotlin.math.abs(diff) else diff * diff
                val w = weights[wOff + i]
                curError += w * diff
            }
            if (curError < bestError) {
                for (i in 0 until n) L[lOff + i] = Laux[lauxOff + i]
                bestError = curError
                scale = thisScale
                min = thisMin
            }
        }
    }
    theMin[minIdx] = -min
    return scale
}

/** Port of `make_qkx3_quants` (C line 931). Like qkx2 but weights can be null. */
private fun make_qkx3_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    weights: FloatArray?, wOff: Int, L: UByteArray, lOff: Int,
    theMin: FloatArray, minIdx: Int, Laux: UByteArray, lauxOff: Int,
    rmin: Float, rdelta: Float, nstep: Int, useMad: Boolean
): Float {
    var min = x[xOff]
    var max = x[xOff]
    var sumW = if (weights != null) weights[wOff] else x[xOff] * x[xOff]
    var sumX = sumW * x[xOff]
    for (i in 1 until n) {
        if (x[xOff + i] < min) min = x[xOff + i]
        if (x[xOff + i] > max) max = x[xOff + i]
        val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
        sumW += w
        sumX += w * x[xOff + i]
    }
    if (min > 0) min = 0f
    if (max <= min) {
        for (i in 0 until n) L[lOff + i] = 0u
        theMin[minIdx] = -min
        return 0f
    }
    var iscale = nmax.toFloat() / (max - min)
    var scale = 1f / iscale
    var bestMad = 0f
    for (i in 0 until n) {
        var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * (x[xOff + i] - min))
        L[lOff + i] = maxOf(0, minOf(nmax, l)).toUByte()
        var diff = scale * L[lOff + i].toInt() + min - x[xOff + i]
        diff = if (useMad) kotlin.math.abs(diff) else diff * diff
        val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
        bestMad += w * diff
    }
    if (nstep < 1) {
        theMin[minIdx] = -min
        return scale
    }
    for (is_ in 0..nstep) {
        iscale = (rmin + rdelta * is_ + nmax) / (max - min)
        var sumL = 0f; var sumL2 = 0f; var sumXL = 0f
        for (i in 0 until n) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * (x[xOff + i] - min))
            l = maxOf(0, minOf(nmax, l))
            Laux[lauxOff + i] = l.toUByte()
            val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
            sumL += w * l
            sumL2 += w * l * l
            sumXL += w * l * x[xOff + i]
        }
        val D = sumW * sumL2 - sumL * sumL
        if (D > 0) {
            var thisScale = (sumW * sumXL - sumX * sumL) / D
            var thisMin = (sumL2 * sumX - sumL * sumXL) / D
            if (thisMin > 0) {
                thisMin = 0f
                thisScale = sumXL / sumL2
            }
            var mad = 0f
            for (i in 0 until n) {
                var diff = thisScale * Laux[lauxOff + i].toInt() + thisMin - x[xOff + i]
                diff = if (useMad) kotlin.math.abs(diff) else diff * diff
                val w = if (weights != null) weights[wOff + i] else x[xOff + i] * x[xOff + i]
                mad += w * diff
            }
            if (mad < bestMad) {
                for (i in 0 until n) L[lOff + i] = Laux[lauxOff + i]
                bestMad = mad
                scale = thisScale
                min = thisMin
            }
        }
    }
    theMin[minIdx] = -min
    return scale
}

/** Port of `make_qp_quants` (C line 1014). Positive-only quantization with iterative refinement. */
private fun make_qp_quants(
    n: Int, nmax: Int, x: FloatArray, xOff: Int,
    L: UByteArray, lOff: Int, quantWeights: FloatArray, qwOff: Int
): Float {
    var max = 0f
    for (i in 0 until n) {
        max = maxOf(max, x[xOff + i])
    }
    if (max < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
        for (i in 0 until n) L[lOff + i] = 0u
        return 0f
    }
    var iscale = nmax.toFloat() / max
    for (i in 0 until n) {
        L[lOff + i] = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i]).toUByte()
    }
    var scale = 1f / iscale
    var bestMse = 0f
    for (i in 0 until n) {
        val diff = x[xOff + i] - scale * L[lOff + i].toInt()
        val w = quantWeights[qwOff + i]
        bestMse += w * diff * diff
    }
    for (is_ in -4..4) {
        if (is_ == 0) continue
        val iscaleIs = (0.1f * is_ + nmax) / max
        val scaleIs = 1f / iscaleIs
        var mse = 0f
        for (i in 0 until n) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(iscaleIs * x[xOff + i])
            l = minOf(nmax, l)
            val diff = x[xOff + i] - scaleIs * l
            val w = quantWeights[qwOff + i]
            mse += w * diff * diff
        }
        if (mse < bestMse) {
            bestMse = mse
            iscale = iscaleIs
        }
    }
    var sumlx = 0f
    var suml2 = 0f
    for (i in 0 until n) {
        var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xOff + i])
        l = minOf(nmax, l)
        L[lOff + i] = l.toUByte()
        val w = quantWeights[qwOff + i]
        sumlx += w * x[xOff + i] * l
        suml2 += w * l * l
    }
    for (itry in 0 until 5) {
        var nChanged = 0
        for (i in 0 until n) {
            val w = quantWeights[qwOff + i]
            val slx = sumlx - w * x[xOff + i] * L[lOff + i].toInt()
            val sl2 = suml2 - w * L[lOff + i].toInt() * L[lOff + i].toInt()
            if (slx > 0 && sl2 > 0) {
                var newL = io.github.kotlinmania.llama.ore.nearest_int(x[xOff + i] * sl2 / slx)
                newL = minOf(nmax, newL)
                if (newL != L[lOff + i].toInt()) {
                    val slxNew = slx + w * x[xOff + i] * newL
                    val sl2New = sl2 + w * newL * newL
                    if (slxNew * slxNew * suml2 > sumlx * sumlx * sl2New) {
                        L[lOff + i] = newL.toUByte(); sumlx = slxNew; suml2 = sl2New
                        ++nChanged
                    }
                }
            }
        }
        if (nChanged == 0) break
    }
    return if (suml2 > 0f) sumlx / suml2 else 0f
}

// ════════════════════════════════════════════════════════════════════════════════
//  ByteArray block readers/writers — read/write fields at known offsets
// ════════════════════════════════════════════════════════════════════════════════

private fun readShortLE(data: ByteArray, off: Int): Short =
    ((data[off].toInt() and 0xFF) or ((data[off + 1].toInt() and 0xFF) shl 8)).toShort()

private fun readIntLE(data: ByteArray, off: Int): Int =
    (data[off].toInt() and 0xFF) or
    ((data[off + 1].toInt() and 0xFF) shl 8) or
    ((data[off + 2].toInt() and 0xFF) shl 16) or
    ((data[off + 3].toInt() and 0xFF) shl 24)

private fun writeShortLE(data: ByteArray, off: Int, v: Short) {
    data[off + 0] = (v.toInt() and 0xFF).toByte()
    data[off + 1] = ((v.toInt() shr 8) and 0xFF).toByte()
}

private fun writeFloatLE(data: ByteArray, off: Int, v: Float) {
    val bits = v.toRawBits()
    data[off + 0] = (bits and 0xFF).toByte()
    data[off + 1] = ((bits shr 8) and 0xFF).toByte()
    data[off + 2] = ((bits shr 16) and 0xFF).toByte()
    data[off + 3] = ((bits shr 24) and 0xFF).toByte()
}

private fun writeIntLE(data: ByteArray, off: Int, v: Int) {
    data[off + 0] = (v and 0xFF).toByte()
    data[off + 1] = ((v shr 8) and 0xFF).toByte()
    data[off + 2] = ((v shr 16) and 0xFF).toByte()
    data[off + 3] = ((v shr 24) and 0xFF).toByte()
}

private fun writeFp16(data: ByteArray, off: Int, v: Float) {
    io.github.kotlinmania.llama.ore.writeShortLE(
        data,
        off,
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(v)
    )
}

private fun readUShortLE(data: ByteArray, off: Int): Int =
    (data[off].toInt() and 0xFF) or ((data[off + 1].toInt() and 0xFF) shl 8)

private fun fp16ToF32(data: ByteArray, off: Int): Float =
    io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
        io.github.kotlinmania.llama.ore.readShortLE(
            data,
            off
        )
    )

// ════════════════════════════════════════════════════════════════════════════════
//  Dequantize functions
// ════════════════════════════════════════════════════════════════════════════════

// --- block_q1_0: d(2) + qs(QK1_0/8) ---
// QK1_0 = 64, block size = 2 + 8 = 10 bytes (but actual block size from GGMLCommon)

fun dequantize_row_q1_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK1_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 8  // d(2) + qs(qk/8)

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val negD = -d
        val qsOff = bOff + 2
        for (j in 0 until qk) {
            val byteIndex = j / 8
            val bitOffset = j % 8
            val bit = ((x[qsOff + byteIndex].toInt() and 0xFF) shr bitOffset) and 1
            y[yOff + i * qk + j] = if (bit != 0) d else negD
        }
    }
}

// --- block_q4_0: d(2) + qs(QK4_0/2) = 18 bytes ---

fun dequantize_row_q4_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK4_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val qsOff = bOff + 2
        for (j in 0 until qk / 2) {
            val x0 = (x[qsOff + j].toInt() and 0x0F) - 8
            val x1 = ((x[qsOff + j].toInt() and 0xFF) shr 4) - 8
            y[yOff + i * qk + j] = x0 * d
            y[yOff + i * qk + j + qk / 2] = x1 * d
        }
    }
}

// --- block_q4_1: d(2) + m(2) + qs(QK4_1/2) = 20 bytes ---

fun dequantize_row_q4_1(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK4_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val m = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 2)
        val qsOff = bOff + 4
        for (j in 0 until qk / 2) {
            val x0 = x[qsOff + j].toInt() and 0x0F
            val x1 = (x[qsOff + j].toInt() and 0xFF) shr 4
            y[yOff + i * qk + j] = x0 * d + m
            y[yOff + i * qk + j + qk / 2] = x1 * d + m
        }
    }
}

// --- block_q5_0: d(2) + qh(4) + qs(QK5_0/2) = 22 bytes ---

fun dequantize_row_q5_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK5_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + 4 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val qh = io.github.kotlinmania.llama.ore.readIntLE(x, bOff + 2).toLong() and 0xFFFFFFFFL
        val qsOff = bOff + 6
        for (j in 0 until qk / 2) {
            val xh0 = (((qh shr (j + 0)) shl 4) and 0x10L).toInt()
            val xh1 = (((qh shr (j + 12))) and 0x10L).toInt()
            val x0 = ((x[qsOff + j].toInt() and 0x0F) or xh0) - 16
            val x1 = (((x[qsOff + j].toInt() and 0xFF) shr 4) or xh1) - 16
            y[yOff + i * qk + j] = x0 * d
            y[yOff + i * qk + j + qk / 2] = x1 * d
        }
    }
}

// --- block_q5_1: d(2) + m(2) + qh(4) + qs(QK5_1/2) = 24 bytes ---

fun dequantize_row_q5_1(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK5_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + 4 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val m = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 2)
        val qh = io.github.kotlinmania.llama.ore.readIntLE(x, bOff + 4).toLong() and 0xFFFFFFFFL
        val qsOff = bOff + 8
        for (j in 0 until qk / 2) {
            val xh0 = (((qh shr (j + 0)) shl 4) and 0x10L).toInt()
            val xh1 = (((qh shr (j + 12))) and 0x10L).toInt()
            val x0 = (x[qsOff + j].toInt() and 0x0F) or xh0
            val x1 = ((x[qsOff + j].toInt() and 0xFF) shr 4) or xh1
            y[yOff + i * qk + j] = x0 * d + m
            y[yOff + i * qk + j + qk / 2] = x1 * d + m
        }
    }
}

// --- block_q8_0: d(2) + qs(QK8_0) = 34 bytes ---

fun dequantize_row_q8_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val qsOff = bOff + 2
        for (j in 0 until qk) {
            y[yOff + i * qk + j] = x[qsOff + j].toInt() * d
        }
    }
}

// --- block_mxfp4: e(1) + qs(QK_MXFP4/2) ---

fun dequantize_row_mxfp4(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK_MXFP4
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 1 + qk / 2

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d =
            io.github.kotlinmania.llama.ore.GGML_E8M0_TO_FP32_HALF((x[bOff].toInt() and 0xFF).toUByte())
        val qsOff = bOff + 1
        for (j in 0 until qk / 2) {
            val x0 = io.github.kotlinmania.llama.ore.kvalues_mxfp4[(x[qsOff + j].toInt() and 0xFF) and 0x0F]
            val x1 = io.github.kotlinmania.llama.ore.kvalues_mxfp4[(x[qsOff + j].toInt() and 0xFF) shr 4]
            y[yOff + i * qk + j] = x0 * d
            y[yOff + i * qk + j + qk / 2] = x1 * d
        }
    }
}

// --- block_nvfp4: d(QK_NVFP4/QK_NVFP4_SUB) + qs(QK_NVFP4/2) ---

fun dequantize_row_nvfp4(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK_NVFP4
    val qkSub = io.github.kotlinmania.llama.ore.QK_NVFP4_SUB
    val nSub = qk / qkSub
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = nSub + qk / 2  // d bytes (nSub UE4M3) + qs bytes

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        for (s in 0 until nSub) {
            val d =
                io.github.kotlinmania.llama.ore.ggml_ue4m3_to_fp32((x[bOff + s].toInt() and 0xFF).toUByte())
            val yBase = yOff + i * qk + s * qkSub
            val qsBase = bOff + nSub + s * (qkSub / 2)
            for (j in 0 until qkSub / 2) {
                val v0 = io.github.kotlinmania.llama.ore.kvalues_mxfp4[(x[qsBase + j].toInt() and 0xFF) and 0x0F]
                val v1 = io.github.kotlinmania.llama.ore.kvalues_mxfp4[(x[qsBase + j].toInt() and 0xFF) shr 4]
                y[yBase + j] = v0 * d
                y[yBase + j + qkSub / 2] = v1 * d
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  K-Quant dequantize
// ════════════════════════════════════════════════════════════════════════════════

// --- block_q2_K: scales(16) + qs(64) + d(2) + dmin(2) = 84 bytes ---
// Layout: qs[QK_K/4]=64, scales[QK_K/16]=16, d(2), dmin(2)
// Actual layout from ggml-common.h: scales(16), qs(64), d(2), dmin(2)

fun dequantize_row_q2_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ2K.SIZE_BYTES

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: scales(16) at 0, qs(64) at 16, d(2) at 80, dmin(2) at 82
        val scalesOff = bOff + 0
        val qsOff = bOff + 16
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 80)
        val min = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 82)

        var qIdx = qsOff
        var is_ = 0
        for (n in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            var shift = 0
            for (j in 0 until 4) {
                val sc = x[scalesOff + is_].toInt() and 0xFF; is_++
                val dl = d * (sc and 0xF)
                val ml = min * (sc shr 4)
                for (l in 0 until 16) {
                    y[yIdx++] = dl * (((x[qIdx + l].toInt() and 0xFF) shr shift) and 3) - ml
                }

                val sc2 = x[scalesOff + is_].toInt() and 0xFF; is_++
                val dl2 = d * (sc2 and 0xF)
                val ml2 = min * (sc2 shr 4)
                for (l in 0 until 16) {
                    y[yIdx++] = dl2 * (((x[qIdx + 16 + l].toInt() and 0xFF) shr shift) and 3) - ml2
                }
                shift += 2
            }
            qIdx += 32
        }
    }
}

// --- block_q3_K: hmask(32) + qs(64) + scales(12) + d(2) = 110 bytes ---

fun dequantize_row_q3_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ3K.SIZE_BYTES

    val kmask1 = 0x03030303
    val kmask2 = 0x0f0f0f0f

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: hmask(32) at 0, qs(64) at 32, scales(12) at 96, d(2) at 108
        val hmOff = bOff + 0
        val qsOff = bOff + 32
        val scOff = bOff + 96
        val dAll = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 108)

        // Decode scales from 12 bytes into 16 int8 values via aux[4]
        val aux = IntArray(4)
        aux[0] = io.github.kotlinmania.llama.ore.readIntLE(x, scOff + 0)
        aux[1] = io.github.kotlinmania.llama.ore.readIntLE(x, scOff + 4)
        aux[2] = io.github.kotlinmania.llama.ore.readIntLE(x, scOff + 8)
        val tmp = aux[2]
        aux[2] = ((aux[0] shr 4) and kmask2) or (((tmp shr 4) and kmask1) shl 4)
        aux[3] = ((aux[1] shr 4) and kmask2) or (((tmp shr 6) and kmask1) shl 4)
        aux[0] = (aux[0] and kmask2) or (((tmp shr 0) and kmask1) shl 4)
        aux[1] = (aux[1] and kmask2) or (((tmp shr 2) and kmask1) shl 4)

        // Convert aux[4] to scales[16] (int8 values)
        val scales = ByteArray(16)
        for (si in 0 until 4) {
            scales[si * 4 + 0] = (aux[si] and 0xFF).toByte()
            scales[si * 4 + 1] = ((aux[si] shr 8) and 0xFF).toByte()
            scales[si * 4 + 2] = ((aux[si] shr 16) and 0xFF).toByte()
            scales[si * 4 + 3] = ((aux[si] shr 24) and 0xFF).toByte()
        }

        var yIdx = yOff + i * io.github.kotlinmania.llama.ore.QK_K
        var qIdx = qsOff
        var m = 1
        var is_ = 0
        for (n in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            var shift = 0
            for (j in 0 until 4) {
                val dl = dAll * (scales[is_].toInt() - 32); is_++
                for (l in 0 until 16) {
                    y[yIdx++] = dl * (((x[qIdx + l].toInt() and 0xFF) shr shift) and 3).let { v ->
                        v - (if ((x[hmOff + l].toInt() and 0xFF) and m != 0) 0 else 4)
                    }
                }
                val dl2 = dAll * (scales[is_].toInt() - 32); is_++
                for (l in 0 until 16) {
                    y[yIdx++] = dl2 * (((x[qIdx + 16 + l].toInt() and 0xFF) shr shift) and 3).let { v ->
                        v - (if ((x[hmOff + 16 + l].toInt() and 0xFF) and m != 0) 0 else 4)
                    }
                }
                shift += 2
                m = m shl 1
            }
            qIdx += 32
        }
    }
}

// --- block_q4_K: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes ---

fun dequantize_row_q4_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ4K.SIZE_BYTES

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(2) at 0, dmin(2) at 2, scales(12) at 4, qs(128) at 16
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val min = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 2)
        val scOff = bOff + 4
        var qIdx = bOff + 16
        var yIdx = yOff + i * io.github.kotlinmania.llama.ore.QK_K

        var is_ = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
            val sm1 = io.github.kotlinmania.llama.ore.get_scale_min_k4(is_ + 0, x, scOff)
            val d1 = d * sm1.d; val m1 = min * sm1.m
            val sm2 = io.github.kotlinmania.llama.ore.get_scale_min_k4(is_ + 1, x, scOff)
            val d2 = d * sm2.d; val m2 = min * sm2.m
            for (l in 0 until 32) {
                y[yIdx++] = d1 * ((x[qIdx + l].toInt() and 0xFF) and 0xF) - m1
            }
            for (l in 0 until 32) {
                y[yIdx++] = d2 * ((x[qIdx + l].toInt() and 0xFF) shr 4) - m2
            }
            qIdx += 32; is_ += 2
        }
    }
}

// --- block_q5_K: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 bytes ---

fun dequantize_row_q5_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ5K.SIZE_BYTES

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(2) at 0, dmin(2) at 2, scales(12) at 4, qh(32) at 16, qs(128) at 48
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val min = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 2)
        val scOff = bOff + 4
        val qhOff = bOff + 16
        var qlIdx = bOff + 48
        var yIdx = yOff + i * io.github.kotlinmania.llama.ore.QK_K

        var is_ = 0
        var u1 = 1; var u2 = 2
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
            val sm1 = io.github.kotlinmania.llama.ore.get_scale_min_k4(is_ + 0, x, scOff)
            val d1 = d * sm1.d; val m1 = min * sm1.m
            val sm2 = io.github.kotlinmania.llama.ore.get_scale_min_k4(is_ + 1, x, scOff)
            val d2 = d * sm2.d; val m2 = min * sm2.m
            for (l in 0 until 32) {
                val ql = x[qlIdx + l].toInt() and 0xFF
                val qh = x[qhOff + l].toInt() and 0xFF
                y[yIdx++] = d1 * ((ql and 0xF) + (if (qh and u1 != 0) 16 else 0)) - m1
            }
            for (l in 0 until 32) {
                val ql = x[qlIdx + l].toInt() and 0xFF
                val qh = x[qhOff + l].toInt() and 0xFF
                y[yIdx++] = d2 * ((ql shr 4) + (if (qh and u2 != 0) 16 else 0)) - m2
            }
            qlIdx += 32; is_ += 2
            u1 = u1 shl 2; u2 = u2 shl 2
        }
    }
}

// --- block_q6_K: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes ---

fun dequantize_row_q6_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ6K.SIZE_BYTES

    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: ql(128) at 0, qh(64) at 128, scales(16) at 192, d(2) at 208
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff + 208)
        var qlIdx = bOff + 0
        var qhIdx = bOff + 128
        var scIdx = bOff + 192
        var yIdx = yOff + i * io.github.kotlinmania.llama.ore.QK_K

        for (n in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                val isv = l / 16
                val qlLo = x[qlIdx + l].toInt() and 0xFF
                val qlHi = x[qlIdx + l + 32].toInt() and 0xFF
                val qhv = x[qhIdx + l].toInt() and 0xFF
                val q1 = ((qlLo and 0xF) or (((qhv shr 0) and 3) shl 4)) - 32
                val q2 = ((qlHi and 0xF) or (((qhv shr 2) and 3) shl 4)) - 32
                val q3 = ((qlLo shr 4) or (((qhv shr 4) and 3) shl 4)) - 32
                val q4 = ((qlHi shr 4) or (((qhv shr 6) and 3) shl 4)) - 32
                y[yIdx + l + 0]  = d * x[scIdx + isv + 0].toInt() * q1
                y[yIdx + l + 32] = d * x[scIdx + isv + 2].toInt() * q2
                y[yIdx + l + 64] = d * x[scIdx + isv + 4].toInt() * q3
                y[yIdx + l + 96] = d * x[scIdx + isv + 6].toInt() * q4
            }
            yIdx += 128; qlIdx += 64; qhIdx += 32; scIdx += 8
        }
    }
}

// --- block_q8_K: d(4 float) + qs(256) + bsums(32) = 292 bytes ---

fun dequantize_row_q8_K(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(4 bytes float) at 0, qs(256) at 4
        val d = Float.fromBits(io.github.kotlinmania.llama.ore.readIntLE(x, bOff))
        val qsOff = bOff + 4
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) {
            y[yIdx++] = d * x[qsOff + j].toInt()
        }
    }
}

// --- block_tq1_0 ---

fun dequantize_row_tq1_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockTQ1_0.SIZE_BYTES
    val pow3 = intArrayOf(1, 3, 9, 27, 81, 243)

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // block_tq1_0: qs[(QK_K-4*QK_K/64)/5] + qh[QK_K/64] + d(2)
        val qsSize = (io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5
        val qhSize = io.github.kotlinmania.llama.ore.QK_K / 64
        val dOff = bOff + qsSize + qhSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, dOff)

        // Process qs in 32-byte chunks
        val fullChunks = qsSize - qsSize % 32
        var qsIdx = bOff
        for (j in 0 until fullChunks step 32) {
            for (n in 0 until 5) {
                for (m in 0 until 32) {
                    val q = ((x[qsIdx + m].toInt() and 0xFF) * pow3[n]) and 0xFF
                    val xi = ((q * 3) shr 8)
                    y[yIdx++] = (xi - 1).toFloat() * d
                }
            }
            qsIdx += 32
        }
        for (j in fullChunks until qsSize step 16) {
            for (n in 0 until 5) {
                for (m in 0 until 16) {
                    val q = ((x[qsIdx + m].toInt() and 0xFF) * pow3[n]) and 0xFF
                    val xi = ((q * 3) shr 8)
                    y[yIdx++] = (xi - 1).toFloat() * d
                }
            }
            qsIdx += 16
        }

        // Process qh
        val qhOff = bOff + qsSize
        for (n in 0 until 4) {
            for (j in 0 until qhSize) {
                val q = ((x[qhOff + j].toInt() and 0xFF) * pow3[n]) and 0xFF
                val xi = ((q * 3) shr 8)
                y[yIdx++] = (xi - 1).toFloat() * d
            }
        }
    }
}

// --- block_tq2_0 ---

fun dequantize_row_tq2_0(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockTQ2_0.SIZE_BYTES
    val qsSize = io.github.kotlinmania.llama.ore.QK_K / 4

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val dOff = bOff + qsSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, dOff)

        var qsIdx = bOff
        for (j in 0 until qsSize step 32) {
            for (l in 0 until 4) {
                for (m in 0 until 32) {
                    val q = ((x[qsIdx + m].toInt() and 0xFF) shr (l * 2)) and 3
                    y[yIdx++] = (q - 1).toFloat() * d
                }
            }
            qsIdx += 32
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  IQ Dequantize — require grid lookup tables
// ════════════════════════════════════════════════════════════════════════════════

fun dequantize_row_iq4_nl(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK4_NL == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK4_NL).toInt()
    val blockSize = 2 + io.github.kotlinmania.llama.ore.QK4_NL / 2  // d(2) + qs(QK4_NL/2)

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val qsOff = bOff + 2
        for (j in 0 until io.github.kotlinmania.llama.ore.QK4_NL / 2) {
            val qByte = x[qsOff + j].toInt() and 0xFF
            y[yIdx + j] = d * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qByte and 0xf]
            y[yIdx + j + io.github.kotlinmania.llama.ore.QK4_NL / 2] = d * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qByte shr 4]
        }
        yIdx += io.github.kotlinmania.llama.ore.QK4_NL
    }
}

fun dequantize_row_iq4_xs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ4XS.SIZE_BYTES

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: d(2) at 0, scales_h(2) at 2, scales_l(QK_K/64) at 4, qs(QK_K/2) at 4+QK_K/64
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        val scalesH = io.github.kotlinmania.llama.ore.readUShortLE(x, bOff + 2)
        val slOff = bOff + 4
        val qsOff = slOff + io.github.kotlinmania.llama.ore.QK_K / 64

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val slByte = x[slOff + ib / 2].toInt() and 0xFF
            val ls = ((slByte shr (4 * (ib % 2))) and 0xf) or (((scalesH shr (2 * ib)) and 3) shl 4)
            val dl = d * (ls - 32)
            val qBase = qsOff + ib * 16
            for (j in 0 until 16) {
                val qByte = x[qBase + j].toInt() and 0xFF
                y[yIdx + j + 0]  = dl * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qByte and 0xf]
                y[yIdx + j + 16] = dl * io.github.kotlinmania.llama.ore.kvalues_iq4nl[qByte shr 4]
            }
            yIdx += 32
        }
    }
}

// IQ dequantize functions — ported from ggml-quants.c

// Helper: extract byte j from a ULong grid entry (treating it as uint8_t[8])
private fun gridByteU(grid: ULong, j: Int): Int = ((grid shr (j * 8)) and 0xFFu).toInt()

// Helper: extract byte j from a ULong grid entry as signed int8
private fun gridByteS(grid: ULong, j: Int): Int = ((grid shr (j * 8)) and 0xFFu).toInt().toByte().toInt()

// Helper: extract byte j from a UInt grid entry (treating it as uint8_t[4])
private fun gridByteU(grid: UInt, j: Int): Int = ((grid shr (j * 8)) and 0xFFu).toInt()

fun dequantize_row_iq2_xxs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ2XXS.SIZE_BYTES // 66

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        // qs is uint16_t[QK_K/8] starting at offset 2
        val qsOff = bOff + 2

        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            // memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t))
            // 4*ib32 uint16_t entries = 8*ib32 bytes
            val a0off = qsOff + 8 * ib32
            val aux32_0 = io.github.kotlinmania.llama.ore.readIntLE(x, a0off)
            val aux32_1 = io.github.kotlinmania.llama.ore.readIntLE(x, a0off + 4)
            val db = d * (0.5f + (aux32_1 ushr 28)) * 0.25f
            for (l in 0 until 4) {
                // aux8[l] = byte l of aux32_0/aux32_1 combined
                val aux8l = (aux32_0 ushr (l * 8)) and 0xFF
                val grid = io.github.kotlinmania.llama.ore.iq2xxs_grid[aux8l]
                val signs = io.github.kotlinmania.llama.ore.ksigns_iq2xs[((aux32_1 ushr (7 * l)) and 127)].toInt() and 0xFF
                for (j in 0 until 8) {
                    val g = io.github.kotlinmania.llama.ore.gridByteU(grid, j)
                    val mask = io.github.kotlinmania.llama.ore.kmask_iq2xs[j].toInt() and 0xFF
                    y[yIdx + j] = db * g * (if (signs and mask != 0) -1f else 1f)
                }
                yIdx += 8
            }
        }
    }
}

fun dequantize_row_iq2_xs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ2XS.SIZE_BYTES // 74

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        // Layout: d(2), qs[QK_K/8] as uint16_t (64 bytes) at 2, scales[QK_K/32] (8 bytes) at 66
        val qsOff = bOff + 2
        val scOff = bOff + 2 + (io.github.kotlinmania.llama.ore.QK_K / 8) * 2 // = bOff + 66

        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val scByte = x[scOff + ib32].toInt() and 0xFF
            val db0 = d * (0.5f + (scByte and 0xf)) * 0.25f
            val db1 = d * (0.5f + (scByte shr 4)) * 0.25f
            for (l in 0 until 4) {
                // x[i].qs[4*ib32 + l] is uint16_t — read 2 bytes LE
                val qval =
                    io.github.kotlinmania.llama.ore.readUShortLE(x, qsOff + (4 * ib32 + l) * 2)
                val grid = io.github.kotlinmania.llama.ore.iq2xs_grid[qval and 511]
                val signs = io.github.kotlinmania.llama.ore.ksigns_iq2xs[(qval shr 9)].toInt() and 0xFF
                for (j in 0 until 8) {
                    val g = io.github.kotlinmania.llama.ore.gridByteU(grid, j)
                    val mask = io.github.kotlinmania.llama.ore.kmask_iq2xs[j].toInt() and 0xFF
                    y[yIdx + j] = (if (l < 2) db0 else db1) * g * (if (signs and mask != 0) -1f else 1f)
                }
                yIdx += 8
            }
        }
    }
}

fun dequantize_row_iq2_s(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ2S.SIZE_BYTES // 82

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        // Layout: d(2), qs[QK_K/4](64) at 2, qh[QK_K/32](8) at 66, scales[QK_K/32](8) at 74
        val qhOff = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4        // = bOff + 66
        val scOff = qhOff + io.github.kotlinmania.llama.ore.QK_K / 32           // = bOff + 74
        var qsPtr = bOff + 2                    // qs pointer (advances by 4 per ib32)
        var signsPtr = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 8      // signs = qs + QK_K/8 (advances by 4 per ib32)

        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val scByte = x[scOff + ib32].toInt() and 0xFF
            val db0 = d * (0.5f + (scByte and 0xf)) * 0.25f
            val db1 = d * (0.5f + (scByte shr 4)) * 0.25f
            for (l in 0 until 4) {
                val dl = if (l < 2) db0 else db1
                val qsByte = x[qsPtr + l].toInt() and 0xFF
                val qhByte = x[qhOff + ib32].toInt() and 0xFF
                // qs[l] | (qh[ib32] << (8 - 2*l) & 0x300)
                val gridIdx = qsByte or ((qhByte shl (8 - 2 * l)) and 0x300)
                val grid = io.github.kotlinmania.llama.ore.iq2s_grid[gridIdx]
                val signsByte = x[signsPtr + l].toInt() and 0xFF
                for (j in 0 until 8) {
                    val g = io.github.kotlinmania.llama.ore.gridByteU(grid, j)
                    val mask = io.github.kotlinmania.llama.ore.kmask_iq2xs[j].toInt() and 0xFF
                    y[yIdx + j] = dl * g * (if (signsByte and mask != 0) -1f else 1f)
                }
                yIdx += 8
            }
            qsPtr += 4
            signsPtr += 4
        }
    }
}

fun dequantize_row_iq3_xxs(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ3XXS.SIZE_BYTES // 98

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        // Layout: d(2), qs[3*QK_K/8] at 2
        // qs = first QK_K/4 = 64 bytes of qs data
        // scales_and_signs = qs + QK_K/4 (= the remaining 32 bytes)
        var qsPtr = bOff + 2
        val sasOff = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4  // scales_and_signs

        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val aux32 = io.github.kotlinmania.llama.ore.readIntLE(x, sasOff + 4 * ib32)
            val db = d * (0.5f + (aux32 ushr 28)) * 0.5f
            for (l in 0 until 4) {
                val signs = io.github.kotlinmania.llama.ore.ksigns_iq2xs[((aux32 ushr (7 * l)) and 127)].toInt() and 0xFF
                val grid1 = io.github.kotlinmania.llama.ore.iq3xxs_grid[x[qsPtr + 2 * l + 0].toInt() and 0xFF]
                val grid2 = io.github.kotlinmania.llama.ore.iq3xxs_grid[x[qsPtr + 2 * l + 1].toInt() and 0xFF]
                for (j in 0 until 4) {
                    val mask0 = io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 0].toInt() and 0xFF
                    val mask4 = io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 4].toInt() and 0xFF
                    y[yIdx + j + 0] = db * io.github.kotlinmania.llama.ore.gridByteU(grid1, j) * (if (signs and mask0 != 0) -1f else 1f)
                    y[yIdx + j + 4] = db * io.github.kotlinmania.llama.ore.gridByteU(grid2, j) * (if (signs and mask4 != 0) -1f else 1f)
                }
                yIdx += 8
            }
            qsPtr += 8
        }
    }
}

fun dequantize_row_iq3_s(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ3S.SIZE_BYTES // 110

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        // Layout: d(2), qs[QK_K/4](64) at 2, qh[QK_K/32](8) at 66, signs[QK_K/8](32) at 74, scales[IQ3S_N_SCALE](4) at 106
        var qsPtr = bOff + 2
        var qhPtr = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4          // = bOff + 66
        var signsPtr = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 32  // = bOff + 74
        val scOff = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 32 + io.github.kotlinmania.llama.ore.QK_K / 8  // = bOff + 106

        for (ib32 in 0 until io.github.kotlinmania.llama.ore.QK_K / 32 step 2) {
            val scByte = x[scOff + ib32 / 2].toInt() and 0xFF
            val db1 = d * (1 + 2 * (scByte and 0xf))
            val db2 = d * (1 + 2 * (scByte shr 4))
            val qh0 = x[qhPtr + 0].toInt() and 0xFF

            // First 32 values (ib32)
            for (l in 0 until 4) {
                val grid1Idx = (x[qsPtr + 2 * l + 0].toInt() and 0xFF) or ((qh0 shl (8 - 2 * l)) and 256)
                val grid2Idx = (x[qsPtr + 2 * l + 1].toInt() and 0xFF) or ((qh0 shl (7 - 2 * l)) and 256)
                val grid1 = io.github.kotlinmania.llama.ore.iq3s_grid[grid1Idx]
                val grid2 = io.github.kotlinmania.llama.ore.iq3s_grid[grid2Idx]
                val signsVal = x[signsPtr + l].toInt() and 0xFF
                for (j in 0 until 4) {
                    val mask0 = io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 0].toInt() and 0xFF
                    val mask4 = io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 4].toInt() and 0xFF
                    y[yIdx + j + 0] = db1 * io.github.kotlinmania.llama.ore.gridByteU(grid1, j) * (if (signsVal and mask0 != 0) -1f else 1f)
                    y[yIdx + j + 4] = db1 * io.github.kotlinmania.llama.ore.gridByteU(grid2, j) * (if (signsVal and mask4 != 0) -1f else 1f)
                }
                yIdx += 8
            }
            qsPtr += 8
            signsPtr += 4

            // Second 32 values (ib32 + 1)
            val qh1 = x[qhPtr + 1].toInt() and 0xFF
            for (l in 0 until 4) {
                val grid1Idx = (x[qsPtr + 2 * l + 0].toInt() and 0xFF) or ((qh1 shl (8 - 2 * l)) and 256)
                val grid2Idx = (x[qsPtr + 2 * l + 1].toInt() and 0xFF) or ((qh1 shl (7 - 2 * l)) and 256)
                val grid1 = io.github.kotlinmania.llama.ore.iq3s_grid[grid1Idx]
                val grid2 = io.github.kotlinmania.llama.ore.iq3s_grid[grid2Idx]
                val signsVal = x[signsPtr + l].toInt() and 0xFF
                for (j in 0 until 4) {
                    val mask0 = io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 0].toInt() and 0xFF
                    val mask4 = io.github.kotlinmania.llama.ore.kmask_iq2xs[j + 4].toInt() and 0xFF
                    y[yIdx + j + 0] = db2 * io.github.kotlinmania.llama.ore.gridByteU(grid1, j) * (if (signsVal and mask0 != 0) -1f else 1f)
                    y[yIdx + j + 4] = db2 * io.github.kotlinmania.llama.ore.gridByteU(grid2, j) * (if (signsVal and mask4 != 0) -1f else 1f)
                }
                yIdx += 8
            }
            qhPtr += 2
            qsPtr += 8
            signsPtr += 4
        }
    }
}

fun dequantize_row_iq1_s(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ1S.SIZE_BYTES // 50

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        val d = io.github.kotlinmania.llama.ore.fp16ToF32(x, bOff)
        // Layout: d(2), qs[QK_K/8](32) at 2, qh[QK_K/32] as uint16_t (16 bytes) at 34
        var qsPtr = bOff + 2
        val qhOff = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 8  // = bOff + 34

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val qhVal = io.github.kotlinmania.llama.ore.readUShortLE(x, qhOff + ib * 2)
            val dl = d * (2 * ((qhVal shr 12) and 7) + 1)
            val delta = if (qhVal and 0x8000 != 0) -io.github.kotlinmania.llama.ore.IQ1S_DELTA else io.github.kotlinmania.llama.ore.IQ1S_DELTA
            for (l in 0 until 4) {
                val gridIdx = (x[qsPtr + l].toInt() and 0xFF) or (((qhVal shr (3 * l)) and 7) shl 8)
                val grid = io.github.kotlinmania.llama.ore.iq1s_grid[gridIdx]
                for (j in 0 until 8) {
                    y[yIdx + j] = dl * (io.github.kotlinmania.llama.ore.gridByteS(grid, j) + delta)
                }
                yIdx += 8
            }
            qsPtr += 4
        }
    }
}

fun dequantize_row_iq1_m(x: ByteArray, xOff: Int, y: FloatArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ1M.SIZE_BYTES // 56

    var yIdx = yOff
    for (i in 0 until nb) {
        val bOff = xOff + i * blockSize
        // Layout: qs[QK_K/8](32) at 0, qh[QK_K/16](16) at 32, scales[QK_K/32](8) at 48
        val qhBase = bOff + io.github.kotlinmania.llama.ore.QK_K / 8         // = bOff + 32
        val scBase = bOff + io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 16  // = bOff + 48

        // Reconstruct fp16 scale from 4-bit pieces of scales[]
        // sc = (const uint16_t *)x[i].scales  (reading scales as uint16_t[4])
        val sc0 = io.github.kotlinmania.llama.ore.readUShortLE(x, scBase + 0)
        val sc1 = io.github.kotlinmania.llama.ore.readUShortLE(x, scBase + 2)
        val sc2 = io.github.kotlinmania.llama.ore.readUShortLE(x, scBase + 4)
        val sc3 = io.github.kotlinmania.llama.ore.readUShortLE(x, scBase + 6)
        val scaleU16 = (sc0 shr 12) or ((sc1 shr 8) and 0x00f0) or ((sc2 shr 4) and 0x0f00) or (sc3 and 0xf000)
        val d = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(scaleU16.toShort())

        var qsPtr = bOff
        var qhPtr = qhBase

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            // sc[ib/2] read as uint16_t
            val scVal = io.github.kotlinmania.llama.ore.readUShortLE(x, scBase + (ib / 2) * 2)
            val dl1 = d * (2 * ((scVal shr (6 * (ib % 2) + 0)) and 0x7) + 1)
            val dl2 = d * (2 * ((scVal shr (6 * (ib % 2) + 3)) and 0x7) + 1)

            val qh0 = x[qhPtr + 0].toInt() and 0xFF
            val qh1 = x[qhPtr + 1].toInt() and 0xFF

            val idx0 = (x[qsPtr + 0].toInt() and 0xFF) or ((qh0 shl 8) and 0x700)
            val idx1 = (x[qsPtr + 1].toInt() and 0xFF) or ((qh0 shl 4) and 0x700)
            val idx2 = (x[qsPtr + 2].toInt() and 0xFF) or ((qh1 shl 8) and 0x700)
            val idx3 = (x[qsPtr + 3].toInt() and 0xFF) or ((qh1 shl 4) and 0x700)

            val delta0 = if (qh0 and 0x08 != 0) -io.github.kotlinmania.llama.ore.IQ1M_DELTA else io.github.kotlinmania.llama.ore.IQ1M_DELTA
            val delta1 = if (qh0 and 0x80 != 0) -io.github.kotlinmania.llama.ore.IQ1M_DELTA else io.github.kotlinmania.llama.ore.IQ1M_DELTA
            val delta2 = if (qh1 and 0x08 != 0) -io.github.kotlinmania.llama.ore.IQ1M_DELTA else io.github.kotlinmania.llama.ore.IQ1M_DELTA
            val delta3 = if (qh1 and 0x80 != 0) -io.github.kotlinmania.llama.ore.IQ1M_DELTA else io.github.kotlinmania.llama.ore.IQ1M_DELTA

            // First two sub-blocks use dl1
            for (l in 0 until 2) {
                val idx = if (l == 0) idx0 else idx1
                val delta = if (l == 0) delta0 else delta1
                val grid = io.github.kotlinmania.llama.ore.iq1s_grid[idx]
                for (j in 0 until 8) {
                    y[yIdx + j] = dl1 * (io.github.kotlinmania.llama.ore.gridByteS(grid, j) + delta)
                }
                yIdx += 8
            }
            // Last two sub-blocks use dl2
            for (l in 2 until 4) {
                val idx = if (l == 2) idx2 else idx3
                val delta = if (l == 2) delta2 else delta3
                val grid = io.github.kotlinmania.llama.ore.iq1s_grid[idx]
                for (j in 0 until 8) {
                    y[yIdx + j] = dl2 * (io.github.kotlinmania.llama.ore.gridByteS(grid, j) + delta)
                }
                yIdx += 8
            }
            qsPtr += 4
            qhPtr += 2
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Quantize-row-ref functions (reference implementations)
// ════════════════════════════════════════════════════════════════════════════════

fun quantize_row_q4_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK4_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 2

    for (i in 0 until nb) {
        var amax = 0f
        var max = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            val v = x[base + j]
            if (amax < kotlin.math.abs(v)) { amax = kotlin.math.abs(v); max = v }
        }
        val d = max / -8f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        // Write d as fp16
        val dFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 2
        for (j in 0 until qk / 2) {
            val x0 = x[base + j] * id
            val x1 = x[base + j + qk / 2] * id
            val xi0 = minOf(15, (x0 + 8.5f).toInt())
            val xi1 = minOf(15, (x1 + 8.5f).toInt())
            y[qsOff + j] = (xi0 or (xi1 shl 4)).toByte()
        }
    }
}

fun quantize_row_q8_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK8_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk

    for (i in 0 until nb) {
        var amax = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            amax = maxOf(amax, kotlin.math.abs(x[base + j]))
        }
        val d = amax / 127f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 2
        for (j in 0 until qk) {
            y[qsOff + j] = (x[base + j] * id + 0.5f).toInt().coerceIn(-128, 127).toByte()
        }
    }
}

fun quantize_row_q4_1_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK4_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + qk / 2

    for (i in 0 until nb) {
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        val base = xOff + i * qk
        for (j in 0 until qk) {
            if (x[base + j] < min) min = x[base + j]
            if (x[base + j] > max) max = x[base + j]
        }
        val d = (max - min) / 15f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val mFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(min)
        y[bOff + 2] = (mFp16.toInt() and 0xFF).toByte()
        y[bOff + 3] = ((mFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 4
        for (j in 0 until qk / 2) {
            val x0 = (x[base + j] - min) * id
            val x1 = (x[base + j + qk / 2] - min) * id
            val xi0 = minOf(15, (x0 + 0.5f).toInt())
            val xi1 = minOf(15, (x1 + 0.5f).toInt())
            y[qsOff + j] = (xi0 or (xi1 shl 4)).toByte()
        }
    }
}

fun quantize_row_q5_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK5_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + 4 + qk / 2

    for (i in 0 until nb) {
        var amax = 0f; var max = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            val v = x[base + j]
            if (amax < kotlin.math.abs(v)) { amax = kotlin.math.abs(v); max = v }
        }
        val d = max / -16f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        // qh at bOff+2..bOff+5
        var qh = 0
        val qsOff = bOff + 6
        for (j in 0 until qk / 2) {
            val x0 = x[base + j] * id
            val x1 = x[base + j + qk / 2] * id
            val xi0 = minOf(31, (x0 + 16.5f).toInt())
            val xi1 = minOf(31, (x1 + 16.5f).toInt())
            y[qsOff + j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
            qh = qh or (((xi0 and 0x10) shr 4) shl (j + 0))
            qh = qh or (((xi1 and 0x10) shr 4) shl (j + qk / 2))
        }
        y[bOff + 2] = (qh and 0xFF).toByte()
        y[bOff + 3] = ((qh shr 8) and 0xFF).toByte()
        y[bOff + 4] = ((qh shr 16) and 0xFF).toByte()
        y[bOff + 5] = ((qh shr 24) and 0xFF).toByte()
    }
}

fun quantize_row_q5_1_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK5_1
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 4 + 4 + qk / 2

    for (i in 0 until nb) {
        var min = Float.MAX_VALUE; var max = -Float.MAX_VALUE
        val base = xOff + i * qk
        for (j in 0 until qk) {
            if (x[base + j] < min) min = x[base + j]
            if (x[base + j] > max) max = x[base + j]
        }
        val d = (max - min) / 31f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val mFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(min)
        y[bOff + 2] = (mFp16.toInt() and 0xFF).toByte()
        y[bOff + 3] = ((mFp16.toInt() shr 8) and 0xFF).toByte()
        var qh = 0
        val qsOff = bOff + 8
        for (j in 0 until qk / 2) {
            val x0 = (x[base + j] - min) * id
            val x1 = (x[base + j + qk / 2] - min) * id
            val xi0 = minOf(31, (x0 + 0.5f).toInt())
            val xi1 = minOf(31, (x1 + 0.5f).toInt())
            y[qsOff + j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
            qh = qh or (((xi0 and 0x10) shr 4) shl (j + 0))
            qh = qh or (((xi1 and 0x10) shr 4) shl (j + qk / 2))
        }
        y[bOff + 4] = (qh and 0xFF).toByte()
        y[bOff + 5] = ((qh shr 8) and 0xFF).toByte()
        y[bOff + 6] = ((qh shr 16) and 0xFF).toByte()
        y[bOff + 7] = ((qh shr 24) and 0xFF).toByte()
    }
}

fun quantize_row_q8_1_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK8_0  // Q8_1 uses same QK as Q8_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    // block_q8_1: d(2) + s(2) + qs(32) = 36 bytes
    val blockSize = 4 + qk

    for (i in 0 until nb) {
        var amax = 0f
        val base = xOff + i * qk
        for (j in 0 until qk) {
            amax = maxOf(amax, kotlin.math.abs(x[base + j]))
        }
        val d = amax / 127f
        val id = if (d != 0f) 1f / d else 0f
        val bOff = yOff + i * blockSize
        val dFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        var sum = 0
        val qsOff = bOff + 4
        for (j in 0 until qk) {
            val v = (x[base + j] * id + 0.5f).toInt().coerceIn(-128, 127)
            y[qsOff + j] = v.toByte()
            sum += v
        }
        // s = d * sum
        val sFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d * sum)
        y[bOff + 2] = (sFp16.toInt() and 0xFF).toByte()
        y[bOff + 3] = ((sFp16.toInt() shr 8) and 0xFF).toByte()
    }
}

fun quantize_row_q1_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK1_0
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = 2 + qk / 8

    for (i in 0 until nb) {
        val base = xOff + i * qk
        // Find scale: d = max(abs(x))
        var amax = 0f
        for (j in 0 until qk) {
            amax = maxOf(amax, kotlin.math.abs(x[base + j]))
        }
        val d = amax
        val bOff = yOff + i * blockSize
        val dFp16 = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(d)
        y[bOff + 0] = (dFp16.toInt() and 0xFF).toByte()
        y[bOff + 1] = ((dFp16.toInt() shr 8) and 0xFF).toByte()
        val qsOff = bOff + 2
        for (j in 0 until qk / 8) y[qsOff + j] = 0
        for (j in 0 until qk) {
            if (x[base + j] > 0f) {
                val byteIdx = j / 8
                val bitOff = j % 8
                y[qsOff + byteIdx] = (y[qsOff + byteIdx].toInt() or (1 shl bitOff)).toByte()
            }
        }
    }
}

// K-quant quantize_row_*_ref — complex implementations, deferred for now

fun quantize_row_q2_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ2K.SIZE_BYTES

    val L = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val Laux = IntArray(16)
    val weights = FloatArray(16)
    val mins = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)

    val q4scale = 15f

    // Temp UByteArray wrappers for make_qkx2_quants
    val Lu = UByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val Lauxu = UByteArray(16)
    val theMin = FloatArray(1)

    var xIdx = xOff
    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        var maxScale = 0f
        var maxMin = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            for (l in 0 until 16) weights[l] = kotlin.math.abs(x[xIdx + 16 * j + l])
            theMin[0] = 0f
            scales[j] = io.github.kotlinmania.llama.ore.make_qkx2_quants(
                16, 3, x, xIdx + 16 * j, weights, 0,
                Lu, 16 * j, theMin, 0, Lauxu, 0, -0.5f, 0.1f, 15, true
            )
            mins[j] = theMin[0]
            if (scales[j] > maxScale) maxScale = scales[j]
            if (mins[j] > maxMin) maxMin = mins[j]
        }

        // scales[QK_K/16] at offset 0
        val scalesOff = bOff + 0
        if (maxScale > 0) {
            val iscale = q4scale / maxScale
            for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
                val l = io.github.kotlinmania.llama.ore.nearest_int(iscale * scales[j])
                y[scalesOff + j] = l.toByte()
            }
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 80, maxScale / q4scale)
        } else {
            for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) y[scalesOff + j] = 0
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 80, 0f)
        }
        if (maxMin > 0) {
            val iscale = q4scale / maxMin
            for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
                val l = io.github.kotlinmania.llama.ore.nearest_int(iscale * mins[j])
                y[scalesOff + j] = (y[scalesOff + j].toInt() or (l shl 4)).toByte()
            }
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 82, maxMin / q4scale)
        } else {
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 82, 0f)
        }
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 80) * ((y[scalesOff + j].toInt() and 0xFF) and 0xF)
            if (d == 0f) continue
            val dm = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 82) * ((y[scalesOff + j].toInt() and 0xFF) shr 4)
            for (ii in 0 until 16) {
                var l =
                    io.github.kotlinmania.llama.ore.nearest_int((x[xIdx + 16 * j + ii] + dm) / d)
                l = maxOf(0, minOf(3, l))
                L[16 * j + ii] = l
            }
        }

        // qs[QK_K/4] at offset 16
        val qsOff = bOff + 16
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                y[qsOff + j / 4 + l] = (L[j + l] or (L[j + l + 32] shl 2) or
                    (L[j + l + 64] shl 4) or (L[j + l + 96] shl 6)).toByte()
            }
        }

        xIdx += io.github.kotlinmania.llama.ore.QK_K
    }
}

fun quantize_row_q3_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ3K.SIZE_BYTES

    val L = ByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)

    var xIdx = xOff
    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        val hmOff = bOff + 0
        val qsOff = bOff + 32
        val scOff = bOff + 96

        var maxScale = 0f
        var amax = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            scales[j] = io.github.kotlinmania.llama.ore.make_q3_quants(
                16,
                4,
                x,
                xIdx + 16 * j,
                L,
                16 * j,
                true
            )
            val scale = kotlin.math.abs(scales[j])
            if (scale > amax) {
                amax = scale; maxScale = scales[j]
            }
        }

        // Clear scales[12]
        for (j in 0 until 12) y[scOff + j] = 0
        if (maxScale != 0f) {
            val iscale = -32f / maxScale
            for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
                var l = io.github.kotlinmania.llama.ore.nearest_int(iscale * scales[j])
                l = maxOf(-32, minOf(31, l)) + 32
                if (j < 8) {
                    y[scOff + j] = (y[scOff + j].toInt() or (l and 0xF)).toByte()
                } else {
                    y[scOff + j - 8] = (y[scOff + j - 8].toInt() or ((l and 0xF) shl 4)).toByte()
                }
                val lh = l shr 4
                y[scOff + j % 4 + 8] = (y[scOff + j % 4 + 8].toInt() or (lh shl (2 * (j / 4)))).toByte()
            }
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 108, 1f / iscale)
        } else {
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 108, 0f)
        }

        var sc: Int
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            sc = if (j < 8) (y[scOff + j].toInt() and 0xFF) and 0xF
                 else (y[scOff + j - 8].toInt() and 0xFF) shr 4
            sc = (sc or ((((y[scOff + 8 + j % 4].toInt() and 0xFF) shr (2 * (j / 4))) and 3) shl 4)) - 32
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 108) * sc
            if (d == 0f) continue
            for (ii in 0 until 16) {
                var l = io.github.kotlinmania.llama.ore.nearest_int(x[xIdx + 16 * j + ii] / d)
                l = maxOf(-4, minOf(3, l))
                L[16 * j + ii] = (l + 4).toByte()
            }
        }

        // hmask[QK_K/8] at offset 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 8) y[hmOff + j] = 0
        var m = 0
        var hm = 1
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) {
            if ((L[j].toInt() and 0xFF) > 3) {
                y[hmOff + m] = (y[hmOff + m].toInt() or hm).toByte()
                L[j] = ((L[j].toInt() and 0xFF) - 4).toByte()
            }
            m++
            if (m == io.github.kotlinmania.llama.ore.QK_K / 8) {
                m = 0; hm = hm shl 1
            }
        }
        // qs[QK_K/4] at offset 32
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                y[qsOff + j / 4 + l] = ((L[j + l].toInt() and 0xFF) or
                    ((L[j + l + 32].toInt() and 0xFF) shl 2) or
                    ((L[j + l + 64].toInt() and 0xFF) shl 4) or
                    ((L[j + l + 96].toInt() and 0xFF) shl 6)).toByte()
            }
        }

        xIdx += io.github.kotlinmania.llama.ore.QK_K
    }
}

fun quantize_row_q4_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ4K.SIZE_BYTES

    val L = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val Laux = IntArray(32)
    val weights = FloatArray(32)
    val mins = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)

    val Lu = UByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val Lauxu = UByteArray(32)
    val theMin = FloatArray(1)

    var xIdx = xOff
    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        val scOff = bOff + 4

        var maxScale = 0f
        var maxMin = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            var sumX2 = 0f
            for (l in 0 until 32) sumX2 += x[xIdx + 32 * j + l] * x[xIdx + 32 * j + l]
            val avX = kotlin.math.sqrt(sumX2 / 32f)
            for (l in 0 until 32) weights[l] = avX + kotlin.math.abs(x[xIdx + 32 * j + l])
            theMin[0] = 0f
            scales[j] = io.github.kotlinmania.llama.ore.make_qkx2_quants(
                32, 15, x, xIdx + 32 * j, weights, 0,
                Lu, 32 * j, theMin, 0, Lauxu, 0, -1f, 0.1f, 20, false
            )
            mins[j] = theMin[0]
            if (scales[j] > maxScale) maxScale = scales[j]
            if (mins[j] > maxMin) maxMin = mins[j]
        }

        val invScale = if (maxScale > 0) 63f / maxScale else 0f
        val invMin = if (maxMin > 0) 63f / maxMin else 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            var ls = io.github.kotlinmania.llama.ore.nearest_int(invScale * scales[j])
            var lm = io.github.kotlinmania.llama.ore.nearest_int(invMin * mins[j])
            ls = minOf(63, ls)
            lm = minOf(63, lm)
            if (j < 4) {
                y[scOff + j] = ls.toByte()
                y[scOff + j + 4] = lm.toByte()
            } else {
                y[scOff + j + 4] = ((ls and 0xF) or ((lm and 0xF) shl 4)).toByte()
                y[scOff + j - 4] = (y[scOff + j - 4].toInt() or ((ls shr 4) shl 6)).toByte()
                y[scOff + j] = (y[scOff + j].toInt() or ((lm shr 4) shl 6)).toByte()
            }
        }
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, maxScale / 63f)
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 2, maxMin / 63f)

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val sm = io.github.kotlinmania.llama.ore.get_scale_min_k4(j, y, scOff)
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 0) * sm.d
            if (d == 0f) continue
            val dm = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 2) * sm.m
            for (ii in 0 until 32) {
                var l =
                    io.github.kotlinmania.llama.ore.nearest_int((x[xIdx + 32 * j + ii] + dm) / d)
                l = maxOf(0, minOf(15, l))
                L[32 * j + ii] = l
            }
        }

        // qs[QK_K/2] at offset 16
        val qsOff = bOff + 16
        var qIdx = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
            for (l in 0 until 32) {
                y[qsOff + qIdx + l] = (L[j + l] or (L[j + l + 32] shl 4)).toByte()
            }
            qIdx += 32
        }

        xIdx += io.github.kotlinmania.llama.ore.QK_K
    }
}

fun quantize_row_q5_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ5K.SIZE_BYTES

    val L = IntArray(io.github.kotlinmania.llama.ore.QK_K)
    val mins = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val weights = FloatArray(32)

    val Lu = UByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val Lauxu = UByteArray(32)
    val theMin = FloatArray(1)

    var xIdx = xOff
    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        val scOff = bOff + 4
        val qhOff = bOff + 16
        val qlOff = bOff + 48

        var maxScale = 0f
        var maxMin = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            var sumX2 = 0f
            for (l in 0 until 32) sumX2 += x[xIdx + 32 * j + l] * x[xIdx + 32 * j + l]
            val avX = kotlin.math.sqrt(sumX2 / 32f)
            for (l in 0 until 32) weights[l] = avX + kotlin.math.abs(x[xIdx + 32 * j + l])
            theMin[0] = 0f
            scales[j] = io.github.kotlinmania.llama.ore.make_qkx2_quants(
                32, 31, x, xIdx + 32 * j, weights, 0,
                Lu, 32 * j, theMin, 0, Lauxu, 0, -0.5f, 0.1f, 15, false
            )
            mins[j] = theMin[0]
            if (scales[j] > maxScale) maxScale = scales[j]
            if (mins[j] > maxMin) maxMin = mins[j]
        }

        val invScale = if (maxScale > 0) 63f / maxScale else 0f
        val invMin = if (maxMin > 0) 63f / maxMin else 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            var ls = io.github.kotlinmania.llama.ore.nearest_int(invScale * scales[j])
            var lm = io.github.kotlinmania.llama.ore.nearest_int(invMin * mins[j])
            ls = minOf(63, ls)
            lm = minOf(63, lm)
            if (j < 4) {
                y[scOff + j] = ls.toByte()
                y[scOff + j + 4] = lm.toByte()
            } else {
                y[scOff + j + 4] = ((ls and 0xF) or ((lm and 0xF) shl 4)).toByte()
                y[scOff + j - 4] = (y[scOff + j - 4].toInt() or ((ls shr 4) shl 6)).toByte()
                y[scOff + j] = (y[scOff + j].toInt() or ((lm shr 4) shl 6)).toByte()
            }
        }
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, maxScale / 63f)
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 2, maxMin / 63f)

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val sm = io.github.kotlinmania.llama.ore.get_scale_min_k4(j, y, scOff)
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 0) * sm.d
            if (d == 0f) continue
            val dm = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 2) * sm.m
            for (ii in 0 until 32) {
                var l =
                    io.github.kotlinmania.llama.ore.nearest_int((x[xIdx + 32 * j + ii] + dm) / d)
                l = maxOf(0, minOf(31, l))
                L[32 * j + ii] = l
            }
        }

        // qh[QK_K/8] at offset 16
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 8) y[qhOff + j] = 0

        var m1 = 1; var m2 = 2
        var qlIdx = 0
        for (n in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
            for (j in 0 until 32) {
                var l1 = L[n + j]
                if (l1 > 15) {
                    l1 -= 16; y[qhOff + j] = (y[qhOff + j].toInt() or m1).toByte()
                }
                var l2 = L[n + j + 32]
                if (l2 > 15) {
                    l2 -= 16; y[qhOff + j] = (y[qhOff + j].toInt() or m2).toByte()
                }
                y[qlOff + qlIdx + j] = (l1 or (l2 shl 4)).toByte()
            }
            m1 = m1 shl 2; m2 = m2 shl 2
            qlIdx += 32
        }

        xIdx += io.github.kotlinmania.llama.ore.QK_K
    }
}

fun quantize_row_q6_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ6K.SIZE_BYTES

    val L = ByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)

    var xIdx = xOff
    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        val qlOff = bOff + 0
        val qhOff = bOff + 128
        val scOff = bOff + 192

        var maxScale = 0f
        var maxAbsScale = 0f

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val scale = io.github.kotlinmania.llama.ore.make_qx_quants(
                16,
                32,
                x,
                xIdx + 16 * ib,
                L,
                16 * ib,
                1,
                null,
                0
            )
            scales[ib] = scale

            val absScale = kotlin.math.abs(scale)
            if (absScale > maxAbsScale) {
                maxAbsScale = absScale
                maxScale = scale
            }
        }

        if (maxAbsScale < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
            // Zero out entire block
            for (j in 0 until blockSize) y[bOff + j] = 0
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 208, 0f)
            xIdx += io.github.kotlinmania.llama.ore.QK_K
            continue
        }

        val iscale = -128f / maxScale
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 208, 1f / iscale)
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            y[scOff + ib] = minOf(127,
                io.github.kotlinmania.llama.ore.nearest_int(iscale * scales[ib])
            ).toByte()
        }

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 208) * y[scOff + j].toInt()
            if (d == 0f) continue
            for (ii in 0 until 16) {
                var l = io.github.kotlinmania.llama.ore.nearest_int(x[xIdx + 16 * j + ii] / d)
                l = maxOf(-32, minOf(31, l))
                L[16 * j + ii] = (l + 32).toByte()
            }
        }

        var qlIdx = 0
        var qhIdx = 0
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                val q1 = (L[j + l + 0].toInt() and 0xFF) and 0xF
                val q2 = (L[j + l + 32].toInt() and 0xFF) and 0xF
                val q3 = (L[j + l + 64].toInt() and 0xFF) and 0xF
                val q4 = (L[j + l + 96].toInt() and 0xFF) and 0xF
                y[qlOff + qlIdx + l + 0] = (q1 or (q3 shl 4)).toByte()
                y[qlOff + qlIdx + l + 32] = (q2 or (q4 shl 4)).toByte()
                y[qhOff + qhIdx + l] = (((L[j + l].toInt() and 0xFF) shr 4) or
                    (((L[j + l + 32].toInt() and 0xFF) shr 4) shl 2) or
                    (((L[j + l + 64].toInt() and 0xFF) shr 4) shl 4) or
                    (((L[j + l + 96].toInt() and 0xFF) shr 4) shl 6)).toByte()
            }
            qlIdx += 64
            qhIdx += 32
        }

        xIdx += io.github.kotlinmania.llama.ore.QK_K
    }
}

fun quantize_row_q8_K_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES

    var xIdx = xOff
    for (i in 0 until nb) {
        var max = 0f
        var amax = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) {
            val ax = kotlin.math.abs(x[xIdx + j])
            if (ax > amax) { amax = ax; max = x[xIdx + j] }
        }
        val bOff = yOff + i * blockSize
        if (amax == 0f) {
            // Write d = 0 as float32 LE
            val dBits = 0f.toRawBits()
            y[bOff + 0] = (dBits and 0xFF).toByte()
            y[bOff + 1] = ((dBits shr 8) and 0xFF).toByte()
            y[bOff + 2] = ((dBits shr 16) and 0xFF).toByte()
            y[bOff + 3] = ((dBits shr 24) and 0xFF).toByte()
            // Zero out qs and bsums
            for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) y[bOff + 4 + j] = 0
            for (j in 0 until (io.github.kotlinmania.llama.ore.QK_K / 16) * 2) y[bOff + 4 + io.github.kotlinmania.llama.ore.QK_K + j] = 0
            xIdx += io.github.kotlinmania.llama.ore.QK_K
            continue
        }
        // -127 not -128, needed for IQ2_XXS AVX compatibility
        val iscale = -127f / max
        val qsOff = bOff + 4
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) {
            val v = io.github.kotlinmania.llama.ore.nearest_int(iscale * x[xIdx + j])
            y[qsOff + j] = minOf(127, v).toByte()
        }
        // Compute bsums (QK_K/16 int16_t values)
        val bsumsOff = bOff + 4 + io.github.kotlinmania.llama.ore.QK_K
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            var sum = 0
            for (ii in 0 until 16) {
                sum += y[qsOff + j * 16 + ii].toInt()
            }
            // Write int16 LE
            y[bsumsOff + j * 2 + 0] = (sum and 0xFF).toByte()
            y[bsumsOff + j * 2 + 1] = ((sum shr 8) and 0xFF).toByte()
        }
        // Write d = 1/iscale as float32 LE
        val d = 1f / iscale
        val dBits = d.toRawBits()
        y[bOff + 0] = (dBits and 0xFF).toByte()
        y[bOff + 1] = ((dBits shr 8) and 0xFF).toByte()
        y[bOff + 2] = ((dBits shr 16) and 0xFF).toByte()
        y[bOff + 3] = ((dBits shr 24) and 0xFF).toByte()
        xIdx += io.github.kotlinmania.llama.ore.QK_K
    }
}

fun quantize_row_tq1_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockTQ1_0.SIZE_BYTES
    val qsSize = (io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5  // 48
    val qhSize = io.github.kotlinmania.llama.ore.QK_K / 64                     // 4

    var xIdx = xOff
    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        val qsOff = bOff + 0
        val qhOff = bOff + qsSize
        val dOff = bOff + qsSize + qhSize

        var amax = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) {
            amax = maxOf(amax, kotlin.math.abs(x[xIdx + j]))
        }
        val d = amax
        val id = if (d != 0f) 1f / d else 0f
        io.github.kotlinmania.llama.ore.writeFp16(y, dOff, d)

        // 5 elements per byte, along 32 bytes
        var xLocal = xIdx
        var j = 0
        while (j < qsSize - qsSize % 32) {
            for (m in 0 until 32) {
                var q = 0
                for (n in 0 until 5) {
                    val xi = kotlin.math.round(x[xLocal + m + n * 32] * id).toInt() + 1  // -1, 0, 1 -> 0, 1, 2
                    q *= 3
                    q += xi
                }
                // ceiling division (243 == pow(3, 5))
                q = ((q * 256 + 242) / 243)
                y[qsOff + j + m] = (q and 0xFF).toByte()
            }
            xLocal += 5 * 32
            j += 32
        }
        // along 16 bytes
        while (j < qsSize) {
            for (m in 0 until 16) {
                var q = 0
                for (n in 0 until 5) {
                    val xi = kotlin.math.round(x[xLocal + m + n * 16] * id).toInt() + 1
                    q *= 3
                    q += xi
                }
                q = ((q * 256 + 242) / 243)
                y[qsOff + j + m] = (q and 0xFF).toByte()
            }
            xLocal += 5 * 16
            j += 16
        }
        // 4 elements per byte
        for (jj in 0 until qhSize) {
            var q = 0
            for (m in 0 until 4) {
                val xi = kotlin.math.round(x[xLocal + jj + m * qhSize] * id).toInt() + 1
                q *= 3
                q += xi
            }
            q *= 3  // shift the first value to the most significant trit
            q = ((q * 256 + 242) / 243)
            y[qhOff + jj] = (q and 0xFF).toByte()
        }
        xIdx += io.github.kotlinmania.llama.ore.QK_K
    }
}

fun quantize_row_tq2_0_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nb = (k / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockTQ2_0.SIZE_BYTES
    val qsSize = io.github.kotlinmania.llama.ore.QK_K / 4  // 64

    var xIdx = xOff
    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        val qsOff = bOff + 0
        val dOff = bOff + qsSize

        var amax = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) {
            amax = maxOf(amax, kotlin.math.abs(x[xIdx + j]))
        }
        val d = amax
        val id = if (d != 0f) 1f / d else 0f
        io.github.kotlinmania.llama.ore.writeFp16(y, dOff, d)

        var j = 0
        while (j < qsSize) {
            for (m in 0 until 32) {
                var q = 0
                for (n in 0 until 4) {
                    val xi = kotlin.math.round(x[xIdx + m + n * 32] * id).toInt() + 1  // -1, 0, 1 -> 0, 1, 2
                    q += (xi and 3) shl (2 * n)
                }
                y[qsOff + j + m] = (q and 0xFF).toByte()
            }
            xIdx += 4 * 32
            j += 32
        }
    }
}

fun quantize_row_mxfp4_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK_MXFP4
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockMXFP4.SIZE_BYTES  // 17

    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        val base = xOff + i * qk
        var amax = 0f
        for (j in 0 until qk) {
            if (amax < kotlin.math.abs(x[base + j])) {
                amax = kotlin.math.abs(x[base + j])
            }
        }
        val e: Int = if (amax > 0f) (kotlin.math.floor(kotlin.math.log2(amax.toDouble())).toInt() - 2 + 127) else 0
        val d = io.github.kotlinmania.llama.ore.GGML_E8M0_TO_FP32_HALF((e and 0xFF).toUByte())

        y[bOff] = (e and 0xFF).toByte()  // block.e

        val qsOff = bOff + 1
        for (j in 0 until qk / 2) {
            val x0 = io.github.kotlinmania.llama.ore.best_index_mxfp4(x[base + j], d)
            val x1 = io.github.kotlinmania.llama.ore.best_index_mxfp4(x[base + j + qk / 2], d)
            y[qsOff + j] = (x0 or (x1 shl 4)).toByte()
        }
    }
}

fun quantize_row_nvfp4_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    val qk = io.github.kotlinmania.llama.ore.QK_NVFP4
    val qkSub = io.github.kotlinmania.llama.ore.QK_NVFP4_SUB
    val nSub = io.github.kotlinmania.llama.ore.QK_NVFP4 / io.github.kotlinmania.llama.ore.QK_NVFP4_SUB
    require(k % qk == 0L)
    val nb = (k / qk).toInt()
    val blockSize = io.github.kotlinmania.llama.ore.BlockNVFP4.SIZE_BYTES  // 36

    for (i in 0 until nb) {
        val bOff = yOff + i * blockSize
        for (s in 0 until nSub) {
            val xbOff = xOff + i * qk + s * qkSub
            var amax = 0f
            for (j in 0 until qkSub) {
                if (amax < kotlin.math.abs(x[xbOff + j])) {
                    amax = kotlin.math.abs(x[xbOff + j])
                }
            }
            val ue = io.github.kotlinmania.llama.ore.ggml_fp32_to_ue4m3(amax / 6f)
            y[bOff + s] = ue.toByte()  // block.d[s]
            val d = io.github.kotlinmania.llama.ore.ggml_ue4m3_to_fp32(ue)

            val qsBase = bOff + nSub + s * (qkSub / 2)
            for (j in 0 until qkSub / 2) {
                val x0 = io.github.kotlinmania.llama.ore.best_index_mxfp4(x[xbOff + j], d)
                val x1 =
                    io.github.kotlinmania.llama.ore.best_index_mxfp4(x[xbOff + j + qkSub / 2], d)
                y[qsBase + j] = (x0 or (x1 shl 4)).toByte()
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  IQ2/IQ3 quantization data infrastructure (C lines 2744-3141)
// ════════════════════════════════════════════════════════════════════════════════

private class IQ2Entry(
    var grid: ULongArray? = null,
    var map: IntArray? = null,
    var neighbours: UShortArray? = null
)
private val iq2Data = Array(4) { io.github.kotlinmania.llama.ore.IQ2Entry() }

// C line 2757
private fun iq2DataIndex(type: io.github.kotlinmania.llama.ore.GGMLType): Int {
    require(type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS || type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_S || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_M || type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_S)
    return when (type) {
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS -> 0
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS -> 1
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M -> 2
        else -> 3 // IQ2_S
    }
}

// C line 2764
private fun iq2GridSize(type: io.github.kotlinmania.llama.ore.GGMLType): Int {
    require(type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS || type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_S || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_M || type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_S)
    return when (type) {
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS -> 256
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS -> 512
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M -> io.github.kotlinmania.llama.ore.NGRID_IQ1S
        else -> 1024 // IQ2_S
    }
}

// C line 3553
private class IQ3Entry(
    var grid: UIntArray? = null,
    var map: IntArray? = null,
    var neighbours: UShortArray? = null
)
private val iq3Data = Array(2) { io.github.kotlinmania.llama.ore.IQ3Entry() }

// C line 3564
private fun iq3DataIndex(gridSize: Int): Int {
    require(gridSize == 256 || gridSize == 512)
    return if (gridSize == 256) 0 else 1
}

// Helper: extract signed byte from a ULong at byte position i
private fun gridByte(gv: ULong, i: Int): Byte =
    ((gv shr (8 * i)) and 0xFFu).toByte()

// Helper: extract signed byte from a UInt at byte position i
private fun gridByte(gv: UInt, i: Int): Byte =
    ((gv shr (8 * i)) and 0xFFu).toByte()

// C line 3143: iq2_find_best_neighbour
private fun iq2FindBestNeighbour(
    neighbours: UShortArray, nOff: Int,
    grid: ULongArray,
    xval: FloatArray, xvOff: Int,
    weight: FloatArray, wOff: Int,
    scale: Float,
    L: ByteArray, lOff: Int
): Int {
    val numNeighbors = neighbours[nOff].toInt()
    require(numNeighbors > 0)
    var bestD2 = Float.MAX_VALUE
    var gridIndex = -1
    for (j in 1..numNeighbors) {
        val gv = grid[neighbours[nOff + j].toInt()]
        var d2 = 0f
        for (i in 0 until 8) {
            val q = io.github.kotlinmania.llama.ore.gridByte(gv, i).toFloat()
            val diff = scale * q - xval[xvOff + i]
            d2 += weight[wOff + i] * diff * diff
        }
        if (d2 < bestD2) {
            bestD2 = d2; gridIndex = neighbours[nOff + j].toInt()
        }
    }
    require(gridIndex >= 0)
    val gv = grid[gridIndex]
    for (i in 0 until 8) L[lOff + i] = ((io.github.kotlinmania.llama.ore.gridByte(gv, i) - 1) / 2).toByte()
    return gridIndex
}

// C line 3742: iq3_find_best_neighbour
private fun iq3FindBestNeighbour(
    neighbours: UShortArray, nOff: Int,
    grid: UIntArray,
    xval: FloatArray, xvOff: Int,
    weight: FloatArray, wOff: Int,
    scale: Float,
    L: ByteArray, lOff: Int
): Int {
    val numNeighbors = neighbours[nOff].toInt()
    require(numNeighbors > 0)
    var bestD2 = Float.MAX_VALUE
    var gridIndex = -1
    for (j in 1..numNeighbors) {
        val gv = grid[neighbours[nOff + j].toInt()]
        var d2 = 0f
        for (i in 0 until 4) {
            val q = io.github.kotlinmania.llama.ore.gridByte(gv, i).toFloat()
            val diff = scale * q - xval[xvOff + i]
            d2 += weight[wOff + i] * diff * diff
        }
        if (d2 < bestD2) {
            bestD2 = d2; gridIndex = neighbours[nOff + j].toInt()
        }
    }
    require(gridIndex >= 0)
    val gv = grid[gridIndex]
    for (i in 0 until 4) L[lOff + i] = ((io.github.kotlinmania.llama.ore.gridByte(gv, i) - 1) / 2).toByte()
    return gridIndex
}

// C line 4211: iq1_find_best_neighbour
private fun iq1FindBestNeighbour(
    neighbours: UShortArray, nOff: Int,
    grid: ULongArray,
    xval: FloatArray, xvOff: Int,
    weight: FloatArray, wOff: Int,
    scaleOut: FloatArray, sOff: Int,
    L: ByteArray, lOff: Int,
    ngrid: Int
): Int {
    val numNeighbors = neighbours[nOff].toInt()
    require(numNeighbors > 0)
    var bestScore = -Float.MAX_VALUE
    var gridIndex = -1
    for (j in 1..numNeighbors) {
        val gv = grid[neighbours[nOff + j].toInt()]
        var sumqx = 0f; var sumq2 = 0f
        for (i in 0 until 8) {
            val q = (io.github.kotlinmania.llama.ore.gridByte(gv, i) - 3).toFloat() / 2f
            val w = weight[wOff + i]
            sumqx += w * q * xval[xvOff + i]
            sumq2 += w * q * q
        }
        if (sumqx > 0 && sumq2 > 0 && sumqx * sumqx > bestScore * sumq2) {
            scaleOut[sOff] = sumqx / sumq2; bestScore = scaleOut[sOff] * sumqx
            gridIndex = neighbours[nOff + j].toInt()
        }
    }
    if (gridIndex < 0) {
        for (i in 0 until ngrid) {
            val gv = grid[i]
            var sumqx = 0f; var sumq2 = 0f
            for (j in 0 until 8) {
                val w = weight[wOff + j]
                val q = (io.github.kotlinmania.llama.ore.gridByte(gv, j) - 3).toFloat() / 2f
                sumqx += w * q * xval[xvOff + j]
                sumq2 += w * q * q
            }
            if (sumqx > 0 && sumq2 > 0 && sumqx * sumqx > bestScore * sumq2) {
                scaleOut[sOff] = sumqx / sumq2; bestScore = scaleOut[sOff] * sumqx
                gridIndex = i
            }
        }
    }
    require(gridIndex >= 0) { "iq1_find_best_neighbour: did not find grid point" }
    scaleOut[sOff] *= 1.05f
    val gv = grid[gridIndex]
    for (i in 0 until 8) L[lOff + i] = ((io.github.kotlinmania.llama.ore.gridByte(gv, i) - 1) / 2).toByte()
    return gridIndex
}

// C line 4271: iq1_find_best_neighbour2
private fun iq1FindBestNeighbour2(
    neighbours: UShortArray, nOff: Int,
    grid: ULongArray,
    xval: FloatArray, xvOff: Int,
    weight: FloatArray, wOff: Int,
    scale: Float,
    xg: FloatArray,
    L: ByteArray, lOff: Int,
    ngrid: Int
): Int {
    val numNeighbors = neighbours[nOff].toInt()
    require(numNeighbors > 0)
    var bestScore = Float.MAX_VALUE
    var gridIndex = -1
    for (j in 1..numNeighbors) {
        val gv = grid[neighbours[nOff + j].toInt()]
        var d2 = 0f
        for (i in 0 until 8) {
            val q = xg[(io.github.kotlinmania.llama.ore.gridByte(gv, i) - 1) / 2]
            val w = weight[wOff + i]
            val diff = scale * q - xval[xvOff + i]
            d2 += w * diff * diff
        }
        if (d2 < bestScore) {
            bestScore = d2; gridIndex = neighbours[nOff + j].toInt()
        }
    }
    if (gridIndex < 0) {
        for (i in 0 until ngrid) {
            val gv = grid[i]
            var d2 = 0f
            for (j in 0 until 8) {
                val w = weight[wOff + j]
                val q = xg[(io.github.kotlinmania.llama.ore.gridByte(gv, j) - 1) / 2]
                val diff = scale * q - xval[xvOff + j]
                d2 += w * diff * diff
            }
            if (d2 < bestScore) {
                bestScore = d2; gridIndex = i
            }
        }
    }
    require(gridIndex >= 0) { "iq1_find_best_neighbour2: did not find grid point" }
    val gv = grid[gridIndex]
    for (i in 0 until 8) L[lOff + i] = ((io.github.kotlinmania.llama.ore.gridByte(gv, i) - 1) / 2).toByte()
    return gridIndex
}

// IQ quantize_row_ref — all require grid tables

fun quantize_row_iq3_xxs_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    io.github.kotlinmania.llama.ore.quantize_row_iq3_xxs_impl(
        256,
        x,
        xOff,
        y,
        yOff,
        k.toInt(),
        null
    )
}

// C line 4794: quantize_row_iq4_nl_impl
// Shared impl for iq4_nl and iq4_xs quantization
private fun quantize_row_iq4_nl_impl(
    superBlockSize: Int, blockSize: Int,
    x: FloatArray, xOff: Int,
    dh: ByteArray, dhOff: Int,         // fp16 output: d
    q4: ByteArray, q4Off: Int,         // uint8_t qs output
    scalesH: ByteArray, shOff: Int,    // uint16_t scales_h (may be unused)
    scalesL: ByteArray?, slOff: Int,   // uint8_t scales_l (null when unused)
    scales: FloatArray, weight: FloatArray, L: UByteArray,
    values: ByteArray,
    quantWeights: FloatArray?, qwOff: Int,
    ntry: Int
) {
    var sigma2 = 0f
    for (j in 0 until superBlockSize) sigma2 += x[xOff + j] * x[xOff + j]
    sigma2 *= 2f / superBlockSize

    for (j in 0 until superBlockSize / 2) q4[q4Off + j] = 0
    io.github.kotlinmania.llama.ore.writeFp16(dh, dhOff, 0f)

    var maxScale = 0f; var amaxScale = 0f
    for (ib in 0 until superBlockSize / blockSize) {
        val xbOff = xOff + ib * blockSize
        val lbOff = ib * blockSize
        if (quantWeights != null) {
            val qwbOff = qwOff + ib * blockSize
            for (j in 0 until blockSize) weight[j] = quantWeights[qwbOff + j] * kotlin.math.sqrt(sigma2 + x[xbOff + j] * x[xbOff + j])
        } else {
            for (j in 0 until blockSize) weight[j] = x[xbOff + j] * x[xbOff + j]
        }
        var amax = 0f; var max = 0f
        for (j in 0 until blockSize) {
            val ax = kotlin.math.abs(x[xbOff + j])
            if (ax > amax) { amax = ax; max = x[xbOff + j] }
        }
        if (amax < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
            scales[ib] = 0f
            continue
        }
        var d = if (ntry > 0) -max / values[0].toInt() else max / values[0].toInt()
        var id = 1f / d
        var sumqx = 0f; var sumq2 = 0f
        for (j in 0 until blockSize) {
            val al = id * x[xbOff + j]
            val l = io.github.kotlinmania.llama.ore.best_index_int8(16, values, 0, al)
            L[lbOff + j] = l.toUByte()
            val q = values[l].toFloat()
            val w = weight[j]
            sumqx += w * q * x[xbOff + j]
            sumq2 += w * q * q
        }
        d = if (sumq2 > 0) sumqx / sumq2 else 0f
        var best = d * sumqx
        for (itry in -ntry..ntry) {
            id = (itry + values[0].toInt()) / max
            sumqx = 0f; sumq2 = 0f
            for (j in 0 until blockSize) {
                val al = id * x[xbOff + j]
                val l = io.github.kotlinmania.llama.ore.best_index_int8(16, values, 0, al)
                val q = values[l].toFloat()
                val w = weight[j]
                sumqx += w * q * x[xbOff + j]
                sumq2 += w * q * q
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                d = sumqx / sumq2; best = d * sumqx
            }
        }
        scales[ib] = d
        val absD = kotlin.math.abs(d)
        if (absD > amaxScale) {
            amaxScale = absD; maxScale = d
        }
    }

    if (superBlockSize / blockSize > 1) {
        val nb = superBlockSize / blockSize
        // zero scales_h
        for (j in 0 until (nb + 7) / 8) io.github.kotlinmania.llama.ore.writeShortLE(
            scalesH,
            shOff + j * 2,
            0
        )
        val d = -maxScale / 32
        io.github.kotlinmania.llama.ore.writeFp16(dh, dhOff, d)
        val id = if (d != 0f) 1f / d else 0f
        for (ib in 0 until superBlockSize / blockSize) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(id * scales[ib])
            l = maxOf(-32, minOf(31, l))
            val dl = d * l
            val idl = if (dl != 0f) 1f / dl else 0f
            val lbOff = ib * blockSize
            val xbOff = xOff + ib * blockSize
            for (j in 0 until blockSize) {
                L[lbOff + j] = io.github.kotlinmania.llama.ore.best_index_int8(
                    16,
                    values,
                    0,
                    idl * x[xbOff + j]
                ).toUByte()
            }
            l += 32
            val lL = l and 0xf
            val lH = l shr 4
            if (ib % 2 == 0) scalesL!![slOff + ib / 2] = lL.toByte()
            else scalesL!![slOff + ib / 2] = (scalesL[slOff + ib / 2].toInt() or (lL shl 4)).toByte()
            // scales_h is uint16_t array — use byte-level writes
            val shIdx = shOff + (ib / 8) * 2
            val curH = io.github.kotlinmania.llama.ore.readShortLE(scalesH, shIdx).toInt() and 0xFFFF
            io.github.kotlinmania.llama.ore.writeShortLE(
                scalesH,
                shIdx,
                (curH or (lH shl (2 * (ib % 8)))).toShort()
            )
        }
    } else {
        io.github.kotlinmania.llama.ore.writeFp16(dh, dhOff, scales[0])
        if (ntry > 0) {
            val id = if (scales[0] != 0f) 1f / scales[0] else 0f
            for (j in 0 until superBlockSize) {
                L[j] = io.github.kotlinmania.llama.ore.best_index_int8(
                    16,
                    values,
                    0,
                    id * x[xOff + j]
                ).toUByte()
            }
        }
    }

    for (i in 0 until superBlockSize / 32) {
        for (j in 0 until 16) {
            q4[q4Off + 16 * i + j] = (L[32 * i + j].toInt() or (L[32 * i + 16 + j].toInt() shl 4)).toByte()
        }
    }
}

fun quantize_row_iq4_nl_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK4_NL == 0L)
    val nblock = (k / io.github.kotlinmania.llama.ore.QK4_NL).toInt()
    val blockSize = 18 // sizeof(block_iq4_nl) = 2 + QK4_NL/2
    val L = UByteArray(io.github.kotlinmania.llama.ore.QK4_NL)
    val weight = FloatArray(io.github.kotlinmania.llama.ore.QK4_NL)
    val scale = FloatArray(1)
    val unusedH = ByteArray(2)
    for (ibl in 0 until nblock) {
        val bOff = yOff + ibl * blockSize
        io.github.kotlinmania.llama.ore.quantize_row_iq4_nl_impl(
            io.github.kotlinmania.llama.ore.QK4_NL,
            32,
            x,
            xOff + io.github.kotlinmania.llama.ore.QK4_NL * ibl,
            y,
            bOff,           // dh at block start
            y,
            bOff + 2,       // qs at offset 2
            unusedH,
            0,
            null,
            0,
            scale,
            weight,
            L,
            io.github.kotlinmania.llama.ore.kvalues_iq4nl,
            null,
            0,
            -1
        )
    }
}

fun quantize_row_iq4_xs_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    io.github.kotlinmania.llama.ore.quantize_iq4_xs(
        x,
        yOff = yOff,
        dst = y,
        nrows = 1,
        nPerRow = k,
        imatrix = null
    )
}

fun quantize_row_iq3_s_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    io.github.kotlinmania.llama.ore.quantize_iq3_s(x, y, nrows = 1, nPerRow = k, imatrix = null)
}

fun quantize_row_iq2_s_ref(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, k: Long) {
    require(k % io.github.kotlinmania.llama.ore.QK_K == 0L)
    io.github.kotlinmania.llama.ore.quantize_iq2_s(x, y, nrows = 1, nPerRow = k, imatrix = null)
}

// ════════════════════════════════════════════════════════════════════════════════
//  Importance-matrix quantize functions
// ════════════════════════════════════════════════════════════════════════════════

fun quantize_q1_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q1_0_ref(src, 0, dst, 0, nrows * nPerRow)
        return nrows * io.github.kotlinmania.llama.ore.ggmlRowSize(
            io.github.kotlinmania.llama.ore.GGMLType.Q1_0,
            nPerRow
        ).toLong()
    }
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0,
        nPerRow
    ).toLong()
    var srcOff = 0
    var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_q1_0_ref(src, srcOff, dst, dstOff, nPerRow)
        srcOff += nPerRow.toInt()
        dstOff += rowSize.toInt()
    }
    return nrows * rowSize
}

// C line 2008: quantize_row_q4_0_impl
// Block layout: d(fp16, 2 bytes at 0), qs(QK4_0/2=16 bytes at 2). Block size = 18.
private fun quantize_row_q4_0_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val weight = FloatArray(io.github.kotlinmania.llama.ore.QK4_0)
    val L = ByteArray(io.github.kotlinmania.llama.ore.QK4_0)

    var sumX2 = 0f
    for (j in 0 until nPerRow) sumX2 += x[xOff + j] * x[xOff + j]
    val sigma2 = sumX2 / nPerRow

    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK4_0
    for (ib in 0 until nb) {
        val xbOff = xOff + io.github.kotlinmania.llama.ore.QK4_0 * ib
        val qwbOff = qwOff + io.github.kotlinmania.llama.ore.QK4_0 * ib
        for (j in 0 until io.github.kotlinmania.llama.ore.QK4_0) weight[j] = quantWeights[qwbOff + j] * kotlin.math.sqrt(sigma2 + x[xbOff + j] * x[xbOff + j])
        val d = io.github.kotlinmania.llama.ore.make_qx_quants(
            io.github.kotlinmania.llama.ore.QK4_0,
            8,
            x,
            xbOff,
            L,
            0,
            1,
            weight,
            0
        )
        val bOff = yOff + 18 * ib
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, d)
        for (j in 0 until 16) {
            y[bOff + 2 + j] = ((L[j].toInt() and 0xFF) or ((L[j + 16].toInt() and 0xFF) shl 4)).toByte()
        }
    }
}

// C line 2052
fun quantize_q4_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q4_0_ref(src, 0, dst, 0, nrows * nPerRow)
        return nrows * io.github.kotlinmania.llama.ore.ggmlRowSize(
            io.github.kotlinmania.llama.ore.GGMLType.Q4_0,
            nPerRow
        ).toLong()
    }
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0,
        nPerRow
    ).toLong()
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_q4_0_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix,
            (row * nPerRow).toInt()
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize.toInt()
    }
    return nrows * rowSize
}

// C line 2067: quantize_row_q4_1_impl
// Block layout: d(fp16, 2 at 0), m(fp16, 2 at 2), qs(16 at 4). Block size = 20.
private fun quantize_row_q4_1_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val weight = FloatArray(io.github.kotlinmania.llama.ore.QK4_1)
    val L = UByteArray(io.github.kotlinmania.llama.ore.QK4_1)
    val Laux = UByteArray(io.github.kotlinmania.llama.ore.QK4_1)

    var sumX2 = 0f
    for (j in 0 until nPerRow) sumX2 += x[xOff + j] * x[xOff + j]
    val sigma2 = sumX2 / nPerRow

    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK4_1
    val theMin = FloatArray(1)
    for (ib in 0 until nb) {
        val xbOff = xOff + io.github.kotlinmania.llama.ore.QK4_1 * ib
        val qwbOff = qwOff + io.github.kotlinmania.llama.ore.QK4_1 * ib
        for (j in 0 until io.github.kotlinmania.llama.ore.QK4_1) weight[j] = quantWeights[qwbOff + j] * kotlin.math.sqrt(sigma2 + x[xbOff + j] * x[xbOff + j])
        val d = io.github.kotlinmania.llama.ore.make_qkx3_quants(
            io.github.kotlinmania.llama.ore.QK4_1,
            15,
            x,
            xbOff,
            weight,
            0,
            L,
            0,
            theMin,
            0,
            Laux,
            0,
            -0.9f,
            0.05f,
            36,
            false
        )
        val bOff = yOff + 20 * ib
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, d)
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 2, -theMin[0])
        for (j in 0 until 16) {
            y[bOff + 4 + j] = (L[j].toInt() or (L[j + 16].toInt() shl 4)).toByte()
        }
    }
}

// C line 2097
fun quantize_q4_1(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q4_1_ref(src, 0, dst, 0, nrows * nPerRow)
        return nrows * io.github.kotlinmania.llama.ore.ggmlRowSize(
            io.github.kotlinmania.llama.ore.GGMLType.Q4_1,
            nPerRow
        ).toLong()
    }
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q4_1,
        nPerRow
    ).toLong()
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_q4_1_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix,
            (row * nPerRow).toInt()
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize.toInt()
    }
    return nrows * rowSize
}

// C line 2112: quantize_row_q5_0_impl
// Block layout: d(fp16, 2 at 0), qh(uint32, 4 at 2), qs(16 at 6). Block size = 22.
private fun quantize_row_q5_0_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val weight = FloatArray(io.github.kotlinmania.llama.ore.QK5_0)
    val L = ByteArray(io.github.kotlinmania.llama.ore.QK5_0)

    var sumX2 = 0f
    for (j in 0 until nPerRow) sumX2 += x[xOff + j] * x[xOff + j]
    val sigma2 = sumX2 / nPerRow

    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK5_0
    for (ib in 0 until nb) {
        val xbOff = xOff + io.github.kotlinmania.llama.ore.QK5_0 * ib
        val qwbOff = qwOff + io.github.kotlinmania.llama.ore.QK5_0 * ib
        for (j in 0 until io.github.kotlinmania.llama.ore.QK5_0) weight[j] = quantWeights[qwbOff + j] * kotlin.math.sqrt(sigma2 + x[xbOff + j] * x[xbOff + j])
        val d = io.github.kotlinmania.llama.ore.make_qx_quants(
            io.github.kotlinmania.llama.ore.QK5_0,
            16,
            x,
            xbOff,
            L,
            0,
            1,
            weight,
            0
        )
        val bOff = yOff + 22 * ib
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, d)

        var qh = 0
        for (j in 0 until 16) {
            val xi0 = L[j].toInt() and 0xFF
            val xi1 = L[j + 16].toInt() and 0xFF
            y[bOff + 6 + j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
            qh = qh or (((xi0 and 0x10) shr 4) shl (j + 0))
            qh = qh or (((xi1 and 0x10) shr 4) shl (j + io.github.kotlinmania.llama.ore.QK5_0 / 2))
        }
        io.github.kotlinmania.llama.ore.writeIntLE(y, bOff + 2, qh)
    }
}

// C line 2151
fun quantize_q5_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = (nPerRow / io.github.kotlinmania.llama.ore.QK5_0) * io.github.kotlinmania.llama.ore.BlockQ5_0.SIZE_BYTES
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q5_0_ref(src, 0, dst, 0, nrows * nPerRow)
        return nrows * rowSize
    }
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_q5_0_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix,
            (row * nPerRow).toInt()
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize.toInt()
    }
    return nrows * rowSize
}

// C line 2166: quantize_row_q5_1_impl
// Block layout: d(fp16, 2 at 0), m(fp16, 2 at 2), qh(uint32, 4 at 4), qs(16 at 8). Block size = 24.
private fun quantize_row_q5_1_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val weight = FloatArray(io.github.kotlinmania.llama.ore.QK5_1)
    val L = UByteArray(io.github.kotlinmania.llama.ore.QK5_1)
    val Laux = UByteArray(io.github.kotlinmania.llama.ore.QK5_1)

    var sumX2 = 0f
    for (j in 0 until nPerRow) sumX2 += x[xOff + j] * x[xOff + j]
    val sigma2 = sumX2 / nPerRow

    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK5_1
    val theMin = FloatArray(1)
    for (ib in 0 until nb) {
        val xbOff = xOff + io.github.kotlinmania.llama.ore.QK5_1 * ib
        val qwbOff = qwOff + io.github.kotlinmania.llama.ore.QK5_1 * ib
        for (j in 0 until io.github.kotlinmania.llama.ore.QK5_1) weight[j] = quantWeights[qwbOff + j] * kotlin.math.sqrt(sigma2 + x[xbOff + j] * x[xbOff + j])
        val d = io.github.kotlinmania.llama.ore.make_qkx3_quants(
            io.github.kotlinmania.llama.ore.QK5_1,
            31,
            x,
            xbOff,
            weight,
            0,
            L,
            0,
            theMin,
            0,
            Laux,
            0,
            -0.9f,
            0.05f,
            36,
            false
        )
        val bOff = yOff + 24 * ib
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, d)
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 2, -theMin[0])

        var qh = 0
        for (j in 0 until 16) {
            val xi0 = L[j].toInt()
            val xi1 = L[j + 16].toInt()
            y[bOff + 8 + j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
            qh = qh or (((xi0 and 0x10) shr 4) shl (j + 0))
            qh = qh or (((xi1 and 0x10) shr 4) shl (j + io.github.kotlinmania.llama.ore.QK5_0 / 2))
        }
        io.github.kotlinmania.llama.ore.writeIntLE(y, bOff + 4, qh)
    }
}

// C line 2204
fun quantize_q5_1(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = (nPerRow / io.github.kotlinmania.llama.ore.QK5_1) * io.github.kotlinmania.llama.ore.BlockQ5_1.SIZE_BYTES
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q5_1_ref(src, 0, dst, 0, nrows * nPerRow)
        return nrows * rowSize
    }
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_q5_1_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix,
            (row * nPerRow).toInt()
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize.toInt()
    }
    return nrows * rowSize
}

// C line 2219
fun quantize_q8_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0,
        nPerRow
    ).toLong()
    io.github.kotlinmania.llama.ore.quantize_row_q8_0_ref(src, 0, dst, 0, nrows * nPerRow)
    return nrows * rowSize
}

// C line 1087: quantize_row_q2_K_impl
// Block layout: scales(16 at 0), qs(64 at 16), d(fp16 at 80), dmin(fp16 at 82). Block size = 84.
private fun quantize_row_q2_K_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK_K

    val L = UByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val Laux = UByteArray(16)
    val mins = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val sw = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val weight = FloatArray(16)
    val Ls = UByteArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val Lm = UByteArray(io.github.kotlinmania.llama.ore.QK_K / 16)

    var xCur = xOff
    for (i in 0 until nb) {
        for (idx in sw.indices) sw[idx] = 0f
        var sumx2 = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xCur + j] * x[xCur + j]
        val sigma2 = sumx2 / io.github.kotlinmania.llama.ore.QK_K
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val qwBase = qwOff + io.github.kotlinmania.llama.ore.QK_K * i + 16 * j
            for (l in 0 until 16) weight[l] = quantWeights[qwBase + l] * kotlin.math.sqrt(sigma2 + x[xCur + 16 * j + l] * x[xCur + 16 * j + l])
            for (l in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) sw[j] += weight[l]
            scales[j] = io.github.kotlinmania.llama.ore.make_qkx3_quants(
                16,
                3,
                x,
                xCur + 16 * j,
                weight,
                0,
                L,
                16 * j,
                mins,
                j,
                Laux,
                0,
                -0.9f,
                0.05f,
                36,
                false
            )
        }

        var dm = io.github.kotlinmania.llama.ore.make_qp_quants(
            io.github.kotlinmania.llama.ore.QK_K / 16,
            15,
            scales,
            0,
            Ls,
            0,
            sw,
            0
        )
        var mm = io.github.kotlinmania.llama.ore.make_qp_quants(
            io.github.kotlinmania.llama.ore.QK_K / 16,
            15,
            mins,
            0,
            Lm,
            0,
            sw,
            0
        )

        val bOff = yOff + 84 * i
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 80, dm)
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 82, mm)
        dm = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 80)
        mm = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 82)

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            y[bOff + j] = (Ls[j].toInt() or (Lm[j].toInt() shl 4)).toByte()
        }

        // requantize
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val d = dm * ((y[bOff + j].toInt() and 0xFF) and 0xF)
            if (d == 0f) continue
            val m = mm * ((y[bOff + j].toInt() and 0xFF) shr 4)
            for (ii in 0 until 16) {
                var l = io.github.kotlinmania.llama.ore.nearest_int((x[xCur + 16 * j + ii] + m) / d)
                l = maxOf(0, minOf(3, l))
                L[16 * j + ii] = l.toUByte()
            }
        }

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                y[bOff + 16 + j / 4 + l] = (L[j + l].toInt() or (L[j + l + 32].toInt() shl 2) or (L[j + l + 64].toInt() shl 4) or (L[j + l + 96].toInt() shl 6)).toByte()
            }
        }

        xCur += io.github.kotlinmania.llama.ore.QK_K
    }
}

// C line 1149
fun quantize_q2_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K,
        nPerRow
    ).toLong()
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q2_K_ref(src, 0, dst, 0, nrows * nPerRow)
    } else {
        var srcOff = 0; var dstOff = 0
        for (row in 0 until nrows) {
            io.github.kotlinmania.llama.ore.quantize_row_q2_K_impl(
                src,
                srcOff,
                dst,
                dstOff,
                nPerRow.toInt(),
                imatrix,
                (row * nPerRow).toInt()
            )
            srcOff += nPerRow.toInt()
            dstOff += rowSize.toInt()
        }
    }
    return nrows * rowSize
}

// C line 1293: quantize_row_q3_K_impl
// Block layout: hmask(32 at 0), qs(64 at 32), scales(12 at 96), d(fp16 at 108). Block size = 110.
private fun quantize_row_q3_K_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK_K

    val L = ByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val weight = FloatArray(16)
    val sw = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val Ls = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 16)

    var xCur = xOff
    for (i in 0 until nb) {
        var sumx2 = 0f
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xCur + j] * x[xCur + j]
        val sigma2 = 2 * sumx2 / io.github.kotlinmania.llama.ore.QK_K

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val qwBase = qwOff + io.github.kotlinmania.llama.ore.QK_K * i + 16 * j
            for (l in 0 until 16) weight[l] = quantWeights[qwBase + l] * kotlin.math.sqrt(sigma2 + x[xCur + 16 * j + l] * x[xCur + 16 * j + l])
            var sumw = 0f
            for (l in 0 until 16) sumw += weight[l]
            sw[j] = sumw
            scales[j] = io.github.kotlinmania.llama.ore.make_qx_quants(
                16,
                4,
                x,
                xCur + 16 * j,
                L,
                16 * j,
                1,
                weight,
                0
            )
        }

        val bOff = yOff + 110 * i
        val scOff = bOff + 96
        for (idx in 0 until 12) y[scOff + idx] = 0

        val dBlock = io.github.kotlinmania.llama.ore.make_qx_quants(
            io.github.kotlinmania.llama.ore.QK_K / 16,
            32,
            scales,
            0,
            Ls,
            0,
            1,
            sw,
            0
        )
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val l = Ls[j].toInt() and 0xFF
            if (j < 8) {
                y[scOff + j] = ((y[scOff + j].toInt() and 0xFF) or (l and 0xF)).toByte()
            } else {
                y[scOff + j - 8] = ((y[scOff + j - 8].toInt() and 0xFF) or ((l and 0xF) shl 4)).toByte()
            }
            val lh = l shr 4
            y[scOff + j % 4 + 8] = ((y[scOff + j % 4 + 8].toInt() and 0xFF) or (lh shl (2 * (j / 4)))).toByte()
        }
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 108, dBlock)

        // requantize
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            var sc = if (j < 8) (y[scOff + j].toInt() and 0xFF) and 0xF else (y[scOff + j - 8].toInt() and 0xFF) shr 4
            sc = (sc or ((((y[scOff + 8 + j % 4].toInt() and 0xFF) shr (2 * (j / 4))) and 3) shl 4)) - 32
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 108) * sc
            if (d == 0f) continue
            for (ii in 0 until 16) {
                var l = io.github.kotlinmania.llama.ore.nearest_int(x[xCur + 16 * j + ii] / d)
                l = maxOf(-4, minOf(3, l))
                L[16 * j + ii] = (l + 4).toByte()
            }
        }

        // hmask
        val hmOff = bOff + 0
        for (idx in 0 until io.github.kotlinmania.llama.ore.QK_K / 8) y[hmOff + idx] = 0
        var m = 0
        var hm = 1
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K) {
            if ((L[j].toInt() and 0xFF) > 3) {
                y[hmOff + m] = ((y[hmOff + m].toInt() and 0xFF) or hm).toByte()
                L[j] = ((L[j].toInt() and 0xFF) - 4).toByte()
            }
            m++
            if (m == io.github.kotlinmania.llama.ore.QK_K / 8) { m = 0; hm = hm shl 1 }
        }

        // pack qs
        val qsOff = bOff + 32
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                y[qsOff + j / 4 + l] = ((L[j + l].toInt() and 0xFF) or ((L[j + l + 32].toInt() and 0xFF) shl 2) or ((L[j + l + 64].toInt() and 0xFF) shl 4) or ((L[j + l + 96].toInt() and 0xFF) shl 6)).toByte()
            }
        }

        xCur += io.github.kotlinmania.llama.ore.QK_K
    }
}

// C line 1377
fun quantize_q3_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q3_K,
        nPerRow
    ).toLong()
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q3_K_ref(src, 0, dst, 0, nrows * nPerRow)
    } else {
        var srcOff = 0; var dstOff = 0
        for (row in 0 until nrows) {
            io.github.kotlinmania.llama.ore.quantize_row_q3_K_impl(
                src,
                srcOff,
                dst,
                dstOff,
                nPerRow.toInt(),
                imatrix,
                (row * nPerRow).toInt()
            )
            srcOff += nPerRow.toInt()
            dstOff += rowSize.toInt()
        }
    }
    return nrows * rowSize
}

// C line 1491: quantize_row_q4_K_impl
// Block layout: d(fp16 at 0), dmin(fp16 at 2), scales(12 at 4), qs(128 at 16). Block size = 144.
private fun quantize_row_q4_K_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK_K

    val L = UByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val Laux = UByteArray(32)
    val Ls = UByteArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val Lm = UByteArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val weights = FloatArray(32)
    val sw = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val mins = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)

    var xCur = xOff
    for (i in 0 until nb) {
        var sumX2 = 0f
        for (l in 0 until io.github.kotlinmania.llama.ore.QK_K) sumX2 += x[xCur + l] * x[xCur + l]
        val sigma2 = 2 * sumX2 / io.github.kotlinmania.llama.ore.QK_K
        val avX = kotlin.math.sqrt(sigma2)

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val qwBase = qwOff + io.github.kotlinmania.llama.ore.QK_K * i + 32 * j
            for (l in 0 until 32) weights[l] = quantWeights[qwBase + l] * kotlin.math.sqrt(sigma2 + x[xCur + 32 * j + l] * x[xCur + 32 * j + l])
            var sumw = 0f
            for (l in 0 until 32) sumw += weights[l]
            sw[j] = sumw
            scales[j] = io.github.kotlinmania.llama.ore.make_qkx3_quants(
                32,
                15,
                x,
                xCur + 32 * j,
                weights,
                0,
                L,
                32 * j,
                mins,
                j,
                Laux,
                0,
                -0.9f,
                0.05f,
                36,
                false
            )
        }

        val dBlock = io.github.kotlinmania.llama.ore.make_qp_quants(
            io.github.kotlinmania.llama.ore.QK_K / 32,
            63,
            scales,
            0,
            Ls,
            0,
            sw,
            0
        )
        val mBlock = io.github.kotlinmania.llama.ore.make_qp_quants(
            io.github.kotlinmania.llama.ore.QK_K / 32,
            63,
            mins,
            0,
            Lm,
            0,
            sw,
            0
        )

        val bOff = yOff + 144 * i
        val scOff = bOff + 4
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val ls = Ls[j].toInt()
            val lm = Lm[j].toInt()
            if (j < 4) {
                y[scOff + j] = ls.toByte()
                y[scOff + j + 4] = lm.toByte()
            } else {
                y[scOff + j + 4] = ((ls and 0xF) or ((lm and 0xF) shl 4)).toByte()
                y[scOff + j - 4] = ((y[scOff + j - 4].toInt() and 0xFF) or ((ls shr 4) shl 6)).toByte()
                y[scOff + j - 0] = ((y[scOff + j - 0].toInt() and 0xFF) or ((lm shr 4) shl 6)).toByte()
            }
        }
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, dBlock)
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 2, mBlock)

        // requantize
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val sm = io.github.kotlinmania.llama.ore.get_scale_min_k4(j, y, scOff)
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 0) * sm.d
            if (d == 0f) continue
            val dm = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 2) * sm.m
            for (ii in 0 until 32) {
                var l =
                    io.github.kotlinmania.llama.ore.nearest_int((x[xCur + 32 * j + ii] + dm) / d)
                l = maxOf(0, minOf(15, l))
                L[32 * j + ii] = l.toUByte()
            }
        }

        // pack qs
        var qOff = bOff + 16
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
            for (l in 0 until 32) y[qOff + l] = (L[j + l].toInt() or (L[j + l + 32].toInt() shl 4)).toByte()
            qOff += 32
        }

        xCur += io.github.kotlinmania.llama.ore.QK_K
    }
}

// C line 1564
fun quantize_q4_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q4_K,
        nPerRow
    ).toLong()
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q4_K_ref(src, 0, dst, 0, nrows * nPerRow)
    } else {
        var srcOff = 0; var dstOff = 0
        for (row in 0 until nrows) {
            io.github.kotlinmania.llama.ore.quantize_row_q4_K_impl(
                src,
                srcOff,
                dst,
                dstOff,
                nPerRow.toInt(),
                imatrix,
                (row * nPerRow).toInt()
            )
            srcOff += nPerRow.toInt()
            dstOff += rowSize.toInt()
        }
    }
    return nrows * rowSize
}

// C line 1696: quantize_row_q5_K_impl
// Block layout: d(fp16 at 0), dmin(fp16 at 2), scales(12 at 4), qh(32 at 16), qs(128 at 48). Block size = 176.
private fun quantize_row_q5_K_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK_K

    val L = UByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val Laux = UByteArray(32)
    val Ls = UByteArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val Lm = UByteArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val mins = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val sw = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val weights = FloatArray(32)

    var xCur = xOff
    for (i in 0 until nb) {
        var sumX2 = 0f
        for (l in 0 until io.github.kotlinmania.llama.ore.QK_K) sumX2 += x[xCur + l] * x[xCur + l]
        val sigma2 = 2 * sumX2 / io.github.kotlinmania.llama.ore.QK_K
        val avX = kotlin.math.sqrt(sigma2)

        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val qwBase = qwOff + io.github.kotlinmania.llama.ore.QK_K * i + 32 * j
            for (l in 0 until 32) weights[l] = quantWeights[qwBase + l] * kotlin.math.sqrt(sigma2 + x[xCur + 32 * j + l] * x[xCur + 32 * j + l])
            var sumw = 0f
            for (l in 0 until 32) sumw += weights[l]
            sw[j] = sumw
            scales[j] = io.github.kotlinmania.llama.ore.make_qkx3_quants(
                32,
                31,
                x,
                xCur + 32 * j,
                weights,
                0,
                L,
                32 * j,
                mins,
                j,
                Laux,
                0,
                -0.9f,
                0.05f,
                36,
                false
            )
        }

        val dBlock = io.github.kotlinmania.llama.ore.make_qp_quants(
            io.github.kotlinmania.llama.ore.QK_K / 32,
            63,
            scales,
            0,
            Ls,
            0,
            sw,
            0
        )
        val mBlock = io.github.kotlinmania.llama.ore.make_qp_quants(
            io.github.kotlinmania.llama.ore.QK_K / 32,
            63,
            mins,
            0,
            Lm,
            0,
            sw,
            0
        )

        val bOff = yOff + 176 * i
        val scOff = bOff + 4
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            var ls = minOf(63, Ls[j].toInt())
            var lm = minOf(63, Lm[j].toInt())
            if (j < 4) {
                y[scOff + j] = ls.toByte()
                y[scOff + j + 4] = lm.toByte()
            } else {
                y[scOff + j + 4] = ((ls and 0xF) or ((lm and 0xF) shl 4)).toByte()
                y[scOff + j - 4] = ((y[scOff + j - 4].toInt() and 0xFF) or ((ls shr 4) shl 6)).toByte()
                y[scOff + j - 0] = ((y[scOff + j - 0].toInt() and 0xFF) or ((lm shr 4) shl 6)).toByte()
            }
        }
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 0, dBlock)
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 2, mBlock)

        // requantize
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val sm = io.github.kotlinmania.llama.ore.get_scale_min_k4(j, y, scOff)
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 0) * sm.d
            if (d == 0f) continue
            val dm = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 2) * sm.m
            for (ii in 0 until 32) {
                var l =
                    io.github.kotlinmania.llama.ore.nearest_int((x[xCur + 32 * j + ii] + dm) / d)
                l = maxOf(0, minOf(31, l))
                L[32 * j + ii] = l.toUByte()
            }
        }

        // pack qh and ql
        val qhOff = bOff + 16
        val qlOff = bOff + 48
        for (idx in 0 until io.github.kotlinmania.llama.ore.QK_K / 8) y[qhOff + idx] = 0

        var m1 = 1
        var m2 = 2
        var qlCur = qlOff
        for (n in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
            for (j in 0 until 32) {
                var l1 = L[n + j].toInt()
                if (l1 > 15) {
                    l1 -= 16; y[qhOff + j] = ((y[qhOff + j].toInt() and 0xFF) or m1).toByte()
                }
                var l2 = L[n + j + 32].toInt()
                if (l2 > 15) {
                    l2 -= 16; y[qhOff + j] = ((y[qhOff + j].toInt() and 0xFF) or m2).toByte()
                }
                y[qlCur + j] = (l1 or (l2 shl 4)).toByte()
            }
            m1 = m1 shl 2; m2 = m2 shl 2
            qlCur += 32
        }

        xCur += io.github.kotlinmania.llama.ore.QK_K
    }
}

// C line 1789
fun quantize_q5_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q5_K,
        nPerRow
    ).toLong()
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q5_K_ref(src, 0, dst, 0, nrows * nPerRow)
    } else {
        var srcOff = 0; var dstOff = 0
        for (row in 0 until nrows) {
            io.github.kotlinmania.llama.ore.quantize_row_q5_K_impl(
                src,
                srcOff,
                dst,
                dstOff,
                nPerRow.toInt(),
                imatrix,
                (row * nPerRow).toInt()
            )
            srcOff += nPerRow.toInt()
            dstOff += rowSize.toInt()
        }
    }
    return nrows * rowSize
}

// C line 1908: quantize_row_q6_K_impl
// Block layout: ql(128 at 0), qh(64 at 128), scales(16 at 192), d(fp16 at 208). Block size = 210.
private fun quantize_row_q6_K_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int,
    nPerRow: Int, quantWeights: FloatArray, qwOff: Int
) {
    val nb = nPerRow / io.github.kotlinmania.llama.ore.QK_K

    val L = ByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)

    var xCur = xOff
    for (i in 0 until nb) {
        var maxScale = 0f
        var maxAbsScale = 0f

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val scale: Float
            val qwBase = qwOff + io.github.kotlinmania.llama.ore.QK_K * i + 16 * ib
            scale = io.github.kotlinmania.llama.ore.make_qx_quants(
                16,
                32,
                x,
                xCur + 16 * ib,
                L,
                16 * ib,
                1,
                quantWeights,
                qwBase
            )
            scales[ib] = scale
            val absScale = kotlin.math.abs(scale)
            if (absScale > maxAbsScale) {
                maxAbsScale = absScale
                maxScale = scale
            }
        }

        val bOff = yOff + 210 * i
        if (maxAbsScale < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
            for (idx in 0 until 210) y[bOff + idx] = 0
            io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 208, 0f)
            xCur += io.github.kotlinmania.llama.ore.QK_K
            continue
        }

        val iscale = -128f / maxScale
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff + 208, 1f / iscale)
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            y[bOff + 192 + ib] = minOf(127,
                io.github.kotlinmania.llama.ore.nearest_int(iscale * scales[ib])
            ).toByte()
        }

        // requantize
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val d = io.github.kotlinmania.llama.ore.fp16ToF32(y, bOff + 208) * y[bOff + 192 + j].toInt()
            if (d == 0f) continue
            for (ii in 0 until 16) {
                var l = io.github.kotlinmania.llama.ore.nearest_int(x[xCur + 16 * j + ii] / d)
                l = maxOf(-32, minOf(31, l))
                L[16 * j + ii] = (l + 32).toByte()
            }
        }

        // pack ql and qh
        var qlOff = bOff + 0
        var qhOff = bOff + 128
        for (j in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
            for (l in 0 until 32) {
                val q1 = (L[j + l + 0].toInt() and 0xFF) and 0xF
                val q2 = (L[j + l + 32].toInt() and 0xFF) and 0xF
                val q3 = (L[j + l + 64].toInt() and 0xFF) and 0xF
                val q4 = (L[j + l + 96].toInt() and 0xFF) and 0xF
                y[qlOff + l + 0] = (q1 or (q3 shl 4)).toByte()
                y[qlOff + l + 32] = (q2 or (q4 shl 4)).toByte()
                y[qhOff + l] = (((L[j + l].toInt() and 0xFF) shr 4) or (((L[j + l + 32].toInt() and 0xFF) shr 4) shl 2) or (((L[j + l + 64].toInt() and 0xFF) shr 4) shl 4) or (((L[j + l + 96].toInt() and 0xFF) shr 4) shl 6)).toByte()
            }
            qlOff += 64
            qhOff += 32
        }

        xCur += io.github.kotlinmania.llama.ore.QK_K
    }
}

// C line 1992
fun quantize_q6_K(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(
        io.github.kotlinmania.llama.ore.GGMLType.Q6_K,
        nPerRow
    ).toLong()
    if (imatrix == null) {
        io.github.kotlinmania.llama.ore.quantize_row_q6_K_ref(src, 0, dst, 0, nrows * nPerRow)
    } else {
        var srcOff = 0; var dstOff = 0
        for (row in 0 until nrows) {
            io.github.kotlinmania.llama.ore.quantize_row_q6_K_impl(
                src,
                srcOff,
                dst,
                dstOff,
                nPerRow.toInt(),
                imatrix,
                (row * nPerRow).toInt()
            )
            srcOff += nPerRow.toInt()
            dstOff += rowSize.toInt()
        }
    }
    return nrows * rowSize
}

// C line 2338
fun quantize_tq1_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = (nPerRow / io.github.kotlinmania.llama.ore.QK_K) * io.github.kotlinmania.llama.ore.BlockTQ1_0.SIZE_BYTES
    io.github.kotlinmania.llama.ore.quantize_row_tq1_0_ref(src, 0, dst, 0, nrows * nPerRow)
    return nrows * rowSize
}

// C line 2345
fun quantize_tq2_0(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    val rowSize = (nPerRow / io.github.kotlinmania.llama.ore.QK_K) * io.github.kotlinmania.llama.ore.BlockTQ2_0.SIZE_BYTES
    io.github.kotlinmania.llama.ore.quantize_row_tq2_0_ref(src, 0, dst, 0, nrows * nPerRow)
    return nrows * rowSize
}

// C line 2226
fun quantize_mxfp4(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    io.github.kotlinmania.llama.ore.quantize_row_mxfp4_ref(src, 0, dst, 0, nrows * nPerRow)
    val rowSize = (nPerRow / io.github.kotlinmania.llama.ore.QK_MXFP4) * io.github.kotlinmania.llama.ore.BlockMXFP4.SIZE_BYTES
    return nrows * rowSize
}

// C line 2232
fun quantize_nvfp4(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    io.github.kotlinmania.llama.ore.quantize_row_nvfp4_ref(src, 0, dst, 0, nrows * nPerRow)
    val rowSize = (nPerRow / io.github.kotlinmania.llama.ore.QK_NVFP4) * io.github.kotlinmania.llama.ore.BlockNVFP4.SIZE_BYTES
    return nrows * rowSize
}

// C line 3167: quantize_row_iq2_xxs_impl
private fun quantize_row_iq2_xxs_impl(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, n: Int, quantWeights: FloatArray?) {
    val gindex =
        io.github.kotlinmania.llama.ore.iq2DataIndex(io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS)
    val kgridQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].grid!!
    val kmapQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].map!!
    val kneighborsQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].neighbours!!
    require(quantWeights != null) { "missing quantization weights" }
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    val kMaxQ = 3
    val nbl = n / io.github.kotlinmania.llama.ore.QK_K
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ2XXS.SIZE_BYTES

    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val weight = FloatArray(32)
    val xval = FloatArray(32)
    val L = ByteArray(32)
    val Laux = ByteArray(32)
    val waux = FloatArray(32)
    val blockSigns = IntArray(4)
    val q2 = IntArray(2 * (io.github.kotlinmania.llama.ore.QK_K / 32))

    for (ibl in 0 until nbl) {
        val bOff = yOff + ibl * blockSize
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, 0f)
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 4) y[bOff + 2 + i] = 0

        var maxScale = 0f
        val xblOff = xOff + io.github.kotlinmania.llama.ore.QK_K * ibl
        var sumx2 = 0f
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xblOff + i] * x[xblOff + i]
        val sigma2 = sumx2 / io.github.kotlinmania.llama.ore.QK_K

        q2.fill(0)

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val xbOff = xblOff + 32 * ib
            val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + 32 * ib
            for (i in 0 until 32) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            for (i in 0 until 32) waux[i] = kotlin.math.sqrt(weight[i])
            for (k in 0 until 4) {
                var nflip = 0
                var s = 0
                for (i in 0 until 8) {
                    if (x[xbOff + 8 * k + i] >= 0) xval[8 * k + i] = x[xbOff + 8 * k + i]
                    else { xval[8 * k + i] = -x[xbOff + 8 * k + i]; nflip++; s = s or (1 shl i) }
                }
                if (nflip % 2 != 0) {
                    var imin = 0; var min = weight[8 * k + imin] * x[xbOff + 8 * k + imin] * x[xbOff + 8 * k + imin]
                    for (i in 1 until 8) {
                        val ax = weight[8 * k + i] * x[xbOff + 8 * k + i] * x[xbOff + 8 * k + i]
                        if (ax < min) { min = ax; imin = i }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin]
                    s = s xor (1 shl imin)
                }
                blockSigns[k] = s and 127
            }
            var max = xval[0]
            for (i in 1 until 32) max = maxOf(max, xval[i])
            if (max < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) {
                scales[ib] = 0f; L.fill(0, 0, 32); continue
            }
            val Lp = UByteArray(32)
            var scale = io.github.kotlinmania.llama.ore.make_qp_quants(
                32,
                kMaxQ + 1,
                xval,
                0,
                Lp,
                0,
                weight,
                0
            )
            for (i in 0 until 32) L[i] = Lp[i].toByte()
            var effMax = scale * kMaxQ
            if (effMax <= 0) { scales[ib] = 0f; L.fill(0, 0, 32); continue }
            var best = 0f
            for (is_ in -6..6) {
                val id = (2 * kMaxQ - 1 + is_ * 0.1f) / effMax
                val thisScale = 1f / id
                for (k in 0 until 4) {
                    for (i in 0 until 8) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[8 * k + i] - 1))
                        Laux[8 * k + i] = maxOf(0, minOf(kMaxQ - 1, l)).toByte()
                    }
                    var u = 0
                    for (i in 0 until 8) u = u or (Laux[8 * k + i].toInt() shl (2 * i))
                    var gridIdx = kmapQ2xs[u]
                    if (gridIdx < 0) {
                        val nOff = -kmapQ2xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq2FindBestNeighbour(
                            kneighborsQ2xs,
                            nOff,
                            kgridQ2xs,
                            xval,
                            8 * k,
                            waux,
                            8 * k,
                            thisScale,
                            Laux,
                            8 * k
                        )
                    }
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 32) {
                    val w = weight[i]; val q = 2f * Laux[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0 && sumqx * sumqx > best * sumq2f) {
                    scale = sumqx / sumq2f; best = scale * sumqx
                    Laux.copyInto(L, 0, 0, 32)
                }
            }
            if (scale > 0) {
                val id = 1f / scale
                for (k in 0 until 4) {
                    var u = 0
                    for (i in 0 until 8) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[8 * k + i] - 1))
                        l = maxOf(0, minOf(kMaxQ - 1, l))
                        u = u or (l shl (2 * i))
                    }
                    var gridIdx = kmapQ2xs[u]
                    if (gridIdx < 0) {
                        val nOff = -kmapQ2xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq2FindBestNeighbour(
                            kneighborsQ2xs,
                            nOff,
                            kgridQ2xs,
                            xval,
                            8 * k,
                            waux,
                            8 * k,
                            scale,
                            L,
                            8 * k
                        )
                    }
                    val gv = kgridQ2xs[gridIdx]
                    for (i in 0 until 8) L[8 * k + i] = ((io.github.kotlinmania.llama.ore.gridByte(
                        gv,
                        i
                    ) - 1) / 2).toByte()
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 32) {
                    val w = weight[i]; val q = 2f * L[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0) scale = sumqx / sumq2f
            }
            if (scale < 0) {
                scale = -scale
                for (k in 0 until 4) blockSigns[k] = (blockSigns[k].inv()) and 127
            }
            for (k in 0 until 4) {
                var u = 0
                for (i in 0 until 8) u = u or (L[8 * k + i].toInt() shl (2 * i))
                val gridIdx = kmapQ2xs[u]
                require(gridIdx >= 0) { "Oops: found point $u not on grid" }
                q2[2 * ib + 0] = q2[2 * ib + 0] or (gridIdx shl (8 * k))
                q2[2 * ib + 1] = q2[2 * ib + 1] or (blockSigns[k] shl (7 * k))
            }
            require(scale >= 0)
            scales[ib] = scale
            maxScale = maxOf(maxScale, scale)
        }

        if (maxScale == 0f) {
            for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 4) y[bOff + 2 + i] = 0
            continue
        }

        val d = maxScale / 31f
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, d)
        val id = 1f / d
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib] - 1))
            l = maxOf(0, minOf(15, l))
            q2[2 * ib + 1] = q2[2 * ib + 1] or (l shl 28)
        }
        // copy q2 to qs as bytes
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 32 * 2) io.github.kotlinmania.llama.ore.writeIntLE(
            y,
            bOff + 2 + i * 4,
            q2[i]
        )
    }
}

// C line 3345: quantize_row_iq2_xs_impl
private fun quantize_row_iq2_xs_impl(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, n: Int, quantWeights: FloatArray?) {
    val gindex =
        io.github.kotlinmania.llama.ore.iq2DataIndex(io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS)
    val kgridQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].grid!!
    val kmapQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].map!!
    val kneighborsQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].neighbours!!
    require(quantWeights != null) { "missing quantization weights" }
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    val kMaxQ = 3
    val nbl = n / io.github.kotlinmania.llama.ore.QK_K
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ2XS.SIZE_BYTES

    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val weight = FloatArray(16)
    val xval = FloatArray(16)
    val L = ByteArray(16)
    val Laux = ByteArray(16)
    val waux = FloatArray(16)
    val isOnGrid = BooleanArray(2)
    val isOnGridAux = BooleanArray(2)
    val blockSigns = IntArray(2)
    val q2 = ShortArray(2 * (io.github.kotlinmania.llama.ore.QK_K / 16))

    for (ibl in 0 until nbl) {
        val bOff = yOff + ibl * blockSize
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, 0f)
        q2.fill(0)
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) y[bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + i] = 0 // scales

        var maxScale = 0f
        val xblOff = xOff + io.github.kotlinmania.llama.ore.QK_K * ibl
        var sumx2 = 0f
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xblOff + i] * x[xblOff + i]
        val sigma2 = sumx2 / io.github.kotlinmania.llama.ore.QK_K

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val xbOff = xblOff + 16 * ib
            val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + 16 * ib
            for (i in 0 until 16) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            for (i in 0 until 16) waux[i] = kotlin.math.sqrt(weight[i])
            for (k in 0 until 2) {
                var nflip = 0; var s = 0
                for (i in 0 until 8) {
                    if (x[xbOff + 8 * k + i] >= 0) xval[8 * k + i] = x[xbOff + 8 * k + i]
                    else { xval[8 * k + i] = -x[xbOff + 8 * k + i]; nflip++; s = s or (1 shl i) }
                }
                if (nflip % 2 != 0) {
                    var imin = 0; var min = weight[8 * k + imin] * x[xbOff + 8 * k + imin] * x[xbOff + 8 * k + imin]
                    for (i in 1 until 8) {
                        val ax = weight[8 * k + i] * x[xbOff + 8 * k + i] * x[xbOff + 8 * k + i]
                        if (ax < min) { min = ax; imin = i }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin]
                    s = s xor (1 shl imin)
                }
                blockSigns[k] = s and 127
            }
            var max = xval[0]
            for (i in 1 until 16) max = maxOf(max, xval[i])
            L.fill(0, 0, 16)
            if (max < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS) { scales[ib] = 0f; continue }
            var best = 0f
            var scale = max / (2 * kMaxQ - 1)
            isOnGrid[0] = true; isOnGrid[1] = true
            for (is_ in -9..9) {
                val id = (2 * kMaxQ - 1 + is_ * 0.1f) / max
                val thisScale = 1f / id
                for (k in 0 until 2) {
                    for (i in 0 until 8) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[8 * k + i] - 1))
                        Laux[8 * k + i] = maxOf(0, minOf(kMaxQ - 1, l)).toByte()
                    }
                    var u = 0
                    for (i in 0 until 8) u = u or (Laux[8 * k + i].toInt() shl (2 * i))
                    var gridIdx = kmapQ2xs[u]
                    isOnGridAux[k] = true
                    if (gridIdx < 0) {
                        isOnGridAux[k] = false
                        val nOff = -kmapQ2xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq2FindBestNeighbour(
                            kneighborsQ2xs,
                            nOff,
                            kgridQ2xs,
                            xval,
                            8 * k,
                            waux,
                            8 * k,
                            thisScale,
                            Laux,
                            8 * k
                        )
                    }
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 16) {
                    val w = weight[i]; val q = 2f * Laux[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0 && sumqx * sumqx > best * sumq2f) {
                    scale = sumqx / sumq2f; best = scale * sumqx
                    for (i in 0 until 16) L[i] = Laux[i]
                    for (k in 0 until 2) isOnGrid[k] = isOnGridAux[k]
                }
            }
            var nNotOnGrid = 0
            for (k in 0 until 2) if (!isOnGrid[k]) nNotOnGrid++
            if (nNotOnGrid > 0 && scale > 0) {
                val id = 1f / scale
                for (k in 0 until 2) {
                    if (isOnGrid[k]) continue
                    var u = 0
                    for (i in 0 until 8) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[8 * k + i] - 1))
                        l = maxOf(0, minOf(kMaxQ - 1, l))
                        u = u or (l shl (2 * i))
                        L[8 * k + i] = l.toByte()
                    }
                    var gridIdx = kmapQ2xs[u]
                    if (gridIdx < 0) {
                        val nOff = -kmapQ2xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq2FindBestNeighbour(
                            kneighborsQ2xs,
                            nOff,
                            kgridQ2xs,
                            xval,
                            8 * k,
                            waux,
                            8 * k,
                            scale,
                            L,
                            8 * k
                        )
                    }
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 16) {
                    val w = weight[i]; val q = 2f * L[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0) scale = sumqx / sumq2f
            }
            if (scale < 0) {
                scale = -scale
                for (k in 0 until 2) blockSigns[k] = (blockSigns[k].inv()) and 127
            }
            for (k in 0 until 2) {
                var u = 0
                for (i in 0 until 8) u = u or (L[8 * k + i].toInt() shl (2 * i))
                val gridIdx = kmapQ2xs[u]
                require(gridIdx >= 0) { "Oops: found point $u not on grid" }
                q2[2 * ib + k] = (gridIdx or (blockSigns[k] shl 9)).toShort()
            }
            require(scale >= 0)
            scales[ib] = scale
            maxScale = maxOf(maxScale, scale)
        }

        if (maxScale == 0f) {
            for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 4) y[bOff + 2 + i] = 0
            continue
        }

        val d = maxScale / 31f
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, d)
        val id = 1f / d
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib] - 1))
            l = maxOf(0, minOf(15, l))
            val scalesOff = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4
            if (ib % 2 == 0) y[scalesOff + ib / 2] = l.toByte()
            else y[scalesOff + ib / 2] = (y[scalesOff + ib / 2].toInt() or (l shl 4)).toByte()
        }
        // copy q2 to qs bytes
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 16 * 2) io.github.kotlinmania.llama.ore.writeShortLE(
            y,
            bOff + 2 + i * 2,
            q2[i]
        )
    }
}

// C line 4970: quantize_row_iq2_s_impl
private fun quantize_row_iq2_s_impl(x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, n: Int, quantWeights: FloatArray?) {
    val gindex =
        io.github.kotlinmania.llama.ore.iq2DataIndex(io.github.kotlinmania.llama.ore.GGMLType.IQ2_S)
    val kgridQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].grid!!
    val kmapQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].map!!
    val kneighborsQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].neighbours!!
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    val kMaxQ = 3
    val nbl = n / io.github.kotlinmania.llama.ore.QK_K
    val blockSize = io.github.kotlinmania.llama.ore.BlockIQ2S.SIZE_BYTES

    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
    val weight = FloatArray(16)
    val xval = FloatArray(16)
    val L = ByteArray(16)
    val Laux = ByteArray(16)
    val waux = FloatArray(16)
    val isOnGrid = BooleanArray(2)
    val isOnGridAux = BooleanArray(2)
    val blockSigns = IntArray(2)

    for (ibl in 0 until nbl) {
        val bOff = yOff + ibl * blockSize
        // zero the block
        for (i in 0 until blockSize) y[bOff + i] = 0
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, 0f)

        var maxScale = 0f
        val xblOff = xOff + io.github.kotlinmania.llama.ore.QK_K * ibl
        var sumx2 = 0f
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xblOff + i] * x[xblOff + i]
        val sigma2 = 2f * sumx2 / io.github.kotlinmania.llama.ore.QK_K

        // block_iq2_s layout: d(2) + qs(QK_K/4=64) + qh(QK_K/32=8) + scales(QK_K/32=8) = 82 (but we also have sign bytes)
        // Actually: d(2) + qs(QK_K/8=32) + qs_signs(QK_K/8=32) + qh(QK_K/32=8) + scales(QK_K/32=8) = 82
        // C layout: d(2), qs[QK_K/4]=64 bytes (first 32 = grid indices, next 32 = signs), qh[QK_K/32]=8, scales[QK_K/32]=8
        val qsOff = bOff + 2           // qs: grid index low bits (32 bytes) then sign bytes (32 bytes)
        val qhOff = bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 4  // qh: 8 bytes
        val scOff = qhOff + io.github.kotlinmania.llama.ore.QK_K / 32     // scales: 8 bytes

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val xbOff = xblOff + 16 * ib
            if (quantWeights != null) {
                val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + 16 * ib
                for (i in 0 until 16) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            } else {
                for (i in 0 until 16) weight[i] = 0.25f * sigma2 + x[xbOff + i] * x[xbOff + i]
            }
            for (i in 0 until 16) waux[i] = kotlin.math.sqrt(weight[i])
            for (k in 0 until 2) {
                var s = 0
                for (i in 0 until 8) {
                    if (x[xbOff + 8 * k + i] >= 0) xval[8 * k + i] = x[xbOff + 8 * k + i]
                    else { xval[8 * k + i] = -x[xbOff + 8 * k + i]; s = s or (1 shl i) }
                }
                blockSigns[k] = s
            }
            var max = xval[0]
            for (i in 1 until 16) max = maxOf(max, xval[i])
            L.fill(0, 0, 16)
            if (max < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS_IQ2_S) { scales[ib] = 0f; continue }
            var best = 0f
            var scale = max / (2 * kMaxQ - 1)
            isOnGrid[0] = true; isOnGrid[1] = true
            for (is_ in -9..9) {
                val id = (2 * kMaxQ - 1 + is_ * 0.1f) / max
                val thisScale = 1f / id
                for (k in 0 until 2) {
                    for (i in 0 until 8) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[8 * k + i] - 1))
                        Laux[8 * k + i] = maxOf(0, minOf(kMaxQ - 1, l)).toByte()
                    }
                    var u = 0
                    for (i in 0 until 8) u = u or (Laux[8 * k + i].toInt() shl (2 * i))
                    var gridIdx = kmapQ2xs[u]
                    isOnGridAux[k] = true
                    if (gridIdx < 0) {
                        isOnGridAux[k] = false
                        val nOff = -kmapQ2xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq2FindBestNeighbour(
                            kneighborsQ2xs,
                            nOff,
                            kgridQ2xs,
                            xval,
                            8 * k,
                            waux,
                            8 * k,
                            thisScale,
                            Laux,
                            8 * k
                        )
                    }
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 16) {
                    val w = weight[i]; val q = 2f * Laux[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0 && sumqx * sumqx > best * sumq2f) {
                    scale = sumqx / sumq2f; best = scale * sumqx
                    for (i in 0 until 16) L[i] = Laux[i]
                    for (k in 0 until 2) isOnGrid[k] = isOnGridAux[k]
                }
            }
            var nNotOnGrid = 0
            for (k in 0 until 2) if (!isOnGrid[k]) nNotOnGrid++
            if (nNotOnGrid > 0 && scale > 0) {
                val id = 1f / scale
                for (k in 0 until 2) {
                    if (isOnGrid[k]) continue
                    var u = 0
                    for (i in 0 until 8) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[8 * k + i] - 1))
                        l = maxOf(0, minOf(kMaxQ - 1, l))
                        u = u or (l shl (2 * i))
                        L[8 * k + i] = l.toByte()
                    }
                    var gridIdx = kmapQ2xs[u]
                    if (gridIdx < 0) {
                        val nOff = -kmapQ2xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq2FindBestNeighbour(
                            kneighborsQ2xs,
                            nOff,
                            kgridQ2xs,
                            xval,
                            8 * k,
                            waux,
                            8 * k,
                            scale,
                            L,
                            8 * k
                        )
                    }
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 16) {
                    val w = weight[i]; val q = 2f * L[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0) scale = sumqx / sumq2f
            }
            if (scale < 0) {
                scale = -scale
                for (k in 0 until 2) blockSigns[k] = blockSigns[k].inv() and 0xFF
            }
            for (k in 0 until 2) {
                var u = 0
                for (i in 0 until 8) u = u or (L[8 * k + i].toInt() shl (2 * i))
                val gridIdx = kmapQ2xs[u]
                require(gridIdx >= 0) { "Oops: found point $u not on grid" }
                val i8 = 2 * ib + k
                y[qsOff + i8] = (gridIdx and 255).toByte()
                y[qhOff + i8 / 4] = (y[qhOff + i8 / 4].toInt() or ((gridIdx shr 8) shl (2 * (i8 % 4)))).toByte()
                y[qsOff + io.github.kotlinmania.llama.ore.QK_K / 8 + i8] = blockSigns[k].toByte()
            }
            require(scale >= 0)
            scales[ib] = scale
            maxScale = maxOf(maxScale, scale)
        }

        if (maxScale == 0f) continue

        val d = maxScale / 31f
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, d * 0.9875f)
        val id = 1f / d
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib] - 1))
            l = maxOf(0, minOf(15, l))
            if (ib % 2 == 0) y[scOff + ib / 2] = l.toByte()
            else y[scOff + ib / 2] = (y[scOff + ib / 2].toInt() or (l shl 4)).toByte()
        }
    }
}

// C line 3766: quantize_row_iq3_xxs_impl
private fun quantize_row_iq3_xxs_impl(gridSize: Int, x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, n: Int, quantWeights: FloatArray?) {
    val gindex = io.github.kotlinmania.llama.ore.iq3DataIndex(gridSize)
    val kgridQ3xs = io.github.kotlinmania.llama.ore.iq3Data[gindex].grid!!
    val kmapQ3xs = io.github.kotlinmania.llama.ore.iq3Data[gindex].map!!
    val kneighborsQ3xs = io.github.kotlinmania.llama.ore.iq3Data[gindex].neighbours!!
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    val kMaxQ = 8
    val nbl = n / io.github.kotlinmania.llama.ore.QK_K

    val blockSizeBl: Int
    val quantSize: Int
    if (gridSize == 256) {
        blockSizeBl = io.github.kotlinmania.llama.ore.BlockIQ3XXS.SIZE_BYTES
    } else {
        blockSizeBl = io.github.kotlinmania.llama.ore.BlockIQ3S.SIZE_BYTES
    }
    quantSize = blockSizeBl - 2 // minus the fp16 d

    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    val weight = FloatArray(32)
    val xval = FloatArray(32)
    val L = ByteArray(32)
    val Laux = ByteArray(32)
    val waux = FloatArray(32)
    val isOnGrid = BooleanArray(8)
    val isOnGridAux = BooleanArray(8)
    val blockSigns = IntArray(8)
    val q3 = ByteArray(3 * (io.github.kotlinmania.llama.ore.QK_K / 8) + io.github.kotlinmania.llama.ore.QK_K / 32)
    // scales_and_signs start at q3[QK_K/4], qh starts at q3[3*(QK_K/8)]

    for (ibl in 0 until nbl) {
        val bOff = yOff + ibl * blockSizeBl
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, 0f)
        q3.fill(0)

        var maxScale = 0f
        val xblOff = xOff + io.github.kotlinmania.llama.ore.QK_K * ibl
        var sumx2 = 0f
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xblOff + i] * x[xblOff + i]
        val sigma2 = 2f * sumx2 / io.github.kotlinmania.llama.ore.QK_K

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            val xbOff = xblOff + 32 * ib
            if (quantWeights != null) {
                val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + 32 * ib
                for (i in 0 until 32) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            } else {
                for (i in 0 until 32) weight[i] = x[xbOff + i] * x[xbOff + i]
            }
            for (i in 0 until 32) waux[i] = kotlin.math.sqrt(weight[i])
            for (k in 0 until 4) {
                var nflip = 0; var s = 0
                for (i in 0 until 8) {
                    if (x[xbOff + 8 * k + i] >= 0) xval[8 * k + i] = x[xbOff + 8 * k + i]
                    else { xval[8 * k + i] = -x[xbOff + 8 * k + i]; nflip++; s = s or (1 shl i) }
                }
                if (nflip % 2 != 0) {
                    var imin = 0; var min = weight[8 * k + imin] * x[xbOff + 8 * k + imin] * x[xbOff + 8 * k + imin]
                    for (i in 1 until 8) {
                        val ax = weight[8 * k + i] * x[xbOff + 8 * k + i] * x[xbOff + 8 * k + i]
                        if (ax < min) { min = ax; imin = i }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin]
                    s = s xor (1 shl imin)
                }
                blockSigns[k] = s and 127
            }
            var max = xval[0]
            for (i in 1 until 32) max = maxOf(max, xval[i])
            L.fill(0, 0, 32)
            if (max < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS_IQ3_XXS) { scales[ib] = 0f; continue }
            var best = 0f
            var scale = max / (2 * kMaxQ - 1)
            for (k in 0 until 8) isOnGrid[k] = true
            for (is_ in -15..15) {
                val id = (2 * kMaxQ - 1 + is_ * 0.2f) / max
                val thisScale = 1f / id
                for (k in 0 until 8) {
                    for (i in 0 until 4) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[4 * k + i] - 1))
                        Laux[4 * k + i] = maxOf(0, minOf(kMaxQ - 1, l)).toByte()
                    }
                    var u = 0
                    for (i in 0 until 4) u = u or (Laux[4 * k + i].toInt() shl (3 * i))
                    var gridIdx = kmapQ3xs[u]
                    isOnGridAux[k] = true
                    if (gridIdx < 0) {
                        isOnGridAux[k] = false
                        val nOff = -kmapQ3xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq3FindBestNeighbour(
                            kneighborsQ3xs,
                            nOff,
                            kgridQ3xs,
                            xval,
                            4 * k,
                            waux,
                            4 * k,
                            thisScale,
                            Laux,
                            4 * k
                        )
                    }
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 32) {
                    val w = weight[i]; val q = 2f * Laux[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0 && sumqx * sumqx > best * sumq2f) {
                    scale = sumqx / sumq2f; best = scale * sumqx
                    for (i in 0 until 32) L[i] = Laux[i]
                    for (k in 0 until 8) isOnGrid[k] = isOnGridAux[k]
                }
            }
            var nNotOnGrid = 0
            for (k in 0 until 8) if (!isOnGrid[k]) nNotOnGrid++
            if (nNotOnGrid > 0 && scale > 0) {
                val id = 1f / scale
                for (k in 0 until 8) {
                    if (isOnGrid[k]) continue
                    var u = 0
                    for (i in 0 until 4) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[4 * k + i] - 1))
                        l = maxOf(0, minOf(kMaxQ - 1, l))
                        u = u or (l shl (3 * i))
                    }
                    var gridIdx = kmapQ3xs[u]
                    if (gridIdx < 0) {
                        val nOff = -kmapQ3xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq3FindBestNeighbour(
                            kneighborsQ3xs,
                            nOff,
                            kgridQ3xs,
                            xval,
                            4 * k,
                            waux,
                            4 * k,
                            scale,
                            L,
                            4 * k
                        )
                    }
                    val gv = kgridQ3xs[gridIdx]
                    for (i in 0 until 4) L[4 * k + i] = ((io.github.kotlinmania.llama.ore.gridByte(
                        gv,
                        i
                    ) - 1) / 2).toByte()
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until 32) {
                    val w = weight[i]; val q = 2f * L[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0) scale = sumqx / sumq2f
            }
            if (scale < 0) {
                scale = -scale
                for (k in 0 until 4) blockSigns[k] = (blockSigns[k].inv()) and 127
            }
            for (k in 0 until 8) {
                var u = 0
                for (i in 0 until 4) u = u or (L[4 * k + i].toInt() shl (3 * i))
                val gridIdx = kmapQ3xs[u]
                require(gridIdx >= 0) { "Oops: found point $u not on grid" }
                if (gridSize == 256) {
                    q3[8 * ib + k] = gridIdx.toByte()
                } else {
                    q3[8 * ib + k] = (gridIdx and 255).toByte()
                    q3[3 * (io.github.kotlinmania.llama.ore.QK_K / 8) + ib] = (q3[3 * (io.github.kotlinmania.llama.ore.QK_K / 8) + ib].toInt() or ((gridIdx shr 8) shl k)).toByte()
                }
            }
            // scales_and_signs at offset QK_K/4 = 64
            val sasOff = io.github.kotlinmania.llama.ore.QK_K / 4 + ib * 4
            val sas = blockSigns[0] or (blockSigns[1] shl 7) or (blockSigns[2] shl 14) or (blockSigns[3] shl 21)
            q3[sasOff + 0] = (sas and 0xFF).toByte()
            q3[sasOff + 1] = ((sas shr 8) and 0xFF).toByte()
            q3[sasOff + 2] = ((sas shr 16) and 0xFF).toByte()
            q3[sasOff + 3] = ((sas shr 24) and 0xFF).toByte()

            require(scale >= 0)
            scales[ib] = scale
            maxScale = maxOf(maxScale, scale)
        }

        if (maxScale == 0f) {
            for (i in 0 until quantSize) y[bOff + 2 + i] = 0
            continue
        }

        val d = maxScale / 31f
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, d * 1.0125f)
        val id = 1f / d
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib] - 1))
            l = maxOf(0, minOf(15, l))
            val sasOff = io.github.kotlinmania.llama.ore.QK_K / 4 + ib * 4
            val existing = (q3[sasOff + 3].toInt() and 0xFF) or ((q3[sasOff + 2].toInt() and 0xFF) shl 8) or
                ((q3[sasOff + 1].toInt() and 0xFF) shl 16) or ((q3[sasOff + 0].toInt() and 0xFF) shl 24)
            // store l in top 4 bits of the uint32
            val sasVal = ((q3[sasOff + 0].toInt() and 0xFF)) or
                ((q3[sasOff + 1].toInt() and 0xFF) shl 8) or
                ((q3[sasOff + 2].toInt() and 0xFF) shl 16) or
                ((q3[sasOff + 3].toInt() and 0xFF) shl 24)
            val updated = sasVal or (l shl 28)
            q3[sasOff + 0] = (updated and 0xFF).toByte()
            q3[sasOff + 1] = ((updated shr 8) and 0xFF).toByte()
            q3[sasOff + 2] = ((updated shr 16) and 0xFF).toByte()
            q3[sasOff + 3] = ((updated shr 24) and 0xFF).toByte()
        }
        q3.copyInto(y, bOff + 2, 0, quantSize)
    }
}

// C line 3997: quantize_row_iq3_s_impl
private fun quantize_row_iq3_s_impl(
    blkSize: Int, x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, n: Int,
    quantWeights: FloatArray?,
    scales: FloatArray, weight: FloatArray, xval: FloatArray,
    L: ByteArray, Laux: ByteArray, waux: FloatArray,
    isOnGrid: BooleanArray, isOnGridAux: BooleanArray, blockSigns: IntArray
) {
    val gindex = io.github.kotlinmania.llama.ore.iq3DataIndex(512)
    val kgridQ3xs = io.github.kotlinmania.llama.ore.iq3Data[gindex].grid!!
    val kmapQ3xs = io.github.kotlinmania.llama.ore.iq3Data[gindex].map!!
    val kneighborsQ3xs = io.github.kotlinmania.llama.ore.iq3Data[gindex].neighbours!!
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    val kMaxQ = 8
    val nbl = n / io.github.kotlinmania.llama.ore.QK_K
    val blockSizeBl = io.github.kotlinmania.llama.ore.BlockIQ3S.SIZE_BYTES
    val bs4 = blkSize / 4
    val bs8 = blkSize / 8

    for (ibl in 0 until nbl) {
        val bOff = yOff + ibl * blockSizeBl
        for (i in 0 until blockSizeBl) y[bOff + i] = 0
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, 0f)

        // block_iq3_s layout: d(2) + qs(QK_K/4=64) + qh(QK_K/32=8) + signs(QK_K/8=32) + scales(IQ3S_N_SCALE=4)
        val qsOff = bOff + 2
        val qhOff = qsOff + io.github.kotlinmania.llama.ore.QK_K / 4
        val signsOff = qhOff + io.github.kotlinmania.llama.ore.QK_K / 32
        val scOff = signsOff + io.github.kotlinmania.llama.ore.QK_K / 8

        var maxScale = 0f
        val xblOff = xOff + io.github.kotlinmania.llama.ore.QK_K * ibl
        var sumx2 = 0f
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xblOff + i] * x[xblOff + i]
        val sigma2 = 2f * sumx2 / io.github.kotlinmania.llama.ore.QK_K

        var qsPtr = qsOff
        var signsPtr = signsOff

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / blkSize) {
            val xbOff = xblOff + blkSize * ib
            if (quantWeights != null) {
                val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + blkSize * ib
                for (i in 0 until blkSize) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            } else {
                for (i in 0 until blkSize) weight[i] = x[xbOff + i] * x[xbOff + i]
            }
            for (i in 0 until blkSize) waux[i] = kotlin.math.sqrt(weight[i])
            for (k in 0 until bs8) {
                var s = 0
                for (i in 0 until 8) {
                    if (x[xbOff + 8 * k + i] >= 0) xval[8 * k + i] = x[xbOff + 8 * k + i]
                    else { xval[8 * k + i] = -x[xbOff + 8 * k + i]; s = s or (1 shl i) }
                }
                blockSigns[k] = s
            }
            var max = xval[0]
            for (i in 1 until blkSize) max = maxOf(max, xval[i])
            L.fill(0, 0, blkSize)
            if (max == 0f) { scales[ib] = 0f; qsPtr += bs4; signsPtr += bs8; continue }
            var best = 0f
            var scale = max / (2 * kMaxQ - 1)
            for (k in 0 until bs4) isOnGrid[k] = false
            for (is_ in -9..9) {
                val id = (2 * kMaxQ - 1 + is_ * 0.2f) / max
                val thisScale = 1f / id
                for (k in 0 until bs4) {
                    for (i in 0 until 4) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[4 * k + i] - 1))
                        Laux[4 * k + i] = maxOf(0, minOf(kMaxQ - 1, l)).toByte()
                    }
                    var u = 0
                    for (i in 0 until 4) u = u or (Laux[4 * k + i].toInt() shl (3 * i))
                    var gridIdx = kmapQ3xs[u]
                    isOnGridAux[k] = true
                    if (gridIdx < 0) {
                        isOnGridAux[k] = false
                        val nOff = -kmapQ3xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq3FindBestNeighbour(
                            kneighborsQ3xs,
                            nOff,
                            kgridQ3xs,
                            xval,
                            4 * k,
                            waux,
                            4 * k,
                            thisScale,
                            Laux,
                            4 * k
                        )
                    }
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until blkSize) {
                    val w = weight[i]; val q = 2f * Laux[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0 && sumqx * sumqx > best * sumq2f) {
                    scale = sumqx / sumq2f; best = scale * sumqx
                    for (i in 0 until blkSize) L[i] = Laux[i]
                    for (k in 0 until bs4) isOnGrid[k] = isOnGridAux[k]
                }
            }
            var nNotOnGrid = 0
            for (k in 0 until bs4) if (!isOnGrid[k]) nNotOnGrid++
            if (nNotOnGrid > 0 && scale > 0) {
                val id = 1f / scale
                for (k in 0 until bs4) {
                    var u = 0
                    for (i in 0 until 4) {
                        var l =
                            io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * xval[4 * k + i] - 1))
                        l = maxOf(0, minOf(kMaxQ - 1, l))
                        u = u or (l shl (3 * i))
                    }
                    var gridIdx = kmapQ3xs[u]
                    if (gridIdx < 0) {
                        val nOff = -kmapQ3xs[u] - 1
                        gridIdx = io.github.kotlinmania.llama.ore.iq3FindBestNeighbour(
                            kneighborsQ3xs,
                            nOff,
                            kgridQ3xs,
                            xval,
                            4 * k,
                            waux,
                            4 * k,
                            scale,
                            L,
                            4 * k
                        )
                    }
                    val gv = kgridQ3xs[gridIdx]
                    for (i in 0 until 4) L[4 * k + i] = ((io.github.kotlinmania.llama.ore.gridByte(
                        gv,
                        i
                    ) - 1) / 2).toByte()
                }
                var sumqx = 0f; var sumq2f = 0f
                for (i in 0 until blkSize) {
                    val w = weight[i]; val q = 2f * L[i] + 1f
                    sumqx += w * xval[i] * q; sumq2f += w * q * q
                }
                if (sumq2f > 0) scale = sumqx / sumq2f
            }
            if (scale < 0) {
                scale = -scale
                for (k in 0 until bs8) blockSigns[k] = blockSigns[k].inv() and 0xFF
            }
            for (k in 0 until bs4) {
                var u = 0
                for (i in 0 until 4) u = u or (L[4 * k + i].toInt() shl (3 * i))
                val gridIdx = kmapQ3xs[u]
                require(gridIdx >= 0) { "Oops: found point $u not on grid" }
                y[qsPtr + k] = (gridIdx and 255).toByte()
                val qhIdx = (ib * bs4 + k)
                y[qhOff + qhIdx / 8] = (y[qhOff + qhIdx / 8].toInt() or ((gridIdx shr 8) shl (qhIdx % 8))).toByte()
            }
            qsPtr += bs4
            for (k in 0 until bs8) y[signsPtr + k] = blockSigns[k].toByte()
            signsPtr += bs8
            require(scale >= 0)
            scales[ib] = scale
            maxScale = maxOf(maxScale, scale)
        }

        if (maxScale == 0f) continue

        val d = maxScale / 31f
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, d * 1.033f)
        val id = 1f / d
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / blkSize step 2) {
            var l1 = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib + 0] - 1))
            l1 = maxOf(0, minOf(15, l1))
            var l2 = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib + 1] - 1))
            l2 = maxOf(0, minOf(15, l2))
            y[scOff + ib / 2] = (l1 or (l2 shl 4)).toByte()
        }
    }
}

// C line 4334-4498: quantize_row_iq1_s_impl
private fun quantize_row_iq1_s_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, n: Int, quantWeights: FloatArray?,
    scales: FloatArray, weight: FloatArray, sumx: FloatArray, sumw: FloatArray,
    pairs: FloatArray, L: ByteArray, index: IntArray, shifts: ByteArray
) {
    val gindex =
        io.github.kotlinmania.llama.ore.iq2DataIndex(io.github.kotlinmania.llama.ore.GGMLType.IQ1_S)
    val kgridQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].grid!!
    val kmapQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].map!!
    val kneighborsQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].neighbours!!
    require(quantWeights != null) { "missing quantization weights" }
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    val nbl = n / io.github.kotlinmania.llama.ore.QK_K
    val blockSizeSub = 32 // IQ1S_BLOCK_SIZE
    val blockSizeBl = io.github.kotlinmania.llama.ore.BlockIQ1S.SIZE_BYTES

    val xP = floatArrayOf(-1f + io.github.kotlinmania.llama.ore.IQ1S_DELTA,
        io.github.kotlinmania.llama.ore.IQ1S_DELTA, 1f + io.github.kotlinmania.llama.ore.IQ1S_DELTA
    )
    val xM = floatArrayOf(-1f - io.github.kotlinmania.llama.ore.IQ1S_DELTA, -io.github.kotlinmania.llama.ore.IQ1S_DELTA, 1f - io.github.kotlinmania.llama.ore.IQ1S_DELTA)

    // idx shares memory with pairs: idx[2*j] is at pairs[2*j+1] reinterpreted as int
    // In Kotlin we use a separate IntArray for idx
    val idx = IntArray(2 * blockSizeSub)

    for (ibl in 0 until nbl) {
        val bOff = yOff + ibl * blockSizeBl
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, 0f)
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 8) y[bOff + 2 + i] = 0 // qs
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) io.github.kotlinmania.llama.ore.writeShortLE(
            y,
            bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 8 + i * 2,
            0
        ) // qh

        var maxScale = 0f
        val xblOff = xOff + io.github.kotlinmania.llama.ore.QK_K * ibl
        var sumx2 = 0f
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xblOff + i] * x[xblOff + i]
        val sigma2 = 2f * sumx2 / io.github.kotlinmania.llama.ore.QK_K

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / blockSizeSub) {
            val xbOff = xblOff + blockSizeSub * ib
            val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + blockSizeSub * ib
            for (i in 0 until blockSizeSub) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            var max = kotlin.math.abs(x[xbOff])
            for (i in 1 until blockSizeSub) max = maxOf(max, kotlin.math.abs(x[xbOff + i]))
            if (max < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS_IQ1_S) {
                scales[ib] = 0f; shifts[ib] = 1; L.fill(1, 0, blockSizeSub); continue
            }
            // sort pairs by value
            for (j in 0 until blockSizeSub) {
                pairs[2 * j] = x[xbOff + j]
                idx[2 * j] = j
            }
            // sort by pairs[2*j] ascending (iq1_sort_helper)
            val indices = (0 until blockSizeSub).sortedBy { pairs[2 * it] }.toIntArray()
            val sortedPairs = FloatArray(2 * blockSizeSub)
            val sortedIdx = IntArray(2 * blockSizeSub)
            for (j in 0 until blockSizeSub) {
                sortedPairs[2 * j] = pairs[2 * indices[j]]
                sortedIdx[2 * j] = idx[2 * indices[j]]
            }
            for (j in 0 until blockSizeSub) { pairs[2 * j] = sortedPairs[2 * j]; idx[2 * j] = sortedIdx[2 * j] }

            sumx[0] = 0f; sumw[0] = 0f
            for (j in 0 until blockSizeSub) {
                val i = idx[2 * j]
                sumx[j + 1] = sumx[j] + weight[i] * x[xbOff + i]
                sumw[j + 1] = sumw[j] + weight[i]
            }
            var bestScore = -Float.MAX_VALUE; var scale = max
            var besti1 = -1; var besti2 = -1; var bestShift = 0
            for (i1 in 0..blockSizeSub) {
                for (i2 in i1..blockSizeSub) {
                    var sqx = (sumx[i1] - sumx[0]) * xP[0] + (sumx[i2] - sumx[i1]) * xP[1] + (sumx[blockSizeSub] - sumx[i2]) * xP[2]
                    var sq2 = (sumw[i1] - sumw[0]) * xP[0] * xP[0] + (sumw[i2] - sumw[i1]) * xP[1] * xP[1] + (sumw[blockSizeSub] - sumw[i2]) * xP[2] * xP[2]
                    if (sq2 > 0 && sqx * sqx > bestScore * sq2) {
                        scale = sqx / sq2; bestScore = scale * sqx
                        besti1 = i1; besti2 = i2; bestShift = 1
                    }
                    sqx = (sumx[i1] - sumx[0]) * xM[0] + (sumx[i2] - sumx[i1]) * xM[1] + (sumx[blockSizeSub] - sumx[i2]) * xM[2]
                    sq2 = (sumw[i1] - sumw[0]) * xM[0] * xM[0] + (sumw[i2] - sumw[i1]) * xM[1] * xM[1] + (sumw[blockSizeSub] - sumw[i2]) * xM[2] * xM[2]
                    if (sq2 > 0 && sqx * sqx > bestScore * sq2) {
                        scale = sqx / sq2; bestScore = scale * sqx
                        besti1 = i1; besti2 = i2; bestShift = -1
                    }
                }
            }
            if (besti1 < 0 || besti2 < 0 || bestShift == 0) {
                scales[ib] = 0f; shifts[ib] = 1; L.fill(1, 0, blockSizeSub); continue
            }
            for (j in 0 until besti1) L[idx[2 * j]] = 0
            for (j in besti1 until besti2) L[idx[2 * j]] = 1
            for (j in besti2 until blockSizeSub) L[idx[2 * j]] = 2
            if (scale < 0) {
                for (j in 0 until blockSizeSub) L[j] = (2 - L[j]).toByte()
                scale = -scale; bestShift = -bestShift
            }
            var allOnGrid = true
            val xx = if (bestShift == 1) xP else xM
            for (k in 0 until blockSizeSub / 8) {
                var u = 0
                for (j in 0 until 8) u = u or (L[8 * k + j].toInt() shl (2 * j))
                var gridIdx = kmapQ2xs[u]
                if (gridIdx < 0) {
                    allOnGrid = false
                    val nOff = -kmapQ2xs[u] - 1
                    gridIdx = io.github.kotlinmania.llama.ore.iq1FindBestNeighbour2(
                        kneighborsQ2xs,
                        nOff,
                        kgridQ2xs,
                        x,
                        xbOff + 8 * k,
                        weight,
                        8 * k,
                        scale,
                        xx,
                        L,
                        8 * k,
                        io.github.kotlinmania.llama.ore.NGRID_IQ1S
                    )
                    require(gridIdx >= 0)
                }
                index[k] = gridIdx
            }
            if (!allOnGrid) {
                var sqx = 0f; var sq2 = 0f
                for (k in 0 until blockSizeSub / 8) {
                    val gv = kgridQ2xs[index[k]]
                    for (j in 0 until 8) {
                        val w = weight[8 * k + j]
                        val q = xx[(io.github.kotlinmania.llama.ore.gridByte(gv, j) - 1) / 2]
                        sqx += w * q * x[xbOff + 8 * k + j]
                        sq2 += w * q * q
                    }
                }
                if (sqx > 0 && sq2 > 0) scale = sqx / sq2
            }
            var h = 0
            for (k in 0 until blockSizeSub / 8) {
                y[bOff + 2 + (blockSizeSub / 8) * ib + k] = (index[k] and 255).toByte()
                h = h or ((index[k] shr 8) shl (3 * k))
            }
            io.github.kotlinmania.llama.ore.writeShortLE(
                y,
                bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 8 + ib * 2,
                h.toShort()
            )
            require(scale >= 0)
            scales[ib] = scale
            shifts[ib] = bestShift.toByte()
            maxScale = maxOf(maxScale, scale)
        }

        if (maxScale == 0f) continue

        val d = maxScale / 15f
        io.github.kotlinmania.llama.ore.writeFp16(y, bOff, d * 1.125f)
        val id = 1f / d
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / blockSizeSub) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib] - 1))
            l = maxOf(0, minOf(7, l))
            if (shifts[ib].toInt() == -1) l = l or 8
            val qhVal = io.github.kotlinmania.llama.ore.readUShortLE(
                y,
                bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 8 + ib * 2
            )
            io.github.kotlinmania.llama.ore.writeShortLE(
                y,
                bOff + 2 + io.github.kotlinmania.llama.ore.QK_K / 8 + ib * 2,
                (qhVal or (l shl 12)).toShort()
            )
        }
    }
}

// C line 4520-4772: quantize_row_iq1_m_impl
private fun quantize_row_iq1_m_impl(
    x: FloatArray, xOff: Int, y: ByteArray, yOff: Int, n: Int, quantWeights: FloatArray?,
    scales: FloatArray, weight: FloatArray, pairs: FloatArray,
    L: ByteArray, index: IntArray, shifts: ByteArray
) {
    val gindex =
        io.github.kotlinmania.llama.ore.iq2DataIndex(io.github.kotlinmania.llama.ore.GGMLType.IQ1_M)
    val kgridQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].grid!!
    val kmapQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].map!!
    val kneighborsQ2xs = io.github.kotlinmania.llama.ore.iq2Data[gindex].neighbours!!
    require(n % io.github.kotlinmania.llama.ore.QK_K == 0)

    val nbl = n / io.github.kotlinmania.llama.ore.QK_K
    val blockSizeSub = 16 // IQ1M_BLOCK_SIZE
    val blockSizeBl = io.github.kotlinmania.llama.ore.BlockIQ1M.SIZE_BYTES

    val xP = floatArrayOf(-1f + io.github.kotlinmania.llama.ore.IQ1M_DELTA,
        io.github.kotlinmania.llama.ore.IQ1M_DELTA, 1f + io.github.kotlinmania.llama.ore.IQ1M_DELTA
    )
    val xM = floatArrayOf(-1f - io.github.kotlinmania.llama.ore.IQ1M_DELTA, -io.github.kotlinmania.llama.ore.IQ1M_DELTA, 1f - io.github.kotlinmania.llama.ore.IQ1M_DELTA)
    val masks = intArrayOf(0x00, 0x80, 0x08, 0x88)

    val idx = IntArray(2 * blockSizeSub)
    val sumqx = FloatArray(4); val sumq2 = FloatArray(4)

    for (ibl in 0 until nbl) {
        val bOff = yOff + ibl * blockSizeBl
        // block_iq1_m: qs[QK_K/8=32] + qh[QK_K/16=16] + scales[QK_K/32=8]
        val qsOff = bOff
        val qhOff = bOff + io.github.kotlinmania.llama.ore.QK_K / 8
        val scOff = bOff + io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 16
        for (i in 0 until blockSizeBl) y[bOff + i] = 0

        var maxScale = 0f
        val xblOff = xOff + io.github.kotlinmania.llama.ore.QK_K * ibl
        var sumx2 = 0f
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) sumx2 += x[xblOff + i] * x[xblOff + i]
        val sigma2 = 2f * sumx2 / io.github.kotlinmania.llama.ore.QK_K

        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / blockSizeSub) {
            val xbOff = xblOff + blockSizeSub * ib
            if (quantWeights != null) {
                val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + blockSizeSub * ib
                for (i in 0 until blockSizeSub) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            } else {
                for (i in 0 until blockSizeSub) weight[i] = x[xbOff + i] * x[xbOff + i]
            }
            var max = kotlin.math.abs(x[xbOff])
            for (i in 1 until blockSizeSub) max = maxOf(max, kotlin.math.abs(x[xbOff + i]))
            if (max < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS_IQ1_M) {
                scales[ib] = 0f; shifts[ib] = 0; L.fill(1, 0, blockSizeSub); continue
            }
            for (j in 0 until blockSizeSub) { pairs[2 * j] = x[xbOff + j]; idx[2 * j] = j }
            val sortOrder = (0 until blockSizeSub).sortedBy { pairs[2 * it] }.toIntArray()
            val sortedP = FloatArray(2 * blockSizeSub); val sortedI = IntArray(2 * blockSizeSub)
            for (j in 0 until blockSizeSub) { sortedP[2 * j] = pairs[2 * sortOrder[j]]; sortedI[2 * j] = idx[2 * sortOrder[j]] }
            for (j in 0 until blockSizeSub) { pairs[2 * j] = sortedP[2 * j]; idx[2 * j] = sortedI[2 * j] }

            var bestScore = -Float.MAX_VALUE; var scale = max
            var besti1 = -1; var besti2 = -1; var bestK = -1
            for (i1 in 0..blockSizeSub) {
                for (i2 in i1..blockSizeSub) {
                    sumqx.fill(0f); sumq2.fill(0f)
                    for (j in 0 until i1) {
                        val ii = idx[2 * j]
                        if (ii < blockSizeSub / 2) {
                            sumqx[0] += weight[ii] * xP[0] * x[xbOff + ii]; sumqx[1] += weight[ii] * xP[0] * x[xbOff + ii]
                            sumqx[2] += weight[ii] * xM[0] * x[xbOff + ii]; sumqx[3] += weight[ii] * xM[0] * x[xbOff + ii]
                            sumq2[0] += weight[ii] * xP[0] * xP[0]; sumq2[1] += weight[ii] * xP[0] * xP[0]
                            sumq2[2] += weight[ii] * xM[0] * xM[0]; sumq2[3] += weight[ii] * xM[0] * xM[0]
                        } else {
                            sumqx[0] += weight[ii] * xP[0] * x[xbOff + ii]; sumqx[2] += weight[ii] * xP[0] * x[xbOff + ii]
                            sumqx[1] += weight[ii] * xM[0] * x[xbOff + ii]; sumqx[3] += weight[ii] * xM[0] * x[xbOff + ii]
                            sumq2[0] += weight[ii] * xP[0] * xP[0]; sumq2[2] += weight[ii] * xP[0] * xP[0]
                            sumq2[1] += weight[ii] * xM[0] * xM[0]; sumq2[3] += weight[ii] * xM[0] * xM[0]
                        }
                    }
                    for (j in i1 until i2) {
                        val ii = idx[2 * j]
                        if (ii < blockSizeSub / 2) {
                            sumqx[0] += weight[ii] * xP[1] * x[xbOff + ii]; sumqx[1] += weight[ii] * xP[1] * x[xbOff + ii]
                            sumqx[2] += weight[ii] * xM[1] * x[xbOff + ii]; sumqx[3] += weight[ii] * xM[1] * x[xbOff + ii]
                            sumq2[0] += weight[ii] * xP[1] * xP[1]; sumq2[1] += weight[ii] * xP[1] * xP[1]
                            sumq2[2] += weight[ii] * xM[1] * xM[1]; sumq2[3] += weight[ii] * xM[1] * xM[1]
                        } else {
                            sumqx[0] += weight[ii] * xP[1] * x[xbOff + ii]; sumqx[2] += weight[ii] * xP[1] * x[xbOff + ii]
                            sumqx[1] += weight[ii] * xM[1] * x[xbOff + ii]; sumqx[3] += weight[ii] * xM[1] * x[xbOff + ii]
                            sumq2[0] += weight[ii] * xP[1] * xP[1]; sumq2[2] += weight[ii] * xP[1] * xP[1]
                            sumq2[1] += weight[ii] * xM[1] * xM[1]; sumq2[3] += weight[ii] * xM[1] * xM[1]
                        }
                    }
                    for (j in i2 until blockSizeSub) {
                        val ii = idx[2 * j]
                        if (ii < blockSizeSub / 2) {
                            sumqx[0] += weight[ii] * xP[2] * x[xbOff + ii]; sumqx[1] += weight[ii] * xP[2] * x[xbOff + ii]
                            sumqx[2] += weight[ii] * xM[2] * x[xbOff + ii]; sumqx[3] += weight[ii] * xM[2] * x[xbOff + ii]
                            sumq2[0] += weight[ii] * xP[2] * xP[2]; sumq2[1] += weight[ii] * xP[2] * xP[2]
                            sumq2[2] += weight[ii] * xM[2] * xM[2]; sumq2[3] += weight[ii] * xM[2] * xM[2]
                        } else {
                            sumqx[0] += weight[ii] * xP[2] * x[xbOff + ii]; sumqx[2] += weight[ii] * xP[2] * x[xbOff + ii]
                            sumqx[1] += weight[ii] * xM[2] * x[xbOff + ii]; sumqx[3] += weight[ii] * xM[2] * x[xbOff + ii]
                            sumq2[0] += weight[ii] * xP[2] * xP[2]; sumq2[2] += weight[ii] * xP[2] * xP[2]
                            sumq2[1] += weight[ii] * xM[2] * xM[2]; sumq2[3] += weight[ii] * xM[2] * xM[2]
                        }
                    }
                    for (k in 0 until 4) {
                        if (sumq2[k] > 0 && sumqx[k] * sumqx[k] > bestScore * sumq2[k]) {
                            scale = sumqx[k] / sumq2[k]; bestScore = scale * sumqx[k]
                            besti1 = i1; besti2 = i2; bestK = k
                        }
                    }
                }
            }
            if (besti1 < 0 || besti2 < 0 || bestK < 0) {
                scales[ib] = 0f; shifts[ib] = 0; L.fill(1, 0, blockSizeSub); continue
            }
            for (j in 0 until besti1) L[idx[2 * j]] = 0
            for (j in besti1 until besti2) L[idx[2 * j]] = 1
            for (j in besti2 until blockSizeSub) L[idx[2 * j]] = 2
            if (scale < 0) {
                for (j in 0 until blockSizeSub) L[j] = (2 - L[j]).toByte()
                scale = -scale
                bestK = when (bestK) { 0 -> 3; 1 -> 2; 2 -> 1; else -> 0 }
            }
            var allOnGrid = true
            for (k in 0 until blockSizeSub / 8) {
                val xx = if (k == 0) { if (bestK < 2) xP else xM } else { if (bestK % 2 == 0) xP else xM }
                var u = 0
                for (j in 0 until 8) u = u or (L[8 * k + j].toInt() shl (2 * j))
                var gridIdx = kmapQ2xs[u]
                if (gridIdx < 0) {
                    allOnGrid = false
                    val nOff = -kmapQ2xs[u] - 1
                    gridIdx = io.github.kotlinmania.llama.ore.iq1FindBestNeighbour2(
                        kneighborsQ2xs,
                        nOff,
                        kgridQ2xs,
                        x,
                        xbOff + 8 * k,
                        weight,
                        8 * k,
                        scale,
                        xx,
                        L,
                        8 * k,
                        io.github.kotlinmania.llama.ore.NGRID_IQ1S
                    )
                    require(gridIdx >= 0)
                }
                index[k] = gridIdx
            }
            if (!allOnGrid) {
                var sqxF = 0f; var sq2F = 0f
                for (k in 0 until blockSizeSub / 8) {
                    val xx = if (k == 0) { if (bestK < 2) xP else xM } else { if (bestK % 2 == 0) xP else xM }
                    val gv = kgridQ2xs[index[k]]
                    for (j in 0 until 8) {
                        val w = weight[8 * k + j]
                        val q = xx[(io.github.kotlinmania.llama.ore.gridByte(gv, j) - 1) / 2]
                        sqxF += w * q * x[xbOff + 8 * k + j]
                        sq2F += w * q * q
                    }
                }
                if (sqxF > 0 && sq2F > 0) scale = sqxF / sq2F
            }
            y[qsOff + 2 * ib + 0] = (index[0] and 255).toByte()
            y[qsOff + 2 * ib + 1] = (index[1] and 255).toByte()
            y[qhOff + ib] = ((index[0] shr 8) or ((index[1] shr 8) shl 4)).toByte()
            require(scale >= 0)
            scales[ib] = scale
            shifts[ib] = bestK.toByte()
            maxScale = maxOf(maxScale, scale)
        }

        if (maxScale == 0f) continue

        // sc is uint16_t * scales, so we read/write pairs of bytes from scOff
        val d = maxScale / 15f
        val id = 1f / d
        var sqxF = 0f; var sq2F = 0f
        for (ib in 0 until io.github.kotlinmania.llama.ore.QK_K / blockSizeSub) {
            var l = io.github.kotlinmania.llama.ore.nearest_int(0.5f * (id * scales[ib] - 1))
            l = maxOf(0, minOf(7, l))
            val scIdx = ib / 4
            val scShift = 3 * (ib % 4)
            val scVal = io.github.kotlinmania.llama.ore.readUShortLE(y, scOff + scIdx * 2)
            io.github.kotlinmania.llama.ore.writeShortLE(
                y,
                scOff + scIdx * 2,
                (scVal or (l shl scShift)).toShort()
            )
            y[qhOff + ib] = (y[qhOff + ib].toInt() or masks[shifts[ib].toInt()]).toByte()

            val xbOff = xblOff + blockSizeSub * ib
            if (quantWeights != null) {
                val qwOff = io.github.kotlinmania.llama.ore.QK_K * ibl + blockSizeSub * ib
                for (i in 0 until blockSizeSub) weight[i] = quantWeights[qwOff + i] * kotlin.math.sqrt(sigma2 + x[xbOff + i] * x[xbOff + i])
            } else {
                for (i in 0 until blockSizeSub) weight[i] = x[xbOff + i] * x[xbOff + i]
            }
            for (k in 0 until blockSizeSub / 8) {
                val xx = if (k == 0) { if (shifts[ib].toInt() < 2) xP else xM } else { if (shifts[ib].toInt() % 2 == 0) xP else xM }
                val qsV = y[qsOff + 2 * ib + k].toInt() and 0xFF
                val qhV = y[qhOff + ib].toInt() and 0xFF
                val gridIdx = qsV + (((qhV shl (8 - 4 * k)) and 0x700))
                val gv = kgridQ2xs[gridIdx]
                for (j in 0 until 8) {
                    val w = weight[8 * k + j]
                    val q = xx[(io.github.kotlinmania.llama.ore.gridByte(gv, j) - 1) / 2] * (2 * l + 1)
                    sqxF += w * q * x[xbOff + 8 * k + j]
                    sq2F += w * q * q
                }
            }
        }
        var dFinal = d
        if (sq2F > 0) dFinal = sqxF / sq2F
        val s = io.github.kotlinmania.llama.ore.IQ1MScale(
            io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(dFinal * 1.1125f)
        )
        val su16 = s.u16
        val sc0 = io.github.kotlinmania.llama.ore.readUShortLE(
            y,
            scOff + 0
        ); io.github.kotlinmania.llama.ore.writeShortLE(
            y,
            scOff + 0,
            (sc0 or ((su16 and 0x000f) shl 12)).toShort()
        )
        val sc1 = io.github.kotlinmania.llama.ore.readUShortLE(
            y,
            scOff + 2
        ); io.github.kotlinmania.llama.ore.writeShortLE(
            y,
            scOff + 2,
            (sc1 or ((su16 and 0x00f0) shl 8)).toShort()
        )
        val sc2 = io.github.kotlinmania.llama.ore.readUShortLE(
            y,
            scOff + 4
        ); io.github.kotlinmania.llama.ore.writeShortLE(
            y,
            scOff + 4,
            (sc2 or ((su16 and 0x0f00) shl 4)).toShort()
        )
        val sc3 = io.github.kotlinmania.llama.ore.readUShortLE(
            y,
            scOff + 6
        ); io.github.kotlinmania.llama.ore.writeShortLE(
            y,
            scOff + 6,
            (sc3 or ((su16 and 0xf000) shl 0)).toShort()
        )
    }
}

// C line 3525
fun quantize_iq2_xxs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val rowSize = nblock * io.github.kotlinmania.llama.ore.BlockIQ2XXS.SIZE_BYTES
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_iq2_xxs_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize
    }
    return nrows * rowSize
}

// C line 3537
fun quantize_iq2_xs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val rowSize = nblock * io.github.kotlinmania.llama.ore.BlockIQ2XS.SIZE_BYTES
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_iq2_xs_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize
    }
    return nrows * rowSize
}

// C line 5139
fun quantize_iq2_s(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val rowSize = nblock * io.github.kotlinmania.llama.ore.BlockIQ2S.SIZE_BYTES
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_iq2_s_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize
    }
    return nrows * rowSize
}

// C line 3980
fun quantize_iq3_xxs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val rowSize = nblock * io.github.kotlinmania.llama.ore.BlockIQ3XXS.SIZE_BYTES
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_iq3_xxs_impl(
            256,
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize
    }
    return nrows * rowSize
}

// C line 4500
fun quantize_iq1_s(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val IQ1S_BLOCK_SIZE = 32
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / IQ1S_BLOCK_SIZE)
    val weight = FloatArray(IQ1S_BLOCK_SIZE)
    val L = ByteArray(IQ1S_BLOCK_SIZE)
    val sumx = FloatArray(IQ1S_BLOCK_SIZE + 1)
    val sumw = FloatArray(IQ1S_BLOCK_SIZE + 1)
    val pairs = FloatArray(2 * IQ1S_BLOCK_SIZE)
    val index = IntArray(IQ1S_BLOCK_SIZE / 8)
    val shifts = ByteArray(io.github.kotlinmania.llama.ore.QK_K / IQ1S_BLOCK_SIZE)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val rowSize = nblock * io.github.kotlinmania.llama.ore.BlockIQ1S.SIZE_BYTES
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_iq1_s_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix,
            scales,
            weight,
            sumx,
            sumw,
            pairs,
            L,
            index,
            shifts
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize
    }
    return nrows * rowSize
}

// C line 4774
fun quantize_iq1_m(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val IQ1M_BLOCK_SIZE = 16
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / IQ1M_BLOCK_SIZE)
    val weight = FloatArray(IQ1M_BLOCK_SIZE)
    val L = ByteArray(IQ1M_BLOCK_SIZE)
    val pairs = FloatArray(2 * IQ1M_BLOCK_SIZE)
    val index = IntArray(IQ1M_BLOCK_SIZE / 8)
    val shifts = ByteArray(io.github.kotlinmania.llama.ore.QK_K / IQ1M_BLOCK_SIZE)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val rowSize = nblock * io.github.kotlinmania.llama.ore.BlockIQ1M.SIZE_BYTES
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_iq1_m_impl(
            src,
            srcOff,
            dst,
            dstOff,
            nPerRow.toInt(),
            imatrix,
            scales,
            weight,
            pairs,
            L,
            index,
            shifts
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize
    }
    return nrows * rowSize
}

// C line 4905
fun quantize_iq4_nl(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK4_NL == 0L)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK4_NL).toInt()
    val blockSize = 18 // sizeof(block_iq4_nl) = 2 + QK4_NL/2
    val L = UByteArray(io.github.kotlinmania.llama.ore.QK4_NL)
    val weight = FloatArray(io.github.kotlinmania.llama.ore.QK4_NL)
    val unusedH = ByteArray(2)
    val scale = FloatArray(1)
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        for (ibl in 0 until nblock) {
            val bOff = dstOff + ibl * blockSize
            val qwOff = if (imatrix != null) io.github.kotlinmania.llama.ore.QK4_NL * ibl else 0
            io.github.kotlinmania.llama.ore.quantize_row_iq4_nl_impl(
                io.github.kotlinmania.llama.ore.QK4_NL,
                32,
                src,
                srcOff + io.github.kotlinmania.llama.ore.QK4_NL * ibl,
                dst,
                bOff,           // dh
                dst,
                bOff + 2,       // qs
                unusedH,
                0,
                null,
                0,
                scale,
                weight,
                L,
                io.github.kotlinmania.llama.ore.kvalues_iq4nl,
                imatrix,
                qwOff,
                7
            )
        }
        srcOff += nPerRow.toInt()
        dstOff += nblock * blockSize
    }
    return nrows * nblock * blockSize.toLong()
}

// C line 4943
fun quantize_iq4_xs(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?, yOff: Int = 0): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    // block_iq4_xs: d(fp16, 2) + scales_h(uint16, 2) + scales_l(QK_K/64=4) + qs(QK_K/2=128) = 136
    val blockSize = 136
    val L = UByteArray(io.github.kotlinmania.llama.ore.QK_K)
    val weight = FloatArray(32)
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
    var srcOff = 0; var dstOff = yOff
    for (row in 0 until nrows) {
        for (ibl in 0 until nblock) {
            val bOff = dstOff + ibl * blockSize
            val qwOff = if (imatrix != null) io.github.kotlinmania.llama.ore.QK_K * ibl else 0
            // d at bOff+0, scales_h at bOff+2, scales_l at bOff+4, qs at bOff+8
            io.github.kotlinmania.llama.ore.quantize_row_iq4_nl_impl(
                io.github.kotlinmania.llama.ore.QK_K,
                32,
                src,
                srcOff + io.github.kotlinmania.llama.ore.QK_K * ibl,
                dst,
                bOff,           // dh (fp16)
                dst,
                bOff + 8,       // qs at offset 8
                dst,
                bOff + 2,       // scales_h (uint16 at offset 2)
                dst,
                bOff + 4,       // scales_l at offset 4
                scales,
                weight,
                L,
                io.github.kotlinmania.llama.ore.kvalues_iq4nl,
                imatrix,
                qwOff,
                7
            )
        }
        srcOff += nPerRow.toInt()
        dstOff += nblock * blockSize
    }
    return nrows * nblock * blockSize.toLong()
}

// C line 4181
fun quantize_iq3_s(src: FloatArray, dst: ByteArray, nrows: Long, nPerRow: Long, imatrix: FloatArray?): Long {
    require(nPerRow % io.github.kotlinmania.llama.ore.QK_K == 0L)
    val nblock = (nPerRow / io.github.kotlinmania.llama.ore.QK_K).toInt()
    val IQ3S_BLOCK_SIZE = 32
    val scales = FloatArray(io.github.kotlinmania.llama.ore.QK_K / IQ3S_BLOCK_SIZE)
    val weight = FloatArray(IQ3S_BLOCK_SIZE)
    val xval = FloatArray(IQ3S_BLOCK_SIZE)
    val L = ByteArray(IQ3S_BLOCK_SIZE)
    val Laux = ByteArray(IQ3S_BLOCK_SIZE)
    val waux = FloatArray(IQ3S_BLOCK_SIZE)
    val isOnGrid = BooleanArray(IQ3S_BLOCK_SIZE / 4)
    val isOnGridAux = BooleanArray(IQ3S_BLOCK_SIZE / 4)
    val blockSigns = IntArray(IQ3S_BLOCK_SIZE / 8)
    val rowSize = nblock * io.github.kotlinmania.llama.ore.BlockIQ3S.SIZE_BYTES
    var srcOff = 0; var dstOff = 0
    for (row in 0 until nrows) {
        io.github.kotlinmania.llama.ore.quantize_row_iq3_s_impl(
            IQ3S_BLOCK_SIZE, src, srcOff, dst, dstOff, nPerRow.toInt(), imatrix,
            scales, weight, xval, L, Laux, waux, isOnGrid, isOnGridAux, blockSigns
        )
        srcOff += nPerRow.toInt()
        dstOff += rowSize
    }
    return nrows * rowSize
}

// IQ init/free

// C line 2777
fun iq2xs_init_impl(type: io.github.kotlinmania.llama.ore.GGMLType) {
    val gindex = io.github.kotlinmania.llama.ore.iq2DataIndex(type)
    val gridSize = io.github.kotlinmania.llama.ore.iq2GridSize(type)
    if (io.github.kotlinmania.llama.ore.iq2Data[gindex].grid != null) return

    val kgrid2bit256 = ushortArrayOf(
        0u, 2u, 5u, 8u, 10u, 17u, 20u, 32u, 34u, 40u, 42u, 65u, 68u, 80u, 88u, 97u,
        100u, 128u, 130u, 138u, 162u, 257u, 260u, 272u, 277u, 320u, 388u, 408u, 512u, 514u, 546u, 642u,
        1025u, 1028u, 1040u, 1057u, 1060u, 1088u, 1090u, 1096u, 1120u, 1153u, 1156u, 1168u, 1188u, 1280u, 1282u, 1288u,
        1312u, 1350u, 1385u, 1408u, 1425u, 1545u, 1552u, 1600u, 1668u, 1700u, 2048u, 2053u, 2056u, 2068u, 2088u, 2113u,
        2116u, 2128u, 2130u, 2184u, 2308u, 2368u, 2562u, 2580u, 4097u, 4100u, 4112u, 4129u, 4160u, 4192u, 4228u, 4240u,
        4245u, 4352u, 4360u, 4384u, 4432u, 4442u, 4480u, 4644u, 4677u, 5120u, 5128u, 5152u, 5157u, 5193u, 5248u, 5400u,
        5474u, 5632u, 5654u, 6145u, 6148u, 6160u, 6208u, 6273u, 6400u, 6405u, 6560u, 6737u, 8192u, 8194u, 8202u, 8260u,
        8289u, 8320u, 8322u, 8489u, 8520u, 8704u, 8706u, 9217u, 9220u, 9232u, 9280u, 9302u, 9472u, 9537u, 9572u, 9872u,
        10248u, 10272u, 10388u, 10820u, 16385u, 16388u, 16400u, 16408u, 16417u, 16420u, 16448u, 16456u, 16470u, 16480u, 16513u, 16516u,
        16528u, 16640u, 16672u, 16737u, 16768u, 16773u, 16897u, 16912u, 16968u, 16982u, 17000u, 17408u, 17416u, 17440u, 17536u, 17561u,
        17682u, 17700u, 17920u, 18433u, 18436u, 18448u, 18496u, 18501u, 18688u, 18776u, 18785u, 18818u, 19013u, 19088u, 20480u, 20488u,
        20497u, 20505u, 20512u, 20608u, 20616u, 20740u, 20802u, 20900u, 21137u, 21648u, 21650u, 21770u, 22017u, 22100u, 22528u, 22545u,
        22553u, 22628u, 22848u, 23048u, 24580u, 24592u, 24640u, 24680u, 24832u, 24917u, 25112u, 25184u, 25600u, 25605u, 25872u, 25874u,
        25988u, 26690u, 32768u, 32770u, 32778u, 32833u, 32898u, 33028u, 33048u, 33088u, 33297u, 33793u, 33796u, 33808u, 33813u, 33856u,
        33888u, 34048u, 34118u, 34196u, 34313u, 34368u, 34400u, 34818u, 35076u, 35345u, 36868u, 36880u, 36900u, 36928u, 37025u, 37142u,
        37248u, 37445u, 37888u, 37922u, 37956u, 38225u, 39041u, 39200u, 40962u, 41040u, 41093u, 41225u, 41472u, 42008u, 43088u, 43268u
    )

    // NOTE: kgrid_2bit_512, kgrid_1bit_2048, kgrid_2bit_1024 are large tables.
    // For brevity we use the grid arrays from iq1s_grid (for IQ1_S/IQ1_M) and
    // construct the grid lookup for IQ2_XS from kgrid_2bit_512 inline.
    // The full tables are defined inline in the C source at lines 2801-3030.

    val kgrid: UShortArray
    val kgridFull: UShortArray?

    // For IQ1_S and IQ1_M, we use a dedicated grid from the C source (kgrid_1bit_2048).
    // For IQ2_S, we use kgrid_2bit_1024.
    // These are very large. We embed them here exactly as in the C source.
    // To keep this function manageable, we store them as file-level lazy vals.
    when {
        type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS -> { kgrid = kgrid2bit256 }
        type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS -> { kgrid =
            io.github.kotlinmania.llama.ore.kgrid2bit512
        }
        type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_S || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_M -> { kgrid =
            io.github.kotlinmania.llama.ore.kgrid1bit2048
        }
        else -> { kgrid = io.github.kotlinmania.llama.ore.kgrid2bit1024
        } // IQ2_S
    }

    val kmapSize = 43692
    val nwant = if (type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_S || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_M) 3 else if (type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_S) 1 else 2

    val theGrid = ULongArray(gridSize)
    for (k in 0 until gridSize) {
        var v = 0uL
        for (i in 0 until 8) {
            val l = ((kgrid[k].toInt() shr (2 * i)) and 0x3)
            val b = (2 * l + 1).toByte()
            v = v or ((b.toULong() and 0xFFuL) shl (8 * i))
        }
        theGrid[k] = v
    }
    io.github.kotlinmania.llama.ore.iq2Data[gindex].grid = theGrid

    val kmapQ2xs = IntArray(kmapSize) { -1 }
    io.github.kotlinmania.llama.ore.iq2Data[gindex].map = kmapQ2xs

    for (i in 0 until gridSize) {
        val gv = theGrid[i]
        var index = 0
        for (k in 0 until 8) {
            val q = ((io.github.kotlinmania.llama.ore.gridByte(gv, k).toInt() and 0xFF) - 1) / 2
            index = index or (q shl (2 * k))
        }
        kmapQ2xs[index] = i
    }

    // First pass: count neighbours
    val dist2 = IntArray(2 * gridSize)
    var numNeighbors = 0; var numNotInMap = 0
    for (i in 0 until kmapSize) {
        if (kmapQ2xs[i] >= 0) continue
        numNotInMap++
        val pos = ByteArray(8)
        for (k in 0 until 8) { val l = (i shr (2 * k)) and 0x3; pos[k] = (2 * l + 1).toByte() }
        for (j in 0 until gridSize) {
            var d2 = 0
            for (k in 0 until 8) { val diff = io.github.kotlinmania.llama.ore.gridByte(
                theGrid[j],
                k
            ) - pos[k]; d2 += diff * diff }
            dist2[2 * j + 0] = d2; dist2[2 * j + 1] = j
        }
        // sort dist2 pairs
        val sortedIndices = (0 until gridSize).sortedWith(compareBy<Int> { dist2[2 * it] }.thenBy { dist2[2 * it + 1] })
        val tmpD = IntArray(2 * gridSize)
        for (j in 0 until gridSize) { tmpD[2 * j] = dist2[2 * sortedIndices[j]]; tmpD[2 * j + 1] = dist2[2 * sortedIndices[j] + 1] }
        for (j in 0 until 2 * gridSize) dist2[j] = tmpD[j]

        var n = 0; var d2v = dist2[0]; var nhave = 1
        for (j in 0 until gridSize) {
            if (dist2[2 * j] > d2v) {
                if (nhave == nwant) break
                d2v = dist2[2 * j]; nhave++
            }
            n++
        }
        numNeighbors += n
    }

    val kneighborsQ2xs = UShortArray(numNeighbors + numNotInMap)
    io.github.kotlinmania.llama.ore.iq2Data[gindex].neighbours = kneighborsQ2xs
    var counter = 0
    for (i in 0 until kmapSize) {
        if (kmapQ2xs[i] >= 0) continue
        val pos = ByteArray(8)
        for (k in 0 until 8) { val l = (i shr (2 * k)) and 0x3; pos[k] = (2 * l + 1).toByte() }
        for (j in 0 until gridSize) {
            var d2 = 0
            for (k in 0 until 8) { val diff = io.github.kotlinmania.llama.ore.gridByte(
                theGrid[j],
                k
            ) - pos[k]; d2 += diff * diff }
            dist2[2 * j + 0] = d2; dist2[2 * j + 1] = j
        }
        val sortedIndices = (0 until gridSize).sortedWith(compareBy<Int> { dist2[2 * it] }.thenBy { dist2[2 * it + 1] })
        val tmpD = IntArray(2 * gridSize)
        for (j in 0 until gridSize) { tmpD[2 * j] = dist2[2 * sortedIndices[j]]; tmpD[2 * j + 1] = dist2[2 * sortedIndices[j] + 1] }
        for (j in 0 until 2 * gridSize) dist2[j] = tmpD[j]

        kmapQ2xs[i] = -(counter + 1)
        val startIdx = counter++
        var d2v = dist2[0]; var n = 0; var nhave = 1
        for (j in 0 until gridSize) {
            if (dist2[2 * j] > d2v) {
                if (nhave == nwant) break
                d2v = dist2[2 * j]; nhave++
            }
            kneighborsQ2xs[counter++] = dist2[2 * j + 1].toUShort()
            n++
        }
        kneighborsQ2xs[startIdx] = n.toUShort()
    }
}

// C line 3133
fun iq2xs_free_impl(type: io.github.kotlinmania.llama.ore.GGMLType) {
    require(type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS || type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_S || type == io.github.kotlinmania.llama.ore.GGMLType.IQ1_M || type == io.github.kotlinmania.llama.ore.GGMLType.IQ2_S)
    val gindex = io.github.kotlinmania.llama.ore.iq2DataIndex(type)
    if (io.github.kotlinmania.llama.ore.iq2Data[gindex].grid != null) {
        io.github.kotlinmania.llama.ore.iq2Data[gindex].grid = null
        io.github.kotlinmania.llama.ore.iq2Data[gindex].map = null
        io.github.kotlinmania.llama.ore.iq2Data[gindex].neighbours = null
    }
}

// C line 3576
fun iq3xs_init_impl(gridSize: Int) {
    val gindex = io.github.kotlinmania.llama.ore.iq3DataIndex(gridSize)
    if (io.github.kotlinmania.llama.ore.iq3Data[gindex].grid != null) return

    val kgrid = if (gridSize == 256) io.github.kotlinmania.llama.ore.kgrid3bit256 else io.github.kotlinmania.llama.ore.kgrid3bit512

    val kmapSize = 4096
    val nwant = if (gridSize == 256) 2 else 3

    val theGrid = UIntArray(gridSize)
    for (k in 0 until gridSize) {
        var v = 0u
        for (i in 0 until 4) {
            val l = (kgrid[k].toInt() shr (3 * i)) and 0x7
            val b = (2 * l + 1).toByte()
            v = v or ((b.toUInt() and 0xFFu) shl (8 * i))
        }
        theGrid[k] = v
    }
    io.github.kotlinmania.llama.ore.iq3Data[gindex].grid = theGrid

    val kmapQ3xs = IntArray(kmapSize) { -1 }
    io.github.kotlinmania.llama.ore.iq3Data[gindex].map = kmapQ3xs

    for (i in 0 until gridSize) {
        val gv = theGrid[i]
        var index = 0
        for (k in 0 until 4) {
            val q = ((io.github.kotlinmania.llama.ore.gridByte(gv, k).toInt() and 0xFF) - 1) / 2
            index = index or (q shl (3 * k))
        }
        kmapQ3xs[index] = i
    }

    val dist2 = IntArray(2 * gridSize)
    var numNeighbors = 0; var numNotInMap = 0
    for (i in 0 until kmapSize) {
        if (kmapQ3xs[i] >= 0) continue
        numNotInMap++
        val pos = ByteArray(4)
        for (k in 0 until 4) { val l = (i shr (3 * k)) and 0x7; pos[k] = (2 * l + 1).toByte() }
        for (j in 0 until gridSize) {
            var d2 = 0
            for (k in 0 until 4) { val diff = io.github.kotlinmania.llama.ore.gridByte(
                theGrid[j],
                k
            ) - pos[k]; d2 += diff * diff }
            dist2[2 * j + 0] = d2; dist2[2 * j + 1] = j
        }
        val sortedIndices = (0 until gridSize).sortedWith(compareBy<Int> { dist2[2 * it] }.thenBy { dist2[2 * it + 1] })
        val tmpD = IntArray(2 * gridSize)
        for (j in 0 until gridSize) { tmpD[2 * j] = dist2[2 * sortedIndices[j]]; tmpD[2 * j + 1] = dist2[2 * sortedIndices[j] + 1] }
        for (j in 0 until 2 * gridSize) dist2[j] = tmpD[j]

        var n = 0; var d2v = dist2[0]; var nhave = 1
        for (j in 0 until gridSize) {
            if (dist2[2 * j] > d2v) { if (nhave == nwant) break; d2v = dist2[2 * j]; nhave++ }
            n++
        }
        numNeighbors += n
    }

    val kneighborsQ3xs = UShortArray(numNeighbors + numNotInMap)
    io.github.kotlinmania.llama.ore.iq3Data[gindex].neighbours = kneighborsQ3xs
    var counter = 0
    for (i in 0 until kmapSize) {
        if (kmapQ3xs[i] >= 0) continue
        val pos = ByteArray(4)
        for (k in 0 until 4) { val l = (i shr (3 * k)) and 0x7; pos[k] = (2 * l + 1).toByte() }
        for (j in 0 until gridSize) {
            var d2 = 0
            for (k in 0 until 4) { val diff = io.github.kotlinmania.llama.ore.gridByte(
                theGrid[j],
                k
            ) - pos[k]; d2 += diff * diff }
            dist2[2 * j + 0] = d2; dist2[2 * j + 1] = j
        }
        val sortedIndices = (0 until gridSize).sortedWith(compareBy<Int> { dist2[2 * it] }.thenBy { dist2[2 * it + 1] })
        val tmpD = IntArray(2 * gridSize)
        for (j in 0 until gridSize) { tmpD[2 * j] = dist2[2 * sortedIndices[j]]; tmpD[2 * j + 1] = dist2[2 * sortedIndices[j] + 1] }
        for (j in 0 until 2 * gridSize) dist2[j] = tmpD[j]

        kmapQ3xs[i] = -(counter + 1)
        val startIdx = counter++
        var d2v = dist2[0]; var n = 0; var nhave = 1
        for (j in 0 until gridSize) {
            if (dist2[2 * j] > d2v) { if (nhave == nwant) break; d2v = dist2[2 * j]; nhave++ }
            kneighborsQ3xs[counter++] = dist2[2 * j + 1].toUShort()
            n++
        }
        kneighborsQ3xs[startIdx] = n.toUShort()
    }
}

// C line 3732
fun iq3xs_free_impl(gridSize: Int) {
    require(gridSize == 256 || gridSize == 512)
    val gindex = io.github.kotlinmania.llama.ore.iq3DataIndex(gridSize)
    if (io.github.kotlinmania.llama.ore.iq3Data[gindex].grid != null) {
        io.github.kotlinmania.llama.ore.iq3Data[gindex].grid = null
        io.github.kotlinmania.llama.ore.iq3Data[gindex].map = null
        io.github.kotlinmania.llama.ore.iq3Data[gindex].neighbours = null
    }
}
