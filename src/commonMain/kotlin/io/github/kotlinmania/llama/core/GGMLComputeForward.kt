// port-lint: source ggml/src/ggml-cpu/ops.cpp
package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setShortLe
import io.github.kotlinmania.llama.lang.fp.ggml_vec_gelu_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_gelu_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_gelu_erf_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_gelu_erf_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_gelu_quick_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_gelu_quick_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_silu_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_silu_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_leaky_relu_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_leaky_relu_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_silu_backward_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_silu_backward_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_reglu_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_reglu_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_geglu_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_geglu_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_swiglu_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_swiglu_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_geglu_erf_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_geglu_erf_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_geglu_quick_f32
import io.github.kotlinmania.llama.lang.fp.ggml_vec_geglu_quick_f16
import io.github.kotlinmania.llama.lang.fp.ggml_vec_scale_f32
import io.github.kotlinmania.llama.lang.fp.ggmlVecCvarF32
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Line-by-line transliteration of ops.cpp lines 15–2096 (DUP through CONCAT).
 *
 * Function names use snake_case to match the C++ originals and maximise
 * ast_distance parity.  Public dispatchers are `fun`; type-specific helpers
 * are `private fun`.
 */

// ============================================================================
// Constants
// ============================================================================

/** Cache-line size in floats (64 bytes / 4 bytes per float). */
private const val CACHE_LINE_SIZE_F32 = 16

// ============================================================================
// ggml_compute_forward_dup — ops.cpp lines 15–575
// ============================================================================

// ops.cpp line 17
private fun ggml_compute_forward_dup_same_cont(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

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
    require(src0.type == dst.type)

    val nb0 = io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type).toLong()

    val ith = params.ith
    val nth = params.nth

    // parallelize by blocks
    val nk = (io.github.kotlinmania.llama.ore.ggmlNelements(src0) / io.github.kotlinmania.llama.ore.ggmlBlckSize(
        src0.type
    )).toInt()
    val dr = (nk + nth - 1) / nth
    val k0 = dr * ith
    val k1 = minOf(k0 + dr, nk)

    if (k0 < k1) {
        val src0Data = src0.data as ByteArray
        val dstData = dst.data as ByteArray
        val srcOff = (k0 * nb0).toInt()
        val len = ((k1 - k0) * nb0).toInt()
        src0Data.copyInto(dstData, srcOff, srcOff, srcOff + len)
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_dup_flt<src_t, dst_t> — template specializations
// ---------------------------------------------------------------------------

// F16 → F16 (same type)
private fun ggmlComputeForwardDupFltF16F16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_same_short(
        params,
        dst,
        sizeofType = 2
    )
}

// F16 → BF16
private fun ggmlComputeForwardDupFltF16BF16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 2, dstSizeof = 2,
        toF32 = { data, off ->
            io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                data.getShortLe(off).toUShort()
            )
        },
        fromF32 = { data, off, v ->
            data.setShortLe(
                off,
                io.github.kotlinmania.llama.ore.ggmlFp32ToBf16(v).bits.toShort()
            )
        }
    )
}

// F16 → F32
private fun ggmlComputeForwardDupFltF16F32(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 2, dstSizeof = 4,
        toF32 = { data, off ->
            io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                data.getShortLe(off).toUShort()
            )
        },
        fromF32 = { data, off, v -> data.setFloatLe(off, v) }
    )
}

// BF16 → F16
private fun ggmlComputeForwardDupFltBF16F16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 2, dstSizeof = 2,
        toF32 = { data, off ->
            io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    data.getShortLe(off).toUShort()
                )
            )
        },
        fromF32 = { data, off, v ->
            data.setShortLe(
                off,
                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(v).toShort()
            )
        }
    )
}

// BF16 → BF16 (same type)
private fun ggmlComputeForwardDupFltBF16BF16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_same_short(
        params,
        dst,
        sizeofType = 2
    )
}

// BF16 → F32
private fun ggmlComputeForwardDupFltBF16F32(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 2, dstSizeof = 4,
        toF32 = { data, off ->
            io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    data.getShortLe(off).toUShort()
                )
            )
        },
        fromF32 = { data, off, v -> data.setFloatLe(off, v) }
    )
}

// F32 → F16
private fun ggmlComputeForwardDupFltF32F16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 4, dstSizeof = 2,
        toF32 = { data, off -> data.getFloatLe(off) },
        fromF32 = { data, off, v ->
            data.setShortLe(
                off,
                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(v).toShort()
            )
        }
    )
}

// F32 → BF16
private fun ggmlComputeForwardDupFltF32BF16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 4, dstSizeof = 2,
        toF32 = { data, off -> data.getFloatLe(off) },
        fromF32 = { data, off, v ->
            data.setShortLe(
                off,
                io.github.kotlinmania.llama.ore.ggmlFp32ToBf16(v).bits.toShort()
            )
        }
    )
}

// F32 → F32 (same type)
private fun ggmlComputeForwardDupFltF32F32(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_same(params, dst, sizeofType = 4)
}

// F32 → I32
private fun ggmlComputeForwardDupFltF32I32(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 4, dstSizeof = 4,
        toF32 = { data, off -> data.getFloatLe(off) },
        fromF32 = { data, off, v -> data.setIntLe(off, v.toRawBits()) }
    )
}

// I32 → F32
private fun ggmlComputeForwardDupFltI32F32(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_flt_conv_short(
        params, dst,
        srcSizeof = 4, dstSizeof = 4,
        toF32 = { data, off -> Float.fromBits(data.getIntLe(off)) },
        fromF32 = { data, off, v -> data.setFloatLe(off, v) }
    )
}

// ---------------------------------------------------------------------------
// Shared implementation for dup_flt when src_t == dst_t (short element sizes: 2 bytes)
// Handles the memcpy paths where no conversion is needed
// ---------------------------------------------------------------------------
private fun ggml_compute_forward_dup_flt_same_short(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    sizeofType: Int
) {
    val src0 = dst.src[0]!!

    require(
        io.github.kotlinmania.llama.ore.ggmlNelements(dst) == io.github.kotlinmania.llama.ore.ggmlNelements(
            src0
        )
    )
    require(!io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type) && !io.github.kotlinmania.llama.ore.ggmlIsQuantized(
        dst.type
    )
    )

    val u = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val ith = params.ith
    val nth = params.nth

    val nr = u.ne01.toInt()
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    // case: type & row size equal
    if (src0.type == dst.type &&
        u.ne00 == u.ne0 &&
        u.nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type)
            .toLong() && u.nb0 == io.github.kotlinmania.llama.ore.ggmlTypeSize(dst.type).toLong()) {
        val rs = (u.ne00 * u.nb00).toInt()
        for (i03 in 0 until u.ne03) {
            for (i02 in 0 until u.ne02) {
                for (i01 in ir0 until ir1) {
                    val dstOff = (i01 * u.nb1 + i02 * u.nb2 + i03 * u.nb3).toInt()
                    val srcOff = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    src0Data.copyInto(dstData, dstOff, srcOff, srcOff + rs)
                }
            }
        }
        return
    }

    // case: dst tensor is contiguous
    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst)) {
        if (u.nb00 == sizeofType.toLong()) {
            // same type → memcpy path
            var id = 0
            val rs = (u.ne00 * u.nb00).toInt()

            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += rs * ir0
                    for (i01 in ir0 until ir1) {
                        val srcOff = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                        src0Data.copyInto(dstData, id, srcOff, srcOff + rs)
                        id += rs
                    }
                    id += rs * (u.ne01.toInt() - ir1)
                }
            }
        } else {
            // not optimal path - element by element memcpy
            var id = 0

            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += (u.ne00 * sizeofType).toInt() * ir0
                    for (i01 in ir0 until ir1) {
                        for (i00 in 0 until u.ne00) {
                            val srcOff = (i00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                            src0Data.copyInto(dstData, id, srcOff, srcOff + sizeofType)
                            id += sizeofType
                        }
                    }
                    id += (u.ne00 * sizeofType).toInt() * (u.ne01.toInt() - ir1)
                }
            }
        }
        return
    }

    // dst not contiguous — element-by-element with index tracking
    var i10 = 0L
    var i11 = 0L
    var i12 = 0L
    var i13 = 0L

    for (i03 in 0 until u.ne03) {
        for (i02 in 0 until u.ne02) {
            i10 += u.ne00 * ir0
            while (i10 >= u.ne0) {
                i10 -= u.ne0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
            for (i01 in ir0.toLong() until ir1.toLong()) {
                for (i00 in 0 until u.ne00) {
                    val srcOff = (i00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    val dstOff = (i10 * u.nb0 + i11 * u.nb1 + i12 * u.nb2 + i13 * u.nb3).toInt()
                    src0Data.copyInto(dstData, dstOff, srcOff, srcOff + sizeofType)

                    if (++i10 == u.ne00) {
                        i10 = 0
                        if (++i11 == u.ne01) {
                            i11 = 0
                            if (++i12 == u.ne02) {
                                i12 = 0
                                if (++i13 == u.ne03) {
                                    i13 = 0
                                }
                            }
                        }
                    }
                }
            }
            i10 += u.ne00 * (u.ne01 - ir1)
            while (i10 >= u.ne0) {
                i10 -= u.ne0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
        }
    }
}

// Same-type dup for 4-byte elements (F32)
private fun ggml_compute_forward_dup_flt_same(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    sizeofType: Int
) {
    val src0 = dst.src[0]!!

    require(
        io.github.kotlinmania.llama.ore.ggmlNelements(dst) == io.github.kotlinmania.llama.ore.ggmlNelements(
            src0
        )
    )
    require(!io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type) && !io.github.kotlinmania.llama.ore.ggmlIsQuantized(
        dst.type
    )
    )

    val u = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val ith = params.ith
    val nth = params.nth

    val nr = u.ne01.toInt()
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    // case: type & row size equal
    if (src0.type == dst.type &&
        u.ne00 == u.ne0 &&
        u.nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type)
            .toLong() && u.nb0 == io.github.kotlinmania.llama.ore.ggmlTypeSize(dst.type).toLong()) {
        val rs = (u.ne00 * u.nb00).toInt()
        for (i03 in 0 until u.ne03) {
            for (i02 in 0 until u.ne02) {
                for (i01 in ir0 until ir1) {
                    val dstOff = (i01 * u.nb1 + i02 * u.nb2 + i03 * u.nb3).toInt()
                    val srcOff = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    src0Data.copyInto(dstData, dstOff, srcOff, srcOff + rs)
                }
            }
        }
        return
    }

    // case: dst tensor is contiguous
    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst)) {
        if (u.nb00 == sizeofType.toLong()) {
            // same type → memcpy path
            var id = 0
            val rs = (u.ne00 * u.nb00).toInt()

            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += rs * ir0
                    for (i01 in ir0 until ir1) {
                        val srcOff = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                        src0Data.copyInto(dstData, id, srcOff, srcOff + rs)
                        id += rs
                    }
                    id += rs * (u.ne01.toInt() - ir1)
                }
            }
        } else {
            // not optimal path - element by element memcpy
            var id = 0

            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += (u.ne00 * sizeofType).toInt() * ir0
                    for (i01 in ir0 until ir1) {
                        for (i00 in 0 until u.ne00) {
                            val srcOff = (i00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                            src0Data.copyInto(dstData, id, srcOff, srcOff + sizeofType)
                            id += sizeofType
                        }
                    }
                    id += (u.ne00 * sizeofType).toInt() * (u.ne01.toInt() - ir1)
                }
            }
        }
        return
    }

    // dst not contiguous — element-by-element with index tracking
    var i10 = 0L
    var i11 = 0L
    var i12 = 0L
    var i13 = 0L

    for (i03 in 0 until u.ne03) {
        for (i02 in 0 until u.ne02) {
            i10 += u.ne00 * ir0
            while (i10 >= u.ne0) {
                i10 -= u.ne0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
            for (i01 in ir0.toLong() until ir1.toLong()) {
                for (i00 in 0 until u.ne00) {
                    val srcOff = (i00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    val dstOff = (i10 * u.nb0 + i11 * u.nb1 + i12 * u.nb2 + i13 * u.nb3).toInt()
                    src0Data.copyInto(dstData, dstOff, srcOff, srcOff + sizeofType)

                    if (++i10 == u.ne00) {
                        i10 = 0
                        if (++i11 == u.ne01) {
                            i11 = 0
                            if (++i12 == u.ne02) {
                                i12 = 0
                                if (++i13 == u.ne03) {
                                    i13 = 0
                                }
                            }
                        }
                    }
                }
            }
            i10 += u.ne00 * (u.ne01 - ir1)
            while (i10 >= u.ne0) {
                i10 -= u.ne0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared implementation for dup_flt when src_t != dst_t (different types, conversion path)
// ---------------------------------------------------------------------------
private fun ggml_compute_forward_dup_flt_conv_short(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    srcSizeof: Int,
    dstSizeof: Int,
    toF32: (ByteArray, Int) -> Float,
    fromF32: (ByteArray, Int, Float) -> Unit
) {
    val src0 = dst.src[0]!!

    require(
        io.github.kotlinmania.llama.ore.ggmlNelements(dst) == io.github.kotlinmania.llama.ore.ggmlNelements(
            src0
        )
    )
    require(!io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type) && !io.github.kotlinmania.llama.ore.ggmlIsQuantized(
        dst.type
    )
    )

    val u = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val ith = params.ith
    val nth = params.nth

    val nr = u.ne01.toInt()
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    // case: type & row size equal — cannot happen when src != dst type, but check anyway
    if (src0.type == dst.type &&
        u.ne00 == u.ne0 &&
        u.nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type)
            .toLong() && u.nb0 == io.github.kotlinmania.llama.ore.ggmlTypeSize(dst.type).toLong()) {
        val rs = (u.ne00 * u.nb00).toInt()
        for (i03 in 0 until u.ne03) {
            for (i02 in 0 until u.ne02) {
                for (i01 in ir0 until ir1) {
                    val dOff = (i01 * u.nb1 + i02 * u.nb2 + i03 * u.nb3).toInt()
                    val sOff = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    src0Data.copyInto(dstData, dOff, sOff, sOff + rs)
                }
            }
        }
        return
    }

    // case: dst tensor is contiguous
    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst)) {
        if (u.nb00 == srcSizeof.toLong()) {
            // casting between non-quantized types
            var id = 0L

            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += u.ne00 * ir0
                    for (i01 in ir0 until ir1) {
                        val src0Off = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                        for (i00 in 0 until u.ne00) {
                            val tmp = toF32(src0Data, src0Off + i00.toInt() * srcSizeof)
                            fromF32(dstData, (id * dstSizeof).toInt(), tmp)
                            id++
                        }
                    }
                    id += u.ne00 * (u.ne01 - ir1)
                }
            }
        } else {
            var id = 0L

            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += u.ne00 * ir0
                    for (i01 in ir0 until ir1) {
                        for (i00 in 0 until u.ne00) {
                            val src0Off = (i00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                            val tmp = toF32(src0Data, src0Off)
                            fromF32(dstData, (id * dstSizeof).toInt(), tmp)
                            id++
                        }
                    }
                    id += u.ne00 * (u.ne01 - ir1)
                }
            }
        }
        return
    }

    // dst not contiguous — element-by-element with conversion and index tracking
    var i10 = 0L
    var i11 = 0L
    var i12 = 0L
    var i13 = 0L

    for (i03 in 0 until u.ne03) {
        for (i02 in 0 until u.ne02) {
            i10 += u.ne00 * ir0
            while (i10 >= u.ne0) {
                i10 -= u.ne0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
            for (i01 in ir0.toLong() until ir1.toLong()) {
                for (i00 in 0 until u.ne00) {
                    val srcOff = (i00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    val dstOff = (i10 * u.nb0 + i11 * u.nb1 + i12 * u.nb2 + i13 * u.nb3).toInt()

                    val tmp = toF32(src0Data, srcOff)
                    fromF32(dstData, dstOff, tmp)

                    if (++i10 == u.ne0) {
                        i10 = 0
                        if (++i11 == u.ne1) {
                            i11 = 0
                            if (++i12 == u.ne2) {
                                i12 = 0
                                if (++i13 == u.ne3) {
                                    i13 = 0
                                }
                            }
                        }
                    }
                }
            }
            i10 += u.ne00 * (u.ne01 - ir1)
            while (i10 >= u.ne0) {
                i10 -= u.ne0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_dup_to_q<src_t> — template specializations
// ---------------------------------------------------------------------------

// F16 → quantized
private fun ggmlComputeForwardDupToQF16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_to_q_impl(
        params,
        dst,
        srcSizeof = 2
    ) { data, off ->
        io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(data.getShortLe(off).toUShort())
    }
}

// BF16 → quantized
private fun ggmlComputeForwardDupToQBF16(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_to_q_impl(
        params,
        dst,
        srcSizeof = 2
    ) { data, off ->
        io.github.kotlinmania.llama.ore.bf16ToF32(
            io.github.kotlinmania.llama.ore.GGMLBF16(
                data.getShortLe(off).toUShort()
            )
        )
    }
}

// F32 → quantized
private fun ggmlComputeForwardDupToQF32(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_to_q_impl(
        params,
        dst,
        srcSizeof = 4
    ) { data, off ->
        data.getFloatLe(off)
    }
}

// Shared implementation for dup_to_q
private fun ggml_compute_forward_dup_to_q_impl(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    srcSizeof: Int,
    toF32: (ByteArray, Int) -> Float
) {
    val src0 = dst.src[0]!!

    require(
        io.github.kotlinmania.llama.ore.ggmlNelements(dst) == io.github.kotlinmania.llama.ore.ggmlNelements(
            src0
        )
    )
    require(!io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type))

    val u = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val ith = params.ith
    val nth = params.nth

    val nr = u.ne01.toInt()
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst) &&
        u.nb00 == srcSizeof.toLong() &&
        io.github.kotlinmania.llama.ore.hasQuantizeRow(dst.type)
    ) {
        val quantizeRowQ: io.github.kotlinmania.llama.ore.GGMLFromFloatFn = io.github.kotlinmania.llama.ore.ggmlGetTypeTraitsCpu(
            dst.type
        ).fromFloat
            ?: error("no fromFloat for ${dst.type}")
        val src0F32 = FloatArray(u.ne00.toInt() + io.github.kotlinmania.llama.ore.CACHE_LINE_SIZE_F32)

        var id = 0L
        val rs = u.nb0 * (u.ne00 / io.github.kotlinmania.llama.ore.ggmlBlckSize(dst.type))
        val dstData = dst.data as ByteArray
        val src0Data = src0.data as ByteArray

        for (i03 in 0 until u.ne03) {
            for (i02 in 0 until u.ne02) {
                id += rs * ir0
                for (i01 in ir0 until ir1) {
                    val src0Off = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()

                    for (i00 in 0 until u.ne00.toInt()) {
                        src0F32[i00] = toF32(src0Data, src0Off + i00 * srcSizeof)
                    }

                    // Extract the destination slice
                    val dstSlice = ByteArray(rs.toInt())
                    quantizeRowQ(src0F32, dstSlice, u.ne00)
                    dstSlice.copyInto(dstData, id.toInt())
                    id += rs
                }
                id += rs * (u.ne01 - ir1)
            }
        }
    } else {
        error("not implemented")
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_dup_bytes — ops.cpp line 369
// ---------------------------------------------------------------------------
private fun ggml_compute_forward_dup_bytes(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(
        io.github.kotlinmania.llama.ore.ggmlNelements(dst) == io.github.kotlinmania.llama.ore.ggmlNelements(
            src0
        )
    )
    require(src0.type == dst.type)

    val u = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(src0) && io.github.kotlinmania.llama.ore.ggmlIsContiguous(
            dst
        )
    ) {
        io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_same_cont(params, dst)
        return
    }

    val typeSize = io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type).toInt()

    val ith = params.ith
    val nth = params.nth

    val nr = u.ne01.toInt()
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    if (src0.type == dst.type &&
        io.github.kotlinmania.llama.ore.ggmlAreSameShape(src0, dst) &&
        u.nb00 == typeSize.toLong() && u.nb0 == typeSize.toLong()) {
        // copy by rows
        val rs = io.github.kotlinmania.llama.ore.ggmlRowSize(src0.type, u.ne00).toInt()
        for (i03 in 0 until u.ne03) {
            for (i02 in 0 until u.ne02) {
                for (i01 in ir0 until ir1) {
                    val dstOff = (i01 * u.nb1 + i02 * u.nb2 + i03 * u.nb3).toInt()
                    val srcOff = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    src0Data.copyInto(dstData, dstOff, srcOff, srcOff + rs)
                }
            }
        }
        return
    }

    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst)) {
        var id = 0
        val rs = (u.ne00 * typeSize).toInt()

        if (u.nb00 == typeSize.toLong()) {
            // src0 is contiguous on first dimension, copy by rows
            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += rs * ir0
                    for (i01 in ir0 until ir1) {
                        val srcOff = (i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                        src0Data.copyInto(dstData, id, srcOff, srcOff + rs)
                        id += rs
                    }
                    id += rs * (u.ne01.toInt() - ir1)
                }
            }
        } else {
            for (i03 in 0 until u.ne03) {
                for (i02 in 0 until u.ne02) {
                    id += rs * ir0
                    for (i01 in ir0 until ir1) {
                        for (i00 in 0 until u.ne00) {
                            val srcOff = (i00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                            src0Data.copyInto(dstData, id, srcOff, srcOff + typeSize)
                            id += typeSize
                        }
                    }
                    id += rs * (u.ne01.toInt() - ir1)
                }
            }
        }

        return
    }

    // dst not contiguous
    var k10 = 0L
    var i11 = 0L
    var i12 = 0L
    var i13 = 0L

    val nk00 = u.ne00 / io.github.kotlinmania.llama.ore.ggmlBlckSize(src0.type)
    val nk0 = u.ne0 / io.github.kotlinmania.llama.ore.ggmlBlckSize(dst.type)

    for (i03 in 0 until u.ne03) {
        for (i02 in 0 until u.ne02) {
            k10 += nk00 * ir0
            while (k10 >= nk0) {
                k10 -= nk0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
            for (i01 in ir0.toLong() until ir1.toLong()) {
                for (k00 in 0 until nk00) {
                    val srcOff = (k00 * u.nb00 + i01 * u.nb01 + i02 * u.nb02 + i03 * u.nb03).toInt()
                    val dstOff = (k10 * u.nb0 + i11 * u.nb1 + i12 * u.nb2 + i13 * u.nb3).toInt()

                    src0Data.copyInto(dstData, dstOff, srcOff, srcOff + typeSize)

                    if (++k10 == nk0) {
                        k10 = 0
                        if (++i11 == u.ne1) {
                            i11 = 0
                            if (++i12 == u.ne2) {
                                i12 = 0
                                if (++i13 == u.ne3) {
                                    i13 = 0
                                }
                            }
                        }
                    }
                }
            }
            k10 += nk00 * (u.ne01 - ir1)
            while (k10 >= nk0) {
                k10 -= nk0
                if (++i11 == u.ne1) {
                    i11 = 0
                    if (++i12 == u.ne2) {
                        i12 = 0
                        if (++i13 == u.ne3) {
                            i13 = 0
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_dup_from_q — ops.cpp line 474
// ---------------------------------------------------------------------------
private fun ggml_compute_forward_dup_from_q(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val b = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val type = src0.type
    val qk = io.github.kotlinmania.llama.ore.ggmlBlckSize(type)
    val nr = io.github.kotlinmania.llama.ore.ggmlNelements(src1) / qk

    // destination must be contiguous in the first dimension
    require(b.nb10 == io.github.kotlinmania.llama.ore.ggmlTypeSize(dst.type).toLong())
    // must either have first dimension large enough to hold a row, or fully contiguous
    require((b.ne10 % qk) == 0L || io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst))

    val ith = params.ith
    val nth = params.nth

    val dr = ((nr + nth - 1) / nth).toInt()

    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr.toInt())

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    val tmpFloat = FloatArray(qk.toInt())

    for (ir in ir0 until ir1) {
        val i = ir.toLong() * qk

        val i03 = i / (b.ne00 * b.ne01 * b.ne02)
        val i02 = (i - i03 * b.ne00 * b.ne01 * b.ne02) / (b.ne00 * b.ne01)
        val i01 = (i - i03 * b.ne00 * b.ne01 * b.ne02 - i02 * b.ne01 * b.ne00) / b.ne00
        val i00 = i - i03 * b.ne00 * b.ne01 * b.ne02 - i02 * b.ne01 * b.ne00 - i01 * b.ne00
        val xOffset = ((i00 / qk) * b.nb00 + i01 * b.nb01 + i02 * b.nb02 + i03 * b.nb03).toInt()

        val i13 = i / (b.ne10 * b.ne11 * b.ne12)
        val i12 = (i - i13 * b.ne10 * b.ne11 * b.ne12) / (b.ne10 * b.ne11)
        val i11p = (i - i13 * b.ne10 * b.ne11 * b.ne12 - i12 * b.ne10 * b.ne11) / b.ne10
        val i10 = i - i13 * b.ne10 * b.ne11 * b.ne12 - i12 * b.ne10 * b.ne11 - i11p * b.ne10
        val dstOffset = (i10 * b.nb10 + i11p * b.nb11 + i12 * b.nb12 + i13 * b.nb13).toInt()

        io.github.kotlinmania.llama.ore.dequantize_row(type, src0Data, xOffset, tmpFloat, 0, qk)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOffset, tmpFloat, 0, qk.toInt())
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_dup — main dispatcher — ops.cpp line 530
// ---------------------------------------------------------------------------
fun ggml_compute_forward_dup(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (src0.type == dst.type) {
        io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_bytes(params, dst)
        return
    }

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            when (dst.type) {
                io.github.kotlinmania.llama.ore.GGMLType.F16  -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltF16F16(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.BF16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltF16BF16(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.F32  -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltF16F32(
                    params,
                    dst
                )
                else          -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupToQF16(
                    params,
                    dst
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.BF16 -> {
            when (dst.type) {
                io.github.kotlinmania.llama.ore.GGMLType.F16  -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltBF16F16(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.BF16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltBF16BF16(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.F32  -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltBF16F32(
                    params,
                    dst
                )
                else          -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupToQBF16(
                    params,
                    dst
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            when (dst.type) {
                io.github.kotlinmania.llama.ore.GGMLType.F16  -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltF32F16(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.BF16 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltF32BF16(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.F32  -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltF32F32(
                    params,
                    dst
                )
                io.github.kotlinmania.llama.ore.GGMLType.I32  -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltF32I32(
                    params,
                    dst
                )
                else          -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupToQF32(
                    params,
                    dst
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            when (dst.type) {
                io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggmlComputeForwardDupFltI32F32(
                    params,
                    dst
                )
                else -> error("not implemented")
            }
        }
        else -> {
            if (io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type) && dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                io.github.kotlinmania.llama.ore.ggml_compute_forward_dup_from_q(params, dst)
            } else {
                error("fatal error")
            }
        }
    }
}

// ============================================================================
// Vector helper functions — scalar implementations matching C++ vec.h
// ============================================================================

/** dst[i] += src[i]  for i in 0 until n */
private fun ggml_vec_acc_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] += src[srcOff + i]
    }
}

/** dst[i] += v  for i in 0 until n */
private fun ggml_vec_acc1_f32(n: Long, dst: FloatArray, dstOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] += v
    }
}

/** dst[i] = src0[i] + src1[i]  for i in 0 until n */
private fun ggml_vec_add_f32(n: Long, dst: FloatArray, dstOff: Int, src0: FloatArray, src0Off: Int, src1: FloatArray, src1Off: Int) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = src0[src0Off + i] + src1[src1Off + i]
    }
}

/** dst[i] = src[i] + v  for i in 0 until n */
private fun ggml_vec_add1_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = src[srcOff + i] + v
    }
}

/** dst[i] = src[i]  for i in 0 until n */
private fun ggml_vec_cpy_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int) {
    src.copyInto(dst, dstOff, srcOff, srcOff + n.toInt())
}

/** dst[i] = v  for i in 0 until n */
private fun ggml_vec_set_f32(n: Long, dst: FloatArray, dstOff: Int, v: Float) {
    for (i in 0 until n.toInt()) {
        dst[dstOff + i] = v
    }
}

/** *result = sum(src[i])  for i in 0 until n  (double accumulator for ggf variant) */
private fun ggml_vec_sum_f32_ggf(n: Long, src: FloatArray, srcOff: Int): Double {
    var sum = 0.0
    for (i in 0 until n.toInt()) {
        sum += src[srcOff + i].toDouble()
    }
    return sum
}

/** *result = sum(src[i])  for i in 0 until n  (float accumulator) */
private fun ggml_vec_sum_f32(n: Long, src: FloatArray, srcOff: Int): Float {
    var sum = 0f
    for (i in 0 until n.toInt()) {
        sum += src[srcOff + i]
    }
    return sum
}

/** *result = sum(fp16_to_f32(src[i]))  for i in 0 until n */
private fun ggml_vec_sum_f16_ggf(n: Long, data: ByteArray, dataOff: Int): Float {
    var sum = 0f
    for (i in 0 until n.toInt()) {
        val bits = data.getShortLe(dataOff + i * 2)
        sum += io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(bits.toUShort())
    }
    return sum
}

/** *result = sum(bf16_to_f32(src[i]))  for i in 0 until n */
private fun ggml_vec_sum_bf16_ggf(n: Long, data: ByteArray, dataOff: Int): Float {
    var sum = 0f
    for (i in 0 until n.toInt()) {
        val bits = data.getShortLe(dataOff + i * 2)
        sum += io.github.kotlinmania.llama.ore.bf16ToF32(
            io.github.kotlinmania.llama.ore.GGMLBF16(
                bits.toUShort()
            )
        )
    }
    return sum
}

/** dst[i] = prefix_sum(src[0..i])  for i in 0 until n */
private fun ggml_vec_cumsum_f32(n: Long, dst: FloatArray, dstOff: Int, src: FloatArray, srcOff: Int) {
    if (n <= 0) return
    dst[dstOff] = src[srcOff]
    for (i in 1 until n.toInt()) {
        dst[dstOff + i] = dst[dstOff + i - 1] + src[srcOff + i]
    }
}

/** *result = argmax(src[0..n)) */
private fun ggml_vec_argmax_f32(n: Long, src: FloatArray, srcOff: Int): Int {
    var maxIdx = 0
    var maxVal = src[srcOff]
    for (i in 1 until n.toInt()) {
        if (src[srcOff + i] > maxVal) {
            maxVal = src[srcOff + i]
            maxIdx = i
        }
    }
    return maxIdx
}

// ============================================================================
// ByteArray data access helpers
// ============================================================================

/** Read a float from a ByteArray at byte offset. */
private fun readFloat(data: ByteArray, off: Int): Float = data.getFloatLe(off)

/** Write a float to a ByteArray at byte offset. */
private fun writeFloat(data: ByteArray, off: Int, v: Float) = data.setFloatLe(off, v)

/** Read an int from a ByteArray at byte offset. */
private fun readInt(data: ByteArray, off: Int): Int = data.getIntLe(off)

/** Write an int to a ByteArray at byte offset. */
private fun writeInt(data: ByteArray, off: Int, v: Int) = data.setIntLe(off, v)

/** Read a long from a ByteArray at byte offset. */
private fun readLong(data: ByteArray, off: Int): Long = data.getLongLe(off)

/** Write a long to a ByteArray at byte offset. */
private fun writeLong(data: ByteArray, off: Int, v: Long) = data.setLongLe(off, v)

/** Read a short from a ByteArray at byte offset. */
private fun readShort(data: ByteArray, off: Int): Short = data.getShortLe(off)

/** Write a short to a ByteArray at byte offset. */
private fun writeShort(data: ByteArray, off: Int, v: Short) = data.setShortLe(off, v)

/** Read ne00..ne03 and nb00..nb03 from src0 plus ne10..ne13, nb10..nb13 from src1
 *  plus ne0..ne3, nb0..nb3 from dst.  Returns a data class for clarity. */
private data class BinaryOpLocals(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne10: Long, val ne11: Long, val ne12: Long, val ne13: Long,
    val nb10: Long, val nb11: Long, val nb12: Long, val nb13: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun binaryOpLocals(dst: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.BinaryOpLocals {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    return io.github.kotlinmania.llama.ore.BinaryOpLocals(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne10 = src1.ne[0], ne11 = src1.ne[1], ne12 = src1.ne[2], ne13 = src1.ne[3],
        nb10 = src1.nb[0].toLong(), nb11 = src1.nb[1].toLong(), nb12 = src1.nb[2].toLong(), nb13 = src1.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

private data class UnaryOpLocals(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun unaryOpLocals(dst: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.UnaryOpLocals {
    val src0 = dst.src[0]!!
    return io.github.kotlinmania.llama.ore.UnaryOpLocals(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

private data class TernaryOpLocals(
    val ne00: Long, val ne01: Long, val ne02: Long, val ne03: Long,
    val nb00: Long, val nb01: Long, val nb02: Long, val nb03: Long,
    val ne10: Long, val ne11: Long, val ne12: Long, val ne13: Long,
    val nb10: Long, val nb11: Long, val nb12: Long, val nb13: Long,
    val ne20: Long, val ne21: Long, val ne22: Long, val ne23: Long,
    val nb20: Long, val nb21: Long, val nb22: Long, val nb23: Long,
    val ne0: Long, val ne1: Long, val ne2: Long, val ne3: Long,
    val nb0: Long, val nb1: Long, val nb2: Long, val nb3: Long
)

private fun ternaryOpLocals(dst: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.TernaryOpLocals {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    val src2 = dst.src[2]!!
    return io.github.kotlinmania.llama.ore.TernaryOpLocals(
        ne00 = src0.ne[0], ne01 = src0.ne[1], ne02 = src0.ne[2], ne03 = src0.ne[3],
        nb00 = src0.nb[0].toLong(), nb01 = src0.nb[1].toLong(), nb02 = src0.nb[2].toLong(), nb03 = src0.nb[3].toLong(),
        ne10 = src1.ne[0], ne11 = src1.ne[1], ne12 = src1.ne[2], ne13 = src1.ne[3],
        nb10 = src1.nb[0].toLong(), nb11 = src1.nb[1].toLong(), nb12 = src1.nb[2].toLong(), nb13 = src1.nb[3].toLong(),
        ne20 = src2.ne[0], ne21 = src2.ne[1], ne22 = src2.ne[2], ne23 = src2.ne[3],
        nb20 = src2.nb[0].toLong(), nb21 = src2.nb[1].toLong(), nb22 = src2.nb[2].toLong(), nb23 = src2.nb[3].toLong(),
        ne0 = dst.ne[0], ne1 = dst.ne[1], ne2 = dst.ne[2], ne3 = dst.ne[3],
        nb0 = dst.nb[0].toLong(), nb1 = dst.nb[1].toLong(), nb2 = dst.nb[2].toLong(), nb3 = dst.nb[3].toLong()
    )
}

// ============================================================================
// Dequantize / quantize dispatch helpers
// ============================================================================

/** Dequantize a row of quantized data to floats (matches ggml_to_float_t). */
private fun dequantize_row(type: io.github.kotlinmania.llama.ore.GGMLType, src: ByteArray, srcOff: Int, dst: FloatArray, dstOff: Int, k: Long) {
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
        else -> error("dequantize_row: unsupported type $type")
    }
}

/** Quantize a row of floats to quantized data (matches ggml_from_float_t). */
private fun quantize_row(type: io.github.kotlinmania.llama.ore.GGMLType, src: FloatArray, dst: ByteArray, k: Long) {
    when (type) {
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0  -> io.github.kotlinmania.llama.ore.quantize_row_q1_0(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0  -> io.github.kotlinmania.llama.ore.quantize_row_q4_0(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_1  -> io.github.kotlinmania.llama.ore.quantize_row_q4_1(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_0  -> io.github.kotlinmania.llama.ore.quantize_row_q5_0(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_1  -> io.github.kotlinmania.llama.ore.quantize_row_q5_1(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0  -> io.github.kotlinmania.llama.ore.quantize_row_q8_0_generic(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K  -> io.github.kotlinmania.llama.ore.quantize_row_q2_K(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q3_K  -> io.github.kotlinmania.llama.ore.quantize_row_q3_K(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q4_K  -> io.github.kotlinmania.llama.ore.quantize_row_q4_K(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q5_K  -> io.github.kotlinmania.llama.ore.quantize_row_q5_K(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.Q6_K  -> io.github.kotlinmania.llama.ore.quantize_row_q6_K(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.TQ1_0 -> io.github.kotlinmania.llama.ore.quantize_row_tq1_0(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.TQ2_0 -> io.github.kotlinmania.llama.ore.quantize_row_tq2_0(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4 -> io.github.kotlinmania.llama.ore.quantize_row_mxfp4(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.NVFP4 -> io.github.kotlinmania.llama.ore.quantize_row_nvfp4(
            src,
            dst,
            k
        )
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL -> io.github.kotlinmania.llama.ore.quantize_row_iq4_nl(
            src,
            dst,
            k
        )
        // F32 "quantize" is just a memcpy — handled by caller
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            // no quantize needed; caller should memcpy
        }
        else -> error("quantize_row: unsupported type $type")
    }
}

/** Whether the given type has a from_float quantize function. */
private fun hasQuantizeRow(type: io.github.kotlinmania.llama.ore.GGMLType): Boolean = when (type) {
    io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1,
    io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K,
    io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0, io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
    io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL -> true
    else -> false
}

// ============================================================================
// Float-array views into ByteArray (read/write rows as FloatArray)
// ============================================================================

/** Read a row of [n] floats from [data] at byte offset [off]. */
private fun readFloatRow(data: ByteArray, off: Int, n: Int): FloatArray {
    val row = FloatArray(n)
    for (i in 0 until n) {
        row[i] = io.github.kotlinmania.llama.ore.readFloat(data, off + i * 4)
    }
    return row
}

/** Write a row of [n] floats into [data] at byte offset [off]. */
private fun writeFloatRow(data: ByteArray, off: Int, src: FloatArray, srcOff: Int, n: Int) {
    for (i in 0 until n) {
        io.github.kotlinmania.llama.ore.writeFloat(data, off + i * 4, src[srcOff + i])
    }
}

// ============================================================================
// ggml_compute_forward_add  — ops.cpp line 576
// ============================================================================

// ops.cpp line 578
private fun ggml_compute_forward_add_q_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(
        io.github.kotlinmania.llama.ore.ggml_are_same_shape(
            src0,
            src1
        ) && io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst)
    )

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val ith = params.ith
    val nth = params.nth

    val type = src0.type
    val dtype = dst.type

    // we don't support permuted src0 or src1
    check(l.nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(type).toLong())
    check(l.nb10 == 4L) // sizeof(float)

    // dst cannot be transposed or permuted
    check(l.nb0 <= l.nb1)
    check(l.nb1 <= l.nb2)
    check(l.nb2 <= l.nb3)

    check(io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type))
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val wdata = FloatArray((l.ne00.toInt() + io.github.kotlinmania.llama.ore.CACHE_LINE_SIZE_F32))

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        // src0 indices
        val i03 = ir / (l.ne02 * l.ne01).toInt()
        val i02 = (ir - i03 * (l.ne02 * l.ne01).toInt()) / l.ne01.toInt()
        val i01 = ir - i03 * (l.ne02 * l.ne01).toInt() - i02 * l.ne01.toInt()

        // src1 and dst are same shape as src0 => same indices
        val i13 = i03; val i12 = i02; val i11 = i01
        val i3 = i03; val i2 = i02; val i1 = i01

        val src0RowOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
        val src1RowOff = (i11 * l.nb11 + i12 * l.nb12 + i13 * l.nb13).toInt()
        val dstRowOff  = (i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()

        check(l.ne00 % 32 == 0L)

        // unquantize row from src0 to temp buffer
        io.github.kotlinmania.llama.ore.dequantize_row(type, src0Data, src0RowOff, wdata, 0, l.ne00)
        // add src1
        val src1Row =
            io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1RowOff, l.ne00.toInt())
        io.github.kotlinmania.llama.ore.ggml_vec_acc_f32(l.ne00, wdata, 0, src1Row, 0)
        // quantize row to dst
        if (io.github.kotlinmania.llama.ore.hasQuantizeRow(dtype)) {
            val tmpDst = ByteArray(io.github.kotlinmania.llama.ore.ggmlRowSize(dtype, l.ne00).toInt())
            io.github.kotlinmania.llama.ore.quantize_row(dtype, wdata, tmpDst, l.ne00)
            tmpDst.copyInto(dstData, dstRowOff, 0, tmpDst.size)
        } else {
            // memcpy(dst_row, wdata, ne0*nb0)
            io.github.kotlinmania.llama.ore.writeFloatRow(
                dstData,
                dstRowOff,
                wdata,
                0,
                l.ne0.toInt()
            )
        }
    }
}

// ops.cpp line 654
fun ggml_compute_forward_add(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32,
        io.github.kotlinmania.llama.ore.GGMLType.F16,
        io.github.kotlinmania.llama.ore.GGMLType.BF16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_add_non_quantized(params, dst)
        }
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
            io.github.kotlinmania.llama.ore.ggml_compute_forward_add_q_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_add_non_quantized — binary-ops.cpp line 140
// ============================================================================

// Transliteration of binary_op<op_add> from binary-ops.cpp.
// This handles the non-quantized cases: F32/F32, F16/F16, BF16/BF16, mixed.
fun ggml_compute_forward_add_non_quantized(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(
        io.github.kotlinmania.llama.ore.ggmlCanRepeat(
            src1,
            src0
        ) && io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst)
    )

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val (ir0, ir1) = io.github.kotlinmania.llama.ore.getThreadRange(params, src0)
    val isContiguousRows = io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src1)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (l.ne02 * l.ne01)
        val i02 = (ir - i03 * l.ne02 * l.ne01) / l.ne01
        val i01 = ir - i03 * l.ne02 * l.ne01 - i02 * l.ne01

        val i13 = i03 % l.ne13
        val i12 = i02 % l.ne12
        val i11 = i01 % l.ne11

        val dstOff  = (i03 * l.nb3 + i02 * l.nb2 + i01 * l.nb1).toInt()
        val src0Off = (i03 * l.nb03 + i02 * l.nb02 + i01 * l.nb01).toInt()
        val src1Off = (i13 * l.nb13 + i12 * l.nb12 + i11 * l.nb11).toInt()

        if (isContiguousRows) {
            val nr0 = l.ne00 / l.ne10
            for (r in 0 until nr0) {
                val rOff = (r * l.ne10).toInt()
                // Apply add element-wise based on types
                io.github.kotlinmania.llama.ore.addRowTyped(
                    src0.type, src1.type, dst.type,
                    dstData, dstOff + rOff * dst.type.sizeBytes,
                    src0Data, src0Off + rOff * src0.type.sizeBytes,
                    src1Data, src1Off,
                    l.ne10
                )
            }
        } else {
            io.github.kotlinmania.llama.ore.addRowNonContiguous(
                src0.type, src1.type, dst.type,
                dstData, dstOff,
                src0Data, src0Off,
                src1Data, src1Off,
                l.ne0, l.ne10, l.nb10
            )
        }
    }
}

/** Element-wise add for contiguous rows with type dispatch. */
private fun addRowTyped(
    src0Type: io.github.kotlinmania.llama.ore.GGMLType, src1Type: io.github.kotlinmania.llama.ore.GGMLType, dstType: io.github.kotlinmania.llama.ore.GGMLType,
    dstData: ByteArray, dstOff: Int,
    src0Data: ByteArray, src0Off: Int,
    src1Data: ByteArray, src1Off: Int,
    n: Long
) {
    val ni = n.toInt()
    // All F32
    if (src0Type == io.github.kotlinmania.llama.ore.GGMLType.F32 && src1Type == io.github.kotlinmania.llama.ore.GGMLType.F32 && dstType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        for (i in 0 until ni) {
            val a = io.github.kotlinmania.llama.ore.readFloat(src0Data, src0Off + i * 4)
            val b = io.github.kotlinmania.llama.ore.readFloat(src1Data, src1Off + i * 4)
            io.github.kotlinmania.llama.ore.writeFloat(dstData, dstOff + i * 4, a + b)
        }
        return
    }
    // All F16
    if (src0Type == io.github.kotlinmania.llama.ore.GGMLType.F16 && src1Type == io.github.kotlinmania.llama.ore.GGMLType.F16 && dstType == io.github.kotlinmania.llama.ore.GGMLType.F16) {
        for (i in 0 until ni) {
            val a = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                io.github.kotlinmania.llama.ore.readShort(
                    src0Data,
                    src0Off + i * 2
                ).toUShort()
            )
            val b = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                io.github.kotlinmania.llama.ore.readShort(
                    src1Data,
                    src1Off + i * 2
                ).toUShort()
            )
            io.github.kotlinmania.llama.ore.writeShort(
                dstData,
                dstOff + i * 2,
                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(a + b).toShort()
            )
        }
        return
    }
    // All BF16
    if (src0Type == io.github.kotlinmania.llama.ore.GGMLType.BF16 && src1Type == io.github.kotlinmania.llama.ore.GGMLType.BF16 && dstType == io.github.kotlinmania.llama.ore.GGMLType.BF16) {
        for (i in 0 until ni) {
            val a = io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    io.github.kotlinmania.llama.ore.readShort(
                        src0Data,
                        src0Off + i * 2
                    ).toUShort()
                )
            )
            val b = io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    io.github.kotlinmania.llama.ore.readShort(
                        src1Data,
                        src1Off + i * 2
                    ).toUShort()
                )
            )
            val r = io.github.kotlinmania.llama.ore.f32ToBf16(a + b)
            io.github.kotlinmania.llama.ore.writeShort(dstData, dstOff + i * 2, r.bits.toShort())
        }
        return
    }
    // BF16 src0, F32 src1, BF16 dst
    if (src0Type == io.github.kotlinmania.llama.ore.GGMLType.BF16 && src1Type == io.github.kotlinmania.llama.ore.GGMLType.F32 && dstType == io.github.kotlinmania.llama.ore.GGMLType.BF16) {
        for (i in 0 until ni) {
            val a = io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    io.github.kotlinmania.llama.ore.readShort(
                        src0Data,
                        src0Off + i * 2
                    ).toUShort()
                )
            )
            val b = io.github.kotlinmania.llama.ore.readFloat(src1Data, src1Off + i * 4)
            val r = io.github.kotlinmania.llama.ore.f32ToBf16(a + b)
            io.github.kotlinmania.llama.ore.writeShort(dstData, dstOff + i * 2, r.bits.toShort())
        }
        return
    }
    // BF16 src0, F32 src1, F32 dst
    if (src0Type == io.github.kotlinmania.llama.ore.GGMLType.BF16 && src1Type == io.github.kotlinmania.llama.ore.GGMLType.F32 && dstType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        for (i in 0 until ni) {
            val a = io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    io.github.kotlinmania.llama.ore.readShort(
                        src0Data,
                        src0Off + i * 2
                    ).toUShort()
                )
            )
            val b = io.github.kotlinmania.llama.ore.readFloat(src1Data, src1Off + i * 4)
            io.github.kotlinmania.llama.ore.writeFloat(dstData, dstOff + i * 4, a + b)
        }
        return
    }
    // F16 src0, F32 src1, F16 dst
    if (src0Type == io.github.kotlinmania.llama.ore.GGMLType.F16 && src1Type == io.github.kotlinmania.llama.ore.GGMLType.F32 && dstType == io.github.kotlinmania.llama.ore.GGMLType.F16) {
        for (i in 0 until ni) {
            val a = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                io.github.kotlinmania.llama.ore.readShort(
                    src0Data,
                    src0Off + i * 2
                ).toUShort()
            )
            val b = io.github.kotlinmania.llama.ore.readFloat(src1Data, src1Off + i * 4)
            io.github.kotlinmania.llama.ore.writeShort(
                dstData,
                dstOff + i * 2,
                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(a + b).toShort()
            )
        }
        return
    }
    // F16 src0, F32 src1, F32 dst
    if (src0Type == io.github.kotlinmania.llama.ore.GGMLType.F16 && src1Type == io.github.kotlinmania.llama.ore.GGMLType.F32 && dstType == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        for (i in 0 until ni) {
            val a = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                io.github.kotlinmania.llama.ore.readShort(
                    src0Data,
                    src0Off + i * 2
                ).toUShort()
            )
            val b = io.github.kotlinmania.llama.ore.readFloat(src1Data, src1Off + i * 4)
            io.github.kotlinmania.llama.ore.writeFloat(dstData, dstOff + i * 4, a + b)
        }
        return
    }
    error("ggml_compute_forward_add_non_quantized: unsupported types: dst=$dstType, src0=$src0Type, src1=$src1Type")
}

/** Element-wise add for non-contiguous rows. */
private fun addRowNonContiguous(
    src0Type: io.github.kotlinmania.llama.ore.GGMLType, src1Type: io.github.kotlinmania.llama.ore.GGMLType, dstType: io.github.kotlinmania.llama.ore.GGMLType,
    dstData: ByteArray, dstOff: Int,
    src0Data: ByteArray, src0Off: Int,
    src1Data: ByteArray, src1Off: Int,
    ne0: Long, ne10: Long, nb10: Long
) {
    val ni = ne0.toInt()
    for (i in 0 until ni) {
        val i10 = (i % ne10.toInt())
        val src0Val = io.github.kotlinmania.llama.ore.readTypedAsF32(
            src0Type,
            src0Data,
            src0Off + i * io.github.kotlinmania.llama.ore.typeSizeOf(src0Type)
        )
        val src1Val = io.github.kotlinmania.llama.ore.readTypedAsF32(
            src1Type,
            src1Data,
            src1Off + (i10 * nb10.toInt())
        )
        io.github.kotlinmania.llama.ore.writeF32AsTyped(
            dstType,
            dstData,
            dstOff + i * io.github.kotlinmania.llama.ore.typeSizeOf(dstType),
            src0Val + src1Val
        )
    }
}

/** Read one element as F32 from the given type. */
private fun readTypedAsF32(type: io.github.kotlinmania.llama.ore.GGMLType, data: ByteArray, off: Int): Float = when (type) {
    io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.readFloat(
        data,
        off
    )
    io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
        io.github.kotlinmania.llama.ore.readShort(data, off).toUShort()
    )
    io.github.kotlinmania.llama.ore.GGMLType.BF16 -> io.github.kotlinmania.llama.ore.bf16ToF32(
        io.github.kotlinmania.llama.ore.GGMLBF16(
            io.github.kotlinmania.llama.ore.readShort(
                data,
                off
            ).toUShort()
        )
    )
    else -> error("readTypedAsF32: unsupported type $type")
}

/** Write one F32 value as the given type. */
private fun writeF32AsTyped(type: io.github.kotlinmania.llama.ore.GGMLType, data: ByteArray, off: Int, v: Float) {
    when (type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.writeFloat(
            data,
            off,
            v
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.writeShort(
            data,
            off,
            io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(v).toShort()
        )
        io.github.kotlinmania.llama.ore.GGMLType.BF16 -> io.github.kotlinmania.llama.ore.writeShort(
            data,
            off,
            io.github.kotlinmania.llama.ore.f32ToBf16(v).bits.toShort()
        )
        else -> error("writeF32AsTyped: unsupported type $type")
    }
}

/** Element byte size for simple non-quantized types. */
private fun typeSizeOf(type: io.github.kotlinmania.llama.ore.GGMLType): Int = when (type) {
    io.github.kotlinmania.llama.ore.GGMLType.F32 -> 4
    io.github.kotlinmania.llama.ore.GGMLType.F16, io.github.kotlinmania.llama.ore.GGMLType.BF16, io.github.kotlinmania.llama.ore.GGMLType.I16 -> 2
    io.github.kotlinmania.llama.ore.GGMLType.I8 -> 1
    io.github.kotlinmania.llama.ore.GGMLType.I32 -> 4
    io.github.kotlinmania.llama.ore.GGMLType.I64 -> 8
    else -> type.sizeBytes
}

// ============================================================================
// ggml_compute_forward_add_id  — ops.cpp line 701
// ============================================================================

// ops.cpp line 703
private fun ggml_compute_forward_add_id_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!
    val src2 = dst.src[2]!!

    check(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    check(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    check(src2.type == io.github.kotlinmania.llama.ore.GGMLType.I32)

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(src1.nb[0] == 4uL)

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.ternaryOpLocals(dst)

    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb10 == 4L)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val src2Data = src2.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        // src0 indices
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        // src1 indices
        val i11 =
            io.github.kotlinmania.llama.ore.readInt(src2Data, (i1 * l.nb20 + i2 * l.nb21).toInt())

        check(i11 >= 0 && i11 < l.ne11.toInt())

        val dstRowOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0RowOff = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()
        val src1RowOff = (i11.toLong() * l.nb11).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val a = io.github.kotlinmania.llama.ore.readFloat(src0Data, src0RowOff + i * 4)
            val b = io.github.kotlinmania.llama.ore.readFloat(src1Data, src1RowOff + i * 4)
            io.github.kotlinmania.llama.ore.writeFloat(dstData, dstRowOff + i * 4, a + b)
        }
    }
}

// ops.cpp line 754
fun ggml_compute_forward_add_id(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_add_id_f32(params, dst)
        }
        else -> {
            error("unsupported type for ggml_compute_forward_add_id: ${
                io.github.kotlinmania.llama.ore.ggmlTypeName(
                    src0.type
                )
            }")
        }
    }
}

// ============================================================================
// ggml_compute_forward_add1  — ops.cpp line 772
// ============================================================================

// ops.cpp line 774
private fun ggml_compute_forward_add1_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(src1))

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb00 == 4L)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    val v = io.github.kotlinmania.llama.ore.readFloat(src1Data, 0)

    for (ir in ir0 until ir1) {
        // src0 and dst are same shape => same indices
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        val src0Row = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, src0Off, l.ne0.toInt())
        for (i in 0 until l.ne0.toInt()) {
            io.github.kotlinmania.llama.ore.writeFloat(dstData, dstOff + i * 4, src0Row[i] + v)
        }
    }
}

// ops.cpp line 824
private fun ggml_compute_forward_add1_f16_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(src1))

    // scalar to add
    val v = io.github.kotlinmania.llama.ore.readFloat(src1.data as ByteArray, 0)

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    check(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F16)

    check(l.nb0 == 2L) // sizeof(ggml_fp16_t)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                io.github.kotlinmania.llama.ore.readShort(
                    src0Data,
                    src0Off + i * 2
                ).toUShort()
            )
            io.github.kotlinmania.llama.ore.writeShort(
                dstData,
                dstOff + i * 2,
                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src0Val + v).toShort()
            )
        }
    }
}

// ops.cpp line 872
private fun ggml_compute_forward_add1_f16_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(src1))

    // scalar to add
    val v = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
        io.github.kotlinmania.llama.ore.readShort(
            src1.data as ByteArray,
            0
        ).toUShort()
    )

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
    check(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F16)

    check(l.nb0 == 2L)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(
                io.github.kotlinmania.llama.ore.readShort(
                    src0Data,
                    src0Off + i * 2
                ).toUShort()
            )
            io.github.kotlinmania.llama.ore.writeShort(
                dstData,
                dstOff + i * 2,
                io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src0Val + v).toShort()
            )
        }
    }
}

// ops.cpp line 920
private fun ggml_compute_forward_add1_q_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(src1))

    // scalar to add
    val v = io.github.kotlinmania.llama.ore.readFloat(src1.data as ByteArray, 0)

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val type = src0.type

    // we don't support permuted src0
    check(l.nb00 == io.github.kotlinmania.llama.ore.ggmlTypeSize(type).toLong())

    // dst cannot be transposed or permuted
    check(l.nb0 <= l.nb1)
    check(l.nb1 <= l.nb2)
    check(l.nb2 <= l.nb3)

    check(io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type))
    check(dst.type == src0.type)
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val wdata = FloatArray(l.ne0.toInt() + io.github.kotlinmania.llama.ore.CACHE_LINE_SIZE_F32)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val src0RowOff = (i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
        val dstRowOff  = (i1 * l.nb1  + i2 * l.nb2  + i3 * l.nb3).toInt()

        check(l.ne0 % 32 == 0L)

        // unquantize row from src0 to temp buffer
        io.github.kotlinmania.llama.ore.dequantize_row(type, src0Data, src0RowOff, wdata, 0, l.ne0)
        // add src1
        io.github.kotlinmania.llama.ore.ggml_vec_acc1_f32(l.ne0, wdata, 0, v)
        // quantize row to dst
        val tmpDst = ByteArray(io.github.kotlinmania.llama.ore.ggmlRowSize(type, l.ne0).toInt())
        io.github.kotlinmania.llama.ore.quantize_row(type, wdata, tmpDst, l.ne0)
        tmpDst.copyInto(dstData, dstRowOff, 0, tmpDst.size)
    }
}

// ops.cpp line 985
private fun ggml_compute_forward_add1_bf16_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(src1))

    // scalar to add
    val v = io.github.kotlinmania.llama.ore.readFloat(src1.data as ByteArray, 0)

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(src0.type == io.github.kotlinmania.llama.ore.GGMLType.BF16)
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    check(dst.type == io.github.kotlinmania.llama.ore.GGMLType.BF16)

    check(l.nb0 == 2L)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    io.github.kotlinmania.llama.ore.readShort(
                        src0Data,
                        src0Off + i * 2
                    ).toUShort()
                )
            )
            io.github.kotlinmania.llama.ore.writeShort(
                dstData,
                dstOff + i * 2,
                io.github.kotlinmania.llama.ore.f32ToBf16(src0Val + v).bits.toShort()
            )
        }
    }
}

// ops.cpp line 1033
private fun ggml_compute_forward_add1_bf16_bf16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(src1))

    // scalar to add
    val v = io.github.kotlinmania.llama.ore.bf16ToF32(
        io.github.kotlinmania.llama.ore.GGMLBF16(
            io.github.kotlinmania.llama.ore.readShort(
                src1.data as ByteArray,
                0
            ).toUShort()
        )
    )

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(src0.type == io.github.kotlinmania.llama.ore.GGMLType.BF16)
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.BF16)
    check(dst.type == io.github.kotlinmania.llama.ore.GGMLType.BF16)

    check(l.nb0 == 2L)
    check(l.nb00 == 2L)

    // rows per thread
    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (l.ne2 * l.ne1).toInt()
        val i2 = (ir - i3 * (l.ne2 * l.ne1).toInt()) / l.ne1.toInt()
        val i1 = ir - i3 * (l.ne2 * l.ne1).toInt() - i2 * l.ne1.toInt()

        val dstOff  = (i3 * l.nb3 + i2 * l.nb2 + i1 * l.nb1).toInt()
        val src0Off = (i3 * l.nb03 + i2 * l.nb02 + i1 * l.nb01).toInt()

        for (i in 0 until l.ne0.toInt()) {
            val src0Val = io.github.kotlinmania.llama.ore.bf16ToF32(
                io.github.kotlinmania.llama.ore.GGMLBF16(
                    io.github.kotlinmania.llama.ore.readShort(
                        src0Data,
                        src0Off + i * 2
                    ).toUShort()
                )
            )
            io.github.kotlinmania.llama.ore.writeShort(
                dstData,
                dstOff + i * 2,
                io.github.kotlinmania.llama.ore.f32ToBf16(src0Val + v).bits.toShort()
            )
        }
    }
}

// ops.cpp line 1081
fun ggml_compute_forward_add1(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_add1_f32(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            if (src1.type == io.github.kotlinmania.llama.ore.GGMLType.F16) {
                io.github.kotlinmania.llama.ore.ggml_compute_forward_add1_f16_f16(params, dst)
            } else if (src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                io.github.kotlinmania.llama.ore.ggml_compute_forward_add1_f16_f32(params, dst)
            } else {
                error("fatal error")
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.BF16 -> {
            if (src1.type == io.github.kotlinmania.llama.ore.GGMLType.BF16) {
                io.github.kotlinmania.llama.ore.ggml_compute_forward_add1_bf16_bf16(params, dst)
            } else if (src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
                io.github.kotlinmania.llama.ore.ggml_compute_forward_add1_bf16_f32(params, dst)
            } else {
                error("fatal error")
            }
        }
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
            io.github.kotlinmania.llama.ore.ggml_compute_forward_add1_q_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_acc  — ops.cpp line 1152
// ============================================================================

// ops.cpp line 1154
private fun ggml_compute_forward_acc_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    check(
        io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst) && io.github.kotlinmania.llama.ore.ggmlIsContiguous(
            src0
        )
    )

    // view src0 and dst with these strides and data offset inbytes during acc
    // nb0 is implicitly element_size because src0 and dst are contiguous
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
            // memcpy needs to be synchronized across threads to avoid race conditions.
            src0Data.copyInto(dstData, 0, 0, io.github.kotlinmania.llama.ore.ggmlNbytes(dst).toInt())
        }
        io.github.kotlinmania.llama.ore.ggmlBarrier(
            params.threadpool ?: io.github.kotlinmania.llama.ore.GGMLThreadpool()
        )
    }

    val ith = params.ith
    val nth = params.nth

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src1).toInt()
    val nc = src1.ne[0]

    // GGML_TENSOR_LOCALS for src1
    val ne10 = src1.ne[0]; val ne11 = src1.ne[1]; val ne12 = src1.ne[2]; val ne13 = src1.ne[3]
    val nb10 = src1.nb[0].toLong(); val nb11 = src1.nb[1].toLong()
    val nb12 = src1.nb[2].toLong(); val nb13 = src1.nb[3].toLong()

    // src0 and dst as viewed during acc
    val nb0 = io.github.kotlinmania.llama.ore.ggmlElementSize(src0).toLong()

    val nb00 = nb0
    val nb01 = nb1
    val nb02 = nb2
    val nb03 = nb3

    check(offset + (if (ne10 == 0L) 0L else ne10 - 1) * nb0 + (if (ne11 == 0L) 0L else ne11 - 1) * nb1 + (if (ne12 == 0L) 0L else ne12 - 1) * nb2 + (if (ne13 == 0L) 0L else ne13 - 1) * nb3 < io.github.kotlinmania.llama.ore.ggmlNbytes(
        dst
    ).toLong())
    check(offset + (if (ne10 == 0L) 0L else ne10 - 1) * nb00 + (if (ne11 == 0L) 0L else ne11 - 1) * nb01 + (if (ne12 == 0L) 0L else ne12 - 1) * nb02 + (if (ne13 == 0L) 0L else ne13 - 1) * nb03 < io.github.kotlinmania.llama.ore.ggmlNbytes(
        src0
    ).toLong())

    check(nb10 == 4L) // sizeof(float)

    // rows per thread
    val dr = (nr + nth - 1) / nth

    // row range for this thread
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    for (ir in ir0 until ir1) {
        // src0 and dst are viewed with shape of src1 and offset
        // => same indices
        val i3 = ir / (ne12 * ne11).toInt()
        val i2 = (ir - i3 * (ne12 * ne11).toInt()) / ne11.toInt()
        val i1 = ir - i3 * (ne12 * ne11).toInt() - i2 * ne11.toInt()

        val dstRowOff  = (i3 * nb3 + i2 * nb2 + i1 * nb1 + offset).toInt()
        val src0RowOff = (i3 * nb03 + i2 * nb02 + i1 * nb01 + offset).toInt()
        val src1RowOff = (i3 * nb13 + i2 * nb12 + i1 * nb11).toInt()

        for (i in 0 until nc.toInt()) {
            val a = io.github.kotlinmania.llama.ore.readFloat(src0Data, src0RowOff + i * 4)
            val b = io.github.kotlinmania.llama.ore.readFloat(src1Data, src1RowOff + i * 4)
            io.github.kotlinmania.llama.ore.writeFloat(dstData, dstRowOff + i * 4, a + b)
        }
    }
}

// ops.cpp line 1234
fun ggml_compute_forward_acc(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_acc_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_sum  — ops.cpp line 1279
// ============================================================================

// ops.cpp line 1281
private fun ggml_compute_forward_sum_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(dst))
    check(src0.nb[0] == 4uL) // sizeof(float)

    val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]; val ne03 = src0.ne[3]
    val nb01 = src0.nb[1].toLong(); val nb02 = src0.nb[2].toLong(); val nb03 = src0.nb[3].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var sum = 0.0

    for (i03 in 0 until ne03) {
        for (i02 in 0 until ne02) {
            for (i01 in 0 until ne01) {
                val rowOff = (i01 * nb01 + i02 * nb02 + i03 * nb03).toInt()
                val rowSum =
                    io.github.kotlinmania.llama.ore.ggml_vec_sum_f32_ggf(ne00, src0Data, rowOff)
                sum += rowSum
            }
        }
    }
    io.github.kotlinmania.llama.ore.writeFloat(dstData, 0, sum.toFloat())
}

/** ggml_vec_sum_f32_ggf for ByteArray (reads floats at byte offset). */
private fun ggml_vec_sum_f32_ggf(n: Long, data: ByteArray, off: Int): Double {
    var sum = 0.0
    for (i in 0 until n.toInt()) {
        sum += io.github.kotlinmania.llama.ore.readFloat(data, off + i * 4).toDouble()
    }
    return sum
}

// ops.cpp line 1313
private fun ggml_compute_forward_sum_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(dst))
    check(src0.nb[0] == 2uL) // sizeof(ggml_fp16_t)

    val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]; val ne03 = src0.ne[3]
    val nb01 = src0.nb[1].toLong(); val nb02 = src0.nb[2].toLong(); val nb03 = src0.nb[3].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var sum = 0f

    for (i03 in 0 until ne03) {
        for (i02 in 0 until ne02) {
            for (i01 in 0 until ne01) {
                val rowOff = (i01 * nb01 + i02 * nb02 + i03 * nb03).toInt()
                val rowSum =
                    io.github.kotlinmania.llama.ore.ggml_vec_sum_f16_ggf(ne00, src0Data, rowOff)
                sum += rowSum
            }
        }
    }
    io.github.kotlinmania.llama.ore.writeShort(
        dstData,
        0,
        io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(sum).toShort()
    )
}

// ops.cpp line 1346
private fun ggml_compute_forward_sum_bf16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(dst))
    check(src0.nb[0] == 2uL) // sizeof(ggml_bf16_t)

    val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]; val ne03 = src0.ne[3]
    val nb01 = src0.nb[1].toLong(); val nb02 = src0.nb[2].toLong(); val nb03 = src0.nb[3].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var sum = 0f

    for (i03 in 0 until ne03) {
        for (i02 in 0 until ne02) {
            for (i01 in 0 until ne01) {
                val rowOff = (i01 * nb01 + i02 * nb02 + i03 * nb03).toInt()
                val rowSum =
                    io.github.kotlinmania.llama.ore.ggml_vec_sum_bf16_ggf(ne00, src0Data, rowOff)
                sum += rowSum
            }
        }
    }
    val bf16Val = io.github.kotlinmania.llama.ore.f32ToBf16(sum)
    io.github.kotlinmania.llama.ore.writeShort(dstData, 0, bf16Val.bits.toShort())
}

// ops.cpp line 1379
fun ggml_compute_forward_sum(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_sum_f32(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_sum_f16(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.BF16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_sum_bf16(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_cumsum  — ops.cpp line 1405
// ============================================================================

// ops.cpp line 1407
private fun ggml_compute_forward_cumsum_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(dst.nb[0] == 4uL)

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(l.ne0 == l.ne00)
    check(l.ne1 == l.ne01)
    check(l.ne2 == l.ne02)
    check(l.ne3 == l.ne03)

    val (ir0, ir1) = io.github.kotlinmania.llama.ore.getThreadRange(params, src0)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (l.ne02 * l.ne01)
        val i02 = (ir - i03 * l.ne02 * l.ne01) / l.ne01
        val i01 = ir - i03 * l.ne02 * l.ne01 - i02 * l.ne01

        val srcRowOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
        val dstRowOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()

        val srcRow =
            io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcRowOff, l.ne00.toInt())
        val dstRow = FloatArray(l.ne00.toInt())
        io.github.kotlinmania.llama.ore.ggml_vec_cumsum_f32(l.ne00, dstRow, 0, srcRow, 0)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstRowOff, dstRow, 0, l.ne00.toInt())
    }
}

// ops.cpp line 1437
fun ggml_compute_forward_cumsum(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_cumsum_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_sum_rows  — ops.cpp line 1455
// ============================================================================

// ops.cpp line 1457
private fun ggml_compute_forward_sum_rows_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(dst.nb[0] == 4uL)

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(l.ne0 == 1L)
    check(l.ne1 == l.ne01)
    check(l.ne2 == l.ne02)
    check(l.ne3 == l.ne03)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i3 in 0 until l.ne03) {
        for (i2 in 0 until l.ne02) {
            for (i1 in 0 until l.ne01) {
                val srcRowOff = (i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
                val dstRowOff = (i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(
                    src0Data,
                    srcRowOff,
                    l.ne00.toInt()
                )
                val rowSum = io.github.kotlinmania.llama.ore.ggml_vec_sum_f32(l.ne00, srcRow, 0)
                io.github.kotlinmania.llama.ore.writeFloat(dstData, dstRowOff, rowSum)
            }
        }
    }
}

// ops.cpp line 1490
fun ggml_compute_forward_sum_rows(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_sum_rows_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_mean  — ops.cpp line 1508
// ============================================================================

// ops.cpp line 1510
private fun ggml_compute_forward_mean_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(src0.nb[0] == 4uL) // sizeof(float)

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    check(l.ne0 == 1L)
    check(l.ne1 == l.ne01)
    check(l.ne2 == l.ne02)
    check(l.ne3 == l.ne03)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            for (i01 in 0 until l.ne01) {
                val srcRowOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(
                    src0Data,
                    srcRowOff,
                    l.ne00.toInt()
                )
                val rowSum = io.github.kotlinmania.llama.ore.ggml_vec_sum_f32(l.ne00, srcRow, 0)
                io.github.kotlinmania.llama.ore.writeFloat(
                    dstData,
                    dstOff,
                    rowSum / l.ne00.toFloat()
                )
            }
        }
    }
}

// ops.cpp line 1547
fun ggml_compute_forward_mean(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_mean_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_argmax  — ops.cpp line 1565
// ============================================================================

// ops.cpp line 1567
private fun ggml_compute_forward_argmax_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(src0.nb[0] == 4uL) // sizeof(float)
    check(dst.nb[0] == 4uL)  // sizeof(float) — dst stores int32 as float-sized

    val ne00 = src0.ne[0]
    val ne01 = src0.ne[1]

    val nb01 = src0.nb[1].toLong()
    val nb0 = dst.nb[0].toLong()

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i1 in 0 until ne01) {
        val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(
            src0Data,
            (i1 * nb01).toInt(),
            ne00.toInt()
        )
        val v = io.github.kotlinmania.llama.ore.ggml_vec_argmax_f32(ne00, srcRow, 0)
        io.github.kotlinmania.llama.ore.writeInt(dstData, (i1 * nb0).toInt(), v)
    }
}

// ops.cpp line 1595
fun ggml_compute_forward_argmax(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_argmax_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_count_equal  — ops.cpp line 1613
// ============================================================================

// ops.cpp line 1615
private fun ggml_compute_forward_count_equal_i32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    check(src0.type == io.github.kotlinmania.llama.ore.GGMLType.I32)
    check(src1.type == io.github.kotlinmania.llama.ore.GGMLType.I32)
    check(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, src1))
    check(io.github.kotlinmania.llama.ore.ggmlIsScalar(dst))
    check(dst.type == io.github.kotlinmania.llama.ore.GGMLType.I64)

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0)

    val ith = params.ith
    val nth = params.nth

    // NOTE: In C++, sums is params->wdata.  We use a shared array approach.
    // For single-threaded, we just accumulate directly.
    var sumThread = 0L

    // rows per thread
    val dr = ((nr + nth - 1) / nth).toInt()

    // row range for this thread
    val ir0 = (dr * ith).toLong()
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (l.ne02 * l.ne01)
        val i02 = (ir - i03 * l.ne03) / l.ne01
        val i01 = ir - i03 * l.ne03 - i02 * l.ne02

        val data0Off = (i03 * l.nb03 + i02 * l.nb02 + i01 * l.nb01).toInt()
        val data1Off = (i03 * l.nb13 + i02 * l.nb12 + i01 * l.nb11).toInt()

        for (i00 in 0 until l.ne00) {
            val val0 =
                io.github.kotlinmania.llama.ore.readInt(src0Data, data0Off + (i00 * l.nb00).toInt())
            val val1 =
                io.github.kotlinmania.llama.ore.readInt(src1Data, data1Off + (i00 * l.nb10).toInt())

            if (val0 == val1) sumThread++
        }
    }

    // NOTE: Multi-threaded accumulation via wdata is simplified here.
    // For single-threaded (nth==1), write directly.
    // For multi-threaded, a proper barrier + accumulation would be needed.
    if (ith == 0) {
        io.github.kotlinmania.llama.ore.writeLong(dstData, 0, sumThread)
    }
}

// ops.cpp line 1675
fun ggml_compute_forward_count_equal(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_count_equal_i32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_repeat  — ops.cpp line 1693
// ============================================================================

// ops.cpp line 1695
private fun ggml_compute_forward_repeat_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(io.github.kotlinmania.llama.ore.ggmlCanRepeat(src0, dst))

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    // guaranteed to be an integer due to the check in ggml_can_repeat
    val nr0 = (l.ne0 / l.ne00).toInt()
    val nr1 = (l.ne1 / l.ne01).toInt()
    val nr2 = (l.ne2 / l.ne02).toInt()
    val nr3 = (l.ne3 / l.ne03).toInt()

    // TODO: support for transposed / permuted tensors
    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb00 == 4L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i3 in 0 until nr3) {
        for (k3 in 0 until l.ne03.toInt()) {
            for (i2 in 0 until nr2) {
                for (k2 in 0 until l.ne02.toInt()) {
                    for (i1 in 0 until nr1) {
                        for (k1 in 0 until l.ne01.toInt()) {
                            for (i0 in 0 until nr0) {
                                val dstOff = ((i3 * l.ne03 + k3) * l.nb3 + (i2 * l.ne02 + k2) * l.nb2 + (i1 * l.ne01 + k1) * l.nb1 + (i0 * l.ne00) * l.nb0).toInt()
                                val srcOff = (k3 * l.nb03 + k2 * l.nb02 + k1 * l.nb01).toInt()
                                // ggml_vec_cpy_f32
                                for (idx in 0 until l.ne00.toInt()) {
                                    io.github.kotlinmania.llama.ore.writeFloat(
                                        dstData,
                                        dstOff + idx * 4,
                                        io.github.kotlinmania.llama.ore.readFloat(
                                            src0Data,
                                            srcOff + idx * 4
                                        )
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ops.cpp line 1739
private fun ggml_compute_forward_repeat_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(io.github.kotlinmania.llama.ore.ggmlCanRepeat(src0, dst))

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    // guaranteed to be an integer due to the check in ggml_can_repeat
    val nr0 = (l.ne0 / l.ne00).toInt()
    val nr1 = (l.ne1 / l.ne01).toInt()
    val nr2 = (l.ne2 / l.ne02).toInt()
    val nr3 = (l.ne3 / l.ne03).toInt()

    // TODO: support for transposed / permuted tensors
    check(l.nb0 == 2L) // sizeof(ggml_fp16_t)
    check(l.nb00 == 2L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i3 in 0 until nr3) {
        for (k3 in 0 until l.ne03.toInt()) {
            for (i2 in 0 until nr2) {
                for (k2 in 0 until l.ne02.toInt()) {
                    for (i1 in 0 until nr1) {
                        for (k1 in 0 until l.ne01.toInt()) {
                            for (i0 in 0 until nr0) {
                                val dstOff = ((i3 * l.ne03 + k3) * l.nb3 + (i2 * l.ne02 + k2) * l.nb2 + (i1 * l.ne01 + k1) * l.nb1 + (i0 * l.ne00) * l.nb0).toInt()
                                val srcOff = (k3 * l.nb03 + k2 * l.nb02 + k1 * l.nb01).toInt()
                                // ggml_vec_cpy_f16 — copy raw bytes
                                for (idx in 0 until l.ne00.toInt()) {
                                    val v = io.github.kotlinmania.llama.ore.readShort(
                                        src0Data,
                                        srcOff + idx * 2
                                    )
                                    io.github.kotlinmania.llama.ore.writeShort(
                                        dstData,
                                        dstOff + idx * 2,
                                        v
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ops.cpp line 1786
fun ggml_compute_forward_repeat(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16, io.github.kotlinmania.llama.ore.GGMLType.BF16, io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_repeat_f16(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32, io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_repeat_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_repeat_back  — ops.cpp line 1817
// ============================================================================

// ops.cpp line 1819
private fun ggml_compute_forward_repeat_back_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    check(io.github.kotlinmania.llama.ore.ggmlCanRepeat(dst, src0))

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    // guaranteed to be an integer due to the check in ggml_can_repeat
    val nr0 = (l.ne00 / l.ne0).toInt()
    val nr1 = (l.ne01 / l.ne1).toInt()
    val nr2 = (l.ne02 / l.ne2).toInt()
    val nr3 = (l.ne03 / l.ne3).toInt()

    // TODO: support for transposed / permuted tensors
    check(l.nb0 == 4L) // sizeof(float)
    check(l.nb00 == 4L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    if (io.github.kotlinmania.llama.ore.ggmlIsContiguous(dst)) {
        val totalN = l.ne0 * l.ne1 * l.ne2 * l.ne3
        for (i in 0 until totalN.toInt()) {
            io.github.kotlinmania.llama.ore.writeFloat(dstData, i * 4, 0f)
        }
    } else {
        for (k3 in 0 until l.ne3.toInt()) {
            for (k2 in 0 until l.ne2.toInt()) {
                for (k1 in 0 until l.ne1.toInt()) {
                    val rowOff = (k1 * l.nb1 + k2 * l.nb2 + k3 * l.nb3).toInt()
                    for (i in 0 until l.ne0.toInt()) {
                        io.github.kotlinmania.llama.ore.writeFloat(dstData, rowOff + i * 4, 0f)
                    }
                }
            }
        }
    }

    // TODO: maybe this is not optimal?
    for (i3 in 0 until nr3) {
        for (k3 in 0 until l.ne3.toInt()) {
            for (i2 in 0 until nr2) {
                for (k2 in 0 until l.ne2.toInt()) {
                    for (i1 in 0 until nr1) {
                        for (k1 in 0 until l.ne1.toInt()) {
                            for (i0 in 0 until nr0) {
                                val dstOff = (k3 * l.nb3 + k2 * l.nb2 + k1 * l.nb1).toInt()
                                val srcOff = ((i3 * l.ne3 + k3) * l.nb03 + (i2 * l.ne2 + k2) * l.nb02 + (i1 * l.ne1 + k1) * l.nb01 + (i0 * l.ne0) * l.nb00).toInt()
                                // ggml_vec_acc_f32
                                for (idx in 0 until l.ne0.toInt()) {
                                    val existing = io.github.kotlinmania.llama.ore.readFloat(
                                        dstData,
                                        dstOff + idx * 4
                                    )
                                    val toAdd = io.github.kotlinmania.llama.ore.readFloat(
                                        src0Data,
                                        srcOff + idx * 4
                                    )
                                    io.github.kotlinmania.llama.ore.writeFloat(
                                        dstData,
                                        dstOff + idx * 4,
                                        existing + toAdd
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ops.cpp line 1877
fun ggml_compute_forward_repeat_back(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_repeat_back_f32(params, dst)
        }
        else -> {
            error("fatal error")
        }
    }
}

// ============================================================================
// ggml_compute_forward_concat  — ops.cpp line 1895
// ============================================================================

// ops.cpp line 1897
private fun ggml_compute_forward_concat_any(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    val len = io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type).toInt()

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val dim = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val xOff: Int
                    val xData: ByteArray
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        xData = src0Data
                        xOff = (i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
                    } else {
                        xData = src1Data
                        xOff = ((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt()
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()

                    xData.copyInto(dstData, yOff, xOff, xOff + len)
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 1940
private fun ggml_compute_forward_concat_i8(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type) == 1uL) // sizeof(int8_t)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val dim = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val x: Byte
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        x = src0Data[(i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()]
                    } else {
                        x = src1Data[((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt()]
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                    dstData[yOff] = x
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 1983
private fun ggml_compute_forward_concat_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type) == 2uL) // sizeof(ggml_fp16_t)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val dim = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val xVal: Short
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        xVal = io.github.kotlinmania.llama.ore.readShort(
                            src0Data,
                            (i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
                        )
                    } else {
                        xVal = io.github.kotlinmania.llama.ore.readShort(
                            src1Data,
                            ((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt()
                        )
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                    io.github.kotlinmania.llama.ore.writeShort(dstData, yOff, xVal)
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 2026
private fun ggml_compute_forward_concat_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]!!

    check(io.github.kotlinmania.llama.ore.ggmlTypeSize(src0.type) == 4uL) // sizeof(float)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val dim = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    check(dim in 0..3)

    val o = longArrayOf(0L, 0L, 0L, 0L)
    o[dim] = src0.ne[dim]

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    // TODO: smarter multi-threading
    for (i3 in 0 until l.ne3.toInt()) {
        var i2 = ith
        while (i2 < l.ne2.toInt()) {
            for (i1 in 0 until l.ne1.toInt()) {
                for (i0 in 0 until l.ne0.toInt()) {
                    val xVal: Float
                    if (i0 < l.ne00.toInt() && i1 < l.ne01.toInt() && i2 < l.ne02.toInt() && i3 < l.ne03.toInt()) {
                        xVal = io.github.kotlinmania.llama.ore.readFloat(
                            src0Data,
                            (i0 * l.nb00 + i1 * l.nb01 + i2 * l.nb02 + i3 * l.nb03).toInt()
                        )
                    } else {
                        xVal = io.github.kotlinmania.llama.ore.readFloat(
                            src1Data,
                            ((i0 - o[0]) * l.nb10 + (i1 - o[1]) * l.nb11 + (i2 - o[2]) * l.nb12 + (i3 - o[3]) * l.nb13).toInt()
                        )
                    }

                    val yOff = (i0 * l.nb0 + i1 * l.nb1 + i2 * l.nb2 + i3 * l.nb3).toInt()
                    io.github.kotlinmania.llama.ore.writeFloat(dstData, yOff, xVal)
                }
            }
            i2 += nth
        }
    }
}

// ops.cpp line 2069
fun ggml_compute_forward_concat(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16, io.github.kotlinmania.llama.ore.GGMLType.BF16, io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_concat_f16(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_concat_i8(params, dst)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32, io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_concat_f32(params, dst)
        }
        else -> {
            io.github.kotlinmania.llama.ore.ggml_compute_forward_concat_any(params, dst)
        }
    }
}

// ============================================================================
// ShortArray (fp16) read/write helpers
// ============================================================================

/** Read a row of [n] shorts (fp16) from [data] at byte offset [off]. */
private fun readShortRow(data: ByteArray, off: Int, n: Int): ShortArray {
    val row = ShortArray(n)
    for (i in 0 until n) {
        row[i] = io.github.kotlinmania.llama.ore.readShort(data, off + i * 2)
    }
    return row
}

/** Write a row of [n] shorts (fp16) into [data] at byte offset [off]. */
private fun writeShortRow(data: ByteArray, off: Int, src: ShortArray, srcOff: Int, n: Int) {
    for (i in 0 until n) {
        io.github.kotlinmania.llama.ore.writeShort(data, off + i * 2, src[srcOff + i])
    }
}

// ============================================================================
// ggml_compute_forward_gelu  — ops.cpp line 2098
// ============================================================================

// ops.cpp line 2100
private fun ggml_compute_forward_gelu_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, nc)
        val dstRow = FloatArray(nc)
        ggml_vec_gelu_f32(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2147
private fun ggml_compute_forward_gelu_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readShortRow(src0Data, srcOff, nc)
        val dstRow = ShortArray(nc)
        ggml_vec_gelu_f16(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeShortRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2195
fun ggml_compute_forward_gelu(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_gelu_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_gelu_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_fill  — ops.cpp line 2217
// ============================================================================

// ops.cpp line 2219
private fun ggml_compute_forward_fill_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val c = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)

    val ne0 = dst.ne[0]
    val ne1 = dst.ne[1]
    val ne2 = dst.ne[2]
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt()
    val nth = params.nth
    val dr = (nr + nth - 1) / nth
    val ir0 = (dr * params.ith).toLong()
    val ir1 = minOf(ir0 + dr, nr.toLong())

    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (ne2 * ne1)
        val i02 = (ir - i03 * ne2 * ne1) / ne1
        val i01 = ir - i03 * ne2 * ne1 - i02 * ne1

        val dstOff = (i03 * nb3 + i02 * nb2 + i01 * nb1).toInt()
        val dstRow = FloatArray(ne0.toInt()) { c }
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, dstRow, 0, ne0.toInt())
    }
}

// ops.cpp line 2238
fun ggml_compute_forward_fill(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    io.github.kotlinmania.llama.ore.ggml_compute_forward_fill_f32(params, dst)
}

// ============================================================================
// ggml_compute_forward_tri  — ops.cpp line 2242
// ============================================================================

// ops.cpp line 2244
private fun ggml_compute_forward_tri_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    val ttype = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(src0))

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()
    val nth = params.nth
    val dr = (nr + nth - 1) / nth
    val ir0 = (dr * params.ith).toLong()
    val ir1 = minOf(ir0 + dr, nr.toLong())

    val bipred: (Int, Int) -> Boolean = when (ttype) {
        io.github.kotlinmania.llama.ore.GGMLTriType.LOWER.value      -> { i, r -> i < r }
        io.github.kotlinmania.llama.ore.GGMLTriType.LOWER_DIAG.value -> { i, r -> i <= r }
        io.github.kotlinmania.llama.ore.GGMLTriType.UPPER.value      -> { i, r -> i > r }
        io.github.kotlinmania.llama.ore.GGMLTriType.UPPER_DIAG.value -> { i, r -> i >= r }
        else -> error("invalid tri type")
    }

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i03 = ir / (l.ne02 * l.ne01)
        val i02 = (ir - i03 * l.ne02 * l.ne01) / l.ne01
        val i01 = ir - i03 * l.ne02 * l.ne01 - i02 * l.ne01

        val srcOff = (i03 * l.nb03 + i02 * l.nb02 + i01 * l.nb01).toInt()
        val dstOff = (i03 * l.nb3 + i02 * l.nb2 + i01 * l.nb1).toInt()

        for (i0 in 0 until l.ne0.toInt()) {
            val srcVal = io.github.kotlinmania.llama.ore.readFloat(src0Data, srcOff + i0 * 4)
            io.github.kotlinmania.llama.ore.writeFloat(
                dstData,
                dstOff + i0 * 4,
                if (bipred(i0, i01.toInt())) srcVal else 0.0f
            )
        }
    }
}

// ops.cpp line 2279
fun ggml_compute_forward_tri(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_tri_f32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_gelu_erf  — ops.cpp line 2294
// ============================================================================

// ops.cpp line 2296
private fun ggml_compute_forward_gelu_erf_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, nc)
        val dstRow = FloatArray(nc)
        ggml_vec_gelu_erf_f32(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2343
private fun ggml_compute_forward_gelu_erf_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readShortRow(src0Data, srcOff, nc)
        val dstRow = ShortArray(nc)
        ggml_vec_gelu_erf_f16(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeShortRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2391
fun ggml_compute_forward_gelu_erf(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_gelu_erf_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_gelu_erf_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_gelu_quick  — ops.cpp line 2413
// ============================================================================

// ops.cpp line 2415
private fun ggml_compute_forward_gelu_quick_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, nc)
        val dstRow = FloatArray(nc)
        ggml_vec_gelu_quick_f32(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2462
private fun ggml_compute_forward_gelu_quick_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readShortRow(src0Data, srcOff, nc)
        val dstRow = ShortArray(nc)
        ggml_vec_gelu_quick_f16(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeShortRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2510
fun ggml_compute_forward_gelu_quick(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_gelu_quick_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_gelu_quick_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_silu  — ops.cpp line 2532
// ============================================================================

// ops.cpp line 2534
private fun ggml_compute_forward_silu_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, nc)
        val dstRow = FloatArray(nc)
        ggml_vec_silu_f32(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2581
private fun ggml_compute_forward_silu_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(src0))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val ne01 = src0.ne[1]
    val ne02 = src0.ne[2]
    val nb01 = src0.nb[1].toLong()
    val nb02 = src0.nb[2].toLong()
    val nb03 = src0.nb[3].toLong()
    val nb1 = dst.nb[1].toLong()
    val nb2 = dst.nb[2].toLong()
    val nb3 = dst.nb[3].toLong()

    val ith = params.ith
    val nth = params.nth

    val nc = src0.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (ir in ir0 until ir1) {
        val i3 = ir / (ne02 * ne01).toInt()
        val i2 = (ir - i3 * (ne02 * ne01).toInt()) / ne01.toInt()
        val i1 = ir - i3 * (ne02 * ne01).toInt() - i2 * ne01.toInt()

        val dstOff = (i3 * nb3 + i2 * nb2 + i1 * nb1).toInt()
        val srcOff = (i3 * nb03 + i2 * nb02 + i1 * nb01).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readShortRow(src0Data, srcOff, nc)
        val dstRow = ShortArray(nc)
        ggml_vec_silu_f16(nc, dstRow, srcRow)
        io.github.kotlinmania.llama.ore.writeShortRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2629
fun ggml_compute_forward_silu(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_silu_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_silu_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_leaky_relu  — ops.cpp line 2650
// ============================================================================

// ops.cpp line 2652
private fun ggml_compute_forward_leaky_relu_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val n = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()
    val nc = src0.ne[0].toInt()

    val negativeSlope = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)

    require(dst.nb[0].toLong() == 4L)  // sizeof(float)
    require(src0.nb[0].toLong() == 4L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i in 0 until n) {
        val dstOff = (i * dst.nb[1].toLong()).toInt()
        val srcOff = (i * src0.nb[1].toLong()).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, nc)
        val dstRow = FloatArray(nc)
        ggml_vec_leaky_relu_f32(nc, dstRow, srcRow, negativeSlope)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2682
private fun ggml_compute_forward_leaky_relu_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    if (params.ith != 0) return

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))

    val n = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()
    val nc = src0.ne[0].toInt()

    val negativeSlope = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)

    require(dst.nb[0].toLong() == 2L)  // sizeof(ggml_fp16_t)
    require(src0.nb[0].toLong() == 2L)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i in 0 until n) {
        val dstOff = (i * dst.nb[1].toLong()).toInt()
        val srcOff = (i * src0.nb[1].toLong()).toInt()

        val srcRow = io.github.kotlinmania.llama.ore.readShortRow(src0Data, srcOff, nc)
        val dstRow = ShortArray(nc)
        ggml_vec_leaky_relu_f16(nc, dstRow, srcRow, negativeSlope)
        io.github.kotlinmania.llama.ore.writeShortRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2712
fun ggml_compute_forward_leaky_relu(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_leaky_relu_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_leaky_relu_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_silu_back  — ops.cpp line 2734
// ============================================================================

// ops.cpp line 2736
private fun ggml_compute_forward_silu_back_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val grad = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(grad))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src1, dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src1, grad))

    val ith = params.ith
    val nth = params.nth

    val nc = src1.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src1).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val gradData = grad.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val src1Off = (i1 * src1.nb[1].toLong()).toInt()
        val gradOff = (i1 * grad.nb[1].toLong()).toInt()

        val src1Row = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1Off, nc)
        val gradRow = io.github.kotlinmania.llama.ore.readFloatRow(gradData, gradOff, nc)
        val dstRow = FloatArray(nc)
        ggml_vec_silu_backward_f32(nc, dstRow, src1Row, gradRow)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2779
private fun ggml_compute_forward_silu_back_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val grad = dst.src[0]!!
    val src1 = dst.src[1]!!

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(grad))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src1, dst))
    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src1, grad))

    val ith = params.ith
    val nth = params.nth

    val nc = src1.ne[0].toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src1).toInt()

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val gradData = grad.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val src1Off = (i1 * src1.nb[1].toLong()).toInt()
        val gradOff = (i1 * grad.nb[1].toLong()).toInt()

        val src1Row = io.github.kotlinmania.llama.ore.readShortRow(src1Data, src1Off, nc)
        val gradRow = io.github.kotlinmania.llama.ore.readShortRow(gradData, gradOff, nc)
        val dstRow = ShortArray(nc)
        ggml_vec_silu_backward_f16(nc, dstRow, src1Row, gradRow)
        io.github.kotlinmania.llama.ore.writeShortRow(dstData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2823
fun ggml_compute_forward_silu_back(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_silu_back_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_silu_back_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_reglu  — ops.cpp line 2845
// ============================================================================

// ops.cpp line 2847
private fun ggml_compute_forward_reglu_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]  // nullable
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 4
            src1Off += (if (swapped != 0) 0 else nc) * 4
        }

        val src0Row = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = FloatArray(nc)
        ggml_vec_reglu_f32(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2906
private fun ggml_compute_forward_reglu_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]  // nullable
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 2
            src1Off += (if (swapped != 0) 0 else nc) * 2
        }

        val src0Row = io.github.kotlinmania.llama.ore.readShortRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readShortRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = ShortArray(nc)
        ggml_vec_reglu_f16(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeShortRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 2966
fun ggml_compute_forward_reglu(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_reglu_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_reglu_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_geglu  — ops.cpp line 2988
// ============================================================================

// ops.cpp line 2990
private fun ggml_compute_forward_geglu_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 4
            src1Off += (if (swapped != 0) 0 else nc) * 4
        }

        val src0Row = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = FloatArray(nc)
        ggml_vec_geglu_f32(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3049
private fun ggml_compute_forward_geglu_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 2
            src1Off += (if (swapped != 0) 0 else nc) * 2
        }

        val src0Row = io.github.kotlinmania.llama.ore.readShortRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readShortRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = ShortArray(nc)
        ggml_vec_geglu_f16(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeShortRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3109
fun ggml_compute_forward_geglu(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_geglu_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_geglu_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_swiglu  — ops.cpp line 3131
// ============================================================================

// ops.cpp line 3133
private fun ggml_compute_forward_swiglu_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 4
            src1Off += (if (swapped != 0) 0 else nc) * 4
        }

        val src0Row = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = FloatArray(nc)
        ggml_vec_swiglu_f32(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3192
private fun ggml_compute_forward_swiglu_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 2
            src1Off += (if (swapped != 0) 0 else nc) * 2
        }

        val src0Row = io.github.kotlinmania.llama.ore.readShortRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readShortRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = ShortArray(nc)
        ggml_vec_swiglu_f16(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeShortRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3252
fun ggml_compute_forward_swiglu(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_swiglu_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_swiglu_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_swiglu_oai  — ops.cpp line 3274
// ============================================================================

// ops.cpp line 3276
private fun ggml_compute_forward_swiglu_oai_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)
    val alpha = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 2)
    val limit = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 3)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 4
            src1Off += (if (swapped != 0) 0 else nc) * 4
        }

        val src0Row = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1Off, nc)

        for (k in 0 until nc) {
            val x = min(src0Row[k], limit)
            val y = src1Row[k].coerceIn(-limit, limit)
            val outGlu = x / (1.0f + exp(alpha * (-x)))
            io.github.kotlinmania.llama.ore.writeFloat(
                dstByteData,
                dstOff + k * 4,
                outGlu * (y + 1.0f)
            )
        }
    }
}

// ops.cpp line 3343
fun ggml_compute_forward_swiglu_oai(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_swiglu_oai_f32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_geglu_erf  — ops.cpp line 3361
// ============================================================================

// ops.cpp line 3363
private fun ggml_compute_forward_geglu_erf_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 4
            src1Off += (if (swapped != 0) 0 else nc) * 4
        }

        val src0Row = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = FloatArray(nc)
        ggml_vec_geglu_erf_f32(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3422
private fun ggml_compute_forward_geglu_erf_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 2
            src1Off += (if (swapped != 0) 0 else nc) * 2
        }

        val src0Row = io.github.kotlinmania.llama.ore.readShortRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readShortRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = ShortArray(nc)
        ggml_vec_geglu_erf_f16(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeShortRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3482
fun ggml_compute_forward_geglu_erf(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_geglu_erf_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_geglu_erf_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_geglu_quick  — ops.cpp line 3504
// ============================================================================

// ops.cpp line 3506
private fun ggml_compute_forward_geglu_quick_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 4
            src1Off += (if (swapped != 0) 0 else nc) * 4
        }

        val src0Row = io.github.kotlinmania.llama.ore.readFloatRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = FloatArray(nc)
        ggml_vec_geglu_quick_f32(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeFloatRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3565
private fun ggml_compute_forward_geglu_quick_f16(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!
    val src1 = dst.src[1]
    val src0Data = src0.data as ByteArray
    val src1Data = (if (src1 != null) src1.data else src0.data) as ByteArray
    val src0o = src0.nb[1].toLong()
    val src1o = if (src1 != null) src1.nb[1].toLong() else src0.nb[1].toLong()

    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src0))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(dst))

    if (src1 != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(src1))
        require(src0.type == src1.type)
    }

    val ith = params.ith
    val nth = params.nth

    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0).toInt()

    require(dst.ne[0].toInt() == nc)
    require(io.github.kotlinmania.llama.ore.ggmlNrows(dst).toInt() == nr)

    val swapped = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)

    val dr = (nr + nth - 1) / nth
    val ir0 = dr * ith
    val ir1 = minOf(ir0 + dr, nr)

    val dstByteData = dst.data as ByteArray

    for (i1 in ir0 until ir1) {
        var src0Off = (i1 * src0o).toInt()
        var src1Off = (i1 * src1o).toInt()

        if (src1 == null) {
            src0Off += (if (swapped != 0) nc else 0) * 2
            src1Off += (if (swapped != 0) 0 else nc) * 2
        }

        val src0Row = io.github.kotlinmania.llama.ore.readShortRow(src0Data, src0Off, nc)
        val src1Row = io.github.kotlinmania.llama.ore.readShortRow(src1Data, src1Off, nc)
        val dstOff = (i1 * dst.nb[1].toLong()).toInt()
        val dstRow = ShortArray(nc)
        ggml_vec_geglu_quick_f16(nc, dstRow, src0Row, src1Row)
        io.github.kotlinmania.llama.ore.writeShortRow(dstByteData, dstOff, dstRow, 0, nc)
    }
}

// ops.cpp line 3625
fun ggml_compute_forward_geglu_quick(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_geglu_quick_f32(
            params,
            dst
        )
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_geglu_quick_f16(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_norm  — ops.cpp line 3647
// ============================================================================

// ops.cpp line 3649
private fun ggml_compute_forward_norm_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    require(src0.nb[0].toLong() == 4L) // sizeof(float)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val eps = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)
    require(eps >= 0.0f)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            var i01 = ith.toLong()
            while (i01 < l.ne01) {
                val srcOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val x =
                    io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, l.ne00.toInt())

                var sum = 0.0f
                for (i in 0 until l.ne00.toInt()) {
                    sum += x[i]
                }
                val mean = sum / l.ne00.toFloat()

                val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                val y = FloatArray(l.ne00.toInt())

                val variance = ggmlVecCvarF32(l.ne00.toInt(), y, x, mean).toFloat()

                val scale = 1.0f / sqrt(variance + eps)
                ggml_vec_scale_f32(l.ne00.toInt(), y, scale)
                io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, y, 0, l.ne00.toInt())

                i01 += nth
            }
        }
    }
}

// ops.cpp line 3696
fun ggml_compute_forward_norm(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_norm_f32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_rms_norm  — ops.cpp line 3714
// ============================================================================

// ops.cpp line 3716
private fun ggml_compute_forward_rms_norm_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    require(src0.nb[0].toLong() == 4L) // sizeof(float)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val eps = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)
    require(eps >= 0.0f)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            var i01 = ith.toLong()
            while (i01 < l.ne01) {
                val srcOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val x =
                    io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, l.ne00.toInt())

                var sum = 0.0
                for (i00 in 0 until l.ne00.toInt()) {
                    sum += (x[i00].toDouble() * x[i00].toDouble())
                }

                val mean = (sum / l.ne00).toFloat()

                val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                val y = FloatArray(l.ne00.toInt())
                x.copyInto(y)

                val scale = 1.0f / sqrt(mean + eps)

                require(scale > 0.0f)

                ggml_vec_scale_f32(l.ne00.toInt(), y, scale)
                io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, y, 0, l.ne00.toInt())

                i01 += nth
            }
        }
    }
}

// ops.cpp line 3767
fun ggml_compute_forward_rms_norm(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_rms_norm_f32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_rms_norm_back  — ops.cpp line 3785
// ============================================================================

// ops.cpp line 3785
private fun ggml_compute_forward_rms_norm_back_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!! // gradients from forward pass output
    val src1 = dst.src[1]!! // src1 from forward pass

    require(
        io.github.kotlinmania.llama.ore.ggml_are_same_shape(
            src0,
            dst
        ) && io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, src1)
    )
    require(src0.nb[0].toLong() == 4L) // sizeof(float)
    require(src1.nb[0].toLong() == 4L)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.binaryOpLocals(dst)

    val eps = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)

    val src0Data = src0.data as ByteArray
    val src1Data = src1.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            var i01 = ith.toLong()
            while (i01 < l.ne01) {
                val i11 = i01
                val i12 = i02
                val i13 = i03

                val dzOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val xOff = (i11 * l.nb11 + i12 * l.nb12 + i13 * l.nb13).toInt()

                val dz =
                    io.github.kotlinmania.llama.ore.readFloatRow(src0Data, dzOff, l.ne00.toInt())
                val x = io.github.kotlinmania.llama.ore.readFloatRow(src1Data, xOff, l.ne00.toInt())

                var sumXx = 0.0
                var sumXdz = 0.0

                for (i00 in 0 until l.ne00.toInt()) {
                    sumXx += (x[i00].toDouble() * x[i00].toDouble())
                    sumXdz += (x[i00].toDouble() * dz[i00].toDouble())
                }

                val meanEps = (sumXx / l.ne00).toFloat() + eps
                val sumEps = sumXx.toFloat() + eps * l.ne00.toFloat()
                val rrms = 1.0f / sqrt(meanEps)

                val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                val dx = FloatArray(l.ne00.toInt())
                x.copyInto(dx)
                ggml_vec_scale_f32(l.ne00.toInt(), dx, (-sumXdz / sumEps).toFloat())
                // dx = dx + dz
                for (i in 0 until l.ne00.toInt()) {
                    dx[i] += dz[i]
                }
                ggml_vec_scale_f32(l.ne00.toInt(), dx, rrms)
                io.github.kotlinmania.llama.ore.writeFloatRow(
                    dstData,
                    dstOff,
                    dx,
                    0,
                    l.ne00.toInt()
                )

                i01 += nth
            }
        }
    }
}

// ops.cpp line 3942
fun ggml_compute_forward_rms_norm_back(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_rms_norm_back_f32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_group_norm  — ops.cpp line 3960
// ============================================================================

// ops.cpp line 3962
private fun ggml_compute_forward_group_norm_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    require(src0.nb[0].toLong() == 4L) // sizeof(float)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val eps =
        io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 1)  // op_params + 1 in C++

    val nChannels = src0.ne[2].toInt()
    val nGroups = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    val nChannelsPerGroup = (nChannels + nGroups - 1) / nGroups

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    var i = ith
    while (i < nGroups) {
        val start = i * nChannelsPerGroup
        var end = start + nChannelsPerGroup
        if (end > nChannels) {
            end = nChannels
        }
        val step = end - start

        for (i03 in 0 until l.ne03) {
            var sum = 0.0
            for (i02 in start until end) {
                for (i01 in 0 until l.ne01.toInt()) {
                    val srcOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                    val x = io.github.kotlinmania.llama.ore.readFloatRow(
                        src0Data,
                        srcOff,
                        l.ne00.toInt()
                    )

                    var sumr = 0.0
                    for (i00 in 0 until l.ne00.toInt()) {
                        sumr += x[i00].toDouble()
                    }
                    sum += sumr
                }
            }
            val mean = (sum / (l.ne00 * l.ne01 * step)).toFloat()

            var sum2 = 0.0
            for (i02 in start until end) {
                for (i01 in 0 until l.ne01.toInt()) {
                    val srcOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                    val x = io.github.kotlinmania.llama.ore.readFloatRow(
                        src0Data,
                        srcOff,
                        l.ne00.toInt()
                    )
                    val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                    val y = FloatArray(l.ne00.toInt())

                    var sumr = 0.0
                    for (i00 in 0 until l.ne00.toInt()) {
                        val v = x[i00] - mean
                        y[i00] = v
                        sumr += (v.toDouble() * v.toDouble())
                    }
                    sum2 += sumr
                    io.github.kotlinmania.llama.ore.writeFloatRow(
                        dstData,
                        dstOff,
                        y,
                        0,
                        l.ne00.toInt()
                    )
                }
            }
            val variance = (sum2 / (l.ne00 * l.ne01 * step)).toFloat()
            val scale = 1.0f / sqrt(variance + eps)

            for (i02 in start until end) {
                for (i01 in 0 until l.ne01.toInt()) {
                    val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                    val y = io.github.kotlinmania.llama.ore.readFloatRow(
                        dstData,
                        dstOff,
                        l.ne00.toInt()
                    )
                    ggml_vec_scale_f32(l.ne00.toInt(), y, scale)
                    io.github.kotlinmania.llama.ore.writeFloatRow(
                        dstData,
                        dstOff,
                        y,
                        0,
                        l.ne00.toInt()
                    )
                }
            }
        }
        i += nth
    }
}

// ops.cpp line 4037
fun ggml_compute_forward_group_norm(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_group_norm_f32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}

// ============================================================================
// ggml_compute_forward_l2_norm  — ops.cpp line 4055
// ============================================================================

// ops.cpp line 4057
private fun ggml_compute_forward_l2_norm_f32(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    require(io.github.kotlinmania.llama.ore.ggml_are_same_shape(src0, dst))
    require(src0.nb[0].toLong() == 4L) // sizeof(float)

    val ith = params.ith
    val nth = params.nth

    val l = io.github.kotlinmania.llama.ore.unaryOpLocals(dst)

    val eps = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)
    require(eps >= 0.0f)

    val src0Data = src0.data as ByteArray
    val dstData = dst.data as ByteArray

    for (i03 in 0 until l.ne03) {
        for (i02 in 0 until l.ne02) {
            var i01 = ith.toLong()
            while (i01 < l.ne01) {
                val srcOff = (i01 * l.nb01 + i02 * l.nb02 + i03 * l.nb03).toInt()
                val x =
                    io.github.kotlinmania.llama.ore.readFloatRow(src0Data, srcOff, l.ne00.toInt())

                var sum = 0.0
                for (i00 in 0 until l.ne00.toInt()) {
                    sum += (x[i00].toDouble() * x[i00].toDouble())
                }

                val dstOff = (i01 * l.nb1 + i02 * l.nb2 + i03 * l.nb3).toInt()
                val y = FloatArray(l.ne00.toInt())
                x.copyInto(y)

                val scale = 1.0f / max(sqrt(sum).toFloat(), eps)

                ggml_vec_scale_f32(l.ne00.toInt(), y, scale)
                io.github.kotlinmania.llama.ore.writeFloatRow(dstData, dstOff, y, 0, l.ne00.toInt())

                i01 += nth
            }
        }
    }
}

// ops.cpp line 4100
fun ggml_compute_forward_l2_norm(
    params: io.github.kotlinmania.llama.ore.GGMLComputeParams,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val src0 = dst.src[0]!!

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.ggml_compute_forward_l2_norm_f32(
            params,
            dst
        )
        else -> error("fatal error")
    }
}
