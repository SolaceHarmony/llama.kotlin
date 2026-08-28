// port-lint: source tmp/llama.cpp/ggml/src/ggml-cpu/common.h
package io.github.kotlinmania.llama.ore

/**
 * Kotlin port of ggml-cpu/common.h — shared CPU backend utilities.
 *
 * This file contains:
 * - Flash-attention tile configuration constants.
 * - Type-conversion helpers (FP16 ↔ FP32, BF16 ↔ FP32, I32 ↔ FP32, identity).
 * - [io.github.kotlinmania.llama.ore.TypeConversionTable] — a sealed interface that lets generic code convert
 *   between storage types and F32 without runtime dispatch.
 * - [io.github.kotlinmania.llama.ore.getThreadRange] — the standard utility for splitting row-based work across
 *   threads, used by almost every CPU kernel.
 */

import kotlin.math.min

// ============================================================================
// Flash-Attention tile constants
// ============================================================================

/** Tile size (number of query rows) used by the fused flash-attention kernel. */
const val GGML_FA_TILE_Q: Int = 64

/** Tile size (number of key/value rows) used by the fused flash-attention kernel. */
const val GGML_FA_TILE_KV: Int = 64

/**
 * Compile-time tile configuration for flash-attention.
 *
 * Mirrors `ggml_fa_tile_config` in C++.
 */
object GGMLFaTileConfig {
    /** Query tile size. */
    val Q: ULong = io.github.kotlinmania.llama.ore.GGML_FA_TILE_Q.toULong()

    /** Key/Value tile size. */
    val KV: ULong = io.github.kotlinmania.llama.ore.GGML_FA_TILE_KV.toULong()
}

// ============================================================================
// Scalar type-conversion helpers
// ============================================================================

/** FP16 → FP32 conversion — delegates to [io.github.kotlinmania.llama.ore.ggmlCpuFp16ToFp32]. */
fun f16ToF32(x: UShort): Float = io.github.kotlinmania.llama.ore.ggmlCpuFp16ToFp32(x)

/** FP32 → FP16 conversion — delegates to [io.github.kotlinmania.llama.ore.ggmlCpuFp32ToFp16]. */
fun f32ToF16(x: Float): UShort = io.github.kotlinmania.llama.ore.ggmlCpuFp32ToFp16(x)

/**
 * BF16 → FP32 conversion.
 *
 * BF16 is the upper 16 bits of an IEEE-754 float, so conversion is a
 * simple shift.
 */
fun bf16ToF32(x: io.github.kotlinmania.llama.ore.GGMLBF16): Float =
    Float.fromBits(x.bits.toInt() shl 16)

/**
 * FP32 → BF16 conversion (round-to-nearest-even).
 *
 * Mirrors `GGML_FP32_TO_BF16` from ggml-impl.h.
 */
fun f32ToBf16(x: Float): io.github.kotlinmania.llama.ore.GGMLBF16 {
    val bits = x.toBits()
    // Round-to-nearest-even: add rounding bias then truncate.
    val rounded = bits + 0x7FFF + ((bits shr 16) and 1)
    return io.github.kotlinmania.llama.ore.GGMLBF16((rounded ushr 16).toUShort())
}

/** Identity conversion for I32 → FP32. */
fun i32ToF32(x: Int): Float = x.toFloat()

/** Truncating conversion for FP32 → I32. */
fun f32ToI32(x: Float): Int = x.toInt()

/** Identity pass-through for FP32 → FP32. */
fun f32ToF32(x: Float): Float = x

// ============================================================================
// TypeConversionTable — generic type-conversion dispatch
// ============================================================================

/**
 * A sealed interface that provides `toF32` and `fromF32` for a given storage
 * type, replacing the C++ `type_conversion_table<T>` template specializations.
 *
 * Usage:
 * ```kotlin
 * val table = TypeConversionTable.fp16()
 * val f32Value = table.toF32(fp16Bits)
 * val fp16Value = table.fromF32(f32Value)
 * ```
 */
sealed interface TypeConversionTable<T> {
    fun toF32(value: T): Float
    fun fromF32(value: Float): T

    companion object {
        /** Conversion table for FP16 (stored as [UShort]). */
        fun fp16(): TypeConversionTable<UShort> = io.github.kotlinmania.llama.ore.Fp16Conversion

        /** Conversion table for BF16 (stored as [io.github.kotlinmania.llama.ore.GGMLBF16]). */
        fun bf16(): TypeConversionTable<io.github.kotlinmania.llama.ore.GGMLBF16> =
            io.github.kotlinmania.llama.ore.Bf16Conversion

        /** Conversion table for FP32 (identity). */
        fun fp32(): TypeConversionTable<Float> = io.github.kotlinmania.llama.ore.Fp32Conversion

        /** Conversion table for I32. */
        fun i32(): TypeConversionTable<Int> = io.github.kotlinmania.llama.ore.I32Conversion
    }
}

private object Fp16Conversion : io.github.kotlinmania.llama.ore.TypeConversionTable<UShort> {
    override fun toF32(value: UShort): Float = io.github.kotlinmania.llama.ore.f16ToF32(value)
    override fun fromF32(value: Float): UShort = io.github.kotlinmania.llama.ore.f32ToF16(value)
}

private object Bf16Conversion :
    io.github.kotlinmania.llama.ore.TypeConversionTable<io.github.kotlinmania.llama.ore.GGMLBF16> {
    override fun toF32(value: io.github.kotlinmania.llama.ore.GGMLBF16): Float =
        io.github.kotlinmania.llama.ore.bf16ToF32(value)
    override fun fromF32(value: Float): io.github.kotlinmania.llama.ore.GGMLBF16 =
        io.github.kotlinmania.llama.ore.f32ToBf16(value)
}

private object Fp32Conversion : io.github.kotlinmania.llama.ore.TypeConversionTable<Float> {
    override fun toF32(value: Float): Float = value
    override fun fromF32(value: Float): Float = value
}

private object I32Conversion : io.github.kotlinmania.llama.ore.TypeConversionTable<Int> {
    override fun toF32(value: Int): Float = io.github.kotlinmania.llama.ore.i32ToF32(value)
    override fun fromF32(value: Float): Int = io.github.kotlinmania.llama.ore.f32ToI32(value)
}

// ============================================================================
// Thread-range utility
// ============================================================================

/**
 * Compute the half-open row range `[ir0, ir1)` that thread [params].ith
 * should process, given a source tensor [src0].
 *
 * This is the standard work-splitting helper used by nearly every CPU kernel
 * in the GGML backend.  Rows are divided as evenly as possible across
 * [params].nth threads; the last thread may receive fewer rows.
 *
 * @return A [Pair] of `(ir0, ir1)` — the start (inclusive) and end (exclusive)
 *         row indices for this thread.
 */
fun getThreadRange(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, src0: io.github.kotlinmania.llama.ore.GGMLTensor): Pair<Long, Long> {
    val ith = params.ith.toLong()
    val nth = params.nth.toLong()

    val nr = io.github.kotlinmania.llama.ore.ggmlNrows(src0)

    // rows per thread (ceiling division)
    val dr = (nr + nth - 1) / nth

    val ir0 = dr * ith
    val ir1 = min(ir0 + dr, nr)

    return Pair(ir0, ir1)
}
