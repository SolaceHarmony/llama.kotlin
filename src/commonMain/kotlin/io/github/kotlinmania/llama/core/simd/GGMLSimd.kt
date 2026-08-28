package io.github.kotlinmania.llama.ore.simd

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe
import io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32
import io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16
import io.github.kotlinmania.llama.ore.halfToFloat
import io.github.kotlinmania.llama.ore.ggmlFp16ToFp32
import io.github.kotlinmania.llama.ore.ggmlFp32ToFp16

// port-lint: source ggml/src/ggml-cpu/simd-mappings.h
// The C header defines SIMD macros per-architecture (ARM NEON, AVX, SVE, etc.).
// In Kotlin/Native we provide scalar fallback implementations for all paths.

// ============================================================================
// FP16 ↔ FP32 conversion macros (scalar fallbacks)
// C: simd-mappings.h lines 38-163
// ============================================================================

/** Scalar fallback for GGML_CPU_FP16_TO_FP32 (all platforms). */
fun GGML_CPU_FP16_TO_FP32(x: Short): Float =
    io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x)

/** Scalar fallback for GGML_CPU_FP32_TO_FP16 (all platforms). */
fun GGML_CPU_FP32_TO_FP16(x: Float): Short =
    io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(x)

/** Scalar fallback for GGML_CPU_COMPUTE_FP16_TO_FP32. */
fun GGML_CPU_COMPUTE_FP16_TO_FP32(x: Short): Float =
    io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x)

/** Scalar fallback for GGML_CPU_COMPUTE_FP32_TO_FP16. */
fun GGML_CPU_COMPUTE_FP32_TO_FP16(x: Float): Short =
    io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(x)

// --- Platform-specific FP16 converters (all route to software conversion) ---

/** NEON FP16→FP32. C: neon_compute_fp16_to_fp32 */
fun neonComputeFp16ToFp32(h: UShort): Float = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(h)

/** NEON FP32→FP16. C: neon_compute_fp32_to_fp16 */
fun neonComputeFp32ToFp16(f: Float): UShort = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(f)

/** POWER9 FP16→FP32. C: power_compute_fp16_to_fp32 */
fun powerComputeFp16ToFp32(h: UShort): Float = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(h)

/** POWER9 FP32→FP16. C: power_compute_fp32_to_fp16 */
fun powerComputeFp32ToFp16(f: Float): UShort = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(f)

/** RISC-V FP16→FP32. C: riscv_compute_fp16_to_fp32 */
fun riscvComputeFp16ToFp32(h: UShort): Float = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(h)

/** RISC-V FP32→FP16. C: riscv_compute_fp32_to_fp16 */
fun riscvComputeFp32ToFp16(f: Float): UShort = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(f)

/** Lookup-table FP16→FP32. C: ggml_lookup_fp16_to_fp32 (line 134) */
fun ggmlLookupFp16ToFp32(f: UShort): Float = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(f)

// ============================================================================
// Endianness helper — C: simd-mappings.h line 741
// ============================================================================

/** Return the i-th byte of the integer 1 in native byte order. C: ggml_endian_byte */
fun ggmlEndianByte(i: Int): UByte {
    val one = 1
    return ((one shr (i * 8)) and 0xFF).toUByte()
}

// ============================================================================
// FP16 vector load/store (scalar fallbacks)
// Each platform has its own variant; we implement all as scalar loops.
// ============================================================================

/** AVX: load 8 FP16 values as FP32 array. C: __avx_f32cx8_load (line 636) */
fun avxF32cx8Load(data: ShortArray, offset: Int = 0): FloatArray {
    return FloatArray(8) { i -> io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(data[offset + i].toUShort()) }
}

/** AVX: store 8 FP32 values as FP16. C: __avx_f32cx8_store (line 645) */
fun avxF32cx8Store(dst: ShortArray, offset: Int, src: FloatArray) {
    for (i in 0 until 8) {
        dst[offset + i] = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src[i]).toShort()
    }
}

/** WASM: load 4 FP16 values as FP32 array. C: __wasm_f16x4_load (line 827) */
fun wasmF16x4Load(data: ShortArray, offset: Int = 0): FloatArray {
    return FloatArray(4) { i -> io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(data[offset + i].toUShort()) }
}

/** WASM: store 4 FP32 values as FP16. C: __wasm_f16x4_store (line 838) */
fun wasmF16x4Store(dst: ShortArray, offset: Int, src: FloatArray) {
    for (i in 0 until 4) {
        dst[offset + i] = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src[i]).toShort()
    }
}

/** SSE: load 4 FP16 values as FP32 array. C: __sse_f16x4_load (line 943) */
fun sseF16x4Load(data: ShortArray, offset: Int = 0): FloatArray {
    return FloatArray(4) { i -> io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(data[offset + i].toUShort()) }
}

/** SSE: store 4 FP32 values as FP16. C: __sse_f16x4_store (line 954) */
fun sseF16x4Store(dst: ShortArray, offset: Int, src: FloatArray) {
    for (i in 0 until 4) {
        dst[offset + i] = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src[i]).toShort()
    }
}

/** LASX: load 8 FP16 values as FP32 array. C: __lasx_f32cx8_load (line 1041) */
fun lasxF32cx8Load(data: ShortArray, offset: Int = 0): FloatArray {
    return FloatArray(8) { i -> io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(data[offset + i].toUShort()) }
}

/** LASX: store 8 FP32 values as FP16. C: __lasx_f32cx8_store (line 1048) */
fun lasxF32cx8Store(dst: ShortArray, offset: Int, src: FloatArray) {
    for (i in 0 until 8) {
        dst[offset + i] = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src[i]).toShort()
    }
}

/** LSX: load 4 FP16 values as FP32 array. C: __lsx_f16x4_load (line 1127) */
fun lsxF16x4Load(data: ShortArray, offset: Int = 0): FloatArray {
    return FloatArray(4) { i -> io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(data[offset + i].toUShort()) }
}

/** LSX: store 4 FP32 values as FP16. C: __lsx_f16x4_store (line 1138) */
fun lsxF16x4Store(dst: ShortArray, offset: Int, src: FloatArray) {
    for (i in 0 until 4) {
        dst[offset + i] = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src[i]).toShort()
    }
}

/** VXE/Z: load 4 FP16 values as FP32 array. C: __lzs_f16cx4_load (line 1226) */
fun lzsF16cx4Load(data: ShortArray, offset: Int = 0): FloatArray {
    return FloatArray(4) { i -> io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(data[offset + i].toUShort()) }
}

/** VXE/Z: store 4 FP32 values as FP16. C: __lzs_f16cx4_store (line 1238) */
fun lzsF16cx4Store(dst: ShortArray, offset: Int, src: FloatArray) {
    for (i in 0 until 4) {
        dst[offset + i] = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(src[i]).toShort()
    }
}

// ============================================================================
// GGML_F32_VEC scalar fallbacks
// The C macros define per-platform vector ops. We provide scalar equivalents.
// ============================================================================

const val GGML_F32_EPR = 8

fun ggmlF32VecZero(): FloatArray = FloatArray(io.github.kotlinmania.llama.ore.simd.GGML_F32_EPR)
fun ggmlF32VecSet1(x: Float): FloatArray = FloatArray(io.github.kotlinmania.llama.ore.simd.GGML_F32_EPR) { x }

fun ggmlF32VecLoad(data: FloatArray, offset: Int = 0): FloatArray =
    data.copyOfRange(offset, offset + io.github.kotlinmania.llama.ore.simd.GGML_F32_EPR)

fun ggmlF32VecStore(dst: FloatArray, offset: Int, src: FloatArray) {
    src.copyInto(dst, offset, 0, io.github.kotlinmania.llama.ore.simd.GGML_F32_EPR)
}

fun ggmlF32VecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray {
    return FloatArray(io.github.kotlinmania.llama.ore.simd.GGML_F32_EPR) { i -> a[i] + b[i] * c[i] }
}

fun ggmlF32VecAdd(a: FloatArray, b: FloatArray): FloatArray {
    return FloatArray(io.github.kotlinmania.llama.ore.simd.GGML_F32_EPR) { i -> a[i] + b[i] }
}

fun ggmlF32VecMul(a: FloatArray, b: FloatArray): FloatArray {
    return FloatArray(io.github.kotlinmania.llama.ore.simd.GGML_F32_EPR) { i -> a[i] * b[i] }
}

fun ggmlF32VecReduce(vararg sums: FloatArray): Float {
    var total = 0.0f
    for (s in sums) {
        for (v in s) total += v
    }
    return total
}

// ============================================================================
// Dot product operations (scalar, unrolled)
// ============================================================================

internal object GGMLSimd {

    private const val UNROLL = 8

    fun dotF32(
        dataA: ByteArray,
        baseOffsetA: Int,
        strideABytes: Int,
        dataB: ByteArray,
        baseOffsetB: Int,
        strideBBytes: Int,
        length: Int
    ): Float {
        var sum = 0.0f
        var idx = 0
        val limit = length - length % UNROLL
        var offsetA = baseOffsetA
        var offsetB = baseOffsetB

        while (idx < limit) {
            sum += dataA.getFloatLe(offsetA) * dataB.getFloatLe(offsetB)
            sum += dataA.getFloatLe(offsetA + strideABytes) * dataB.getFloatLe(offsetB + strideBBytes)
            sum += dataA.getFloatLe(offsetA + 2 * strideABytes) * dataB.getFloatLe(offsetB + 2 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 3 * strideABytes) * dataB.getFloatLe(offsetB + 3 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 4 * strideABytes) * dataB.getFloatLe(offsetB + 4 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 5 * strideABytes) * dataB.getFloatLe(offsetB + 5 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 6 * strideABytes) * dataB.getFloatLe(offsetB + 6 * strideBBytes)
            sum += dataA.getFloatLe(offsetA + 7 * strideABytes) * dataB.getFloatLe(offsetB + 7 * strideBBytes)

            idx += UNROLL
            offsetA += UNROLL * strideABytes
            offsetB += UNROLL * strideBBytes
        }

        while (idx < length) {
            sum += dataA.getFloatLe(offsetA) * dataB.getFloatLe(offsetB)
            idx += 1
            offsetA += strideABytes
            offsetB += strideBBytes
        }

        return sum
    }

    fun dotF16(
        dataA: ByteArray,
        baseOffsetA: Int,
        strideABytes: Int,
        dataB: ByteArray,
        baseOffsetB: Int,
        strideBBytes: Int,
        length: Int
    ): Float {
        var sum = 0.0f
        var idx = 0
        val limit = length - length % UNROLL
        var offsetA = baseOffsetA
        var offsetB = baseOffsetB

        while (idx < limit) {
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB)
            )
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA + strideABytes)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB + strideBBytes)
            )
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA + 2 * strideABytes)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB + 2 * strideBBytes)
            )
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA + 3 * strideABytes)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB + 3 * strideBBytes)
            )
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA + 4 * strideABytes)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB + 4 * strideBBytes)
            )
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA + 5 * strideABytes)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB + 5 * strideBBytes)
            )
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA + 6 * strideABytes)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB + 6 * strideBBytes)
            )
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA + 7 * strideABytes)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB + 7 * strideBBytes)
            )

            idx += UNROLL
            offsetA += UNROLL * strideABytes
            offsetB += UNROLL * strideBBytes
        }

        while (idx < length) {
            sum += io.github.kotlinmania.llama.ore.halfToFloat(dataA.getShortLe(offsetA)) * io.github.kotlinmania.llama.ore.halfToFloat(
                dataB.getShortLe(offsetB)
            )
            idx += 1
            offsetA += strideABytes
            offsetB += strideBBytes
        }

        return sum
    }
}
