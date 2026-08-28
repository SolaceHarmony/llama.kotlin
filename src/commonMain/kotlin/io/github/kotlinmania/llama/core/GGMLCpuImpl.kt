// port-lint: source ggml/src/ggml-cpu/ggml-cpu-impl.h
package io.github.kotlinmania.llama.ore

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Kotlin port of ggml-cpu-impl.h — CPU backend implementation internals.
 *
 * Contains:
 * - [io.github.kotlinmania.llama.ore.GGMLComputeParams]: per-thread compute context passed to every CPU kernel.
 * - Threadpool barrier and chunk-scheduling functions.
 * - Scalar FP16 ↔ FP32 helpers.
 * - SIMD vector type wrappers (scalar fallbacks for NEON, VXE, LoongArch).
 *
 * SIMD intrinsics have no direct Kotlin/Native equivalent, so all vector
 * operations are implemented as scalar array-based fallbacks matching the
 * C++ reference implementations in ggml-cpu-impl.h.
 */

// ============================================================================
// ggml_threadpool — opaque handle (defined in ggml-cpu internals)
// ============================================================================

@OptIn(ExperimentalAtomicApi::class)
class GGMLThreadpool {
    var nThreads: Int = 1
    private val currentChunk = AtomicInt(0)
    private val nBarrierPassed = AtomicInt(0)

    fun chunkGet(): Int = currentChunk.load()
    fun chunkSet(value: Int) { currentChunk.store(value) }
    fun chunkAdd(value: Int): Int = currentChunk.fetchAndAdd(value)
}

// ============================================================================
// ggml_compute_params — C: ggml-cpu-impl.h line 18
// ============================================================================

data class GGMLComputeParams(
    val ith: Int,
    val nth: Int,
    val wsize: ULong = 0uL,
    val wdata: ByteArray? = null,
    val threadpool: io.github.kotlinmania.llama.ore.GGMLThreadpool? = null,
    val useRef: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGMLComputeParams) return false
        return ith == other.ith && nth == other.nth && wsize == other.wsize && useRef == other.useRef
    }

    override fun hashCode(): Int {
        var result = ith
        result = 31 * result + nth
        result = 31 * result + wsize.hashCode()
        result = 31 * result + useRef.hashCode()
        return result
    }
}

// ============================================================================
// Threadpool helpers — C: ggml-cpu-impl.h line 532-535
// ============================================================================

fun ggmlBarrier(tp: io.github.kotlinmania.llama.ore.GGMLThreadpool) {
    if (tp.nThreads == 1) return
    // Multi-threaded spin barrier deferred until threading is wired up
}

fun ggmlThreadpoolChunkSet(tp: io.github.kotlinmania.llama.ore.GGMLThreadpool, value: Int) {
    tp.chunkSet(value)
}

fun ggmlThreadpoolChunkAdd(tp: io.github.kotlinmania.llama.ore.GGMLThreadpool, value: Int): Int {
    return tp.chunkAdd(value)
}

// ============================================================================
// Scalar FP16 helpers
// ============================================================================

fun ggmlCpuFp32ToFp16(x: Float): UShort = io.github.kotlinmania.llama.ore.ggmlFp32ToFp16(x)
fun ggmlCpuFp16ToFp32(x: UShort): Float = io.github.kotlinmania.llama.ore.ggmlFp16ToFp32(x)

// ============================================================================
// NEON scalar fallbacks — C: ggml-cpu-impl.h line 87-331
// These are 32-bit ARM compatibility fallbacks. On Kotlin we implement them
// as scalar operations on arrays (matching the C scalar fallback semantics).
// ============================================================================

/** Scalar: sum of all int16 elements widened to int32. C: vaddlvq_s16 */
fun vaddlvqS16(v: ShortArray): Int {
    require(v.size == 8)
    var sum = 0
    for (i in 0 until 8) sum += v[i].toInt()
    return sum
}

/** Scalar: pairwise add int16x8. C: vpaddq_s16 */
fun vpaddqS16(a: ShortArray, b: ShortArray): ShortArray {
    require(a.size == 8 && b.size == 8)
    return shortArrayOf(
        (a[0] + a[1]).toShort(), (a[2] + a[3]).toShort(), (a[4] + a[5]).toShort(), (a[6] + a[7]).toShort(),
        (b[0] + b[1]).toShort(), (b[2] + b[3]).toShort(), (b[4] + b[5]).toShort(), (b[6] + b[7]).toShort()
    )
}

/** Scalar: pairwise add int32x4. C: vpaddq_s32 */
fun vpaddqS32(a: IntArray, b: IntArray): IntArray {
    require(a.size == 4 && b.size == 4)
    return intArrayOf(a[0] + a[1], a[2] + a[3], b[0] + b[1], b[2] + b[3])
}

/** Scalar: sum of int32x4 elements. C: vaddvq_s32 */
fun vaddvqS32(v: IntArray): Int {
    require(v.size == 4)
    return v[0] + v[1] + v[2] + v[3]
}

/** Scalar: sum of float32x4 elements. C: vaddvq_f32 */
fun vaddvqF32(v: FloatArray): Float {
    require(v.size == 4)
    return v[0] + v[1] + v[2] + v[3]
}

/** Scalar: max of float32x4 elements. C: vmaxvq_f32 */
fun vmaxvqF32(v: FloatArray): Float {
    require(v.size == 4)
    return max(max(v[0], v[1]), max(v[2], v[3]))
}

/** Scalar: convert float32x4 to int32x4 with round-to-nearest. C: vcvtnq_s32_f32 */
fun vcvtnqS32F32(v: FloatArray): IntArray {
    require(v.size == 4)
    return intArrayOf(v[0].roundToInt(), v[1].roundToInt(), v[2].roundToInt(), v[3].roundToInt())
}

/** Scalar: interleave low halves of two uint8x8. C: vzip1_u8 */
fun vzip1U8(a: UByteArray, b: UByteArray): UByteArray {
    require(a.size == 8 && b.size == 8)
    return ubyteArrayOf(a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3])
}

/** Scalar: interleave high halves of two uint8x8. C: vzip2_u8 */
fun vzip2U8(a: UByteArray, b: UByteArray): UByteArray {
    require(a.size == 8 && b.size == 8)
    return ubyteArrayOf(a[4], b[4], a[5], b[5], a[6], b[6], a[7], b[7])
}

// --- NEON multi-element loads (scalar fallbacks) ---

/** Scalar: load 2×int16x8 from pointer. C: ggml_vld1q_s16_x2 */
fun ggmlVld1qS16X2(data: ShortArray, offset: Int = 0): Array<ShortArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 8),
        data.copyOfRange(offset + 8, offset + 16)
    )
}

/** Scalar: load 2×uint8x16 from pointer. C: ggml_vld1q_u8_x2 */
fun ggmlVld1qU8X2(data: UByteArray, offset: Int = 0): Array<UByteArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 16),
        data.copyOfRange(offset + 16, offset + 32)
    )
}

/** Scalar: load 4×uint8x16 from pointer. C: ggml_vld1q_u8_x4 */
fun ggmlVld1qU8X4(data: UByteArray, offset: Int = 0): Array<UByteArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 16),
        data.copyOfRange(offset + 16, offset + 32),
        data.copyOfRange(offset + 32, offset + 48),
        data.copyOfRange(offset + 48, offset + 64)
    )
}

/** Scalar: load 2×int8x16 from pointer. C: ggml_vld1q_s8_x2 */
fun ggmlVld1qS8X2(data: ByteArray, offset: Int = 0): Array<ByteArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 16),
        data.copyOfRange(offset + 16, offset + 32)
    )
}

/** Scalar: load 4×int8x16 from pointer. C: ggml_vld1q_s8_x4 */
fun ggmlVld1qS8X4(data: ByteArray, offset: Int = 0): Array<ByteArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 16),
        data.copyOfRange(offset + 16, offset + 32),
        data.copyOfRange(offset + 32, offset + 48),
        data.copyOfRange(offset + 48, offset + 64)
    )
}

/** Scalar: table lookup int8x16 by uint8x16 indices. C: ggml_vqtbl1q_s8 */
fun ggmlVqtbl1qS8(a: ByteArray, b: UByteArray): ByteArray {
    require(a.size == 16 && b.size == 16)
    return ByteArray(16) { i -> a[b[i].toInt()] }
}

/** Scalar: table lookup uint8x16 by uint8x16 indices. C: ggml_vqtbl1q_u8 */
fun ggmlVqtbl1qU8(a: UByteArray, b: UByteArray): UByteArray {
    require(a.size == 16 && b.size == 16)
    return UByteArray(16) { i -> a[b[i].toInt()] }
}

/** Scalar: dot product accumulate int8x16. C: ggml_vdotq_s32 */
fun ggmlVdotqS32(acc: IntArray, a: ByteArray, b: ByteArray): IntArray {
    require(acc.size == 4 && a.size == 16 && b.size == 16)
    val result = acc.copyOf()
    for (lane in 0 until 4) {
        var sum = 0
        for (j in 0 until 4) {
            val idx = lane * 4 + j
            sum += a[idx].toInt() * b[idx].toInt()
        }
        result[lane] += sum
    }
    return result
}

/** Scalar: NVFP4 dot product of 8 elements. C: ggml_nvfp4_dot8 */
fun ggmlNvfp4Dot8(q4Lo: ByteArray, q8Lo: ByteArray, q4Hi: ByteArray, q8Hi: ByteArray): IntArray {
    require(q4Lo.size == 8 && q8Lo.size == 8 && q4Hi.size == 8 && q8Hi.size == 8)
    // p_lo = vmull_s8(q4_lo, q8_lo) → 8 int16
    // p_hi = vmull_s8(q4_hi, q8_hi) → 8 int16
    // sum_lo = vpaddlq_s16(p_lo) → 4 int32 (pairwise widening add)
    // sum_hi = vpaddlq_s16(p_hi) → 4 int32
    // return vaddq_s32(sum_lo, sum_hi)
    val result = IntArray(4)
    for (lane in 0 until 4) {
        val loIdx = lane * 2
        val sumLo = q4Lo[loIdx].toInt() * q8Lo[loIdx].toInt() + q4Lo[loIdx + 1].toInt() * q8Lo[loIdx + 1].toInt()
        val sumHi = q4Hi[loIdx].toInt() * q8Hi[loIdx].toInt() + q4Hi[loIdx + 1].toInt() * q8Hi[loIdx + 1].toInt()
        result[lane] = sumLo + sumHi
    }
    return result
}

// ============================================================================
// VXE (IBM Z) scalar fallbacks — C: ggml-cpu-impl.h line 361-512
// ============================================================================

/** Scalar: load 2×uint8x16 from array. C: ggml_vec_xl_u8x2 */
fun ggmlVecXlU8X2(data: UByteArray, offset: Int = 0): Array<UByteArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 16),
        data.copyOfRange(offset + 16, offset + 32)
    )
}

/** Scalar: load 4×uint8x16 from array. C: ggml_vec_xl_u8x4 */
fun ggmlVecXlU8X4(data: UByteArray, offset: Int = 0): Array<UByteArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 16),
        data.copyOfRange(offset + 16, offset + 32),
        data.copyOfRange(offset + 32, offset + 48),
        data.copyOfRange(offset + 48, offset + 64)
    )
}

/** Scalar: load 4×int8x16 from array. C: ggml_vec_xl_s8x4 */
fun ggmlVecXlS8X4(data: ByteArray, offset: Int = 0): Array<ByteArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 16),
        data.copyOfRange(offset + 16, offset + 32),
        data.copyOfRange(offset + 32, offset + 48),
        data.copyOfRange(offset + 48, offset + 64)
    )
}

/** Scalar: load 2×int16x8 from array. C: ggml_vec_xl_s16x2 */
fun ggmlVecXlS16X2(data: ShortArray, offset: Int = 0): Array<ShortArray> {
    return arrayOf(
        data.copyOfRange(offset, offset + 8),
        data.copyOfRange(offset + 8, offset + 16)
    )
}

/** Scalar: table lookup int8x16 by uint8x16 indices. C: ggml_vec_tbl */
fun ggmlVecTbl(a: ByteArray, b: UByteArray): ByteArray {
    require(a.size == 16 && b.size == 16)
    return ByteArray(16) { i -> a[b[i].toInt()] }
}

/** Scalar: pairwise add int16x8. C: vec_padd_s16 */
fun vecPaddS16(a: ShortArray, b: ShortArray): ShortArray {
    require(a.size == 8 && b.size == 8)
    // Pack even elements and add odd elements
    return shortArrayOf(
        (a[0] + a[1]).toShort(), (a[2] + a[3]).toShort(), (a[4] + a[5]).toShort(), (a[6] + a[7]).toShort(),
        (b[0] + b[1]).toShort(), (b[2] + b[3]).toShort(), (b[4] + b[5]).toShort(), (b[6] + b[7]).toShort()
    )
}

/** Scalar: horizontal sum of float32x4. C: vec_hsum_f32x4 */
fun vecHsumF32X4(v: FloatArray): Float {
    require(v.size == 4)
    return v[0] + v[1] + v[2] + v[3]
}

/** Scalar: horizontal sum of int32x4. C: vec_hsum_i32x4 */
fun vecHsumI32X4(v: IntArray): Int {
    require(v.size == 4)
    return v[0] + v[1] + v[2] + v[3]
}

/** Scalar: dot product accumulate for VXE. C: ggml_vec_dot (VXE version) */
fun ggmlVecDotVxe(acc: IntArray, a: ByteArray, b: ByteArray): IntArray {
    require(acc.size == 4 && a.size == 16 && b.size == 16)
    val result = acc.copyOf()
    for (lane in 0 until 4) {
        var sum = 0
        for (j in 0 until 4) {
            val idx = lane * 4 + j
            sum += a[idx].toInt() * b[idx].toInt()
        }
        result[lane] += sum
    }
    return result
}

// ============================================================================
// LoongArch scalar fallbacks — C: ggml-cpu-impl.h line 516-529
// ============================================================================

/** Scalar: broadcast float to float32x4. C: __lsx_vreplfr2vr_s */
fun lsxVreplfr2vrS(v: Float): FloatArray = floatArrayOf(v, v, v, v)

/** Scalar: broadcast float to float32x8. C: __lasx_xvreplfr2vr_s */
fun lasxXvreplfr2vrS(v: Float): FloatArray = floatArrayOf(v, v, v, v, v, v, v, v)
