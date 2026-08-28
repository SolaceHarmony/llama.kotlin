// port-lint: source ggml/src/ggml-impl.h
package io.github.kotlinmania.llama.ore

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

// ============================================================================
// Constants (from ggml-impl.h)
// ============================================================================

/** Required for mmap as gguf only guarantees 32-byte alignment. */
const val TENSOR_ALIGNMENT: Int = 32

// ============================================================================
// Alignment helpers — transliteration of C macros GGML_UP / GGML_PAD
// ============================================================================

/** Round [n] up to the next multiple of 32. */
fun ggml_up32(n: Int): Int = (n + 31) and 31.inv()

/**
 * Round [n] up to the next multiple of [m].
 * [m] **must** be a power of two.
 */
fun ggml_up(n: Int, m: Int): Int {
    require((m and (m - 1)) == 0) { "m must be a power of 2, got $m" }
    return (n + m - 1) and (m - 1).inv()
}

/**
 * Pad [n] up to the next multiple of [m] (general, not limited to power-of-two).
 * Equivalent to the C macro `GGML_PAD(n, m)`.
 */
fun ggml_pad(n: Int, m: Int): Int {
    return ((n + m - 1) / m) * m
}

/** ULong overload for sizes. */
fun ggml_pad(n: ULong, m: ULong): ULong {
    return ((n + m - 1u) / m) * m
}

// ============================================================================
// FP16 ↔ FP32 conversion routines
// ref: https://github.com/Maratyszcza/FP16
// ============================================================================

/**
 * Reinterpret a raw 32-bit pattern as a Float (equivalent to C `fp32_from_bits`).
 */
fun fp32_from_bits(w: Int): Float = Float.fromBits(w)

/**
 * Reinterpret a Float as its raw 32-bit pattern (equivalent to C `fp32_to_bits`).
 */
fun fp32_to_bits(f: Float): Int = f.toRawBits()

/**
 * Converts a 16-bit half-precision float (ggml_fp16_t) to a 32-bit single-precision float.
 *
 * Port of `ggml_compute_fp16_to_fp32` from ggml-impl.h.
 * Uses the FP16 library approach (Maratyszcza) with magic-number denormal handling.
 */
fun ggml_compute_fp16_to_fp32(h: Short): Float {
    val w = (h.toInt() and 0xFFFF) shl 16
    val sign = w and Int.MIN_VALUE
    val two_w = w + w

    val exp_offset = 0xE0 shl 23
    // 0x1.0p-112f == 2^(-112) == Float.fromBits(0x07800000)
    val exp_scale = io.github.kotlinmania.llama.ore.fp32_from_bits(0x07800000)
    val normalized_value = io.github.kotlinmania.llama.ore.fp32_from_bits(((two_w ushr 4) + exp_offset)) * exp_scale

    val magic_mask = 126 shl 23
    val magic_bias = 0.5f
    val denormalized_value = io.github.kotlinmania.llama.ore.fp32_from_bits((two_w ushr 17) or magic_mask) - magic_bias

    val denormalized_cutoff = 1 shl 27
    val result = sign or
        if ((two_w ushr 0).toLong() and 0xFFFFFFFFL < (denormalized_cutoff.toLong() and 0xFFFFFFFFL))
            io.github.kotlinmania.llama.ore.fp32_to_bits(denormalized_value)
        else
            io.github.kotlinmania.llama.ore.fp32_to_bits(normalized_value)
    return io.github.kotlinmania.llama.ore.fp32_from_bits(result)
}

/**
 * Converts a 32-bit single-precision float to a 16-bit half-precision float (ggml_fp16_t).
 *
 * Port of `ggml_compute_fp32_to_fp16` from ggml-impl.h.
 */
fun ggml_compute_fp32_to_fp16(f: Float): Short {
    // scale_to_inf  = 0x1.0p+112f = Float.fromBits(0x77800000)
    val scale_to_inf = io.github.kotlinmania.llama.ore.fp32_from_bits(0x77800000)
    // scale_to_zero = 0x1.0p-110f = Float.fromBits(0x08800000)
    val scale_to_zero = io.github.kotlinmania.llama.ore.fp32_from_bits(0x08800000)
    var base = (abs(f) * scale_to_inf) * scale_to_zero

    val w = io.github.kotlinmania.llama.ore.fp32_to_bits(f)
    val shl1_w = w + w
    val sign = w and Int.MIN_VALUE
    var bias = shl1_w and -0x01000000
    if ((bias.toLong() and 0xFFFFFFFFL) < (0x71000000L)) {
        bias = 0x71000000
    }

    base = io.github.kotlinmania.llama.ore.fp32_from_bits((bias ushr 1) + 0x07800000) + base
    val bits = io.github.kotlinmania.llama.ore.fp32_to_bits(base)
    val exp_bits = (bits ushr 13) and 0x00007C00
    val mantissa_bits = bits and 0x00000FFF
    val nonsign = exp_bits + mantissa_bits
    val result = (sign ushr 16) or
        if ((shl1_w.toLong() and 0xFFFFFFFFL) > 0xFF000000L) 0x7E00 else nonsign
    return result.toShort()
}

// Convenience aliases matching C macros
/** Alias for [io.github.kotlinmania.llama.ore.ggml_compute_fp16_to_fp32]. */
fun GGML_FP16_TO_FP32(x: Short): Float =
    io.github.kotlinmania.llama.ore.ggml_compute_fp16_to_fp32(x)
/** Alias for [io.github.kotlinmania.llama.ore.ggml_compute_fp32_to_fp16]. */
fun GGML_FP32_TO_FP16(x: Float): Short =
    io.github.kotlinmania.llama.ore.ggml_compute_fp32_to_fp16(x)

// ============================================================================
// Legacy half ↔ float helpers (kept for backward compatibility)
// ============================================================================

/**
 * Converts a 16-bit half-precision float to a 32-bit single-precision float.
 * Based on public domain code by Fabian "ryg" Giesen and others.
 */
internal fun halfToFloat(h_bits: Short): Float {
    val h = h_bits.toInt() and 0xFFFF // Ensure we're working with 16 bits, unsigned

    val signMaskF32 = Int.MIN_VALUE
    val f32Infinity = 0x7F800000 // Positive infinity in F32

    val hSign = (h ushr 15)
    val hExp = (h ushr 10) and 0x1F
    val hMant = h and 0x03FF

    if (hExp == 0) { // Denormalized or zero
        if (hMant == 0) { // Zero
            return Float.fromBits(hSign shl 31)
        } else { // Denormalized F16; convert to normalized F32
            var mant = hMant
            var exp = hExp
            // Normalize: shift mantissa left until the leading bit is 1
            while ((mant and 0x0400) == 0) { // 0x0400 is 10th bit (implicit 1 for F16 normalized)
                mant = mant shl 1
                exp-- // Adjust exponent downwards
            }
            // Remove the implicit leading 1 from mantissa (which is now explicit)
            mant = mant and 0x03FF
            // Adjusted exponent for F32: (F16 denorm exp is effectively -14)
            // F32 exp = (exp + 1) - 15 (F16 bias) + 127 (F32 bias)
            val f32Exp = (exp + 1) + (127 - 15)
            val f32Mant = mant shl 13 // Align F16 10-bit mantissa to F32 23-bit
            return Float.fromBits((hSign shl 31) or (f32Exp shl 23) or f32Mant)
        }
    } else if (hExp == 0x1F) { // Infinity or NaN
        if (hMant == 0) { // Infinity
            return Float.fromBits((hSign shl 31) or f32Infinity)
        } else { // NaN
            // Propagate NaN payload (hMant) to F32. Make it a quiet NaN.
            // A common way is to set MSB of mantissa. F32 NaN: exp all 1s, mant non-zero.
            return Float.fromBits((hSign shl 31) or f32Infinity or (hMant shl 13) or (1 shl 22)) // set a high bit in mantissa for qNaN
        }
    } else { // Normalized F16
        val f32Sign = hSign shl 31
        // F32 exp = F16 exp - F16 bias + F32 bias
        val f32Exp = (hExp - 15 + 127) shl 23
        // F32 mant = F16 mant left shifted by (23 - 10) = 13 bits
        val f32Mant = hMant shl 13
        return Float.fromBits(f32Sign or f32Exp or f32Mant)
    }
}

/**
 * Converts a 32-bit single-precision float to a 16-bit half-precision float.
 * Implements Round-to-Nearest-Ties-to-Even.
 * Based on public domain code by Fabian "ryg" Giesen (gist:2156668).
 */
internal fun floatToHalf(f_val: Float): Short {
    val f32bits = f_val.toRawBits()
    val fSign = (f32bits ushr 16) and 0x8000 // F16 sign bit (already shifted)
    val absF = f32bits and 0x7FFFFFFF // Absolute value of F32

    // Handle special cases
    if (absF > 0x47FFEFFF) { // F32 value is too large for F16 normal => F16 Inf or NaN
        // NaN if F32 mantissa is non-zero, else Inf
    val mantissaIsNonZero = (absF and 0x007FFFFF) != 0
        // F16 Inf/NaN: exp all 1s (0x1F for 5 bits -> 0x7C00 when shifted)
        // For NaN, set MSB of mantissa (e.g., 0x0200 for F16) or any non-zero pattern
        return (fSign or 0x7C00 or if (mantissaIsNonZero) 0x0200 else 0).toShort()
    }

    if (absF < 0x38800000) { // F32 value is too small for F16 normal => F16 denormal or zero
        // Convert F32 to F16 denormal
        // Add implicit F32 '1' bit to mantissa
    val fMant = (absF and 0x007FFFFF) or 0x00800000
        // Calculate F16 denormal shift amount
        // F32 exp is (absF ushr 23). Denormal F16 exp is effectively -14.
        // Shift needed = 24 (F32 mant+implicit_1 bits) - (10 (F16 mant bits) + ( (absF ushr 23) - 127 (F32 bias) - (-14 (F16 denorm_exp)) ) )
        // shift = 24 - (10 + ( (absF ushr 23) - 113) ) = 24 - 10 - (absF ushr 23) + 113 = 127 - (absF ushr 23)
    val shift = 127 - (absF ushr 23) // Number of positions to shift right to align for F16 denormal mantissa

    val hMant = if (shift < 24) (fMant ushr shift) else 0

        // Rounding (RTNE for denormals requires checking bits shifted out)
    val roundBits = fMant and ((1 shl shift) - 1) // Bits lost
        // Tie-breaking: if exactly halfway, round to even (LSB of hMant is 0 after rounding)
        // Threshold for rounding up is halfway mark (1 << (shift - 1))
    if (roundBits > (1 shl (shift - 1)) || (roundBits == (1 shl (shift - 1)) && (hMant and 1) != 0)) {
           var h_temp = hMant + 1
           // If rounding caused overflow into implicit leading bit of a normal number
           if(h_temp == 0x0400) { // 0x0400 is 1024, meaning it became 1.0 * 2^(exp_min_norm_f16)
               return (fSign or (1 shl 10)).toShort() // Smallest normalized F16
           }
           return (fSign or h_temp).toShort()
        }
        return (fSign or hMant).toShort()
    }

    // Handle normalized F16
    // Shift and mask F32 exponent and mantissa to F16 form
    // F32 exp bias 127, F16 exp bias 15. Diff = 112.
    // F16 exp = (F32 exp - 112)
    // F32 mantissa has 23 bits, F16 has 10. Diff = 13.
    val hExp = ((absF ushr 23) - 112) shl 10 // Shifted F16 exponent
    var hMant = (absF and 0x007FFFFF) ushr 13 // Shifted F16 mantissa

    // Rounding for normalized numbers (RTNE)
    // Check the MSB of the bits that were shifted out (the rounding bit)
    if ((absF and 0x00001000) != 0) { // If rounding bit is 1 (0x1000 is 2^12, MSB of 13 shifted bits)
        // Check for tie-breaking (if remaining shifted bits are zero AND LSB of hMant is 1)
    if ((absF and 0x00000FFF) != 0 || (hMant and 1) != 0) {
            hMant++
            if (hMant == 0x0400) { // Mantissa overflowed to 1024
                hMant = 0 // Reset mantissa
                // Increment exponent (already shifted)
                return (fSign or (hExp + (1 shl 10)) or hMant).toShort()
            }
        }
    }
    return (fSign or hExp or hMant).toShort()
}

// ============================================================================
// E8M0 conversion helpers (MX / MXFP scaling exponents)
// ============================================================================

/**
 * Converts an E8M0 exponent-only byte to float32.
 *
 * E8M0 encodes pure powers of two: value = 2^(x − 127).
 * Special case: x == 0 maps to 2^(−127) (the smallest representable value).
 */
fun ggml_e8m0_to_fp32(x: UByte): Float {
    val bits: Int = if (x.toInt() == 0) {
        // 2^(-127) expressed as a denormal float: sign=0, exp=0, mantissa=0x400000
        0x00400000
    } else {
        x.toInt() shl 23
    }
    return Float.fromBits(bits)
}

/**
 * Equal to [io.github.kotlinmania.llama.ore.ggml_e8m0_to_fp32] / 2.
 * Useful with MXFP4 quantization since the E0M2 values are doubled.
 */
fun ggml_e8m0_to_fp32_half(x: UByte): Float {
    val bits: Int = if (x.toInt() < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        0x00200000 shl x.toInt()
    } else {
        // 0.5 * 2^(x-127) = 2^(x-128) → normalized with exponent (x-1)
        (x.toInt() - 1) shl 23
    }
    return Float.fromBits(bits)
}

/** Alias for [io.github.kotlinmania.llama.ore.ggml_e8m0_to_fp32]. */
fun GGML_E8M0_TO_FP32(x: UByte): Float = io.github.kotlinmania.llama.ore.ggml_e8m0_to_fp32(x)
/** Alias for [io.github.kotlinmania.llama.ore.ggml_e8m0_to_fp32_half]. */
fun GGML_E8M0_TO_FP32_HALF(x: UByte): Float =
    io.github.kotlinmania.llama.ore.ggml_e8m0_to_fp32_half(x)

// ============================================================================
// UE4M3 conversion helpers (unsigned, 4 exp bits bias=7, 3 mantissa bits)
// ============================================================================

/**
 * Converts an unsigned E4M3 byte to float32.
 *
 * Returns value × 0.5 to match `kvalues_mxfp4` convention (kvalues = 2 × E2M1_float).
 */
fun ggml_ue4m3_to_fp32(x: UByte): Float {
    val xi = x.toInt()
    if (xi == 0 || xi == 0x7F) return 0.0f
    val exp = (xi ushr 3) and 0xF
    val man = xi and 0x7
    val raw: Float = if (exp == 0) {
        // subnormal: man * 2^(-9)
        man.toFloat() * (1.0f / 512.0f)
    } else {
        // normalized: (1 + man/8) * 2^(exp-7)
        (1.0f + man.toFloat() / 8.0f) * io.github.kotlinmania.llama.ore.twoToThe(exp - 7)
    }
    return raw * 0.5f
}

/**
 * Converts a float32 to an unsigned E4M3 byte.
 */
fun ggml_fp32_to_ue4m3(x: Float): UByte {
    var x = x
    if (!(x > 0.0f)) return 0u
    if (x > 448.0f) x = 448.0f

    val bits = io.github.kotlinmania.llama.ore.fp32_to_bits(x)
    val fp32_exp = ((bits ushr 23) and 0xFF) - 127
    val fp32_man = (bits ushr 20) and 0x7
    var ue4m3_exp = fp32_exp + 7
    if (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        var man = (x * 512.0f + 0.5f).toInt()
        if (man > 7) man = 7
        if (man < 1) return 0u
        return man.toUByte()
    }
    if (ue4m3_exp >= 15) return 0x7Eu
    val round_bit = (bits ushr 19) and 1
    var ue4m3_man = fp32_man + round_bit
    if (ue4m3_man > 7) {
        ue4m3_man = 0
        ue4m3_exp++
        if (ue4m3_exp >= 15) return 0x7Eu
    }
    return ((ue4m3_exp shl 3) or ue4m3_man).toUByte()
}

/** Fast 2^n for small integer n (positive or negative). */
private fun twoToThe(n: Int): Float {
    // Construct the float directly from the exponent field
    return if (n >= -126 && n <= 127) {
        Float.fromBits((n + 127) shl 23)
    } else {
        // Fallback for out-of-range (shouldn't happen for E4M3)
        var result = 1.0f
        if (n > 0) { repeat(n) { result *= 2.0f } }
        else { repeat(-n) { result *= 0.5f } }
        result
    }
}

// ============================================================================
// BF16 (Brain Float 16) ↔ FP32 conversion
// ============================================================================

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32-bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * @see [io.github.kotlinmania.llama.ore.GGMLBF16] defined in GGMLTypes.kt
 */
fun ggml_compute_bf16_to_fp32(h: io.github.kotlinmania.llama.ore.GGMLBF16): Float {
    return Float.fromBits(h.bits.toInt() shl 16)
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 */
fun ggml_compute_fp32_to_bf16(s: Float): io.github.kotlinmania.llama.ore.GGMLBF16 {
    val i = io.github.kotlinmania.llama.ore.fp32_to_bits(s)
    if (((i and 0x7fffffff).toLong() and 0xFFFFFFFFL) > 0x7f800000L) {
        // NaN → force to quiet
        return io.github.kotlinmania.llama.ore.GGMLBF16(((i ushr 16) or 64).toUShort())
    }
    val rounded = i + (0x7fff + ((i ushr 16) and 1))
    return io.github.kotlinmania.llama.ore.GGMLBF16((rounded ushr 16).toUShort())
}

/** Alias for [io.github.kotlinmania.llama.ore.ggml_compute_fp32_to_bf16]. */
fun GGML_FP32_TO_BF16(x: Float): io.github.kotlinmania.llama.ore.GGMLBF16 =
    io.github.kotlinmania.llama.ore.ggml_compute_fp32_to_bf16(x)
/** Alias for [io.github.kotlinmania.llama.ore.ggml_compute_bf16_to_fp32]. */
fun GGML_BF16_TO_FP32(x: io.github.kotlinmania.llama.ore.GGMLBF16): Float =
    io.github.kotlinmania.llama.ore.ggml_compute_bf16_to_fp32(x)

// ============================================================================
// Softplus helper
// ============================================================================

/**
 * Computes softplus: log(1 + exp(input)).
 * For large [input] (> 20), returns [input] directly to avoid overflow.
 */
fun ggml_compute_softplus_f32(input: Float): Float {
    return if (input > 20.0f) input else ln(1.0f + exp(input))
}

// ============================================================================
// Tensor layout / operation helpers
// ============================================================================

/**
 * Returns `true` if tensors [a] and [b] have identical type, element counts,
 * and byte strides across every dimension.
 *
 * Port of `ggml_are_same_layout` from ggml-impl.h.
 */
fun ggml_are_same_layout(a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    if (a.type != b.type) return false
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) return false
        if (a.nb[i] != b.nb[i]) return false
    }
    return true
}

/**
 * Returns `true` if the given [op] is an "empty" operation that doesn't
 * perform actual computation (metadata-only reshaping / viewing).
 */
fun ggml_op_is_empty(op: io.github.kotlinmania.llama.ore.GGMLOp): Boolean {
    return when (op) {
        io.github.kotlinmania.llama.ore.GGMLOp.NONE,
        io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE,
        io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE,
        io.github.kotlinmania.llama.ore.GGMLOp.VIEW,
        io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE -> true
        else -> false
    }
}

/**
 * Returns `true` if tensor [t] is a view of another tensor.
 */
fun ggml_impl_is_view(t: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean = t.viewSrc != null

// ============================================================================
// Op-params accessors
// ============================================================================

/**
 * Copies [paramsSize] bytes from [params] into [tensor]'s opParams.
 * [params] is interpreted as an IntArray where each Int occupies 4 bytes.
 */
fun ggml_set_op_params(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, params: IntArray, paramsSize: Int) {
    require(paramsSize <= io.github.kotlinmania.llama.ore.GGML_MAX_OP_PARAMS) {
        "paramsSize ($paramsSize) exceeds GGML_MAX_OP_PARAMS (${io.github.kotlinmania.llama.ore.GGML_MAX_OP_PARAMS})"
    }
    val intCount = paramsSize / Int.SIZE_BYTES
    for (j in 0 until intCount) {
        tensor.opParams[j] = params[j]
    }
}

/** Reads the [i]-th Int32 from [tensor]'s op_params. */
fun ggml_get_op_params_i32(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, i: Int): Int {
    require(i < io.github.kotlinmania.llama.ore.GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { "i ($i) out of range" }
    return tensor.opParams[i]
}

/** Reads the [i]-th Float from [tensor]'s op_params (reinterpreted from Int bits). */
fun ggml_get_op_params_f32(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, i: Int): Float {
    require(i < io.github.kotlinmania.llama.ore.GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { "i ($i) out of range" }
    return Float.fromBits(tensor.opParams[i])
}

/** Writes an Int32 [value] at position [i] in [tensor]'s op_params. */
fun ggml_set_op_params_i32(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, i: Int, value: Int) {
    require(i < io.github.kotlinmania.llama.ore.GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { "i ($i) out of range" }
    tensor.opParams[i] = value
}

/** Writes a Float [value] at position [i] in [tensor]'s op_params (stored as raw bits). */
fun ggml_set_op_params_f32(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, i: Int, value: Float) {
    require(i < io.github.kotlinmania.llama.ore.GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { "i ($i) out of range" }
    tensor.opParams[i] = value.toRawBits()
}

// ============================================================================
// Custom op-param structures
// ============================================================================

/**
 * Parameters for a custom unary operation.
 *
 * Port of `struct ggml_map_custom1_op_params`.
 */
class GGMLMapCustom1OpParams(
    val fun_: ((io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Any?) -> Unit)? = null,
    val nTasks: Int = 1,
    val userdata: Any? = null
)

/**
 * Parameters for a custom binary operation.
 *
 * Port of `struct ggml_map_custom2_op_params`.
 */
class GGMLMapCustom2OpParams(
    val fun_: ((io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Any?) -> Unit)? = null,
    val nTasks: Int = 1,
    val userdata: Any? = null
)

/**
 * Parameters for a custom ternary operation.
 *
 * Port of `struct ggml_map_custom3_op_params`.
 */
class GGMLMapCustom3OpParams(
    val fun_: ((io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Any?) -> Unit)? = null,
    val nTasks: Int = 1,
    val userdata: Any? = null
)

/**
 * Parameters for a general custom operation.
 *
 * Port of `struct ggml_custom_op_params`.
 */
class GGMLCustomOpParams(
    val fun_: ((io.github.kotlinmania.llama.ore.GGMLTensor, Int, Any?) -> Unit)? = null,
    val nTasks: Int = 1,
    val userdata: Any? = null
)

// ============================================================================
// Bitset — port of ggml_bitset_t helpers
// ============================================================================

/** Number of bits to right-shift an index to find its word: log2(32). */
private const val BITSET_SHR = 5
/** Mask for the bit position within a single word (31 = 32-1). */
private const val BITSET_MASK = 31

/**
 * A simple fixed-size bitset backed by an [IntArray].
 *
 * Each `Int` holds 32 bits.  Port of the C `ggml_bitset_t *` pattern in ggml-impl.h.
 */
class GGMLBitset(private val data: IntArray) {

    /** Number of *words* in the backing array. */
    val wordCount: Int get() = data.size

    /** Returns `true` if bit [i] is set. */
    fun get(i: Int): Boolean {
        return (data[i ushr io.github.kotlinmania.llama.ore.BITSET_SHR] and (1 shl (i and io.github.kotlinmania.llama.ore.BITSET_MASK))) != 0
    }

    /** Sets bit [i] to 1. */
    fun set(i: Int) {
        data[i ushr io.github.kotlinmania.llama.ore.BITSET_SHR] = data[i ushr io.github.kotlinmania.llama.ore.BITSET_SHR] or (1 shl (i and io.github.kotlinmania.llama.ore.BITSET_MASK))
    }

    /** Clears bit [i] to 0. */
    fun clear(i: Int) {
        data[i ushr io.github.kotlinmania.llama.ore.BITSET_SHR] = data[i ushr io.github.kotlinmania.llama.ore.BITSET_SHR] and (1 shl (i and io.github.kotlinmania.llama.ore.BITSET_MASK)).inv()
    }

    /** Resets every bit to 0. */
    fun reset() {
        data.fill(0)
    }

    companion object {
        /**
         * Returns the number of 32-bit words needed to hold [n] bits.
         * Equivalent to the C function `ggml_bitset_size`.
         */
        fun wordsForBits(n: Int): Int = (n + io.github.kotlinmania.llama.ore.BITSET_MASK) ushr io.github.kotlinmania.llama.ore.BITSET_SHR

        /** Creates a new bitset large enough to hold [n] bits, all initialized to 0. */
        fun create(n: Int): GGMLBitset = GGMLBitset(IntArray(wordsForBits(n)))
    }
}

// ============================================================================
// Hash set — port of struct ggml_hash_set
// ============================================================================

/** Sentinel: the hash table is completely full. */
const val GGML_HASHSET_FULL: Int = -1
/** Sentinel: the key already exists in the hash table. */
const val GGML_HASHSET_ALREADY_EXISTS: Int = -2

/**
 * Open-addressed hash set keyed by [io.github.kotlinmania.llama.ore.GGMLTensor] identity (reference equality).
 *
 * Port of `struct ggml_hash_set` from ggml-impl.h.
 * Uses linear probing. The slot capacity is stored in [size]; actual occupancy
 * is tracked by the [used] bitset.
 *
 * @property size the number of *slots* in the table (not the number of elements)
 * @property used bitset indicating which slots contain a valid key
 * @property keys the tensor references, valid only where `used.get(i)` is true
 */
class GGMLHashSet(
    val size: Int,
    val used: io.github.kotlinmania.llama.ore.GGMLBitset = io.github.kotlinmania.llama.ore.GGMLBitset.create(size),
    val keys: Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = arrayOfNulls(size)
) {
    /** Removes all elements from the hash set. */
    fun reset() {
        used.reset()
        keys.fill(null)
    }

    companion object {
        /**
         * Returns the minimum table size for a hash set that can hold [minSz] elements.
         *
         * Rounds up to a prime-like size for better distribution.
         * Port of the C function `ggml_hash_size`.
         */
        fun ggml_hash_size(minSz: Int): Int {
            // Simple strategy: next power-of-two × 2 for ~50% load factor
            var sz = maxOf(minSz * 2, 16)
            // Round up to next power of two
            var v = sz - 1
            v = v or (v ushr 1)
            v = v or (v ushr 2)
            v = v or (v ushr 4)
            v = v or (v ushr 8)
            v = v or (v ushr 16)
            return v + 1
        }

        /** Creates a new hash set with capacity for at least [minSz] elements. */
        fun new(minSz: Int): GGMLHashSet {
            val sz = ggml_hash_size(minSz)
            return GGMLHashSet(sz)
        }
    }
}

// ============================================================================
// Hash functions operating on GGMLHashSet
// ============================================================================

/**
 * Hash function for a tensor.
 *
 * In C this is `(uintptr_t)p >> 4`. Since Kotlin doesn't expose raw addresses,
 * we use [System.identityHashCode] (or its equivalent) as a proxy.
 */
fun ggml_hash(p: io.github.kotlinmania.llama.ore.GGMLTensor): Int {
    // identityHashCode is the closest Kotlin analogue to address-based hashing
    return p.hashCode() ushr 4
}

/**
 * Finds the slot for [key] in [hashSet] using linear probing.
 *
 * @return the slot index, or [io.github.kotlinmania.llama.ore.GGML_HASHSET_FULL] if the table is full.
 */
fun ggml_hash_find(hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet, key: io.github.kotlinmania.llama.ore.GGMLTensor): Int {
    val h = (io.github.kotlinmania.llama.ore.ggml_hash(key) and 0x7FFFFFFF) % hashSet.size
    var i = h
    while (hashSet.used.get(i) && hashSet.keys[i] !== key) {
        i = (i + 1) % hashSet.size
        if (i == h) return io.github.kotlinmania.llama.ore.GGML_HASHSET_FULL
    }
    return i
}

/**
 * Returns `true` if [key] is present in [hashSet].
 */
fun ggml_hash_contains(hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet, key: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    val i = io.github.kotlinmania.llama.ore.ggml_hash_find(hashSet, key)
    return i != io.github.kotlinmania.llama.ore.GGML_HASHSET_FULL && hashSet.used.get(i)
}

/**
 * Inserts [key] into [hashSet].
 *
 * @return the slot index, or [io.github.kotlinmania.llama.ore.GGML_HASHSET_ALREADY_EXISTS] if the key was already present.
 * @throws IllegalStateException if the table is completely full.
 */
fun ggml_hash_insert(hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet, key: io.github.kotlinmania.llama.ore.GGMLTensor): Int {
    val h = (io.github.kotlinmania.llama.ore.ggml_hash(key) and 0x7FFFFFFF) % hashSet.size
    var i = h
    do {
        if (!hashSet.used.get(i)) {
            hashSet.used.set(i)
            hashSet.keys[i] = key
            return i
        }
        if (hashSet.keys[i] === key) {
            return io.github.kotlinmania.llama.ore.GGML_HASHSET_ALREADY_EXISTS
        }
        i = (i + 1) % hashSet.size
    } while (i != h)

    error("ggml_hash_insert: hash table is full")
}

/**
 * Finds the slot for [key], inserting it if absent.
 *
 * @return the slot index (whether newly inserted or already present).
 * @throws IllegalStateException if the table is completely full.
 */
fun ggml_hash_find_or_insert(hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet, key: io.github.kotlinmania.llama.ore.GGMLTensor): Int {
    val h = (io.github.kotlinmania.llama.ore.ggml_hash(key) and 0x7FFFFFFF) % hashSet.size
    var i = h
    do {
        if (!hashSet.used.get(i)) {
            hashSet.used.set(i)
            hashSet.keys[i] = key
            return i
        }
        if (hashSet.keys[i] === key) {
            return i
        }
        i = (i + 1) % hashSet.size
    } while (i != h)

    error("ggml_hash_find_or_insert: hash table is full")
}

// ============================================================================
// Computation graph (cgraph) utility helpers
// ============================================================================

/**
 * Returns the use count of the node at [nodeIdx] in [cgraph].
 *
 * Port of `ggml_node_get_use_count` from ggml-impl.h.
 */
fun ggml_node_get_use_count(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, nodeIdx: Int): Int {
    val node = cgraph.nodes[nodeIdx] ?: return 0
    val hashSet = cgraph.visitedHashSet as? io.github.kotlinmania.llama.ore.GGMLHashSet ?: return 0
    val useCounts = cgraph.useCounts ?: return 0

    val hashPos = io.github.kotlinmania.llama.ore.ggml_hash_find(hashSet, node)
    if (hashPos == io.github.kotlinmania.llama.ore.GGML_HASHSET_FULL || !hashSet.used.get(hashPos)) {
        return 0
    }
    return useCounts[hashPos]
}

/**
 * Returns `true` if [nodeIdx]'s results are used by exactly [nUses] other nodes
 * and can be fused into their calculations.
 *
 * Port of `ggml_node_has_n_uses` from ggml-impl.h.
 */
fun ggml_node_has_n_uses(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, nodeIdx: Int, nUses: Int): Boolean {
    val node = cgraph.nodes[nodeIdx] ?: return false

    if (io.github.kotlinmania.llama.ore.ggml_node_get_use_count(cgraph, nodeIdx) != nUses) return false

    // If node is a view, some other node might be using the intermediate result
    // via the view source.
    if (node.viewSrc != null) return false

    // If the user requested output for the node, can't fuse
    if (node.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_OUTPUT != 0) return false

    return true
}

/**
 * Returns `true` if nodes at [nodeIdxs] form a sequence of the given [ops]
 * and are fusable.
 *
 * Port of `ggml_can_fuse_ext` from ggml-impl.h.
 */
fun ggml_can_fuse_ext(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, nodeIdxs: IntArray, ops: Array<io.github.kotlinmania.llama.ore.GGMLOp>, numOps: Int): Boolean {
    for (i in 0 until numOps) {
        if (nodeIdxs[i] >= cgraph.nNodes) return false

        val node = cgraph.nodes[nodeIdxs[i]] ?: return false
        if (node.op != ops[i]) return false
        if (i < numOps - 1 && !io.github.kotlinmania.llama.ore.ggml_node_has_n_uses(
                cgraph,
                nodeIdxs[i],
                1
            )
        ) return false
        if (i > 0) {
            val prev = cgraph.nodes[nodeIdxs[i - 1]] ?: return false
            if (node.src[0] !== prev && node.src[1] !== prev) return false
            // Same shape check
            if (!io.github.kotlinmania.llama.ore.ggml_are_same_shape(node, prev)) return false
        }
    }
    return true
}

/**
 * Returns `true` if sequential nodes starting at [nodeIdx] match [ops] and are fusable.
 *
 * Port of `ggml_can_fuse` from ggml-impl.h.
 */
fun ggml_can_fuse(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, nodeIdx: Int, ops: Array<io.github.kotlinmania.llama.ore.GGMLOp>, numOps: Int): Boolean {
    require(numOps < 32)
    if (nodeIdx + numOps > cgraph.nNodes) return false

    val idxs = IntArray(numOps) { nodeIdx + it }
    return io.github.kotlinmania.llama.ore.ggml_can_fuse_ext(cgraph, idxs, ops, numOps)
}

/**
 * Returns `true` if two tensors have the same shape (element counts in every dimension).
 */
fun ggml_are_same_shape(a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) return false
    }
    return true
}

/**
 * Returns a "view" of [cgraph] containing only nodes [i0, i1).
 * The slice has no leafs or gradients.
 *
 * Port of `ggml_graph_view` from ggml-impl.h.
 */
fun ggml_graph_view(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, i0: Int, i1: Int): io.github.kotlinmania.llama.ore.GGMLCGraph {
    require(i0 in 0..i1 && i1 <= cgraph.nNodes)
    val sliceNodes = cgraph.nodes.sliceArray(i0 until i1)
    return io.github.kotlinmania.llama.ore.GGMLCGraph(
        size = i1 - i0,
        nNodes = i1 - i0,
        nLeafs = 0,
        nodes = sliceNodes,
        grads = emptyArray(),
        leafs = emptyArray(),
        visitedHashSet = cgraph.visitedHashSet,
        order = cgraph.order
    )
}

// ============================================================================
// Global UID counter for computation graphs
// ============================================================================

private var _ggml_graph_uid_counter: Long = 0L

/**
 * Returns the next unique ID for a computation graph.
 * Port of `ggml_graph_next_uid`.
 */
fun ggml_graph_next_uid(): Long = ++io.github.kotlinmania.llama.ore._ggml_graph_uid_counter

// ============================================================================
// Logging level enum
// ============================================================================

/**
 * Logging levels matching GGML's log level constants.
 */
enum class GGMLLogLevel {
    NONE,
    INFO,
    WARN,
    ERROR,
    DEBUG,
    CONT
}

// ============================================================================
// Consolidated numeric utility functions (pre-existing)
// ============================================================================

/**
 * CONSOLIDATION: Numeric utility functions to reduce code duplication
 * These functions consolidate common numeric operations found throughout the codebase
 */

/**
 * Consolidated array conversion function
 * Replaces repeated float-to-half conversion loops
 */
fun convertFloatArrayToHalf(floatArray: FloatArray): ShortArray {
    return ShortArray(floatArray.size) { i -> io.github.kotlinmania.llama.ore.floatToHalf(floatArray[i]) }
}

/**
 * Consolidated array conversion function  
 * Replaces repeated half-to-float conversion loops
 */
fun convertHalfArrayToFloat(halfArray: ShortArray): FloatArray {
    return FloatArray(halfArray.size) { i -> io.github.kotlinmania.llama.ore.halfToFloat(halfArray[i]) }
}

/**
 * Consolidated numeric validation utility
 * Replaces scattered isFinite and range checks
 */
fun validateNumericArray(array: FloatArray, allowInfinite: Boolean = false, allowNaN: Boolean = false): Boolean {
    for (value in array) {
        if (!allowNaN && value.isNaN()) return false
        if (!allowInfinite && value.isInfinite()) return false
    }
    return true
}

/**
 * Consolidated clamping utility
 * Replaces repeated min/max clamping patterns
 */
fun clampFloatArray(array: FloatArray, minValue: Float, maxValue: Float): FloatArray {
    return FloatArray(array.size) { i -> 
        array[i].coerceIn(minValue, maxValue)
    }
}

/**
 * Consolidated numeric precision comparison
 * Replaces inconsistent floating-point comparison patterns
 */
/**
 * Port of `ggml_can_fuse_subgraph_ext` — declared in ggml-impl.h, implemented in ggml.c.
 * Returns true if the subgraph formed by [nodeIdxs] can be fused.
 */
fun ggml_can_fuse_subgraph_ext(
    cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph,
    nodeIdxs: IntArray,
    count: Int,
    ops: Array<io.github.kotlinmania.llama.ore.GGMLOp>,
    outputs: IntArray,
    numOutputs: Int
): Boolean {
    error("ggml_can_fuse_subgraph_ext not yet ported")
}

/**
 * Port of `ggml_can_fuse_subgraph` — convenience wrapper for sequential node indices.
 */
fun ggml_can_fuse_subgraph(
    cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph,
    nodeIdx: Int,
    count: Int,
    ops: Array<io.github.kotlinmania.llama.ore.GGMLOp>,
    outputs: IntArray = intArrayOf(),
    numOutputs: Int = outputs.size
): Boolean {
    require(count < 32) { "count must be < 32" }
    if (nodeIdx + count > cgraph.nNodes) return false
    val idxs = IntArray(count) { nodeIdx + it }
    return io.github.kotlinmania.llama.ore.ggml_can_fuse_subgraph_ext(
        cgraph,
        idxs,
        count,
        ops,
        outputs,
        numOutputs
    )
}

/**
 * Vararg convenience overloads (C++ initializer_list equivalents).
 */
fun ggml_can_fuse(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, nodeIdx: Int, vararg ops: io.github.kotlinmania.llama.ore.GGMLOp): Boolean =
    io.github.kotlinmania.llama.ore.ggml_can_fuse(cgraph, nodeIdx, arrayOf(*ops), ops.size)

fun ggml_can_fuse_subgraph(
    cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph,
    startIdx: Int,
    ops: List<io.github.kotlinmania.llama.ore.GGMLOp>,
    outputs: List<Int> = emptyList()
): Boolean =
    io.github.kotlinmania.llama.ore.ggml_can_fuse_subgraph(
        cgraph,
        startIdx,
        ops.size,
        ops.toTypedArray(),
        outputs.toIntArray(),
        outputs.size
    )

/**
 * Port of `ggml_check_edges` — validates graph edge connectivity.
 * Each edge is [dstNode, srcIdx, srcNode] relative to [startIdx].
 */
fun ggml_check_edges(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, startIdx: Int, edges: List<Triple<Int, Int, Int>>): Boolean {
    for ((dstNode, srcIdx, srcNode) in edges) {
        val dstTensor = cgraph.nodes[startIdx + dstNode] ?: return false
        val srcTensor = cgraph.nodes[startIdx + srcNode] ?: return false
        if (srcIdx >= dstTensor.src.size || dstTensor.src[srcIdx] !== srcTensor) {
            return false
        }
    }
    return true
}

// ============================================================================
// Standalone bitset functions — match C signatures from ggml-impl.h
// ============================================================================

/** Port of `ggml_bitset_size(n)` from ggml-impl.h. */
fun ggml_bitset_size(n: Int): Int = (n + 31) / 32

/** Port of `ggml_bitset_get(bitset, i)` from ggml-impl.h line 209. */
fun ggml_bitset_get(bitset: io.github.kotlinmania.llama.ore.GGMLBitset, i: Int): Boolean = bitset.get(i)

/** Port of `ggml_bitset_set(bitset, i)` from ggml-impl.h line 213. */
fun ggml_bitset_set(bitset: io.github.kotlinmania.llama.ore.GGMLBitset, i: Int) = bitset.set(i)

/** Port of `ggml_bitset_clear(bitset, i)` from ggml-impl.h line 217. */
fun ggml_bitset_clear(bitset: io.github.kotlinmania.llama.ore.GGMLBitset, i: Int) = bitset.clear(i)

// ============================================================================
// Standalone hash set functions — match C signatures from ggml-impl.h
// ============================================================================

/** Port of `ggml_hash_set_new(size)` from ggml-impl.h line 232. */
fun ggml_hash_set_new(size: Int): io.github.kotlinmania.llama.ore.GGMLHashSet = io.github.kotlinmania.llama.ore.GGMLHashSet.new(size)

/** Port of `ggml_hash_set_free(hash_set)` from ggml-impl.h line 233. */
fun ggml_hash_set_free(hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet) {
    // In Kotlin, GC handles deallocation. Reset to release tensor references.
    hashSet.reset()
}

/** Port of `ggml_hash_set_reset(hash_set)` from ggml-impl.h line 239. */
fun ggml_hash_set_reset(hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet) = hashSet.reset()

// ============================================================================
// ggml_op_can_inplace — port from ggml-alloc.c (declared in ggml-impl.h line 355)
// ============================================================================

fun ggml_op_can_inplace(op: io.github.kotlinmania.llama.ore.GGMLOp): Boolean {
    return when (op) {
        io.github.kotlinmania.llama.ore.GGMLOp.FILL,
        io.github.kotlinmania.llama.ore.GGMLOp.SCALE,
        io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_ZERO,
        io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_INF,
        io.github.kotlinmania.llama.ore.GGMLOp.ADD,
        io.github.kotlinmania.llama.ore.GGMLOp.ADD_ID,
        io.github.kotlinmania.llama.ore.GGMLOp.ADD1,
        io.github.kotlinmania.llama.ore.GGMLOp.SUB,
        io.github.kotlinmania.llama.ore.GGMLOp.MUL,
        io.github.kotlinmania.llama.ore.GGMLOp.DIV,
        io.github.kotlinmania.llama.ore.GGMLOp.SQR,
        io.github.kotlinmania.llama.ore.GGMLOp.SQRT,
        io.github.kotlinmania.llama.ore.GGMLOp.LOG,
        io.github.kotlinmania.llama.ore.GGMLOp.UNARY,
        io.github.kotlinmania.llama.ore.GGMLOp.ROPE,
        io.github.kotlinmania.llama.ore.GGMLOp.ROPE_BACK,
        io.github.kotlinmania.llama.ore.GGMLOp.SILU_BACK,
        io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM,
        io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM_BACK,
        io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX,
        io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX_BACK -> true
        else -> false
    }
}

// ============================================================================
// Logging (ggml-impl.h lines 115-123, ggml.c lines 296-308)
// ============================================================================

/** Callback signature for logging. Mirrors `ggml_log_callback`. */
typealias GGMLLogCallback = (level: io.github.kotlinmania.llama.ore.GGMLLogLevel, message: String) -> Unit

/** Default callback: prints to stdout. Port of `ggml_log_callback_default`. */
val ggmlLogCallbackDefault: io.github.kotlinmania.llama.ore.GGMLLogCallback = { _, msg -> print(msg) }

internal var gLogCallback: io.github.kotlinmania.llama.ore.GGMLLogCallback =
    io.github.kotlinmania.llama.ore.ggmlLogCallbackDefault

/** `ggml_log_internal` — C: ggml-impl.h line 115 / ggml.c line 301. */
fun ggmlLogInternal(level: io.github.kotlinmania.llama.ore.GGMLLogLevel, message: String) {
    io.github.kotlinmania.llama.ore.gLogCallback(level, message)
}

// ggmlLogSet moved to GGMLOps.kt (ggml.h line 2729)

/**
 * Port of `ggml_log_callback_default` from ggml.c line 308.
 */
fun ggml_log_callback_default(level: io.github.kotlinmania.llama.ore.GGMLLogLevel, text: String, userData: Any? = null) {
    io.github.kotlinmania.llama.ore.ggmlLogCallbackDefault(level, text)
}

/**
 * Port of `ggml_log_internal` from ggml.c line 301 (snake_case alias).
 */
fun ggml_log_internal(level: io.github.kotlinmania.llama.ore.GGMLLogLevel, text: String) {
    io.github.kotlinmania.llama.ore.ggmlLogInternal(level, text)
}

/** Set the global log callback (snake_case alias). */
fun ggml_log_set(callback: io.github.kotlinmania.llama.ore.GGMLLogCallback?, userData: Any? = null) {
    io.github.kotlinmania.llama.ore.gLogCallback = callback ?: io.github.kotlinmania.llama.ore.ggmlLogCallbackDefault
}

// ============================================================================
// ggml_print_backtrace — port from ggml.c line 152
// ============================================================================

fun ggml_print_backtrace() {
    val trace = Throwable("backtrace").stackTraceToString()
    print(trace)
}

// ============================================================================
// ggml_aligned_malloc / ggml_aligned_free — ggml-impl.h line 360-361
//
// On native targets, these allocate/free truly aligned memory via posix_memalign.
// On jvm/js targets, they fall back to ByteArray (GC-managed, no alignment).
// ============================================================================

expect fun ggml_aligned_malloc(size: Long): Any?

expect fun ggml_aligned_free(ptr: Any?, size: Long)

// ============================================================================
// Platform-specific CPU buffer type factory.
//
// On native targets returns GGMLCpuBufferTypeNative (posix_memalign-backed).
// On jvm/js returns GGMLCpuBufferType (ByteArray-backed).
// ============================================================================

expect fun createDefaultCpuBufferType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType

// ============================================================================
// ggml_up64 — commented out in ggml-impl.h line 64-66, but included for parity
// ============================================================================

fun ggml_up64(n: Int): Int = (n + 63) and 63.inv()

// ---- Utility extensions (not from ggml-impl.h, part of original Kotlin port) ----

fun arraysEqualWithinTolerance(a: FloatArray, b: FloatArray, tolerance: Float = 1e-6f): Boolean {
    if (a.size != b.size) return false
    
    for (i in a.indices) {
        val diff = abs(a[i] - b[i])
        val threshold = tolerance * max(abs(a[i]), abs(b[i])).coerceAtLeast(tolerance)
        if (diff > threshold) return false
    }
    return true
}
