package ai.solace.llamakotlin.core

// Half/Float conversion routines.

/**
 * Converts a 16-bit half-precision float to a 32-bit single-precision float.
 * Based on public domain code by Fabian "ryg" Giesen and others.
 */
internal fun halfToFloat(h_bits: Short): Float {
    val h = h_bits.toInt() and 0xFFFF // Ensure we're working with 16 bits, unsigned

    val signMaskF32 = 0x80000000
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
    if (absF > 0x47FFEFFFU) { // F32 value is too large for F16 normal => F16 Inf or NaN
        // NaN if F32 mantissa is non-zero, else Inf
        val mantissaIsNonZero = (absF and 0x007FFFFFU) != 0
        // F16 Inf/NaN: exp all 1s (0x1F for 5 bits -> 0x7C00 when shifted)
        // For NaN, set MSB of mantissa (e.g., 0x0200 for F16) or any non-zero pattern
        return (fSign or 0x7C00 or if (mantissaIsNonZero) 0x0200 else 0).toShort()
    }

    if (absF < 0x38800000U) { // F32 value is too small for F16 normal => F16 denormal or zero
        // Convert F32 to F16 denormal
        // Add implicit F32 '1' bit to mantissa
        val fMant = (absF and 0x007FFFFFU) or 0x00800000U
        // Calculate F16 denormal shift amount
        // F32 exp is (absF ushr 23). Denormal F16 exp is effectively -14.
        // Shift needed = 24 (F32 mant+implicit_1 bits) - (10 (F16 mant bits) + ( (absF ushr 23) - 127 (F32 bias) - (-14 (F16 denorm_exp)) ) )
        // shift = 24 - (10 + ( (absF ushr 23) - 113) ) = 24 - 10 - (absF ushr 23) + 113 = 127 - (absF ushr 23)
        val shift = 127 - (absF ushr 23) // Number of positions to shift right to align for F16 denormal mantissa

        val hMant = if (shift < 24) (fMant ushr shift) else 0

        // Rounding (RTNE for denormals requires checking bits shifted out)
        val roundBits = fMant and ((1U shl shift) - 1U) // Bits lost
        // Tie-breaking: if exactly halfway, round to even (LSB of hMant is 0 after rounding)
        // Threshold for rounding up is halfway mark (1 << (shift - 1))
        if (roundBits > (1U shl (shift - 1)) || (roundBits == (1U shl (shift - 1)) && (hMant and 1) != 0)) {
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
    var hMant = (absF and 0x007FFFFFU) ushr 13 // Shifted F16 mantissa

    // Rounding for normalized numbers (RTNE)
    // Check the MSB of the bits that were shifted out (the rounding bit)
    if ((absF and 0x00001000U) != 0U) { // If rounding bit is 1 (0x1000 is 2^12, MSB of 13 shifted bits)
        // Check for tie-breaking (if remaining shifted bits are zero AND LSB of hMant is 1)
        if ((absF and 0x00000FFFU) != 0U || (hMant and 1U) != 0U) {
            hMant++
            if (hMant == 0x0400U) { // Mantissa overflowed to 1024
                hMant = 0U // Reset mantissa
                // Increment exponent (already shifted)
                return (fSign or (hExp + (1 shl 10)) or hMant).toShort()
            }
        }
    }
    return (fSign or hExp or hMant).toShort()
}
