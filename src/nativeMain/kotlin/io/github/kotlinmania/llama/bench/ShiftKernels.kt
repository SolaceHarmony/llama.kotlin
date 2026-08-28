package io.github.kotlinmania.llama.ench

// Optimized 3-pass (in-place) left shift on 16-bit limbs in IntArray
// Assumptions: 0 < s < 16, a.size > 0
internal class ShiftScratch(size: Int) {
    val low = IntArray(size)
    val high = IntArray(size)
}

internal fun shl16ThreePassInPlace(
    a: IntArray,
    s: Int,
    scratch: ShiftScratch
): Int {
    val len = a.size
    val lowArr = scratch.low
    val highArr = scratch.high
    val mask16 = 0xFFFF
    val sInv = 16 - s
    val highMask = (1 shl s) - 1

    // Pass 1: compute shifted low part and raw high bits
    var i = 0
    while (i < len) {
        val cur = a[i] and mask16
        lowArr[i] = (cur shl s) and mask16
        highArr[i] = (cur ushr sInv) and highMask
        i++
    }
    // Pass 2: link carries forward (merge)
    var carry = 0
    i = 0
    while (i < len) {
        val merged = lowArr[i] or carry
        a[i] = merged
        carry = highArr[i]
        i++
    }
    return carry
}

// Optimized 3-pass (in-place) unsigned right shift on 16-bit limbs in IntArray.
// Preconditions: 0 < s < 16, a.isNotEmpty().
// Returns the dropped low-s bits of limb 0 (before shift).
internal fun rsh16ThreePassInPlace(
    a: IntArray,
    s: Int,
    scratch: ShiftScratch
): Int {
    require(s in 1..15) { "shift must be 1..15 (was $s)" }
    val len = a.size
    if (len == 0) return 0
    val lowArr = scratch.low   // here used to hold hi part (cur ushr s)
    val highArr = scratch.high // here used to hold dropped low bits
    val mask16 = 0xFFFF
    val sInv = 16 - s
    val carryMask = (1 shl s) - 1

    // Pass 1: compute hi part and dropped bits
    var i = 0
    while (i < len) {
        val cur = a[i] and mask16
        lowArr[i] = (cur ushr s) and mask16
        highArr[i] = cur and carryMask
        i++
    }
    val droppedLimb0 = highArr[0]

    // Pass 2: merge from high->low with carry from neighbor's dropped bits
    var carry = 0
    i = len - 1
    while (i >= 0) {
        val merged = lowArr[i] or carry
        a[i] = merged and mask16
        val d = highArr[i]
        carry = (d shl sInv) and mask16
        i--
    }
    return droppedLimb0 and carryMask
}
