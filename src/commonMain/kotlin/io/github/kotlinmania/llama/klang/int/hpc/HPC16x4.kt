package io.github.kotlinmania.llama.lang.int.hpc

import io.github.kotlinmania.llama.lang.bitwise.ArrayBitShifts

/**
 * HPC16x4: 64-bit unsigned integer stored as 4 little-endian 16-bit limbs.
 * Limb 0 is least significant 16 bits.
 */
class HPC16x4 internal constructor(private val limbs: UShortArray) {
    init {
        require(limbs.size == 4)
    }

    fun copy(): HPC16x4 = HPC16x4(limbs.copyOf())

    fun toULong(): ULong {
        var v = 0uL
        for (i in 3 downTo 0) {
            v = (v shl 16) or limbs[i].toULong()
        }
        return v
    }

    fun limb(index: Int): UShort = limbs[index]

    fun compareTo(other: HPC16x4): Int {
        for (i in 3 downTo 0) {
            val a = limbs[i].toUInt()
            val b = other.limbs[i].toUInt()
            if (a != b) return if (a > b) 1 else -1
        }
        return 0
    }

    fun add(other: HPC16x4): Pair<HPC16x4, UShort> {
        val out = UShortArray(4)
        var carry = 0u
        for (i in 0..3) {
            val sum = limbs[i].toUInt() + other.limbs[i].toUInt() + carry
            out[i] = (sum and 0xFFFFu).toUShort()
            carry = (sum shr 16)
        }
        return HPC16x4(out) to carry.toUShort()
    }

    fun sub(other: HPC16x4): Pair<HPC16x4, UShort> {
        val out = UShortArray(4)
        var borrow = 0
        for (i in 0..3) {
            var diff = limbs[i].toInt() - other.limbs[i].toInt() - borrow
            if (diff < 0) {
                diff += 1 shl 16
                borrow = 1
            } else borrow = 0
            out[i] = (diff and 0xFFFF).toUShort()
        }
        return HPC16x4(out) to borrow.toUShort()
    }

    /** left shift by k bits, 0 <= k < 16 */
    fun shlBits(k: Int): Pair<HPC16x4, UShort> {
        require(k in 0..15)
        if (k == 0) return this.copy() to 0u
        val buf = io.github.kotlinmania.llama.lang.buffer.LimbBuffer.allocate(4)
        for (i in 0..3) buf.setU16(i, limbs[i].toInt() and 0xFFFF)
        val res = ArrayBitShifts.shl16LEInPlace(buf, 0, 4, k)
        val arr = UShortArray(4) { i -> buf.getU16(i).toUShort() }
        return HPC16x4(arr) to (res.carryOut and 0xFFFF).toUShort()
    }

    /** right shift by k bits, 0 <= k < 16 */
    fun shrBits(k: Int): Pair<HPC16x4, UShort> {
        require(k in 0..15)
        if (k == 0) return this.copy() to 0u
        val buf = io.github.kotlinmania.llama.lang.buffer.LimbBuffer.allocate(4)
        for (i in 0..3) buf.setU16(i, limbs[i].toInt() and 0xFFFF)
        val res = ArrayBitShifts.rsh16LEInPlace(buf, 0, 4, k)
        val arr = UShortArray(4) { i -> buf.getU16(i).toUShort() }
        return HPC16x4(arr) to (res.carryOut and 0xFFFF).toUShort()
    }

    fun shlWords(words: Int): HPC16x4 {
        require(words in 0..4)
        if (words == 0) return copy()
        val out = UShortArray(4)
        for (i in 3 downTo words) out[i] = limbs[i - words]
        for (i in words - 1 downTo 0) out[i] = 0u
        return HPC16x4(out)
    }

    fun shrWords(words: Int): HPC16x4 {
        require(words in 0..4)
        if (words == 0) return copy()
        val out = UShortArray(4)
        for (i in 0..3 - words) out[i] = limbs[i + words]
        for (i in 4 - words..3) out[i] = 0u
        return HPC16x4(out)
    }

    companion object {
        fun zero(): HPC16x4 = HPC16x4(ushortArrayOf(0u, 0u, 0u, 0u))
        fun ofLimbsLE(l0: UShort, l1: UShort, l2: UShort, l3: UShort): HPC16x4 = HPC16x4(ushortArrayOf(l0,l1,l2,l3))
        fun fromULong(v: ULong): HPC16x4 {
            val a = UShortArray(4)
            var x = v
            for (i in 0..3) {
                a[i] = (x and 0xFFFFu).toUShort()
                x = x shr 16
            }
            return HPC16x4(a)
        }
    }
}
