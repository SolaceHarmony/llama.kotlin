package io.github.kotlinmania.llama.lang.int.hpc

/**
 * HPC16x8: 128-bit unsigned integer stored as 8 little-endian 16-bit limbs.
 */
class HPC16x8 private constructor(private val limbs: UShortArray) {
    init { require(limbs.size == 8) }

    fun copy(): HPC16x8 = HPC16x8(limbs.copyOf())

    // Operators for convenience
    operator fun div(den: HPC16x4): HPC16x4 = divRem(den).first
    operator fun rem(den: HPC16x4): HPC16x4 = divRem(den).second
    fun divRem(den: HPC16x4): Pair<HPC16x4, HPC16x4> = div128by64(this, den)

    fun limb(i: Int): UShort = limbs[i]

    fun add(other: HPC16x8): Pair<HPC16x8, UShort> {
        val out = UShortArray(8)
        var carry = 0u
        for (i in 0..7) {
            val sum = limbs[i].toUInt() + other.limbs[i].toUInt() + carry
            out[i] = (sum and 0xFFFFu).toUShort()
            carry = sum shr 16
        }
        return HPC16x8(out) to carry.toUShort()
    }

    fun sub(other: HPC16x8): Pair<HPC16x8, UShort> {
        val out = UShortArray(8)
        var borrow = 0
        for (i in 0..7) {
            var diff = limbs[i].toInt() - other.limbs[i].toInt() - borrow
            if (diff < 0) { diff += 1 shl 16; borrow = 1 } else borrow = 0
            out[i] = (diff and 0xFFFF).toUShort()
        }
        return HPC16x8(out) to borrow.toUShort()
    }

    fun compareTo(other: HPC16x8): Int {
        for (i in 7 downTo 0) {
            val a = limbs[i].toUInt(); val b = other.limbs[i].toUInt()
            if (a != b) return if (a > b) 1 else -1
        }
        return 0
    }

    /** base-2^16 left shift by k bits, 0<=k<16 */
    fun shlBits(k: Int): Pair<HPC16x8, UShort> {
        require(k in 0..15)
        if (k == 0) return this.copy() to 0u
        val buf = io.github.kotlinmania.llama.lang.buffer.LimbBuffer.allocate(8)
        for (i in 0..7) buf.setU16(i, limb(i).toInt() and 0xFFFF)
        val res = io.github.kotlinmania.llama.lang.bitwise.ArrayBitShifts.shl16LEInPlace(buf, 0, 8, k)
        val out = UShortArray(8) { i -> buf.getU16(i).toUShort() }
        return HPC16x8(out) to (res.carryOut and 0xFFFF).toUShort()
    }

    fun shrBits(k: Int): Pair<HPC16x8, UShort> {
        require(k in 0..15)
        if (k == 0) return this.copy() to 0u
        val buf = io.github.kotlinmania.llama.lang.buffer.LimbBuffer.allocate(8)
        for (i in 0..7) buf.setU16(i, limb(i).toInt() and 0xFFFF)
        val res = io.github.kotlinmania.llama.lang.bitwise.ArrayBitShifts.rsh16LEInPlace(buf, 0, 8, k)
        val out = UShortArray(8) { i -> buf.getU16(i).toUShort() }
        return HPC16x8(out) to (res.carryOut and 0xFFFF).toUShort()
    }

    fun shlWords(words: Int): HPC16x8 {
        require(words in 0..8)
        if (words == 0) return copy()
        val out = UShortArray(8)
        for (i in 7 downTo words) out[i] = limbs[i - words]
        for (i in 0 until words) out[i] = 0u
        return HPC16x8(out)
    }

    fun shrWords(words: Int): HPC16x8 {
        require(words in 0..8)
        if (words == 0) return copy()
        val out = UShortArray(8)
        for (i in 0..7 - words) out[i] = limbs[i + words]
        for (i in 8 - words..7) out[i] = 0u
        return HPC16x8(out)
    }

    companion object {
        fun zero(): HPC16x8 = HPC16x8(UShortArray(8) { 0u })

        fun mul64x64To128(a: HPC16x4, b: HPC16x4): HPC16x8 {
            val res = UIntArray(8) { 0u }
            for (i in 0..3) {
                var carry = 0u
                val ai = a.limb(i).toUInt()
                for (j in 0..3) {
                    val idx = i + j
                    val sum = ai * b.limb(j).toUInt() + res[idx] + carry
                    res[idx] = sum and 0xFFFFu
                    carry = sum shr 16
                }
                // propagate remaining carry
                var k = i + 4
                while (carry != 0u && k < 8) {
                    val sum = res[k] + carry
                    res[k] = sum and 0xFFFFu
                    carry = sum shr 16
                    k++
                }
            }
            val out = UShortArray(8) { i -> res[i].toUShort() }
            return HPC16x8(out)
        }

        private fun toULong64(x: HPC16x4): ULong {
            var v = 0uL
            for (i in 3 downTo 0) {
                v = v * 65536u + (x.limb(i).toULong() and 0xFFFFu)
            }
            return v
        }

        fun ofLimbsLE(l0: UShort, l1: UShort, l2: UShort, l3: UShort, l4: UShort, l5: UShort, l6: UShort, l7: UShort): HPC16x8 =
            HPC16x8(ushortArrayOf(l0, l1, l2, l3, l4, l5, l6, l7))

        private fun hiLoULong(x: HPC16x8): Pair<ULong, ULong> {
            var lo = 0uL
            for (i in 3 downTo 0) lo = lo * 65536u + (x.limb(i).toULong() and 0xFFFFu)
            var hi = 0uL
            for (i in 7 downTo 4) hi = hi * 65536u + (x.limb(i).toULong() and 0xFFFFu)
            return hi to lo
        }

        private fun fromULong64(v: ULong): HPC16x4 = HPC16x4.fromULong(v)

        /** 128/64 division (Knuth D, base 2^16). */
        fun div128by64(num: HPC16x8, den: HPC16x4): Pair<HPC16x4, HPC16x4> {
            val U = IntArray(9) { 0 }
            val V = IntArray(4) { 0 }
            for (i in 0..7) U[i] = num.limb(i).toInt() and 0xFFFF
            for (i in 0..3) V[i] = den.limb(i).toInt() and 0xFFFF
            require((V[0] or V[1] or V[2] or V[3]) != 0) { "division by zero" }

            var n = 4
            while (n > 1 && V[n - 1] == 0) n--
            val m = 8 - n

            fun clz16_arith(x0: Int): Int {
                var x = x0 and 0xFFFF
                if (x == 0) return 16
                var s = 0
                while (x < 32768 && s < 15) { x *= 2; s++ }
                return s
            }
            val s = clz16_arith(V[n - 1]).coerceIn(0, 15)
            if (s > 0) {
                val uRes = io.github.kotlinmania.llama.lang.bitwise.ArrayBitShifts.shl16LEInPlace(U, 0, m + n, s)
                io.github.kotlinmania.llama.lang.bitwise.ArrayBitShifts.shl16LEInPlace(V, 0, n, s)
                U[m + n] = uRes.carryOut and 0xFFFF
            }

            val Q = IntArray(m) { 0 }
            val base = 65536
            for (j in (m - 1) downTo 0) {
                val ujn = U[j + n]
                val ujn1 = U[j + n - 1]
                val ujn2 = if (n >= 2) U[j + n - 2] else 0
                val vn1 = V[n - 1]
                val vn2 = if (n >= 2) V[n - 2] else 0

                var numHi = ujn.toLong() * base + ujn1.toLong()
                var qhat = if (vn1 != 0) numHi / vn1.toLong() else (base - 1).toLong()
                var rhat = if (vn1 != 0) numHi % vn1.toLong() else ujn1.toLong()
                if (qhat >= base) { qhat = (base - 1).toLong(); rhat = numHi - qhat * vn1.toLong() }

                while (n >= 2 && qhat * vn2.toLong() > rhat * base + ujn2.toLong()) {
                    qhat -= 1
                    rhat += vn1.toLong()
                    if (rhat >= base) break
                }

                var carry = 0L
                var borrow = 0
                for (i in 0 until n) {
                    val p = qhat * V[i].toLong() + carry
                    carry = p / base
                    val low = (p % base).toInt()
                    var t = U[j + i] - low - borrow
                    if (t < 0) { t += base; borrow = 1 } else borrow = 0
                    U[j + i] = t and 0xFFFF
                }
                var tTop = U[j + n] - carry.toInt() - borrow
                val negative = tTop < 0
                U[j + n] = (tTop % base + base) % base

                if (negative) {
                    qhat -= 1
                    var c = 0
                    for (i in 0 until n) {
                        val sum = U[j + i] + V[i] + c
                        U[j + i] = sum % base
                        c = sum / base
                    }
                    tTop = U[j + n] + c
                    U[j + n] = tTop % base
                }
                Q[j] = (qhat % base).toInt() and 0xFFFF
            }

            val R = IntArray(n) { U[it] }
            if (s > 0) io.github.kotlinmania.llama.lang.bitwise.ArrayBitShifts.rsh16LEInPlace(R, 0, n, s)

            val qOut = UShortArray(4) { i -> ((if (i < m) Q[i] else 0) and 0xFFFF).toUShort() }
            if (m > 4) {
                for (k in 4 until m) require(Q[k] == 0) { "quotient exceeds 64 bits for 128/64 division" }
            }
            val rOut = UShortArray(4) { i -> ((if (i < n) R[i] else 0) and 0xFFFF).toUShort() }
            val qRes = HPC16x4(qOut)
            val rRes = HPC16x4(rOut)
            // Debug recomposition check (throws on mismatch)
            run {
                val prod = mul64x64To128(qRes, den)
                val arr = UShortArray(8) { prod.limb(it) }
                var carryU = 0u
                for (i in 0..3) {
                    val sum = arr[i].toUInt() + rRes.limb(i).toUInt() + carryU
                    arr[i] = (sum and 0xFFFFu).toUShort()
                    carryU = sum shr 16
                }
                var idx = 4
                while (carryU != 0u && idx < 8) {
                    val sum = arr[idx].toUInt() + carryU
                    arr[idx] = (sum and 0xFFFFu).toUShort()
                    carryU = sum shr 16
                    idx++
                }
                var mismatch = false
                for (i in 0..7) if (arr[i] != num.limb(i)) { mismatch = true; break }
                if (mismatch) {
                    fun limb4(a: HPC16x8, i: Int) = a.limb(i).toString(16).padStart(4, '0')
                    val msg = StringBuilder().apply {
                        append("KnuthD recomposition mismatch\n")
                        append("den=0x").append(den.toULong().toString(16)).append('\n')
                        append("q=0x").append(qRes.toULong().toString(16)).append(" r=0x").append(rRes.toULong().toString(16)).append('\n')
                        append("num    = [").append(limb4(num,7)).append(' ').append(limb4(num,6)).append(' ').append(limb4(num,5)).append(' ').append(limb4(num,4)).append(" | ")
                            .append(limb4(num,3)).append(' ').append(limb4(num,2)).append(' ').append(limb4(num,1)).append(' ').append(limb4(num,0)).append("]\n")
                        append("recomp = [").append(arr[7].toString(16).padStart(4,'0')).append(' ').append(arr[6].toString(16).padStart(4,'0')).append(' ').append(arr[5].toString(16).padStart(4,'0')).append(' ').append(arr[4].toString(16).padStart(4,'0')).append(" | ")
                            .append(arr[3].toString(16).padStart(4,'0')).append(' ').append(arr[2].toString(16).padStart(4,'0')).append(' ').append(arr[1].toString(16).padStart(4,'0')).append(' ').append(arr[0].toString(16).padStart(4,'0')).append("]\n")
                    }.toString()
                    throw IllegalStateException(msg)
                }
            }
            return qRes to rRes
        }

        private data class UInt128(var hi: ULong, var lo: ULong) {
            fun isZero(): Boolean = hi == 0uL && lo == 0uL
            fun compare(other: UInt128): Int = when {
                hi != other.hi -> if (hi > other.hi) 1 else -1
                lo != other.lo -> if (lo > other.lo) 1 else -1
                else -> 0
            }
            fun bitLength(): Int {
                if (hi != 0uL) return 128 - hi.countLeadingZeroBits()
                if (lo != 0uL) return 64 - lo.countLeadingZeroBits()
                return 0
            }
            fun shl(bits: Int): UInt128 {
                if (bits <= 0) return copy()
                if (bits >= 128) return UInt128(0u, 0u)
                return when {
                    bits >= 64 -> UInt128(lo shl (bits - 64), 0u)
                    else -> {
                        val newHi = (hi shl bits) or (lo shr (64 - bits))
                        val newLo = lo shl bits
                        UInt128(newHi, newLo)
                    }
                }
            }
            fun shr(bits: Int): UInt128 {
                if (bits <= 0) return copy()
                if (bits >= 128) return UInt128(0u, 0u)
                return when {
                    bits >= 64 -> UInt128(0u, hi shr (bits - 64))
                    else -> {
                        val newLo = (lo shr bits) or (hi shl (64 - bits))
                        val newHi = hi shr bits
                        UInt128(newHi, newLo)
                    }
                }
            }
            fun subtractAssign(other: UInt128) {
                val loTmp = lo
                lo -= other.lo
                val borrow = if (loTmp < other.lo) 1uL else 0uL
                hi = hi - other.hi - borrow
            }
            fun orBit(bit: Int) {
                require(bit in 0..127)
                if (bit >= 64) {
                    hi = hi or (1uL shl (bit - 64))
                } else {
                    lo = lo or (1uL shl bit)
                }
            }
            fun copy(): UInt128 = UInt128(hi, lo)
        }

        private fun HPC16x8.toUInt128(): UInt128 {
            val (hi, lo) = hiLoULong(this)
            return UInt128(hi, lo)
        }

        private fun fromUInt128(value: UInt128): HPC16x8 {
            val limbs = UShortArray(8)
            var tmpLo = value.lo
            for (i in 0..3) {
                limbs[i] = (tmpLo and 0xFFFFu).toUShort()
                tmpLo = tmpLo shr 16
            }
            var tmpHi = value.hi
            for (i in 4..7) {
                limbs[i] = (tmpHi and 0xFFFFu).toUShort()
                tmpHi = tmpHi shr 16
            }
            return HPC16x8(limbs)
        }

        fun mul128(a: HPC16x8, b: HPC16x8): Pair<HPC16x8, HPC16x8> {
            val acc = UIntArray(16) { 0u }
            for (i in 0..7) {
                var carry = 0uL
                val ai = a.limb(i).toULong()
                for (j in 0..7) {
                    val idx = i + j
                    val prod = ai * b.limb(j).toULong() + acc[idx].toULong() + carry
                    acc[idx] = (prod and 0xFFFFu).toUInt()
                    carry = prod shr 16
                }
                var k = i + 8
                while (carry != 0uL && k < 16) {
                    val sum = acc[k].toULong() + carry
                    acc[k] = (sum and 0xFFFFu).toUInt()
                    carry = sum shr 16
                    k++
                }
            }
            val loLimbs = UShortArray(8) { acc[it].toUShort() }
            val hiLimbs = UShortArray(8) { acc[it + 8].toUShort() }
            return HPC16x8(loLimbs) to HPC16x8(hiLimbs)
        }

        fun mulHi128(a: HPC16x8, b: HPC16x8): HPC16x8 = mul128(a, b).second

        fun umul128(a: ULong, b: ULong): Pair<ULong, ULong> {
            val result = mul64x64To128(HPC16x4.fromULong(a), HPC16x4.fromULong(b))
            val (hi, lo) = hiLoULong(result)
            return hi to lo
        }

        fun divmod128(num: HPC16x8, den: HPC16x8): Pair<HPC16x8, HPC16x8> {
            val divisor = den.toUInt128()
            require(!divisor.isZero()) { "division by zero" }
            val remainder = num.toUInt128()
            if (remainder.compare(divisor) < 0) return zero() to num.copy()

            var rem = remainder.copy()
            var shift = remainder.bitLength() - divisor.bitLength()
            var shiftedDiv = divisor.shl(shift)
            var quotient = UInt128(0u, 0u)

            while (shift >= 0) {
                if (rem.compare(shiftedDiv) >= 0) {
                    rem.subtractAssign(shiftedDiv)
                    quotient.orBit(shift)
                }
                shiftedDiv = shiftedDiv.shr(1)
                shift--
            }
            return fromUInt128(quotient) to fromUInt128(rem)
        }

        fun bitLength(value: HPC16x8): Int = value.toUInt128().bitLength()
        // (removed raw clz16; normalization uses arithmetic-only clz)

        // Note: all limb shifts must route through ArrayBitShifts/BitShiftEngine.
    }
}
