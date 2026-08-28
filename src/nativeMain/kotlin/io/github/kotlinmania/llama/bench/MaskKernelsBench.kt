package io.github.kotlinmania.llama.ench

import kotlin.time.TimeSource

private const val BASE16 = 65536
private const val INV_BASE16 = 1.0 / 65536.0
private val POW2 = IntArray(17).apply { for (i in 0..16) this[i] = if (i == 0) 1 else (1 shl i) }

// Bitwise baseline (left shift). Returns carry-out of last limb.
internal fun shl16Bitwise(a: IntArray, s: Int, carryIn: Int = 0): Int {
    require(s in 1..15)
    val mask16 = 0xFFFF
    val carryMask = POW2[s] - 1
    val sInv = 16 - s
    var carry = carryIn and carryMask
    var i = 0
    while (i < a.size) {
        val cur = a[i] and mask16
        val newVal = ((cur shl s) and mask16) or carry
        a[i] = newVal
        carry = (cur ushr sInv) and carryMask
        i++
    }
    return carry
}

// Arithmetic-only: integer division/remainder form. Returns carry-out of last limb.
internal fun shl16Div(a: IntArray, s: Int, carryIn: Int = 0): Int {
    require(s in 1..15)
    val pow = POW2[s]
    var carry = carryIn
    var i = 0
    while (i < a.size) {
        val t = a[i] * pow + carry
        val q = t / BASE16
        val r = t - q * BASE16
        a[i] = r
        carry = q
        i++
    }
    return carry
}

// Arithmetic-only: floating reciprocal form. Returns carry-out of last limb.
internal fun shl16Recip(a: IntArray, s: Int, carryIn: Int = 0): Int {
    require(s in 1..15)
    val pow = POW2[s]
    var carry = carryIn
    var i = 0
    while (i < a.size) {
        val t = a[i] * pow + carry
        val q = (t * INV_BASE16).toInt()  // exact for t < 2^32
        val r = t - q * BASE16
        a[i] = r
        carry = q
        i++
    }
    return carry
}

private fun checksum16(a: IntArray): Long {
    var s = 0L
    for (v in a) s = (s + (v and 0xFFFF)) and 0x7FFF_FFFF
    return s
}

private fun Double.fx3(): String {
    val v = (this * 1000.0).toLong()
    val i = v / 1000
    val d = (v % 1000).toInt()
    return "$i.${d.toString().padStart(3,'0')}"
}

internal fun runMaskBenchSuite() {
    println("variant,size,shift,iters,ns,GBps,checksum")
    val sizes = listOf(8, 64, 4096, 262144)
    val shifts = listOf(1, 5, 8, 13, 15)
    fun itersFor(size: Int) = when {
        size <= 64 -> 200_000
        size <= 4096 -> 50_000
        else -> 2_000
    }
    for (sz in sizes) for (s in shifts) {
        val iters = itersFor(sz)
        // bitwise
        run {
            val a = IntArray(sz) { (it * 23 + 7) and 0xFFFF }
            var carry = 0
            val t0 = TimeSource.Monotonic.markNow()
            repeat(iters) { carry = shl16Bitwise(a, s, carry) }
            val ns = t0.elapsedNow().inWholeNanoseconds
            val gbps = (sz * 2.0 * iters / 1e9) / (ns.toDouble() / 1e9)
            println("mask-left-bitwise,$sz,$s,$iters,$ns,${gbps.fx3()},${(checksum16(a)+carry) and 0x7FFF_FFFFL}")
        }
        // integer div
        run {
            val a = IntArray(sz) { (it * 23 + 7) and 0xFFFF }
            var carry = 0
            val t0 = TimeSource.Monotonic.markNow()
            repeat(iters) { carry = shl16Div(a, s, carry) }
            val ns = t0.elapsedNow().inWholeNanoseconds
            val gbps = (sz * 2.0 * iters / 1e9) / (ns.toDouble() / 1e9)
            println("mask-left-intdiv,$sz,$s,$iters,$ns,${gbps.fx3()},${(checksum16(a)+carry) and 0x7FFF_FFFFL}")
        }
        // reciprocal
        run {
            val a = IntArray(sz) { (it * 23 + 7) and 0xFFFF }
            var carry = 0
            val t0 = TimeSource.Monotonic.markNow()
            repeat(iters) { carry = shl16Recip(a, s, carry) }
            val ns = t0.elapsedNow().inWholeNanoseconds
            val gbps = (sz * 2.0 * iters / 1e9) / (ns.toDouble() / 1e9)
            println("mask-left-fprecip,$sz,$s,$iters,$ns,${gbps.fx3()},${(checksum16(a)+carry) and 0x7FFF_FFFFL}")
        }
    }
}

