package io.github.kotlinmania.llama.ench

import io.github.kotlinmania.llama.lang.stringshift.leftShiftHexString
import io.github.kotlinmania.llama.lang.stringshift.rightShiftHexString
import io.github.kotlinmania.llama.lang.stringshift.limbsToHex
import kotlin.time.TimeSource

private fun checksumHex(s: String): Long {
    var h = 0L
    for (ch in s) h = (h * 131 + ch.code) and 0x7FFF_FFFFL
    return (h + s.length) and 0x7FFF_FFFFL
}

internal fun runHexStringShiftBenchSuite() {
    println("variant,size,shift,iters,ns,GBps,checksum")
    val sizes = listOf(8, 64, 4096, 262144)
    val shifts = listOf(1, 5, 8, 13, 15)
    fun itersFor(size: Int) = when {
        size <= 64 -> 50_000
        size <= 4096 -> 10_000
        else -> 1_000
    }
    for (sz in sizes) for (s in shifts) {
        val limbs = IntArray(sz) { (it * 23 + 7) and 0xFFFF }
        val baseHex = limbsToHex(limbs)
        // left shift (string)
        run {
            val iters = itersFor(sz)
            var acc = 0L
            val t0 = TimeSource.Monotonic.markNow()
            repeat(iters) {
                val out = leftShiftHexString(baseHex, s)
                acc = (acc + checksumHex(out)) and 0x7FFF_FFFFL
            }
            val ns = t0.elapsedNow().inWholeNanoseconds
            // Report logical GBps like limb benches: 2 bytes per limb per iter
            val logicalBytes = sz * 2.0 * iters
            val gbps = (logicalBytes / 1e9) / (ns.toDouble() / 1e9)
            val gbpsFmt = (kotlin.math.round(gbps * 1000.0) / 1000.0)
            println("string-left-hex,$sz,$s,$iters,$ns,${gbpsFmt},$acc")
        }
        // right shift (string)
        run {
            val iters = itersFor(sz)
            var acc = 0L
            val t0 = TimeSource.Monotonic.markNow()
            repeat(iters) {
                val out = rightShiftHexString(baseHex, s)
                acc = (acc + checksumHex(out)) and 0x7FFF_FFFFL
            }
            val ns = t0.elapsedNow().inWholeNanoseconds
            val logicalBytes = sz * 2.0 * iters
            val gbps = (logicalBytes / 1e9) / (ns.toDouble() / 1e9)
            val gbpsFmt = (kotlin.math.round(gbps * 1000.0) / 1000.0)
            println("string-right-hex,$sz,$s,$iters,$ns,${gbpsFmt},$acc")
        }
    }
}
