package io.github.kotlinmania.llama.ench

// Engine-based benchmark: use ArrayBitShifts/BitShiftEngine for apples-to-apples comparison.
import io.github.kotlinmania.llama.lang.bitwise.ArrayBitShifts
import kotlin.time.TimeSource
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking

private fun Double.toStringAsFixed3(): String {
    val v = this
    val scaled = (v * 1000.0)
    val rounded = kotlin.math.round(scaled) / 1000.0
    // Manual fixed formatting to 3 decimals
    val s = rounded.toString()
    return if (s.contains('.')) {
        val parts = s.split('.', limit = 2)
        val frac = (parts.getOrNull(1) ?: "")
        parts[0] + "." + (frac + "000").substring(0, 3)
    } else s + ".000"
}
private fun checksumInt(a: IntArray): Long {
    var s = 0L
    for (v in a) s = (s + (v and 0xFFFF)) and 0x7FFF_FFFF
    return s
}

data class BenchCase(val name: String, val size: Int, val shift: Int)

// Engine-based LEFT shift on IntArray (ArrayBitShifts)
private fun benchLeftEngineIntArray(c: BenchCase, iters: Int): Triple<Long, Double, Long> {
    val a = IntArray(c.size) { (it * 23 + 7) and 0xFFFF }
    var carry = 0
    val t0 = TimeSource.Monotonic.markNow()
    repeat(iters) {
        val res = ArrayBitShifts.shl16LEInPlace(a, 0, a.size, c.shift, carry)
        carry = res.carryOut
    }
    val ns = t0.elapsedNow().inWholeNanoseconds
    val logicalBytes = a.size * 2L * iters
    val gbps = (logicalBytes.toDouble() / 1e9) / (ns.toDouble() / 1e9)
    return Triple(ns, gbps, (checksumInt(a) + carry) and 0x7FFF_FFFFL)
}

private fun printCsvHeader() {
    println("variant,size,shift,iters,ns,GBps,checksum")
}

private fun runShiftSuite() {
    val sizes = listOf(8, 64, 4096, 262144)
    val shifts = listOf(1, 5, 8, 13, 15)
    val itersFor = fun(size: Int): Int {
        return when {
            size <= 64 -> 200_000
            size <= 4096 -> 50_000
            else -> 2_000
        }
    }
    printCsvHeader()
    for (sz in sizes) for (s in shifts) {
        val c = BenchCase("shift", sz, s)
        val iters = itersFor(sz)
        val (nsE, gbE, csE) = benchLeftEngineIntArray(c, iters)
        println("engine-left-intarray,${sz},${s},${iters},${nsE},${gbE.toStringAsFixed3()},${csE}")
    }
}

// ===== Right-shift (unsigned) benchmarks =====

// Engine-based RIGHT shift on IntArray (ArrayBitShifts)
private fun benchRightEngineIntArray(c: BenchCase, iters: Int): Triple<Long, Double, Long> {
    val a = IntArray(c.size) { (it * 23 + 7) and 0xFFFF }
    var carry = 0
    val t0 = TimeSource.Monotonic.markNow()
    repeat(iters) {
        val res = ArrayBitShifts.rsh16LEInPlace(a, 0, a.size, c.shift)
        carry = res.carryOut
    }
    val ns = t0.elapsedNow().inWholeNanoseconds
    val logicalBytes = a.size * 2L * iters
    val gbps = (logicalBytes.toDouble() / 1e9) / (ns.toDouble() / 1e9)
    return Triple(ns, gbps, (checksumInt(a) + carry) and 0x7FFF_FFFFL)
}

private fun runRightShiftSuite() {
    val sizes = listOf(8, 64, 4096, 262144)
    val shifts = listOf(1, 5, 8, 13, 15)
    fun itersFor(size: Int): Int = when {
        size <= 64 -> 200_000
        size <= 4096 -> 50_000
        else -> 2_000
    }
    printCsvHeader()
    for (sz in sizes) for (s in shifts) {
        val c = BenchCase("rshift", sz, s)
        val iters = itersFor(sz)
        val (nsE1, gbE1, csE1) = benchRightEngineIntArray(c, iters)
        println("engine-right-intarray,${sz},${s},${iters},${nsE1},${gbE1.toStringAsFixed3()},${csE1}")
    }
}

private fun usage() {
    println("bench usage:")
    println("  --hex            : run hex string shift bench only")
    println("  --engine-left    : run engine-based left shift bench (IntArray)")
    println("  --engine-right   : run engine-based right shift bench (IntArray)")
    println("  --engine-left-par: run parallel engine-based left shift bench")
    println("  --engine-right-par: run parallel engine-based right shift bench")
    println("  --engine-left-native : run engine-based left shift with NATIVE mode")
    println("  --engine-right-native: run engine-based right shift with NATIVE mode")
    println("  --masks          : run arithmetic mask vs bitwise (left shift) bench")
    println("  --swar-avg-par   : run SWAR arithmetic-only (u8) parallel bench")
    println("  --swar-stream    : run SWAR stream/tile channel bench")
    println("  --all            : run all benches in sequence")
}

fun main(args: Array<String>) = runBlocking {
    benchMain(args)
}

private suspend fun benchMain(args: Array<String>) {
    if (args.isEmpty()) {
        usage()
        return
    }
    when (args[0]) {
        "--hex" -> runHexStringShiftBenchSuite()
        "--engine-left" -> runShiftSuite()
        "--engine-right" -> runRightShiftSuite()
        "--engine-left-native" -> runShiftSuiteWithMode(io.github.kotlinmania.llama.lang.bitwise.BitShiftMode.NATIVE)
        "--engine-right-native" -> runRightShiftSuiteWithMode(io.github.kotlinmania.llama.lang.bitwise.BitShiftMode.NATIVE)
        "--engine-left-par" -> runShiftSuiteParallel()
        "--engine-right-par" -> runRightShiftSuiteParallel()
        "--masks" -> runMaskBenchSuite()
        "--swar-avg-par" -> runSwarAvgBenchSuiteParallel()
        "--swar-stream" -> runSwarAvgStreamBench()
        "--all" -> {
            runHexStringShiftBenchSuite()
            runShiftSuite()
            runRightShiftSuite()
            runMaskBenchSuite()
            runSwarAvgBenchSuiteParallel()
        }
        else -> usage()
    }
}

private fun runShiftSuiteWithMode(mode: io.github.kotlinmania.llama.lang.bitwise.BitShiftMode) {
    io.github.kotlinmania.llama.lang.bitwise.BitShiftConfig.withMode(mode) { runShiftSuite() }
}

private fun runRightShiftSuiteWithMode(mode: io.github.kotlinmania.llama.lang.bitwise.BitShiftMode) {
    io.github.kotlinmania.llama.lang.bitwise.BitShiftConfig.withMode(mode) { runRightShiftSuite() }
}

// ===== Parallel variants =====

private suspend fun benchLeftEngineIntArrayParallel(c: BenchCase, iters: Int): Triple<Long, Double, Long> {
    val a = IntArray(c.size) { (it * 23 + 7) and 0xFFFF }
    var carry = 0
    val t0 = TimeSource.Monotonic.markNow()
    repeat(iters) {
        val res = ArrayBitShifts.shl16LEInPlaceParallel(a, 0, a.size, c.shift, carry)
        carry = res.carryOut
    }
    val ns = t0.elapsedNow().inWholeNanoseconds
    val logicalBytes = a.size * 2L * iters
    val gbps = (logicalBytes.toDouble() / 1e9) / (ns.toDouble() / 1e9)
    return Triple(ns, gbps, (checksumInt(a) + carry) and 0x7FFF_FFFFL)
}

private suspend fun runShiftSuiteParallel() {
    val sizes = listOf(8, 64, 4096, 262144)
    val shifts = listOf(1, 5, 8, 13, 15)
    fun itersFor(size: Int): Int = when {
        size <= 64 -> 200_000
        size <= 4096 -> 50_000
        else -> 2_000
    }
    printCsvHeader()
    for (sz in sizes) for (s in shifts) {
        val c = BenchCase("shift-par", sz, s)
        val iters = itersFor(sz)
        val (nsE, gbE, csE) = benchLeftEngineIntArrayParallel(c, iters)
        println("engine-left-par-intarray,${sz},${s},${iters},${nsE},${gbE.toStringAsFixed3()},${csE}")
    }
}

private suspend fun benchRightEngineIntArrayParallel(c: BenchCase, iters: Int): Triple<Long, Double, Long> {
    val a = IntArray(c.size) { (it * 23 + 7) and 0xFFFF }
    var carry = 0
    val t0 = TimeSource.Monotonic.markNow()
    repeat(iters) {
        val res = ArrayBitShifts.rsh16LEInPlaceParallel(a, 0, a.size, c.shift)
        carry = res.carryOut
    }
    val ns = t0.elapsedNow().inWholeNanoseconds
    val logicalBytes = a.size * 2L * iters
    val gbps = (logicalBytes.toDouble() / 1e9) / (ns.toDouble() / 1e9)
    return Triple(ns, gbps, (checksumInt(a) + carry) and 0x7FFF_FFFFL)
}

private suspend fun runRightShiftSuiteParallel() {
    val sizes = listOf(8, 64, 4096, 262144)
    val shifts = listOf(1, 5, 8, 13, 15)
    fun itersFor(size: Int): Int = when {
        size <= 64 -> 200_000
        size <= 4096 -> 50_000
        else -> 2_000
    }
    printCsvHeader()
    for (sz in sizes) for (s in shifts) {
        val c = BenchCase("rshift-par", sz, s)
        val iters = itersFor(sz)
        val (nsE1, gbE1, csE1) = benchRightEngineIntArrayParallel(c, iters)
        println("engine-right-par-intarray,${sz},${s},${iters},${nsE1},${gbE1.toStringAsFixed3()},${csE1}")
    }
}
