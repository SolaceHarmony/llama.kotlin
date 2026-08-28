package io.github.kotlinmania.llama.ench

import io.github.kotlinmania.llama.lang.bitwise.SwAR
import kotlin.time.TimeSource
import kotlinx.coroutines.*

// Serial SWAR bench archived (2025-10-06) under archive/bench/SwARBenchSerial.kt. Only the
// parallel suite remains in active use.

private fun checksumIntArray(a: IntArray): Long {
    var s = 0L
    for (v in a) s = (s * 131 + (v and -1)) and 0x7FFF_FFFFL
    return s
}

private fun genPacks(size: Int): Pair<IntArray, IntArray> {
    val a = IntArray(size) { i ->
        val b0 = (i * 17 + 5) and 0xFF
        val b1 = (i * 29 + 7) and 0xFF
        val b2 = (i * 43 + 11) and 0xFF
        val b3 = (i * 61 + 13) and 0xFF
        b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
    val b = IntArray(size) { i ->
        val b0 = (i * 31 + 3) and 0xFF
        val b1 = (i * 37 + 9) and 0xFF
        val b2 = (i * 19 + 15) and 0xFF
        val b3 = (i * 23 + 21) and 0xFF
        b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
    return a to b
}

private fun emitCsv(
    variant: String,
    sz: Int,
    iters: Int,
    ns: Long,
    checksum: Long,
    bytesPerPackLogical: Double = 8.0,
    bytesPerPackEffective: Double = 12.0,
) {
    val gbps = (sz * bytesPerPackLogical * iters / 1e9) / (ns.toDouble() / 1e9)
    val gbeff = (sz * bytesPerPackEffective * iters / 1e9) / (ns.toDouble() / 1e9)
    val gbpsFmt = ((gbps * 1000.0).toLong() / 1000.0)
    val gbeffFmt = ((gbeff * 1000.0).toLong() / 1000.0)
    println("$variant,$sz,$iters,$ns,$gbpsFmt,$gbeffFmt,$checksum")
}

internal fun runSwarAvgBenchSuiteParallel() {
    println("variant,size,iters,ns,GBps,GBps_eff,checksum")
    val sizes = listOf(8, 64, 4096, 262144)
    fun itersFor(size: Int) = when {
        size <= 64 -> 200_000
        size <= 4096 -> 50_000
        else -> 2_000
    }
    val dispatcher = Dispatchers.Default
    for (sz in sizes) {
        val (a0, b0) = genPacks(sz)
        // arithmetic-only trunc parallel
        run {
            val iters = itersFor(sz)
            val out = IntArray(sz)
            val t0 = TimeSource.Monotonic.markNow()
            runBlocking {
                repeat(iters) {
                    val chunks = 4
                    val chunkSize = (sz + chunks - 1) / chunks
                    coroutineScope {
                        var ck = 0
                        while (ck < chunks) {
                            val start = ck * chunkSize
                            val end = minOf(sz, start + chunkSize)
                            if (start < end) {
                                launch(dispatcher) {
                                    var i = start
                                    while (i < end) {
                                        out[i] = SwAR.avgU8TruncArith(a0[i], b0[i])
                                        i++
                                    }
                                }
                            }
                            ck++
                        }
                    }
                }
            }
            val ns = t0.elapsedNow().inWholeNanoseconds
            emitCsv("swar-u8-trunc-arith-par", sz, iters, ns, checksumIntArray(out))
        }
        // arithmetic-only round parallel
        run {
            val iters = itersFor(sz)
            val out = IntArray(sz)
            val t0 = TimeSource.Monotonic.markNow()
            runBlocking {
                repeat(iters) {
                    val chunks = 4
                    val chunkSize = (sz + chunks - 1) / chunks
                    coroutineScope {
                        var ck = 0
                        while (ck < chunks) {
                            val start = ck * chunkSize
                            val end = minOf(sz, start + chunkSize)
                            if (start < end) {
                                launch(dispatcher) {
                                    var i = start
                                    while (i < end) {
                                        out[i] = SwAR.avgU8RoundArith(a0[i], b0[i])
                                        i++
                                    }
                                }
                            }
                            ck++
                        }
                    }
                }
            }
            val ns = t0.elapsedNow().inWholeNanoseconds
            emitCsv("swar-u8-round-arith-par", sz, iters, ns, checksumIntArray(out))
        }

        // u16 trunc parallel (arith-only)
        run {
            val iters = itersFor(sz)
            val out = IntArray(sz)
            val t0 = TimeSource.Monotonic.markNow()
            runBlocking {
                repeat(iters) {
                    val chunks = 4
                    val chunkSize = (sz + chunks - 1) / chunks
                    coroutineScope {
                        var ck = 0
                        while (ck < chunks) {
                            val start = ck * chunkSize
                            val end = minOf(sz, start + chunkSize)
                            if (start < end) {
                                launch(dispatcher) {
                                    var i = start
                                    while (i < end) {
                                        out[i] = SwAR.avgU16TruncArith(a0[i], b0[i])
                                        i++
                                    }
                                }
                            }
                            ck++
                        }
                    }
                }
            }
            val ns = t0.elapsedNow().inWholeNanoseconds
            emitCsv("swar-u16-trunc-arith-par", sz, iters, ns, checksumIntArray(out))
        }

        // u16 round parallel (arith-only)
        run {
            val iters = itersFor(sz)
            val out = IntArray(sz)
            val t0 = TimeSource.Monotonic.markNow()
            runBlocking {
                repeat(iters) {
                    val chunks = 4
                    val chunkSize = (sz + chunks - 1) / chunks
                    coroutineScope {
                        var ck = 0
                        while (ck < chunks) {
                            val start = ck * chunkSize
                            val end = minOf(sz, start + chunkSize)
                            if (start < end) {
                                launch(dispatcher) {
                                    var i = start
                                    while (i < end) {
                                        out[i] = SwAR.avgU16RoundArith(a0[i], b0[i])
                                        i++
                                    }
                                }
                            }
                            ck++
                        }
                    }
                }
            }
            val ns = t0.elapsedNow().inWholeNanoseconds
            emitCsv("swar-u16-round-arith-par", sz, iters, ns, checksumIntArray(out))
        }
    }
}
