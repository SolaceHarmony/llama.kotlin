package io.github.kotlinmania.llama.ench

import io.github.kotlinmania.llama.lang.bitwise.SwAR
import kotlin.math.min
import kotlin.time.TimeSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch

private const val DEFAULT_STREAM_WORKERS = 8

internal suspend fun runSwarAvgStreamBench(workerCount: Int = DEFAULT_STREAM_WORKERS) {
    println("variant,size,iters,ns,GBps,checksum,workers")
    val sizes = listOf(8, 64, 4096, 262144)
    fun itersFor(size: Int) = when {
        size <= 64 -> 200_000
        size <= 4096 -> 50_000
        else -> 2_000
    }
    for (sz in sizes) {
        val iters = itersFor(sz)
        val (a0, b0) = genStreamPacks(sz)
        val out = IntArray(sz)
        val dispatcher = Dispatchers.Default
        val t0 = TimeSource.Monotonic.markNow()
        coroutineScope {
            val tileChannel = Channel<Int>(capacity = min(sz, 1024))
            val workerChannels = List(workerCount) { Channel<Int>(capacity = min(sz, 1024)) }

            val workers: List<Job> = workerChannels.map { channel ->
                launch(dispatcher) {
                    for (idx in channel) {
                        out[idx] = SwAR.avgU8TruncArith(a0[idx], b0[idx])
                    }
                }
            }

            val router = launch(dispatcher) {
                var nextWorker = 0
                for (idx in tileChannel) {
                    workerChannels[nextWorker].send(idx)
                    nextWorker = (nextWorker + 1) % workerCount
                }
                workerChannels.forEach { it.close() }
            }

            val producer = launch(dispatcher) {
                repeat(iters) {
                    var idx = 0
                    while (idx < sz) {
                        tileChannel.send(idx)
                        idx++
                    }
                }
                tileChannel.close()
            }

            producer.join()
            router.join()
            workers.joinAll()
        }
        val ns = t0.elapsedNow().inWholeNanoseconds
        val gbps = (sz * STREAM_LOGICAL_BYTES_PER_PACK * iters / 1e9) / (ns.toDouble() / 1e9)
        val checksum = checksumStream(out)
        val gbpsFmt = ((gbps * 1000.0).toLong() / 1000.0)
        println("swar-u8-trunc-stream,$sz,$iters,$ns,$gbpsFmt,$checksum,$workerCount")
    }
}

private const val STREAM_LOGICAL_BYTES_PER_PACK = 8.0

private fun genStreamPacks(size: Int): Pair<IntArray, IntArray> {
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

private fun checksumStream(a: IntArray): Long {
    var s = 0L
    for (v in a) s = (s * 131 + (v and -1)) and 0x7FFF_FFFFL
    return s
}
