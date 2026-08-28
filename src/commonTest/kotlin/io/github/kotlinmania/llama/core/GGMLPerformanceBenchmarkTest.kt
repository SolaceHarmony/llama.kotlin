package io.github.kotlinmania.llama.core

import kotlin.math.sqrt
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource

/**
 * Performance benchmarking suite for GGML operations using the destination-based API.
 */
class GGMLPerformanceBenchmarkTest {

    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    private val bufferSize = 16 * 1024 * 1024 // 16 MB shared test buffer
    private val context = io.github.kotlinmania.llama.ore.GGMLContext()

    data class BenchmarkResult(
        val operationName: String,
        val dataSize: Int,
        val timeMillis: Long,
        val throughputMBps: Double,
        val operationsPerSecond: Double
    )

    @BeforeTest
    fun setup() {
        val (allocator, _) = GGMLTestUtils.createTestAllocator(bufferSize)
        graphAllocator = allocator
        resetAllocatorTracking(graphAllocator)
    }

    private fun allocateVector(
        name: String,
        type: io.github.kotlinmania.llama.ore.GGMLType,
        size: Int,
        initializer: (Int) -> Float
    ): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor = GGMLTestUtils.allocateDestinationTensor(
            graphAllocator = graphAllocator,
            name = name,
            type = type,
            shape = longArrayOf(size.toLong())
        )

        when (type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> repeat(size) { idx -> tensor.setFloat(graphAllocator, initializer(idx), idx) }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> repeat(size) { idx -> tensor.setHalf(graphAllocator, initializer(idx), idx) }
            else -> throw IllegalArgumentException("Unsupported vector type $type for benchmark")
        }
        return tensor
    }

    private fun allocateMatrix(
        name: String,
        rows: Int,
        cols: Int,
        initializer: (Int, Int) -> Float
    ): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor = GGMLTestUtils.allocateDestinationTensor(
            graphAllocator = graphAllocator,
            name = name,
            type = io.github.kotlinmania.llama.ore.GGMLType.F32,
            shape = longArrayOf(cols.toLong(), rows.toLong())
        )

        for (r in 0 until rows) {
            for (c in 0 until cols) {
                tensor.setFloat(graphAllocator, initializer(r, c), c, r)
            }
        }
        return tensor
    }

    private fun allocateDestination(
        reference: io.github.kotlinmania.llama.ore.GGMLTensor,
        name: String = "dst_${reference.name}",
        type: io.github.kotlinmania.llama.ore.GGMLType = reference.type
    ): io.github.kotlinmania.llama.ore.GGMLTensor {
        return GGMLTestUtils.allocateDestinationTensor(
            graphAllocator = graphAllocator,
            name = name,
            type = type,
            shape = reference.ne.copyOf()
        )
    }

    private fun benchmarkOperation(
        operationName: String,
        dataSize: Int,
        warmupRuns: Int = 3,
        measureRuns: Int = 10,
        block: () -> Unit
    ): BenchmarkResult {
        val perf = GGMLTestUtils.PerformanceUtils.benchmark(
            operationName = operationName,
            dataSize = dataSize,
            warmupRuns = warmupRuns,
            measureRuns = measureRuns,
            operation = block
        )
        return BenchmarkResult(
            operationName = perf.operationName,
            dataSize = perf.dataSize,
            timeMillis = perf.avgTimeMillis,
            throughputMBps = perf.throughputMBps,
            operationsPerSecond = perf.operationsPerSecond
        )
    }

    @Test
    fun benchmarkElementWiseOperationsF32() {
        val sizes = arrayOf(1024, 4096, 16384)
        val results = mutableListOf<BenchmarkResult>()

        for (size in sizes) {
            resetAllocatorTracking(graphAllocator)
            val tensorA = allocateVector("ew_a_$size", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 17) / 17.0f }
            val tensorB = allocateVector("ew_b_$size", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 13 - 6).toFloat() }
            val addDst = allocateDestination(tensorA, name = "add_dst_$size")
            val mulDst = allocateDestination(tensorA, name = "mul_dst_$size")
            val subDst = allocateDestination(tensorA, name = "sub_dst_$size")
            val divDst = allocateDestination(tensorA, name = "div_dst_$size")

            results += benchmarkOperation("ADD_F32", size) {
                io.github.kotlinmania.llama.ore.computeAdd(graphAllocator, tensorA, tensorB, addDst)
            }
            results += benchmarkOperation("MUL_F32", size) {
                io.github.kotlinmania.llama.ore.computeMul(graphAllocator, tensorA, tensorB, mulDst)
            }
            results += benchmarkOperation("SUB_F32", size) {
                io.github.kotlinmania.llama.ore.computeSub(graphAllocator, tensorA, tensorB, subDst)
            }
            results += benchmarkOperation("DIV_F32", size) {
                io.github.kotlinmania.llama.ore.computeDiv(graphAllocator, tensorA, tensorB, divDst)
            }
        }

        println("\n=== F32 Element-wise Operations ===")
        println("Operation\tElements\tTime(ms)\tThroughput(MB/s)")
        for (result in results) {
            println("${result.operationName}\t${result.dataSize}\t${result.timeMillis}\t${io.github.kotlinmania.llama.ore.GGMLUtilities.formatDouble(result.throughputMBps)}")
        }

        results.filter { it.timeMillis < Long.MAX_VALUE }.forEach {
            assertTrue(it.timeMillis < 2_000, "${it.operationName} took too long: ${it.timeMillis}ms")
        }
    }

    @Test
    fun benchmarkUnaryOperations() {
        val size = 16_384
        resetAllocatorTracking(graphAllocator)
        val source = allocateVector("unary_src", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 11 - 5).toFloat() }
        val negDst = allocateDestination(source, "neg_dst")
        val sqrDst = allocateDestination(source, "sqr_dst")
        val sqrtDst = allocateDestination(source, "sqrt_dst")
        val reluDst = allocateDestination(source, "relu_dst")
        val geluDst = allocateDestination(source, "gelu_dst")

        val operations = listOf(
            Triple("NEG_F32", negDst) { dst: io.github.kotlinmania.llama.ore.GGMLTensor ->
                io.github.kotlinmania.llama.ore.computeNeg(
                    graphAllocator,
                    source,
                    dst
                )
            },
            Triple("SQR_F32", sqrDst) { dst: io.github.kotlinmania.llama.ore.GGMLTensor ->
                io.github.kotlinmania.llama.ore.computeSqr(
                    graphAllocator,
                    source,
                    dst
                )
            },
            Triple("SQRT_F32", sqrtDst) { dst: io.github.kotlinmania.llama.ore.GGMLTensor ->
                io.github.kotlinmania.llama.ore.computeSqrt(
                    graphAllocator,
                    source,
                    dst
                )
            },
            Triple("RELU_F32", reluDst) { dst: io.github.kotlinmania.llama.ore.GGMLTensor ->
                io.github.kotlinmania.llama.ore.computeRelu(
                    graphAllocator,
                    source,
                    dst
                )
            },
            Triple("GELU_F32", geluDst) { dst: io.github.kotlinmania.llama.ore.GGMLTensor ->
                io.github.kotlinmania.llama.ore.computeGelu(
                    graphAllocator,
                    source,
                    dst
                )
            }
        )

        val results = operations.map { (name, dst, op) ->
            benchmarkOperation(name, size) { op(dst) }
        }

        println("\n=== Unary Operations (size=$size) ===")
        println("Operation\tTime(ms)\tOps/sec")
        results.forEach {
            println("${it.operationName}\t${it.timeMillis}\t${io.github.kotlinmania.llama.ore.GGMLUtilities.formatDouble(it.operationsPerSecond)}")
        }
    }

    @Test
    fun benchmarkMatrixMultiplication() {
        val sizes = arrayOf(64, 128, 256)
        val results = mutableListOf<BenchmarkResult>()

        for (n in sizes) {
            resetAllocatorTracking(graphAllocator)
            val matrixA = allocateMatrix("matA_$n", n, n) { r, c -> (r + c).toFloat() * 0.01f }
            val matrixB = allocateMatrix("matB_$n", n, n) { r, c -> (r - c).toFloat() * 0.01f }
            val dst = allocateDestination(matrixA, name = "matmul_dst_$n")

            val elements = n * n
            results += benchmarkOperation("MATMUL_${n}x$n", elements, warmupRuns = 1, measureRuns = 3) {
                io.github.kotlinmania.llama.ore.computeMatMul(graphAllocator, matrixA, matrixB, dst)
            }
        }

        println("\n=== Matrix Multiplication ===")
        println("Size\tTime(ms)\tGFLOPs")
        results.forEach { result ->
            val dim = sqrt(result.dataSize.toDouble()).toInt()
            val flops = 2.0 * dim * dim * dim
            val gflops = if (result.timeMillis > 0) (flops / (result.timeMillis / 1000.0)) / 1e9 else 0.0
            println("${dim}x$dim\t${result.timeMillis}\t${io.github.kotlinmania.llama.ore.GGMLUtilities.formatDouble(gflops)}")
        }
    }

    @Test
    fun benchmarkQuantizationOperations() {
        val sizes = arrayOf(1024, 4096, 16_384)
        val results = mutableListOf<BenchmarkResult>()

        for (size in sizes) {
            resetAllocatorTracking(graphAllocator)
            val source = allocateVector("quant_src_$size", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 31 - 15).toFloat() / 10f }
            results += benchmarkOperation("QUANT_Q8_0", size) {
                io.github.kotlinmania.llama.ore.quantizeTensor(
                    graphAllocator,
                    source,
                    io.github.kotlinmania.llama.ore.GGMLType.Q8_0
                )
            }
            results += benchmarkOperation("QUANT_Q4_0", size) {
                io.github.kotlinmania.llama.ore.quantizeTensor(
                    graphAllocator,
                    source,
                    io.github.kotlinmania.llama.ore.GGMLType.Q4_0
                )
            }
            results += benchmarkOperation("QUANT_Q4_1", size) {
                io.github.kotlinmania.llama.ore.quantizeTensor(
                    graphAllocator,
                    source,
                    io.github.kotlinmania.llama.ore.GGMLType.Q4_1
                )
            }
        }

        println("\n=== Quantization Benchmarks ===")
        println("Operation\tSize\tTime(ms)")
        results.forEach { println("${it.operationName}\t${it.dataSize}\t${it.timeMillis}") }
    }

    @Test
    fun benchmarkDequantizationOperations() {
        val size = 8192
        resetAllocatorTracking(graphAllocator)
        val source = allocateVector("dequant_src", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 29 - 14).toFloat() / 8f }
        val q8 = io.github.kotlinmania.llama.ore.quantizeTensor(
            graphAllocator,
            source,
            io.github.kotlinmania.llama.ore.GGMLType.Q8_0
        )
        val q4 = io.github.kotlinmania.llama.ore.quantizeTensor(
            graphAllocator,
            source,
            io.github.kotlinmania.llama.ore.GGMLType.Q4_0
        )
        val q41 = io.github.kotlinmania.llama.ore.quantizeTensor(
            graphAllocator,
            source,
            io.github.kotlinmania.llama.ore.GGMLType.Q4_1
        )

        val results = listOf(
            benchmarkOperation("DEQUANT_Q8_0", size) {
                io.github.kotlinmania.llama.ore.dequantizeTensor(
                    graphAllocator,
                    q8
                )
            },
            benchmarkOperation("DEQUANT_Q4_0", size) {
                io.github.kotlinmania.llama.ore.dequantizeTensor(
                    graphAllocator,
                    q4
                )
            },
            benchmarkOperation("DEQUANT_Q4_1", size) {
                io.github.kotlinmania.llama.ore.dequantizeTensor(
                    graphAllocator,
                    q41
                )
            }
        )

        println("\n=== Dequantization Benchmarks (size=$size) ===")
        println("Operation\tTime(ms)\tThroughput(MB/s)")
        results.forEach {
            println("${it.operationName}\t${it.timeMillis}\t${io.github.kotlinmania.llama.ore.GGMLUtilities.formatDouble(it.throughputMBps)}")
        }
    }

    @Test
    @OptIn(ExperimentalTime::class)
    fun benchmarkMemoryAllocation() {
        val sizes = arrayOf(1024, 4096, 16_384, 65_536)
        println("\n=== Memory Allocation Benchmark ===")
        println("Elements\tAllocation Time(ms)\tTensors/sec")

        for (size in sizes) {
            resetAllocatorTracking(graphAllocator)
            val tensorCount = 64
            val mark = TimeSource.Monotonic.markNow()
            repeat(tensorCount) { index ->
                allocateVector("alloc_${size}_$index", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 5).toFloat() }
            }
            val elapsed = mark.elapsedNow().inWholeMilliseconds
            val tensorsPerSec = if (elapsed > 0) (tensorCount * 1000.0) / elapsed else 0.0
            println("$size\t$elapsed\t${io.github.kotlinmania.llama.ore.GGMLUtilities.formatDouble(tensorsPerSec)}")
            assertTrue(elapsed < 5_000L, "Allocating $tensorCount tensors of size $size took too long: ${elapsed}ms")
        }
    }

    @Test
    fun performanceSummaryReport() {
        val size = 4096
        resetAllocatorTracking(graphAllocator)
        val tensorA = allocateVector("summary_a", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 19).toFloat() / 5f }
        val tensorB = allocateVector("summary_b", io.github.kotlinmania.llama.ore.GGMLType.F32, size) { idx -> (idx % 23 - 11).toFloat() / 7f }
        val dst = allocateDestination(tensorA, name = "summary_dst")

        val operations = mapOf<String, (io.github.kotlinmania.llama.ore.GGMLTensor) -> Unit>(
            "ADD" to {
                io.github.kotlinmania.llama.ore.computeAdd(
                    graphAllocator,
                    tensorA,
                    tensorB,
                    dst
                )
            },
            "MUL" to {
                io.github.kotlinmania.llama.ore.computeMul(
                    graphAllocator,
                    tensorA,
                    tensorB,
                    dst
                )
            },
            "SUB" to {
                io.github.kotlinmania.llama.ore.computeSub(
                    graphAllocator,
                    tensorA,
                    tensorB,
                    dst
                )
            },
            "DIV" to {
                io.github.kotlinmania.llama.ore.computeDiv(
                    graphAllocator,
                    tensorA,
                    tensorB,
                    dst
                )
            },
            "NEG" to { io.github.kotlinmania.llama.ore.computeNeg(graphAllocator, tensorA, dst) }
        )

        println("\n${"=".repeat(60)}")
        println("GGML KOTLIN PERFORMANCE SUMMARY")
        println("${"=".repeat(60)}")
        println("Operation\tTime(ms)\tThroughput(MB/s)")

        operations.forEach { (name, op) ->
            val result = benchmarkOperation(name, size, warmupRuns = 2, measureRuns = 5) { op(dst) }
            println("$name\t${result.timeMillis}\t${io.github.kotlinmania.llama.ore.GGMLUtilities.formatDouble(result.throughputMBps)}")
        }
        println("Summary complete.")
    }
}
