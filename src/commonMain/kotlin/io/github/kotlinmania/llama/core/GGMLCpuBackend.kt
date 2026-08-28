// port-lint: source ggml/src/ggml-cpu/ggml-cpu.cpp
package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setShortLe

/**
 * CPU backend implementation for GGML operations.
 *
 * Ported from `ggml-cpu.cpp`. This file contains:
 * 1. CPU buffer type implementation (alloc, get_name, get_alignment, etc.)
 * 2. CPU backend implementation (get_name, graph_plan, graph_compute, supports_op, etc.)
 * 3. CPU device implementation (get_name, get_type, get_memory, get_props)
 * 4. CPU backend registration (ggml_backend_cpu_reg)
 * 5. Graph plan and compute functions
 * 6. Thread management functions
 */

// ============================================================================
// Extra buffer type helpers
// ============================================================================

/**
 * Returns the list of extra buffer types supported by the CPU backend.
 *
 * In the C++ original this collects AMX, RISC-V SpaceMIT, KleidiAI and repack
 * buffer types. In the Kotlin port no hardware-specific extra buffer types
 * exist yet, so the list starts empty.
 *
 * Mirrors `ggml_backend_cpu_get_extra_buffer_types()`.
 */
fun ggmlBackendCpuGetExtraBufferTypes(): MutableList<io.github.kotlinmania.llama.ore.GGMLBackendBufferType> {
    // Empty — add extra buffer types here when accelerator extensions are ported
    return mutableListOf()
}

/**
 * Returns the extra buffer types array with a trailing `null` sentinel,
 * suitable for device queries.
 *
 * Mirrors `ggml_backend_cpu_device_get_extra_buffers_type()`.
 */
private fun ggmlBackendCpuDeviceGetExtraBufferTypes(): List<io.github.kotlinmania.llama.ore.GGMLBackendBufferType> {
    return io.github.kotlinmania.llama.ore.ggmlBackendCpuGetExtraBufferTypes()
}

/**
 * Returns `true` when [buft] is one of the extra buffer types registered
 * for the CPU backend.
 *
 * Mirrors `ggml_backend_cpu_is_extra_buffer_type()`.
 */
fun ggmlBackendCpuIsExtraBufferType(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean {
    return io.github.kotlinmania.llama.ore.ggmlBackendCpuGetExtraBufferTypes().any { it === buft }
}

// ============================================================================
// CPU buffer type  (ggml_backend_cpu_buffer_type)
// ============================================================================

/**
 * CPU backend buffer type.
 *
 * Mirrors the buffer-type vtable wired into `ggml_backend_cpu_buffer_type()`.
 */
class GGMLCpuBufferType : io.github.kotlinmania.llama.ore.GGMLBackendBufferType {
    companion object {
        private const val TENSOR_ALIGNMENT = 32u // 32-byte alignment as in C++ implementation
        private const val MAX_BUFFER_SIZE = ULong.MAX_VALUE // No practical limit for CPU
    }

    override fun getName(): String = "CPU"

    override fun allocBuffer(size: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBuffer? {
        if (size == 0uL) return null

        return try {
            // Allocate with extra space for alignment
            val alignedSize = (size + TENSOR_ALIGNMENT).toInt()
            val data = ByteArray(alignedSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLCpuBuffer(this, data, size)
        } catch (t: Throwable) {
            println("GGMLCpuBufferType: Failed to allocate buffer of size $size (${t.message})")
            null
        }
    }

    override fun getAlignment(): UInt = TENSOR_ALIGNMENT

    override fun getMaxSize(): ULong = MAX_BUFFER_SIZE

    override fun isHost(): Boolean = true
}

// ============================================================================
// CPU buffer  (ggml_backend_cpu_buffer)
// ============================================================================

/**
 * CPU backend buffer implementation using ByteArray.
 *
 * Mirrors the buffer vtable wired into `ggml_backend_cpu_buffer_type_alloc_buffer`.
 */
class GGMLCpuBuffer(
    private val bufferType: io.github.kotlinmania.llama.ore.GGMLCpuBufferType,
    private val data: ByteArray,
    private val size: ULong
) : io.github.kotlinmania.llama.ore.GGMLBackendBuffer {

    override fun getType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType = bufferType

    override fun getName(): String = "CPU"

    override fun getBase(): Any = data

    override fun getSize(): ULong = size

    override fun free() {
        // ByteArray will be garbage collected, nothing to do explicitly
    }

    override fun initTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLStatus {
        // CPU tensors don't need special initialization
        tensor.buffer = this
        return io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS
    }

    override fun setTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val tensorData = getBase() as ByteArray
        val srcStart = offset.toInt()
        val srcEnd = (offset + size).toInt()
        val dstStart = tensor.dataOffset.toInt()

        if (srcEnd > data.size) {
            throw IndexOutOfBoundsException("Source data out of bounds: $srcEnd > ${data.size}")
        }

        if (dstStart + size.toInt() > tensorData.size) {
            throw IndexOutOfBoundsException("Destination buffer out of bounds: ${dstStart + size.toInt()} > ${tensorData.size}")
        }

        data.copyInto(tensorData, dstStart, srcStart, srcEnd)
    }

    override fun getTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val tensorData = getBase() as ByteArray
        val srcStart = tensor.dataOffset.toInt()
        val srcEnd = srcStart + size.toInt()
        val dstStart = offset.toInt()

        if (srcEnd > tensorData.size) {
            throw IndexOutOfBoundsException("Source buffer out of bounds: $srcEnd > ${tensorData.size}")
        }

        if (dstStart + size.toInt() > data.size) {
            throw IndexOutOfBoundsException("Destination data out of bounds: ${dstStart + size.toInt()} > ${data.size}")
        }

        tensorData.copyInto(data, dstStart, srcStart, srcEnd)
    }

    override fun copyTensor(src: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
        // Check if source is from a host buffer (CPU or compatible)
        val srcBuffer = src.buffer
        if (srcBuffer == null || !srcBuffer.getType().isHost()) {
            return false
        }

        val srcData = srcBuffer.getBase() as? ByteArray ?: return false
        val dstData = getBase() as ByteArray

        val srcByteSize = io.github.kotlinmania.llama.ore.calculateTensorByteSize(src)
        val dstByteSize = io.github.kotlinmania.llama.ore.calculateTensorByteSize(dst)

        if (srcByteSize != dstByteSize) {
            return false
        }

        val srcStart = src.dataOffset.toInt()
        val srcEnd = srcStart + srcByteSize.toInt()
        val dstStart = dst.dataOffset.toInt()

        if (srcEnd > srcData.size || dstStart + srcByteSize.toInt() > dstData.size) {
            return false
        }

        srcData.copyInto(dstData, dstStart, srcStart, srcEnd)
        return true
    }

    override fun clear(value: UByte) {
        data.fill(value.toByte())
    }

    // Helper methods for ByteArray access (forwarded from GGMLTypes.kt)
    fun getIntLe(offset: Int): Int = data.getIntLe(offset)
    fun getFloatLe(offset: Int): Float = data.getFloatLe(offset)
    fun getShortLe(offset: Int): Short = data.getShortLe(offset)
    fun getLongLe(offset: Int): Long = data.getLongLe(offset)

    fun setIntLe(offset: Int, value: Int) = data.setIntLe(offset, value)
    fun setFloatLe(offset: Int, value: Float) = data.setFloatLe(offset, value)
    fun setShortLe(offset: Int, value: Short) = data.setShortLe(offset, value)
    fun setLongLe(offset: Int, value: Long) = data.setLongLe(offset, value)
}

// ============================================================================
// CPU backend context  (ggml_backend_cpu_context)
// ============================================================================

/**
 * Internal mutable context carried by every [io.github.kotlinmania.llama.ore.GGMLCpuBackend] instance.
 *
 * Mirrors `struct ggml_backend_cpu_context` (lines 99-110 in ggml-cpu.cpp).
 *
 * @property nThreads     Number of threads for graph computation.
 * @property threadpool   Optional threadpool; when non-null, compute uses it
 *                        instead of spawning ad-hoc threads.
 * @property workData     Scratch buffer re-used across consecutive
 *                        [io.github.kotlinmania.llama.ore.GGMLCpuBackend.graphCompute] calls.
 * @property workSize     Current size (in bytes) of [workData].
 * @property abortCallback If non-null, checked between compute steps;
 *                          returning `true` aborts the run.
 * @property abortCallbackData Opaque data forwarded to [abortCallback].
 * @property useRef       When `true`, only reference (non-optimised)
 *                        implementations are used.
 */
class GGMLCpuBackendContext(
    var nThreads: Int = io.github.kotlinmania.llama.ore.GGML_DEFAULT_N_THREADS,
    var threadpool: io.github.kotlinmania.llama.ore.GGMLThreadpool? = null,
    var workData: Any? = null,
    var workSize: ULong = 0uL,
    var abortCallback: ((data: Any?) -> Boolean)? = null,
    var abortCallbackData: Any? = null,
    var useRef: Boolean = false
)

// ============================================================================
// CPU graph plan  (ggml_backend_plan_cpu)
// ============================================================================

/**
 * A pre-computed execution plan for a computation graph on the CPU backend.
 *
 * Mirrors `struct ggml_backend_plan_cpu` (lines 125-128 in ggml-cpu.cpp).
 *
 * @property cplan  The compute plan (thread counts, work buffer, callbacks).
 * @property cgraph A snapshot of the computation graph at plan-creation time.
 */
class GGMLCpuGraphPlan(
    val cplan: io.github.kotlinmania.llama.ore.GGMLCPlan,
    val cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph
)

// ============================================================================
// CPU backend GUID
// ============================================================================

/**
 * Stable identifier for the CPU backend.
 *
 * Mirrors `ggml_backend_cpu_guid()` — the C version uses a 16-byte array;
 * here we use a human-readable string since Kotlin has no equivalent of
 * `ggml_guid`.
 */
private const val GGML_BACKEND_CPU_GUID = "CPU-KOTLIN-NATIVE"

// ============================================================================
// CPU backend  (ggml_backend_cpu)
// ============================================================================

/**
 * CPU backend implementation.
 *
 * Mirrors the vtable `ggml_backend_cpu_i` and the factory function
 * `ggml_backend_cpu_init()` (lines 193-247 in ggml-cpu.cpp).
 */
class GGMLCpuBackend : io.github.kotlinmania.llama.ore.GGMLBackend {
    companion object {
        private const val BACKEND_GUID = io.github.kotlinmania.llama.ore.GGML_BACKEND_CPU_GUID
    }

    /** Internal CPU context — mirrors `ggml_backend_cpu_context`. */
    internal val cpuCtx = io.github.kotlinmania.llama.ore.GGMLCpuBackendContext()

    private val bufferType = io.github.kotlinmania.llama.ore.createDefaultCpuBufferType()
    // -- GGMLBackend interface ------------------------------------------------

    override fun getGuid(): String = BACKEND_GUID

    /** Mirrors `ggml_backend_cpu_get_name()`. */
    override fun getName(): String = "CPU"

    /** Mirrors `ggml_backend_cpu_free()`. */
    override fun free() {
        cpuCtx.workData = null
        cpuCtx.workSize = 0uL
    }

    override fun getDefaultBufferType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType = bufferType

    /**
     * Create an execution plan for [graph].
     *
     * Mirrors `ggml_backend_cpu_graph_plan_create()` (lines 130-151).
     */
    override fun graphPlanCreate(graph: io.github.kotlinmania.llama.ore.GGMLCGraph): Any? {
        val cplan =
            io.github.kotlinmania.llama.ore.ggmlGraphPlan(graph, cpuCtx.nThreads, cpuCtx.threadpool)

        if (cplan.workSize > 0uL) {
            cplan.workData =
                io.github.kotlinmania.llama.ore.ggml_aligned_malloc(cplan.workSize.toLong())
        }

        cplan.abortCallback = cpuCtx.abortCallback
        cplan.abortCallbackData = cpuCtx.abortCallbackData
        cplan.useRef = cpuCtx.useRef

        return io.github.kotlinmania.llama.ore.GGMLCpuGraphPlan(cplan, graph)
    }

    /**
     * Free a previously created [plan].
     *
     * Mirrors `ggml_backend_cpu_graph_plan_free()` (lines 153-160).
     */
    override fun graphPlanFree(plan: Any?) {
        if (plan is io.github.kotlinmania.llama.ore.GGMLCpuGraphPlan) {
            io.github.kotlinmania.llama.ore.ggml_aligned_free(
                plan.cplan.workData,
                plan.cplan.workSize.toLong()
            )
            plan.cplan.workData = null
        }
    }

    /**
     * Execute a previously created [plan].
     *
     * Mirrors `ggml_backend_cpu_graph_plan_compute()` (lines 162-168).
     */
    override fun graphPlanCompute(plan: Any?): io.github.kotlinmania.llama.ore.GGMLStatus {
        require(plan is io.github.kotlinmania.llama.ore.GGMLCpuGraphPlan) { "Expected GGMLCpuGraphPlan" }
        return io.github.kotlinmania.llama.ore.ggmlGraphCompute(plan.cgraph, plan.cplan)
    }

    /**
     * Plan and execute [graph] in one shot, re-using the work buffer when
     * possible.
     *
     * Mirrors `ggml_backend_cpu_graph_compute()` (lines 170-191).
     */
    override fun graphCompute(graph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLStatus {
        val cplan =
            io.github.kotlinmania.llama.ore.ggmlGraphPlan(graph, cpuCtx.nThreads, cpuCtx.threadpool)

        // Grow the work buffer if necessary
        if (cpuCtx.workSize < cplan.workSize) {
            // Free old work buffer if it was native-allocated
            io.github.kotlinmania.llama.ore.ggml_aligned_free(
                cpuCtx.workData,
                cpuCtx.workSize.toLong()
            )
            val newBuf =
                io.github.kotlinmania.llama.ore.ggml_aligned_malloc(cplan.workSize.toLong())
            if (newBuf == null) {
                cpuCtx.workSize = 0uL
                cpuCtx.workData = null
                return io.github.kotlinmania.llama.ore.GGMLStatus.ALLOC_FAILED
            }
            cpuCtx.workData = newBuf
            cpuCtx.workSize = cplan.workSize
        }
        cplan.workData = cpuCtx.workData

        cplan.abortCallback = cpuCtx.abortCallback
        cplan.abortCallbackData = cpuCtx.abortCallbackData
        cplan.useRef = cpuCtx.useRef

        return io.github.kotlinmania.llama.ore.ggmlGraphCompute(graph, cplan)
    }

    /**
     * Check whether the CPU backend can compute [tensor].
     *
     * Mirrors `ggml_backend_cpu_device_supports_op()` (lines 423-473).
     * The full C++ version checks extra buffer types and per-op type
     * constraints; here we check the known supported op set and delegate
     * shape-ops unconditionally.
     */
    override fun supportsOp(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
        // Shape-only ops are always supported
        if (tensor.op == io.github.kotlinmania.llama.ore.GGMLOp.NONE || tensor.op == io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE ||
            tensor.op == io.github.kotlinmania.llama.ore.GGMLOp.VIEW || tensor.op == io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE ||
            tensor.op == io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE
        ) {
            return true
        }

        return when (tensor.op) {
            io.github.kotlinmania.llama.ore.GGMLOp.DUP -> true
            io.github.kotlinmania.llama.ore.GGMLOp.ADD -> true
            io.github.kotlinmania.llama.ore.GGMLOp.ADD1 -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SUB -> true
            io.github.kotlinmania.llama.ore.GGMLOp.MUL -> true
            io.github.kotlinmania.llama.ore.GGMLOp.DIV -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SQR -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SQRT -> true
            io.github.kotlinmania.llama.ore.GGMLOp.LOG -> true
            io.github.kotlinmania.llama.ore.GGMLOp.NEG -> true
            io.github.kotlinmania.llama.ore.GGMLOp.ABS -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SGN -> true
            io.github.kotlinmania.llama.ore.GGMLOp.RELU -> true
            io.github.kotlinmania.llama.ore.GGMLOp.GELU -> true
            io.github.kotlinmania.llama.ore.GGMLOp.GELU_QUICK -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SILU -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SILU_BACK -> true
            io.github.kotlinmania.llama.ore.GGMLOp.NORM -> true
            io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM -> true
            io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM_BACK -> true
            io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT -> true
            io.github.kotlinmania.llama.ore.GGMLOp.REPEAT -> true
            io.github.kotlinmania.llama.ore.GGMLOp.REPEAT_BACK -> true
            io.github.kotlinmania.llama.ore.GGMLOp.CONCAT -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SUM -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SUM_ROWS -> true
            io.github.kotlinmania.llama.ore.GGMLOp.MEAN -> true
            io.github.kotlinmania.llama.ore.GGMLOp.ARGMAX -> true
            io.github.kotlinmania.llama.ore.GGMLOp.CPY -> true
            io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE -> true
            io.github.kotlinmania.llama.ore.GGMLOp.VIEW -> true
            io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE -> true
            io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE -> true
            io.github.kotlinmania.llama.ore.GGMLOp.GET_ROWS -> true
            io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_INF -> true
            io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX -> true
            io.github.kotlinmania.llama.ore.GGMLOp.ROPE -> true
            io.github.kotlinmania.llama.ore.GGMLOp.COUNT -> false
            else -> false
        }
    }

    /**
     * Mirrors `ggml_backend_cpu_device_supports_buft()` (line 476).
     */
    override fun supportsBufferType(bufferType: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean {
        return bufferType.isHost() || io.github.kotlinmania.llama.ore.ggmlBackendCpuIsExtraBufferType(
            bufferType
        )
    }

    /**
     * CPU backend never offloads — it is the fallback.
     *
     * Mirrors the `NULL` offload_op in `ggml_backend_cpu_device_i`.
     */
    override fun offloadOp(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean = false

    // -- Thread management (mirrors free functions in ggml-cpu.cpp) -----------

    /**
     * Set the number of compute threads.
     *
     * Mirrors `ggml_backend_cpu_set_n_threads()` (lines 253-258).
     */
    fun setThreadCount(threadCount: Int) {
        cpuCtx.nThreads = threadCount
    }

    fun getThreadCount(): Int = cpuCtx.nThreads

    /**
     * Assign a threadpool to the backend.
     *
     * Mirrors `ggml_backend_cpu_set_threadpool()` (lines 260-270).
     */
    fun setThreadpool(threadpool: io.github.kotlinmania.llama.ore.GGMLThreadpool?) {
        val prev = cpuCtx.threadpool
        if (prev != null && prev !== threadpool) {
            // Already had a different threadpool — pause it before switching
        }
        cpuCtx.threadpool = threadpool
    }

    /**
     * Install an abort callback.
     *
     * Mirrors `ggml_backend_cpu_set_abort_callback()` (lines 272-278).
     */
    fun setAbortCallback(
        callback: ((data: Any?) -> Boolean)?,
        callbackData: Any? = null
    ) {
        cpuCtx.abortCallback = callback
        cpuCtx.abortCallbackData = callbackData
    }

    /**
     * Force the backend to use reference (non-optimised) implementations only.
     *
     * Mirrors `ggml_backend_cpu_set_use_ref()` (lines 280-285).
     */
    fun setUseRef(useRef: Boolean) {
        cpuCtx.useRef = useRef
    }

    fun getUseRef(): Boolean = cpuCtx.useRef
}

// ============================================================================
// Free-standing backend identity helpers
// ============================================================================

/** `ggml_backend_is_cpu` — C: ggml-cpu.cpp lines 249-251. */
fun ggmlBackendIsCpu(backend: io.github.kotlinmania.llama.ore.GGMLBackend): Boolean {
    return backend is io.github.kotlinmania.llama.ore.GGMLCpuBackend
}

// ============================================================================
// CPU device context  (ggml_backend_cpu_device_context)
// ============================================================================

/**
 * Device-level context that caches the CPU description string.
 *
 * Mirrors `struct ggml_backend_cpu_device_context` (lines 289-351).
 * The C++ version reads `machdep.cpu.brand_string` (macOS),
 * `/proc/cpuinfo` (Linux), or the Windows registry. Here we default to "CPU"
 * and leave platform-specific detection as a TODO for the Kotlin/Native
 * `expect/actual` layer.
 */
class GGMLCpuDeviceContext {
    val description: String = detectCpuDescription()

    companion object {
        private fun detectCpuDescription(): String {
            return "CPU"
        }
    }
}

// ============================================================================
// CPU device  (ggml_backend_cpu_device)
// ============================================================================

/**
 * The single CPU device exposed by the CPU backend registration.
 *
 * Mirrors `ggml_backend_cpu_device_i` vtable (lines 481-497) and the static
 * device instance created inside `ggml_backend_cpu_reg_get_device()`.
 */
class GGMLCpuDevice(
    private val reg: io.github.kotlinmania.llama.ore.GGMLBackendReg? = null
) : io.github.kotlinmania.llama.ore.GGMLBackendDevice {

    private val ctx = io.github.kotlinmania.llama.ore.GGMLCpuDeviceContext()

    /** Mirrors `ggml_backend_cpu_device_get_name()` (line 353). */
    override fun getName(): String = "CPU"

    /** Mirrors `ggml_backend_cpu_device_get_description()` (line 359). */
    override fun getDescription(): String = ctx.description

    /**
     * Query system memory.
     *
     * Mirrors `ggml_backend_cpu_device_get_memory()` (lines 365-382).
     * Returns (free, total) in bytes. The C++ version calls `sysconf` on
     * POSIX and `GlobalMemoryStatusEx` on Windows. The Kotlin/Native port
     * returns a best-effort estimate via [Runtime] or defaults.
     */
    override fun getMemory(): Pair<ULong, ULong> {
        // Best-effort: 8 GB skeleton so callers get something reasonable.
        // A nativeMain actual can call sysconf(_SC_PHYS_PAGES) for real values.
        val totalBytes = 8uL * 1024uL * 1024uL * 1024uL
        return Pair(totalBytes, totalBytes)
    }

    /** Mirrors `ggml_backend_cpu_device_get_type()` (line 384). */
    override fun getType(): io.github.kotlinmania.llama.ore.GGMLBackendDeviceType = io.github.kotlinmania.llama.ore.GGMLBackendDeviceType.CPU

    /**
     * Mirrors `ggml_backend_cpu_device_get_props()` (lines 390-401).
     */
    override fun getProps(): io.github.kotlinmania.llama.ore.GGMLBackendDeviceProps {
        val (free, total) = getMemory()
        return io.github.kotlinmania.llama.ore.GGMLBackendDeviceProps(
            name = getName(),
            description = getDescription(),
            memoryFree = free,
            memoryTotal = total,
            type = getType(),
            caps = io.github.kotlinmania.llama.ore.GGMLBackendDeviceCaps(
                async = false,
                hostBuffer = false,
                bufferFromHostPtr = true,
                events = false
            )
        )
    }

    /** Mirrors `ggml_backend_cpu_reg()` accessor. */
    override fun getBackendReg(): io.github.kotlinmania.llama.ore.GGMLBackendReg? = reg

    /**
     * Create a new CPU backend instance.
     *
     * Mirrors `ggml_backend_cpu_device_init_backend()` (lines 403-408).
     */
    override fun initBackend(params: String?): io.github.kotlinmania.llama.ore.GGMLBackend {
        return io.github.kotlinmania.llama.ore.GGMLCpuBackend()
    }

    /**
     * Return the CPU buffer type singleton.
     *
     * Mirrors `ggml_backend_cpu_device_get_buffer_type()` (lines 410-414).
     */
    override fun getBufferType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType =
        io.github.kotlinmania.llama.ore.createDefaultCpuBufferType()

    /**
     * Create a CPU buffer wrapping an existing host [ptr].
     *
     * Mirrors `ggml_backend_cpu_device_buffer_from_host_ptr()` (lines 416-421).
     */
    override fun bufferFromHostPtr(
        ptr: ByteArray,
        size: ULong,
        maxTensorSize: ULong
    ): io.github.kotlinmania.llama.ore.GGMLBackendBuffer {
        return io.github.kotlinmania.llama.ore.GGMLCpuBuffer(
            io.github.kotlinmania.llama.ore.GGMLCpuBufferType(),
            ptr,
            size
        )
    }

    /**
     * Check whether the CPU device can compute [op].
     *
     * Mirrors `ggml_backend_cpu_device_supports_op()` (lines 423-474).
     */
    override fun supportsOp(op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
        // Shape-only ops are always supported
        if (op.op == io.github.kotlinmania.llama.ore.GGMLOp.NONE || op.op == io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE ||
            op.op == io.github.kotlinmania.llama.ore.GGMLOp.VIEW || op.op == io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE ||
            op.op == io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE
        ) {
            return true
        }

        // Check extra buffer types on sources
        for (i in 0 until minOf(4, op.src.size)) {
            val src = op.src[i] ?: continue
            val srcBuf = src.buffer ?: continue
            val srcBuft = srcBuf.getType()
            if (io.github.kotlinmania.llama.ore.ggmlBackendCpuIsExtraBufferType(srcBuft)) {
                // Extra buffer type would handle the op — delegate to it
            }
        }

        // Per-op type constraints (simplified from C++ switch)
        return when (op.op) {
            io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT -> {
                val src1 = op.src.getOrNull(1)
                src1 != null && (src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || src1.type == io.github.kotlinmania.llama.ore.GGMLType.F16)
            }
            else -> true
        }
    }

    /**
     * Mirrors `ggml_backend_cpu_device_supports_buft()` (line 476).
     */
    override fun supportsBufferType(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean {
        return buft.isHost() || io.github.kotlinmania.llama.ore.ggmlBackendCpuIsExtraBufferType(buft)
    }
}

// ============================================================================
// CPU backend feature detection
// ============================================================================

/**
 * Returns the list of CPU features detected at runtime.
 *
 * Mirrors `ggml_backend_cpu_get_features()` (lines 528-640). The C++ version
 * probes for SSE3, AVX, NEON, SVE etc. The Kotlin port delegates to
 * [io.github.kotlinmania.llama.ore.ggmlCpuDetectFeatures] (defined in GGMLCpuExecutor.kt) and converts the
 * result into the [io.github.kotlinmania.llama.ore.GGMLBackendFeature] list format.
 */
fun ggmlBackendCpuGetFeatures(): List<io.github.kotlinmania.llama.ore.GGMLBackendFeature> {
    // C++: returns detected CPU features — empty until ggmlCpuDetectFeatures() is ported
    return emptyList()
}

// ============================================================================
// CPU backend proc-address lookup
// ============================================================================

/**
 * Look up an extension function by [name].
 *
 * Mirrors `ggml_backend_cpu_get_proc_address()` (lines 642-681).
 * In C++ this returns raw function pointers; in Kotlin we return lambda
 * wrappers or sentinel objects that the caller can cast.
 */
fun ggmlBackendCpuGetProcAddress(name: String): Any? {
    // Each branch returns a lambda cast to Any? so the when-expression has a
    // uniform type. Callers must cast back to the expected function type.
    val result: Any? = when (name) {
        "ggml_backend_set_n_threads" -> object : Any() {
            fun invoke(backend: io.github.kotlinmania.llama.ore.GGMLBackend, nThreads: Int) {
                require(backend is io.github.kotlinmania.llama.ore.GGMLCpuBackend); backend.setThreadCount(nThreads)
            }
        }
        "ggml_backend_get_features" -> object : Any() {
            fun invoke(): List<io.github.kotlinmania.llama.ore.GGMLBackendFeature> =
                io.github.kotlinmania.llama.ore.ggmlBackendCpuGetFeatures()
        }
        "ggml_backend_set_abort_callback" -> object : Any() {
            fun invoke(backend: io.github.kotlinmania.llama.ore.GGMLBackend, cb: ((Any?) -> Boolean)?, data: Any?) {
                require(backend is io.github.kotlinmania.llama.ore.GGMLCpuBackend); backend.setAbortCallback(cb, data)
            }
        }
        "ggml_backend_cpu_set_use_ref" -> object : Any() {
            fun invoke(backend: io.github.kotlinmania.llama.ore.GGMLBackend, useRef: Boolean) {
                require(backend is io.github.kotlinmania.llama.ore.GGMLCpuBackend); backend.setUseRef(useRef)
            }
        }
        "ggml_backend_cpu_set_threadpool" -> object : Any() {
            fun invoke(backend: io.github.kotlinmania.llama.ore.GGMLBackend, tp: io.github.kotlinmania.llama.ore.GGMLThreadpool) {
                require(backend is io.github.kotlinmania.llama.ore.GGMLCpuBackend); backend.setThreadpool(tp)
            }
        }
        "ggml_threadpool_new" -> object : Any() {
            fun invoke(params: io.github.kotlinmania.llama.ore.GGMLThreadpoolParams): io.github.kotlinmania.llama.ore.GGMLThreadpool =
                io.github.kotlinmania.llama.ore.ggmlThreadpoolNew(params)
        }
        "ggml_threadpool_free" -> object : Any() {
            fun invoke(tp: io.github.kotlinmania.llama.ore.GGMLThreadpool) =
                io.github.kotlinmania.llama.ore.ggmlThreadpoolFree(tp)
        }
        "ggml_backend_cpu_numa_init" -> object : Any() {
            fun invoke(strategy: io.github.kotlinmania.llama.ore.GGMLNumaStrategy) =
                io.github.kotlinmania.llama.ore.ggmlNumaInit(strategy)
        }
        "ggml_backend_cpu_is_numa" -> object : Any() {
            fun invoke(): Boolean = io.github.kotlinmania.llama.ore.ggmlIsNuma()
        }
        else -> null
    }
    return result
}

// ============================================================================
// CPU backend registration  (ggml_backend_cpu_reg)
// ============================================================================

/**
 * CPU backend registration singleton.
 *
 * Mirrors `ggml_backend_cpu_reg()` (lines 690-701) and the static
 * `ggml_backend_cpu_reg_i` vtable (lines 683-688).
 */
class GGMLCpuBackendReg private constructor() : io.github.kotlinmania.llama.ore.GGMLBackendReg {

    companion object {
        /** Lazy singleton mirroring the C static local. */
        val instance: GGMLCpuBackendReg by lazy { GGMLCpuBackendReg() }
    }

    /** The single CPU device. */
    private val device: io.github.kotlinmania.llama.ore.GGMLCpuDevice by lazy {
        io.github.kotlinmania.llama.ore.GGMLCpuDevice(
            reg = this
        )
    }

    /** Mirrors `ggml_backend_cpu_reg_get_name()` (line 501). */
    override fun getName(): String = "CPU"

    /** Mirrors `ggml_backend_cpu_reg_get_device_count()` (line 507). */
    override fun getDeviceCount(): ULong = 1uL

    /**
     * Mirrors `ggml_backend_cpu_reg_get_device()` (lines 513-524).
     */
    override fun getDevice(index: ULong): io.github.kotlinmania.llama.ore.GGMLBackendDevice {
        require(index == 0uL) { "CPU backend has exactly one device (index must be 0)" }
        return device
    }

    /**
     * Mirrors `ggml_backend_cpu_get_proc_address()` (lines 642-681).
     */
    override fun getProcAddress(name: String): Any? {
        return io.github.kotlinmania.llama.ore.ggmlBackendCpuGetProcAddress(name)
    }

    /**
     * Mirrors `ggml_backend_cpu_get_features()`.
     */
    override fun getFeatures(): List<io.github.kotlinmania.llama.ore.GGMLBackendFeature> {
        return io.github.kotlinmania.llama.ore.ggmlBackendCpuGetFeatures()
    }
}

/**
 * Return the global CPU backend registration singleton.
 *
 * Mirrors `ggml_backend_cpu_reg()` (lines 690-701).
 */
fun ggmlBackendCpuRegSingleton(): io.github.kotlinmania.llama.ore.GGMLBackendReg = io.github.kotlinmania.llama.ore.GGMLCpuBackendReg.instance

// ============================================================================
// Free-standing function wrappers (match C++ ggml-cpu.cpp function names)
// ============================================================================

// C++: ggml_backend_cpu_get_name (line 112)
fun ggmlBackendCpuGetName(backend: io.github.kotlinmania.llama.ore.GGMLBackend): String {
    return "CPU"
}

// C++: ggml_backend_cpu_free (line 118)
fun ggmlBackendCpuFree(backend: io.github.kotlinmania.llama.ore.GGMLCpuBackend) {
    backend.free()
}

// C++: ggml_backend_cpu_graph_plan_create (line 130)
fun ggmlBackendCpuGraphPlanCreate(backend: io.github.kotlinmania.llama.ore.GGMLCpuBackend, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): Any? {
    return backend.graphPlanCreate(cgraph)
}

// C++: ggml_backend_cpu_graph_plan_free (line 153)
fun ggmlBackendCpuGraphPlanFree(backend: io.github.kotlinmania.llama.ore.GGMLCpuBackend, plan: Any?) {
    backend.graphPlanFree(plan)
}

// C++: ggml_backend_cpu_graph_plan_compute (line 162)
fun ggmlBackendCpuGraphPlanCompute(backend: io.github.kotlinmania.llama.ore.GGMLCpuBackend, plan: Any?): io.github.kotlinmania.llama.ore.GGMLStatus {
    return backend.graphPlanCompute(plan)
}

// C++: ggml_backend_cpu_graph_compute (line 170)
fun ggmlBackendCpuGraphCompute(backend: io.github.kotlinmania.llama.ore.GGMLCpuBackend, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLStatus {
    return backend.graphCompute(cgraph)
}

// C++: ggml_backend_cpu_guid (line 212)
fun ggmlBackendCpuGuid(): ByteArray {
    return byteArrayOf(
        0xaa.toByte(), 0x67, 0xc7.toByte(), 0x43, 0x96.toByte(), 0xe6.toByte(),
        0xa3.toByte(), 0x8a.toByte(), 0xe3.toByte(), 0xaf.toByte(), 0xea.toByte(),
        0x92.toByte(), 0x36, 0xbc.toByte(), 0xfc.toByte(), 0x89.toByte()
    )
}

// C++: ggml_backend_cpu_device_get_name (line 353)
fun ggmlBackendCpuDeviceGetName(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice): String {
    return "CPU"
}

// C++: ggml_backend_cpu_device_get_description (line 359)
fun ggmlBackendCpuDeviceGetDescription(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice): String {
    return dev.getDescription()
}

// C++: ggml_backend_cpu_device_get_memory (line 365)
fun ggmlBackendCpuDeviceGetMemory(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice): Pair<ULong, ULong> {
    return dev.getMemory()
}

// C++: ggml_backend_cpu_device_get_type (line 384)
fun ggmlBackendCpuDeviceGetType(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice): io.github.kotlinmania.llama.ore.GGMLBackendDeviceType {
    return io.github.kotlinmania.llama.ore.GGMLBackendDeviceType.CPU
}

// C++: ggml_backend_cpu_device_get_props (line 390)
fun ggmlBackendCpuDeviceGetProps(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice): Any {
    return dev.getProps()
}

// C++: ggml_backend_cpu_device_init_backend (line 403)
fun ggmlBackendCpuDeviceInitBackend(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice, params: String?): io.github.kotlinmania.llama.ore.GGMLBackend {
    return io.github.kotlinmania.llama.ore.GGMLCpuBackend()
}

// C++: ggml_backend_cpu_device_get_buffer_type (line 410)
fun ggmlBackendCpuDeviceGetBufferType(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice): io.github.kotlinmania.llama.ore.GGMLBackendBufferType {
    return io.github.kotlinmania.llama.ore.createDefaultCpuBufferType()
}

// C++: ggml_backend_cpu_device_buffer_from_host_ptr (line 416)
fun ggmlBackendCpuDeviceBufferFromHostPtr(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice, data: ByteArray, size: ULong, maxTensorSize: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBuffer {
    return io.github.kotlinmania.llama.ore.GGMLCpuBuffer(
        io.github.kotlinmania.llama.ore.GGMLCpuBufferType(),
        data,
        size
    )
}

// C++: ggml_backend_cpu_device_supports_op (line 423)
fun ggmlBackendCpuDeviceSupportsOp(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    val src0 = op.src.getOrNull(0)
    val src1 = op.src.getOrNull(1)

    if (op.op == io.github.kotlinmania.llama.ore.GGMLOp.NONE || op.op == io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE || op.op == io.github.kotlinmania.llama.ore.GGMLOp.VIEW ||
        op.op == io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE || op.op == io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE) {
        return true
    }

    // Check extra buffer types on sources
    for (i in 0 until minOf(4, op.src.size)) {
        val srcI = op.src[i] ?: continue
        val srcBuf = srcI.buffer ?: continue
        if (io.github.kotlinmania.llama.ore.ggmlBackendCpuIsExtraBufferType(srcBuf.getType())) {
            return true
        }
    }

    return when (op.op) {
        io.github.kotlinmania.llama.ore.GGMLOp.CPY, io.github.kotlinmania.llama.ore.GGMLOp.SET_ROWS ->
            op.type != io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS && op.type != io.github.kotlinmania.llama.ore.GGMLType.IQ3_S &&
            op.type != io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS && op.type != io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS &&
            op.type != io.github.kotlinmania.llama.ore.GGMLType.IQ2_S && op.type != io.github.kotlinmania.llama.ore.GGMLType.IQ1_S &&
            op.type != io.github.kotlinmania.llama.ore.GGMLType.IQ1_M
        io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT -> {
            src1 != null && (src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || src0 != null)
        }
        io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX_BACK -> {
            if (op.src.getOrNull(0)?.type != io.github.kotlinmania.llama.ore.GGMLType.F32 || op.src.getOrNull(1)?.type != io.github.kotlinmania.llama.ore.GGMLType.F32) {
                false
            } else {
                val maxBias = op.opParams.getOrNull(1) ?: 0
                maxBias == 0
            }
        }
        io.github.kotlinmania.llama.ore.GGMLOp.IM2COL_BACK -> src0?.type == io.github.kotlinmania.llama.ore.GGMLType.F32 && src1?.type == io.github.kotlinmania.llama.ore.GGMLType.F32
        io.github.kotlinmania.llama.ore.GGMLOp.GET_ROWS_BACK -> src0?.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || src0?.type == io.github.kotlinmania.llama.ore.GGMLType.F16
        io.github.kotlinmania.llama.ore.GGMLOp.OUT_PROD -> {
            val quantizedOk = src0 != null && io.github.kotlinmania.llama.ore.ggmlIsQuantized(src0.type) &&
                src0.ne[2] == (src1?.ne?.getOrNull(2) ?: 0L) &&
                src0.ne[3] == (src1?.ne?.getOrNull(3) ?: 0L)
            (src0?.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || quantizedOk) &&
            src1?.type == io.github.kotlinmania.llama.ore.GGMLType.F32 && op.type == io.github.kotlinmania.llama.ore.GGMLType.F32
        }
        else -> true
    }
}

// C++: ggml_backend_cpu_device_supports_buft (line 476)
fun ggmlBackendCpuDeviceSupportsBuft(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice, buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean {
    return buft.isHost() || io.github.kotlinmania.llama.ore.ggmlBackendCpuIsExtraBufferType(buft)
}

// C++: ggml_backend_cpu_reg_get_name (line 501)
fun ggmlBackendCpuRegGetName(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg): String {
    return "CPU"
}

// C++: ggml_backend_cpu_reg_get_device_count (line 507)
fun ggmlBackendCpuRegGetDeviceCount(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg): Long {
    return 1L
}

// C++: ggml_backend_cpu_reg_get_device (line 513)
fun ggmlBackendCpuRegGetDevice(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg, index: Long): io.github.kotlinmania.llama.ore.GGMLBackendDevice {
    require(index == 0L)
    return io.github.kotlinmania.llama.ore.GGMLCpuDevice(reg)
}
