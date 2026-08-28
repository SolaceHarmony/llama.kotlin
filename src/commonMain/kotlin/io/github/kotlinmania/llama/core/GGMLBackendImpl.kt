// port-lint: source ggml/src/ggml-backend-impl.h
package io.github.kotlinmania.llama.ore

/**
 * Kotlin Native port of the GGML backend **implementation** header.
 *
 * This file mirrors `ggml-backend-impl.h` — the internal interface that every
 * concrete backend (CPU, CUDA, Metal …) must implement. It complements the
 * public-facing API already defined in `GGMLBackend.kt`.
 *
 * ## Mapping conventions
 * | C / C++                         | Kotlin                              |
 * |---------------------------------|-------------------------------------|
 * | function-pointer vtable struct  | Kotlin `interface`                  |
 * | `size_t`                        | `ULong`                             |
 * | `int64_t`                       | `Long`                              |
 * | `void *`                        | `Any?` (or a typed parameter)       |
 * | `uint8_t`                       | `UByte`                             |
 * | nullable function pointer       | nullable return / default body      |
 */

// ---------------------------------------------------------------------------
// API version constant
// ---------------------------------------------------------------------------

/** Backend API version — bump when the interface contract changes. */
const val GGML_BACKEND_API_VERSION: Int = 2

// ---------------------------------------------------------------------------
// Backend buffer type — implementation interface
// ---------------------------------------------------------------------------

/**
 * Internal vtable that a backend buffer type must implement.
 *
 * Mirrors `ggml_backend_buffer_type_i` in the C source.
 * The public convenience API is [io.github.kotlinmania.llama.ore.GGMLBackendBufferType] in `GGMLBackend.kt`;
 * this interface adds the lower-level hooks that only backend authors need.
 */
interface GGMLBackendBufferTypeImpl {

    /** Return a short human-readable name for this buffer type. */
    fun getName(): String

    /**
     * Allocate a buffer of [size] bytes.
     *
     * @return a new [io.github.kotlinmania.llama.ore.GGMLBackendBufferImpl], or `null` on failure.
     */
    fun allocBuffer(size: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBufferImpl?

    /** Tensor alignment in bytes required by this buffer type. */
    fun getAlignment(): ULong

    /**
     * (optional) Maximum buffer size that can be allocated.
     *
     * Defaults to [ULong.MAX_VALUE] (≈ `SIZE_MAX`).
     */
    fun getMaxSize(): ULong = ULong.MAX_VALUE

    /**
     * (optional) Data size needed to allocate [tensor], including padding.
     *
     * Defaults to `ggml_nbytes(tensor)`.
     */
    fun getAllocSize(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong { return io.github.kotlinmania.llama.ore.ggmlNbytes(
        tensor
    )
    }

    /**
     * (optional) Check if tensor data is in host memory and uses the
     * standard GGML tensor layout.
     *
     * Defaults to `false`.
     */
    fun isHost(): Boolean = false
}

// ---------------------------------------------------------------------------
// Backend buffer type — concrete holder
// ---------------------------------------------------------------------------

/**
 * Concrete backend buffer type wrapping an [iface] vtable, a [device]
 * reference, and an opaque [context].
 *
 * Mirrors `struct ggml_backend_buffer_type` in C.
 */
open class GGMLBackendBufferTypeHolder(
    /** The implementation vtable. */
    val iface: io.github.kotlinmania.llama.ore.GGMLBackendBufferTypeImpl,
    /** The device that owns this buffer type (`null` until assigned). */
    var device: io.github.kotlinmania.llama.ore.GGMLBackendDeviceHolder? = null,
    /** Opaque backend-specific context. */
    var context: Any? = null
)

// ---------------------------------------------------------------------------
// Backend buffer — implementation interface
// ---------------------------------------------------------------------------

/**
 * Internal vtable that a backend buffer must implement.
 *
 * Mirrors `ggml_backend_buffer_i` in C.
 */
interface GGMLBackendBufferImpl {

    /** (optional) Free the buffer and any associated resources. */
    fun freeBuffer() {}

    /** Return the base address / backing object of the buffer. */
    fun getBase(): Any?

    /**
     * (optional) Initialise a tensor that resides in this buffer
     * (e.g. attach tensor extras).
     *
     * @return [io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS] on success.
     */
    fun initTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLStatus = io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS

    /** Fill [size] bytes of [tensor] data starting at [offset] with [value]. */
    fun memsetTensor(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        value: UByte,
        offset: ULong,
        size: ULong
    ) {
    }

    /**
     * Copy host [data] into [tensor] storage starting at byte [offset].
     *
     * @param data  source bytes in host memory.
     * @param offset byte offset into the tensor's backing store.
     * @param size   number of bytes to copy.
     */
    fun setTensor(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong
    )

    /**
     * Copy [tensor] data back to host [data] starting at byte [offset].
     *
     * @param data  destination array in host memory.
     * @param offset byte offset into the tensor's backing store.
     * @param size   number of bytes to copy.
     */
    fun getTensor(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong
    )

    /**
     * (optional) 2-D strided copy: host → tensor.
     *
     * @param data          source bytes in host memory.
     * @param offset        byte offset into the tensor's backing store.
     * @param size          size of a single copy chunk in bytes.
     * @param nCopies       number of chunks to copy.
     * @param strideTensor  byte stride between chunks in the tensor.
     * @param strideData    byte stride between chunks in [data].
     */
    fun setTensor2D(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
    }

    /**
     * (optional) 2-D strided copy: tensor → host.
     *
     * Parameters mirror [setTensor2D].
     */
    fun getTensor2D(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
    }

    /**
     * (optional) Copy tensor data from [src] (which may live in any buffer,
     * including a buffer owned by a different backend) into [dst] which lives
     * in **this** buffer.
     *
     * @return `true` if the copy was performed, `false` if unsupported.
     */
    fun cpyTensor(src: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean = false

    /** Clear the entire buffer by filling every byte with [value]. */
    fun clear(value: UByte)

    /**
     * (optional) Reset any internal state due to tensor initialisation,
     * such as tensor extras.
     */
    fun reset() {}
}

// ---------------------------------------------------------------------------
// Backend buffer — concrete holder
// ---------------------------------------------------------------------------

/**
 * Concrete backend buffer wrapping an [iface] vtable plus bookkeeping.
 *
 * Mirrors `struct ggml_backend_buffer` in C.
 */
open class GGMLBackendBufferHolder(
    /** The implementation vtable. */
    val iface: io.github.kotlinmania.llama.ore.GGMLBackendBufferImpl,
    /** The buffer type that created this buffer. */
    val buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferTypeHolder,
    /** Opaque backend-specific context. */
    var context: Any? = null,
    /** Total size of the buffer in bytes. */
    var size: ULong = 0UL,
    /** How this buffer will be used. */
    var usage: io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage = io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage.COMPUTE
)

// ---------------------------------------------------------------------------
// Backend buffer — factory helpers
// ---------------------------------------------------------------------------

/**
 * Create and initialise a new [io.github.kotlinmania.llama.ore.GGMLBackendBufferHolder].
 *
 * Mirrors `ggml_backend_buffer_init()` in C.
 */
fun ggmlBackendBufferInit(
    buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferTypeHolder,
    iface: io.github.kotlinmania.llama.ore.GGMLBackendBufferImpl,
    context: Any?,
    size: ULong
): io.github.kotlinmania.llama.ore.GGMLBackendBufferHolder {
    return io.github.kotlinmania.llama.ore.GGMLBackendBufferHolder(
        iface = iface,
        buft = buft,
        context = context,
        size = size
    )
}

// ---------------------------------------------------------------------------
// Backend (stream) — implementation interface
// ---------------------------------------------------------------------------

/**
 * Internal vtable that every backend (stream) must implement.
 *
 * Mirrors `ggml_backend_i` in C.
 */
interface GGMLBackendIface {

    /** Return the backend's short name (e.g. "CPU", "CUDA0"). */
    fun getName(): String

    /** Free the backend and release all associated resources. */
    fun free()

    // -- (optional) asynchronous tensor data access --

    /** Asynchronously copy host [data] into [tensor]. */
    fun setTensorAsync(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong
    ) {
    }

    /** Asynchronously copy [tensor] data back to host [data]. */
    fun getTensorAsync(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong
    ) {
    }

    /** Asynchronously set tensor with 2-D strided layout. */
    fun setTensor2DAsync(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
    }

    /** Asynchronously get tensor with 2-D strided layout. */
    fun getTensor2DAsync(
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        data: ByteArray,
        offset: ULong,
        size: ULong,
        nCopies: ULong,
        strideTensor: ULong,
        strideData: ULong
    ) {
    }

    /**
     * (optional) Asynchronous cross-backend tensor copy.
     *
     * @return `true` if the copy was initiated, `false` if unsupported.
     */
    fun cpyTensorAsync(
        backendSrc: io.github.kotlinmania.llama.ore.GGMLBackendHolder,
        backendDst: io.github.kotlinmania.llama.ore.GGMLBackendHolder,
        src: io.github.kotlinmania.llama.ore.GGMLTensor,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ): Boolean = false

    /**
     * (optional) Block until all pending asynchronous operations complete.
     *
     * Required if the backend supports any async operations above.
     */
    fun synchronize() {}

    // -- graph plans (not currently used) --

    /** Create an execution plan for [cgraph]. */
    fun graphPlanCreate(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): Any? { return null }

    /** Free a previously created [plan]. */
    fun graphPlanFree(plan: Any?) {}

    /**
     * Update an existing [plan] with a new [cgraph] that shares the same
     * topology — faster than creating a fresh plan.
     */
    fun graphPlanUpdate(plan: Any?, cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {}

    /** Execute the pre-compiled [plan]. */
    fun graphPlanCompute(plan: Any?): io.github.kotlinmania.llama.ore.GGMLStatus { return io.github.kotlinmania.llama.ore.GGMLStatus.FAILED }

    /**
     * Compute [cgraph] — always asynchronous when the backend supports it.
     *
     * This is the primary entry point for graph execution.
     */
    fun graphCompute(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLStatus

    // -- (optional) event synchronisation --

    /** Record an event on this stream. */
    fun eventRecord(event: io.github.kotlinmania.llama.ore.GGMLBackendEventHolder) {
    }

    /** Wait for an event that was recorded on a different stream. */
    fun eventWait(event: io.github.kotlinmania.llama.ore.GGMLBackendEventHolder) {
    }

    // -- (optional) graph optimisation --

    /** Sort / optimise the nodes in [cgraph] before execution. */
    fun graphOptimize(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
        // identity by default
    }
}

// ---------------------------------------------------------------------------
// Backend (stream) — concrete holder
// ---------------------------------------------------------------------------

/**
 * Concrete backend object wrapping an [iface] vtable.
 *
 * Mirrors `struct ggml_backend` in C.
 *
 * @property guid   globally-unique identifier (128-bit UUID stored as a [String]).
 * @property iface  the backend implementation vtable.
 * @property device the device that created this backend stream.
 * @property context opaque backend-specific context.
 */
open class GGMLBackendHolder(
    val guid: String,
    val iface: io.github.kotlinmania.llama.ore.GGMLBackendIface,
    var device: io.github.kotlinmania.llama.ore.GGMLBackendDeviceHolder? = null,
    var context: Any? = null
)

// ---------------------------------------------------------------------------
// Backend event
// ---------------------------------------------------------------------------

/**
 * A synchronisation event tied to a specific [device].
 *
 * Mirrors `struct ggml_backend_event` in C.
 */
open class GGMLBackendEventHolder(
    /** The device that created this event. */
    val device: io.github.kotlinmania.llama.ore.GGMLBackendDeviceHolder,
    /** Opaque backend-specific context. */
    var context: Any? = null
)

// ---------------------------------------------------------------------------
// Backend device — implementation interface
// ---------------------------------------------------------------------------

/**
 * Device type classification.
 *
 * Re-exported here for completeness; if already defined in `GGMLBackend.kt`
 * remove this definition and use the existing one.
 */
enum class GGMLBackendDevType(val value: Int) {
    CPU(0),
    GPU(1),
    ACCELERATOR(2)
}

/**
 * Memory information returned by [io.github.kotlinmania.llama.ore.GGMLBackendDeviceIface.getMemory].
 */
data class GGMLBackendDevMemory(
    /** Free device memory in bytes (0 = not reported). */
    val free: ULong = 0UL,
    /** Total device memory in bytes (0 = not reported). */
    val total: ULong = 0UL
)

/**
 * Device properties bundle.
 *
 * Mirrors `struct ggml_backend_dev_props` in C.
 */
data class GGMLBackendDevProps(
    val name: String = "",
    val description: String = "",
    val memory: io.github.kotlinmania.llama.ore.GGMLBackendDevMemory = io.github.kotlinmania.llama.ore.GGMLBackendDevMemory(),
    val type: io.github.kotlinmania.llama.ore.GGMLBackendDevType = io.github.kotlinmania.llama.ore.GGMLBackendDevType.CPU
)

/**
 * Internal vtable that every backend device must implement.
 *
 * Mirrors `ggml_backend_device_i` in C.
 */
interface GGMLBackendDeviceIface {

    /** Short identifier for this device (e.g. "CPU", "CUDA0"). */
    fun getName(): String

    /** Short informative description (e.g. model name, chip revision). */
    fun getDescription(): String

    /**
     * Query device memory.
     *
     * Return [io.github.kotlinmania.llama.ore.GGMLBackendDevMemory] with both fields set to 0 when the
     * device has no memory to report.
     */
    fun getMemory(): io.github.kotlinmania.llama.ore.GGMLBackendDevMemory

    /** Classify the device. */
    fun getType(): io.github.kotlinmania.llama.ore.GGMLBackendDevType

    /** Retrieve the full set of device properties. */
    fun getProps(): io.github.kotlinmania.llama.ore.GGMLBackendDevProps

    /**
     * Create a new backend (stream) on this device.
     *
     * @param params optional backend-specific parameter string.
     */
    fun initBackend(params: String?): io.github.kotlinmania.llama.ore.GGMLBackendHolder?

    /** Return the preferred buffer type for this device. */
    fun getBufferType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferTypeHolder

    /**
     * (optional) Return a host-memory buffer type, typically a pinned-memory
     * buffer for faster host↔device transfers.
     */
    fun getHostBufferType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferTypeHolder? = null

    /**
     * (optional) Create a buffer directly from a host pointer.
     *
     * Useful for memory-mapped models and data imported from other libraries.
     *
     * @param ptr           the host pointer (represented as [Any?] in Kotlin).
     * @param size          total size in bytes.
     * @param maxTensorSize largest single tensor that will be stored.
     * @return a buffer backed by the host pointer, or `null` if unsupported.
     */
    fun bufferFromHostPtr(
        ptr: Any?,
        size: ULong,
        maxTensorSize: ULong
    ): io.github.kotlinmania.llama.ore.GGMLBackendBufferHolder? = null

    /** Check whether this device can compute [op]. */
    fun supportsOp(op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean

    /** Check whether this device can use tensors in a [buft] buffer. */
    fun supportsBuft(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferTypeHolder): Boolean

    /**
     * (optional) Check whether this device **wants** to run [op], even when
     * the tensor weights live in an incompatible buffer.
     *
     * Typically returns `true` only for expensive operations that benefit
     * significantly from running on an accelerator.
     */
    fun offloadOp(op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean = false

    // -- (optional) event synchronisation --

    /** Create a new synchronisation event. */
    fun eventNew(): io.github.kotlinmania.llama.ore.GGMLBackendEventHolder? { return null }

    /** Free a previously created [event]. */
    fun eventFree(event: io.github.kotlinmania.llama.ore.GGMLBackendEventHolder) {
    }

    /** Block until [event] has completed. */
    fun eventSynchronize(event: io.github.kotlinmania.llama.ore.GGMLBackendEventHolder) {
    }
}

// ---------------------------------------------------------------------------
// Backend device — concrete holder
// ---------------------------------------------------------------------------

/**
 * Concrete backend device wrapping an [iface] vtable.
 *
 * Mirrors `struct ggml_backend_device` in C.
 */
open class GGMLBackendDeviceHolder(
    /** The device implementation vtable. */
    val iface: io.github.kotlinmania.llama.ore.GGMLBackendDeviceIface,
    /** The registry entry that owns this device. */
    var reg: io.github.kotlinmania.llama.ore.GGMLBackendRegHolder? = null,
    /** Opaque backend-specific context. */
    var context: Any? = null
)

// ---------------------------------------------------------------------------
// Backend registry — implementation interface
// ---------------------------------------------------------------------------

/**
 * Internal vtable for a backend registry entry.
 *
 * Mirrors `ggml_backend_reg_i` in C.
 */
interface GGMLBackendRegIface {

    /** Return the registry entry name (e.g. "CPU", "CUDA"). */
    fun getName(): String

    /** Number of devices exposed by this registry entry. */
    fun getDeviceCount(): ULong

    /** Return the device at [index]. */
    fun getDevice(index: ULong): io.github.kotlinmania.llama.ore.GGMLBackendDeviceHolder

    /**
     * (optional) Look up a custom extension function by [name].
     *
     * Backends can register functions that are not part of the standard
     * ggml-backend interface.
     *
     * @return an opaque handle to the function, or `null` if not found.
     */
    fun getProcAddress(name: String): Any? = null
}

// ---------------------------------------------------------------------------
// Backend registry — concrete holder
// ---------------------------------------------------------------------------

/**
 * Concrete backend registry entry.
 *
 * Mirrors `struct ggml_backend_reg` in C.
 *
 * @property apiVersion must equal [io.github.kotlinmania.llama.ore.GGML_BACKEND_API_VERSION].
 */
open class GGMLBackendRegHolder(
    val apiVersion: Int = io.github.kotlinmania.llama.ore.GGML_BACKEND_API_VERSION,
    val iface: io.github.kotlinmania.llama.ore.GGMLBackendRegIface,
    var context: Any? = null
)

// ---------------------------------------------------------------------------
// Dynamic-loading support (informational — not applicable in Kotlin/Native)
// ---------------------------------------------------------------------------

/**
 * Type alias for the backend initialisation entry point used by the C
 * dynamic-loading mechanism (`ggml_backend_init_t`).
 *
 * In Kotlin/Native there is no direct equivalent of `dlsym`-based loading,
 * but the alias is kept for documentation parity with the C header.
 */
typealias GGMLBackendInitFn = () -> io.github.kotlinmania.llama.ore.GGMLBackendRegHolder?

/**
 * Type alias for the optional backend score function
 * (`ggml_backend_score_t`).
 *
 * Higher scores are preferred; 0 means the backend is not supported on
 * the current system.
 */
typealias GGMLBackendScoreFn = () -> Int

// ---------------------------------------------------------------------------
// Meta backend query functions (from ggml-backend-meta.cpp)
// ---------------------------------------------------------------------------

/**
 * Marker interface for meta backends (backends that wrap multiple simple backends).
 * In C++ this is checked via vtable pointer comparison; in Kotlin we use `is` checks.
 */
interface GGMLMetaBackendMarker

/** Port of `ggml_backend_is_meta` from ggml-backend-meta.cpp line 1953. */
fun ggmlBackendIsMeta(backend: io.github.kotlinmania.llama.ore.GGMLBackend?): Boolean =
    backend is io.github.kotlinmania.llama.ore.GGMLMetaBackendMarker

/** Port of `ggml_backend_buffer_is_meta` from ggml-backend-meta.cpp line 1372. */
fun ggmlBackendBufferIsMeta(buf: io.github.kotlinmania.llama.ore.GGMLBackendBuffer?): Boolean =
    buf is io.github.kotlinmania.llama.ore.GGMLMetaBackendMarker

/** Port of `ggml_backend_buft_is_meta` from ggml-backend-meta.cpp line 339. */
fun ggmlBackendBuftIsMeta(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType?): Boolean =
    buft is io.github.kotlinmania.llama.ore.GGMLMetaBackendMarker

/**
 * Port of `ggml_backend_meta_n_backends`.
 * Awaiting meta backend port — depends on holder/context plumbing not yet ported.
 */
fun ggmlBackendMetaNBackends(@Suppress("UNUSED_PARAMETER") metaBackend: io.github.kotlinmania.llama.ore.GGMLBackend): Int =
    error("ggmlBackendMetaNBackends not yet ported")

/**
 * Port of `ggml_backend_meta_simple_backend`.
 * Awaiting meta backend port.
 */
fun ggmlBackendMetaSimpleBackend(
    @Suppress("UNUSED_PARAMETER") metaBackend: io.github.kotlinmania.llama.ore.GGMLBackend,
    @Suppress("UNUSED_PARAMETER") index: Int
): io.github.kotlinmania.llama.ore.GGMLBackend = error("ggmlBackendMetaSimpleBackend not yet ported")

/**
 * Port of `ggml_backend_meta_alloc_ctx_tensors_from_buft`.
 * Awaiting meta backend port.
 */
fun ggmlBackendMetaAllocCtxTensorsFromBuft(
    @Suppress("UNUSED_PARAMETER") ctx: io.github.kotlinmania.llama.ore.GGMLContext,
    @Suppress("UNUSED_PARAMETER") buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType
): io.github.kotlinmania.llama.ore.GGMLBackendBuffer = error("ggmlBackendMetaAllocCtxTensorsFromBuft not yet ported")

// --- Functions declared in ggml-backend-impl.h, moved here for ast_distance parity ---

/**
 * ggml_backend_buffer_copy_tensor — ggml-backend-impl.h line 79 / ggml-backend.cpp line 205.
 * Copies src tensor data using the destination buffer's copyTensor hook.
 */
fun ggmlBackendBufferCopyTensor(src: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    val dstBuf = dst.viewSrc?.buffer ?: dst.buffer ?: return false
    return dstBuf.copyTensor(src, dst)
}

/**
 * ggml_backend_buffer_is_multi_buffer — ggml-backend-impl.h line 84 / ggml-backend.cpp line 723.
 * Returns true if [buffer] is a multi-buffer wrapper.
 */
fun ggmlBackendBufferIsMultiBuffer(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): Boolean {
    return buffer is io.github.kotlinmania.llama.ore.GGMLBackendMultiBufferWrapper
}

// ggmlBackendMultiBufferSetUsage moved to GGMLBackendUtils.kt
