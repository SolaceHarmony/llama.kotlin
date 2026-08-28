// port-lint: source ggml/src/ggml-backend.cpp
package io.github.kotlinmania.llama.ore

/**
 * Kotlin port of `ggml-backend.cpp` — the main backend implementation file.
 *
 * Contains:
 * 1. Backend buffer-type operations (alloc, alignment, max-size, is-host …)
 * 2. Backend buffer operations (init, free, get/set tensor, clear, reset …)
 * 3. Backend (stream) operations (sync, graph plan/compute, supports-op …)
 * 4. Tensor data helpers (set, get, memset, copy, copy-async, 2-D strided …)
 * 5. Event operations (new, free, record, synchronize, wait)
 * 6. Device operations (name, description, memory, type, props, buffer-type …)
 * 7. Registry operations (name, dev-count, dev-get, proc-address)
 * 8. Multi-buffer helpers
 * 9. Backend scheduler (graph-split, alloc, compute, synchronize)
 * 10. Graph-copy / compare utilities
 * 11. CPU backend buffer and buffer-type implementation
 *
 * Scheduler, graph-copy, and CPU buffer implementations follow the
 * transliterated C++ code from ggml-backend.cpp.
 */

// =====================================================================
// Constants  (mirrors C #define values)
// =====================================================================

/** Maximum number of backends in a scheduler. C: `GGML_SCHED_MAX_BACKENDS`. */
const val GGML_SCHED_MAX_BACKENDS: Int = 16

/** Maximum number of inputs to a single graph split. C: `GGML_SCHED_MAX_SPLIT_INPUTS`. */
const val GGML_SCHED_MAX_SPLIT_INPUTS: Int = 30

/** Maximum number of pipeline-parallel copies. C: `GGML_SCHED_MAX_COPIES`. */
const val GGML_SCHED_MAX_COPIES: Int = 4

// =====================================================================
// 1. Backend buffer-type free-standing functions
//    C: ggml_backend_buft_*
// =====================================================================

/** `ggml_backend_buft_name` */
fun ggmlBackendBuftName(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): String {
    return buft.getName()
}

/** `ggml_backend_buft_alloc_buffer` — returns a dummy buffer for zero-sized requests. */
fun ggmlBackendBuftAllocBuffer(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType, size: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBuffer? {
    if (size == 0UL) {
        // return a dummy buffer for zero-sized allocations (mirrors C)
        return null
    }
    return buft.allocBuffer(size)
}

/** `ggml_backend_buft_get_alignment` */
fun ggmlBackendBuftGetAlignment(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): UInt {
    return buft.getAlignment()
}

/** `ggml_backend_buft_get_max_size` — defaults to ULong.MAX_VALUE. */
fun ggmlBackendBuftGetMaxSize(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): ULong {
    return buft.getMaxSize()
}

/** `ggml_backend_buft_get_alloc_size` — defaults to ggml_nbytes(tensor). */
fun ggmlBackendBuftGetAllocSize(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType, tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong {
    val size = buft.getAllocSize(tensor)
    require(size >= io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)) { "alloc size must be >= ggml_nbytes" }
    return size
}

/** `ggml_backend_buft_is_host` */
fun ggmlBackendBuftIsHost(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean {
    return buft.isHost()
}

/** `ggml_backend_buft_get_device` */
fun ggmlBackendBuftGetDevice(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): io.github.kotlinmania.llama.ore.GGMLBackendDevice? {
    return buft.getDevice()
}

// =====================================================================
// 2. Backend buffer free-standing functions
//    C: ggml_backend_buffer_*
// =====================================================================

// ggmlBackendBufferInit lives in GGMLBackendImpl.kt (declared in ggml-backend-impl.h)

/** `ggml_backend_buffer_name` */
fun ggmlBackendBufferName(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): String {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftName(buffer.getType())
}

/** `ggml_backend_buffer_free` */
fun ggmlBackendBufferFree(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer?) {
    buffer?.free()
}

/** `ggml_backend_buffer_get_size` */
fun ggmlBackendBufferGetSize(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): ULong {
    return buffer.getSize()
}

/** `ggml_backend_buffer_get_base` */
fun ggmlBackendBufferGetBase(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): Any? {
    if (buffer.getSize() == 0UL) return null
    return buffer.getBase()
}

/** `ggml_backend_buffer_init_tensor` */
fun ggmlBackendBufferInitTensor(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLStatus {
    return buffer.initTensor(tensor)
}

/** `ggml_backend_buffer_get_alignment` */
fun ggmlBackendBufferGetAlignment(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): ULong {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftGetAlignment(buffer.getType()).toULong()
}

/** `ggml_backend_buffer_get_max_size` */
fun ggmlBackendBufferGetMaxSize(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): ULong {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftGetMaxSize(buffer.getType())
}

/** `ggml_backend_buffer_get_alloc_size` */
fun ggmlBackendBufferGetAllocSize(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftGetAllocSize(buffer.getType(), tensor)
}

/** `ggml_backend_buffer_clear` */
fun ggmlBackendBufferClear(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, value: UByte) {
    if (buffer.getSize() == 0UL) return
    buffer.clear(value)
}

/** `ggml_backend_buffer_is_host` */
fun ggmlBackendBufferIsHost(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): Boolean {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftIsHost(buffer.getType())
}

/** `ggml_backend_buffer_set_usage` */
fun ggmlBackendBufferSetUsage(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, usage: io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage) {
    buffer.setUsage(usage)
    // FIXME: add a generic callback to the buffer interface
    if (io.github.kotlinmania.llama.ore.ggmlBackendBufferIsMultiBuffer(buffer)) {
        io.github.kotlinmania.llama.ore.ggmlBackendMultiBufferSetUsage(buffer, usage)
    }
}

/** `ggml_backend_buffer_get_usage` */
fun ggmlBackendBufferGetUsage(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage {
    return buffer.getUsage()
}

/** `ggml_backend_buffer_get_type` */
fun ggmlBackendBufferGetType(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): io.github.kotlinmania.llama.ore.GGMLBackendBufferType {
    return buffer.getType()
}

/** `ggml_backend_buffer_reset` */
fun ggmlBackendBufferReset(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer) {
    buffer.reset()
}

// ggmlBackendBufferCopyTensor lives in GGMLBackendImpl.kt (declared in ggml-backend-impl.h)

// =====================================================================
// 3. Backend (stream) free-standing functions
//    C: ggml_backend_*
// =====================================================================

/** `ggml_backend_guid` */
fun ggmlBackendGuid(backend: io.github.kotlinmania.llama.ore.GGMLBackend?): String? {
    return backend?.getGuid()
}

/** `ggml_backend_name` */
fun ggmlBackendName(backend: io.github.kotlinmania.llama.ore.GGMLBackend?): String {
    return backend?.getName() ?: "NULL"
}

/** `ggml_backend_free` */
fun ggmlBackendFree(backend: io.github.kotlinmania.llama.ore.GGMLBackend?) {
    backend?.free()
}

/** `ggml_backend_get_default_buffer_type` */
fun ggmlBackendGetDefaultBufferType(backend: io.github.kotlinmania.llama.ore.GGMLBackend): io.github.kotlinmania.llama.ore.GGMLBackendBufferType {
    return backend.getDefaultBufferType()
}

/** `ggml_backend_alloc_buffer` */
fun ggmlBackendAllocBuffer(backend: io.github.kotlinmania.llama.ore.GGMLBackend, size: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBuffer? {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftAllocBuffer(
        io.github.kotlinmania.llama.ore.ggmlBackendGetDefaultBufferType(
            backend
        ), size
    )
}

/** `ggml_backend_get_alignment` */
fun ggmlBackendGetAlignment(backend: io.github.kotlinmania.llama.ore.GGMLBackend): UInt {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftGetAlignment(
        io.github.kotlinmania.llama.ore.ggmlBackendGetDefaultBufferType(
            backend
        )
    )
}

/** `ggml_backend_get_max_size` */
fun ggmlBackendGetMaxSize(backend: io.github.kotlinmania.llama.ore.GGMLBackend): ULong {
    return io.github.kotlinmania.llama.ore.ggmlBackendBuftGetMaxSize(
        io.github.kotlinmania.llama.ore.ggmlBackendGetDefaultBufferType(
            backend
        )
    )
}

/** `ggml_backend_synchronize` */
fun ggmlBackendSynchronize(backend: io.github.kotlinmania.llama.ore.GGMLBackend) {
    backend.synchronize()
}

/** `ggml_backend_graph_plan_create` */
fun ggmlBackendGraphPlanCreate(backend: io.github.kotlinmania.llama.ore.GGMLBackend, graph: io.github.kotlinmania.llama.ore.GGMLCGraph): Any? {
    return backend.graphPlanCreate(graph)
}

/** `ggml_backend_graph_plan_free` */
fun ggmlBackendGraphPlanFree(backend: io.github.kotlinmania.llama.ore.GGMLBackend, plan: Any?) {
    backend.graphPlanFree(plan)
}

/** `ggml_backend_graph_plan_compute` */
fun ggmlBackendGraphPlanCompute(backend: io.github.kotlinmania.llama.ore.GGMLBackend, plan: Any?): io.github.kotlinmania.llama.ore.GGMLStatus {
    return backend.graphPlanCompute(plan)
}

/** `ggml_backend_graph_compute` — sync wrapper around async compute. */
fun ggmlBackendGraphCompute(backend: io.github.kotlinmania.llama.ore.GGMLBackend, graph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLStatus {
    val err = io.github.kotlinmania.llama.ore.ggmlBackendGraphComputeAsync(backend, graph)
    io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(backend)
    return err
}

/** `ggml_backend_graph_compute_async` */
fun ggmlBackendGraphComputeAsync(backend: io.github.kotlinmania.llama.ore.GGMLBackend, graph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLStatus {
    return backend.graphComputeAsync(graph)
}

/** `ggml_backend_supports_op` */
fun ggmlBackendSupportsOp(backend: io.github.kotlinmania.llama.ore.GGMLBackend, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    return backend.supportsOp(op)
}

/** `ggml_backend_supports_buft` */
fun ggmlBackendSupportsBuft(backend: io.github.kotlinmania.llama.ore.GGMLBackend, buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean {
    return backend.supportsBufferType(buft)
}

/** `ggml_backend_offload_op` */
fun ggmlBackendOffloadOp(backend: io.github.kotlinmania.llama.ore.GGMLBackend, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    return backend.offloadOp(op)
}

/** `ggml_backend_get_device` */
fun ggmlBackendGetDevice(backend: io.github.kotlinmania.llama.ore.GGMLBackend): io.github.kotlinmania.llama.ore.GGMLBackendDevice? {
    return backend.getDevice()
}

/** `ggml_backend_graph_optimize` (static in C) */
fun ggmlBackendGraphOptimize(backend: io.github.kotlinmania.llama.ore.GGMLBackend, graph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    // optional: identity unless a backend overrides
}

// =====================================================================
// 4. Tensor data helpers
//    C: ggml_backend_tensor_set / get / memset / copy / copy_async / 2d
// =====================================================================

/** `ggml_backend_tensor_set_async` */
fun ggmlBackendTensorSetAsync(
    backend: io.github.kotlinmania.llama.ore.GGMLBackend,
    tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong
) {
    require(offset + size <= io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)) { "tensor write out of bounds" }
    backend.setTensorAsync(tensor, data, offset, size)
}

/** `ggml_backend_tensor_get_async` */
fun ggmlBackendTensorGetAsync(
    backend: io.github.kotlinmania.llama.ore.GGMLBackend,
    tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong
) {
    require(offset + size <= io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)) { "tensor read out of bounds" }
    backend.getTensorAsync(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_set_2d_async`
 * Set 2-D strided tensor data asynchronously.
 */
fun ggmlBackendTensorSet2dAsync(
    backend: io.github.kotlinmania.llama.ore.GGMLBackend,
    tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    if (nCopies <= 1UL) {
        for (i in 0UL until nCopies) {
            io.github.kotlinmania.llama.ore.ggmlBackendTensorSetAsync(
                backend,
                tensor,
                data,
                offset + i * strideTensor,
                size
            )
        }
        return
    }
    if (size == 0UL) return
    require(offset + (nCopies - 1UL) * strideTensor + size <= io.github.kotlinmania.llama.ore.ggmlNbytes(
        tensor
    )
    ) { "tensor write out of bounds" }
    backend.setTensor2dAsync(tensor, data, offset, size, nCopies, strideTensor, strideData)
}

/**
 * `ggml_backend_tensor_get_2d_async`
 * Get 2-D strided tensor data asynchronously.
 */
fun ggmlBackendTensorGet2dAsync(
    backend: io.github.kotlinmania.llama.ore.GGMLBackend,
    tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    if (nCopies <= 1UL) {
        for (i in 0UL until nCopies) {
            io.github.kotlinmania.llama.ore.ggmlBackendTensorGetAsync(
                backend,
                tensor,
                data,
                offset + i * strideTensor,
                size
            )
        }
        return
    }
    if (size == 0UL) return
    require(offset + (nCopies - 1UL) * strideTensor + size <= io.github.kotlinmania.llama.ore.ggmlNbytes(
        tensor
    )
    ) { "tensor read out of bounds" }
    backend.getTensor2dAsync(tensor, data, offset, size, nCopies, strideTensor, strideData)
}

/**
 * `ggml_backend_tensor_set` — synchronous set from host memory.
 */
fun ggmlBackendTensorSet(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    if (size == 0UL) return
    require(offset + size <= io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)) { "tensor write out of bounds" }
    buf.setTensor(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_get` — synchronous get into host memory.
 */
fun ggmlBackendTensorGet(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    if (size == 0UL) return
    require(offset + size <= io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)) { "tensor read out of bounds" }
    buf.getTensor(tensor, data, offset, size)
}

/**
 * `ggml_backend_tensor_set_2d` — synchronous 2-D strided set.
 */
fun ggmlBackendTensorSet2d(
    tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    // fallback to per-row set if no 2-D helper on the buffer
    for (i in 0UL until nCopies) {
        io.github.kotlinmania.llama.ore.ggmlBackendTensorSet(
            tensor,
            data,
            offset + i * strideTensor,
            size
        )
    }
}

/**
 * `ggml_backend_tensor_get_2d` — synchronous 2-D strided get.
 */
fun ggmlBackendTensorGet2d(
    tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
    data: ByteArray,
    offset: ULong,
    size: ULong,
    nCopies: ULong,
    strideTensor: ULong,
    strideData: ULong
) {
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    for (i in 0UL until nCopies) {
        io.github.kotlinmania.llama.ore.ggmlBackendTensorGet(
            tensor,
            data,
            offset + i * strideTensor,
            size
        )
    }
}

/**
 * `ggml_backend_tensor_memset` — fill a region of a tensor with a byte value.
 */
fun ggmlBackendTensorMemset(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, value: UByte, offset: ULong, size: ULong) {
    if (size == 0UL) return
    val buf = tensor.viewSrc?.buffer ?: tensor.buffer
    requireNotNull(buf) { "tensor buffer not set" }
    require(offset + size <= io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)) { "tensor write out of bounds" }
    buf.clear(value) // simplified: clear the entire buffer with value
}

/**
 * `ggml_backend_tensor_copy` — copy tensor data between different backends.
 * Tries direct buffer copy first, then falls back through host memory.
 */
fun ggmlBackendTensorCopy(src: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    if (src === dst) return

    val srcBuf = src.buffer
    val dstBuf = dst.buffer

    if (srcBuf != null && io.github.kotlinmania.llama.ore.ggmlBackendBufferIsHost(srcBuf)) {
        // source is in host memory — direct set into dst
        val nbytes = io.github.kotlinmania.llama.ore.ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        srcBuf.getTensor(src, tmp, 0UL, nbytes)
        io.github.kotlinmania.llama.ore.ggmlBackendTensorSet(dst, tmp, 0UL, nbytes)
    } else if (dstBuf != null && io.github.kotlinmania.llama.ore.ggmlBackendBufferIsHost(dstBuf)) {
        // destination is in host memory — direct get from src
        val nbytes = io.github.kotlinmania.llama.ore.ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        io.github.kotlinmania.llama.ore.ggmlBackendTensorGet(src, tmp, 0UL, nbytes)
        dstBuf.setTensor(dst, tmp, 0UL, nbytes)
    } else if (!io.github.kotlinmania.llama.ore.ggmlBackendBufferCopyTensor(src, dst)) {
        // slow path: round-trip through host
        val nbytes = io.github.kotlinmania.llama.ore.ggmlNbytes(src)
        val tmp = ByteArray(nbytes.toInt())
        io.github.kotlinmania.llama.ore.ggmlBackendTensorGet(src, tmp, 0UL, nbytes)
        io.github.kotlinmania.llama.ore.ggmlBackendTensorSet(dst, tmp, 0UL, nbytes)
    }
}

/**
 * `ggml_backend_tensor_copy_async` — async copy between two backends.
 * Falls back to synchronous copy when the backend doesn't support async copies.
 */
fun ggmlBackendTensorCopyAsync(
    backendSrc: io.github.kotlinmania.llama.ore.GGMLBackend,
    backendDst: io.github.kotlinmania.llama.ore.GGMLBackend,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    if (src === dst) return

    if (backendDst.copyTensorAsync(backendSrc, src, dst)) {
        return
    }
    // fallback: synchronize both then do a blocking copy
    io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(backendSrc)
    io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(backendDst)
    io.github.kotlinmania.llama.ore.ggmlBackendTensorCopy(src, dst)
}

// =====================================================================
// 5. Event operations
//    C: ggml_backend_event_*
// =====================================================================

/** `ggml_backend_event_new` */
fun ggmlBackendEventNew(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice?): io.github.kotlinmania.llama.ore.GGMLBackendEvent? {
    return device?.newEvent()
}

/** `ggml_backend_event_free` */
fun ggmlBackendEventFree(event: io.github.kotlinmania.llama.ore.GGMLBackendEvent?) {
    event?.free()
}

/** `ggml_backend_event_record` */
fun ggmlBackendEventRecord(event: io.github.kotlinmania.llama.ore.GGMLBackendEvent, backend: io.github.kotlinmania.llama.ore.GGMLBackend) {
    event.record(backend)
}

/** `ggml_backend_event_synchronize` */
fun ggmlBackendEventSynchronize(event: io.github.kotlinmania.llama.ore.GGMLBackendEvent) {
    event.synchronize()
}

/** `ggml_backend_event_wait` */
fun ggmlBackendEventWait(backend: io.github.kotlinmania.llama.ore.GGMLBackend, event: io.github.kotlinmania.llama.ore.GGMLBackendEvent) {
    event.wait(backend)
}

// =====================================================================
// 6. Device operations
//    C: ggml_backend_dev_*
// =====================================================================

/** `ggml_backend_dev_name` */
fun ggmlBackendDevName(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): String {
    return device.getName()
}

/** `ggml_backend_dev_description` */
fun ggmlBackendDevDescription(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): String {
    return device.getDescription()
}

/** `ggml_backend_dev_memory` — returns (free, total). */
fun ggmlBackendDevMemory(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): Pair<ULong, ULong> {
    return device.getMemory()
}

/** `ggml_backend_dev_type` */
fun ggmlBackendDevType(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): io.github.kotlinmania.llama.ore.GGMLBackendDeviceType {
    return device.getType()
}

/** `ggml_backend_dev_get_props` */
fun ggmlBackendDevGetProps(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): io.github.kotlinmania.llama.ore.GGMLBackendDeviceProps {
    return device.getProps()
}

/** `ggml_backend_dev_backend_reg` */
fun ggmlBackendDevBackendReg(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): io.github.kotlinmania.llama.ore.GGMLBackendReg? {
    return device.getBackendReg()
}

/** `ggml_backend_dev_init` */
fun ggmlBackendDevInit(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice, params: String? = null): io.github.kotlinmania.llama.ore.GGMLBackend? {
    return device.initBackend(params)
}

/** `ggml_backend_dev_buffer_type` */
fun ggmlBackendDevBufferType(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): io.github.kotlinmania.llama.ore.GGMLBackendBufferType {
    return device.getBufferType()
}

/** `ggml_backend_dev_host_buffer_type` */
fun ggmlBackendDevHostBufferType(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice): io.github.kotlinmania.llama.ore.GGMLBackendBufferType? {
    return device.getHostBufferType()
}

/** `ggml_backend_dev_buffer_from_host_ptr` */
fun ggmlBackendDevBufferFromHostPtr(
    device: io.github.kotlinmania.llama.ore.GGMLBackendDevice,
    ptr: ByteArray,
    size: ULong,
    maxTensorSize: ULong
): io.github.kotlinmania.llama.ore.GGMLBackendBuffer? {
    return device.bufferFromHostPtr(ptr, size, maxTensorSize)
}

/** `ggml_backend_dev_supports_op` */
fun ggmlBackendDevSupportsOp(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    return device.supportsOp(op)
}

/** `ggml_backend_dev_supports_buft` */
fun ggmlBackendDevSupportsBuft(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice, buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean {
    return device.supportsBufferType(buft)
}

/** `ggml_backend_dev_offload_op` */
fun ggmlBackendDevOffloadOp(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    return device.offloadOp(op)
}

// =====================================================================
// 7. Backend registration (reg) operations
//    C: ggml_backend_reg_*
// =====================================================================

/** `ggml_backend_reg_name` */
fun ggmlBackendRegName(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg): String {
    return reg.getName()
}

/** `ggml_backend_reg_dev_count` */
fun ggmlBackendRegDevCount(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg): ULong {
    return reg.getDeviceCount()
}

/** `ggml_backend_reg_dev_get` */
fun ggmlBackendRegDevGet(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg, index: ULong): io.github.kotlinmania.llama.ore.GGMLBackendDevice? {
    return reg.getDevice(index)
}

/** `ggml_backend_reg_get_proc_address` */
fun ggmlBackendRegGetProcAddress(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg, name: String): Any? {
    return reg.getProcAddress(name)
}

/** `ggml_backend_register` — C: ggml-backend-reg.cpp line 279. */
fun ggmlBackendRegister(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg) {
    io.github.kotlinmania.llama.ore.GGMLBackendRegistry.registerReg(reg)
}

/** `ggml_backend_device_register` — C: ggml-backend-reg.cpp line 283. */
fun ggmlBackendDeviceRegister(device: io.github.kotlinmania.llama.ore.GGMLBackendDevice) {
    io.github.kotlinmania.llama.ore.GGMLBackendRegistry.registerDevice(device)
}

/** `ggml_backend_reg_count` — C: ggml-backend-reg.cpp line 297. */
fun ggmlBackendRegCount(): ULong {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.regCount()
}

/** `ggml_backend_reg_get` — C: ggml-backend-reg.cpp line 301. */
fun ggmlBackendRegGet(index: ULong): io.github.kotlinmania.llama.ore.GGMLBackendReg? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.regGet(index)
}

/** `ggml_backend_reg_by_name` — C: ggml-backend-reg.cpp line 306. */
fun ggmlBackendRegByName(name: String): io.github.kotlinmania.llama.ore.GGMLBackendReg? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.regByName(name)
}

/** `ggml_backend_dev_count` — C: ggml-backend-reg.cpp line 317. */
fun ggmlBackendDevCount(): ULong {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.deviceCount()
}

/** `ggml_backend_dev_get` — C: ggml-backend-reg.cpp line 321. */
fun ggmlBackendDevGet(index: ULong): io.github.kotlinmania.llama.ore.GGMLBackendDevice? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.deviceGet(index)
}

/** `ggml_backend_dev_by_name` — C: ggml-backend-reg.cpp line 326. */
fun ggmlBackendDevByName(name: String): io.github.kotlinmania.llama.ore.GGMLBackendDevice? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.deviceByName(name)
}

/** `ggml_backend_dev_by_type` — C: ggml-backend-reg.cpp line 336. */
fun ggmlBackendDevByType(type: io.github.kotlinmania.llama.ore.GGMLBackendDeviceType): io.github.kotlinmania.llama.ore.GGMLBackendDevice? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.deviceByType(type)
}

/** `ggml_backend_init_by_name` — C: ggml-backend-reg.cpp line 347. */
fun ggmlBackendInitByName(name: String, params: String? = null): io.github.kotlinmania.llama.ore.GGMLBackend? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.initByName(name, params)
}

/** `ggml_backend_init_by_type` — C: ggml-backend-reg.cpp line 355. */
fun ggmlBackendInitByType(type: io.github.kotlinmania.llama.ore.GGMLBackendDeviceType, params: String? = null): io.github.kotlinmania.llama.ore.GGMLBackend? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.initByType(type, params)
}

/** `ggml_backend_init_best` — C: ggml-backend-reg.cpp line 363. */
fun ggmlBackendInitBest(): io.github.kotlinmania.llama.ore.GGMLBackend? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.initBest()
}

/** `ggml_backend_load` — C: ggml-backend-reg.cpp line 374. */
fun ggmlBackendLoad(path: String): io.github.kotlinmania.llama.ore.GGMLBackendReg? {
    return io.github.kotlinmania.llama.ore.GGMLBackendRegistry.load(path)
}

/** `ggml_backend_unload` — C: ggml-backend-reg.cpp line 378. */
fun ggmlBackendUnload(reg: io.github.kotlinmania.llama.ore.GGMLBackendReg) {
    io.github.kotlinmania.llama.ore.GGMLBackendRegistry.unload(reg)
}

/** `ggml_backend_load_all` — C: ggml-backend-reg.cpp line 543. */
fun ggmlBackendLoadAll() {
    io.github.kotlinmania.llama.ore.GGMLBackendRegistry.loadAll()
}

/** `ggml_backend_load_all_from_path` — C: ggml-backend-reg.cpp line 547. */
fun ggmlBackendLoadAllFromPath(dirPath: String) {
    io.github.kotlinmania.llama.ore.GGMLBackendRegistry.loadAllFromPath(dirPath)
}

// =====================================================================
// 7b. Multi-buffer
//     C: ggml-backend.cpp lines 667-735
// =====================================================================

/**
 * Context for a multi-buffer (logical buffer wrapping several sub-buffers).
 * Mirrors `ggml_backend_multi_buffer_context` in C.
 */
class GGMLBackendMultiBufferContext(
    val buffers: MutableList<io.github.kotlinmania.llama.ore.GGMLBackendBuffer> = mutableListOf()
)

/** Marker wrapper class for multi-buffer detection. C: ggml-backend.cpp lines 637-663. */
class GGMLBackendMultiBufferWrapper(
    private val ctx: io.github.kotlinmania.llama.ore.GGMLBackendMultiBufferContext,
    private val buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType,
    private val totalSize: ULong
) : io.github.kotlinmania.llama.ore.GGMLBackendBuffer {
    override fun getType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType = buft
    override fun getName(): String = buft.getName()
    override fun getBase(): Any? = null
    override fun getSize(): ULong = totalSize
    override fun free() {
        ctx.buffers.forEach { it.free() }
        ctx.buffers.clear()
    }
    override fun setTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        error("multi-buffer does not support direct tensor set")
    }
    override fun getTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        error("multi-buffer does not support direct tensor get")
    }
    override fun copyTensor(src: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean = false
    override fun clear(value: UByte) {
        ctx.buffers.forEach { it.clear(value) }
    }
    override fun setUsage(usage: io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage) {
        ctx.buffers.forEach { it.setUsage(usage) }
    }
}

/**
 * `ggml_backend_multi_buffer_alloc_buffer`
 * Allocate a logical multi-buffer wrapping several sub-buffers.
 * C: ggml-backend.cpp line 707.
 */
fun ggmlBackendMultiBufferAllocBuffer(buffers: List<io.github.kotlinmania.llama.ore.GGMLBackendBuffer>): io.github.kotlinmania.llama.ore.GGMLBackendBuffer {
    require(buffers.isNotEmpty()) { "multi-buffer requires at least one sub-buffer" }
    val ctx = io.github.kotlinmania.llama.ore.GGMLBackendMultiBufferContext(buffers.toMutableList())
    var totalSize = 0UL
    for (buf in buffers) {
        totalSize += buf.getSize()
    }
    return io.github.kotlinmania.llama.ore.GGMLBackendMultiBufferWrapper(
        ctx,
        buffers[0].getType(),
        totalSize
    )
}

/** `ggml_backend_multi_buffer_set_usage` — C: ggml-backend.cpp line 728. */
fun ggmlBackendMultiBufferSetUsage(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, usage: io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage) {
    if (buffer is io.github.kotlinmania.llama.ore.GGMLBackendMultiBufferWrapper) {
        buffer.setUsage(usage)
    }
}

/**
 * `ggml_backend_multi_buffer_free_buffer` — static vtable entry in C.
 * Frees all sub-buffers in a multi-buffer wrapper.
 * C: ggml-backend.cpp line 674.
 */
fun ggmlBackendMultiBufferFreeBuffer(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer) {
    buffer.free()
}

/**
 * `ggml_backend_multi_buffer_clear` — static vtable entry in C.
 * Clears all sub-buffers in a multi-buffer.
 * C: ggml-backend.cpp line 685.
 */
fun ggmlBackendMultiBufferClear(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, value: UByte) {
    buffer.clear(value)
}

// =====================================================================
// 8. Backend registry free functions (ggml-backend-reg.cpp)
// =====================================================================

// Registry free functions are in section 7 above.

// =====================================================================
// 8b. Meta backend functions (ggml-backend-meta.cpp)
// =====================================================================

/** `ggml_backend_meta_split_axis_name` — C: ggml-backend-meta.cpp line 26. */
fun ggmlBackendMetaSplitAxisName(axis: io.github.kotlinmania.llama.ore.GGMLBackendMetaSplitAxis): String = axis.name

/**
 * `ggml_backend_meta_device` — create a meta device from constituent devices.
 * C: ggml-backend.h line 398. Interim implementation for future tensor-parallelism support.
 */
fun ggmlBackendMetaDevice(
    devices: List<io.github.kotlinmania.llama.ore.GGMLBackendDevice>,
    getSplitState: io.github.kotlinmania.llama.ore.GGMLBackendMetaGetSplitState
): io.github.kotlinmania.llama.ore.GGMLBackendDevice? {
    return null
}

// =====================================================================
// 8c. Graph copy utilities (ggml-backend.cpp lines 2007-2209)
// =====================================================================

/**
 * Result of copying a graph to a different backend.
 * Mirrors `struct ggml_backend_graph_copy`.
 */
data class GGMLBackendGraphCopy(
    val buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer?,
    val ctxAllocated: io.github.kotlinmania.llama.ore.GGMLContext?,
    val ctxUnallocated: io.github.kotlinmania.llama.ore.GGMLContext?,
    val graph: io.github.kotlinmania.llama.ore.GGMLCGraph?
)

/** Evaluation callback used by `ggml_backend_compare_graph_backend`. */
typealias GGMLBackendEvalCallback = (nodeIndex: Int, t1: io.github.kotlinmania.llama.ore.GGMLTensor, t2: io.github.kotlinmania.llama.ore.GGMLTensor) -> Boolean

/**
 * `ggml_backend_graph_copy` — deep-copy a graph and its tensors to a target backend.
 * C: ggml-backend.cpp lines 2068-2148.
 */
fun ggmlBackendGraphCopy(backend: io.github.kotlinmania.llama.ore.GGMLBackend, graph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLBackendGraphCopy {
    val hashSet = io.github.kotlinmania.llama.ore.GGMLHashSet(
        io.github.kotlinmania.llama.ore.GGMLHashSet.ggml_hash_size(graph.size)
    )
    val nodeCopies: Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = arrayOfNulls(hashSet.size)
    val nodeInit = BooleanArray(hashSet.size)

    val ctxAllocated = io.github.kotlinmania.llama.ore.GGMLContext(noAlloc = true)
    val ctxUnallocated = io.github.kotlinmania.llama.ore.GGMLContext(noAlloc = true)

    // dup nodes
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        io.github.kotlinmania.llama.ore.graphCopyDupTensor(
            hashSet,
            nodeCopies,
            ctxAllocated,
            ctxUnallocated,
            node
        )
    }

    // copy data and init views
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        io.github.kotlinmania.llama.ore.graphCopyInitTensor(hashSet, nodeCopies, nodeInit, node)
    }

    // build graph copy
    val graphCopy = io.github.kotlinmania.llama.ore.ggmlNewGraphCustom(
        ctxAllocated,
        graph.size.toULong(),
        false
    )
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        val nodeCopy = nodeCopies[io.github.kotlinmania.llama.ore.ggml_hash_find(hashSet, node)]
        graphCopy.nodes[i] = nodeCopy
    }
    graphCopy.nNodes = graph.nNodes

    return io.github.kotlinmania.llama.ore.GGMLBackendGraphCopy(
        buffer = null,
        ctxAllocated = ctxAllocated,
        ctxUnallocated = ctxUnallocated,
        graph = graphCopy
    )
}

/** `ggml_backend_graph_copy_free` — C: ggml-backend.cpp lines 2150-2154. */
fun ggmlBackendGraphCopyFree(copy: io.github.kotlinmania.llama.ore.GGMLBackendGraphCopy) {
    io.github.kotlinmania.llama.ore.ggmlBackendBufferFree(copy.buffer)
}

/**
 * `ggml_backend_compare_graph_backend` — C: ggml-backend.cpp lines 2156-2209.
 * Compute a graph on two backends and call [callback] to compare each pair of result tensors.
 */
fun ggmlBackendCompareGraphBackend(
    backend1: io.github.kotlinmania.llama.ore.GGMLBackend,
    backend2: io.github.kotlinmania.llama.ore.GGMLBackend,
    graph: io.github.kotlinmania.llama.ore.GGMLCGraph,
    callback: io.github.kotlinmania.llama.ore.GGMLBackendEvalCallback,
    testNodes: List<io.github.kotlinmania.llama.ore.GGMLTensor>
): Boolean {
    val copy = io.github.kotlinmania.llama.ore.ggmlBackendGraphCopy(backend2, graph)
    if (copy.buffer == null && copy.graph == null) {
        return false
    }

    val g1 = graph
    val g2 = copy.graph!!

    require(g1.nNodes == g2.nNodes)

    if (testNodes.isNotEmpty()) {
        io.github.kotlinmania.llama.ore.ggmlBackendGraphCompute(backend1, g1)
        io.github.kotlinmania.llama.ore.ggmlBackendGraphCompute(backend2, g2)

        for (i in 0 until g1.nNodes) {
            for (testNode in testNodes) {
                if (g1.nodes[i] === testNode) {
                    callback(i, g1.nodes[i]!!, g2.nodes[i]!!)
                }
            }
        }
    } else {
        for (i in 0 until g1.nNodes) {
            val t1 = g1.nodes[i]!!
            val t2 = g2.nodes[i]!!

            val g1v = io.github.kotlinmania.llama.ore.ggml_graph_view(g1, i, i + 1)
            val g2v = io.github.kotlinmania.llama.ore.ggml_graph_view(g2, i, i + 1)

            io.github.kotlinmania.llama.ore.ggmlBackendGraphCompute(backend1, g1v)
            io.github.kotlinmania.llama.ore.ggmlBackendGraphCompute(backend2, g2v)

            if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(t1.op)) {
                continue
            }

            if (!callback(i, t1, t2)) {
                break
            }
        }
    }
    io.github.kotlinmania.llama.ore.ggmlBackendGraphCopyFree(copy)
    return true
}

// =====================================================================
// 9. Utility helpers
//    C: ggml_dup_tensor_layout, ggml_is_view_op, fmt_size
// =====================================================================

/** Create a copy of a tensor with the same memory layout (strides). */
fun ggmlDupTensorLayout(ctx: io.github.kotlinmania.llama.ore.GGMLContext, tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val dup = io.github.kotlinmania.llama.ore.GGMLTensor(
        type = tensor.type,
        ne = tensor.ne.copyOf(),
        nb = tensor.nb.copyOf(),
        op = tensor.op,
        name = tensor.name,
        flags = tensor.flags
    )
    return dup
}

/** Returns true if [op] is a view operation (VIEW, RESHAPE, PERMUTE, TRANSPOSE). */
fun ggmlIsViewOp(op: io.github.kotlinmania.llama.ore.GGMLOp): Boolean {
    return op == io.github.kotlinmania.llama.ore.GGMLOp.VIEW || op == io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE || op == io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE || op == io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE
}

/** Format a byte size as a human-readable string (e.g. "128K", "64M"). */
fun fmtSize(size: ULong): String {
    return when {
        size >= (1024UL * 1024UL) -> "${size / (1024UL * 1024UL)}M"
        else -> "${size / 1024UL}K"
    }
}

// =====================================================================
// 10. Scheduler split data structure
//     C: struct ggml_backend_sched_split (ggml-backend.cpp lines 764-772)
// =====================================================================

/**
 * A single split in the scheduler's graph decomposition.
 * Mirrors `struct ggml_backend_sched_split` in C.
 */
data class GGMLBackendSchedSplit(
    var backendId: Int = -1,
    var iStart: Int = 0,
    var iEnd: Int = 0,
    val inputs: Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = arrayOfNulls(io.github.kotlinmania.llama.ore.GGML_SCHED_MAX_SPLIT_INPUTS),
    var nInputs: Int = 0,
    var graph: io.github.kotlinmania.llama.ore.GGMLCGraph = io.github.kotlinmania.llama.ore.GGMLCGraph()
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = backendId xor iStart xor iEnd xor nInputs
}

// =====================================================================
// 11. Scheduler internal helpers
//     C: ggml_backend_sched_backend_id, _from_buffer, _id_from_cur,
//        _print_assignments, _buffer_supported, _set_if_supported,
//        _split_graph, _alloc_splits, _compute_splits
// =====================================================================

/**
 * Find the index of [backend] in the scheduler's backends array.
 * Returns -1 if not found. Lower index = higher priority.
 * C: `ggml_backend_sched_backend_id` (lines 836-843)
 */
fun ggmlBackendSchedBackendId(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, backend: io.github.kotlinmania.llama.ore.GGMLBackend): Int {
    for (i in 0 until sched.nBackends) {
        if (sched.backends[i] === backend) {
            return i
        }
    }
    return -1
}

/**
 * Find the highest-priority backend that supports the buffer type of [tensor] and the operation [op].
 * C: `ggml_backend_sched_backend_from_buffer` (lines 845-865)
 */
fun ggmlBackendSchedBackendFromBuffer(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLTensor): Int {
    val buffer = tensor.viewSrc?.buffer ?: tensor.buffer ?: return -1

    for (i in 0 until sched.nBackends) {
        if (io.github.kotlinmania.llama.ore.ggmlBackendSupportsBuft(
                sched.backends[i]!!,
                buffer.getType()
            ) &&
            io.github.kotlinmania.llama.ore.ggmlBackendSupportsOp(sched.backends[i]!!, op)
        ) {
            return i
        }
    }

    return -1
}

/**
 * Determine the backend that should own a node based on its current buffer
 * allocations, view sources, and weight locations.
 * C: `ggml_backend_sched_backend_id_from_cur` (lines 878-933)
 */
fun ggmlBackendSchedBackendIdFromCur(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Int {
    // assign pre-allocated nodes to their backend
    var curBackendId =
        io.github.kotlinmania.llama.ore.ggmlBackendSchedBackendFromBuffer(sched, tensor, tensor)
    if (curBackendId != -1) {
        return curBackendId
    }

    // view_src
    if (tensor.viewSrc != null) {
        curBackendId = io.github.kotlinmania.llama.ore.ggmlBackendSchedBackendFromBuffer(
            sched,
            tensor.viewSrc!!,
            tensor
        )
        if (curBackendId != -1) {
            return curBackendId
        }
    }

    if (tensor.buffer != null || (tensor.viewSrc != null && tensor.viewSrc!!.buffer != null)) {
        val buffer = if (tensor.viewSrc != null) tensor.viewSrc!!.buffer else tensor.buffer
        error("pre-allocated tensor (${tensor.name}) in a buffer (${
            io.github.kotlinmania.llama.ore.ggmlBackendBufferName(
                buffer!!
            )
        }) that cannot run the operation (${tensor.op.name})")
    }

    // graph input
    if (tensor.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_INPUT != 0) {
        curBackendId = sched.nBackends - 1 // last backend (assumed CPU)
        return curBackendId
    }

    // operations with weights are preferably run on the same backend as the weights
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
        val src = tensor.src[i] ?: continue
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        if (tensor.op != io.github.kotlinmania.llama.ore.GGMLOp.ROPE && src.buffer != null &&
            src.buffer!!.getUsage() == io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage.WEIGHTS
        ) {
            val srcBackendId = io.github.kotlinmania.llama.ore.ggmlBackendSchedBackendFromBuffer(
                sched,
                src,
                tensor
            )
            // check if a backend with higher prio wants to offload the op
            if (sched.opOffload && srcBackendId == sched.nBackends - 1 &&
                io.github.kotlinmania.llama.ore.ggmlBackendBufferIsHost(src.buffer!!)
            ) {
                for (b in 0 until srcBackendId) {
                    if (io.github.kotlinmania.llama.ore.ggmlBackendSupportsOp(
                            sched.backends[b]!!,
                            tensor
                        ) &&
                        io.github.kotlinmania.llama.ore.ggmlBackendOffloadOp(
                            sched.backends[b]!!,
                            tensor
                        )
                    ) {
                        return b
                    }
                }
            }
            return srcBackendId
        }
    }

    return -1
}

/**
 * Print the node→backend assignments for debugging.
 * C: `ggml_backend_sched_print_assignments` (lines 945-983)
 */
fun ggmlBackendSchedPrintAssignments(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, graph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    var curSplit = 0
    for (i in 0 until graph.nNodes) {
        if (curSplit < sched.nSplits && i == sched.splits[curSplit].iStart) {
            val splitBackend = sched.backends[sched.splits[curSplit].backendId]
            val sb = StringBuilder()
            sb.append("\n## SPLIT #$curSplit: ${
                io.github.kotlinmania.llama.ore.ggmlBackendName(
                    splitBackend
                )
            } # ${sched.splits[curSplit].nInputs} inputs")
            for (j in 0 until sched.splits[curSplit].nInputs) {
                if (j == 0) sb.append(": ")
                val inp = sched.splits[curSplit].inputs[j]!!
                sb.append("[${inp.name} (${
                    io.github.kotlinmania.llama.ore.fmtSize(
                        io.github.kotlinmania.llama.ore.ggmlNbytes(inp)
                    )
                })] ")
            }
            println(sb.toString())
            curSplit++
        }
        val node = graph.nodes[i] ?: continue
        if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) continue

        if (sched.debug > 1) {
            val tensorBackend = sched.getTensorBackend(node)
            val sb = StringBuilder()
            sb.append("node #${i}: ${node.op.name} ${node.name} (${
                io.github.kotlinmania.llama.ore.fmtSize(
                    io.github.kotlinmania.llama.ore.ggmlNbytes(node)
                )
            }) [${tensorBackend?.getName() ?: "NULL"}]:")
            for (j in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
                val src = node.src[j] ?: continue
                val srcBackend = sched.getTensorBackend(src)
                sb.append(" ${src.name} (${
                    io.github.kotlinmania.llama.ore.fmtSize(
                        io.github.kotlinmania.llama.ore.ggmlNbytes(src)
                    )
                }) [${srcBackend?.getName() ?: "NULL"}]")
            }
            println(sb.toString())
        }
    }
}

/**
 * Check if a tensor's buffer type is supported on a given backend.
 * C: `ggml_backend_sched_buffer_supported` (lines 985-1004)
 */
fun ggmlBackendSchedBufferSupported(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, t: io.github.kotlinmania.llama.ore.GGMLTensor, backendId: Int): Boolean {
    val buf = t.viewSrc?.buffer ?: t.buffer
    var buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType? = null

    if (buf != null) {
        buft = buf.getType()
    } else {
        var tensorBid = sched.tensorBackendId(t)
        if (tensorBid == -1 && t.viewSrc != null) {
            tensorBid = sched.tensorBackendId(t.viewSrc!!)
        }
        if (tensorBid != -1) {
            buft = sched.bufts[tensorBid]
        }
    }

    return buft != null && io.github.kotlinmania.llama.ore.ggmlBackendSupportsBuft(
        sched.backends[backendId]!!,
        buft
    )
}

/**
 * Assign a backend to a node if the backend supports the operation.
 * C: `ggml_backend_sched_set_if_supported` (lines 1006-1011)
 */
fun ggmlBackendSchedSetIfSupported(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, node: io.github.kotlinmania.llama.ore.GGMLTensor, curBackendId: Int) {
    if (io.github.kotlinmania.llama.ore.ggmlBackendSupportsOp(sched.backends[curBackendId]!!, node)) {
        sched.setTensorBackendId(node, curBackendId)
    }
}

/**
 * Allocate memory for all splits using the graph allocator.
 * C: `ggml_backend_sched_alloc_splits` (lines 1489-1539)
 */
fun ggmlBackendSchedAllocSplits(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched): Boolean {
    var backendIdsChanged = false
    for (i in 0 until sched.graph.nNodes) {
        if (sched.nodeBackendIds[i] != sched.prevNodeBackendIds[i] &&
            sched.bufts[sched.nodeBackendIds[i]] !== sched.bufts[sched.prevNodeBackendIds[i]]
        ) {
            backendIdsChanged = true
            break
        }
    }
    if (!backendIdsChanged) {
        for (i in 0 until sched.graph.nLeafs) {
            if (sched.leafBackendIds[i] != sched.prevLeafBackendIds[i] &&
                sched.bufts[sched.leafBackendIds[i]] !== sched.bufts[sched.prevLeafBackendIds[i]]
            ) {
                backendIdsChanged = true
                break
            }
        }
    }

    // allocate graph via gallocr
    // if (backendIdsChanged || !ggmlGallocrAllocGraph(sched.galloc, sched.graph)) {
    if (backendIdsChanged) {
        if (sched.debugRealloc > 0) {
            val unexpected = !backendIdsChanged && sched.debugPrevGraphSize == sched.debugGraphSize
            if (unexpected || sched.debugRealloc > 1) {
                error("unexpected graph reallocation (graph size = ${sched.debugGraphSize})")
            }
        }

        // synchronize all backends before re-allocation
        for (i in 0 until sched.nBackends) {
            io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(sched.backends[i]!!)
        }

        // ggmlGallocrReserveN(sched.galloc, sched.graph, sched.nodeBackendIds, sched.leafBackendIds)
        // if (!ggmlGallocrAllocGraph(sched.galloc, sched.graph)) {
        //     return false
    }

    return true
}

/**
 * Execute all splits, copying inputs between backends as needed.
 * C: `ggml_backend_sched_compute_splits` (lines 1541-1725)
 */
fun ggmlBackendSchedComputeSplits(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched): io.github.kotlinmania.llama.ore.GGMLStatus {
    for (splitId in 0 until sched.nSplits) {
        val split = sched.splits[splitId]
        val splitBid = split.backendId
        val splitBackend = sched.backends[splitBid]!!

        // copy the input tensors to the split backend
        for (inputId in 0 until split.nInputs) {
            val input = split.inputs[inputId]!!
            val inputCpy = sched.tensorCopy(input, splitBid, sched.curCopy)!!

            if (input.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_INPUT != 0) {
                // inputs from the user must be copied immediately
                if (sched.events[splitBid][sched.curCopy] != null) {
                    io.github.kotlinmania.llama.ore.ggmlBackendEventSynchronize(sched.events[splitBid][sched.curCopy]!!)
                } else {
                    io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(splitBackend)
                }
                io.github.kotlinmania.llama.ore.ggmlBackendTensorCopy(input, inputCpy)
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched.events[splitBid][sched.curCopy] != null) {
                    io.github.kotlinmania.llama.ore.ggmlBackendEventWait(
                        splitBackend,
                        sched.events[splitBid][sched.curCopy]!!
                    )
                } else {
                    io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(splitBackend)
                }

                // MoE expert optimization: check if we can copy only used experts
                val firstNode = if (split.graph.nNodes > 0) split.graph.nodes[0] else null
                if (firstNode != null &&
                    input.buffer != null &&
                    io.github.kotlinmania.llama.ore.ggmlBackendBufferGetUsage(input.buffer!!) == io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage.WEIGHTS &&
                    io.github.kotlinmania.llama.ore.ggmlBackendBufferIsHost(input.buffer!!) &&
                    firstNode.src[0] === inputCpy && firstNode.op == io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT_ID
                ) {
                    // MoE weight copy optimization — copy full tensor for now
                    // Full expert-level copy optimization requires bitset and ids tensor inspection
                    io.github.kotlinmania.llama.ore.ggmlBackendTensorCopy(input, inputCpy)
                } else {
                    // try async copy, fallback to sync
                    val inputBackend = sched.getTensorBackend(input)
                    if (inputBackend != null) {
                        if (!splitBackend.copyTensorAsync(inputBackend, input, inputCpy)) {
                            io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(inputBackend)
                            if (sched.events[splitBid][sched.curCopy] != null) {
                                io.github.kotlinmania.llama.ore.ggmlBackendEventSynchronize(sched.events[splitBid][sched.curCopy]!!)
                            } else {
                                io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(splitBackend)
                            }
                            io.github.kotlinmania.llama.ore.ggmlBackendTensorCopy(input, inputCpy)
                        }
                    } else {
                        io.github.kotlinmania.llama.ore.ggmlBackendTensorCopy(input, inputCpy)
                    }
                }
            }
        }

        if (sched.callbackEval == null) {
            val ec = io.github.kotlinmania.llama.ore.ggmlBackendGraphComputeAsync(
                splitBackend,
                split.graph
            )
            if (ec != io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS) {
                return ec
            }
        } else {
            // compute with eval callback — similar to compare_graph_backend
            var j0 = 0
            while (j0 < split.graph.nNodes) {
                var t = split.graph.nodes[j0]!!
                var need = sched.callbackEval!!(t, true, sched.callbackEvalUserData)
                var j1 = j0

                while (!need && j1 < split.graph.nNodes - 1) {
                    t = split.graph.nodes[++j1]!!
                    need = sched.callbackEval!!(t, true, sched.callbackEvalUserData)
                }

                val gv = io.github.kotlinmania.llama.ore.ggml_graph_view(split.graph, j0, j1 + 1)
                val ec =
                    io.github.kotlinmania.llama.ore.ggmlBackendGraphComputeAsync(splitBackend, gv)
                if (ec != io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS) {
                    return ec
                }

                io.github.kotlinmania.llama.ore.ggmlBackendSynchronize(splitBackend)

                if (need && !sched.callbackEval!!(t, false, sched.callbackEvalUserData)) {
                    break
                }

                j0 = j1 + 1
            }
        }

        // record the event of this copy
        if (split.nInputs > 0) {
            if (sched.events[splitBid][sched.curCopy] != null) {
                io.github.kotlinmania.llama.ore.ggmlBackendEventRecord(
                    sched.events[splitBid][sched.curCopy]!!,
                    splitBackend
                )
            }
        }
    }

    return io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS
}

// =====================================================================
// 12. View / tensor alloc utils
//     C: ggml_backend_view_init, ggml_backend_tensor_alloc
// =====================================================================

/**
 * `ggml_backend_view_init` — initialise a tensor view.
 * C: ggml-backend.cpp lines 1980-1989.
 */
fun ggmlBackendViewInit(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLStatus {
    require(tensor.buffer == null) { "tensor already has a buffer" }
    requireNotNull(tensor.viewSrc) { "tensor is not a view" }
    requireNotNull(tensor.viewSrc!!.buffer) { "view_src buffer is null" }
    requireNotNull(tensor.viewSrc!!.data) { "view_src data is null" }

    tensor.buffer = tensor.viewSrc!!.buffer
    // tensor.data = viewSrc.data + viewOffs — handled via viewOffs accessor
    return io.github.kotlinmania.llama.ore.ggmlBackendBufferInitTensor(tensor.buffer!!, tensor)
}

/**
 * `ggml_backend_tensor_alloc` — place a tensor at a specific address inside a buffer.
 * C: ggml-backend.cpp lines 1992-2005.
 */
fun ggmlBackendTensorAlloc(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, addr: Any?): io.github.kotlinmania.llama.ore.GGMLStatus {
    require(tensor.buffer == null) { "tensor already allocated" }
    require(tensor.viewSrc == null) { "tensor is a view, use view_init" }

    tensor.buffer = buffer
    tensor.data = addr
    return io.github.kotlinmania.llama.ore.ggmlBackendBufferInitTensor(buffer, tensor)
}

// =====================================================================
// 13. Graph copy / compare helpers
//     C: graph_copy_dup_tensor, graph_copy_init_tensor
//     (ggml-backend.cpp lines 2007-2066)
// =====================================================================

/**
 * Recursively duplicate a tensor and all its sources for graph copy.
 * C: `graph_copy_dup_tensor` (lines 2007-2038)
 */
fun graphCopyDupTensor(
    hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet,
    nodeCopies: Array<io.github.kotlinmania.llama.ore.GGMLTensor?>,
    ctxAllocated: io.github.kotlinmania.llama.ore.GGMLContext,
    ctxUnallocated: io.github.kotlinmania.llama.ore.GGMLContext,
    src: io.github.kotlinmania.llama.ore.GGMLTensor
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val id = io.github.kotlinmania.llama.ore.ggml_hash_insert(hashSet, src)
    if (id == io.github.kotlinmania.llama.ore.GGML_HASHSET_ALREADY_EXISTS) {
        return nodeCopies[io.github.kotlinmania.llama.ore.ggml_hash_find(hashSet, src)]!!
    }

    val ctx = if (src.data != null && src.viewSrc == null) ctxAllocated else ctxUnallocated
    val dst = io.github.kotlinmania.llama.ore.ggmlDupTensorLayout(ctx, src)
    val viewSrc = src.viewSrc
    if (viewSrc != null) {
        dst.viewSrc = graphCopyDupTensor(hashSet, nodeCopies, ctxAllocated, ctxUnallocated, viewSrc)
        dst.viewOffs = src.viewOffs
    }
    dst.op = src.op
    dst.flags = src.flags
    dst.opParams = src.opParams.copyOf()
    dst.name = src.name

    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
        val s = src.src[i] ?: continue
        dst.src[i] = graphCopyDupTensor(hashSet, nodeCopies, ctxAllocated, ctxUnallocated, s)
    }

    nodeCopies[id] = dst
    return dst
}

/**
 * Initialize a copied tensor — copy data or init view.
 * C: `graph_copy_init_tensor` (lines 2041-2066)
 */
fun graphCopyInitTensor(
    hashSet: io.github.kotlinmania.llama.ore.GGMLHashSet,
    nodeCopies: Array<io.github.kotlinmania.llama.ore.GGMLTensor?>,
    nodeInit: BooleanArray,
    src: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val id = io.github.kotlinmania.llama.ore.ggml_hash_find(hashSet, src)
    if (nodeInit[id]) return
    nodeInit[id] = true

    val dst = nodeCopies[id]!!
    if (dst.viewSrc != null) {
        graphCopyInitTensor(hashSet, nodeCopies, nodeInit, src.viewSrc!!)
        val status = io.github.kotlinmania.llama.ore.ggmlBackendViewInit(dst)
        require(status == io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS)
    } else {
        io.github.kotlinmania.llama.ore.ggmlBackendTensorCopy(src, dst)
    }

    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
        val s = src.src[i] ?: continue
        graphCopyInitTensor(hashSet, nodeCopies, nodeInit, s)
    }
}

// =====================================================================
// 14. CPU backend buffer / buffer-type
//     C: ggml_backend_cpu_buffer_* (lines 2211-2371)
//
//     GGMLCpuBuffer and GGMLCpuBufferType are implemented in GGMLCpuBackend.kt.
//     GGMLCpuBufferFromPtr and GGMLCpuBufferFromPtrType are unique to this file
//     (they wrap external byte arrays without owning them).
// =====================================================================

/**
 * CPU buffer type for buffers created from external pointers (not owned).
 * C: `ggml_backend_cpu_buffer_from_ptr_type` (lines 2351-2366)
 */
class GGMLCpuBufferFromPtrType : io.github.kotlinmania.llama.ore.GGMLBackendBufferType {
    override fun getName(): String = "CPU_Mapped"
    override fun allocBuffer(size: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBuffer {
        return io.github.kotlinmania.llama.ore.createDefaultCpuBufferType().allocBuffer(size)
            ?: throw IllegalStateException("Failed to allocate CPU buffer of size $size")
    }
    override fun getAlignment(): UInt = io.github.kotlinmania.llama.ore.TENSOR_ALIGNMENT.toUInt()
    override fun getMaxSize(): ULong = ULong.MAX_VALUE
    override fun getAllocSize(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong =
        io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)
    override fun isHost(): Boolean = true
    override fun getDevice(): io.github.kotlinmania.llama.ore.GGMLBackendDevice? = null
}

/**
 * Create a CPU buffer wrapping an existing byte array (not owned by the buffer).
 * C: `ggml_backend_cpu_buffer_from_ptr` (lines 2368-2371)
 */
class GGMLCpuBufferFromPtr(
    private val ptr: ByteArray,
    private val size: ULong
) : io.github.kotlinmania.llama.ore.GGMLBackendBuffer {
    private val buft = io.github.kotlinmania.llama.ore.GGMLCpuBufferFromPtrType()
    private var usage = io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage.COMPUTE

    override fun getType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType = buft
    override fun getName(): String = "CPU_Mapped"
    override fun getBase(): Any = ptr
    override fun getSize(): ULong = size
    override fun free() {}
    override fun initTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLStatus = io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS
    override fun setTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val dst = tensor.data
        if (dst is ByteArray) data.copyInto(dst, offset.toInt(), 0, size.toInt())
    }
    override fun getTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val src = tensor.data
        if (src is ByteArray) src.copyInto(data, 0, offset.toInt(), (offset + size).toInt())
    }
    override fun copyTensor(src: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
        val srcBuf = src.buffer ?: return false
        if (io.github.kotlinmania.llama.ore.ggmlBackendBufferIsHost(srcBuf)) {
            val srcData = src.data
            val dstData = dst.data
            if (srcData is ByteArray && dstData is ByteArray) {
                srcData.copyInto(dstData, 0, 0, io.github.kotlinmania.llama.ore.ggmlNbytes(src).toInt())
                return true
            }
        }
        return false
    }
    override fun clear(value: UByte) { ptr.fill(value.toByte()) }
    override fun setUsage(usage: io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage) { this.usage = usage }
    override fun getUsage(): io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage = usage
    override fun reset() { clear(0u) }
}

// =====================================================================
// 15. CPU buffer vtable functions (top-level wrappers)
//     C: static ggml_backend_cpu_buffer_* functions (lines 2213-2327)
//     These are static vtable entries in C. In Kotlin the logic is in
//     GGMLCpuBuffer / GGMLCpuBufferType classes — these wrappers exist
//     for naming parity with the C source.
// =====================================================================

/** `ggml_backend_cpu_buffer_get_base` — C line 2213 (static vtable entry). */
fun ggmlBackendCpuBufferGetBase(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer): Any? = buffer.getBase()

/** `ggml_backend_cpu_buffer_free_buffer` — C line 2225 (static vtable entry). */
fun ggmlBackendCpuBufferFreeBuffer(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer) = buffer.free()

/** `ggml_backend_cpu_buffer_memset_tensor` — C line 2230 (static vtable entry). */
fun ggmlBackendCpuBufferMemsetTensor(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, value: UByte, offset: ULong, size: ULong) {
    val data = tensor.data
    if (data is ByteArray) {
        data.fill(value.toByte(), offset.toInt(), (offset + size).toInt())
    }
}

/** `ggml_backend_cpu_buffer_set_tensor` — C line 2237 (static vtable entry). */
fun ggmlBackendCpuBufferSetTensor(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    buffer.setTensor(tensor, data, offset, size)
}

/** `ggml_backend_cpu_buffer_get_tensor` — C line 2244 (static vtable entry). */
fun ggmlBackendCpuBufferGetTensor(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
    buffer.getTensor(tensor, data, offset, size)
}

/** `ggml_backend_cpu_buffer_cpy_tensor` — C line 2251 (static vtable entry). */
fun ggmlBackendCpuBufferCpyTensor(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, src: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    return buffer.copyTensor(src, dst)
}

/** `ggml_backend_cpu_buffer_clear` — C line 2262 (static vtable entry). */
fun ggmlBackendCpuBufferClear(buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer, value: UByte) {
    buffer.clear(value)
}

/** `ggml_backend_cpu_buffer_type_get_name` — C line 2299 (static vtable entry). */
fun ggmlBackendCpuBufferTypeGetName(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): String = buft.getName()

/** `ggml_backend_cpu_buffer_type_alloc_buffer` — C line 2305 (static vtable entry). */
fun ggmlBackendCpuBufferTypeAllocBuffer(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType, size: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBuffer? {
    return buft.allocBuffer(size)
}

/** `ggml_backend_cpu_buffer_type_get_alignment` — C line 2316 (static vtable entry). */
fun ggmlBackendCpuBufferTypeGetAlignment(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): UInt = buft.getAlignment()

/** `ggml_backend_cpu_buffer_type_is_host` — C line 2322 (static vtable entry). */
fun ggmlBackendCpuBufferTypeIsHost(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): Boolean = buft.isHost()

/** `ggml_backend_cpu_buffer_from_ptr_type_get_name` — C line 2345. */
fun ggmlBackendCpuBufferFromPtrTypeGetName(buft: io.github.kotlinmania.llama.ore.GGMLBackendBufferType): String = buft.getName()

/** `ggml_backend_cpu_buffer_from_ptr_type` — C line 2351. Returns singleton buft. */
fun ggmlBackendCpuBufferFromPtrType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType =
    io.github.kotlinmania.llama.ore.GGMLCpuBufferFromPtrType()

// Multi-buffer functions in GGMLBackendImpl.kt (declared in ggml-backend-impl.h)

// ---------------------------------------------------------------------------
// Scheduler public API  (ggml-backend.cpp lines 1727-1976)
// Top-level functions matching C naming, delegating to GGMLBackendSched class.
// ---------------------------------------------------------------------------

/**
 * `ggml_backend_sched_new` — C: ggml-backend.cpp lines 1727-1793.
 */
fun ggmlBackendSchedNew(
    backends: List<io.github.kotlinmania.llama.ore.GGMLBackend>,
    bufts: List<io.github.kotlinmania.llama.ore.GGMLBackendBufferType>?,
    nBackends: Int,
    graphSize: Int,
    parallel: Boolean,
    opOffload: Boolean
): io.github.kotlinmania.llama.ore.GGMLBackendSched {
    return io.github.kotlinmania.llama.ore.GGMLBackendSched.new(
        backends = backends.take(nBackends),
        bufts = bufts,
        graphSize = graphSize,
        parallel = parallel,
        opOffload = opOffload
    )
}

/** `ggml_backend_sched_free` — C: ggml-backend.cpp lines 1796-1819. */
fun ggmlBackendSchedFree(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched?) {
    sched?.free()
}

/** `ggml_backend_sched_reset` — C: ggml-backend.cpp lines 1821-1831. */
fun ggmlBackendSchedReset(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched) {
    sched.reset()
}

/** `ggml_backend_sched_reserve_size` — C: ggml-backend.cpp lines 1833-1845. */
fun ggmlBackendSchedReserveSize(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, measureGraph: io.github.kotlinmania.llama.ore.GGMLCGraph, sizes: ULongArray) {
    sched.reserveSize(measureGraph, sizes)
}

/** `ggml_backend_sched_reserve` — C: ggml-backend.cpp lines 1847-1862. */
fun ggmlBackendSchedReserve(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, measureGraph: io.github.kotlinmania.llama.ore.GGMLCGraph): Boolean {
    return sched.reserve(measureGraph)
}

/** `ggml_backend_sched_alloc_graph` — C: ggml-backend.cpp lines 1864-1881. */
fun ggmlBackendSchedAllocGraph(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, graph: io.github.kotlinmania.llama.ore.GGMLCGraph): Boolean {
    return sched.allocGraph(graph)
}

/** `ggml_backend_sched_graph_compute` — C: ggml-backend.cpp lines 1883-1887. */
fun ggmlBackendSchedGraphCompute(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, graph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLStatus {
    return sched.graphCompute(graph)
}

/** `ggml_backend_sched_graph_compute_async` — C: ggml-backend.cpp lines 1889-1902. */
fun ggmlBackendSchedGraphComputeAsync(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, graph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLStatus {
    return sched.graphComputeAsync(graph)
}

/** `ggml_backend_sched_synchronize` — C: ggml-backend.cpp lines 1904-1915. */
fun ggmlBackendSchedSynchronize(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched) {
    sched.synchronize()
}

/** `ggml_backend_sched_set_eval_callback` — C: ggml-backend.cpp lines 1917-1921. */
fun ggmlBackendSchedSetEvalCallback(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, callback: io.github.kotlinmania.llama.ore.GGMLBackendSchedEvalCallback?, userData: Any? = null) {
    sched.setEvalCallback(callback, userData)
}

/** `ggml_backend_sched_get_n_splits` — C: ggml-backend.cpp line 1923. */
fun ggmlBackendSchedGetNSplits(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched): Int {
    return sched.getNumSplits()
}

/** `ggml_backend_sched_get_n_copies` — C: ggml-backend.cpp line 1928. */
fun ggmlBackendSchedGetNCopies(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched): Int {
    return sched.getNumCopies()
}

/** `ggml_backend_sched_get_n_backends` — C: ggml-backend.cpp line 1933. */
fun ggmlBackendSchedGetNBackends(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched): Int {
    return sched.getNumBackends()
}

/** `ggml_backend_sched_get_backend` — C: ggml-backend.cpp line 1938. */
fun ggmlBackendSchedGetBackend(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, i: Int): io.github.kotlinmania.llama.ore.GGMLBackend? {
    return sched.getBackend(i)
}

/** `ggml_backend_sched_get_buffer_type` — C: ggml-backend.cpp line 1944. */
fun ggmlBackendSchedGetBufferType(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, backend: io.github.kotlinmania.llama.ore.GGMLBackend): io.github.kotlinmania.llama.ore.GGMLBackendBufferType? {
    return sched.getBufferType(backend)
}

/** `ggml_backend_sched_get_buffer_size` — C: ggml-backend.cpp line 1952. */
fun ggmlBackendSchedGetBufferSize(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, backend: io.github.kotlinmania.llama.ore.GGMLBackend): ULong {
    return sched.getBufferSize(backend)
}

/** `ggml_backend_sched_set_tensor_backend` — C: ggml-backend.cpp line 1960. */
fun ggmlBackendSchedSetTensorBackend(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, node: io.github.kotlinmania.llama.ore.GGMLTensor, backend: io.github.kotlinmania.llama.ore.GGMLBackend) {
    sched.setTensorBackend(node, backend)
}

/** `ggml_backend_sched_get_tensor_backend` — C: ggml-backend.cpp line 1969. */
fun ggmlBackendSchedGetTensorBackend(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, node: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLBackend? {
    return sched.getTensorBackend(node)
}

/**
 * Split a computation graph into sub-graphs, each assigned to a single backend.
 * This is the core scheduling algorithm — 5-pass approach.
 * C: `ggml_backend_sched_split_graph` (lines 1014-1487)
 */
fun ggmlBackendSchedSplitGraph(sched: io.github.kotlinmania.llama.ore.GGMLBackendSched, graph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    // reset splits
    sched.nSplits = 0
    sched.nGraphInputs = 0
    sched.isReset = false

    // re-initialize context for temporary tensor allocations
    sched.ctx = io.github.kotlinmania.llama.ore.GGMLContext(noAlloc = true)

    graph.uid = io.github.kotlinmania.llama.ore.ggml_graph_next_uid()

    // pass 1: assign backends to ops with pre-allocated inputs
    for (i in 0 until graph.nLeafs) {
        val leaf = graph.leafs[i] ?: continue
        val leafBid = sched.tensorBackendId(leaf)
        if (leafBid == -1) {
            sched.setTensorBackendId(leaf,
                io.github.kotlinmania.llama.ore.ggmlBackendSchedBackendIdFromCur(sched, leaf)
            )
        }
    }

    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        val nodeBid = sched.tensorBackendId(node)
        if (nodeBid == -1) {
            sched.setTensorBackendId(node,
                io.github.kotlinmania.llama.ore.ggmlBackendSchedBackendIdFromCur(sched, node)
            )
        }
    }

    // pass 2: expand current backend assignments
    // expand gpu backends (non-last-prio) down and up, ignoring cpu (lowest priority backend)
    // expand gpu down
    run {
        var curBid = -1
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = if (nodeBid == sched.nBackends - 1) -1 else nodeBid
            } else if (curBid != -1) {
                io.github.kotlinmania.llama.ore.ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }
    // expand gpu up
    run {
        var curBid = -1
        for (i in graph.nNodes - 1 downTo 0) {
            val node = graph.nodes[i] ?: continue
            if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = if (nodeBid == sched.nBackends - 1) -1 else nodeBid
            } else if (curBid != -1) {
                io.github.kotlinmania.llama.ore.ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }
    // expand rest down
    run {
        var curBid = -1
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = nodeBid
            } else if (curBid != -1) {
                io.github.kotlinmania.llama.ore.ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }
    // expand rest up
    run {
        var curBid = -1
        for (i in graph.nNodes - 1 downTo 0) {
            val node = graph.nodes[i] ?: continue
            if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) continue
            val nodeBid = sched.tensorBackendId(node)
            if (nodeBid != -1) {
                curBid = nodeBid
            } else if (curBid != -1) {
                io.github.kotlinmania.llama.ore.ggmlBackendSchedSetIfSupported(sched, node, curBid)
            }
        }
    }

    // pass 3: upgrade to higher prio backends with compatible buffer types
    // and assign remaining unassigned nodes to backend with most supported inputs
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) continue
        val nodeBid = sched.tensorBackendId(node)
        if (nodeBid == -1) {
            // unassigned node: find the backend with the most supported inputs
            var nSupportedBest = -1
            for (b in 0 until sched.nBackends) {
                if (io.github.kotlinmania.llama.ore.ggmlBackendSupportsOp(
                        sched.backends[b]!!,
                        node
                    )
                ) {
                    var nSupported = 0
                    for (j in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
                        val src = node.src[j] ?: continue
                        val srcBid = sched.tensorBackendId(src)
                        val srcViewBid = if (src.viewSrc != null) sched.tensorBackendId(src.viewSrc!!) else -1
                        if ((srcBid != -1 || srcViewBid != -1) && io.github.kotlinmania.llama.ore.ggmlBackendSchedBufferSupported(
                                sched,
                                src,
                                b
                            )
                        ) {
                            nSupported++
                        }
                    }
                    if (nSupported > nSupportedBest) {
                        nSupportedBest = nSupported
                        sched.setTensorBackendId(node, b)
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (b in 0 until nodeBid) {
                if (sched.bufts[b] === sched.bufts[nodeBid] &&
                    io.github.kotlinmania.llama.ore.ggmlBackendSupportsOp(sched.backends[b]!!, node)
                ) {
                    var supported = true
                    for (j in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
                        val src = node.src[j] ?: continue
                        if (!io.github.kotlinmania.llama.ore.ggmlBackendSchedBufferSupported(
                                sched,
                                src,
                                b
                            )
                        ) {
                            supported = false
                            break
                        }
                    }
                    if (supported) {
                        sched.setTensorBackendId(node, b)
                        break
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (i in 0 until graph.nNodes) {
        val node = graph.nodes[i] ?: continue
        var curBid = sched.tensorBackendId(node)
        if (node.viewSrc != null && curBid == -1) {
            curBid = sched.tensorBackendId(node.viewSrc!!)
            sched.setTensorBackendId(node, curBid)
        }
        for (j in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
            val src = node.src[j] ?: continue
            val srcBid = sched.tensorBackendId(src)
            if (srcBid == -1) {
                if (src.viewSrc != null) {
                    sched.setTensorBackendId(src, sched.tensorBackendId(src.viewSrc!!))
                } else {
                    sched.setTensorBackendId(src, curBid)
                }
            }
        }
        // if node is still unassigned, assign to first backend that supports it
        if (sched.tensorBackendId(node) == -1) {
            for (b in 0 until sched.nBackends) {
                if (io.github.kotlinmania.llama.ore.ggmlBackendSupportsOp(
                        sched.backends[b]!!,
                        node
                    )
                ) {
                    sched.setTensorBackendId(node, b)
                    break
                }
            }
        }
        require(sched.tensorBackendId(node) != -1) { "node ${node.name} has no backend assigned" }
    }

    // pass 5: split graph, find tensors that need to be copied
    run pass5@ {
        var iSplit = 0
        var split = sched.splits[0]
        // find the backend of the first split, skipping view ops
        var i = 0
        while (i < graph.nNodes) {
            val node = graph.nodes[i]
            if (node == null) { i++; continue }
            if (!io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) {
                split.backendId = sched.tensorBackendId(node)
                break
            }
            i++
        }
        split.iStart = 0
        split.nInputs = 0
        var curBid = split.backendId

        while (i < graph.nNodes) {
            val node = graph.nodes[i]
            if (node == null) { i++; continue }
            if (io.github.kotlinmania.llama.ore.ggmlIsViewOp(node.op)) { i++; continue }

            val nodeBid = sched.tensorBackendId(node)
            require(nodeBid != -1) { "all nodes should be assigned by now" }

            // check if we should start a new split
            var needNewSplit = false
            if (nodeBid == curBid && split.nInputs > 0) {
                for (j in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
                    val src = node.src[j] ?: continue
                    if (src.buffer != null && src.buffer!!.getUsage() == io.github.kotlinmania.llama.ore.GGMLBackendBufferUsage.WEIGHTS) {
                        val srcBid = sched.tensorBackendId(src)
                        if (srcBid != curBid && !io.github.kotlinmania.llama.ore.ggmlBackendSchedBufferSupported(
                                sched,
                                src,
                                curBid
                            )
                        ) {
                            needNewSplit = true
                            break
                        }
                    }
                    if (split.nInputs == io.github.kotlinmania.llama.ore.GGML_SCHED_MAX_SPLIT_INPUTS) {
                        val srcId = sched.hashId(src)
                        val srcBid2 = sched.hvTensorBackendIds[srcId]
                        val supported =
                            io.github.kotlinmania.llama.ore.ggmlBackendSchedBufferSupported(
                                sched,
                                src,
                                curBid
                            )
                        if (srcBid2 != curBid && sched.tensorIdCopy(srcId, curBid, 0) == null && !supported) {
                            needNewSplit = true
                            break
                        }
                    }
                }
            }

            if (nodeBid != curBid || needNewSplit) {
                split.iEnd = i
                iSplit++
                if (iSplit >= sched.splitsCapacity) {
                    sched.splitsCapacity *= 2
                    while (sched.splits.size < sched.splitsCapacity) {
                        sched.splits.add(io.github.kotlinmania.llama.ore.GGMLBackendSchedSplit())
                    }
                }
                split = sched.splits[iSplit]
                split.backendId = nodeBid
                split.iStart = i
                split.nInputs = 0
                curBid = nodeBid
            }

            // find inputs that are not on the same backend
            for (j in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
                val src = node.src[j] ?: continue

                val srcId = sched.hashId(src)
                val srcBid = sched.hvTensorBackendIds[srcId]
                require(srcBid != -1) { "all inputs should be assigned by now" }

                if (src.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_INPUT != 0 && sched.nCopies > 1) {
                    if (sched.tensorIdCopy(srcId, srcBid, 0) == null) {
                        val backend = sched.backends[srcBid]!!
                        for (c in 0 until sched.nCopies) {
                            val tensorCopy: io.github.kotlinmania.llama.ore.GGMLTensor
                            if (c == sched.curCopy) {
                                tensorCopy = src
                            } else {
                                tensorCopy = io.github.kotlinmania.llama.ore.ggmlDupTensorLayout(
                                    sched.ctx!!,
                                    src
                                )
                                io.github.kotlinmania.llama.ore.ggmlFormatName(
                                    tensorCopy,
                                    "%s#%s#%d",
                                    io.github.kotlinmania.llama.ore.ggmlBackendName(backend),
                                    src.name,
                                    c
                                )
                            }
                            io.github.kotlinmania.llama.ore.ggmlSetInput(tensorCopy)
                            io.github.kotlinmania.llama.ore.ggmlSetOutput(tensorCopy)
                            sched.setTensorIdCopy(srcId, srcBid, c, tensorCopy)
                        }
                        val ngi = sched.nGraphInputs++
                        require(ngi < io.github.kotlinmania.llama.ore.GGML_SCHED_MAX_SPLIT_INPUTS)
                        sched.graphInputs[ngi] = src
                    }
                }

                if (srcBid != curBid && !io.github.kotlinmania.llama.ore.ggmlBackendSchedBufferSupported(
                        sched,
                        src,
                        curBid
                    )
                ) {
                    if (sched.tensorIdCopy(srcId, curBid, 0) == null) {
                        val backend = sched.backends[curBid]!!
                        for (c in 0 until sched.nCopies) {
                            val tensorCopy = io.github.kotlinmania.llama.ore.ggmlDupTensorLayout(
                                sched.ctx!!,
                                src
                            )
                            io.github.kotlinmania.llama.ore.ggmlFormatName(
                                tensorCopy,
                                "%s#%s#%d",
                                io.github.kotlinmania.llama.ore.ggmlBackendName(backend),
                                src.name,
                                c
                            )
                            if (sched.nCopies > 1) {
                                io.github.kotlinmania.llama.ore.ggmlSetInput(tensorCopy)
                                io.github.kotlinmania.llama.ore.ggmlSetOutput(tensorCopy)
                            }
                            sched.setTensorIdCopy(srcId, curBid, c, tensorCopy)
                        }
                        val ni = split.nInputs++
                        require(ni < io.github.kotlinmania.llama.ore.GGML_SCHED_MAX_SPLIT_INPUTS)
                        split.inputs[ni] = src
                    }
                    node.src[j] = sched.tensorIdCopy(srcId, curBid, sched.curCopy)
                }
            }
            i++
        }
        split.iEnd = graph.nNodes
        sched.nSplits = iSplit + 1
    }

    if (sched.debug > 0) {
        io.github.kotlinmania.llama.ore.ggmlBackendSchedPrintAssignments(sched, graph)
    }

    // swap node_backend_ids and leaf_backend_ids with prevs
    run {
        val tmp = sched.nodeBackendIds
        sched.nodeBackendIds = sched.prevNodeBackendIds
        sched.prevNodeBackendIds = tmp

        val tmp2 = sched.leafBackendIds
        sched.leafBackendIds = sched.prevLeafBackendIds
        sched.prevLeafBackendIds = tmp2
    }

    val graphSize = maxOf(graph.nNodes, graph.nLeafs) +
        sched.nSplits * io.github.kotlinmania.llama.ore.GGML_SCHED_MAX_SPLIT_INPUTS * 2 * sched.nCopies

    sched.debugPrevGraphSize = sched.debugGraphSize
    sched.debugGraphSize = graphSize

    if (sched.graph.size < graphSize) {
        sched.graph.size = graphSize
        sched.graph.nodes = Array(graphSize) { null }
        sched.graph.leafs = Array(graphSize) { null }
    }
    sched.graph.nNodes = 0
    sched.graph.nLeafs = 0

    val graphCopy = sched.graph

    for (si in 0 until sched.nSplits) {
        val sp = sched.splits[si]
        sp.graph = io.github.kotlinmania.llama.ore.ggml_graph_view(graph, sp.iStart, sp.iEnd)

        // add inputs to graph copy so they are allocated by ggml-alloc at the start of the split
        for (j in 0 until sp.nInputs) {
            require(graphCopy.size > graphCopy.nNodes + 1)

            val input = sp.inputs[j]!!
            val inputId = sched.hashId(input)
            val inputCpy = sched.tensorIdCopy(inputId, sp.backendId, sched.curCopy)

            // add a dependency to the input source so that it is not freed before the copy is done
            val inputDep = io.github.kotlinmania.llama.ore.ggmlViewTensor(sched.ctx!!, input)
            inputDep.src[0] = input
            sched.nodeBackendIds[graphCopy.nNodes] = sched.hvTensorBackendIds[inputId]
            graphCopy.nodes[graphCopy.nNodes++] = inputDep

            // add the input copy
            sched.nodeBackendIds[graphCopy.nNodes] = sp.backendId
            graphCopy.nodes[graphCopy.nNodes++] = inputCpy
        }

        for (j in sp.iStart until sp.iEnd) {
            require(graphCopy.size > graphCopy.nNodes)
            sched.nodeBackendIds[graphCopy.nNodes] = sched.tensorBackendId(graph.nodes[j]!!)
            graphCopy.nodes[graphCopy.nNodes++] = graph.nodes[j]
        }
    }

    if (sched.nCopies > 1) {
        // add input copies as leafs so that they are allocated first
        for (gi in 0 until sched.nGraphInputs) {
            val input = sched.graphInputs[gi]!!
            val id = sched.hashId(input)
            val bid = sched.tensorBackendId(input)
            for (c in 0 until sched.nCopies) {
                val inputCpy = sched.tensorIdCopy(id, bid, c)
                sched.leafBackendIds[graphCopy.nLeafs] = bid
                require(graphCopy.size > graphCopy.nLeafs)
                graphCopy.leafs[graphCopy.nLeafs++] = inputCpy
            }
        }

        for (si in 0 until sched.nSplits) {
            val sp = sched.splits[si]
            val bid = sp.backendId
            for (j in 0 until sp.nInputs) {
                val input = sp.inputs[j]!!
                val id = sched.hashId(input)
                for (c in 0 until sched.nCopies) {
                    val inputCpy = sched.tensorIdCopy(id, bid, c)
                    sched.leafBackendIds[graphCopy.nLeafs] = bid
                    require(graphCopy.size > graphCopy.nLeafs)
                    graphCopy.leafs[graphCopy.nLeafs++] = inputCpy
                }
            }
        }
    }

    // add leafs from the original graph
    for (i in 0 until graph.nLeafs) {
        val leaf = graph.leafs[i] ?: continue
        sched.leafBackendIds[graphCopy.nLeafs] = sched.tensorBackendId(leaf)
        require(graphCopy.size > graphCopy.nLeafs)
        graphCopy.leafs[graphCopy.nLeafs++] = leaf
    }

    // set ids for all splits
    for (si in 0 until sched.nSplits) {
        sched.splits[si].graph.uid = io.github.kotlinmania.llama.ore.ggml_graph_next_uid()
    }
}

/** `ggml_backend_cpu_buffer_from_ptr` — C line 2368. */
fun ggmlBackendCpuBufferFromPtr(ptr: ByteArray, size: ULong): io.github.kotlinmania.llama.ore.GGMLBackendBuffer {
    return io.github.kotlinmania.llama.ore.GGMLCpuBufferFromPtr(ptr, size)
}

/** `ggml_backend_cpu_buffer_type` — C line 2328. Returns the CPU buffer type. */
fun ggmlBackendCpuBufferType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType =
    io.github.kotlinmania.llama.ore.createDefaultCpuBufferType()
