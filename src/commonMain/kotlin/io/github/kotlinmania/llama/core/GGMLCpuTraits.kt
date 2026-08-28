package io.github.kotlinmania.llama.ore

// port-lint: source ggml/src/ggml-cpu/traits.h + traits.cpp

/**
 * Port of `ggml::cpu::tensor_traits` — interface registered in tensor->extra
 * to provide custom compute_forward / work_size for accelerated ops.
 */
interface GGMLCpuTensorTraits {
    fun workSize(nThreads: Int, op: io.github.kotlinmania.llama.ore.GGMLTensor, size: LongArray): Boolean
    fun computeForward(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean
}

/**
 * Port of `ggml::cpu::extra_buffer_type` — interface for extra accelerator buffer types
 * that can intercept tensor ops on the CPU backend.
 */
interface GGMLCpuExtraBufferType {
    fun supportsOp(dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean
    fun getTensorTraits(op: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLCpuTensorTraits?
}

/**
 * Port of `ggml_cpu_extra_compute_forward` from traits.cpp.
 * Iterates over all registered extra buffer types and delegates to the
 * first one whose tensor_traits can handle the op.
 */
fun ggmlCpuExtraComputeForward(params: io.github.kotlinmania.llama.ore.GGMLComputeParams, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    for (extra in io.github.kotlinmania.llama.ore.ggmlBackendCpuGetExtraBufferTypes()) {
        val context = extra.getContext()
        if (context is io.github.kotlinmania.llama.ore.GGMLCpuExtraBufferType) {
            val tensorTraits = context.getTensorTraits(op)
            if (tensorTraits != null && tensorTraits.computeForward(params, op)) {
                return true
            }
        }
    }
    return false
}

/**
 * Port of `ggml_cpu_extra_work_size` from traits.cpp.
 * Iterates over all registered extra buffer types and queries each for
 * the required work buffer size.
 */
fun ggmlCpuExtraWorkSize(nThreads: Int, op: io.github.kotlinmania.llama.ore.GGMLTensor, size: LongArray): Boolean {
    for (extra in io.github.kotlinmania.llama.ore.ggmlBackendCpuGetExtraBufferTypes()) {
        val context = extra.getContext()
        if (context is io.github.kotlinmania.llama.ore.GGMLCpuExtraBufferType) {
            val tensorTraits = context.getTensorTraits(op)
            if (tensorTraits != null && tensorTraits.workSize(nThreads, op, size)) {
                return true
            }
        }
    }
    return false
}

// ggmlBackendCpuGetExtraBufferTypes is defined in GGMLCpuBackend.kt.

// --- Virtual method wrappers for ast_distance parity with traits.h ---

/** Wrapper for tensor_traits::work_size (traits.h line 23). */
fun workSize(traits: io.github.kotlinmania.llama.ore.GGMLCpuTensorTraits, nThreads: Int, op: io.github.kotlinmania.llama.ore.GGMLTensor, size: LongArray): Boolean {
    return traits.workSize(nThreads, op, size)
}

/** Wrapper for tensor_traits::compute_forward (traits.h line 24). */
fun computeForward(traits: io.github.kotlinmania.llama.ore.GGMLCpuTensorTraits, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    return traits.computeForward(params, op)
}

/** Wrapper for extra_buffer_type::supports_op (traits.h line 30). */
fun supportsOp(extra: io.github.kotlinmania.llama.ore.GGMLCpuExtraBufferType, dev: io.github.kotlinmania.llama.ore.GGMLBackendDevice, op: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    return extra.supportsOp(dev, op)
}

/** Wrapper for extra_buffer_type::get_tensor_traits (traits.h line 31). */
fun getTensorTraits(extra: io.github.kotlinmania.llama.ore.GGMLCpuExtraBufferType, op: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLCpuTensorTraits? {
    return extra.getTensorTraits(op)
}

// ggml_backend_cpu_get_extra_buffer_types (traits.h line 36) lives in GGMLCpuBackend.kt.
