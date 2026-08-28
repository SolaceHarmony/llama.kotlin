package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.threadlocal.ThreadLocal

/**
 * Thread-local compute context for GGML operations.
 *
 * Provides thread-local storage for [io.github.kotlinmania.llama.ore.GGMLComputeParams] to eliminate parameter passing
 * through deep call stacks in the compute graph. Each thread maintains its own compute
 * parameters (thread ID, thread count, etc.) without contention.
 *
 * ## Problem Solved
 *
 * GGML computation is multi-threaded with per-thread context:
 * - Each thread has unique parameters (ith = thread ID, nth = total threads)
 * - Parameters must flow through every kernel function
 * - Deep call stacks make parameter passing verbose
 * - Thread-local storage provides cleaner APIs
 *
 * ## Architecture
 *
 * ```
 * Thread 0 ──→ ThreadLocal Params ──→ GGMLComputeParams(ith=0, nth=4)
 * Thread 1 ──→ ThreadLocal Params ──→ GGMLComputeParams(ith=1, nth=4)
 * Thread 2 ──→ ThreadLocal Params ──→ GGMLComputeParams(ith=2, nth=4)
 * Thread 3 ──→ ThreadLocal Params ──→ GGMLComputeParams(ith=3, nth=4)
 * ```
 *
 * Each thread has:
 * - Dedicated compute parameters
 * - Thread ID and count
 * - Work distribution metadata
 * - No contention with other threads
 *
 * ## Usage Example
 *
 * ### Traditional Parameter Passing (verbose)
 * ```kotlin
 * fun ggmlComputeForward(params: GGMLComputeParams, tensor: GGMLTensor) {
 *     // Pass params to every function
 *     val result = computeLayer1(params, tensor)
 *     val final = computeLayer2(params, result)
 * }
 *
 * fun computeLayer1(params: GGMLComputeParams, tensor: GGMLTensor): GGMLTensor {
 *     return kernelOp(params, tensor)
 * }
 *
 * fun kernelOp(params: GGMLComputeParams, tensor: GGMLTensor): GGMLTensor {
 *     val ith = params.ith  // Finally use it here
 *     // ...
 * }
 * ```
 *
 * ### Thread-Local Access (clean)
 * ```kotlin
 * fun ggmlComputeForward(tensor: GGMLTensor) {
 *     val result = computeLayer1(tensor)
 *     val final = computeLayer2(result)
 * }
 *
 * fun computeLayer1(tensor: GGMLTensor): GGMLTensor {
 *     return kernelOp(tensor)
 * }
 *
 * fun kernelOp(tensor: GGMLTensor): GGMLTensor {
 *     val params = GGMLThreadLocalContext.getCurrentParams()
 *     val ith = params.ith  // Access thread-local params directly
 *     // ...
 * }
 * ```
 *
 * ### Multi-threaded Computation
 * ```kotlin
 * fun parallelCompute(tensor: GGMLTensor, nThreads: Int) {
 *     List(nThreads) { threadId ->
 *         thread {
 *             // Set params for this thread
 *             GGMLThreadLocalContext.setCurrentParams(
 *                 GGMLComputeParams(ith = threadId, nth = nThreads)
 *             )
 *
 *             // Compute without passing params
 *             val result = ggmlComputeForward(tensor)
 *
 *             // Clean up
 *             GGMLThreadLocalContext.clearCurrentParams()
 *         }
 *     }
 * }
 * ```
 *
 * ### Scoped Computation
 * ```kotlin
 * fun computeWithContext(tensor: GGMLTensor): GGMLTensor {
 *     return GGMLThreadLocalContext.withParams(
 *         GGMLComputeParams(ith = 0, nth = 1)
 *     ) {
 *         // Params available in this scope
 *         ggmlComputeForward(tensor)
 *     }
 *     // Params automatically cleared
 * }
 * ```
 *
 * ## Benefits
 *
 * **Cleaner APIs**:
 * - Eliminate parameter drilling through call stacks
 * - Reduce function signatures
 * - More readable code
 *
 * **Thread Safety**:
 * - Each thread has isolated parameters
 * - No contention or data races
 * - Inherently thread-safe access
 *
 * **Performance**:
 * - No parameter passing overhead
 * - Better inlining opportunities
 * - Reduced stack usage
 *
 * **Correctness**:
 * - Thread ID always matches actual thread
 * - No accidental parameter reuse across threads
 * - Clear scoping with withParams
 *
 * ## Trade-offs
 *
 * **Implicit State**:
 * - Parameters not visible in function signatures
 * - Must remember to set params before use
 * - Error if accessing before set
 *
 * **Memory**:
 * - Small overhead per thread (few bytes for params)
 * - Negligible compared to compute buffers
 *
 * **Compatibility**:
 * - Existing code using explicit params still works
 * - Can migrate incrementally
 * - Both styles can coexist
 *
 * ## API
 *
 * ### Setting Parameters
 * ```kotlin
 * GGMLThreadLocalContext.setCurrentParams(params)  // Set for current thread
 * GGMLThreadLocalContext.clearCurrentParams()      // Remove from current thread
 * ```
 *
 * ### Accessing Parameters
 * ```kotlin
 * val params = GGMLThreadLocalContext.getCurrentParams()  // Get or null
 * val params = GGMLThreadLocalContext.getCurrentParamsOr { default }  // Get or default
 * val params = GGMLThreadLocalContext.requireCurrentParams()  // Get or error
 * ```
 *
 * ### Scoped Usage
 * ```kotlin
 * GGMLThreadLocalContext.withParams(params) {
 *     // Code that uses params
 * }
 * // Params automatically cleared
 * ```
 *
 * ## Integration with GGML
 *
 * ### Option 1: Hybrid Approach (recommended)
 * Keep explicit parameter passing but also set thread-local for nested functions:
 * ```kotlin
 * fun ggmlComputeForward(params: GGMLComputeParams, tensor: GGMLTensor) {
 *     GGMLThreadLocalContext.setCurrentParams(params)
 *     try {
 *         // Nested functions can use thread-local access
 *         computeKernels(tensor)
 *     } finally {
 *         GGMLThreadLocalContext.clearCurrentParams()
 *     }
 * }
 * ```
 *
 * ### Option 2: Full Migration
 * Remove explicit params from all functions and use thread-local only.
 *
 * ### Option 3: New API Variants
 * Create `*ThreadLocal` variants of functions:
 * ```kotlin
 * fun ggmlComputeForward(params: GGMLComputeParams, tensor: GGMLTensor)  // Old
 * fun ggmlComputeForwardThreadLocal(tensor: GGMLTensor)  // New
 * ```
 *
 * ## Comparison with Explicit Parameters
 *
 * | Aspect | Explicit Params | Thread-Local |
 * |--------|----------------|--------------|
 * | Clarity | ✅ Visible in signatures | ⚠️ Implicit state |
 * | Verbosity | ⚠️ Parameter drilling | ✅ Clean APIs |
 * | Thread safety | ✅ Explicit threading | ✅ Automatic isolation |
 * | Performance | ⚠️ Passing overhead | ✅ No passing |
 * | Debuggability | ✅ Clear data flow | ⚠️ Hidden state |
 * | Migration | ❌ Requires rewrite | ✅ Incremental |
 *
 * ## Implementation Notes
 *
 * Uses threadlocal-kotlin (io.github.kotlinmania.llama.hreadlocal-kotlin:0.3.1) which provides
 * true thread-local storage across all Kotlin Multiplatform targets:
 * - JVM: java.lang.ThreadLocal
 * - Native: pthread_key_t / Windows TLS
 * - JS: Map keyed by coroutine context
 * - Wasm: Linear memory slots
 *
 * ## Future Enhancements
 *
 * - Stack of contexts for nested computations
 * - Automatic context propagation in coroutines
 * - Context inheritance for child threads
 * - Debug mode to track context lifetime
 * - Statistics on context usage
 *
 * @see io.github.kotlinmania.llama.ore.GGMLComputeParams For the compute parameters stored
 * @since 0.3.1 (requires threadlocal-kotlin:0.3.1)
 */
object GGMLThreadLocalContext {
    /**
     * Thread-local mutable cell holding compute parameters.
     */
    private data class ParamsCell(var value: GGMLComputeParams? = null)

    /**
     * Thread-local compute parameters holder.
     * Each thread gets its own ParamsCell instance.
     */
    private val computeParams = ThreadLocal<ParamsCell>()

    /**
     * Get or create the params cell for the current thread.
     */
    private fun getCell(): ParamsCell {
        return computeParams.getOr { ParamsCell() }
    }

    /**
     * Set the compute parameters for the current thread.
     *
     * @param params The compute parameters to associate with this thread
     */
    fun setCurrentParams(params: GGMLComputeParams) {
        getCell().value = params
    }

    /**
     * Get the compute parameters for the current thread.
     *
     * @return The compute parameters, or null if not set for this thread
     */
    fun getCurrentParams(): GGMLComputeParams? {
        return getCell().value
    }

    /**
     * Get the compute parameters for the current thread, or use the provided default.
     *
     * @param default The default parameters to return if none are set
     * @return The compute parameters for this thread, or the default
     */
    fun getCurrentParamsOr(default: () -> GGMLComputeParams): GGMLComputeParams {
        return getCell().value ?: default()
    }

    /**
     * Get the compute parameters for the current thread, or throw an error.
     *
     * @return The compute parameters for this thread
     * @throws IllegalStateException if no parameters are set
     */
    fun requireCurrentParams(): GGMLComputeParams {
        return getCell().value
            ?: error("No compute params set for current thread. Call setCurrentParams() first.")
    }

    /**
     * Remove the compute parameters for the current thread.
     *
     * Should be called when thread computation is complete to avoid memory leaks.
     */
    fun clearCurrentParams() {
        getCell().value = null
    }

    /**
     * Execute a block with the given compute parameters set for the current thread.
     * Automatically clears the parameters when the block completes.
     *
     * @param params The compute parameters to use
     * @param block The code to execute with these parameters
     * @return The result of the block
     */
    inline fun <T> withParams(params: GGMLComputeParams, block: () -> T): T {
        setCurrentParams(params)
        try {
            return block()
        } finally {
            clearCurrentParams()
        }
    }

    /**
     * Get the current thread ID from the thread-local params.
     *
     * @return Thread ID (ith), or null if no params set
     */
    fun getCurrentThreadId(): Int? {
        return getCurrentParams()?.ith
    }

    /**
     * Get the total thread count from the thread-local params.
     *
     * @return Thread count (nth), or null if no params set
     */
    fun getTotalThreads(): Int? {
        return getCurrentParams()?.nth
    }

    /**
     * Check if compute parameters are set for the current thread.
     *
     * @return True if params are set, false otherwise
     */
    fun hasCurrentParams(): Boolean {
        return computeParams.get() != null
    }
}
