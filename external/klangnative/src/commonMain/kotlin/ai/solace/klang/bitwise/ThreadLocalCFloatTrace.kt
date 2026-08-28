package io.github.kotlinmania.llama.lang.bitwise

import io.github.kotlinmania.llama.hreadlocal.ThreadLocal

/**
 * Thread-local CFloat32 operation tracer.
 *
 * This wrapper provides thread-safe tracing by maintaining separate CFloatTrace instances
 * per thread, enabling parallel numerical analysis without contention.
 *
 * ## Problem Solved
 *
 * CFloatTrace is not thread-safe:
 * - Shared mutable list of entries
 * - No synchronization on enable/disable
 * - Concurrent access causes corruption and lost traces
 *
 * Instead of using a global lock (which would serialize all tracing), ThreadLocalCFloatTrace
 * gives each thread its own isolated tracer.
 *
 * ## Architecture
 *
 * ```
 * Thread 1 ──→ ThreadLocal Tracer 1 ──→ List<Entry>
 * Thread 2 ──→ ThreadLocal Tracer 2 ──→ List<Entry>
 * Thread 3 ──→ ThreadLocal Tracer 3 ──→ List<Entry>
 * ```
 *
 * Each thread has:
 * - Dedicated trace buffer
 * - Independent enable/disable state
 * - No contention with other threads
 *
 * ## Usage Example
 *
 * ### Basic Thread-Local Tracing
 * ```kotlin
 * fun analyzeAlgorithm() {
 *     ThreadLocalCFloatTrace.enable()
 *
 *     val a = CFloat32(1.0f)
 *     val b = CFloat32(3.0f)
 *     val result = a / b  // Logs to this thread's buffer
 *
 *     val entries = ThreadLocalCFloatTrace.getEntries()
 *     println("Traced ${entries.size} operations")
 *
 *     ThreadLocalCFloatTrace.disable()
 * }
 * ```
 *
 * ### Parallel Numerical Analysis
 * ```kotlin
 * fun parallelAnalysis() = runBlocking {
 *     List(10) {
 *         async(Dispatchers.Default) {
 *             ThreadLocalCFloatTrace.enable()
 *
 *             // Each coroutine traces independently
 *             val result = complexCalculation()
 *
 *             val threadTraces = ThreadLocalCFloatTrace.getEntries()
 *             println("Thread ${Thread.currentThread().name}: ${threadTraces.size} ops")
 *
 *             ThreadLocalCFloatTrace.reset()
 *         }
 *     }.awaitAll()
 * }
 * ```
 *
 * ### Scoped Tracing with Automatic Cleanup
 * ```kotlin
 * fun scopedTrace() {
 *     val (result, traces) = ThreadLocalCFloatTrace.withTracing {
 *         // Code to trace
 *         val x = CFloat32(2.0f)
 *         val y = CFloat32(0.5f)
 *         x * y
 *     }
 *
 *     println("Result: $result")
 *     println("Operations: ${traces.size}")
 *     // Tracing automatically disabled and reset
 * }
 * ```
 *
 * ## Benefits Over Global Tracing
 *
 * **Thread Safety**:
 * - No data races or corruption
 * - Each thread sees only its own traces
 * - No locking overhead
 *
 * **Correctness**:
 * - Operations traced only in the thread that executed them
 * - No mixing of traces from different computations
 * - Clear attribution of operations to threads
 *
 * **Performance**:
 * - Lock-free tracing
 * - Better cache locality (thread-local buffers)
 * - No contention between threads
 *
 * **Usability**:
 * - Enable tracing in one thread without affecting others
 * - Independent trace buffers per thread
 * - Easy to correlate traces with specific work
 *
 * ## API
 *
 * ### Enabling/Disabling
 * ```kotlin
 * ThreadLocalCFloatTrace.enable()   // Enable for current thread
 * ThreadLocalCFloatTrace.disable()  // Disable for current thread
 * ThreadLocalCFloatTrace.isEnabled()  // Check if enabled
 * ```
 *
 * ### Retrieving Traces
 * ```kotlin
 * val entries = ThreadLocalCFloatTrace.getEntries()  // Get current thread's traces
 * ThreadLocalCFloatTrace.reset()  // Clear current thread's buffer
 * ```
 *
 * ### Scoped Tracing
 * ```kotlin
 * val (result, traces) = ThreadLocalCFloatTrace.withTracing {
 *     // Code to trace
 * }
 * ```
 *
 * ## Comparison with Global CFloatTrace
 *
 * | Feature | CFloatTrace (global) | ThreadLocalCFloatTrace |
 * |---------|----------------------|------------------------|
 * | Thread safety | ❌ Not safe | ✅ Thread-safe |
 * | Multi-threaded | ❌ Requires lock | ✅ Lock-free |
 * | Trace isolation | ❌ Mixed traces | ✅ Per-thread traces |
 * | Performance (MT) | ⚠️ Serialized | ✅ Parallel |
 * | Memory usage | ✅ Single buffer | ⚠️ Buffer per thread |
 *
 * ## Integration with CFloat32
 *
 * If CFloat32 operations call `CFloatTrace.log()` directly, they will only trace to the
 * global instance. To use thread-local tracing, either:
 *
 * 1. **Modify CFloat32** to call `ThreadLocalCFloatTrace.log()` instead
 * 2. **Use wrapper functions** that manually log operations
 * 3. **Post-process** the global CFloatTrace buffer into thread-local buffers
 *
 * Example wrapper approach:
 * ```kotlin
 * fun CFloat32.divTracked(other: CFloat32): CFloat32 {
 *     val result = this / other
 *     ThreadLocalCFloatTrace.log("/", this.toFloat(), other.toFloat(), result.toFloat())
 *     return result
 * }
 * ```
 *
 * ## Memory Considerations
 *
 * Each thread maintains its own trace buffer:
 * - Buffer grows unbounded until reset
 * - 10 threads × 1000 entries × ~64 bytes = ~640 KB
 * - Call `reset()` periodically to free memory
 *
 * ## Limitations
 *
 * - **Requires threadlocal-kotlin 0.3.1**: Must be in dependencies
 * - **Not integrated with CFloat32**: Needs manual logging or CFloat32 changes
 * - **Memory overhead**: One buffer per thread that enables tracing
 * - **No cross-thread aggregation**: Each thread sees only its traces
 *
 * ## Future Enhancements
 *
 * Potential improvements:
 * - Automatic integration with CFloat32 operations
 * - Cross-thread trace aggregation and analysis
 * - Filtering by thread ID or computation phase
 * - Export thread-specific traces to files
 * - Statistics per thread (precision loss, operation counts)
 *
 * @see CFloatTrace For the underlying global tracer
 * @see CFloat32 For the floating-point type being traced
 * @since 0.3.1 (requires threadlocal-kotlin:0.3.1)
 */
object ThreadLocalCFloatTrace {
    /**
     * Per-thread tracer state.
     */
    private data class TracerState(
        var enabled: Boolean = false,
        val entries: MutableList<Entry> = mutableListOf()
    )

    /**
     * Trace entry representing one operation.
     */
    data class Entry(
        val operation: String,
        val lhs: Float,
        val rhs: Float,
        val result: Float
    )

    /**
     * Thread-local tracer state.
     * Each thread gets its own TracerState instance via lazy initialization.
     */
    private val tracer = ThreadLocal<TracerState>()

    /**
     * Get or create the current thread's tracer state.
     */
    private fun getState(): TracerState {
        return tracer.getOr { TracerState() }
    }

    /**
     * Enable tracing for the current thread.
     */
    fun enable() {
        getState().enabled = true
    }

    /**
     * Disable tracing for the current thread.
     */
    fun disable() {
        getState().enabled = false
    }

    /**
     * Check if tracing is enabled for the current thread.
     */
    fun isEnabled(): Boolean {
        return getState().enabled
    }

    /**
     * Log a floating-point operation to the current thread's trace buffer.
     *
     * @param operation Operation name ("+", "-", "*", "/", etc.)
     * @param lhs Left-hand side value
     * @param rhs Right-hand side value
     * @param result Result value
     */
    fun log(operation: String, lhs: Float, rhs: Float, result: Float) {
        val state = getState()
        if (state.enabled) {
            state.entries.add(Entry(operation, lhs, rhs, result))
        }
    }

    /**
     * Get all trace entries for the current thread.
     *
     * @return List of trace entries (may be empty)
     */
    fun getEntries(): List<Entry> {
        return getState().entries.toList()
    }

    /**
     * Clear the current thread's trace buffer.
     */
    fun reset() {
        val state = getState()
        state.entries.clear()
    }

    /**
     * Execute a block with tracing enabled, then return both the result and traces.
     * Automatically disables tracing and resets buffer after the block completes.
     *
     * @param block Code to trace
     * @return Pair of (result, trace entries)
     */
    inline fun <T> withTracing(block: () -> T): Pair<T, List<Entry>> {
        enable()
        reset()
        try {
            val result = block()
            val traces = getEntries()
            return result to traces
        } finally {
            disable()
            reset()
        }
    }

    /**
     * Get statistics about the current thread's traced operations.
     *
     * @return Statistics summary
     */
    fun getStats(): TraceStats {
        val entries = getEntries()
        return TraceStats(
            totalOperations = entries.size,
            operationCounts = entries.groupingBy { it.operation }.eachCount(),
            enabled = isEnabled()
        )
    }

    /**
     * Statistics about traced operations.
     */
    data class TraceStats(
        /** Total number of operations traced */
        val totalOperations: Int,
        /** Count of each operation type */
        val operationCounts: Map<String, Int>,
        /** Whether tracing is currently enabled */
        val enabled: Boolean
    )
}
