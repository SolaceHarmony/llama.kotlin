// port-lint: source ggml/src/ggml-threading.cpp
package io.github.kotlinmania.llama.ore

import kotlin.concurrent.atomics.AtomicInt
import kotlin.concurrent.atomics.ExperimentalAtomicApi

/**
 * Kotlin Native port of GGML scheduler functionality.
 * This file contains the multi-threaded and multi-backend scheduler implementation.
 */

// ---------------------------------------------------------------------------
// ggml-threading.h / ggml-threading.cpp — critical section (global mutex)
// ---------------------------------------------------------------------------

/**
 * Simple spin-lock based critical section matching the C++ std::mutex usage.
 * In Kotlin/Native single-threaded mode this is effectively a identity guard;
 * with the new memory model it provides basic mutual exclusion.
 */
@OptIn(ExperimentalAtomicApi::class)
private val criticalSectionLock = AtomicInt(0)

@OptIn(ExperimentalAtomicApi::class)
fun ggml_critical_section_start() {
    while (!io.github.kotlinmania.llama.ore.criticalSectionLock.compareAndSet(0, 1)) {
        // spin
    }
}

@OptIn(ExperimentalAtomicApi::class)
fun ggml_critical_section_end() {
    io.github.kotlinmania.llama.ore.criticalSectionLock.store(0)
}

/**
 * High-level backend status used by scheduler/examples
 */
enum class GGMLBackendStatus {
    SUCCESS,
    FAILED,
    ALLOC_FAILED,
    NOT_SUPPORTED
}

/**
 * Scheduling strategy enumeration
 */
enum class GGMLSchedulingStrategy {
    /** Execute operations sequentially */
    SEQUENTIAL,
    /** Execute independent operations in parallel */
    PARALLEL,
    /** Use multiple backends when available */
    MULTI_BACKEND
}

/**
 * Graph split for backend execution
 */
data class GGMLGraphSplit(
    val backendName: String,
    val startNode: Int,
    val endNode: Int,
    val nodes: List<io.github.kotlinmania.llama.ore.GGMLTensor>,
    val inputs: List<io.github.kotlinmania.llama.ore.GGMLTensor> = emptyList()
)

/**
 * Execution context for a scheduled operation
 */
data class GGMLExecutionContext(
    val backend: io.github.kotlinmania.llama.ore.GGMLBackend,
    val split: io.github.kotlinmania.llama.ore.GGMLGraphSplit,
    val context: io.github.kotlinmania.llama.ore.GGMLContext
)

/**
 * Dependency tracker for graph nodes
 */
class GGMLDependencyTracker {
    private val dependencies = mutableMapOf<io.github.kotlinmania.llama.ore.GGMLTensor, MutableSet<io.github.kotlinmania.llama.ore.GGMLTensor>>()
    private val dependents = mutableMapOf<io.github.kotlinmania.llama.ore.GGMLTensor, MutableSet<io.github.kotlinmania.llama.ore.GGMLTensor>>()
    private val completed = mutableSetOf<io.github.kotlinmania.llama.ore.GGMLTensor>()
    
    /**
     * Build dependency graph for a computation graph
     */
    fun buildDependencies(graph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
        dependencies.clear()
        dependents.clear()
        completed.clear()
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            dependencies[node] = mutableSetOf()
            dependents[node] = mutableSetOf()
            
            // Add source dependencies
            for (src in node.src) {
                if (src != null) {
                    dependencies[node]!!.add(src)
                    dependents.getOrPut(src) { mutableSetOf() }.add(node)
                }
            }
            
            // Add view source dependency
            node.viewSrc?.let { viewSrc ->
                dependencies[node]!!.add(viewSrc)
                dependents.getOrPut(viewSrc) { mutableSetOf() }.add(node)
            }
        }
    }
    
    /**
     * Get nodes that are ready to execute (all dependencies completed)
     */
    fun getReadyNodes(): List<io.github.kotlinmania.llama.ore.GGMLTensor> {
        return dependencies.keys.filter { node ->
            node !in completed && dependencies[node]!!.all { it in completed }
        }
    }
    
    /**
     * Mark a node as completed
     */
    fun markCompleted(node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        completed.add(node)
    }
    
    /**
     * Check if all nodes are completed
     */
    fun isComplete(): Boolean {
        return completed.size == dependencies.size
    }
    
    /**
     * Reset the tracker
     */
    fun reset() {
        completed.clear()
    }
}

/**
 * Multi-threaded scheduler for computation graphs
 */
class GGMLScheduler(
    private val backends: List<io.github.kotlinmania.llama.ore.GGMLBackend>,
    private val strategy: io.github.kotlinmania.llama.ore.GGMLSchedulingStrategy = io.github.kotlinmania.llama.ore.GGMLSchedulingStrategy.PARALLEL
) {
    private val dependencyTracker = io.github.kotlinmania.llama.ore.GGMLDependencyTracker()
    private var maxWorkers = 4
    
    private fun primaryBackend(): io.github.kotlinmania.llama.ore.GGMLBackend? = backends.firstOrNull()
    
    fun setMaxWorkers(workers: Int) {
        maxWorkers = workers.coerceAtLeast(1)
    }
    
    fun execute(graph: io.github.kotlinmania.llama.ore.GGMLCGraph, context: io.github.kotlinmania.llama.ore.GGMLContext): io.github.kotlinmania.llama.ore.GGMLBackendStatus {
        return when (strategy) {
            io.github.kotlinmania.llama.ore.GGMLSchedulingStrategy.SEQUENTIAL -> executeSequential(graph, context)
            io.github.kotlinmania.llama.ore.GGMLSchedulingStrategy.PARALLEL -> executeParallel(graph, context)
            io.github.kotlinmania.llama.ore.GGMLSchedulingStrategy.MULTI_BACKEND -> executeMultiBackend(graph, context)
        }
    }
    
    private fun executeSequential(graph: io.github.kotlinmania.llama.ore.GGMLCGraph, context: io.github.kotlinmania.llama.ore.GGMLContext): io.github.kotlinmania.llama.ore.GGMLBackendStatus {
        val backend = primaryBackend() ?: return io.github.kotlinmania.llama.ore.GGMLBackendStatus.FAILED
        return try {
            val allocator = graph.allocator ?: io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
                .also { graph.allocator = it }
            if (!allocator.allocateGraph(graph)) return io.github.kotlinmania.llama.ore.GGMLBackendStatus.ALLOC_FAILED
            val status = backend.graphCompute(graph)
            if (status == io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS) io.github.kotlinmania.llama.ore.GGMLBackendStatus.SUCCESS else io.github.kotlinmania.llama.ore.GGMLBackendStatus.FAILED
        } catch (e: Exception) {
            io.github.kotlinmania.llama.ore.GGMLBackendStatus.FAILED
        }
    }
    
    private fun executeParallel(graph: io.github.kotlinmania.llama.ore.GGMLCGraph, context: io.github.kotlinmania.llama.ore.GGMLContext): io.github.kotlinmania.llama.ore.GGMLBackendStatus {
        dependencyTracker.buildDependencies(graph)
        return executeSequentialWithDependencies(graph, context)
    }
    
    private fun executeSequentialWithDependencies(graph: io.github.kotlinmania.llama.ore.GGMLCGraph, context: io.github.kotlinmania.llama.ore.GGMLContext): io.github.kotlinmania.llama.ore.GGMLBackendStatus {
        dependencyTracker.buildDependencies(graph)
        
        while (!dependencyTracker.isComplete()) {
            val readyNodes = dependencyTracker.getReadyNodes()
            if (readyNodes.isEmpty()) return io.github.kotlinmania.llama.ore.GGMLBackendStatus.FAILED
            
            for (node in readyNodes) {
                try {
                    val backend = primaryBackend() ?: return io.github.kotlinmania.llama.ore.GGMLBackendStatus.FAILED
                    executeNode(node, context, backend)
                    dependencyTracker.markCompleted(node)
                } catch (e: Exception) {
                    return io.github.kotlinmania.llama.ore.GGMLBackendStatus.FAILED
                }
            }
        }
        
        return io.github.kotlinmania.llama.ore.GGMLBackendStatus.SUCCESS
    }
    
    private fun executeMultiBackend(graph: io.github.kotlinmania.llama.ore.GGMLCGraph, context: io.github.kotlinmania.llama.ore.GGMLContext): io.github.kotlinmania.llama.ore.GGMLBackendStatus {
        val splits = createGraphSplits(graph)
        
        if (splits.size <= 1) {
            return executeSequential(graph, context)
        }
        
        for (split in splits) {
            val backend = backends.firstOrNull { it.getName() == split.backendName } ?: continue
            val status = executeGraphSplit(split, context, backend)
            if (status != io.github.kotlinmania.llama.ore.GGMLBackendStatus.SUCCESS) return status
        }
        
        backends.forEach { it.synchronize() }
        return io.github.kotlinmania.llama.ore.GGMLBackendStatus.SUCCESS
    }
    
    private fun executeNode(node: io.github.kotlinmania.llama.ore.GGMLTensor, context: io.github.kotlinmania.llama.ore.GGMLContext, backend: io.github.kotlinmania.llama.ore.GGMLBackend) {
        val tempGraph = io.github.kotlinmania.llama.ore.createGraph(1)
        tempGraph.nodes[0] = node
        tempGraph.nNodes = 1
        val allocator = tempGraph.allocator ?: io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
            .also { tempGraph.allocator = it }
        allocator.allocateGraph(tempGraph)
        backend.graphCompute(tempGraph)
    }
    
    private fun executeGraphSplit(split: io.github.kotlinmania.llama.ore.GGMLGraphSplit, context: io.github.kotlinmania.llama.ore.GGMLContext, backend: io.github.kotlinmania.llama.ore.GGMLBackend): io.github.kotlinmania.llama.ore.GGMLBackendStatus {
        val subGraph = io.github.kotlinmania.llama.ore.createGraph(split.nodes.size)
        for (i in split.nodes.indices) {
            subGraph.nodes[i] = split.nodes[i]
        }
        subGraph.nNodes = split.nodes.size
        val allocator = subGraph.allocator ?: io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
            .also { subGraph.allocator = it }
        if (!allocator.allocateGraph(subGraph)) return io.github.kotlinmania.llama.ore.GGMLBackendStatus.ALLOC_FAILED
        val status = backend.graphCompute(subGraph)
        return if (status == io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS) io.github.kotlinmania.llama.ore.GGMLBackendStatus.SUCCESS else io.github.kotlinmania.llama.ore.GGMLBackendStatus.FAILED
    }
    
    private fun createGraphSplits(graph: io.github.kotlinmania.llama.ore.GGMLCGraph): List<io.github.kotlinmania.llama.ore.GGMLGraphSplit> {
        val splits = mutableListOf<io.github.kotlinmania.llama.ore.GGMLGraphSplit>()
        if (backends.isEmpty()) return splits
        
        var currentBackend = primaryBackend() ?: return splits
        var splitStart = 0
        val currentSplitNodes = mutableListOf<io.github.kotlinmania.llama.ore.GGMLTensor>()
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            val bestBackend = primaryBackend() ?: currentBackend
            if (bestBackend.getName() != currentBackend.getName() && currentSplitNodes.isNotEmpty()) {
                splits.add(
                    io.github.kotlinmania.llama.ore.GGMLGraphSplit(
                        backendName = currentBackend.getName(),
                        startNode = splitStart,
                        endNode = i - 1,
                        nodes = currentSplitNodes.toList()
                    )
                )
                currentSplitNodes.clear()
                splitStart = i
                currentBackend = bestBackend
            }
            currentSplitNodes.add(node)
        }
        
        if (currentSplitNodes.isNotEmpty()) {
            splits.add(
                io.github.kotlinmania.llama.ore.GGMLGraphSplit(
                    backendName = currentBackend.getName(),
                    startNode = splitStart,
                    endNode = graph.nNodes - 1,
                    nodes = currentSplitNodes.toList()
                )
            )
        }
        
        return splits
    }
    
    fun getStats(): io.github.kotlinmania.llama.ore.SchedulerStats {
        return io.github.kotlinmania.llama.ore.SchedulerStats(
            maxWorkers = maxWorkers,
            strategy = strategy,
            availableBackends = backends.map { it.getName() }
        )
    }
}

/**
 * Scheduler statistics
 */
data class SchedulerStats(
    val maxWorkers: Int,
    val strategy: io.github.kotlinmania.llama.ore.GGMLSchedulingStrategy,
    val availableBackends: List<String>
)
