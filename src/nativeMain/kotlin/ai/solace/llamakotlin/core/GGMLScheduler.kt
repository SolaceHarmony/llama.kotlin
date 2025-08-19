package ai.solace.llamakotlin.core

/**
 * Kotlin Native port of GGML scheduler functionality.
 * This file contains the multi-threaded and multi-backend scheduler implementation.
 */

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
    val backendType: GGMLBackendType,
    val startNode: Int,
    val endNode: Int,
    val nodes: List<GGMLTensor>,
    val inputs: List<GGMLTensor> = emptyList()
)

/**
 * Execution context for a scheduled operation
 */
data class GGMLExecutionContext(
    val backend: GGMLBackend,
    val split: GGMLGraphSplit,
    val context: GGMLContext
)

/**
 * Dependency tracker for graph nodes
 */
class GGMLDependencyTracker {
    private val dependencies = mutableMapOf<GGMLTensor, MutableSet<GGMLTensor>>()
    private val dependents = mutableMapOf<GGMLTensor, MutableSet<GGMLTensor>>()
    private val completed = mutableSetOf<GGMLTensor>()
    
    /**
     * Build dependency graph for a computation graph
     */
    fun buildDependencies(graph: GGMLCGraph) {
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
    fun getReadyNodes(): List<GGMLTensor> {
        return dependencies.keys.filter { node ->
            node !in completed && dependencies[node]!!.all { it in completed }
        }
    }
    
    /**
     * Mark a node as completed
     */
    fun markCompleted(node: GGMLTensor) {
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
    private val backendManager: GGMLBackendManager,
    private val strategy: GGMLSchedulingStrategy = GGMLSchedulingStrategy.PARALLEL
) {
    private val dependencyTracker = GGMLDependencyTracker()
    private var maxWorkers = 4
    
    /**
     * Set maximum number of worker threads
     */
    fun setMaxWorkers(workers: Int) {
        maxWorkers = workers.coerceAtLeast(1)
    }
    
    /**
     * Execute a computation graph using the scheduler
     */
    fun execute(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        return when (strategy) {
            GGMLSchedulingStrategy.SEQUENTIAL -> executeSequential(graph, context)
            GGMLSchedulingStrategy.PARALLEL -> executeParallel(graph, context)
            GGMLSchedulingStrategy.MULTI_BACKEND -> executeMultiBackend(graph, context)
        }
    }
    
    /**
     * Execute graph sequentially
     */
    private fun executeSequential(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        val backend = backendManager.getPrimaryBackend() ?: return GGMLBackendStatus.FAILED
        
        try {
            return backend.compute(graph, context)
        } catch (e: Exception) {
            return GGMLBackendStatus.FAILED
        }
    }
    
    /**
     * Execute graph with parallel operations
     */
    private fun executeParallel(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        dependencyTracker.buildDependencies(graph)
        
        // For now, fall back to sequential execution with dependency tracking
        // Workers in Kotlin Native can be complex in some environments
        return executeSequentialWithDependencies(graph, context)
    }
    
    /**
     * Execute graph sequentially but with proper dependency tracking
     */
    private fun executeSequentialWithDependencies(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        dependencyTracker.buildDependencies(graph)
        
        while (!dependencyTracker.isComplete()) {
            val readyNodes = dependencyTracker.getReadyNodes()
            
            if (readyNodes.isEmpty()) {
                // This shouldn't happen if dependencies are correctly tracked
                return GGMLBackendStatus.FAILED
            }
            
            for (node in readyNodes) {
                try {
                    val backend = backendManager.getBestBackend(node) ?: return GGMLBackendStatus.FAILED
                    executeNode(node, context, backend)
                    dependencyTracker.markCompleted(node)
                } catch (e: Exception) {
                    return GGMLBackendStatus.FAILED
                }
            }
        }
        
        return GGMLBackendStatus.SUCCESS
    }
    
    /**
     * Execute graph with multiple backends
     */
    private fun executeMultiBackend(graph: GGMLCGraph, context: GGMLContext): GGMLBackendStatus {
        val splits = createGraphSplits(graph)
        
        if (splits.size == 1) {
            // Only one backend available, use sequential execution
            return executeSequential(graph, context)
        }
        
        // Execute splits sequentially for now (can be enhanced later with true parallelism)
        for (split in splits) {
            val backend = backendManager.getBackends().find { it.type == split.backendType }
                ?: continue
            
            val status = executeGraphSplit(split, context, backend)
            if (status != GGMLBackendStatus.SUCCESS) {
                return status
            }
        }
        
        // Synchronize all backends
        for (backend in backendManager.getBackends()) {
            backend.synchronize()
        }
        
        return GGMLBackendStatus.SUCCESS
    }
    
    /**
     * Execute a single node
     */
    private fun executeNode(node: GGMLTensor, context: GGMLContext, backend: GGMLBackend) {
        // Create a temporary graph with just this node
        val tempGraph = createGraph(1)
        tempGraph.nodes[0] = node
        tempGraph.nNodes = 1
        
        // Execute the single-node graph
        backend.compute(tempGraph, context)
    }
    
    /**
     * Execute a graph split on a specific backend
     */
    private fun executeGraphSplit(split: GGMLGraphSplit, context: GGMLContext, backend: GGMLBackend): GGMLBackendStatus {
        // Create a subgraph for this split
        val subGraph = createGraph(split.nodes.size)
        
        for (i in split.nodes.indices) {
            subGraph.nodes[i] = split.nodes[i]
        }
        subGraph.nNodes = split.nodes.size
        
        return backend.compute(subGraph, context)
    }
    
    /**
     * Create graph splits for different backends
     */
    private fun createGraphSplits(graph: GGMLCGraph): List<GGMLGraphSplit> {
        val splits = mutableListOf<GGMLGraphSplit>()
        val backends = backendManager.getBackends()
        
        if (backends.isEmpty()) {
            return splits
        }
        
        var currentBackend = backends[0]
        var splitStart = 0
        val currentSplitNodes = mutableListOf<GGMLTensor>()
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            
            val bestBackend = backendManager.getBestBackend(node) ?: currentBackend
            
            if (bestBackend != currentBackend && currentSplitNodes.isNotEmpty()) {
                // Create split for previous backend
                splits.add(GGMLGraphSplit(
                    backendType = currentBackend.type,
                    startNode = splitStart,
                    endNode = i - 1,
                    nodes = currentSplitNodes.toList()
                ))
                
                currentSplitNodes.clear()
                splitStart = i
                currentBackend = bestBackend
            }
            
            currentSplitNodes.add(node)
        }
        
        // Add final split
        if (currentSplitNodes.isNotEmpty()) {
            splits.add(GGMLGraphSplit(
                backendType = currentBackend.type,
                startNode = splitStart,
                endNode = graph.nNodes - 1,
                nodes = currentSplitNodes.toList()
            ))
        }
        
        return splits
    }
    
    /**
     * Get scheduler statistics
     */
    fun getStats(): SchedulerStats {
        return SchedulerStats(
            maxWorkers = maxWorkers,
            strategy = strategy,
            availableBackends = backendManager.getBackends().map { it.name }
        )
    }
}

/**
 * Scheduler statistics
 */
data class SchedulerStats(
    val maxWorkers: Int,
    val strategy: GGMLSchedulingStrategy,
    val availableBackends: List<String>
)