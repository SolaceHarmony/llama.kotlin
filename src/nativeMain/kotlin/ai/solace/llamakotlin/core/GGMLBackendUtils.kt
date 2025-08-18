package ai.solace.llamakotlin.core

/**
 * Backend utilities and integration helpers for GGML operations.
 * 
 * This file provides utility functions for working with backends,
 * hybrid execution, and backend selection logic.
 */

/**
 * Backend selection strategy for hybrid execution
 */
enum class GGMLBackendSelectionStrategy {
    /** Always use CPU backend */
    CPU_ONLY,
    /** Always use Metal backend (if available) */
    METAL_ONLY,
    /** Use Metal for large operations, CPU for small ones */
    HYBRID_SIZE_BASED,
    /** Use Metal for supported ops, CPU for unsupported ones */
    HYBRID_OP_BASED,
    /** Automatic selection based on operation and tensor properties */
    AUTO
}

/**
 * Backend manager for handling multiple backends and hybrid execution
 */
class GGMLBackendManager {
    private val availableBackends = mutableMapOf<String, GGMLBackend>()
    private var primaryBackend: GGMLBackend? = null
    private var fallbackBackend: GGMLBackend? = null
    private var selectionStrategy = GGMLBackendSelectionStrategy.AUTO
    
    init {
        initializeBackends()
    }
    
    /**
     * Initialize available backends
     */
    private fun initializeBackends() {
        // Initialize CPU backend (always available)
        val cpuBackend = GGMLCpuBackend()
        availableBackends[cpuBackend.getName()] = cpuBackend
        fallbackBackend = cpuBackend
        
        // Try to initialize Metal backend
        val metalBackend = GGMLMetalBackend()
        if (metalBackend is GGMLMetalBackend) {
            if (metalBackend.initialize()) {
                availableBackends[metalBackend.getName()] = metalBackend
                primaryBackend = metalBackend
            }
        }
        
        // Set primary backend to CPU if Metal failed
        if (primaryBackend == null) {
            primaryBackend = cpuBackend
        }
    }
    
    /**
     * Get the list of available backend names
     */
    fun getAvailableBackends(): List<String> {
        return availableBackends.keys.toList()
    }
    
    /**
     * Get a backend by name
     */
    fun getBackend(name: String): GGMLBackend? {
        return availableBackends[name]
    }
    
    /**
     * Get the primary backend
     */
    fun getPrimaryBackend(): GGMLBackend? = primaryBackend
    
    /**
     * Get the fallback backend (usually CPU)
     */
    fun getFallbackBackend(): GGMLBackend? = fallbackBackend
    
    /**
     * Set the backend selection strategy
     */
    fun setSelectionStrategy(strategy: GGMLBackendSelectionStrategy) {
        selectionStrategy = strategy
    }
    
    /**
     * Select the best backend for a given operation
     */
    fun selectBackend(tensor: GGMLTensor): GGMLBackend? {
        return when (selectionStrategy) {
            GGMLBackendSelectionStrategy.CPU_ONLY -> fallbackBackend
            GGMLBackendSelectionStrategy.METAL_ONLY -> primaryBackend
            GGMLBackendSelectionStrategy.HYBRID_SIZE_BASED -> selectBackendBySize(tensor)
            GGMLBackendSelectionStrategy.HYBRID_OP_BASED -> selectBackendByOperation(tensor)
            GGMLBackendSelectionStrategy.AUTO -> selectBackendAuto(tensor)
        }
    }
    
    /**
     * Select backend based on tensor size
     */
    private fun selectBackendBySize(tensor: GGMLTensor): GGMLBackend? {
        val elementCount = tensor.numElements()
        val threshold = 1000L // Arbitrary threshold for demonstration
        
        return if (elementCount > threshold && primaryBackend != fallbackBackend) {
            // Use Metal for large tensors
            if (primaryBackend?.supportsOp(tensor) == true) {
                primaryBackend
            } else {
                fallbackBackend
            }
        } else {
            // Use CPU for small tensors
            fallbackBackend
        }
    }
    
    /**
     * Select backend based on operation type
     */
    private fun selectBackendByOperation(tensor: GGMLTensor): GGMLBackend? {
        // Check if primary backend supports the operation
        if (primaryBackend?.supportsOp(tensor) == true) {
            return primaryBackend
        }
        
        // Fall back to CPU
        return if (fallbackBackend?.supportsOp(tensor) == true) {
            fallbackBackend
        } else {
            null
        }
    }
    
    /**
     * Automatic backend selection based on multiple factors
     */
    private fun selectBackendAuto(tensor: GGMLTensor): GGMLBackend? {
        // Combine size and operation considerations
        val elementCount = tensor.numElements()
        val sizeThreshold = 10000L
        
        // For very small operations, always use CPU
        if (elementCount < 100L) {
            return fallbackBackend
        }
        
        // For large operations that Metal supports, use Metal
        if (elementCount > sizeThreshold && primaryBackend?.supportsOp(tensor) == true) {
            return primaryBackend
        }
        
        // For medium operations, check operation type preference
        return when (tensor.op) {
            GGMLOp.MUL_MAT -> primaryBackend?.takeIf { it.supportsOp(tensor) } ?: fallbackBackend
            GGMLOp.ADD, GGMLOp.MUL -> primaryBackend?.takeIf { it.supportsOp(tensor) } ?: fallbackBackend
            else -> fallbackBackend
        }
    }
    
    /**
     * Create a graph with automatic backend selection
     */
    fun createGraphWithBackend(size: Int, strategy: GGMLBackendSelectionStrategy? = null): GGMLCGraph {
        val oldStrategy = selectionStrategy
        strategy?.let { setSelectionStrategy(it) }
        
        val backend = when (selectionStrategy) {
            GGMLBackendSelectionStrategy.CPU_ONLY -> fallbackBackend
            GGMLBackendSelectionStrategy.METAL_ONLY -> primaryBackend
            else -> primaryBackend // Default to primary backend for graph creation
        }
        
        val graph = createGraph(size, backend)
        setSelectionStrategy(oldStrategy) // Restore old strategy
        return graph
    }
    
    /**
     * Compute a graph with hybrid execution
     */
    fun computeGraphHybrid(graph: GGMLCGraph): GGMLStatus {
        val allocator = graph.allocator
        if (allocator == null) {
            return GGMLStatus.FAILED
        }
        
        // For now, use the graph's associated backend
        // In a full hybrid implementation, we would analyze each node
        // and potentially execute different nodes on different backends
        return computeGraphWithBackend(graph)
    }
    
    /**
     * Get backend information for debugging
     */
    fun getBackendInfo(): Map<String, Any> {
        val info = mutableMapOf<String, Any>()
        
        info["availableBackends"] = availableBackends.keys.toList()
        info["primaryBackend"] = primaryBackend?.getName() ?: "None"
        info["fallbackBackend"] = fallbackBackend?.getName() ?: "None" 
        info["selectionStrategy"] = selectionStrategy.name
        
        // Add backend-specific info
        availableBackends.forEach { (name, backend) ->
            info["${name}_guid"] = backend.getGuid()
            info["${name}_bufferType"] = backend.getDefaultBufferType().getName()
            info["${name}_alignment"] = backend.getAlignment()
            info["${name}_maxSize"] = backend.getMaxSize()
            
            // Add Metal-specific info if available
            if (backend is GGMLMetalBackend) {
                info["${name}_deviceInfo"] = backend.getDeviceInfo()
            }
        }
        
        return info
    }
    
    /**
     * Clean up all backends
     */
    fun cleanup() {
        availableBackends.values.forEach { backend ->
            try {
                backend.free()
            } catch (e: Exception) {
                println("Error freeing backend ${backend.getName()}: ${e.message}")
            }
        }
        availableBackends.clear()
        primaryBackend = null
        fallbackBackend = null
    }
}

/**
 * Global backend manager instance
 */
val globalBackendManager = GGMLBackendManager()

/**
 * Convenience function to create a graph with the global backend manager
 */
fun createGraphWithGlobalBackend(
    size: Int, 
    strategy: GGMLBackendSelectionStrategy = GGMLBackendSelectionStrategy.AUTO
): GGMLCGraph {
    return globalBackendManager.createGraphWithBackend(size, strategy)
}

/**
 * Convenience function to compute a graph with hybrid execution
 */
fun computeGraphHybrid(graph: GGMLCGraph): GGMLStatus {
    return globalBackendManager.computeGraphHybrid(graph)
}