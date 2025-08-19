package ai.solace.llamakotlin.core

/**
 * Kotlin Native port of GGML graph optimization functionality.
 * This file contains graph optimization passes to improve performance and reduce redundancy.
 */

/**
 * Optimization pass interface
 */
interface GGMLOptimizationPass {
    /**
     * Apply the optimization pass to a graph
     * @param graph The graph to optimize
     * @param context The GGML context
     * @return True if the graph was modified, false otherwise
     */
    fun apply(graph: GGMLCGraph, context: GGMLContext): Boolean
    
    /**
     * Get the name of this optimization pass
     */
    fun getName(): String
}

/**
 * Dead code elimination pass - removes unreachable operations
 */
class DeadCodeEliminationPass : GGMLOptimizationPass {
    
    override fun apply(graph: GGMLCGraph, context: GGMLContext): Boolean {
        if (graph.nNodes == 0) return false
        
        val reachable = mutableSetOf<GGMLTensor>()
        val visited = mutableSetOf<GGMLTensor>()
        
        // Mark all leaf nodes as reachable (outputs and parameters)
        for (i in 0 until graph.nLeafs) {
            val leaf = graph.leafs[i] ?: continue
            if (leaf.isOutput() || (leaf.flags and GGML_TENSOR_FLAG_PARAM) != 0) {
                markReachable(leaf, reachable, visited)
            }
        }
        
        // Mark all nodes that are used by reachable nodes
        for (i in graph.nNodes - 1 downTo 0) {
            val node = graph.nodes[i] ?: continue
            if (node in reachable) {
                markReachable(node, reachable, visited)
            }
        }
        
        // Remove unreachable nodes
        var removed = 0
        var writeIndex = 0
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            if (node in reachable) {
                graph.nodes[writeIndex] = node
                if (i < graph.grads.size) {
                    graph.grads[writeIndex] = graph.grads[i]
                }
                writeIndex++
            } else {
                removed++
                graph.nodes[i] = null
                if (i < graph.grads.size) {
                    graph.grads[i] = null
                }
            }
        }
        
        graph.nNodes = writeIndex
        
        return removed > 0
    }
    
    private fun markReachable(
        tensor: GGMLTensor, 
        reachable: MutableSet<GGMLTensor>, 
        visited: MutableSet<GGMLTensor>
    ) {
        if (tensor in visited) return
        visited.add(tensor)
        reachable.add(tensor)
        
        // Mark all source tensors as reachable
        for (src in tensor.src) {
            if (src != null) {
                markReachable(src, reachable, visited)
            }
        }
        
        // Mark view source as reachable
        tensor.viewSrc?.let { markReachable(it, reachable, visited) }
    }
    
    override fun getName(): String = "DeadCodeElimination"
}

/**
 * Redundant operation removal pass - removes duplicate operations
 */
class RedundantOpRemovalPass : GGMLOptimizationPass {
    
    override fun apply(graph: GGMLCGraph, context: GGMLContext): Boolean {
        if (graph.nNodes == 0) return false
        
        val opSignatures = mutableMapOf<String, GGMLTensor>()
        var removed = 0
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            
            // Skip operations that can't be deduplicated
            if (!canDeduplicate(node)) continue
            
            val signature = createOperationSignature(node)
            val existing = opSignatures[signature]
            
            if (existing != null && isEquivalentOperation(node, existing)) {
                // Replace this node with the existing equivalent one
                replaceNodeReferences(graph, node, existing)
                graph.nodes[i] = null
                removed++
            } else {
                opSignatures[signature] = node
            }
        }
        
        // Compact the graph if nodes were removed
        if (removed > 0) {
            compactGraph(graph)
        }
        
        return removed > 0
    }
    
    private fun canDeduplicate(tensor: GGMLTensor): Boolean {
        // Only deduplicate pure operations without side effects
        return when (tensor.op) {
            GGMLOp.ADD, GGMLOp.MUL, GGMLOp.SUB, GGMLOp.NEG, GGMLOp.DIV,
            GGMLOp.SQR, GGMLOp.SQRT, GGMLOp.ABS, GGMLOp.RELU, GGMLOp.GELU,
            GGMLOp.SILU -> true
            else -> false
        }
    }
    
    private fun createOperationSignature(tensor: GGMLTensor): String {
        val sb = StringBuilder()
        sb.append(tensor.op.name)
        sb.append(":")
        sb.append(tensor.type.name)
        
        // Add source tensor signatures
        for (src in tensor.src) {
            if (src != null) {
                sb.append(":").append(System.identityHashCode(src))
            } else {
                sb.append(":null")
            }
        }
        
        // Add dimensions
        for (i in 0 until GGML_MAX_DIMS) {
            sb.append(":").append(tensor.ne[i])
        }
        
        return sb.toString()
    }
    
    private fun isEquivalentOperation(a: GGMLTensor, b: GGMLTensor): Boolean {
        if (a.op != b.op || a.type != b.type) return false
        
        // Check dimensions match
        for (i in 0 until GGML_MAX_DIMS) {
            if (a.ne[i] != b.ne[i]) return false
        }
        
        // Check sources match
        for (i in 0 until GGML_MAX_SRC) {
            if (a.src[i] != b.src[i]) return false
        }
        
        return true
    }
    
    private fun replaceNodeReferences(graph: GGMLCGraph, oldNode: GGMLTensor, newNode: GGMLTensor) {
        // Replace references in all nodes
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            for (j in 0 until GGML_MAX_SRC) {
                if (node.src[j] == oldNode) {
                    node.src[j] = newNode
                }
            }
            if (node.viewSrc == oldNode) {
                node.viewSrc = newNode
            }
        }
        
        // Replace references in leaf nodes
        for (i in 0 until graph.nLeafs) {
            if (graph.leafs[i] == oldNode) {
                graph.leafs[i] = newNode
            }
        }
    }
    
    private fun compactGraph(graph: GGMLCGraph) {
        var writeIndex = 0
        for (i in 0 until graph.nNodes) {
            if (graph.nodes[i] != null) {
                graph.nodes[writeIndex] = graph.nodes[i]
                if (i < graph.grads.size) {
                    graph.grads[writeIndex] = graph.grads[i]
                }
                writeIndex++
            }
        }
        graph.nNodes = writeIndex
    }
    
    override fun getName(): String = "RedundantOpRemoval"
}

/**
 * Constant folding pass - pre-compute operations on constants
 */
class ConstantFoldingPass : GGMLOptimizationPass {
    
    override fun apply(graph: GGMLCGraph, context: GGMLContext): Boolean {
        if (graph.nNodes == 0) return false
        
        var folded = 0
        
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            
            if (canConstantFold(node)) {
                try {
                    // Compute the constant value
                    val result = computeConstantValue(node, context)
                    if (result != null) {
                        // Replace the operation with a constant
                        node.op = GGMLOp.NONE
                        node.data = result.data
                        node.src.fill(null)
                        folded++
                    }
                } catch (e: Exception) {
                    // If constant folding fails, skip this node
                    continue
                }
            }
        }
        
        return folded > 0
    }
    
    private fun canConstantFold(tensor: GGMLTensor): Boolean {
        // Check if all sources are constants (have data but no operation)
        for (src in tensor.src) {
            if (src == null) continue
            if (src.op != GGMLOp.NONE || src.data == null) return false
        }
        
        // Only fold simple operations
        return when (tensor.op) {
            GGMLOp.ADD, GGMLOp.MUL, GGMLOp.SUB, GGMLOp.NEG -> true
            else -> false
        }
    }
    
    private fun computeConstantValue(tensor: GGMLTensor, context: GGMLContext): GGMLTensor? {
        // Create a temporary allocator for constant computation
        val allocator = GGMLGraphAllocator()
        
        return when (tensor.op) {
            GGMLOp.ADD -> {
                val src0 = tensor.src[0] ?: return null
                val src1 = tensor.src[1] ?: return null
                computeAdd(allocator, context, src0, src1)
            }
            GGMLOp.MUL -> {
                val src0 = tensor.src[0] ?: return null
                val src1 = tensor.src[1] ?: return null
                computeMul(allocator, context, src0, src1)
            }
            GGMLOp.SUB -> {
                val src0 = tensor.src[0] ?: return null
                val src1 = tensor.src[1] ?: return null
                computeSub(allocator, context, src0, src1)
            }
            GGMLOp.NEG -> {
                val src0 = tensor.src[0] ?: return null
                computeNeg(allocator, context, src0)
            }
            else -> null
        }
    }
    
    override fun getName(): String = "ConstantFolding"
}

/**
 * Memory optimization pass - optimize memory usage patterns
 */
class MemoryOptimizationPass : GGMLOptimizationPass {
    
    override fun apply(graph: GGMLCGraph, context: GGMLContext): Boolean {
        if (graph.nNodes == 0) return false
        
        var optimized = 0
        val tensorUsage = analyzeTensorUsage(graph)
        
        // Mark tensors for in-place operations where possible
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            
            if (canBeInPlace(node, tensorUsage)) {
                // Mark for in-place optimization
                // This would be used by the allocator
                optimized++
            }
        }
        
        return optimized > 0
    }
    
    private fun analyzeTensorUsage(graph: GGMLCGraph): Map<GGMLTensor, Int> {
        val usage = mutableMapOf<GGMLTensor, Int>()
        
        // Count how many times each tensor is used
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            for (src in node.src) {
                if (src != null) {
                    usage[src] = (usage[src] ?: 0) + 1
                }
            }
        }
        
        return usage
    }
    
    private fun canBeInPlace(tensor: GGMLTensor, usage: Map<GGMLTensor, Int>): Boolean {
        if (!tensor.op.canBeInplace) return false
        
        val src0 = tensor.src[0] ?: return false
        
        // Can only do in-place if the source tensor is used only once
        return (usage[src0] ?: 0) <= 1 && !src0.isOutput()
    }
    
    override fun getName(): String = "MemoryOptimization"
}

/**
 * Graph optimizer that applies multiple optimization passes
 */
class GGMLGraphOptimizer {
    private val passes = mutableListOf<GGMLOptimizationPass>()
    private var maxIterations = 3
    
    init {
        // Default optimization passes
        passes.add(DeadCodeEliminationPass())
        passes.add(RedundantOpRemovalPass())
        passes.add(ConstantFoldingPass())
        passes.add(MemoryOptimizationPass())
    }
    
    /**
     * Add an optimization pass
     */
    fun addPass(pass: GGMLOptimizationPass) {
        passes.add(pass)
    }
    
    /**
     * Remove an optimization pass
     */
    fun removePass(pass: GGMLOptimizationPass) {
        passes.remove(pass)
    }
    
    /**
     * Set maximum number of iterations
     */
    fun setMaxIterations(iterations: Int) {
        maxIterations = iterations.coerceAtLeast(1)
    }
    
    /**
     * Optimize a computation graph
     */
    fun optimize(graph: GGMLCGraph, context: GGMLContext): OptimizationResult {
        val result = OptimizationResult()
        
        for (iteration in 0 until maxIterations) {
            var anyChanges = false
            
            for (pass in passes) {
                val changed = pass.apply(graph, context)
                if (changed) {
                    anyChanges = true
                    result.passResults[pass.getName()] = (result.passResults[pass.getName()] ?: 0) + 1
                }
            }
            
            result.iterations++
            
            if (!anyChanges) {
                // No more changes, optimization converged
                break
            }
        }
        
        return result
    }
}

/**
 * Result of graph optimization
 */
data class OptimizationResult(
    var iterations: Int = 0,
    val passResults: MutableMap<String, Int> = mutableMapOf()
)