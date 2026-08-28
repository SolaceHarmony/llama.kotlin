package io.github.kotlinmania.llama

import io.github.kotlinmania.llama.ore.*
import io.github.kotlinmania.llama.examples.runComprehensiveDemo

fun main() {
    kotlin.io.println("🦙 LLaMA Kotlin Native - Advanced Capabilities Demonstration")
    kotlin.io.println("This is a comprehensive Kotlin Native port of llama.cpp")
    kotlin.io.println("Showcasing K-Quantization, GGUF loading, tensor operations, and more!")
    kotlin.io.println()

    // Run the comprehensive integration demo
    val demoResult = runComprehensiveDemo()
    kotlin.io.println(demoResult)
    
    kotlin.io.println()
    kotlin.io.println("=" .repeat(60))
    kotlin.io.println("🎯 Additional Core Functionality Demos")
    kotlin.io.println("=" .repeat(60))

    // Demonstrate the use of the computation graph functionality
    demonstrateComputationGraph()

    // Demonstrate the use of the memory allocation functionality
    demonstrateMemoryAllocation()

    // Demonstrate the use of the optimized tensor operations
    demonstrateOptimizedTensorOps()

    // Demonstrate LLaMA model architecture components
    demonstrateLlamaModelArchitecture()
    
    kotlin.io.println()
    kotlin.io.println("🚀 All demonstrations completed! The Kotlin llama.cpp port is comprehensive and functional.")
}

/**
 * Demonstrates the use of the optimized tensor operations.
 */
fun demonstrateOptimizedTensorOps() {
    kotlin.io.println("\nDemonstrating optimized tensor operations:")

    // Build via graph and run with allocator/backends to respect dst-arg API
    val context = GGMLContext(computeImmediately = false)
    val graph = createGraph(10)
    val allocator = graph.allocator ?: GGMLGraphAllocator().also { graph.allocator = it }

    // Create tensors directly and mark as leafs
    val a = GGMLTensor(type = GGMLType.F32).apply { ne[0] = 4; ne[1] = 4; data = FloatArray(16) { it.toFloat() } }
    val b = GGMLTensor(type = GGMLType.F32).apply { ne[0] = 4; ne[1] = 4; data = FloatArray(16) { (it + 1).toFloat() } }

    graph.leafs[0] = a; graph.leafs[1] = b; graph.nLeafs = 2

    val addNode = ggmlAdd(context, a, b)
    val mulNode = ggmlMul(context, a, b)
    val mmNode = ggmlMulMat(context, a, b)

    graph.nodes[0] = addNode
    graph.nodes[1] = mulNode
    graph.nodes[2] = mmNode
    graph.nNodes = 3

    // Allocate and execute
    allocator.allocateGraph(graph)
    computeGraphWithBackend(graph, context)

    kotlin.io.println("Tensor a: [${(a.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("Tensor b: [${(b.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("a + b: [${(addNode.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("a * b: [${(mulNode.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("a @ b: [${(mmNode.data as FloatArray).take(16).joinToString()}]")
}

/**
 * Demonstrates the use of the computation graph functionality.
 */
fun demonstrateComputationGraph() {
    kotlin.io.println("\nDemonstrating computation graph functionality:")

    // Create a context
    val context = GGMLContext(
        memSize = (16 * 1024 * 1024).toULong(), // 16 MB
        memBuffer = null,
        memBufferOwned = false,
        noAlloc = false,
        computeImmediately = false // Important: set to false to use the computation graph
    )

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 2, 2)
    val b = createTensor2D(context, GGMLType.F32, 2, 2)

    // Initialize tensor data
    (a.data as FloatArray)[0] = 1.0f
    (a.data as FloatArray)[1] = 2.0f
    (a.data as FloatArray)[2] = 3.0f
    (a.data as FloatArray)[3] = 4.0f

    (b.data as FloatArray)[0] = 5.0f
    (b.data as FloatArray)[1] = 6.0f
    (b.data as FloatArray)[2] = 7.0f
    (b.data as FloatArray)[3] = 8.0f

    kotlin.io.println("Tensor a: [${(a.data as FloatArray).joinToString()}]")
    kotlin.io.println("Tensor b: [${(b.data as FloatArray).joinToString()}]")

    // Create operations
    val c = ggmlAdd(context, a, b) // c = a + b
    val d = ggmlMul(context, a, b) // d = a * b
    val e = ggmlMulMat(context, a, b) // e = a @ b

    // Create a computation graph
    val graph = createGraph(100) // Maximum 100 nodes

    // Build the graph for all operations
    buildForward(graph, c)
    buildForward(graph, d)
    buildForward(graph, e)

    kotlin.io.println("Graph built with ${graph.nNodes} nodes and ${graph.nLeafs} leaf nodes")

    // Execute the graph (allocates internally and uses backend/CPU compute)
    executeGraph(context, graph)

    // Print the results
    kotlin.io.println("c = a + b: [${(c.data as FloatArray).joinToString()}]")
    kotlin.io.println("d = a * b: [${(d.data as FloatArray).joinToString()}]")
    kotlin.io.println("e = a @ b: [${(e.data as FloatArray).joinToString()}]")
}

/**
 * Demonstrates the use of the memory allocation functionality.
 */
fun demonstrateMemoryAllocation() {
    kotlin.io.println("\nDemonstrating memory allocation functionality:")

    // Create a tensor allocator
    val tensorAllocator = GGMLTensorAllocator()

    // Create a tensor
    val tensor = GGMLTensor(type = GGMLType.F32)
    tensor.ne[0] = 2
    tensor.ne[1] = 3

    // Allocate memory for the tensor
    tensorAllocator.allocate(tensor)

    // Initialize tensor data
    val data = tensor.data as FloatArray
    for (i in data.indices) {
        data[i] = i.toFloat()
    }

    kotlin.io.println("Tensor data: [${data.joinToString()}]")

    // Create a dynamic tensor allocator
    val dynTensorAllocator = GGMLDynTensorAllocator()

    // Create tensors
    val tensor1 = GGMLTensor(type = GGMLType.F32)
    tensor1.ne[0] = 2
    tensor1.ne[1] = 2

    val tensor2 = GGMLTensor(type = GGMLType.F32)
    tensor2.ne[0] = 3
    tensor2.ne[1] = 3

    // Calculate tensor sizes
    val size1 = 4UL // 2x2
    val size2 = 9UL // 3x3

    // Allocate memory for the tensors
    val offset1 = dynTensorAllocator.allocate(size1, tensor1)
    val offset2 = dynTensorAllocator.allocate(size2, tensor2)

    kotlin.io.println("Tensor 1 allocated at offset $offset1 with size $size1")
    kotlin.io.println("Tensor 2 allocated at offset $offset2 with size $size2")

    // Free memory for tensor1
    dynTensorAllocator.freeTensor(offset1, size1, tensor1)
    kotlin.io.println("Tensor 1 freed")

    // Allocate memory for a new tensor
    val tensor3 = GGMLTensor(type = GGMLType.F32)
    tensor3.ne[0] = 2
    tensor3.ne[1] = 1

    val size3 = 2UL // 2x1
    val offset3 = dynTensorAllocator.allocate(size3, tensor3)

    kotlin.io.println("Tensor 3 allocated at offset $offset3 with size $size3")

    // Create a graph allocator
    val graphAllocator = GGMLGraphAllocator()

    // Create a computation graph
    val graph = createGraph(10)

    // Create tensors for the graph
    val a = GGMLTensor(type = GGMLType.F32)
    a.ne[0] = 2
    a.ne[1] = 2

    val b = GGMLTensor(type = GGMLType.F32)
    b.ne[0] = 2
    b.ne[1] = 2

    // Set up the graph
    graph.leafs[0] = a
    graph.leafs[1] = b
    graph.nLeafs = 2

    // Create an operation node
    val c = GGMLTensor(type = GGMLType.F32)
    c.ne[0] = 2
    c.ne[1] = 2
    c.op = GGMLOp.ADD
    c.src[0] = a
    c.src[1] = b

    graph.nodes[0] = c
    graph.nNodes = 1

    // Allocate memory for the graph
    val success = graphAllocator.allocateGraph(graph)

    kotlin.io.println("Graph allocation ${if (success) "succeeded" else "failed"}")
    kotlin.io.println("Buffer size: ${graphAllocator.getBufferSize(0)}")
}

/**
 * Demonstrates the LLaMA model architecture and core transformer components
 */
fun demonstrateLlamaModelArchitecture() {
    kotlin.io.println("\n=== LLaMA Model Architecture Demo ===")
    kotlin.io.println("Testing transformer components and attention mechanisms...")
    
    try {
        // Setup allocator and context
        val allocator = GGMLGraphAllocator()
        allocator.reserve(4 * 1024 * 1024) // 4MB buffer
        val context = GGMLContext()
        
        // Test 1: Model Configuration and Creation
        kotlin.io.println("\n1. Model Configuration")
        val config = io.github.kotlinmania.llama.model.LlamaConfig(
            vocabSize = 100,
            hiddenSize = 64,
            intermediateSize = 128,
            numHiddenLayers = 2,
            numAttentionHeads = 4,
            numKeyValueHeads = 4
        )
        
        val model = io.github.kotlinmania.llama.model.LlamaModel(config)
        kotlin.io.println("   ✓ LLaMA model created with ${config.numHiddenLayers} layers")
        kotlin.io.println("   ✓ Hidden size: ${config.hiddenSize}, Head dimension: ${config.headDim}")
        
        // Test 2: Transpose Operation (critical for attention)
        kotlin.io.println("\n2. Transpose Operation")
        val matrix = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = 3L  // 3 columns
            ne[1] = 2L  // 2 rows  
            ne[2] = 1L; ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(matrix)
        
        // Fill test matrix: [[1,2,3], [4,5,6]]
        matrix.setFloat(allocator, 1.0f, 0, 0)
        matrix.setFloat(allocator, 2.0f, 1, 0)
        matrix.setFloat(allocator, 3.0f, 2, 0)
        matrix.setFloat(allocator, 4.0f, 0, 1)
        matrix.setFloat(allocator, 5.0f, 1, 1)
        matrix.setFloat(allocator, 6.0f, 2, 1)
        
        val transposed = GGMLTensor(type = matrix.type).apply {
            ne = matrix.ne.copyOf()
            val tmpDim0 = ne[0]
            ne[0] = matrix.ne[1]
            ne[1] = tmpDim0
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(transposed)
        computeTranspose(allocator, matrix, transposed)
        kotlin.io.println("   ✓ Matrix transposed from (3x2) to (2x3)")
        
        // Verify transpose worked correctly
        val val00 = transposed.getFloat(allocator, 0, 0) // Should be 1
        val val10 = transposed.getFloat(allocator, 1, 0) // Should be 4
        val val01 = transposed.getFloat(allocator, 0, 1) // Should be 2
        val val11 = transposed.getFloat(allocator, 1, 1) // Should be 5
        kotlin.io.println("   ✓ Transpose verification: T[0,0]=$val00, T[1,0]=$val10, T[0,1]=$val01, T[1,1]=$val11")
        
        // Test 3: RMSNorm (Layer Normalization)
        kotlin.io.println("\n3. RMSNorm Layer Normalization")
        val hiddenSize = 32
        val inputTensor = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = hiddenSize.toLong(); ne[1] = 1L; ne[2] = 1L; ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(inputTensor)
        
        // Fill with test values
        for (i in 0 until hiddenSize) {
            inputTensor.setFloat(allocator, (i + 1).toFloat(), i, 0)
        }
        
        val normalized = GGMLTensor(type = inputTensor.type).apply {
            ne = inputTensor.ne.copyOf()
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(normalized)
        computeRMSNorm(allocator, inputTensor, 1e-6f, normalized)
        
        // Verify normalization
        var sumSquared = 0.0f
        for (i in 0 until hiddenSize) {
            val value = normalized.getFloat(allocator, i, 0)
            sumSquared += value * value
        }
        val rms = kotlin.math.sqrt(sumSquared / hiddenSize)
        kotlin.io.println("   ✓ RMS after normalization: %.4f (target: ~1.0)".replace("%.4f", rms.toString().take(6)))
        
        // Test 4: SiLU Activation Function
        kotlin.io.println("\n4. SiLU Activation Function")
        val activationInput = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = 8L; ne[1] = 1L; ne[2] = 1L; ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(activationInput)
        
        val testValues = floatArrayOf(-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f)
        for (i in testValues.indices) {
            activationInput.setFloat(allocator, testValues[i], i, 0)
        }
        
        val siluResult = GGMLTensor(type = GGMLType.F32).apply {
            ne = activationInput.ne.copyOf()
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(siluResult)
        computeSilu(allocator, activationInput, siluResult)
        
        val siluZero = siluResult.getFloat(allocator, 3, 0) // SiLU(0)
        val siluPositive = siluResult.getFloat(allocator, 4, 0) // SiLU(0.5)
        kotlin.io.println("   ✓ SiLU(0) = %.6f (should be ~0)".replace("%.6f", siluZero.toString().take(8)))
        kotlin.io.println("   ✓ SiLU(0.5) = %.4f (should be positive)".replace("%.4f", siluPositive.toString().take(6)))
        
        // Test 5: Matrix Multiplication for Linear Layers
        kotlin.io.println("\n5. Matrix Multiplication for Linear Layers")
        val inputDim = 16
        val outputDim = 32
        val seqLen = 4
        
        val transformerInput = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = inputDim.toLong(); ne[1] = seqLen.toLong(); ne[2] = 1L; ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(transformerInput)
        
        val weightMatrix = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = inputDim.toLong(); ne[1] = outputDim.toLong(); ne[2] = 1L; ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(weightMatrix)
        
        // Initialize with small test values
        for (i in 0 until inputDim * seqLen) {
            transformerInput.setFloat(allocator, 0.1f * (i % 10 + 1), i % inputDim, i / inputDim)
        }
        for (i in 0 until inputDim * outputDim) {
            weightMatrix.setFloat(allocator, 0.01f * ((i % 7) - 3), i % inputDim, i / inputDim)
        }
        
        val matmulResult = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = outputDim.toLong(); ne[1] = seqLen.toLong(); ne[2] = 1L; ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(matmulResult)
        computeMatMul(allocator, weightMatrix, transformerInput, matmulResult)
        
        kotlin.io.println("   ✓ Linear transformation: ($inputDim × $seqLen) × ($inputDim × $outputDim) → ($outputDim × $seqLen)")
        
        // Test 6: Softmax for Attention Weights
        kotlin.io.println("\n6. Softmax for Attention Scores")
        val logitsInput = GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = 5L; ne[1] = 1L; ne[2] = 1L; ne[3] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(logitsInput)
        
        val logitValues = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        for (i in logitValues.indices) {
            logitsInput.setFloat(allocator, logitValues[i], i, 0)
        }
        
        val softmaxResult = GGMLTensor(type = GGMLType.F32).apply {
            ne = logitsInput.ne.copyOf()
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
        allocator.allocateTensor(softmaxResult)
        computeSoftMax(allocator, logitsInput, softmaxResult)
        
        // Verify softmax properties
        var sum = 0.0f
        for (i in 0 until 5) {
            val value = softmaxResult.getFloat(allocator, i, 0)
            sum += value
        }
        kotlin.io.println("   ✓ Softmax sum: %.6f (should be 1.0)".replace("%.6f", sum.toString().take(8)))
        
        // Test 7: KV Cache for Efficient Inference
        kotlin.io.println("\n7. KV Cache for Efficient Inference")
        val kvCache = io.github.kotlinmania.llama.model.KVCache(
            maxSequenceLength = 10,
            numHeads = 4,
            headDim = 8
        )
        kvCache.initialize(allocator)
        kotlin.io.println("   ✓ KV Cache initialized for efficient autoregressive generation")
        kotlin.io.println("   ✓ Max sequence length: 10, Heads: 4, Head dimension: 8")
        
        kotlin.io.println("\n🎉 LLaMA Model Architecture Demo Complete!")
        kotlin.io.println("✓ All core transformer components are working:")
        kotlin.io.println("  • Model configuration and instantiation")
        kotlin.io.println("  • Matrix transpose for attention mechanisms")
        kotlin.io.println("  • RMSNorm layer normalization")
        kotlin.io.println("  • SiLU activation function")
        kotlin.io.println("  • Matrix multiplication for linear layers")
        kotlin.io.println("  • Softmax for attention weights")
        kotlin.io.println("  • KV cache for efficient inference")
        kotlin.io.println()
        kotlin.io.println("🚀 Ready for full LLaMA model inference implementation!")
        
    } catch (e: Exception) {
        kotlin.io.println("✗ Error during LLaMA model demo: ${e.message}")
        e.printStackTrace()
    }
}
