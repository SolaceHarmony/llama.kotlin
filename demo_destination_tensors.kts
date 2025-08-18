#!/usr/bin/env kotlin

@file:Import("src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLTypes.kt")
@file:Import("src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLAlloc.kt")
@file:Import("src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLComputeOps.kt")
@file:Import("src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLOps.kt")

/**
 * Simple demonstration script showing the new destination tensor interface
 * for compute operations working correctly with graph allocator managed buffers.
 */

fun main() {
    println("=== Destination Tensor Interface Demo ===")
    
    val graphAllocator = GGMLGraphAllocator()
    val context = GGMLContext()
    
    // Create simple 1D tensors for testing
    val dims = longArrayOf(3)
    
    // Helper to create tensor
    fun createTensor(name: String, type: GGMLType, dims: LongArray, offset: ULong): GGMLTensor {
        val tensor = GGMLTensor(type = type)
        tensor.name = name
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dimSize ->
            if (index < GGML_MAX_DIMS) tensor.ne[index] = dimSize
        }
        val nb = ULongArray(GGML_MAX_DIMS) { 0uL }
        if (type.byteSize > 0uL) {
            nb[0] = type.byteSize
            for (d in 1 until GGML_MAX_DIMS) {
                val prevDimSize = tensor.ne.getOrElse(d - 1) { 1L }
                nb[d] = nb[d-1] * (if (prevDimSize > 0) prevDimSize.toULong() else 1uL)
            }
        }
        tensor.nb = nb
        tensor.bufferId = 0
        tensor.dataOffset = offset
        return tensor
    }
    
    // Calculate tensor byte size
    fun calcSize(type: GGMLType, ne: LongArray): ULong {
        var elements = 1UL
        for (dim in ne) {
            if (dim > 0L) elements *= dim.toULong()
        }
        return elements * type.byteSize
    }
    
    var offset = 0uL
    val size = calcSize(GGMLType.F32, dims)
    
    // Create source tensors
    val src0 = createTensor("src0", GGMLType.F32, dims, offset)
    offset += size
    val src1 = createTensor("src1", GGMLType.F32, dims, offset) 
    offset += size
    val dst = createTensor("dst", GGMLType.F32, dims, offset)
    
    // Initialize source data
    println("Initializing source tensors...")
    val src0Values = floatArrayOf(1.0f, 2.0f, 3.0f)
    val src1Values = floatArrayOf(4.0f, 5.0f, 6.0f)
    
    for (i in src0Values.indices) {
        src0.setFloat(graphAllocator, src0Values[i], i)
        src1.setFloat(graphAllocator, src1Values[i], i)
        dst.setFloat(graphAllocator, 0.0f, i) // Initialize to zero
    }
    
    println("Source 0: [${src0Values.joinToString(", ")}]")
    println("Source 1: [${src1Values.joinToString(", ")}]")
    
    // Test ADD operation
    println("\n=== Testing ADD Operation ===")
    computeAdd(graphAllocator, context, src0, src1, dst)
    
    print("Result:   [")
    for (i in 0 until dims[0].toInt()) {
        val result = dst.getFloat(graphAllocator, i)
        print("$result")
        if (i < dims[0].toInt() - 1) print(", ")
        
        // Verify correctness
        val expected = src0Values[i] + src1Values[i]
        if (kotlin.math.abs(result - expected) > 0.001f) {
            println("\nERROR: Expected $expected, got $result at index $i")
            return
        }
    }
    println("]")
    println("✓ ADD operation successful!")
    
    // Test MUL operation  
    println("\n=== Testing MUL Operation ===")
    // Reset destination
    for (i in 0 until dims[0].toInt()) {
        dst.setFloat(graphAllocator, 0.0f, i)
    }
    
    computeMul(graphAllocator, context, src0, src1, dst)
    
    print("Result:   [")
    for (i in 0 until dims[0].toInt()) {
        val result = dst.getFloat(graphAllocator, i)
        print("$result")
        if (i < dims[0].toInt() - 1) print(", ")
        
        // Verify correctness
        val expected = src0Values[i] * src1Values[i]
        if (kotlin.math.abs(result - expected) > 0.001f) {
            println("\nERROR: Expected $expected, got $result at index $i")
            return
        }
    }
    println("]")
    println("✓ MUL operation successful!")
    
    println("\n=== All Tests Passed! ===")
    println("The new destination tensor interface is working correctly.")
    println("Compute operations now write directly into allocator-managed buffers!")
}