package ai.solace.llamakotlin.core

import ai.solace.llamakotlin.backends.cpu.CPUBackend
import ai.solace.llamakotlin.ggml.GGMLType
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GGMLGraphConcurrencyTest {

    private fun createTestContext(): GGMLContext {
        val context = GGMLContext()
        // Ensure graphAllocator is initialized. Let's assume a default constructor or a simple setup.
        // If GGMLGraphAllocator needs specific buffer sizes, those would be configured here.
        context.graphAllocator = GGMLGraphAllocator()
        context.backend = CPUBackend
        // The GGMLGraphAllocator's constructor should initialize a default buffer.
        // Tests will rely on this or allocateGraph resizing if necessary.
        return context
    }

    // Helper to create a leaf tensor with F32 type and initial data.
    // This data will be copied by allocateGraph into the graph allocator's managed buffer if tensor.data is not null.
    private fun createF32LeafTensor(value: Float, vararg ne: Long, name: String = ""): GGMLTensor {
        // GGMLTensor constructor handles shape initialization (e.g., scalar if ne is empty or all zeros)
        val tensor = GGMLTensor(type = GGMLType.F32, shape = ne, name = name)
        val numElements = tensor.numElements().toInt()
        assertTrue(numElements > 0, "Tensor '${name}' must have elements to hold data. Shape: ${ne.joinToString()}. Calculated elements: $numElements")
        tensor.data = FloatArray(numElements) { value } // Set data directly
        return tensor
    }

    // Helper to get data from a tensor after execution.
    // It uses tensor.getFloat(allocator, indices...) which is the standard way to access data.
    private fun getTensorData(context: GGMLContext, tensor: GGMLTensor): FloatArray {
        val allocator = context.graphAllocator ?: throw IllegalStateException("Graph allocator must be initialized.")
        assertTrue(tensor.bufferId != -1, "Tensor '${tensor.name}' must be allocated to a buffer to read its data.")

        val numElements = tensor.numElements().toInt()
        assertTrue(numElements > 0, "Tensor '${tensor.name}' must have elements to read. Shape: ${tensor.ne.joinToString()}.")
        val result = FloatArray(numElements)

        if (numElements == 1 && (tensor.nDims == 0 || tensor.ne.all { it == 1L })) { // Scalar case
            result[0] = tensor.getFloat(allocator)
        } else if (tensor.nDims == 1) { // 1D tensor
            for (i in 0 until numElements) {
                result[i] = tensor.getFloat(allocator, i.toLong())
            }
        } else {
            // For multi-dimensional tensors, a flat index access might not be what getFloat expects
            // or might not be implemented. For simplicity, current tests will focus on scalar/1D results.
            // If multi-D results are needed, this helper must be adapted or use specific per-dimension indices.
            // For now, let's assume tests will verify simple (scalar/1D) outputs or use ops that produce them.
            // As a placeholder for flat access if tensor.getFloat supports it:
            // TODO: Confirm/adjust nD access for getFloat if tests need it.
            // For now, this loop is okay if getFloat(allocator, flatIndex) is valid.
            // Let's refine this to be more explicit for common test cases (1D).
            // If a tensor is multi-dimensional but numElements is used, it implies flat iteration.
            // The `getFloat(allocator, vararg indices: Long)` is the primary accessor.
            // We need to convert flat index `i` to `vararg indices`.
            // This is complex. Let's simplify: test results that are scalar or 1D for now.
            // The existing loop for 1D (nDims == 1) is fine.
            // If a tensor is, for example, [1,1,5], numElements is 5, but nDims could be 3.
            // A truly generic solution needs to map flat index i to (idx0, idx1, idx2, ...).
            // Let's restrict test verification to tensors where this is simple.
            // For now, if not scalar and not 1D, this helper will fail or misbehave.
            // This is a limitation of the test helper, not necessarily the core API.
            // A common pattern: result tensor E is [1,1,1,1] (scalar-like).
            // Let's handle the common case of a single value first.
            if (numElements == 1) { // Covers shapes like [1], [1,1], [1,1,1], [1,1,1,1]
                 result[0] = tensor.getFloat(allocator) // For scalar, no indices
            } else {
                // If we need to test multi-element tensors that aren't strictly 1D,
                // this part of the helper needs to be more robust.
                // For now, assuming tests will mostly deal with single float results or true 1D arrays.
                // The previous 1D check (tensor.nDims == 1) is good.
                // If numElements > 1 AND nDims > 1, this flat loop is likely wrong for getFloat.
                // For now, this helper will primarily support scalar or 1D tensors.
                // Add a check:
                assertTrue(tensor.nDims <= 1 || tensor.ne.count { it > 1L } <= 1,
                    "getTensorData helper currently supports scalar or 1D tensors for multi-element data. Tensor '${tensor.name}' shape: ${tensor.ne.joinToString()}")
                for (i in 0 until numElements) {
                     result[i] = tensor.getFloat(allocator, i.toLong()) // Works for 1D, or if getFloat can take flat index
                }
            }
        }
        return result
    }

    @Test
    fun testSimpleParallelGraphExecution() {
        val context = createTestContext()
        val graphAllocator = context.graphAllocator!!

        // Setup Leaf Tensors
        // Leaf tensor data is initially in their .data field (as FloatArray)
        // allocateGraph will copy this data into the graphAllocator's managed buffer.
        val tensorA = createF32LeafTensor(1.0f, name = "A") // Scalar tensor
        val tensorB = createF32LeafTensor(2.0f, name = "B") // Scalar tensor

        // Construct Graph: E = (A+A) + (B+B)
        // C = A + A
        val tensorC = add(context, tensorA, tensorA)
        tensorC.name = "C"
        // D = B + B
        val tensorD = add(context, tensorB, tensorB)
        tensorD.name = "D"
        // E = C + D
        val tensorE = add(context, tensorC, tensorD)
        tensorE.name = "E"
        tensorE.isOutput = true // Mark E as an output tensor

        // Build and Execute
        val cgraph = createGraph(size = 10) // Size should be enough for nodes + leafs
        buildForward(cgraph, tensorE)

        assertTrue(graphAllocator.allocateGraph(cgraph), "Graph allocation failed")

        // Before execution, check initial values if they are in graph allocator
        // For leafs A and B, their data should have been copied to the allocator by allocateGraph.
        val initialAData = getTensorData(context, tensorA)
        assertEquals(1.0f, initialAData[0], "Initial data for A is incorrect.")
        val initialBData = getTensorData(context, tensorB)
        assertEquals(2.0f, initialBData[0], "Initial data for B is incorrect.")

        executeGraph(context, cgraph)

        // Verification
        // Retrieve data for E
        val resultEData = getTensorData(context, tensorE)
        assertEquals(1, resultEData.size, "Result E should be a scalar.")
        assertEquals(6.0f, resultEData[0], 0.001f, "Result of E = (1+1) + (2+2) should be 6.0.")

        // Optionally, verify intermediate tensors if their memory wasn't immediately reused
        // This depends on allocation strategy and graph structure.
        // For this simple graph, C and D might still be valid.
        val resultCData = getTensorData(context, tensorC)
        assertEquals(2.0f, resultCData[0], 0.001f, "Intermediate C = A+A should be 2.0.")

        val resultDData = getTensorData(context, tensorD)
        assertEquals(4.0f, resultDData[0], 0.001f, "Intermediate D = B+B should be 4.0.")

        // Check memory deallocation (indirectly)
        // We expect that A and B (inputs to C and D) should have their numChildren decremented.
        // If C and D are not views and not outputs, their memory might be freed after E uses them.
        val usageA = graphAllocator.getTensorUsageInfo(tensorA)
        assertNotNull(usageA, "UsageInfo for A not found")
        assertEquals(0, usageA.numChildren.value, "Tensor A should have 0 children remaining after C is computed twice (or C's children are done).")
        // numChildren for A is tricky here if C = A+A. It's used twice by the same node.
        // The current logic in GGMLGraph.executeForward iterates node.src. For C=A+A, src[0]=A, src[1]=A.
        // So, A's numChildren will be decremented twice by the job for C.
        // If initial numChildren for A (due to C=A+A) was 2, it should become 0.
        // Let's trace: analyzeTensorUsage: C depends on A twice. So, A.numChildren becomes 2.
        // When C's job finishes, it loops src:
        // srcTensor = A. numChildren becomes 1. freeTensorDataIfUnused(A) -> no, children=1
        // srcTensor = A. numChildren becomes 0. freeTensorDataIfUnused(A) -> yes, if not output & owns memory
        // This seems correct.

        val usageB = graphAllocator.getTensorUsageInfo(tensorB)
        assertNotNull(usageB, "UsageInfo for B not found")
        assertEquals(0, usageB.numChildren.value, "Tensor B should have 0 children remaining.")

        val usageC = graphAllocator.getTensorUsageInfo(tensorC)
        assertNotNull(usageC, "UsageInfo for C not found")
        assertEquals(0, usageC.numChildren.value, "Tensor C should have 0 children remaining after E is computed.")
        // If C was not inplace for E and C is not an output, its memory might be freed.
        // Check if C's memory was freed (if it owned it and was not an output)
        if (usageC.ownsMemory && !usageC.isOutputTensor) { // This would be true if it was freed
             // This assertion is tricky because ownsMemory becomes false *after* freeing.
             // So, if it *was* freed, usageC.ownsMemory should be false now.
            assertEquals(false, usageC.ownsMemory, "Tensor C's memory should have been marked as not owned if freed.")
        }


        val usageD = graphAllocator.getTensorUsageInfo(tensorD)
        assertNotNull(usageD, "UsageInfo for D not found")
        assertEquals(0, usageD.numChildren.value, "Tensor D should have 0 children remaining.")
         if (usageD.ownsMemory && !usageD.isOutputTensor) {
            assertEquals(false, usageD.ownsMemory, "Tensor D's memory should have been marked as not owned if freed.")
        }

        // Tensor E is an output, so it should not be freed.
        val usageE = graphAllocator.getTensorUsageInfo(tensorE)
        assertNotNull(usageE, "UsageInfo for E not found")
        assertTrue(usageE.isOutputTensor, "Tensor E should be an output tensor.")
        assertTrue(usageE.ownsMemory, "Tensor E (output) should still own its memory.") // Assuming it wasn't inplace from a non-output.
    }

    @Test
    fun testGraphWithSharedInput() {
        val context = createTestContext()
        val graphAllocator = context.graphAllocator!!

        // Setup Leaf Tensor
        val tensorA = createF32LeafTensor(3.0f, name = "A")

        // Construct Graph: D = (A+A) + (A*A)
        // B = A + A
        val tensorB = add(context, tensorA, tensorA)
        tensorB.name = "B"
        // C = A * A
        val tensorC = mul(context, tensorA, tensorA) // Assuming mul operator exists and is accessible
        tensorC.name = "C"
        // D = B + C
        val tensorD = add(context, tensorB, tensorC)
        tensorD.name = "D"
        tensorD.isOutput = true

        // Build and Execute
        val cgraph = createGraph(size = 10)
        buildForward(cgraph, tensorD)

        assertTrue(graphAllocator.allocateGraph(cgraph), "Graph allocation failed for shared input test")

        val initialAData = getTensorData(context, tensorA)
        assertEquals(3.0f, initialAData[0], "Initial data for A is incorrect.")

        executeGraph(context, cgraph)

        // Verification
        val resultDData = getTensorData(context, tensorD)
        assertEquals(1, resultDData.size, "Result D should be a scalar.")
        // (3+3) + (3*3) = 6 + 9 = 15.0f
        assertEquals(15.0f, resultDData[0], 0.001f, "Result of D should be 15.0.")

        // Verify intermediate tensors
        val resultBData = getTensorData(context, tensorB)
        assertEquals(6.0f, resultBData[0], 0.001f, "Intermediate B = A+A should be 6.0.")

        val resultCData = getTensorData(context, tensorC)
        assertEquals(9.0f, resultCData[0], 0.001f, "Intermediate C = A*A should be 9.0.")

        // Check memory deallocation for A
        // A is used by B and C. Initial numChildren for A should be 2.
        // After B's job, A.numChildren becomes 1.
        // After C's job, A.numChildren becomes 0. Then A can be freed.
        val usageA = graphAllocator.getTensorUsageInfo(tensorA)
        assertNotNull(usageA, "UsageInfo for A not found")
        assertEquals(0, usageA.numChildren.value, "Tensor A should have 0 children remaining.")
        if (!usageA.isOutputTensor) { // A is not an output
             assertEquals(false, usageA.ownsMemory, "Tensor A's memory should have been marked as not owned if freed.")
        }
    }

    @Test
    fun testMultipleIndependentOperations() {
        val context = createTestContext()
        val graphAllocator = context.graphAllocator!!
        val numOps = 20 // Number of independent operations

        val inputsA = mutableListOf<GGMLTensor>()
        val inputsB = mutableListOf<GGMLTensor>()
        val outputs = mutableListOf<GGMLTensor>()

        // Setup: Create many independent operations
        for (i in 0 until numOps) {
            val a = createF32LeafTensor(i.toFloat(), name = "A$i")
            val b = createF32LeafTensor((i * 2).toFloat(), name = "B$i")
            inputsA.add(a)
            inputsB.add(b)

            val out = add(context, a, b) // out = A[i] + B[i]
            out.name = "Out$i"
            out.isOutput = true // Mark each result as an output to prevent premature deallocation for verification
            outputs.add(out)
        }

        // To build the graph for all these operations, we need a single root or multiple roots.
        // If buildForward handles multiple tensors, that's fine.
        // Otherwise, create a dummy final node that consumes all outputs.
        // For this test, let's assume we can build for a list or use a dummy.
        // Let's create a dummy final tensor that takes all 'outputs' as input.
        // This ensures all operations are part of the graph processed by buildForward.
        // Note: this dummy node might not be strictly necessary if buildForward can take multiple roots
        // or if executeGraph processes all nodes in cgraph.nodes regardless of a single end target.
        // However, many graph runners expect a single output tensor to define the part of the graph to run.
        // Let's assume we need a single "final" tensor for buildForward.

        // Create a dummy "collector" node if needed. For simplicity, let's try building for the list of outputs
        // if the API allows, or focus on buildForward(last_output) and ensure others are connected.
        // The current `buildForward` takes a single tensor. So, we need a dummy.
        // However, `cgraph.nodes` will contain all reachable nodes from this final tensor.
        // If all `outputs[i]` are marked as `isOutput = true` and `allocateGraph` considers this,
        // their memory won't be freed. `executeGraph` executes all nodes in `cgraph.nNodes`.

        // Let's make a dummy sum of all outputs.
        // This might create a very wide node.
        var finalAggregator = outputs[0]
        if (outputs.size > 1) {
            for (i in 1 until outputs.size) {
                finalAggregator = add(context, finalAggregator, outputs[i])
                finalAggregator.name = "FinalAgg$i"
                 // Only the very last aggregation needs to be the graph's explicit output for buildForward
                if (i < outputs.size -1) finalAggregator.isOutput = false
            }
        }
        finalAggregator.isOutput = true


        val cgraph = createGraph(size = numOps * 3 + 5) // Estimate graph size
        buildForward(cgraph, finalAggregator) // Build graph ending with the last output or dummy

        assertTrue(graphAllocator.allocateGraph(cgraph), "Graph allocation failed for multiple ops test")

        // Execute
        executeGraph(context, cgraph)

        // Verification
        for (i in 0 until numOps) {
            val expectedValue = i.toFloat() + (i * 2).toFloat()
            val outputTensor = outputs[i]
            val resultData = getTensorData(context, outputTensor)
            assertEquals(1, resultData.size, "Output $i should be scalar")
            assertEquals(expectedValue, resultData[0], 0.001f, "Incorrect result for Out$i = A$i + B$i")

            val usageA = graphAllocator.getTensorUsageInfo(inputsA[i])
            assertNotNull(usageA)
            assertEquals(0, usageA.numChildren.value, "Input A$i should have 0 children after Out$i is computed.")
            // If A$i was not an output and owned memory, it should be freed.
            if (!usageA.isOutputTensor) { // Leaf inputs are not outputs by default
                 assertEquals(false, usageA.ownsMemory, "Input A$i's memory should be freed.")
            }


            val usageB = graphAllocator.getTensorUsageInfo(inputsB[i])
            assertNotNull(usageB)
            assertEquals(0, usageB.numChildren.value, "Input B$i should have 0 children after Out$i is computed.")
            if (!usageB.isOutputTensor) {
                 assertEquals(false, usageB.ownsMemory, "Input B$i's memory should be freed.")
            }

            val usageOut = graphAllocator.getTensorUsageInfo(outputTensor)
            assertNotNull(usageOut)
            // If `finalAggregator` is used, `outputs[i]` (for i < numOps-1) will have one child (the next aggregation step).
            // `outputs[numOps-1]` will be one of the sources for the last aggregation.
            // All `outputs[i]` were marked `isOutput=true` initially.
            // The dummy aggregation means only `finalAggregator` is the true "end" of the graph for `buildForward`.
            // The `isOutput=true` on `outputs[i]` should prevent their memory from being freed if they are part of `finalAggregator`.
            // Let's simplify: we marked all `outputs[i]` as `isOutput = true`.
            // This test is primarily for concurrency robustness, not complex memory freeing validation of intermediate aggregations.
            assertTrue(usageOut.isOutputTensor, "Output tensor Out$i was marked as output.")
            assertTrue(usageOut.ownsMemory, "Output tensor Out$i should retain its memory.")
        }
    }
}
