package ai.solace.llamakotlin.core

import kotlin.test.*

// Assuming GGML_MAX_DIMS and GGMLType are accessible from this test package
// For example, by being in the same module or via appropriate imports.
// internal const val GGML_MAX_DIMS = 4 // Define if not accessible otherwise for tests

class GGMLAllocTest {

    private fun createDummyTensor(name: String = "dummy", type: GGMLType = GGMLType.F32): GGMLTensor {
        val tensor = GGMLTensor(type = type) // ne defaults to [0,0,0,0]
        tensor.name = name
        // For tests, we mainly care about size, not ne/nb structure unless specifically testing views.
        // The allocator uses the size passed to allocate(), not from tensor.ne/nb directly.
        // However, for completeness if any part of allocator might inspect ne/nb:
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L } // Treat as scalar for simplicity of ne/nb
        if (type.byteSize > 0u) {
            tensor.nb[0] = type.byteSize
            for (i in 1 until GGML_MAX_DIMS) {
                tensor.nb[i] = tensor.nb[i-1] * tensor.ne[i-1].toULong() // Will be type.byteSize for all nb if ne is all 1s
            }
        }
        return tensor
    }

    private val dummyTensorF32 = createDummyTensor("tF32", GGMLType.F32) // size 4
    // private val dummyTensorI16 = createDummyTensor("tI16", GGMLType.I16) // size 2 // Not used in current tests explicitly

    // --- GGMLDynTensorAllocator Tests End ---

    // --- GGMLGraphAllocator Tests Start ---

    private fun setupTensorForGraph(
        name: String,
        type: GGMLType,
        dims: LongArray,
        op: GGMLOp = GGMLOp.NONE,
        isOutput: Boolean = false
    ): GGMLTensor {
        val tensor = GGMLTensor(type = type)
        tensor.name = name
        tensor.op = op
        if (isOutput) {
            tensor.flags = tensor.flags or GGML_TENSOR_FLAG_OUTPUT
        }

        // Ensure dims is GGML_MAX_DIMS long
        if (dims.size < GGML_MAX_DIMS) {
            tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
            dims.copyInto(tensor.ne, 0, 0, dims.size)
        } else if (dims.size == GGML_MAX_DIMS) {
            tensor.ne = dims.copyOf()
        } else {
            throw IllegalArgumentException("Dimensions array size ${dims.size} exceeds GGML_MAX_DIMS $GGML_MAX_DIMS")
        }

        // Simplified stride calculation for test setup
        if (type.byteSize > 0u) {
            tensor.nb[0] = type.byteSize
            for (i in 1 until GGML_MAX_DIMS) {
                tensor.nb[i] = tensor.nb[i-1] * (if (tensor.ne[i-1] > 0) tensor.ne[i-1].toULong() else 1uL)
            }
        } else { // For types with 0 byteSize (like quantized), set nb to 0 or based on some convention if needed for tests
            tensor.nb.fill(0uL)
        }
        tensor.data = null // Ensure data is null for graph allocation tests
        return tensor
    }


    // Helper to calculate the padded size of a requested allocation based on GGMLDynTensorAllocator's logic
    private fun calculatePaddedSize(requestedSize: ULong, alignment: UInt = 16u): ULong {
        // Copied from alignedOffset in GGMLAlloc.kt, but ensuring it matches the allocator's internal padding logic
        // which is typically just aligning the offset, and the block size becomes aligned implicitly.
        // The allocator allocates a block starting at an aligned offset. The size of this block is effectively padded.
        // No, the allocator takes the requested size, aligns IT, then finds a block.
        // So, calculatePaddedSize should be alignedOffset(requestedSize, alignment) from the perspective of block size needed.
        // Let's re-verify GGMLDynTensorAllocator.allocate internal logic for size padding.
        // It does: `val alignedSize = alignedOffset(size, alignment)`
        return alignedOffset(requestedSize, alignment.toUInt()) // Use the actual alignedOffset function
    }

    // Actual alignedOffset function as defined in GGMLAlloc.kt for testing purposes
    private fun alignedOffset(offset: ULong, alignment: UInt): ULong {
        assert(alignment > 0u && (alignment and (alignment - 1u)) == 0u)
        val align = (alignment - (offset % alignment)) % alignment
        return offset + align
    }


    @Test
    fun testInitialState() {
        val allocator = GGMLDynTensorAllocator(bufferSize = 1024uL)
        assertEquals(0uL, allocator.getMaxSize(), "Initial maxSize should be 0")
        assertEquals(1, allocator.freeBlocks.size, "Should have one initial free block")
        assertEquals(0uL, allocator.freeBlocks[0].offset, "Initial free block offset should be 0")
        assertEquals(1024uL, allocator.freeBlocks[0].size, "Initial free block size should match bufferSize")
    }

    @Test
    fun testSingleAllocationAndMaxSize() {
        val allocator = GGMLDynTensorAllocator(bufferSize = 1024uL, alignment = 16u)
        val size1 = 100uL
        val paddedSize1 = calculatePaddedSize(size1, 16u) // 112uL

        val offset1 = allocator.allocate(size1, dummyTensorF32)
        assertEquals(0uL, offset1, "Offset of first allocation should be 0")
        assertEquals(paddedSize1, allocator.getMaxSize(), "maxSize after first allocation")

        val size2 = 200uL
        val paddedSize2 = calculatePaddedSize(size2, 16u) // 208uL
        val offset2 = allocator.allocate(size2, dummyTensorF32)
        // Offset of second should be at offset of first (0) + paddedSize of first
        assertEquals(paddedSize1, offset2, "Offset of second allocation should be after first padded block")
        assertEquals(paddedSize1 + paddedSize2, allocator.getMaxSize(), "maxSize after second allocation")
    }

    @Test
    fun testAllocationExceedsBuffer() {
        val allocator = GGMLDynTensorAllocator(bufferSize = 100uL, alignment = 16u) // Buffer 100
        val size1 = 90uL
        val paddedSize1 = calculatePaddedSize(size1, 16u) // 96uL
        allocator.allocate(size1, dummyTensorF32) // Allocates 0-95. MaxSize = 96. Free block: offset 96, size 4

        assertFailsWith<IllegalStateException>("Should fail if allocation exceeds buffer size") {
            // Requesting 10uL, padded is 16uL. The remaining free block (size 4) is too small.
            allocator.allocate(10uL, dummyTensorF32)
        }
    }

    @Test
    fun testAllocationExceedsBufferImmediately() {
        val allocator = GGMLDynTensorAllocator(bufferSize = 100uL)
        assertFailsWith<IllegalStateException>("Should fail if allocation exceeds buffer size") {
            allocator.allocate(120uL, dummyTensorF32)
        }
    }

    @Test
    fun testFreeAndReuseExactSize() {
        val allocator = GGMLDynTensorAllocator(bufferSize = 1024uL, alignment = 16u)
        val size1 = 100uL
        val paddedSize1 = calculatePaddedSize(size1, 16u)
        val size2 = 200uL
        val paddedSize2 = calculatePaddedSize(size2, 16u)

        val offset1 = allocator.allocate(size1, dummyTensorF32)
        val offset2 = allocator.allocate(size2, dummyTensorF32)

        // Free the second block
        allocator.freeTensor(offset2, size2, dummyTensorF32)

        // Allocate same size as freed block
        val offset3 = allocator.allocate(size2, createDummyTensor("t3"))
        assertEquals(offset2, offset3, "Should reuse the exactly sized freed block")
        assertEquals(paddedSize1 + paddedSize2, allocator.getMaxSize(), "MaxSize should remain the same")
    }

    @Test
    fun testFreeAndReuseSmallerThanFreed() {
        val allocator = GGMLDynTensorAllocator(bufferSize = 1024uL, alignment = 16u)
        val size1 = 100uL
        val paddedSize1 = calculatePaddedSize(size1, 16u)
        val sizeToFree = 200uL
        val paddedSizeToFree = calculatePaddedSize(sizeToFree, 16u)

        val smallerSizeAlloc = 50uL
        val paddedSmallerSizeAlloc = calculatePaddedSize(smallerSizeAlloc, 16u)

        val offset1 = allocator.allocate(size1, dummyTensorF32)
        val offsetToFree = allocator.allocate(sizeToFree, dummyTensorF32)

        allocator.freeTensor(offsetToFree, sizeToFree, dummyTensorF32)

        val offsetSmaller = allocator.allocate(smallerSizeAlloc, createDummyTensor("ts1"))
        assertEquals(offsetToFree, offsetSmaller, "Should use start of the freed block")

        // Check that the remaining part of the freed block is correct
        // Expected: one block from 0 to paddedSize1-1 (used)
        //           one block from offsetToFree to offsetToFree+paddedSmallerSizeAlloc-1 (used by ts1)
        //           one free block from offsetToFree+paddedSmallerSizeAlloc to offsetToFree+paddedSizeToFree-1
        //           one free block at the end of the buffer
        val expectedRemainingFreeBlockOffset = offsetToFree + paddedSmallerSizeAlloc
        val expectedRemainingFreeBlockSize = paddedSizeToFree - paddedSmallerSizeAlloc

        val foundRemainingBlock = allocator.freeBlocks.find { it.offset == expectedRemainingFreeBlockOffset && it.size == expectedRemainingFreeBlockSize }
        assertNotNull(foundRemainingBlock, "Should find a free block representing the remainder of the split block.")

        assertEquals(paddedSize1 + paddedSizeToFree, allocator.getMaxSize(), "MaxSize should not change")
    }

    @Test
    fun testFreeAndMergeAdjacentBlocks() {
        // Use alignment = 1u to make sizes exact and predictable without padding interference for this test
        val allocator = GGMLDynTensorAllocator(alignment = 1u, bufferSize = 1024uL)
        val s = 100uL
        val t1 = createDummyTensor("t1"); val t2 = createDummyTensor("t2"); val t3 = createDummyTensor("t3")

        val off1 = allocator.allocate(s, t1) // 0-99
        assertEquals(0uL, off1)
        val off2 = allocator.allocate(s, t2) // 100-199
        assertEquals(100uL, off2)
        allocator.allocate(s, t3)      // 200-299 // t_placeholder

        assertEquals(1, allocator.freeBlocks.size, "One free block should remain at the end initially")
        assertEquals(300uL, allocator.getMaxSize())

        allocator.freeTensor(off1, s, t1) // Free [0-99]. Free blocks: {[0,100], [300,724]}
        assertEquals(2, allocator.freeBlocks.size)
        assertTrue(allocator.freeBlocks.any { it.offset == 0uL && it.size == 100uL })


        allocator.freeTensor(off2, s, t2) // Free [100-199]. Should merge with [0-99].
                                          // Free blocks: {[0,200], [300,724]}
        assertEquals(2, allocator.freeBlocks.size, "Freeing adjacent block should merge blocks.")
        assertTrue(allocator.freeBlocks.any { it.offset == 0uL && it.size == 200uL }, "Merged block [0,200] not found.")

        val mergedAllocOffset = allocator.allocate(150uL, createDummyTensor("merged"))
        assertEquals(0uL, mergedAllocOffset, "Should use the merged block at offset 0")
        // After allocation: Used [0-149]. Free blocks: {[150,50], [300,724]}
        assertEquals(2, allocator.freeBlocks.size)
        assertTrue(allocator.freeBlocks.any { it.offset == 150uL && it.size == 50uL })
    }

    @Test
    fun testFreeMergeMoreComplex() {
        val allocator = GGMLDynTensorAllocator(alignment = 1u, bufferSize = 500uL)
        val s = 100uL
        val t1 = createDummyTensor("t1"); val t2 = createDummyTensor("t2");
        val t3 = createDummyTensor("t3"); val t4 = createDummyTensor("t4")
        val t5 = createDummyTensor("t5")

        val off1 = allocator.allocate(s, t1) // 0-99
        val off2 = allocator.allocate(s, t2) // 100-199
        val off3 = allocator.allocate(s, t3) // 200-299
        allocator.allocate(s, t4)      // 300-399. MaxSize = 400. Free: [400, 100]

        allocator.freeTensor(off2, s, t2) // Free [100-199]. Free: {[100,100], [400,100]}
        allocator.freeTensor(off1, s, t1) // Free [0-99]. Merges with [100-199]. Free: {[0,200], [400,100]}

        assertEquals(2, allocator.freeBlocks.size)
        assertTrue(allocator.freeBlocks.any { it.offset == 0uL && it.size == 200uL })

        val off5 = allocator.allocate(150uL, t5) // Uses [0-149] from [0,200].
        assertEquals(0uL, off5, "t5 should be allocated at offset 0 from merged block")
        // Free blocks: {[150,50], [400,100]}
        assertEquals(2, allocator.freeBlocks.size)
        assertTrue(allocator.freeBlocks.any { it.offset == 150uL && it.size == 50uL })


        allocator.freeTensor(off3,s,t3) // Free [200-299].
                                        // Free blocks: {[150,50], [200,100], [400,100]}.
                                        // Block [150,50] and [200,100] should merge.
                                        // Expected merge: if block B is freed and B.offset == A.offset + A.size, A absorbs B.
                                        // If block B is freed and A.offset == B.offset + B.size, B absorbs A.
                                        // Here, freeing [200,100]. Previous block is [150,50]. Not adjacent by start.
                                        // Next block is [400,100]. Not adjacent by end.
                                        // It seems my merge logic understanding for this case might be off or the implementation detail matters.
                                        // The code tries to merge with previous and next blocks in the list if they become adjacent.
                                        // After freeing [200,100], it should be inserted.
                                        // Freeing [200,100]. Free list: {[150,50], [400,100]}
                                        // Insert [200,100] -> list: {[150,50], [200,100], [400,100]} (sorted by offset)
                                        // Check if [200,100] merges with [150,50] (no: 150+50 != 200)
                                        // Check if [200,100] merges with [400,100] (no: 200+100 != 400)
                                        // So, 3 free blocks expected.
        assertEquals(3, allocator.freeBlocks.size, "Should have 3 distinct free blocks after freeing t3.")
        assertTrue(allocator.freeBlocks.any { it.offset == 200uL && it.size == 100uL })

        // Now, if we free the block at 150, it *should* merge with 200.
        // To test this, let's allocate something in [400,100] to keep it separate.
        allocator.allocate(50uL, createDummyTensor("t_filler_end")) // uses [400-449]
        // Free blocks: {[150,50], [200,100], [450,50]}

        // To free the block at 150, we need a tensor that was allocated there.
        // We don't have one directly. Let's re-evaluate the test logic slightly.
        // The goal is to see if freeing a block makes it merge with a subsequent block.
        // Current state: Free: {[150,50], [200,100], [450,50]}
        // Let's try to allocate 150uL, it should take [150,50] and then [200,100] -> split [200,100]

        // Simpler test for merge with next:
        allocator.reset(500uL)
        val oA = allocator.allocate(100uL, createDummyTensor("A")) // 0-99
        val oB = allocator.allocate(100uL, createDummyTensor("B")) // 100-199
        val oC = allocator.allocate(100uL, createDummyTensor("C")) // 200-299
        allocator.freeTensor(oA, 100uL, createDummyTensor("A")) // Free [0-99]
        allocator.freeTensor(oC, 100uL, createDummyTensor("C")) // Free [200-299]
        // Free blocks are now: [0,100], [200,100], [300,200]
        assertEquals(3, allocator.freeBlocks.size)
        allocator.freeTensor(oB, 100uL, createDummyTensor("B")) // Free [100-199]
        // Should merge all three into [0,300]. Then [300,200] from end. Total 2 blocks.
        // Freeing [100,100]:
        // 1. Merges with [0,100] -> new block [0,200]
        // 2. Then this new block [0,200] merges with [200,100] -> final [0,300]
        assertEquals(2, allocator.freeBlocks.size, "All three adjacent freed blocks should merge.")
        assertTrue(allocator.freeBlocks.any { it.offset == 0uL && it.size == 300uL })

    }


    @Test
    fun testResetAllocator() {
        val allocator = GGMLDynTensorAllocator(bufferSize = 1024uL, alignment = 16u)
        allocator.allocate(100uL, dummyTensorF32)

        val newBufferSize = 512uL
        allocator.reset(newBufferSize)

        assertEquals(0uL, allocator.getMaxSize(), "maxSize should be 0 after reset")
        assertEquals(1, allocator.freeBlocks.size, "Should have one free block after reset")
        assertEquals(newBufferSize, allocator.freeBlocks[0].size, "Free block size should be newBufferSize")

        val sizeAfterReset = 50uL
        val paddedSizeAfterReset = calculatePaddedSize(sizeAfterReset, 16u)
        val offsetAfterReset = allocator.allocate(sizeAfterReset, dummyTensorF32)
        assertEquals(0uL, offsetAfterReset, "First allocation after reset should start at 0")
        assertEquals(paddedSizeAfterReset, allocator.getMaxSize(), "maxSize after allocation post-reset")

        assertFailsWith<IllegalStateException>("Should fail if allocation exceeds new buffer size after reset") {
            // Padded size of 449uL (requested) with 16u alignment is 464uL.
            // 512 (buffer) - 64 (used by 50uL alloc) = 448 remaining. 464 > 448.
            allocator.allocate(449uL, createDummyTensor("t_exceed"))
        }
    }

    @Test
    fun testReserveGraphSimple() {
        val graphAllocator = GGMLGraphAllocator()
        val graph = GGMLCGraph()

        val tensorA = setupTensorForGraph("A", GGMLType.F32, longArrayOf(10))    // 10*4 = 40 bytes
        val tensorB = setupTensorForGraph("B", GGMLType.I16, longArrayOf(5, 2)) // 5*2*2 = 20 bytes
        val tensorC = setupTensorForGraph("C", GGMLType.I8, longArrayOf(3))     // 3*1 = 3 bytes

        graph.leafs = arrayOf(tensorA, tensorB, tensorC)
        graph.nLeafs = 3
        graph.nodes = arrayOf(tensorA, tensorB, tensorC) // For simple reserve, nodes can be same as leafs
        graph.nNodes = 3

        // Default alignment is 16u
        val paddedSizeA = calculatePaddedSize(40uL, 16u) // 48
        val paddedSizeB = calculatePaddedSize(20uL, 16u) // 32
        val paddedSizeC = calculatePaddedSize(3uL, 16u)  // 16
        val expectedTotalSize = paddedSizeA + paddedSizeB + paddedSizeC // 48 + 32 + 16 = 96

        graphAllocator.reserveGraph(graph)

        // getBufferSize(0) gets the maxSize from the dynamic allocator, which includes padding for each allocation
        assertEquals(expectedTotalSize, graphAllocator.getBufferSize(0), "Buffer size after reserveGraph not as expected")
    }

    @Test
    fun testAllocateGraphBasic_TwoInputs_OneAddOutput() {
        val graphAllocator = GGMLGraphAllocator()
        val graph = GGMLCGraph()

        val tensorA = setupTensorForGraph("A", GGMLType.F32, longArrayOf(10), op = GGMLOp.NONE)  // 40 bytes
        val tensorB = setupTensorForGraph("B", GGMLType.F32, longArrayOf(10), op = GGMLOp.NONE)  // 40 bytes
        val tensorC = setupTensorForGraph("C", GGMLType.F32, longArrayOf(10), op = GGMLOp.ADD, isOutput = true) // 40 bytes

        tensorC.src[0] = tensorA
        tensorC.src[1] = tensorB

        graph.leafs = arrayOf(tensorA, tensorB)
        graph.nLeafs = 2
        graph.nodes = arrayOf(tensorA, tensorB, tensorC) // Order: A, B, then C
        graph.nNodes = 3

        graphAllocator.allocateGraph(graph)

        val alignment = 16u // Default alignment in GGMLDynTensorAllocator
        val paddedSizeA = calculatePaddedSize(40uL, alignment) // 48
        val paddedSizeB = calculatePaddedSize(40uL, alignment) // 48
        val paddedSizeC = calculatePaddedSize(40uL, alignment) // 48


        assertNotNull(graphAllocator.tensorUsageMap[tensorA])
        assertNotNull(graphAllocator.tensorUsageMap[tensorB])
        assertNotNull(graphAllocator.tensorUsageMap[tensorC])

        assertEquals(0, tensorA.bufferId, "Tensor A bufferId")
        assertEquals(0uL, tensorA.dataOffset, "Tensor A dataOffset")

        assertEquals(0, tensorB.bufferId, "Tensor B bufferId")
        assertEquals(paddedSizeA, tensorB.dataOffset, "Tensor B dataOffset")

        assertEquals(0, tensorC.bufferId, "Tensor C bufferId")
        assertEquals(paddedSizeA + paddedSizeB, tensorC.dataOffset, "Tensor C dataOffset")
        assertTrue(graphAllocator.tensorUsageMap[tensorC]!!.isOutputTensor, "Tensor C should be output")
        assertTrue(graphAllocator.tensorUsageMap[tensorC]!!.ownsMemory, "Tensor C should own its memory")


        val expectedTotalMaxSize = paddedSizeA + paddedSizeB + paddedSizeC
        assertEquals(expectedTotalMaxSize, graphAllocator.tensorAllocators[0].getMaxSize(), "Allocator max size not as expected")
    }


    @Test
    fun testAllocateGraphInplace() {
        val graphAllocator = GGMLGraphAllocator()
        val graph = GGMLCGraph()

        val tensorInput = setupTensorForGraph("INPUT", GGMLType.F32, longArrayOf(10)) // 40 bytes
        val tensorReluOut = setupTensorForGraph("RELU_OUT", GGMLType.F32, longArrayOf(10), op = GGMLOp.RELU, isOutput = true)

        tensorReluOut.src[0] = tensorInput

        graph.leafs = arrayOf(tensorInput)
        graph.nLeafs = 1
        // Order: INPUT, then RELU_OUT. This is important for inplace and freeing logic.
        graph.nodes = arrayOf(tensorInput, tensorReluOut)
        graph.nNodes = 2

        // Ensure RELU is inplace for this test
        assertTrue(GGMLOp.RELU.canBeInplace, "RELU op should be marked as canBeInplace for this test to be valid")

        graphAllocator.allocateGraph(graph)

        assertNotNull(graphAllocator.tensorUsageMap[tensorInput])
        assertNotNull(graphAllocator.tensorUsageMap[tensorReluOut])

        assertEquals(0, tensorInput.bufferId, "INPUT bufferId")
        assertEquals(0uL, tensorInput.dataOffset, "INPUT dataOffset")

        // RELU_OUT should reuse INPUT's memory
        assertEquals(tensorInput.bufferId, tensorReluOut.bufferId, "RELU_OUT should reuse INPUT's bufferId")
        assertEquals(tensorInput.dataOffset, tensorReluOut.dataOffset, "RELU_OUT should reuse INPUT's dataOffset")

        assertTrue(graphAllocator.tensorUsageMap[tensorReluOut]!!.ownsMemory, "RELU_OUT should own the memory segment")
        assertFalse(graphAllocator.tensorUsageMap[tensorInput]!!.ownsMemory, "INPUT should have relinquished memory ownership")
        assertTrue(graphAllocator.tensorUsageMap[tensorReluOut]!!.isOutputTensor, "RELU_OUT is an output tensor")

        val paddedSizeInput = calculatePaddedSize(40uL, 16u) // 48
        assertEquals(paddedSizeInput, graphAllocator.tensorAllocators[0].getMaxSize(), "Allocator max size should be size of one tensor")
    }

    @Test
    fun testAllocateGraphMemoryFreeing() {
        val graphAllocator = GGMLGraphAllocator()
        val graph = GGMLCGraph()
        val alignment = 16u // Default

        // Setup tensors
        val tensorA = setupTensorForGraph("A", GGMLType.F32, longArrayOf(10)) // 40 bytes, padded 48
        val tensorB = setupTensorForGraph("B", GGMLType.F32, longArrayOf(10)) // 40 bytes, padded 48
        // Ensure ADD is NOT inplace for this test to make TEMP distinct.
        // We can achieve this by making its output different or ensuring conditions aren't met,
        // or by temporarily overriding canBeInplace (not clean for test).
        // For this test, we assume ADD is not inplace or its conditions for inplace are not met.
        // (e.g. if we were to add another child to A or B before TEMP is computed)
        val tensorTemp = setupTensorForGraph("TEMP", GGMLType.F32, longArrayOf(10), op = GGMLOp.ADD)
        val tensorOutput = setupTensorForGraph("OUTPUT", GGMLType.F32, longArrayOf(10), op = GGMLOp.RELU, isOutput = true)

        tensorTemp.src[0] = tensorA
        tensorTemp.src[1] = tensorB
        tensorOutput.src[0] = tensorTemp

        graph.leafs = arrayOf(tensorA, tensorB)
        graph.nLeafs = 2
        // Order: A, B, TEMP, OUTPUT
        graph.nodes = arrayOf(tensorA, tensorB, tensorTemp, tensorOutput)
        graph.nNodes = 4

        val originalAddCanBeInplace = GGMLOp.ADD.canBeInplace
        val originalReluCanBeInplace = GGMLOp.RELU.canBeInplace

        // Force ADD to not be inplace for this test if it was marked as such
        // This is a bit hacky for a test; ideally, the graph structure would ensure this.
        // Forcing by temporarily changing enum state is not robust if tests run in parallel or state leaks.
        // A better way is to ensure inplace conditions for ADD are not met (e.g. A or B has other children).
        // For simplicity here, we rely on default ADD behavior or assume it won't be inplace.
        // And ensure RELU is inplace.
        if (!GGMLOp.RELU.canBeInplace) {
            // This would be a test setup issue, not a library issue.
            // For this test, we'll assume it's true or skip.
            println("Warning: RELU not marked as canBeInplace, testAllocateGraphMemoryFreeing might not be fully valid.")
        }


        graphAllocator.allocateGraph(graph)

        val paddedSizeA = calculatePaddedSize(40uL, alignment) // 48
        val paddedSizeB = calculatePaddedSize(40uL, alignment) // 48
        val paddedSizeTemp = calculatePaddedSize(40uL, alignment) // 48

        assertEquals(0uL, tensorA.dataOffset, "Tensor A offset")
        assertEquals(paddedSizeA, tensorB.dataOffset, "Tensor B offset")
        assertEquals(paddedSizeA + paddedSizeB, tensorTemp.dataOffset, "Tensor TEMP offset")

        // OUTPUT (RELU) should be inplace on TEMP
        assertEquals(tensorTemp.dataOffset, tensorOutput.dataOffset, "Tensor OUTPUT should reuse TEMP's offset")

        // Max size should be up to where TEMP/OUTPUT is.
        // A (0-47), B (48-95), TEMP/OUTPUT (96-143)
        assertEquals(paddedSizeA + paddedSizeB + paddedSizeTemp, graphAllocator.tensorAllocators[0].getMaxSize(), "Max size after graph allocation")

        // Check if memory for A and B was freed (their numChildren would be 0)
        // This is implicitly tested by trying to reallocate.
        // The allocator's free list should now contain blocks for A and B.

        val dynAlloc = graphAllocator.tensorAllocators[0]

        // Try to allocate new tensors of same size as A and B
        val newTensorX = setupTensorForGraph("X", GGMLType.F32, longArrayOf(10))
        val newTensorY = setupTensorForGraph("Y", GGMLType.F32, longArrayOf(10))

        val offsetX = dynAlloc.allocate(calculateTensorSize(newTensorX), newTensorX)
        // First free block should be A's old space
        assertEquals(tensorA.dataOffset, offsetX, "New tensor X should reuse A's freed memory")

        val offsetY = dynAlloc.allocate(calculateTensorSize(newTensorY), newTensorY)
        // Next free block should be B's old space
        assertEquals(tensorB.dataOffset, offsetY, "New tensor Y should reuse B's freed memory")
    }
}
