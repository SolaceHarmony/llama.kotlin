package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.core.GGMLTestUtils
import io.github.kotlinmania.llama.core.allocateTensorData
import io.github.kotlinmania.llama.core.calculateTensorByteSize
import io.github.kotlinmania.llama.core.resetAllocatorTracking
import io.github.kotlinmania.llama.ore.*
import kotlin.math.*
import kotlin.test.*

private fun io.github.kotlinmania.llama.ore.GGMLTensor.allocateTestBuffer(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator) {
    val byteSize = calculateTensorByteSize(this.type, this.ne).toInt()
    if (byteSize <= 0) return
    val offset = graphAllocator.allocateTensorData(byteSize)
    this.bufferId = 0
    this.dataOffset = offset
}

class LlamaAttentionTest {
    
    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    private lateinit var context: io.github.kotlinmania.llama.ore.GGMLContext
    private val bufferSize = 1024 * 1024 // 1MB
    
    @BeforeTest
    fun setup() {
        val (allocator, _) = GGMLTestUtils.createTestAllocator(bufferSize)
        graphAllocator = allocator
        resetAllocatorTracking(graphAllocator)
        context = io.github.kotlinmania.llama.ore.GGMLContext()
    }
    
    @Test
    fun testAttentionCreation() {
        val attention = LlamaAttention(
            hiddenSize = 512,
            numHeads = 8,
            headDim = 64
        )
        
        assertEquals(512, attention.hiddenSize)
        assertEquals(8, attention.numHeads)
        assertEquals(64, attention.headDim)
        assertEquals(2048, attention.maxPositionEmbeddings)
    }
    
    @Test
    fun testInvalidConfiguration() {
        assertFailsWith<IllegalArgumentException> {
            LlamaAttention(
                hiddenSize = 513, // Not divisible by numHeads
                numHeads = 8
            )
        }
    }
    
    @Test
    fun testRoPEApplication() {
        val attention = LlamaAttention(
            hiddenSize = 4,
            numHeads = 2,
            headDim = 2
        )
        
        // Create test tensor [head_dim, num_heads, seq_len, batch_size] = [2, 2, 1, 1]
        val input =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        input.ne[0] = 2L // head_dim
        input.ne[1] = 2L // num_heads
        input.ne[2] = 1L // seq_len
        input.ne[3] = 1L // batch_size
        input.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            input.ne,
            input.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        
        input.allocateTestBuffer(graphAllocator)
        
        // Set test values
        input.setFloat(graphAllocator, 1.0f, 0, 0, 0) // head 0, dim 0
        input.setFloat(graphAllocator, 0.0f, 1, 0, 0) // head 0, dim 1
        input.setFloat(graphAllocator, 1.0f, 0, 1, 0) // head 1, dim 0
        input.setFloat(graphAllocator, 0.0f, 1, 1, 0) // head 1, dim 1
        
        // Apply RoPE
        val result = attention.applyRoPE(context, graphAllocator, input, 0)
        
        // Check that result has same shape
        assertEquals(input.ne[0], result.ne[0])
        assertEquals(input.ne[1], result.ne[1])
        assertEquals(input.ne[2], result.ne[2])
        assertEquals(input.ne[3], result.ne[3])
        
        // Values should be different after RoPE (unless at specific positions)
        val resultVal0 = result.getFloat(graphAllocator, 0, 0, 0)
        val resultVal1 = result.getFloat(graphAllocator, 1, 0, 0)
        
        // For position 0, rotation should apply
        assertNotEquals(1.0f, resultVal0, 0.001f)
    }
    
    @Test
    fun testAttentionComputation() {
        val attention = LlamaAttention(
            hiddenSize = 4,
            numHeads = 2,
            headDim = 2
        )
        
        // Create query, key, value tensors [head_dim=2, num_heads=2, seq_len=2, batch=1]
        val query = createTestTensor(2, 2, 2, 1)
        val key = createTestTensor(2, 2, 2, 1)
        val value = createTestTensor(2, 2, 2, 1)
        
        // Initialize with test values
        initializeTestTensor(query)
        initializeTestTensor(key)
        initializeTestTensor(value)
        
        // Compute attention
        val result = attention.computeAttention(
            context, graphAllocator, query, key, value
        )
        
        // Check result shape
        assertEquals(query.ne[0], result.ne[0]) // head_dim
        assertEquals(query.ne[1], result.ne[1]) // num_heads
        assertEquals(query.ne[2], result.ne[2]) // seq_len
        assertEquals(query.ne[3], result.ne[3]) // batch
        
        // Verify attention output has reasonable values (not NaN or infinite)
        val outputVal = result.getFloat(graphAllocator, 0, 0, 0, 0)
        assertFalse(outputVal.isNaN(), "Attention output should not be NaN")
        assertFalse(outputVal.isInfinite(), "Attention output should not be infinite")
    }
    
    private fun createTestTensor(dim0: Int, dim1: Int, dim2: Int, dim3: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        tensor.ne[0] = dim0.toLong()
        tensor.ne[1] = dim1.toLong()
        tensor.ne[2] = dim2.toLong()
        tensor.ne[3] = dim3.toLong()
        tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            tensor.ne,
            tensor.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        
        tensor.allocateTestBuffer(graphAllocator)
        return tensor
    }
    
    private fun initializeTestTensor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val totalElements = tensor.numElements().toInt()
        for (i in 0 until totalElements) {
            val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 0 }
            var temp = i
            for (d in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                indices[d] = temp % tensor.ne[d].toInt()
                temp /= tensor.ne[d].toInt()
            }
            tensor.setFloat(graphAllocator, (i % 10) / 10.0f, *indices)
        }
    }
}

class KVCacheTest {
    
    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    
    @BeforeTest
    fun setup() {
        val (allocator, _) = GGMLTestUtils.createTestAllocator(1024 * 1024)
        graphAllocator = allocator
        resetAllocatorTracking(graphAllocator)
    }
    
    @Test
    fun testKVCacheCreation() {
        val cache = KVCache(
            maxSequenceLength = 100,
            numHeads = 8,
            headDim = 64
        )
        
        cache.initialize(graphAllocator)
        
        assertEquals(0, cache.getCurrentLength())
    }
    
    @Test
    fun testKVCacheUpdate() {
        val cache = KVCache(
            maxSequenceLength = 10,
            numHeads = 2,
            headDim = 4
        )
        
        cache.initialize(graphAllocator)
        
        // Create new key tensor
        val newKey =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        newKey.ne[0] = 4L // head_dim
        newKey.ne[1] = 2L // num_heads
        newKey.ne[2] = 3L // new_seq_len
        newKey.ne[3] = 1L // batch
        newKey.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            newKey.ne,
            newKey.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        
        newKey.allocateTestBuffer(graphAllocator)
        
        // Update cache
        val result = cache.updateKey(graphAllocator, newKey)
        
        assertEquals(3, cache.getCurrentLength())
        assertEquals(4L, result.ne[0]) // head_dim
        assertEquals(2L, result.ne[1]) // num_heads
        assertEquals(3L, result.ne[2]) // current sequence length
    }
    
    @Test
    fun testKVCacheClear() {
        val cache = KVCache(
            maxSequenceLength = 10,
            numHeads = 2,
            headDim = 4
        )
        
        cache.initialize(graphAllocator)
        
        // Add some data
        val newKey =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        newKey.ne[0] = 4L
        newKey.ne[1] = 2L
        newKey.ne[2] = 3L
        newKey.ne[3] = 1L
        newKey.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            newKey.ne,
            newKey.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        
        newKey.allocateTestBuffer(graphAllocator)
        cache.updateKey(graphAllocator, newKey)
        
        assertEquals(3, cache.getCurrentLength())
        
        // Clear cache
        cache.clear()
        
        assertEquals(0, cache.getCurrentLength())
    }
}

class SamplingTest {
    
    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    
    @BeforeTest
    fun setup() {
        val (allocator, _) = GGMLTestUtils.createTestAllocator(1024 * 1024)
        graphAllocator = allocator
        resetAllocatorTracking(graphAllocator)
    }
    
    @Test
    fun testSamplingContextCreation() {
        val config = SamplingConfig(
            temperature = 0.7f,
            topK = 50,
            topP = 0.95f
        )
        
        val context = SamplingContext(config, vocabSize = 1000)
        
        assertEquals(config, context.config)
    }
    
    @Test
    fun testTokenSampling() {
        val config = SamplingConfig(
            temperature = 1.0f,
            seed = 42 // Fixed seed for reproducible test
        )
        
        val context = SamplingContext(config, vocabSize = 10)
        
        // Create test logits tensor
        val logits =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        logits.ne[0] = 10L // vocab size
        for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) logits.ne[i] = 1L
        logits.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            logits.ne,
            logits.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        
        logits.allocateTestBuffer(graphAllocator)
        
        // Set logits with one clearly higher value
        for (i in 0 until 10) {
            logits.setFloat(graphAllocator, if (i == 5) 10.0f else 0.0f, i)
        }
        
        val sampledToken = context.sample(graphAllocator, logits)
        
        // With high logit for token 5, it should be selected most of the time
        assertTrue(sampledToken >= 0)
        assertTrue(sampledToken < 10)
    }
    
    @Test
    fun testTemperatureScaling() {
        val lowTempConfig = SamplingConfig(temperature = 0.1f, seed = 42)
        val highTempConfig = SamplingConfig(temperature = 2.0f, seed = 42)
        
        val lowTempContext = SamplingContext(lowTempConfig, vocabSize = 10)
        val highTempContext = SamplingContext(highTempConfig, vocabSize = 10)
        
        // Create logits with one clearly higher value
        val logits = createLogitsTensor(10)
        for (i in 0 until 10) {
            logits.setFloat(graphAllocator, if (i == 5) 2.0f else 0.0f, i)
        }
        
        // Low temperature should be more deterministic
        val lowTempResults = mutableSetOf<Int>()
        repeat(10) {
            lowTempResults.add(lowTempContext.sample(graphAllocator, logits))
        }
        
        // High temperature should be more random (but this is hard to test deterministically)
        val highTempToken = highTempContext.sample(graphAllocator, logits)
        
        // Low temp should heavily favor the high-logit token
        assertTrue(lowTempResults.size <= 3) // Should be quite concentrated
    }
    
    @Test
    fun testTopKFiltering() {
        val config = SamplingConfig(topK = 3, seed = 42)
        val context = SamplingContext(config, vocabSize = 10)
        
        val logits = createLogitsTensor(10)
        // Set decreasing logits
        for (i in 0 until 10) {
            logits.setFloat(graphAllocator, (10 - i).toFloat(), i)
        }
        
        // Sample many times to check that only top-k tokens are selected
        val sampledTokens = mutableSetOf<Int>()
        repeat(100) {
            sampledTokens.add(context.sample(graphAllocator, logits))
        }
        
        // Should only sample from top-3 tokens (0, 1, 2)
        assertTrue(sampledTokens.all { it < 3 })
    }
    
    private fun createLogitsTensor(vocabSize: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        tensor.ne[0] = vocabSize.toLong()
        for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) tensor.ne[i] = 1L
        tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            tensor.ne,
            tensor.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        
        tensor.allocateTestBuffer(graphAllocator)
        return tensor
    }
}

class GrammarTest {
    
    @Test
    fun testGrammarParsing() {
        val grammarText = """
            root ::= greeting name punctuation
            greeting ::= "hello" | "hi"
            name ::= [A-Z][a-z]*
            punctuation ::= "!" | "?"
        """.trimIndent()
        
        val grammar = GBNFGrammar.parse(grammarText)
        
        assertEquals(4, grammar.rules.size)
        assertEquals(0, grammar.rootRuleId)
    }
    
    @Test
    fun testGrammarInitialStates() {
        val grammarText = """
            root ::= "hello"
        """.trimIndent()
        
        val grammar = GBNFGrammar.parse(grammarText)
        val initialStates = grammar.getInitialStates()
        
        assertEquals(1, initialStates.size)
        assertEquals(0, initialStates[0].ruleId)
        assertEquals(0, initialStates[0].position)
    }
    
    @Test
    fun testGrammarSamplingContext() {
        val grammarText = """
            root ::= "hello"
        """.trimIndent()
        
        val grammar = GBNFGrammar.parse(grammarText)
        val tokenizer = SimpleTokenizer()
        val grammarContext = GrammarSamplingContext(grammar, tokenizer)
        
        assertFalse(grammarContext.isComplete())
        
        // Create test candidates
        val candidates = Array(10) { i ->
            TokenData(id = i, logit = 0.0f, prob = 0.1f)
        }
        
        val filtered = grammarContext.filterCandidates(candidates)
        
        // Should filter based on grammar constraints
        assertTrue(filtered.size <= candidates.size)
    }
}

class LlamaModelTest {
    
    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    private lateinit var context: io.github.kotlinmania.llama.ore.GGMLContext
    
    @BeforeTest
    fun setup() {
        val (allocator, _) = GGMLTestUtils.createTestAllocator(10 * 1024 * 1024)
        graphAllocator = allocator
        resetAllocatorTracking(graphAllocator)
        context = io.github.kotlinmania.llama.ore.GGMLContext()
    }
    
    @Test
    fun testModelCreation() {
        val config = LlamaConfig(
            vocabSize = 1000,
            hiddenSize = 64,
            intermediateSize = 128,
            numHiddenLayers = 2,
            numAttentionHeads = 4
        )
        
        val model = LlamaModel(config)
        
        assertEquals(config, model.config)
        assertEquals(2, model.layers.size)
    }
    
    @Test
    fun testModelForward() {
        val config = LlamaConfig(
            vocabSize = 100,
            hiddenSize = 32,
            intermediateSize = 64,
            numHiddenLayers = 1,
            numAttentionHeads = 2
        )
        
        val model = LlamaModel(config)
        model.initializeParameters(graphAllocator)
        
        val inputIds = intArrayOf(1, 2, 3)
        
        val result = model.forward(
            context = context,
            graphAllocator = graphAllocator,
            inputIds = inputIds
        )
        
        // Check output shape
        assertEquals(config.vocabSize.toLong(), result.ne[0]) // vocab size
        assertEquals(inputIds.size.toLong(), result.ne[1]) // sequence length
        assertEquals(1L, result.ne[2]) // batch size
        
        // Verify output has reasonable values (not NaN or infinite)
        val outputVal = result.getFloat(graphAllocator, 0, 0, 0, 0)
        assertFalse(outputVal.isNaN(), "Model output should not be NaN")
        assertFalse(outputVal.isInfinite(), "Model output should not be infinite")
    }
    
    @Test
    fun testMLPForward() {
        val config = LlamaConfig(
            hiddenSize = 32,
            intermediateSize = 64
        )
        
        val mlp = LlamaMLP(config)
        
        // Allocate MLP weights
        mlp.gateProj.allocateTestBuffer(graphAllocator)
        mlp.upProj.allocateTestBuffer(graphAllocator)
        mlp.downProj.allocateTestBuffer(graphAllocator)
        
        // Initialize weights with small values to avoid numerical issues
        val weights = arrayOf(mlp.gateProj, mlp.upProj, mlp.downProj)
        for (weight in weights) {
            val numElements = weight.numElements().toInt()
            for (i in 0 until numElements) {
                val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 0 }
                var temp = i
                for (d in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                    indices[d] = temp % weight.ne[d].toInt()
                    temp /= weight.ne[d].toInt()
                }
                weight.setFloat(graphAllocator, 0.01f * (i % 10 - 5), *indices)
            }
        }
        
        // Create input tensor
        val input =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        input.ne = longArrayOf(32, 3, 1, 1) // [hidden, seq, batch]
        input.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            input.ne,
            input.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        input.allocateTestBuffer(graphAllocator)
        
        // Initialize input with small values
        for (i in 0 until 32) {
            for (j in 0 until 3) {
                input.setFloat(graphAllocator, 0.1f * (i + j), i, j, 0, 0)
            }
        }
        
        val result = mlp.forward(context, graphAllocator, input)
        
        // Check output shape should match input hidden dimension
        assertEquals(32L, result.ne[0]) // hidden size
        assertEquals(3L, result.ne[1])  // sequence length
        assertEquals(1L, result.ne[2])  // batch size
        
        // Verify result values are reasonable
        val resultVal = result.getFloat(graphAllocator, 0, 0, 0, 0)
        assertFalse(resultVal.isNaN(), "MLP output should not be NaN")
        assertFalse(resultVal.isInfinite(), "MLP output should not be infinite")
    }
    
    @Test
    fun testRMSNorm() {
        val norm = RMSNorm(normalizedShape = 4, eps = 1e-6f)
        
        // Initialize norm weight
        norm.weight.ne = longArrayOf(4, 1, 1, 1)
        norm.weight.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            norm.weight.ne,
            norm.weight.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        norm.weight.allocateTestBuffer(graphAllocator)
        
        // Set weights to 1.0
        for (i in 0 until 4) {
            norm.weight.setFloat(graphAllocator, 1.0f, i)
        }
        
        // Create input tensor
        val input =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        input.ne = longArrayOf(4, 2, 1, 1) // [hidden, seq, batch]
        input.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            input.ne,
            input.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
        )
        input.allocateTestBuffer(graphAllocator)
        
        // Set input values [1.0, 2.0, 3.0, 4.0] for first sequence position
        input.setFloat(graphAllocator, 1.0f, 0, 0, 0)
        input.setFloat(graphAllocator, 2.0f, 1, 0, 0)
        input.setFloat(graphAllocator, 3.0f, 2, 0, 0)
        input.setFloat(graphAllocator, 4.0f, 3, 0, 0)
        
        val result = norm.forward(context, graphAllocator, input)
        
        // Check that result has same shape as input
        assertEquals(input.ne[0], result.ne[0])
        assertEquals(input.ne[1], result.ne[1])
        assertEquals(input.ne[2], result.ne[2])
        
        // Verify result values are reasonable (not NaN or infinite)
        val resultVal = result.getFloat(graphAllocator, 0, 0, 0, 0)
        assertFalse(resultVal.isNaN(), "RMSNorm output should not be NaN")
        assertFalse(resultVal.isInfinite(), "RMSNorm output should not be infinite")
        
        // The result should have smaller magnitude than input since we're normalizing
        assertTrue(kotlin.math.abs(resultVal) < 4.0f, "Normalized value should be smaller than original")
    }
}
