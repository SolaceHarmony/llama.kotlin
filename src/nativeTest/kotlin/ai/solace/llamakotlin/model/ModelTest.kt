package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.test.*
import kotlin.math.*

class LlamaAttentionTest {
    
    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var context: GGMLContext
    private val bufferSize = 1024 * 1024 // 1MB
    
    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        graphAllocator.reserve(bufferSize.toLong())
        context = GGMLContext()
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
        val input = GGMLTensor(type = GGMLType.F32)
        input.ne[0] = 2L // head_dim
        input.ne[1] = 2L // num_heads
        input.ne[2] = 1L // seq_len
        input.ne[3] = 1L // batch_size
        input.nb = calculateContiguousStrides(input.ne, input.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(input)
        
        // Set test values
        input.setFloat(graphAllocator, 0, 0, 0, 1.0f) // head 0, dim 0
        input.setFloat(graphAllocator, 1, 0, 0, 0.0f) // head 0, dim 1
        input.setFloat(graphAllocator, 0, 1, 0, 1.0f) // head 1, dim 0
        input.setFloat(graphAllocator, 1, 1, 0, 0.0f) // head 1, dim 1
        
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
        
        // Create query, key, value tensors
        val query = createTestTensor(2, 2, 2, 1) // [head_dim, num_heads, seq_len, batch]
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
    }
    
    private fun createTestTensor(dim0: Int, dim1: Int, dim2: Int, dim3: Int): GGMLTensor {
        val tensor = GGMLTensor(type = GGMLType.F32)
        tensor.ne[0] = dim0.toLong()
        tensor.ne[1] = dim1.toLong()
        tensor.ne[2] = dim2.toLong()
        tensor.ne[3] = dim3.toLong()
        tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(tensor)
        return tensor
    }
    
    private fun initializeTestTensor(tensor: GGMLTensor) {
        val totalElements = tensor.numElements().toInt()
        for (i in 0 until totalElements) {
            val indices = IntArray(GGML_MAX_DIMS) { 0 }
            var temp = i
            for (d in 0 until GGML_MAX_DIMS) {
                indices[d] = temp % tensor.ne[d].toInt()
                temp /= tensor.ne[d].toInt()
            }
            tensor.setFloat(graphAllocator, *indices, (i % 10) / 10.0f)
        }
    }
}

class KVCacheTest {
    
    private lateinit var graphAllocator: GGMLGraphAllocator
    
    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        graphAllocator.reserve(1024 * 1024)
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
        val newKey = GGMLTensor(type = GGMLType.F32)
        newKey.ne[0] = 4L // head_dim
        newKey.ne[1] = 2L // num_heads
        newKey.ne[2] = 3L // new_seq_len
        newKey.ne[3] = 1L // batch
        newKey.nb = calculateContiguousStrides(newKey.ne, newKey.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(newKey)
        
        // Update cache
        val result = cache.updateKey(newKey)
        
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
        val newKey = GGMLTensor(type = GGMLType.F32)
        newKey.ne[0] = 4L
        newKey.ne[1] = 2L
        newKey.ne[2] = 3L
        newKey.ne[3] = 1L
        newKey.nb = calculateContiguousStrides(newKey.ne, newKey.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(newKey)
        cache.updateKey(newKey)
        
        assertEquals(3, cache.getCurrentLength())
        
        // Clear cache
        cache.clear()
        
        assertEquals(0, cache.getCurrentLength())
    }
}

class SamplingTest {
    
    private lateinit var graphAllocator: GGMLGraphAllocator
    
    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        graphAllocator.reserve(1024 * 1024)
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
        val logits = GGMLTensor(type = GGMLType.F32)
        logits.ne[0] = 10L // vocab size
        for (i in 1 until GGML_MAX_DIMS) logits.ne[i] = 1L
        logits.nb = calculateContiguousStrides(logits.ne, logits.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(logits)
        
        // Set logits with one clearly higher value
        for (i in 0 until 10) {
            logits.setFloat(graphAllocator, i, if (i == 5) 10.0f else 0.0f)
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
            logits.setFloat(graphAllocator, i, if (i == 5) 2.0f else 0.0f)
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
            logits.setFloat(graphAllocator, i, (10 - i).toFloat())
        }
        
        // Sample many times to check that only top-k tokens are selected
        val sampledTokens = mutableSetOf<Int>()
        repeat(100) {
            sampledTokens.add(context.sample(graphAllocator, logits))
        }
        
        // Should only sample from top-3 tokens (0, 1, 2)
        assertTrue(sampledTokens.all { it < 3 })
    }
    
    private fun createLogitsTensor(vocabSize: Int): GGMLTensor {
        val tensor = GGMLTensor(type = GGMLType.F32)
        tensor.ne[0] = vocabSize.toLong()
        for (i in 1 until GGML_MAX_DIMS) tensor.ne[i] = 1L
        tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(tensor)
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
    
    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var context: GGMLContext
    
    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        graphAllocator.reserve(10 * 1024 * 1024) // 10MB
        context = GGMLContext()
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
    }
    
    @Test
    fun testRMSNorm() {
        val norm = RMSNorm(normalizedShape = 4, eps = 1e-6f)
        
        // Initialize norm weight
        norm.weight.ne = longArrayOf(4, 1, 1, 1)
        norm.weight.nb = calculateContiguousStrides(norm.weight.ne, norm.weight.type, GGML_MAX_DIMS)
        graphAllocator.allocateTensor(norm.weight)
        
        // Set weights to 1.0
        for (i in 0 until 4) {
            norm.weight.setFloat(graphAllocator, i, 1.0f)
        }
        
        // Create input tensor
        val input = GGMLTensor(type = GGMLType.F32)
        input.ne = longArrayOf(4, 2, 1, 1) // [hidden, seq, batch]
        input.nb = calculateContiguousStrides(input.ne, input.type, GGML_MAX_DIMS)
        graphAllocator.allocateTensor(input)
        
        // Set input values
        input.setFloat(graphAllocator, 0, 0, 0, 1.0f)
        input.setFloat(graphAllocator, 1, 0, 0, 2.0f)
        input.setFloat(graphAllocator, 2, 0, 0, 3.0f)
        input.setFloat(graphAllocator, 3, 0, 0, 4.0f)
        
        val result = norm.forward(context, graphAllocator, input)
        
        // Check that result has same shape as input
        assertEquals(input.ne[0], result.ne[0])
        assertEquals(input.ne[1], result.ne[1])
        assertEquals(input.ne[2], result.ne[2])
    }
}