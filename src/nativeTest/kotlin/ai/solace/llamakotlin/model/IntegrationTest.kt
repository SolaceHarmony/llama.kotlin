package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*
import kotlin.test.*

class InferencePipelineIntegrationTest {
    
    @Test
    fun testPipelineCreation() {
        val config = LlamaConfig(
            vocabSize = 100,
            hiddenSize = 32,
            intermediateSize = 64,
            numHiddenLayers = 1,
            numAttentionHeads = 2,
            maxPositionEmbeddings = 128
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .samplingConfig(SamplingConfig(temperature = 0.8f))
            .build()
        
        assertNotNull(pipeline)
        
        val modelInfo = pipeline.getModelInfo()
        assertEquals(config.vocabSize, modelInfo.config.vocabSize)
        assertEquals(config.hiddenSize, modelInfo.config.hiddenSize)
    }
    
    @Test
    fun testParameterCountEstimation() {
        val config = LlamaConfig(
            vocabSize = 1000,
            hiddenSize = 64,
            intermediateSize = 128,
            numHiddenLayers = 2,
            numAttentionHeads = 4
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        val modelInfo = pipeline.getModelInfo()
        
        // Parameter count should be positive and reasonable
        assertTrue(modelInfo.parameterCount > 0)
        
        // Rough estimate check - should be in the right ballpark
        val expectedApprox = config.vocabSize * config.hiddenSize + 
                            config.numHiddenLayers * config.hiddenSize * 10 // Rough multiplier
        assertTrue(modelInfo.parameterCount > expectedApprox / 2)
        assertTrue(modelInfo.parameterCount < expectedApprox * 2)
    }
    
    @Test
    fun testSimpleGeneration() {
        val config = LlamaConfig(
            vocabSize = 50,
            hiddenSize = 16,
            intermediateSize = 32,
            numHiddenLayers = 1,
            numAttentionHeads = 2,
            maxPositionEmbeddings = 64
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val samplingConfig = SamplingConfig(
            temperature = 1.0f,
            seed = 42 // Fixed seed for reproducibility
        )
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .samplingConfig(samplingConfig)
            .build()
        
        // Test generation (will be mostly random due to uninitialized model)
        val result = pipeline.generate(
            prompt = "hello",
            maxNewTokens = 3
        )
        
        assertEquals("hello", result.prompt)
        assertTrue(result.generatedText.isNotEmpty() || result.generatedTokens.isEmpty()) // Might be empty if EOS is generated
        assertTrue(result.generatedTokens.size <= 3)
        assertTrue(result.inputTokens.isNotEmpty())
        assertEquals(FinishReason.MAX_TOKENS, result.finishReason)
    }
    
    @Test
    fun testStreamingGeneration() {
        val config = LlamaConfig(
            vocabSize = 30,
            hiddenSize = 8,
            intermediateSize = 16,
            numHiddenLayers = 1,
            numAttentionHeads = 2
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        val streamingResults = mutableListOf<StreamingResult>()
        
        pipeline.generateStream(
            prompt = "hi",
            maxNewTokens = 2
        ) { result ->
            streamingResults.add(result)
            true // Continue generation
        }
        
        // Should have received some streaming results
        assertTrue(streamingResults.size <= 2) // Up to maxNewTokens
        
        if (streamingResults.isNotEmpty()) {
            val lastResult = streamingResults.last()
            if (streamingResults.size == 2 || lastResult.isFinished) {
                assertTrue(lastResult.isFinished)
                assertNotNull(lastResult.finishReason)
            }
        }
    }
    
    @Test
    fun testGrammarConstrainedGeneration() {
        val config = LlamaConfig(
            vocabSize = 30,
            hiddenSize = 8,
            intermediateSize = 16,
            numHiddenLayers = 1,
            numAttentionHeads = 2
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val grammar = """
            root ::= "yes" | "no"
        """.trimIndent()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .grammar(grammar)
            .build()
        
        // Test that pipeline can be created with grammar
        assertNotNull(pipeline)
        
        // Generation test - results will depend on tokenizer and model initialization
        val result = pipeline.generate(
            prompt = "answer:",
            maxNewTokens = 5
        )
        
        // Should complete normally even with grammar constraints
        assertTrue(result.generatedTokens.size <= 5)
    }
    
    @Test
    fun testBuilderValidation() {
        val builder = LlamaInferencePipelineBuilder()
        
        // Should fail without model
        assertFailsWith<IllegalStateException> {
            builder.tokenizer(SimpleTokenizer()).build()
        }
        
        // Should fail without tokenizer
        assertFailsWith<IllegalStateException> {
            val config = LlamaConfig(vocabSize = 10, hiddenSize = 4, numAttentionHeads = 1, numHiddenLayers = 1, intermediateSize = 8)
            builder.model(LlamaModel(config)).build()
        }
    }
    
    @Test
    fun testModelConfigValidation() {
        // Test invalid hidden size / attention heads ratio
        assertFailsWith<IllegalArgumentException> {
            LlamaConfig(
                hiddenSize = 13, // Not divisible by numAttentionHeads
                numAttentionHeads = 4
            )
        }
        
        // Test valid configuration
        val validConfig = LlamaConfig(
            hiddenSize = 16,
            numAttentionHeads = 4,
            numKeyValueHeads = 2
        )
        
        assertEquals(4, validConfig.headDim)
    }
    
    @Test
    fun testSamplingConfigOptions() {
        val config = SamplingConfig(
            temperature = 0.7f,
            topK = 50,
            topP = 0.95f,
            penaltyRepeat = 1.1f,
            penaltyFreq = 0.1f,
            penaltyPresent = 0.1f,
            mirostat = 1,
            mirostatTau = 5.0f,
            mirostatEta = 0.1f
        )
        
        val model = LlamaModel(LlamaConfig(
            vocabSize = 20,
            hiddenSize = 8,
            numAttentionHeads = 2,
            numHiddenLayers = 1,
            intermediateSize = 16
        ))
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(SimpleTokenizer())
            .samplingConfig(config)
            .build()
        
        // Should create successfully with all sampling options
        assertNotNull(pipeline)
    }
    
    @Test
    fun testFinishReasons() {
        // Test that all finish reasons are available
        val reasons = FinishReason.values()
        
        assertTrue(reasons.contains(FinishReason.MAX_TOKENS))
        assertTrue(reasons.contains(FinishReason.STOP_TOKEN))
        assertTrue(reasons.contains(FinishReason.GRAMMAR_COMPLETE))
        assertTrue(reasons.contains(FinishReason.USER_STOP))
        assertTrue(reasons.contains(FinishReason.ERROR))
    }
    
    @Test
    fun testStopTokens() {
        val config = LlamaConfig(
            vocabSize = 30,
            hiddenSize = 8,
            intermediateSize = 16,
            numHiddenLayers = 1,
            numAttentionHeads = 2
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        // The actual stopping behavior depends on model output and tokenizer
        // But we can test that the API accepts stop tokens
        val result = pipeline.generate(
            prompt = "hello",
            maxNewTokens = 10,
            stopTokens = listOf("\n", ".", "!")
        )
        
        // Should complete without error
        assertTrue(result.generatedTokens.size <= 10)
    }
}

/**
 * Performance and stress tests for the inference pipeline.
 */
class InferencePipelinePerformanceTest {
    
    @Test
    fun testLargeVocabularyModel() {
        val config = LlamaConfig(
            vocabSize = 1000, // Larger vocabulary
            hiddenSize = 64,
            intermediateSize = 128,
            numHiddenLayers = 2,
            numAttentionHeads = 8
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        // Should handle larger models
        val result = pipeline.generate(
            prompt = "test",
            maxNewTokens = 2
        )
        
        assertTrue(result.generatedTokens.size <= 2)
    }
    
    @Test
    fun testLongSequenceGeneration() {
        val config = LlamaConfig(
            vocabSize = 50,
            hiddenSize = 16,
            intermediateSize = 32,
            numHiddenLayers = 1,
            numAttentionHeads = 2
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        // Test longer sequence generation
        val result = pipeline.generate(
            prompt = "a",
            maxNewTokens = 20
        )
        
        assertTrue(result.generatedTokens.size <= 20)
        assertTrue(result.allTokens.size >= 1) // At least the input token
    }
    
    @Test
    fun testMemoryUsage() {
        // Test with multiple models to check memory management
        repeat(3) { iteration ->
            val config = LlamaConfig(
                vocabSize = 30,
                hiddenSize = 8,
                intermediateSize = 16,
                numHiddenLayers = 1,
                numAttentionHeads = 2
            )
            
            val model = LlamaModel(config)
            val tokenizer = SimpleTokenizer()
            
            val pipeline = LlamaInferencePipelineBuilder()
                .model(model)
                .tokenizer(tokenizer)
                .build()
            
            val result = pipeline.generate("test$iteration", maxNewTokens = 1)
            
            // Should complete without memory issues
            assertTrue(result.generatedTokens.size <= 1)
        }
    }
}