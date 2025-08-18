package ai.solace.llamakotlin.gguf

import ai.solace.llamakotlin.core.*
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertNotNull

class GGUFIntegrationTest {
    
    @Test
    fun testCompleteModelLoadingWorkflow() {
        // Create a larger test model with multiple tensors
        val generator = TestGGUFGenerator()
        val ggufData = generator.generateTestFile()
        
        // Load the model
        val loader = ModelLoader()
        val model = loader.loadFromBytes(ggufData)
        
        // Verify model metadata
        val architecture = model.ggufContext.getStringValue("general.architecture")
        assertNotNull(architecture)
        assertTrue(architecture == "test")
        
        val modelName = model.ggufContext.getStringValue("general.name")
        assertNotNull(modelName)
        assertTrue(modelName == "test-model")
        
        // Create context for tensor operations
        val context = GGMLContext(
            memSize = 4096uL,
            noAlloc = false
        )
        
        // Load and verify first tensor
        val tensor0 = model.getTensor("weight.0", context)
        assertNotNull(tensor0)
        assertTrue(tensor0.type == GGMLType.F32)
        assertTrue(tensor0.ne[0] == 2L && tensor0.ne[1] == 2L)
        
        // Verify tensor data was loaded correctly
        if (tensor0.data is FloatArray) {
            val data = tensor0.data as FloatArray
            assertTrue(data.size == 4)
            // Check the test data values from generator
            assertTrue(data[0] == 1.0f)
            assertTrue(data[1] == 2.0f)
            assertTrue(data[2] == 3.0f)
            assertTrue(data[3] == 4.0f)
        }
        
        // Load and verify second tensor
        val tensor1 = model.getTensor("weight.1", context)
        assertNotNull(tensor1)
        assertTrue(tensor1.type == GGMLType.F32)
        assertTrue(tensor1.ne[0] == 3L && tensor1.ne[1] == 3L)
        
        if (tensor1.data is FloatArray) {
            val data = tensor1.data as FloatArray
            assertTrue(data.size == 9)
            // This should be the identity-like matrix from generator
            assertTrue(data[0] == 1.0f) // [0,0]
            assertTrue(data[4] == 1.0f) // [1,1]  
            assertTrue(data[8] == 1.0f) // [2,2]
        }
        
        // Test basic matrix operations
        val testSuccess = model.performForwardTest(context)
        assertTrue(testSuccess)
    }
    
    @Test
    fun testTensorMetadataExtraction() {
        val generator = TestGGUFGenerator()
        val ggufData = generator.generateTestFile()
        
        val parser = GGUFParser(ggufData)
        val context = parser.parse()
        
        // Verify GGUF structure
        assertTrue(context.tensors.size == 2)
        assertTrue(context.metadata.size == 3)
        
        // Check first tensor metadata
        val tensor0Info = context.findTensor("weight.0")
        assertNotNull(tensor0Info)
        assertTrue(tensor0Info.dimensions.contentEquals(intArrayOf(2, 2)))
        assertTrue(tensor0Info.type == GGMLType.F32)
        assertTrue(tensor0Info.offset == 0L)
        
        // Check second tensor metadata  
        val tensor1Info = context.findTensor("weight.1")
        assertNotNull(tensor1Info)
        assertTrue(tensor1Info.dimensions.contentEquals(intArrayOf(3, 3)))
        assertTrue(tensor1Info.type == GGMLType.F32)
        assertTrue(tensor1Info.offset == 16L) // After first tensor: 2*2*4 = 16 bytes
        
        // Verify we can extract tensor data
        val tensor0Data = context.getTensorData(tensor0Info)
        assertTrue(tensor0Data.size == 16) // 2*2*4 bytes
        
        val tensor1Data = context.getTensorData(tensor1Info)
        assertTrue(tensor1Data.size == 36) // 3*3*4 bytes
    }
    
    @Test 
    fun testGGUFVersionAndAlignment() {
        val generator = TestGGUFGenerator()
        val ggufData = generator.generateTestFile()
        
        val parser = GGUFParser(ggufData)
        val context = parser.parse()
        
        // Verify header information
        assertTrue(context.version == 3u)
        assertTrue(context.alignment == 32L)
        assertTrue(context.dataOffset > 0)
        
        // The data offset should be aligned to the specified alignment
        assertTrue(context.dataOffset % context.alignment == 0L)
    }
}