package ai.solace.llamakotlin.gguf

import ai.solace.llamakotlin.core.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFailsWith

class GGUFTest {
    
    @Test
    fun testGGUFParsingBasic() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val parser = GGUFParser(testData)
        val context = parser.parse()
        
        // Verify basic properties
        assertEquals(3u, context.version)
        assertEquals(2, context.tensors.size)
        assertEquals(3, context.metadata.size)
    }
    
    @Test
    fun testGGUFMetadata() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val parser = GGUFParser(testData)
        val context = parser.parse()
        
        // Check metadata values
        assertEquals("test", context.getStringValue("general.architecture"))
        assertEquals("test-model", context.getStringValue("general.name"))
        assertEquals(32L, context.getLongValue("general.alignment"))
    }
    
    @Test
    fun testGGUFTensors() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val parser = GGUFParser(testData)
        val context = parser.parse()
        
        // Check tensor info
        val tensor0 = context.findTensor("weight.0")
        assertNotNull(tensor0)
        assertEquals("weight.0", tensor0.name)
        assertEquals(GGMLType.F32, tensor0.type)
        assertEquals(2, tensor0.dimensions.size)
        assertEquals(2, tensor0.dimensions[0])
        assertEquals(2, tensor0.dimensions[1])
        assertEquals(0L, tensor0.offset)
        
        val tensor1 = context.findTensor("weight.1")
        assertNotNull(tensor1)
        assertEquals("weight.1", tensor1.name)
        assertEquals(GGMLType.F32, tensor1.type)
        assertEquals(3, tensor1.dimensions[0])
        assertEquals(3, tensor1.dimensions[1])
        assertEquals(16L, tensor1.offset)
    }
    
    @Test
    fun testGGUFTensorData() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val parser = GGUFParser(testData)
        val context = parser.parse()
        
        val tensor0 = context.findTensor("weight.0")
        assertNotNull(tensor0)
        
        val data = context.getTensorData(tensor0)
        assertEquals(16, data.size) // 2*2*4 bytes for F32
        
        // Check tensor data values [1.0, 2.0, 3.0, 4.0]
        assertEquals(1.0f, data.getFloatLe(0))
        assertEquals(2.0f, data.getFloatLe(4))
        assertEquals(3.0f, data.getFloatLe(8))
        assertEquals(4.0f, data.getFloatLe(12))
    }
    
    @Test
    fun testModelLoader() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val loader = ModelLoader()
        val model = loader.loadFromBytes(testData)
        
        // Check model info
        val info = model.getModelInfo()
        assertTrue(info.contains("test-model"))
        assertTrue(info.contains("test"))
        
        // Check tensor names
        val tensorNames = model.getTensorNames()
        assertEquals(2, tensorNames.size)
        assertTrue(tensorNames.contains("weight.0"))
        assertTrue(tensorNames.contains("weight.1"))
    }
    
    @Test
    fun testModelLoaderTensorCreation() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val loader = ModelLoader()
        val model = loader.loadFromBytes(testData)
        
        // Create context for tensor creation
        val context = GGMLContext(
            memSize = 1024uL,
            noAlloc = false
        )
        
        val tensor = model.getTensor("weight.0", context)
        assertNotNull(tensor)
        assertEquals("weight.0", tensor.name)
        assertEquals(GGMLType.F32, tensor.type)
        assertEquals(2, tensor.ne[0])
        assertEquals(2, tensor.ne[1])
        
        // Verify tensor data was loaded correctly if it has FloatArray data
        if (tensor.data is FloatArray) {
            val floatData = tensor.data as FloatArray
            assertEquals(1.0f, floatData[0])
            assertEquals(2.0f, floatData[1])
            assertEquals(3.0f, floatData[2])
            assertEquals(4.0f, floatData[3])
        }
    }
    
    @Test
    fun testForwardPass() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val loader = ModelLoader()
        val model = loader.loadFromBytes(testData)
        
        // Create context for tensor creation
        val context = GGMLContext(
            memSize = 2048uL,
            noAlloc = false
        )
        
        // Test basic forward pass with simple matrix multiplication
        val success = model.performForwardTest(context)
        assertTrue(success)
    }
    
    @Test
    fun testInvalidMagic() {
        val badData = "XXXX".encodeToByteArray()
        val parser = GGUFParser(badData)
        
        assertFailsWith<IllegalArgumentException> {
            parser.parse()
        }
    }
    
    @Test
    fun testContextPrintSummary() {
        val generator = TestGGUFGenerator()
        val testData = generator.generateTestFile()
        
        val parser = GGUFParser(testData)
        val context = parser.parse()
        
        // This should not throw an exception
        context.printSummary()
    }
}