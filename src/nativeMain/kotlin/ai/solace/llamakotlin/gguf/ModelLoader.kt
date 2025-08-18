package ai.solace.llamakotlin.gguf

import ai.solace.llamakotlin.core.*
import kotlin.io.path.Path
import kotlin.io.path.readBytes

/**
 * Model loader that can load GGUF files and create tensors
 */
class ModelLoader {
    
    /**
     * Load model from GGUF file
     */
    fun loadFromFile(filePath: String): LoadedModel {
        val data = Path(filePath).readBytes()
        return loadFromBytes(data)
    }
    
    /**
     * Load model from byte array
     */
    fun loadFromBytes(data: ByteArray): LoadedModel {
        val parser = GGUFParser(data)
        val context = parser.parse()
        
        return LoadedModel(context)
    }
}

/**
 * Loaded model containing GGUF context and tensor access methods
 */
class LoadedModel(val ggufContext: GGUFContext) {
    private val tensorCache = mutableMapOf<String, GGMLTensor>()
    
    /**
     * Get tensor by name, creating it if not cached
     */
    fun getTensor(name: String, context: GGMLContext): GGMLTensor? {
        return tensorCache[name] ?: run {
            val tensorInfo = ggufContext.findTensor(name) ?: return null
            val tensor = createTensorFromInfo(tensorInfo, context)
            tensorCache[name] = tensor
            tensor
        }
    }
    
    /**
     * Get all available tensor names
     */
    fun getTensorNames(): List<String> {
        return ggufContext.tensors.map { it.name }
    }
    
    /**
     * Create GGML tensor from GGUF tensor info
     */
    private fun createTensorFromInfo(info: GGUFTensorInfo, context: GGMLContext): GGMLTensor {
        // Create tensor with appropriate dimensions
        val tensor = when (info.dimensions.size) {
            1 -> createTensor1D(context, info.type, info.dimensions[0])
            2 -> createTensor2D(context, info.type, info.dimensions[0], info.dimensions[1])
            3 -> createTensor3D(context, info.type, info.dimensions[0], info.dimensions[1], info.dimensions[2])
            4 -> createTensor4D(context, info.type, info.dimensions[0], info.dimensions[1], info.dimensions[2], info.dimensions[3])
            else -> throw IllegalArgumentException("Unsupported tensor dimension count: ${info.dimensions.size}")
        }
        
        // Set the tensor name
        tensor.name = info.name
        
        // Load tensor data
        loadTensorData(tensor, info)
        
        return tensor
    }
    
    /**
     * Load tensor data from GGUF into GGML tensor
     */
    private fun loadTensorData(tensor: GGMLTensor, info: GGUFTensorInfo) {
        val data = ggufContext.getTensorData(info)
        
        // For F32 tensors, we can copy data directly to the tensor's FloatArray
        if (info.type == GGMLType.F32 && tensor.data is FloatArray) {
            val floatArray = tensor.data as FloatArray
            val expectedSize = floatArray.size * 4 // 4 bytes per float
            if (data.size != expectedSize) {
                throw IllegalArgumentException("Tensor data size mismatch: expected $expectedSize, got ${data.size}")
            }
            // Copy float data from bytes
            for (i in floatArray.indices) {
                floatArray[i] = data.getFloatLe(i * 4)
            }
        } else {
            // For other types, store raw data (to be implemented as needed)
            println("Warning: Loading data for tensor type ${info.type} not fully implemented")
        }
    }
    
    /**
     * Create a minimal test tensor for validation
     */
    fun createTestTensor(context: GGMLContext): GGMLTensor {
        // Create a simple 2x2 F32 tensor for testing
        val tensor = createTensor2D(context, GGMLType.F32, 2, 2)
        tensor.name = "test_tensor"
        
        // Fill with test data if tensor has FloatArray data
        if (tensor.data is FloatArray) {
            val data = tensor.data as FloatArray
            data[0] = 1.0f
            data[1] = 2.0f
            data[2] = 3.0f
            data[3] = 4.0f
        }
        
        return tensor
    }
    
    /**
     * Perform a simple forward pass test using matrix multiplication
     */
    fun performForwardTest(context: GGMLContext): Boolean {
        return try {
            // Create test matrices
            val a = createTestTensor(context)
            val b = createTestTensor(context)
            
            // Perform matrix multiplication using context
            val result = matMul(context, a, b)
            
            // Compute the result
            GGMLComputeOps.compute(result)
            
            // Check if we got expected results (basic validation)
            val expected = 7.0f // (1*1 + 2*3) = 7 for element [0,0]
            
            // Access result data
            if (result.data is FloatArray) {
                val resultData = result.data as FloatArray
                val actual = resultData[0]
                val tolerance = 0.001f
                kotlin.math.abs(actual - expected) < tolerance
            } else {
                false
            }
        } catch (e: Exception) {
            println("Forward test failed: ${e.message}")
            false
        }
    }
    
    /**
     * Get model metadata summary
     */
    fun getModelInfo(): String {
        return buildString {
            appendLine("Model: ${ggufContext.getModelName() ?: "Unknown"}")
            appendLine("Architecture: ${ggufContext.getArchitecture() ?: "Unknown"}")
            appendLine("Tensors: ${ggufContext.tensors.size}")
            appendLine("Version: ${ggufContext.version}")
        }
    }
}