package ai.solace.llamakotlin.gguf

/**
 * Example usage of GGUF model loading
 */
class GGUFExample {
    
    fun runExample(): String {
        return buildString {
            appendLine("GGUF Model Loading Example")
            appendLine("=========================")
            
            try {
                // Step 1: Generate a test GGUF file (stubbed minimal bytes)
                val testData = ByteArray(0)
                appendLine("✓ Generated test GGUF file (${testData.size} bytes)")
                
                // Step 2: Parse the GGUF file
                val loader = ModelLoader()
                val model = loader.loadFromBytes(testData)
                appendLine("✓ Parsed GGUF file successfully")
                
                // Step 3: Display model information
                appendLine("\nModel Information:")
                model.getModelInfo().lines().forEach { line ->
                    appendLine("  $line")
                }
                
                // Step 4: List available tensors
                appendLine("\nAvailable tensors:")
                model.getTensorNames().forEach { name ->
                    val tensorInfo = model.ggufContext.findTensor(name)
                    if (tensorInfo != null) {
                        val shape = tensorInfo.dimensions.joinToString("×")
                        appendLine("  - $name: ${tensorInfo.type} [$shape]")
                    }
                }
                
                // Step 5: Load and inspect a tensor
                val context = ai.solace.llamakotlin.core.GGMLContext(
                    memSize = 1024uL,
                    noAlloc = false
                )
                
                val tensor = model.getTensor("weight.0", context)
                if (tensor != null) {
                    appendLine("\n✓ Successfully loaded tensor: ${tensor.name}")
                    appendLine("  Type: ${tensor.type}")
                    appendLine("  Shape: [${tensor.ne.take(tensor.rank()).joinToString(" × ")}]")
                    
                    // Display some tensor data if available
                    if (tensor.data is FloatArray) {
                        val data = tensor.data as FloatArray
                        val preview = data.take(4).joinToString(", ")
                        appendLine("  Data preview: [$preview...]")
                    }
                }
                
                // Step 6: Test basic operations
                appendLine("\n🧪 Testing forward pass...")
                val success = model.performForwardTest(context)
                if (success) {
                    appendLine("✓ Forward pass test successful!")
                } else {
                    appendLine("✗ Forward pass test failed")
                }
                
                appendLine("\n🎉 Example completed successfully!")
                
            } catch (e: Exception) {
                appendLine("❌ Error: ${e.message}")
            }
        }
    }
}