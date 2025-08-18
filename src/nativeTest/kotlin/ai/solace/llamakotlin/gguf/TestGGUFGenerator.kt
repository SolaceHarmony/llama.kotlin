package ai.solace.llamakotlin.gguf

import ai.solace.llamakotlin.core.*

/**
 * Simple GGUF file generator for testing purposes
 */
class TestGGUFGenerator {
    
    /**
     * Generate a minimal test GGUF file as ByteArray
     */
    fun generateTestFile(): ByteArray {
        val buffer = mutableListOf<Byte>()
        
        // Header: magic + version + tensor count + metadata count
        buffer.addAll("GGUF".encodeToByteArray().toList())
        buffer.addAll(writeUInt32(3u)) // version 3
        buffer.addAll(writeUInt64(2uL)) // 2 tensors
        buffer.addAll(writeUInt64(3uL)) // 3 metadata entries
        
        // Metadata
        // 1. general.architecture
        buffer.addAll(writeString("general.architecture"))
        buffer.addAll(writeUInt32(GGUFType.STRING.value.toUInt()))
        buffer.addAll(writeString("test"))
        
        // 2. general.name
        buffer.addAll(writeString("general.name"))
        buffer.addAll(writeUInt32(GGUFType.STRING.value.toUInt()))
        buffer.addAll(writeString("test-model"))
        
        // 3. general.alignment
        buffer.addAll(writeString("general.alignment"))
        buffer.addAll(writeUInt32(GGUFType.UINT64.value.toUInt()))
        buffer.addAll(writeUInt64(32uL))
        
        // Tensor info
        // Tensor 1: weight.0 - 2x2 F32 matrix
        buffer.addAll(writeString("weight.0"))
        buffer.addAll(writeUInt32(2u)) // 2 dimensions
        buffer.addAll(writeUInt64(2uL)) // dim 0: 2
        buffer.addAll(writeUInt64(2uL)) // dim 1: 2
        buffer.addAll(writeUInt32(0u)) // F32 type (value 0)
        buffer.addAll(writeUInt64(0uL)) // offset 0
        
        // Tensor 2: weight.1 - 3x3 F32 matrix  
        buffer.addAll(writeString("weight.1"))
        buffer.addAll(writeUInt32(2u)) // 2 dimensions
        buffer.addAll(writeUInt64(3uL)) // dim 0: 3
        buffer.addAll(writeUInt64(3uL)) // dim 1: 3
        buffer.addAll(writeUInt32(0u)) // F32 type (value 0)
        buffer.addAll(writeUInt64(16uL)) // offset 16 (after first tensor: 2*2*4 = 16 bytes)
        
        // Align to 32 bytes
        val currentSize = buffer.size
        val alignment = 32
        val paddingNeeded = (alignment - (currentSize % alignment)) % alignment
        repeat(paddingNeeded) {
            buffer.add(0)
        }
        
        // Tensor data
        // Tensor 1: 2x2 F32 matrix [[1.0, 2.0], [3.0, 4.0]]
        buffer.addAll(writeFloat32(1.0f))
        buffer.addAll(writeFloat32(2.0f))
        buffer.addAll(writeFloat32(3.0f))
        buffer.addAll(writeFloat32(4.0f))
        
        // Tensor 2: 3x3 F32 matrix (identity-like)
        buffer.addAll(writeFloat32(1.0f))
        buffer.addAll(writeFloat32(0.0f))
        buffer.addAll(writeFloat32(0.0f))
        buffer.addAll(writeFloat32(0.0f))
        buffer.addAll(writeFloat32(1.0f))
        buffer.addAll(writeFloat32(0.0f))
        buffer.addAll(writeFloat32(0.0f))
        buffer.addAll(writeFloat32(0.0f))
        buffer.addAll(writeFloat32(1.0f))
        
        return buffer.toByteArray()
    }
    
    private fun writeString(str: String): List<Byte> {
        val bytes = str.encodeToByteArray()
        val result = mutableListOf<Byte>()
        result.addAll(writeUInt64(bytes.size.toULong()))
        result.addAll(bytes.toList())
        return result
    }
    
    private fun writeUInt32(value: UInt): List<Byte> {
        val bytes = ByteArray(4)
        bytes.setIntLe(0, value.toInt())
        return bytes.toList()
    }
    
    private fun writeUInt64(value: ULong): List<Byte> {
        val bytes = ByteArray(8)
        bytes.setLongLe(0, value.toLong())
        return bytes.toList()
    }
    
    private fun writeFloat32(value: Float): List<Byte> {
        val bytes = ByteArray(4)
        bytes.setFloatLe(0, value)
        return bytes.toList()
    }
}