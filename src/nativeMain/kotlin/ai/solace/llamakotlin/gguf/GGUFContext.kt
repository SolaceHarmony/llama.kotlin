package ai.solace.llamakotlin.gguf

/**
 * GGUF context holding parsed model data
 */
class GGUFContext(
    val version: UInt,
    val metadata: Map<String, GGUFKeyValue>,
    val tensors: List<GGUFTensorInfo>,
    val dataOffset: Long,
    val alignment: Long,
    val rawData: ByteArray
) {
    /**
     * Get metadata value by key
     */
    fun getMetadataValue(key: String): Any? {
        return metadata[key]?.value
    }

    /**
     * Get string metadata value
     */
    fun getStringValue(key: String): String? {
        return getMetadataValue(key) as? String
    }

    /**
     * Get integer metadata value
     */
    fun getIntValue(key: String): Int? {
        val value = getMetadataValue(key)
        return when (value) {
            is Int -> value
            is UInt -> value.toInt()
            is Long -> value.toInt()
            is ULong -> value.toInt()
            else -> null
        }
    }

    /**
     * Get long metadata value
     */
    fun getLongValue(key: String): Long? {
        val value = getMetadataValue(key)
        return when (value) {
            is Int -> value.toLong()
            is UInt -> value.toLong()
            is Long -> value
            is ULong -> value.toLong()
            else -> null
        }
    }

    /**
     * Get float metadata value
     */
    fun getFloatValue(key: String): Float? {
        val value = getMetadataValue(key)
        return when (value) {
            is Float -> value
            is Double -> value.toFloat()
            else -> null
        }
    }

    /**
     * Get boolean metadata value
     */
    fun getBooleanValue(key: String): Boolean? {
        return getMetadataValue(key) as? Boolean
    }

    /**
     * Find tensor by name
     */
    fun findTensor(name: String): GGUFTensorInfo? {
        return tensors.find { it.name == name }
    }

    /**
     * Get tensor data as ByteArray
     */
    fun getTensorData(tensorInfo: GGUFTensorInfo): ByteArray {
        val tensorSize = calculateTensorSize(tensorInfo)
        val startOffset = (dataOffset + tensorInfo.offset).toInt()
        val endOffset = startOffset + tensorSize
        
        if (endOffset > rawData.size) {
            throw IndexOutOfBoundsException("Tensor data extends beyond file: $endOffset > ${rawData.size}")
        }
        
        return rawData.sliceArray(startOffset until endOffset)
    }

    /**
     * Calculate tensor size in bytes
     */
    private fun calculateTensorSize(tensorInfo: GGUFTensorInfo): Int {
        val elementCount = tensorInfo.dimensions.fold(1) { acc, dim -> acc * dim }
        return elementCount * tensorInfo.type.sizeBytes
    }

    /**
     * Get model architecture name if available
     */
    fun getArchitecture(): String? {
        return getStringValue("general.architecture")
    }

    /**
     * Get model name if available
     */
    fun getModelName(): String? {
        return getStringValue("general.name")
    }

    /**
     * Print summary of the model
     */
    fun printSummary() {
        println("GGUF Model Summary:")
        println("  Version: $version")
        println("  Architecture: ${getArchitecture() ?: "unknown"}")
        println("  Model Name: ${getModelName() ?: "unknown"}")
        println("  Tensors: ${tensors.size}")
        println("  Data Offset: $dataOffset")
        println("  Alignment: $alignment")
        
        println("\nMetadata (${metadata.size} entries):")
        metadata.entries.sortedBy { it.key }.take(10).forEach { (key, kv) ->
            val valueStr = when (kv.value) {
                is String -> "\"${kv.value}\""
                is List<*> -> "[${(kv.value as List<*>).size} items]"
                else -> kv.value.toString()
            }
            println("  $key: $valueStr")
        }
        if (metadata.size > 10) {
            println("  ... and ${metadata.size - 10} more")
        }
        
        println("\nTensors (${tensors.size} entries):")
        tensors.take(10).forEach { tensor ->
            val shape = tensor.dimensions.joinToString("×")
            println("  ${tensor.name}: ${tensor.type} [$shape] @ ${tensor.offset}")
        }
        if (tensors.size > 10) {
            println("  ... and ${tensors.size - 10} more")
        }
    }
}