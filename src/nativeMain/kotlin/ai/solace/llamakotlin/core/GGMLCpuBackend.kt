package ai.solace.llamakotlin.core

/**
 * CPU backend implementation for GGML operations.
 * 
 * This backend uses ByteArray for memory allocation and performs computations
 * on the CPU using the existing GGMLComputeOps implementations.
 */

/**
 * CPU backend buffer type
 */
class GGMLCpuBufferType : GGMLBackendBufferType {
    companion object {
        private const val TENSOR_ALIGNMENT = 32u // 32-byte alignment as in C++ implementation
        private const val MAX_BUFFER_SIZE = ULong.MAX_VALUE // No practical limit for CPU
    }
    
    override fun getName(): String = "CPU"
    
    override fun allocBuffer(size: ULong): GGMLBackendBuffer? {
        if (size == 0uL) return null
        
        try {
            // Allocate with extra space for alignment
            val alignedSize = (size + TENSOR_ALIGNMENT).toInt()
            val data = ByteArray(alignedSize) { 0 }
            return GGMLCpuBuffer(this, data, size)
        } catch (e: OutOfMemoryError) {
            println("GGMLCpuBufferType: Failed to allocate buffer of size $size")
            return null
        }
    }
    
    override fun getAlignment(): UInt = TENSOR_ALIGNMENT
    
    override fun getMaxSize(): ULong = MAX_BUFFER_SIZE
    
    override fun isHost(): Boolean = true
}

/**
 * CPU backend buffer implementation using ByteArray
 */
class GGMLCpuBuffer(
    private val bufferType: GGMLCpuBufferType,
    private val data: ByteArray,
    private val size: ULong
) : GGMLBackendBuffer {
    
    override fun getType(): GGMLBackendBufferType = bufferType
    
    override fun getName(): String = "CPU"
    
    override fun getBase(): Any = data
    
    override fun getSize(): ULong = size
    
    override fun free() {
        // ByteArray will be garbage collected, nothing to do explicitly
    }
    
    override fun initTensor(tensor: GGMLTensor) {
        // CPU tensors don't need special initialization
        tensor.buffer = this
    }
    
    override fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val tensorData = getBase() as ByteArray
        val srcStart = offset.toInt()
        val srcEnd = (offset + size).toInt()
        val dstStart = tensor.dataOffset.toInt()
        
        if (srcEnd > data.size) {
            throw IndexOutOfBoundsException("Source data out of bounds: $srcEnd > ${data.size}")
        }
        
        if (dstStart + size.toInt() > tensorData.size) {
            throw IndexOutOfBoundsException("Destination buffer out of bounds: ${dstStart + size.toInt()} > ${tensorData.size}")
        }
        
        data.copyInto(tensorData, dstStart, srcStart, srcEnd)
    }
    
    override fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val tensorData = getBase() as ByteArray
        val srcStart = tensor.dataOffset.toInt()
        val srcEnd = srcStart + size.toInt()
        val dstStart = offset.toInt()
        
        if (srcEnd > tensorData.size) {
            throw IndexOutOfBoundsException("Source buffer out of bounds: $srcEnd > ${tensorData.size}")
        }
        
        if (dstStart + size.toInt() > data.size) {
            throw IndexOutOfBoundsException("Destination data out of bounds: ${dstStart + size.toInt()} > ${data.size}")
        }
        
        tensorData.copyInto(data, dstStart, srcStart, srcEnd)
    }
    
    override fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean {
        // Check if source is from a host buffer (CPU or compatible)
        val srcBuffer = src.buffer
        if (srcBuffer == null || !srcBuffer.getType().isHost()) {
            return false
        }
        
        val srcData = srcBuffer.getBase() as? ByteArray ?: return false
        val dstData = getBase() as ByteArray
        
        val srcByteSize = calculateTensorByteSize(src)
        val dstByteSize = calculateTensorByteSize(dst)
        
        if (srcByteSize != dstByteSize) {
            return false
        }
        
        val srcStart = src.dataOffset.toInt()
        val srcEnd = srcStart + srcByteSize.toInt()
        val dstStart = dst.dataOffset.toInt()
        
        if (srcEnd > srcData.size || dstStart + srcByteSize.toInt() > dstData.size) {
            return false
        }
        
        srcData.copyInto(dstData, dstStart, srcStart, srcEnd)
        return true
    }
    
    override fun clear(value: UByte) {
        data.fill(value.toByte())
    }
    
    // Helper methods for ByteArray access (forwarded from GGMLTypes.kt)
    fun getIntLe(offset: Int): Int = data.getIntLe(offset)
    fun getFloatLe(offset: Int): Float = data.getFloatLe(offset)
    fun getShortLe(offset: Int): Short = data.getShortLe(offset)
    fun getLongLe(offset: Int): Long = data.getLongLe(offset)
    
    fun setIntLe(offset: Int, value: Int) = data.setIntLe(offset, value)
    fun setFloatLe(offset: Int, value: Float) = data.setFloatLe(offset, value)
    fun setShortLe(offset: Int, value: Short) = data.setShortLe(offset, value)
    fun setLongLe(offset: Int, value: Long) = data.setLongLe(offset, value)
}

/**
 * CPU backend implementation
 */
class GGMLCpuBackend : GGMLBackend {
    companion object {
        private const val BACKEND_GUID = "CPU-KOTLIN-NATIVE"
    }
    
    private val bufferType = GGMLCpuBufferType()
    
    override fun getGuid(): String = BACKEND_GUID
    
    override fun getName(): String = "CPU"
    
    override fun free() {
        // Nothing to free for CPU backend
    }
    
    override fun getDefaultBufferType(): GGMLBackendBufferType = bufferType
    
    override fun graphCompute(graph: GGMLCGraph): GGMLStatus {
        return try {
            // Use existing GGMLComputeOps to compute the graph
            GGMLComputeOps.computeGraph(graph)
            GGMLStatus.SUCCESS
        } catch (e: Exception) {
            println("GGMLCpuBackend: Error computing graph: ${e.message}")
            GGMLStatus.FAILED
        }
    }
    
    override fun supportsOp(tensor: GGMLTensor): Boolean {
        // CPU backend supports all operations that are implemented in GGMLComputeOps
        return when (tensor.op) {
            GGMLOp.NONE -> false
            GGMLOp.DUP -> true
            GGMLOp.ADD -> true
            GGMLOp.ADD1 -> true
            GGMLOp.SUB -> true
            GGMLOp.MUL -> true
            GGMLOp.DIV -> true
            GGMLOp.SQR -> true
            GGMLOp.SQRT -> true
            GGMLOp.LOG -> true
            GGMLOp.NEG -> true
            GGMLOp.ABS -> true
            GGMLOp.SGN -> true
            GGMLOp.RELU -> true
            GGMLOp.GELU -> true
            GGMLOp.GELU_QUICK -> true
            GGMLOp.SILU -> true
            GGMLOp.SILU_BACK -> true
            GGMLOp.NORM -> true
            GGMLOp.RMS_NORM -> true
            GGMLOp.RMS_NORM_BACK -> true
            GGMLOp.MUL_MAT -> true
            GGMLOp.REPEAT -> true
            GGMLOp.REPEAT_BACK -> true
            GGMLOp.CONCAT -> true
            GGMLOp.SUM -> true
            GGMLOp.SUM_ROWS -> true
            GGMLOp.MEAN -> true
            GGMLOp.ARGMAX -> true
            GGMLOp.COUNT -> false
            // Add more operations as they are implemented
            else -> false
        }
    }
    
    override fun supportsBufferType(bufferType: GGMLBackendBufferType): Boolean {
        // CPU backend supports its own buffer type and any host buffer type
        return bufferType.isHost()
    }
    
    override fun offloadOp(tensor: GGMLTensor): Boolean {
        // CPU backend doesn't need to offload operations - it's the fallback
        return false
    }
}