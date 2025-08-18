package ai.solace.llamakotlin.core

/**
 * Kotlin Native port of GGML backend abstraction.
 * 
 * This file defines the backend interfaces that abstract over different compute backends
 * (CPU, Metal, etc.) similar to ggml_backend_t in the C++ implementation.
 */

/**
 * Backend buffer usage enum - how the buffer will be used
 */
enum class GGMLBackendBufferUsage {
    /** Buffer will be allocated and written to on the backend */
    COMPUTE,
    /** Buffer will be uploaded from host */
    HOST
}

/**
 * Interface for backend buffer types - defines properties of a buffer type
 * Similar to ggml_backend_buffer_type in C++
 */
interface GGMLBackendBufferType {
    /** Get the name of this buffer type */
    fun getName(): String
    
    /** Allocate a buffer of the specified size */
    fun allocBuffer(size: ULong): GGMLBackendBuffer?
    
    /** Get the alignment requirements for this buffer type */
    fun getAlignment(): UInt
    
    /** Get the maximum size that can be allocated */
    fun getMaxSize(): ULong
    
    /** Check if this buffer type represents host-accessible memory */
    fun isHost(): Boolean
}

/**
 * Interface for backend buffers - represents allocated memory
 * Similar to ggml_backend_buffer in C++
 */
interface GGMLBackendBuffer {
    /** Get the buffer type that created this buffer */
    fun getType(): GGMLBackendBufferType
    
    /** Get the name of this buffer */
    fun getName(): String
    
    /** Get the base address/data of the buffer */
    fun getBase(): Any?
    
    /** Get the size of the buffer */
    fun getSize(): ULong
    
    /** Free the buffer */
    fun free()
    
    /** Initialize a tensor with this buffer */
    fun initTensor(tensor: GGMLTensor) {}
    
    /** Set tensor data from host memory */
    fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong)
    
    /** Get tensor data to host memory */
    fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong)
    
    /** Copy tensor data from another buffer */
    fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean
    
    /** Clear buffer with specified value */
    fun clear(value: UByte)
}

/**
 * Backend computation status
 */
enum class GGMLStatus {
    SUCCESS,
    FAILED,
    ABORTED
}

/**
 * Main backend interface - represents a compute backend
 * Similar to ggml_backend in C++
 */
interface GGMLBackend {
    /** Get a unique identifier for this backend */
    fun getGuid(): String
    
    /** Get the name of this backend */
    fun getName(): String
    
    /** Free the backend */
    fun free()
    
    /** Get the default buffer type for this backend */
    fun getDefaultBufferType(): GGMLBackendBufferType
    
    /** Allocate a buffer using the default buffer type */
    fun allocBuffer(size: ULong): GGMLBackendBuffer? {
        return getDefaultBufferType().allocBuffer(size)
    }
    
    /** Get alignment requirements */
    fun getAlignment(): UInt {
        return getDefaultBufferType().getAlignment()
    }
    
    /** Get maximum buffer size */
    fun getMaxSize(): ULong {
        return getDefaultBufferType().getMaxSize()
    }
    
    /** Set tensor data asynchronously (optional) */
    fun setTensorAsync(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        // Default implementation uses synchronous tensor set
        val buffer = tensor.buffer
        if (buffer != null) {
            buffer.setTensor(tensor, data, offset, size)
        }
    }
    
    /** Get tensor data asynchronously (optional) */
    fun getTensorAsync(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        // Default implementation uses synchronous tensor get
        val buffer = tensor.buffer
        if (buffer != null) {
            buffer.getTensor(tensor, data, offset, size)
        }
    }
    
    /** Copy tensor between backends (optional) */
    fun copyTensorAsync(backendSrc: GGMLBackend, src: GGMLTensor, dst: GGMLTensor): Boolean {
        return false // Default: not supported
    }
    
    /** Synchronize all pending operations (optional) */
    fun synchronize() {}
    
    /** Compute a graph on this backend */
    fun graphCompute(graph: GGMLCGraph): GGMLStatus
    
    /** Check if this backend supports the specified operation */
    fun supportsOp(tensor: GGMLTensor): Boolean
    
    /** Check if this backend supports the specified buffer type */
    fun supportsBufferType(bufferType: GGMLBackendBufferType): Boolean
    
    /** Check if this backend wants to offload an operation */
    fun offloadOp(tensor: GGMLTensor): Boolean {
        return supportsOp(tensor)
    }
}

/**
 * Backend registry entry
 */
data class GGMLBackendRegistration(
    val name: String,
    val initFunction: (String?) -> GGMLBackend?,
    val defaultBufferType: GGMLBackendBufferType,
    val userData: Any? = null
)

/**
 * Global backend registry for managing available backends
 */
object GGMLBackendRegistry {
    private val backends = mutableListOf<GGMLBackendRegistration>()
    private var initialized = false
    
    /**
     * Register a backend with the registry
     */
    fun register(registration: GGMLBackendRegistration) {
        backends.add(registration)
    }
    
    /**
     * Initialize the registry with built-in backends
     */
    fun init() {
        if (initialized) return
        initialized = true
        
        // Register CPU backend
        register(GGMLBackendRegistration(
            name = "CPU",
            initFunction = { _ -> GGMLCpuBackend() },
            defaultBufferType = GGMLCpuBufferType()
        ))
        
        // Register Metal backend (stub for now)
        register(GGMLBackendRegistration(
            name = "Metal", 
            initFunction = { _ -> GGMLMetalBackend() },
            defaultBufferType = GGMLMetalBufferType()
        ))
    }
    
    /**
     * Get the number of registered backends
     */
    fun getCount(): Int {
        init()
        return backends.size
    }
    
    /**
     * Find backend by name
     */
    fun findByName(name: String): Int? {
        init()
        return backends.indexOfFirst { it.name.equals(name, ignoreCase = true) }.takeIf { it >= 0 }
    }
    
    /**
     * Get backend name by index
     */
    fun getName(index: Int): String? {
        init()
        return backends.getOrNull(index)?.name
    }
    
    /**
     * Initialize backend by index
     */
    fun initBackend(index: Int, params: String? = null): GGMLBackend? {
        init()
        return backends.getOrNull(index)?.initFunction?.invoke(params)
    }
    
    /**
     * Get default buffer type by index
     */
    fun getDefaultBufferType(index: Int): GGMLBackendBufferType? {
        init()
        return backends.getOrNull(index)?.defaultBufferType
    }
    
    /**
     * Allocate buffer by backend index
     */
    fun allocBuffer(index: Int, size: ULong): GGMLBackendBuffer? {
        return getDefaultBufferType(index)?.allocBuffer(size)
    }
}