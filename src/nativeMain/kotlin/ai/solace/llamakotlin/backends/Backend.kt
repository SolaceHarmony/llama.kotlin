package ai.solace.llamakotlin.backends

import ai.solace.llamakotlin.core.GGMLTensor
import ai.solace.llamakotlin.core.GGMLGraphAllocator
import ai.solace.llamakotlin.core.GGMLContext

/**
 * Interface for a computation backend.
 * A backend is responsible for executing operations defined in a GGML computation graph.
 */
interface Backend {
    /**
     * Executes a single operation (node) from the computation graph.
     *
     * @param node The tensor representing the operation to execute. Its `op` field defines the operation,
     *             and `src` fields point to its input tensors.
     * @param graphAllocator The allocator used for managing tensor data buffers for the graph.
     *                       The backend may need this to access input tensor data and to allocate
     *                       space for the output tensor's data.
     * @param context The GGML context, which might provide additional backend-specific resources or settings.
     * @return The resulting tensor from the operation. This might be the `node` tensor itself if the
     *         operation was performed in-place (where applicable and if the backend chooses to do so),
     *         or a new tensor containing the result. The result tensor's data should be managed
     *         within the provided `graphAllocator`.
     */
    fun executeOp(
        node: GGMLTensor,
        graphAllocator: GGMLGraphAllocator,
        context: GGMLContext
    ): GGMLTensor
}
