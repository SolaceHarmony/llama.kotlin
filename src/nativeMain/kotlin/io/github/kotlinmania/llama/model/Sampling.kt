// port-lint: source llama.cpp/src/llama-sampling.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Token data structure for sampling.
 *
 * Holds the token id, its raw logit from the model, and (after softmax) the
 * normalised probability. The [logit] and [prob] fields are mutable because
 * the various sampling stages (penalties, temperature, filtering) modify them
 * in-place for efficiency.
 *
 * Note: llama.cpp does not have a separate `llama-sampling.cpp` source file;
 * sampling logic is spread across `llama-sampler.cpp` and the higher-level
 * `common/sampling.cpp`. This Kotlin module consolidates the core sampling
 * strategies (temperature, top-k, top-p, min-p, TFS, locally-typical,
 * mirostat, repetition/frequency/presence penalties) in one place.
 */
data class TokenData(
    val id: Int,
    var logit: Float,
    var prob: Float = 0.0f
)

/**
 * Sampling configuration parameters.
 *
 * All fields have sensible defaults that correspond to "greedy with full
 * distribution" (temperature 1, no filtering). Callers typically customise a
 * few fields — e.g. `temperature = 0.8f, topP = 0.95f` — and leave the rest
 * at their defaults.
 *
 * @property temperature Softmax temperature. Values < 1 sharpen, > 1 flatten.
 *   A value of 0 selects greedy (argmax) sampling.
 * @property topK Keep only the top-k most probable tokens (-1 = disabled).
 * @property topP Nucleus sampling: keep tokens whose cumulative probability
 *   reaches this threshold (1.0 = disabled).
 * @property minP Minimum probability threshold, expressed as a fraction of the
 *   highest-probability token (0.0 = disabled).
 * @property tfsZ Tail Free Sampling z parameter (1.0 = disabled).
 * @property locallyTypical Locally-typical sampling p parameter (1.0 = disabled).
 * @property penaltyRepeat Repetition penalty multiplier (1.0 = disabled).
 * @property penaltyFreq Frequency penalty: subtracted proportional to how many
 *   times a token has appeared (0.0 = disabled).
 * @property penaltyPresent Presence penalty: subtracted once for any token that
 *   has appeared before (0.0 = disabled).
 * @property mirostat Mirostat mode: 0 = disabled, 1 = v1, 2 = v2.
 * @property mirostatTau Target surprise (entropy) for Mirostat.
 * @property mirostatEta Mirostat learning rate.
 * @property penalizeNl Whether to apply repetition penalties to newline tokens.
 * @property seed Random seed for reproducibility.
 * @property grammar Optional grammar constraint applied before sampling.
 */
data class SamplingConfig(
    val temperature: Float = 1.0f,
    val topK: Int = -1,
    val topP: Float = 1.0f,
    val minP: Float = 0.0f,
    val tfsZ: Float = 1.0f,
    val locallyTypical: Float = 1.0f,
    val penaltyRepeat: Float = 1.0f,
    val penaltyFreq: Float = 0.0f,
    val penaltyPresent: Float = 0.0f,
    val mirostat: Int = 0,
    val mirostatTau: Float = 5.0f,
    val mirostatEta: Float = 0.1f,
    val penalizeNl: Boolean = false,
    val seed: Int = Random.nextInt(),
    val grammar: LlamaGrammar? = null
)

/**
 * Sampling context that maintains state across sampling operations.
 */
class SamplingContext(
    val config: SamplingConfig,
    val vocabSize: Int
) {
    private val rng = Random(config.seed)
    private var mirostatMu = 2.0f * config.mirostatTau
    
    // Token history for repetition penalties
    private val tokenHistory = mutableListOf<Int>()
    private val tokenCounts = mutableMapOf<Int, Int>()
    
    /**
     * Sample a token from the given logits tensor.
     *
     * The full pipeline is:
     * 1. Convert logits to [TokenData] candidates.
     * 2. Apply grammar constraints (if configured).
     * 3. Apply repetition / frequency / presence penalties.
     * 4. Apply temperature scaling.
     * 5. Softmax → probabilities.
     * 6. Apply distribution filters (top-k, top-p, min-p, TFS, locally-typical).
     * 7. Sample from the filtered distribution (or use Mirostat).
     * 8. Feed the accepted token back to the grammar (if configured).
     *
     * @param tokenPieceFn Optional function mapping token id to its string piece.
     *   Required when grammar constraints are active.
     * @param isEogFn Optional function returning true for end-of-generation tokens.
     *   Required when grammar constraints are active.
     */
    fun sample(
        graphAllocator: GGMLGraphAllocator,
        logits: GGMLTensor,
        penaltyTokens: List<Int> = emptyList(),
        tokenPieceFn: ((Int) -> String)? = null,
        isEogFn: ((Int) -> Boolean)? = null
    ): Int {
        require(logits.type == GGMLType.F32) { "Logits must be F32 tensor" }
        require(logits.ne[0].toInt() == vocabSize) { "Logits dimension mismatch" }

        // Convert logits to token data array
        val candidates = Array(vocabSize) { i ->
            TokenData(
                id = i,
                logit = logits.getFloat(graphAllocator, i)
            )
        }

        // Apply grammar constraints (before penalties, as in llama.cpp)
        config.grammar?.let { grammar ->
            val pieceFn = tokenPieceFn
                ?: throw IllegalStateException("tokenPieceFn required when grammar is set")
            val eogFn = isEogFn ?: { false }
            grammar.apply(candidates, pieceFn, eogFn)
        }

        // Apply penalties
        applyPenalties(candidates, penaltyTokens)

        // Apply temperature
        if (config.temperature != 1.0f && config.temperature > 0.0f) {
            for (candidate in candidates) {
                candidate.logit /= config.temperature
            }
        }

        // Convert logits to probabilities
        softmax(candidates)

        // Apply sampling strategies
        val sampledId = when {
            config.mirostat == 1 -> sampleMirostatV1(candidates)
            config.mirostat == 2 -> sampleMirostatV2(candidates)
            else -> {
                if (config.topK > 0) topKFilter(candidates, config.topK)
                if (config.topP < 1.0f) topPFilter(candidates, config.topP)
                if (config.minP > 0.0f) minPFilter(candidates, config.minP)
                if (config.tfsZ < 1.0f) tailFreeSampling(candidates, config.tfsZ)
                if (config.locallyTypical < 1.0f) locallyTypicalSampling(candidates, config.locallyTypical)
                sampleToken(candidates)
            }
        }

        // Feed accepted token to grammar
        config.grammar?.let { grammar ->
            val piece = tokenPieceFn?.invoke(sampledId) ?: ""
            if (piece.isNotEmpty()) grammar.acceptToken(sampledId, piece)
        }
        
        // Update token history
        tokenHistory.add(sampledId)
    tokenCounts[sampledId] = (tokenCounts[sampledId] ?: 0) + 1
        
        return sampledId
    }
    
    /**
     * Apply repetition penalties to candidates.
     */
    private fun applyPenalties(candidates: Array<TokenData>, penaltyTokens: List<Int>) {
        val allPenaltyTokens = (tokenHistory + penaltyTokens).distinct()
        
        for (tokenId in allPenaltyTokens) {
            if (tokenId >= 0 && tokenId < candidates.size) {
                val candidate = candidates[tokenId]
                
                // Repetition penalty
                if (config.penaltyRepeat != 1.0f) {
                    candidate.logit = if (candidate.logit <= 0.0f) {
                        candidate.logit * config.penaltyRepeat
                    } else {
                        candidate.logit / config.penaltyRepeat
                    }
                }
                
                // Frequency penalty
                if (config.penaltyFreq != 0.0f) {
                    val count = tokenCounts[tokenId] ?: 0
                    candidate.logit -= config.penaltyFreq * count
                }
                
                // Presence penalty
                if (config.penaltyPresent != 0.0f && tokenCounts.containsKey(tokenId)) {
                    candidate.logit -= config.penaltyPresent
                }
            }
        }
    }
    
    /**
     * Apply softmax to convert logits to probabilities.
     */
    private fun softmax(candidates: Array<TokenData>) {
        // Find max for numerical stability
        var maxLogit = Float.NEGATIVE_INFINITY
        for (candidate in candidates) {
            if (candidate.logit > maxLogit) {
                maxLogit = candidate.logit
            }
        }
        
        // Compute exp(x - max) and sum
        var sum = 0.0f
        for (candidate in candidates) {
            candidate.prob = exp(candidate.logit - maxLogit)
            sum += candidate.prob
        }
        
        // Normalize
        for (candidate in candidates) {
            candidate.prob /= sum
        }
    }
    
    /**
     * Apply top-k filtering.
     */
    private fun topKFilter(candidates: Array<TokenData>, k: Int) {
        if (k >= candidates.size) return
        
        // Sort by probability (descending)
        candidates.sortByDescending { it.prob }
        
        // Zero out probabilities beyond top-k
        for (i in k until candidates.size) {
            candidates[i].prob = 0.0f
        }
        
        // Renormalize
        var sum = 0.0f
        for (i in 0 until k) {
            sum += candidates[i].prob
        }
        for (i in 0 until k) {
            candidates[i].prob /= sum
        }
    }
    
    /**
     * Apply top-p (nucleus) filtering.
     */
    private fun topPFilter(candidates: Array<TokenData>, p: Float) {
        // Sort by probability (descending)
        candidates.sortByDescending { it.prob }
        
        var cumulativeProb = 0.0f
        var cutoffIndex = candidates.size
        
        for (i in candidates.indices) {
            cumulativeProb += candidates[i].prob
            if (cumulativeProb >= p) {
                cutoffIndex = i + 1
                break
            }
        }
        
        // Zero out probabilities beyond cutoff
        for (i in cutoffIndex until candidates.size) {
            candidates[i].prob = 0.0f
        }
        
        // Renormalize
        var sum = 0.0f
        for (i in 0 until cutoffIndex) {
            sum += candidates[i].prob
        }
        for (i in 0 until cutoffIndex) {
            candidates[i].prob /= sum
        }
    }
    
    /**
     * Apply minimum probability filtering.
     */
    private fun minPFilter(candidates: Array<TokenData>, minP: Float) {
        // Find max probability
        var maxProb = 0.0f
        for (candidate in candidates) {
            if (candidate.prob > maxProb) {
                maxProb = candidate.prob
            }
        }
        
        val threshold = minP * maxProb
        
        // Filter out tokens below threshold
        for (candidate in candidates) {
            if (candidate.prob < threshold) {
                candidate.prob = 0.0f
            }
        }
        
        // Renormalize
        var sum = 0.0f
        for (candidate in candidates) {
            sum += candidate.prob
        }
        for (candidate in candidates) {
            candidate.prob /= sum
        }
    }
    
    /**
     * Apply tail free sampling.
     */
    private fun tailFreeSampling(candidates: Array<TokenData>, z: Float) {
        if (z >= 1.0f) return
        
        // Sort by probability (descending)
        candidates.sortByDescending { it.prob }
        
        // Compute second derivatives
        val derivatives = FloatArray(candidates.size - 2)
        for (i in 1 until candidates.size - 1) {
            derivatives[i - 1] = candidates[i - 1].prob - 2 * candidates[i].prob + candidates[i + 1].prob
        }
        
        // Find cutoff based on cumulative absolute second derivative
        var sum = 0.0f
        var cutoff = candidates.size
        
        for (i in derivatives.indices) {
            sum += abs(derivatives[i])
            if (sum > z) {
                cutoff = i + 1
                break
            }
        }
        
        // Zero out probabilities beyond cutoff
        for (i in cutoff until candidates.size) {
            candidates[i].prob = 0.0f
        }
        
        // Renormalize
        sum = 0.0f
        for (i in 0 until cutoff) {
            sum += candidates[i].prob
        }
        for (i in 0 until cutoff) {
            candidates[i].prob /= sum
        }
    }
    
    /**
     * Apply locally typical sampling.
     */
    private fun locallyTypicalSampling(candidates: Array<TokenData>, p: Float) {
        if (p >= 1.0f) return
        
        // Compute entropy
        var entropy = 0.0f
        for (candidate in candidates) {
            if (candidate.prob > 0.0f) {
                entropy -= candidate.prob * ln(candidate.prob)
            }
        }
        
        // Compute absolute difference from entropy for each token
        val diffs = FloatArray(candidates.size)
        for (i in candidates.indices) {
            if (candidates[i].prob > 0.0f) {
                diffs[i] = abs(-ln(candidates[i].prob) - entropy)
            } else {
                diffs[i] = Float.MAX_VALUE
            }
        }
        
        // Sort by difference (ascending)
        val indices = (0 until candidates.size).sortedBy { diffs[it] }
        
        // Find cutoff
        var cumulativeProb = 0.0f
        var cutoffIndex = candidates.size
        
        for (i in indices) {
            cumulativeProb += candidates[i].prob
            if (cumulativeProb >= p) {
                cutoffIndex = indices.indexOf(i) + 1
                break
            }
        }
        
        // Zero out probabilities not in selected set
        val selectedSet = indices.take(cutoffIndex).toSet()
        for (i in candidates.indices) {
            if (i !in selectedSet) {
                candidates[i].prob = 0.0f
            }
        }
        
        // Renormalize
        var sum = 0.0f
        for (candidate in candidates) {
            sum += candidate.prob
        }
        for (candidate in candidates) {
            candidate.prob /= sum
        }
    }
    
    /**
     * Sample token using Mirostat v1.
     */
    private fun sampleMirostatV1(candidates: Array<TokenData>): Int {
        // Sort by probability (descending)
        candidates.sortByDescending { it.prob }
        
        // Estimate surprise and compute k
        val surprise = -ln(candidates[0].prob)
        val errorSurprise = surprise - config.mirostatTau
        mirostatMu -= config.mirostatEta * errorSurprise
        
        val k = max(1, (exp(mirostatMu) + 0.5f).toInt())
        
        // Apply top-k filtering
        topKFilter(candidates, min(k, candidates.size))
        
        return sampleToken(candidates)
    }
    
    /**
     * Sample token using Mirostat v2.
     */
    private fun sampleMirostatV2(candidates: Array<TokenData>): Int {
        // Sort by probability (descending)
        candidates.sortByDescending { it.prob }
        
        // Estimate surprise
        val surprise = -ln(candidates[0].prob)
        val errorSurprise = surprise - config.mirostatTau
        mirostatMu -= config.mirostatEta * errorSurprise
        
        // Compute tau threshold
        val tau = max(0.0f, config.mirostatTau + mirostatMu)
        
        // Apply threshold
        for (candidate in candidates) {
            if (-ln(candidate.prob) > tau) {
                candidate.prob = 0.0f
            }
        }
        
        // Renormalize
        var sum = 0.0f
        for (candidate in candidates) {
            sum += candidate.prob
        }
        for (candidate in candidates) {
            candidate.prob /= sum
        }
        
        return sampleToken(candidates)
    }
    
    /**
     * Sample a token from the probability distribution.
     */
    private fun sampleToken(candidates: Array<TokenData>): Int {
        val rand = rng.nextFloat()
        var cumulativeProb = 0.0f
        
        for (candidate in candidates) {
            cumulativeProb += candidate.prob
            if (cumulativeProb >= rand) {
                return candidate.id
            }
        }
        
        // Fallback to last candidate if numerical issues
        return candidates.lastOrNull { it.prob > 0.0f }?.id ?: 0
    }
    
    /**
     * Reset the sampling context.
     */
    fun reset() {
        tokenHistory.clear()
        tokenCounts.clear()
        mirostatMu = 2.0f * config.mirostatTau
    }
    
    /**
     * Get token history.
     */
    fun getTokenHistory(): List<Int> = tokenHistory.toList()
}
