// port-lint: source llama.cpp/src/llama-sampler.h llama.cpp/src/llama-sampler.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.*
import kotlin.random.Random
import kotlin.time.TimeSource

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default seed sentinel – when used, the sampler picks a random seed. */
const val LLAMA_DEFAULT_SEED: UInt = 0xFFFF_FFFFu

// ---------------------------------------------------------------------------
// LlamaTokenDataArray  –  sortable array of (token, logit, p) triples
// ---------------------------------------------------------------------------

/**
 * A single candidate token during sampling. Mutable so samplers can adjust
 * [logit] and [p] in-place without copying.
 */
data class SamplerTokenData(
    val id: LlamaToken,
    var logit: Float,
    var p: Float = 0.0f
)

/**
 * Wrapper around a mutable list of [SamplerTokenData] that tracks the
 * current working set size, whether it is sorted by logit (descending),
 * and which index was finally selected.
 *
 * Samplers operate on this array, progressively filtering and re-weighting
 * candidates until one is selected.
 */
class LlamaTokenDataArray(
    val data: MutableList<SamplerTokenData>,
    var size: Int = data.size,
    var sorted: Boolean = false,
    var selected: Int = -1
) {
    companion object {
        /** Build from a raw logit array (one entry per vocab token). */
        fun fromLogits(logits: FloatArray): LlamaTokenDataArray {
            val list = ArrayList<SamplerTokenData>(logits.size)
            for (i in logits.indices) {
                list.add(SamplerTokenData(id = i, logit = logits[i]))
            }
            return LlamaTokenDataArray(list)
        }
    }
}

// ---------------------------------------------------------------------------
// RingBuffer  –  fixed-capacity circular buffer used by Penalties & DRY
// ---------------------------------------------------------------------------

/**
 * Fixed-capacity ring buffer. When full, [pushBack] overwrites the oldest
 * element automatically.
 */
class RingBuffer<T>(val capacity: Int, private val default: T) {
    private val data: MutableList<T> = MutableList(capacity) { default }
    private var first = 0
    private var pos = 0
    var sz = 0
        private set

    fun size(): Int = sz
    fun isEmpty(): Boolean = sz == 0

    fun front(): T {
        check(sz > 0) { "ring buffer is empty" }
        return data[first]
    }

    fun back(): T {
        check(sz > 0) { "ring buffer is empty" }
        return data[pos]
    }

    fun popFront(): T {
        check(sz > 0) { "ring buffer is empty" }
        val value = data[first]
        first = (first + 1) % capacity
        sz--
        return value
    }

    fun pushBack(value: T) {
        check(capacity > 0) { "ring buffer: capacity is zero" }
        if (sz == capacity) {
            first = (first + 1) % capacity
        } else {
            sz++
        }
        data[pos] = value
        pos = (pos + 1) % capacity
    }

    /** Access element [i] positions from the *end* (0 = most recent). */
    fun rat(i: Int): T {
        check(i in 0 until sz) { "ring buffer: index out of bounds" }
        return data[(first + sz - i - 1) % capacity]
    }

    fun clear() { sz = 0; first = 0; pos = 0 }

    fun toList(): List<T> = List(sz) { i -> data[(first + i) % capacity] }
}

// ---------------------------------------------------------------------------
// LlamaSampler interface
// ---------------------------------------------------------------------------

/**
 * Common interface for every sampling stage. Each stage transforms a
 * [LlamaTokenDataArray] in-place – typically filtering candidates, adjusting
 * logits, computing softmax probabilities, and/or choosing a selected index.
 *
 * Lifecycle:
 * 1. [apply] is called once per generation step with the current candidates.
 * 2. After a token is chosen, [accept] is called so stateful samplers (e.g.
 *    penalties, mirostat) can update their history.
 * 3. [reset] restores the sampler to its initial state (e.g. between prompts).
 * 4. [clone] creates an independent deep copy.
 * 5. [name] returns a short human-readable identifier.
 */
interface LlamaSampler {
    fun name(): String
    fun accept(token: LlamaToken) {}
    fun apply(curP: LlamaTokenDataArray)
    fun reset() {}
    fun clone(): LlamaSampler
    fun backendInit(buft: GGMLBackendBufferType): Boolean = false
    fun backendAccept(ctx: GGMLContext, gf: GGMLCGraph, selectedToken: GGMLTensor) {}
    fun backendApply(ctx: GGMLContext, gf: GGMLCGraph, data: LlamaSamplerData) {}
    fun backendSetInput() {}
}

// LlamaSamplerData, LlamaSamplerChainParams, LlamaPerfSamplerData defined in
// LlamaHTypes.kt and LlamaApi.kt — kept here previously as duplicates.

/**
 * Common backend sampler functionality. Samplers that can run on a backend
 * (GPU etc.) extend this class. The name is prefixed with +/- to indicate
 * whether the backend supports the required ops.
 */
open class LlamaSamplerBackend(private val baseName: String) {
    private var nameExt: String = baseName
    private var isInit: Boolean = false
    private var support: Boolean = false

    fun getName(): String {
        if (!isInit) return baseName
        return if (support) "+$baseName" else "-$baseName"
    }

    fun init(support: Boolean) {
        check(!isInit) { "backend already initialized" }
        isInit = true
        this.support = support
    }
}

// ---------------------------------------------------------------------------
// Utility helpers (private to this file)
// ---------------------------------------------------------------------------

private fun getRngSeed(seed: UInt): Long {
    return if (seed == LLAMA_DEFAULT_SEED) {
        Random.nextLong()
    } else {
        seed.toLong()
    }
}

/**
 * Non-inplace partial sort via bucket sort. Writes result to [res], does not mutate [curP].
 * Matches C++ `llama_token_data_array_partial_sort`.
 */
private fun partialSort(curP: LlamaTokenDataArray, nPartial: Int, res: MutableList<SamplerTokenData>) {
    val nBuckets = 128
    val bucketLow = -10.0f
    val bucketHigh = 10.0f
    val bucketScale = nBuckets / (bucketHigh - bucketLow)
    val bucketInter = -bucketLow * bucketScale

    val bucketIdx = IntArray(curP.size)
    val histo = IntArray(nBuckets)

    for (i in 0 until curP.size) {
        val v = curP.data[i].logit
        var ib = (bucketScale * v + bucketInter).toInt()
        ib = maxOf(0, minOf(nBuckets - 1, ib))
        bucketIdx[i] = ib
        histo[ib]++
    }

    var nHave = 0
    var ib = nBuckets - 1
    while (ib >= 0) {
        nHave += histo[ib]
        if (nHave >= nPartial) break
        ib--
    }

    res.clear()
    for (i in 0 until nHave) res.add(SamplerTokenData(0, 0f))

    // compute per-bucket write offsets
    val bucketOffsets = IntArray(nBuckets - ib)
    var offset = 0
    for (j in nBuckets - 1 downTo ib) {
        bucketOffsets[nBuckets - 1 - j] = offset
        offset += histo[j]
    }
    val bucketPtrs = IntArray(nBuckets - ib)
    for (i in bucketPtrs.indices) bucketPtrs[i] = bucketOffsets[i]

    for (i in 0 until curP.size) {
        val j = bucketIdx[i]
        if (j >= ib) {
            res[bucketPtrs[nBuckets - 1 - j]++] = curP.data[i]
        }
    }

    // sort each bucket, partial sort last bucket
    var ptr = 0
    var nDone = 0
    for (j in nBuckets - 1 downTo ib + 1) {
        val sub = res.subList(ptr, ptr + histo[j])
        sub.sortByDescending { it.logit }
        ptr += histo[j]
        nDone += histo[j]
    }
    // partial sort the boundary bucket
    val remaining = nPartial - nDone
    if (remaining > 0 && histo[ib] > 0) {
        val sub = res.subList(ptr, ptr + histo[ib])
        sub.sortByDescending { it.logit }
    }
}

/**
 * In-place partial sort: keep only the top [k] elements (by descending logit).
 * For small k (≤ 128) uses a simple sort; for larger k uses bucket sort.
 * Matches C++ `llama_token_data_array_partial_sort_inplace`.
 */
private fun partialSortInPlace(curP: LlamaTokenDataArray, k: Int) {
    val n = minOf(k, curP.size)
    if (n <= 0) return

    if (n <= 128) {
        // simple partial sort for small sizes
        val sub = curP.data.subList(0, curP.size)
        sub.sortByDescending { it.logit }
        curP.size = n
        curP.sorted = true
        return
    }

    val tmp = mutableListOf<SamplerTokenData>()
    partialSort(curP, n, tmp)
    for (i in 0 until n) curP.data[i] = tmp[i]
    curP.size = n
    curP.sorted = true
}

/**
 * Softmax over the current candidates, optionally sorting first.
 */
private fun softmaxImpl(curP: LlamaTokenDataArray, doSort: Boolean) {
    require(curP.size > 0)

    if (doSort && !curP.sorted) {
        partialSortInPlace(curP, curP.size)
    }

    var maxL = curP.data[0].logit
    if (!curP.sorted) {
        for (i in 1 until curP.size) {
            if (curP.data[i].logit > maxL) maxL = curP.data[i].logit
        }
    }

    var cumSum = 0.0f
    for (i in 0 until curP.size) {
        val p = exp((curP.data[i].logit - maxL).toDouble()).toFloat()
        curP.data[i].p = p
        cumSum += p
    }
    for (i in 0 until curP.size) {
        curP.data[i].p /= cumSum
    }
}

/**
 * Temperature scaling implementation. If temp ≤ 0 the highest-logit token
 * is kept and the rest are set to -Inf (greedy).
 */
private fun tempImpl(curP: LlamaTokenDataArray, temp: Float) {
    if (temp <= 0.0f) {
        var maxI = 0
        var maxL = curP.data[0].logit
        for (i in 1 until curP.size) {
            if (curP.data[i].logit > maxL) {
                curP.data[maxI].logit = Float.NEGATIVE_INFINITY
                maxI = i
                maxL = curP.data[i].logit
            } else {
                curP.data[i].logit = Float.NEGATIVE_INFINITY
            }
        }
        return
    }
    for (i in 0 until curP.size) {
        curP.data[i].logit /= temp
    }
}

/**
 * Top-k implementation shared between the TopK sampler and Mirostat.
 */
private fun topKImpl(curP: LlamaTokenDataArray, k: Int) {
    if (k <= 0) return
    val n = minOf(k, curP.size)
    if (!curP.sorted) {
        partialSortInPlace(curP, n)
    }
    curP.size = n
}

/**
 * Sample an index from the probability distribution in [curP] using the
 * cumulative-sum walk method (matches llama.cpp `llama_sampler_dist_apply`).
 */
private fun sampleDist(curP: LlamaTokenDataArray, rng: Random): Int {
    if (curP.size == 0) return -1
    if (curP.size == 1) return 0

    // compute softmax if not already done (needs valid .p)
    var maxL = curP.data[0].logit
    if (!curP.sorted) {
        for (i in 1 until curP.size) {
            if (curP.data[i].logit > maxL) maxL = curP.data[i].logit
        }
    }
    var cumSum = 0.0
    for (i in 0 until curP.size) {
        val p = exp((curP.data[i].logit - maxL).toDouble()).toFloat()
        curP.data[i].p = p
        cumSum += p
    }

    val rnd = rng.nextDouble()
    var runSum = 0.0
    val target = cumSum * rnd
    for (i in 0 until curP.size) {
        runSum += curP.data[i].p
        curP.data[i].p = (curP.data[i].p / cumSum).toFloat()
        if (runSum >= target) {
            // normalise remaining
            for (j in i + 1 until curP.size) {
                curP.data[j].p = (curP.data[j].p / cumSum).toFloat()
            }
            return i
        }
    }
    // normalise all probs
    return curP.size - 1
}

// ---------------------------------------------------------------------------
// 1. Chain sampler
// ---------------------------------------------------------------------------

/**
 * Parameters for the sampler chain.
 *
 * @property noPerf When `true`, skip timing measurements.
 */
/**
 * Sequences multiple [LlamaSampler] stages. Calling [apply] runs each stage
 * in order; [accept] is forwarded to every stage that cares about history.
 */
class LlamaSamplerChain(
    val params: LlamaSamplerChainParams = LlamaSamplerChainParams()
) : LlamaSampler {

    private val samplers = mutableListOf<LlamaSampler>()

    /** Timing (microseconds). */
    var tSampleUs: Long = 0L
        private set
    var nSample: Int = 0
        private set

    private val timeSource = TimeSource.Monotonic
    private var mark = timeSource.markNow()

    override fun name(): String = "chain"

    override fun accept(token: LlamaToken) {
        val start = if (!params.noPerf) timeSource.markNow() else mark
        for (s in samplers) s.accept(token)
        nSample++
        if (!params.noPerf) tSampleUs += start.elapsedNow().inWholeMicroseconds
    }

    override fun apply(curP: LlamaTokenDataArray) {
        val start = if (!params.noPerf) timeSource.markNow() else mark
        for (s in samplers) s.apply(curP)
        if (!params.noPerf) tSampleUs += start.elapsedNow().inWholeMicroseconds
    }

    override fun reset() { for (s in samplers) s.reset() }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerChain(params)
        for (s in samplers) c.add(s.clone())
        return c
    }

    // -- chain management --

    fun add(sampler: LlamaSampler) { samplers.add(sampler) }

    fun get(i: Int): LlamaSampler? = samplers.getOrNull(i)

    fun remove(i: Int): LlamaSampler? {
        if (i !in samplers.indices) return null
        return samplers.removeAt(i)
    }

    fun n(): Int = samplers.size

    fun resetPerf() { tSampleUs = 0; nSample = 0 }

    /** Sample a token from [logits] using this chain. */
    fun sample(logits: FloatArray): LlamaToken {
        val curP = LlamaTokenDataArray.fromLogits(logits)
        apply(curP)
        require(curP.selected in 0 until curP.size) { "no token selected by sampler chain" }
        val token = curP.data[curP.selected].id
        accept(token)
        return token
    }

    /** Walk backwards through the chain looking for the first seeded sampler. */
    fun getSeed(): UInt {
        for (s in samplers.reversed()) {
            val seed = when (s) {
                is LlamaSamplerDist -> s.seedCur
                is LlamaSamplerMirostat -> s.seedCur
                is LlamaSamplerMirostatV2 -> s.seedCur
                is LlamaSamplerChain -> {
                    val inner = s.getSeed()
                    if (inner != LLAMA_DEFAULT_SEED) inner else continue
                }
                else -> continue
            }
            if (seed != LLAMA_DEFAULT_SEED) return seed
        }
        return LLAMA_DEFAULT_SEED
    }

}

// ---------------------------------------------------------------------------
// 2. Greedy sampler
// ---------------------------------------------------------------------------

/** Picks the token with the highest logit (argmax). */
class LlamaSamplerGreedy : LlamaSampler {
    override fun name(): String = "greedy"
    override fun apply(curP: LlamaTokenDataArray) {
        curP.selected = 0
        for (i in 1 until curP.size) {
            if (curP.data[i].logit > curP.data[curP.selected].logit) {
                curP.selected = i
            }
        }
    }
    override fun clone(): LlamaSampler = LlamaSamplerGreedy()
}

// ---------------------------------------------------------------------------
// 3. Dist sampler (temperature-scaled random)
// ---------------------------------------------------------------------------

/**
 * Distribution sampler: applies softmax then samples proportional to
 * probability using the cumulative-sum walk.
 */
class LlamaSamplerDist(val seed: UInt = LLAMA_DEFAULT_SEED) : LlamaSampler {

    internal var seedCur: UInt = getRngSeed(seed).toUInt()
    private var rng = Random(seedCur.toLong())

    override fun name(): String = "dist"

    override fun apply(curP: LlamaTokenDataArray) {
        if (curP.size == 0) { curP.selected = -1; return }
        curP.selected = 0
        if (curP.size == 1) { curP.data[0].p = 1.0f; return }

        // softmax + cumulative walk in one pass
        var maxL = curP.data[0].logit
        if (!curP.sorted) {
            for (i in 1 until curP.size) maxL = maxOf(maxL, curP.data[i].logit)
        }
        var cumSum = 0.0
        for (i in 0 until curP.size) {
            val p = exp((curP.data[i].logit - maxL).toDouble()).toFloat()
            curP.data[i].p = p
            cumSum += p
        }

        val rnd = rng.nextDouble()
        var runSum = 0.0
        val target = cumSum * rnd
        var found = false
        for (i in 0 until curP.size) {
            if (!found) {
                runSum += curP.data[i].p
                if (runSum >= target) { curP.selected = i; found = true }
            }
            curP.data[i].p = (curP.data[i].p / cumSum).toFloat()
        }
        if (!found) curP.selected = curP.size - 1
    }

    override fun reset() {
        seedCur = getRngSeed(seed).toUInt()
        rng = Random(seedCur.toLong())
    }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerDist(seed)
        c.seedCur = seedCur
        c.rng = Random(seedCur.toLong())
        return c
    }
}

// ---------------------------------------------------------------------------
// 4. Top-K sampler
// ---------------------------------------------------------------------------

/** Keep only the top [k] candidates by logit. */
class LlamaSamplerTopK(val k: Int) : LlamaSampler {
    override fun name(): String = "top-k"
    override fun apply(curP: LlamaTokenDataArray) = topKImpl(curP, k)
    override fun clone(): LlamaSampler = LlamaSamplerTopK(k)
}

// ---------------------------------------------------------------------------
// 5. Top-P (nucleus) sampler
// ---------------------------------------------------------------------------

/**
 * Nucleus sampling: keep the smallest set of tokens whose cumulative
 * probability reaches [p], retaining at least [minKeep] tokens.
 */
class LlamaSamplerTopP(val p: Float, val minKeep: Int = 1) : LlamaSampler {

    override fun name(): String = "top-p"

    override fun apply(curP: LlamaTokenDataArray) {
        if (p >= 1.0f) return

        softmaxImpl(curP, false)

        var k = curP.size
        var pdata = curP.data
        val bufSort = mutableListOf<SamplerTokenData>()

        // if not sorted, try adaptive top-k sorting
        if (!curP.sorted && curP.size > 1024) {
            k = minOf(256, curP.size)
            partialSort(curP, k, bufSort)
            pdata = bufSort
        } else if (!curP.sorted) {
            partialSortInPlace(curP, k)
        }

        var cumSum = 0.0f
        var lastIdx = curP.size
        for (i in 0 until curP.size) {
            cumSum += pdata[i].p
            if (cumSum >= p && i + 1 >= minKeep) {
                lastIdx = i + 1
                break
            }
            // exceeded current top-k heuristic -> expand and re-sort
            if (!curP.sorted && i == k - 1) {
                k = curP.size
                partialSort(curP, k, bufSort)
                pdata = bufSort
            }
        }

        if (!curP.sorted) {
            for (i in 0 until lastIdx) curP.data[i] = bufSort[i]
            curP.sorted = true
        }
        curP.size = lastIdx
    }

    override fun clone(): LlamaSampler = LlamaSamplerTopP(p, minKeep)
}

// ---------------------------------------------------------------------------
// 6. Min-P sampler
// ---------------------------------------------------------------------------

/**
 * Keep tokens whose probability is at least [p] × max_prob. In logit space
 * this becomes: keep if logit ≥ max_logit + ln(p).
 */
class LlamaSamplerMinP(val p: Float, val minKeep: Int = 1) : LlamaSampler {

    override fun name(): String = "min-p"

    override fun apply(curP: LlamaTokenDataArray) {
        if (p <= 0.0f || curP.size == 0) return

        // Try unsorted fast path
        if (!curP.sorted) {
            var maxLogit = Float.NEGATIVE_INFINITY
            for (i in 0 until curP.size) {
                if (curP.data[i].logit > maxLogit) maxLogit = curP.data[i].logit
            }
            val minLogit = maxLogit + ln(p)
            val filtered = ArrayList<SamplerTokenData>()
            for (i in 0 until curP.size) {
                if (curP.data[i].logit >= minLogit) filtered.add(curP.data[i])
            }
            if (filtered.isNotEmpty() && filtered.size >= minKeep) {
                for (i in filtered.indices) curP.data[i] = filtered[i]
                curP.size = filtered.size
                return
            }
        }

        // Sorted fallback
        if (!curP.sorted) partialSortInPlace(curP, curP.size)
        val minLogit = curP.data[0].logit + ln(p)
        var i = 1
        while (i < curP.size) {
            if (curP.data[i].logit < minLogit && i >= minKeep) break
            i++
        }
        curP.size = i
    }

    override fun clone(): LlamaSampler = LlamaSamplerMinP(p, minKeep)
}

// ---------------------------------------------------------------------------
// 7. Typical-P sampler
// ---------------------------------------------------------------------------

/**
 * Locally-typical sampling: keep tokens whose information content is closest
 * to the entropy of the distribution, up to cumulative probability [p].
 *
 * Reference: <https://arxiv.org/abs/2202.00666>
 */
class LlamaSamplerTypical(val p: Float, val minKeep: Int = 1) : LlamaSampler {

    override fun name(): String = "typical"

    override fun apply(curP: LlamaTokenDataArray) {
        if (p >= 1.0f) return

        softmaxImpl(curP, true)

        var entropy = 0.0f
        for (i in 0 until curP.size) {
            if (curP.data[i].p > 0f) entropy += -curP.data[i].p * ln(curP.data[i].p)
        }

        // shifted scores = |−log(p_i) − entropy|
        val shifted = FloatArray(curP.size) { i ->
            abs(-ln(curP.data[i].p) - entropy)
        }

        // sort indices by shifted score ascending
        val indices = (0 until curP.size).sortedBy { shifted[it] }

        var cumSum = 0.0f
        var lastIdx = indices.size
        for (i in indices.indices) {
            cumSum += curP.data[indices[i]].p
            if (cumSum > p && (minKeep == 0 || i >= minKeep - 1)) {
                lastIdx = i + 1; break
            }
        }

        val newData = ArrayList<SamplerTokenData>(lastIdx)
        for (i in 0 until lastIdx) newData.add(curP.data[indices[i]])
        for (i in newData.indices) curP.data[i] = newData[i]
        curP.size = newData.size
        curP.sorted = false
    }

    override fun clone(): LlamaSampler = LlamaSamplerTypical(p, minKeep)
}

// ---------------------------------------------------------------------------
// 8. XTC (cross-token consistency) sampler
// ---------------------------------------------------------------------------

/**
 * XTC sampling: with [probability] chance, remove the top tokens whose
 * softmax probability exceeds [threshold], keeping at least [minKeep].
 */
class LlamaSamplerXtc(
    val probability: Float,
    val threshold: Float,
    val minKeep: Int = 1,
    val seed: UInt = LLAMA_DEFAULT_SEED
) : LlamaSampler {

    internal var seedCur: UInt = getRngSeed(seed).toUInt()
    private var rng = Random(seedCur.toLong())

    override fun name(): String = "xtc"

    override fun apply(curP: LlamaTokenDataArray) {
        if (probability <= 0.0f || threshold > 0.5f || curP.size < 2) return

        if (rng.nextFloat() > probability) return

        softmaxImpl(curP, true)

        var posLast = 0
        for (i in 0 until curP.size) {
            if (curP.data[i].p >= threshold) posLast = i else break
        }

        if (curP.size - posLast >= minKeep && posLast > 0) {
            // shift data left by posLast
            for (i in posLast until curP.size) {
                curP.data[i - posLast] = curP.data[i]
            }
            curP.size -= posLast
        }
    }

    override fun reset() {
        seedCur = getRngSeed(seed).toUInt()
        rng = Random(seedCur.toLong())
    }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerXtc(probability, threshold, minKeep, seed)
        c.seedCur = seedCur
        c.rng = Random(seedCur.toLong())
        return c
    }
}

// ---------------------------------------------------------------------------
// 9. Temperature sampler
// ---------------------------------------------------------------------------

/** Simple temperature scaling. */
class LlamaSamplerTemp(val temp: Float) : LlamaSampler {
    override fun name(): String = "temp"
    override fun apply(curP: LlamaTokenDataArray) = tempImpl(curP, temp)
    override fun clone(): LlamaSampler = LlamaSamplerTemp(temp)
}

// ---------------------------------------------------------------------------
// 10. Temperature-ext (dynamic temperature with smoothing)
// ---------------------------------------------------------------------------

/**
 * Dynamic temperature: when [delta] > 0, the effective temperature is
 * interpolated between `temp ± delta` based on the normalised entropy of
 * the softmax distribution, raised to the [exponent] power.
 */
class LlamaSamplerTempExt(
    val temp: Float,
    val delta: Float,
    val exponent: Float
) : LlamaSampler {

    override fun name(): String = "temp-ext"

    override fun apply(curP: LlamaTokenDataArray) {
        if (delta > 0f) {
            val minTemp = maxOf(0.0f, temp - delta)
            val maxTemp = temp + delta

            if (curP.size <= 1) return

            val maxEntropy = -ln(1.0f / curP.size)
            softmaxImpl(curP, true)

            var entropy = 0.0f
            for (i in 0 until curP.size) {
                val prob = curP.data[i].p
                if (prob > 0.0f) entropy -= prob * ln(prob)
            }
            val normEntropy = entropy / maxEntropy
            val dynTemp = minTemp + (maxTemp - minTemp) * normEntropy.toDouble().pow(exponent.toDouble()).toFloat()

            tempImpl(curP, dynTemp)

            // recompute softmax after temperature scaling
            val maxLD = curP.data[0].logit.toDouble()
            var cumSum = 0.0
            for (i in 0 until curP.size) {
                val p = exp(curP.data[i].logit.toDouble() - maxLD)
                curP.data[i].p = p.toFloat()
                cumSum += p
            }
            for (i in 0 until curP.size) {
                curP.data[i].p = (curP.data[i].p / cumSum).toFloat()
            }
        } else {
            tempImpl(curP, temp)
        }
    }

    override fun clone(): LlamaSampler = LlamaSamplerTempExt(temp, delta, exponent)
}

// ---------------------------------------------------------------------------
// 11. Penalties (repetition / frequency / presence)
// ---------------------------------------------------------------------------

/**
 * Applies repetition, frequency, and presence penalties based on recent token
 * history. Uses a [RingBuffer] of size [penaltyLastN] to track context.
 */
class LlamaSamplerPenalties(
    val penaltyLastN: Int,
    val penaltyRepeat: Float = 1.0f,
    val penaltyFreq: Float = 0.0f,
    val penaltyPresent: Float = 0.0f
) : LlamaSampler {

    private val prev = RingBuffer<LlamaToken>(maxOf(penaltyLastN, 0), 0)
    private val tokenCount = HashMap<LlamaToken, Int>()

    override fun name(): String = "penalties"

    override fun accept(token: LlamaToken) {
        if (penaltyLastN == 0) return

        tokenCount[token] = (tokenCount[token] ?: 0) + 1

        if (prev.size() >= penaltyLastN) {
            val old = prev.front()
            val cnt = (tokenCount[old] ?: 1) - 1
            if (cnt <= 0) tokenCount.remove(old) else tokenCount[old] = cnt
        }
        prev.pushBack(token)
    }

    override fun apply(curP: LlamaTokenDataArray) {
        if (penaltyLastN == 0) return
        if (penaltyRepeat == 1.0f && penaltyFreq == 0.0f && penaltyPresent == 0.0f) return

        for (i in 0 until curP.size) {
            val count = tokenCount[curP.data[i].id] ?: continue
            if (count <= 0) continue

            if (curP.data[i].logit <= 0f) {
                curP.data[i].logit *= penaltyRepeat
            } else {
                curP.data[i].logit /= penaltyRepeat
            }
            curP.data[i].logit -= count.toFloat() * penaltyFreq + (if (count > 0) 1f else 0f) * penaltyPresent
        }
        curP.sorted = false
    }

    override fun reset() { prev.clear(); tokenCount.clear() }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerPenalties(penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent)
        // copy history
        for (t in prev.toList()) c.accept(t)
        return c
    }
}

// ---------------------------------------------------------------------------
// 12. Top-N-Sigma sampler
// ---------------------------------------------------------------------------

/**
 * Masks tokens whose logit is more than [n] standard deviations below the
 * maximum logit (ignoring already-masked tokens at -Inf).
 */
class LlamaSamplerTopNSigma(val n: Float) : LlamaSampler {

    override fun name(): String = "top-n-sigma"

    override fun apply(curP: LlamaTokenDataArray) {
        if (n <= 0.0f || curP.size <= 1) return

        var maxLogit = curP.data[0].logit
        var logitsSum = 0.0f
        var validCount = 0
        for (i in 0 until curP.size) {
            if (curP.data[i].logit != Float.NEGATIVE_INFINITY) {
                if (curP.data[i].logit > maxLogit) maxLogit = curP.data[i].logit
                logitsSum += curP.data[i].logit
                validCount++
            }
        }
        val mean = if (validCount > 0) logitsSum / validCount else 0f

        var acc = 0.0f
        for (i in 0 until curP.size) {
            if (curP.data[i].logit != Float.NEGATIVE_INFINITY) {
                acc += (curP.data[i].logit - mean).let { it * it }
            }
        }
        val std = if (validCount > 0) sqrt(acc / validCount) else 0f

        for (i in 0 until curP.size) {
            if (curP.data[i].logit < maxLogit - n * std) {
                curP.data[i].logit = Float.NEGATIVE_INFINITY
            }
        }
        softmaxImpl(curP, true)
    }

    override fun clone(): LlamaSampler = LlamaSamplerTopNSigma(n)
}

// ---------------------------------------------------------------------------
// 13. DRY (Don't Repeat Yourself) penalty
// ---------------------------------------------------------------------------

/**
 * Detects repeated suffixes in recent context using the Z-algorithm and
 * applies an exponential penalty `multiplier × base^(repeat_len − allowed_length)`
 * to discourage continuation of those patterns.
 *
 * Ported from Koboldcpp (original author: pi6am).
 *
 * @property totalContextSize Maximum context window the model was trained on.
 * @property dryMultiplier    Penalty strength. 0 = disabled.
 * @property dryBase          Exponential base (must be ≥ 1.0).
 * @property dryAllowedLength Repetitions shorter than this are ignored.
 * @property dryPenaltyLastN  How many recent tokens to inspect (-1 = full context).
 * @property processedBreakers Map from a "head" token to lists of "tail" tokens
 *           forming sequence breakers (e.g. punctuation).
 */
class LlamaSamplerDry(
    val totalContextSize: Int,
    val dryMultiplier: Float,
    val dryBase: Float,
    val dryAllowedLength: Int,
    val dryPenaltyLastN: Int,
    val processedBreakers: HashMap<LlamaToken, MutableList<List<LlamaToken>>> = HashMap()
) : LlamaSampler {

    private val lastTokens = RingBuffer<LlamaToken>(
        if (dryPenaltyLastN == -1) totalContextSize
        else maxOf(dryPenaltyLastN, 0), 0
    )
    private var dryRepeatCount = IntArray(0)
    private val dryMaxTokenRepeat = HashMap<LlamaToken, Int>()

    override fun name(): String = "dry"

    override fun accept(token: LlamaToken) {
        if (dryMultiplier == 0.0f || dryBase < 1.0f || dryPenaltyLastN == 0) return
        lastTokens.pushBack(token)
    }

    override fun apply(curP: LlamaTokenDataArray) {
        if (dryMultiplier == 0.0f || dryBase < 1.0f || dryPenaltyLastN == 0) return

        val effectiveLast = if (dryPenaltyLastN == -1) totalContextSize else maxOf(dryPenaltyLastN, 0)
        val lastNRepeat = minOf(lastTokens.size(), effectiveLast, totalContextSize)

        if (lastNRepeat <= dryAllowedLength) return

        dryRepeatCount = IntArray(lastNRepeat)
        dryMaxTokenRepeat.clear()

        // Step 1: find rep_limit via restart sequences
        var repLimit = lastNRepeat
        for (i in 0 until lastNRepeat) {
            val token = lastTokens.rat(i)
            val seqs = processedBreakers[token] ?: continue
            var longestMatch = -1
            for (seq in seqs) {
                val seqLen = seq.size
                if (seqLen > longestMatch && seqLen <= i) {
                    var match = true
                    for (offset in 0 until seqLen) {
                        if (seq[offset] != lastTokens.rat(i - offset - 1)) { match = false; break }
                    }
                    if (match) longestMatch = seqLen
                }
            }
            if (longestMatch >= 0) {
                repLimit = i - longestMatch
                break
            }
        }
        if (repLimit < dryAllowedLength) return

        // Step 2: Z-algorithm (reversed) to find repeat counts
        val last = lastNRepeat - 1
        var rt = 0; var lt = 0
        for (k in 1 until lastNRepeat) {
            if (k > rt) {
                var n = 0
                while (n + k < lastNRepeat && lastTokens.rat(n) == lastTokens.rat(n + k)) n++
                dryRepeatCount[last - k] = minOf(n, repLimit)
                if (n > 0) { lt = k; rt = k + n - 1 }
            } else {
                val p = k - lt
                val rightPartLen = rt - k + 1
                if (dryRepeatCount[last - p] < rightPartLen) {
                    dryRepeatCount[last - k] = minOf(dryRepeatCount[last - p], repLimit)
                } else {
                    var ii = rt + 1
                    while (ii < lastNRepeat && lastTokens.rat(ii) == lastTokens.rat(ii - k)) ii++
                    dryRepeatCount[last - k] = minOf(ii - k, repLimit)
                    lt = k; rt = ii - 1
                }
            }
        }

        // Step 3: find max repeat length per "next token"
        for (i in 0 until lastNRepeat - 1) {
            val repeatLen = dryRepeatCount[i]
            if (repeatLen >= dryAllowedLength) {
                val token = lastTokens.rat(lastNRepeat - 2 - i)
                val existing = dryMaxTokenRepeat[token]
                if (existing == null || existing < repeatLen) {
                    dryMaxTokenRepeat[token] = repeatLen
                }
            }
        }

        // Step 4: apply penalties
        val floatMaxLog = 88.7228391f
        val maxExponent = if (dryBase > 1.000001f) (floatMaxLog / ln(dryBase)).toInt() else 0

        for (i in 0 until curP.size) {
            val repeatLen = dryMaxTokenRepeat[curP.data[i].id] ?: continue

            // check if it's a single-token breaker
            val seqs = processedBreakers[curP.data[i].id]
            var isSingleTokenBreaker = false
            if (seqs != null) {
                for (seq in seqs) { if (seq.isEmpty()) { isSingleTokenBreaker = true; break } }
            }
            if (isSingleTokenBreaker) continue

            var repeatExp = repeatLen - dryAllowedLength
            if (maxExponent > 0 && repeatExp > maxExponent) repeatExp = maxExponent
            val penalty = dryMultiplier * dryBase.toDouble().pow(repeatExp.toDouble()).toFloat()
            curP.data[i].logit -= penalty
        }
        curP.sorted = false
    }

    override fun reset() {
        lastTokens.clear()
        dryRepeatCount = IntArray(0)
        dryMaxTokenRepeat.clear()
    }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerDry(
            totalContextSize, dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN,
            HashMap(processedBreakers)
        )
        for (t in lastTokens.toList()) c.accept(t)
        return c
    }
}

// ---------------------------------------------------------------------------
// 14. Logit bias
// ---------------------------------------------------------------------------

/** A single logit bias entry: add [bias] to the logit of [token]. */
data class LlamaLogitBias(val token: LlamaToken, val bias: Float)

/**
 * Adds manual bias values to specified token logits.
 */
class LlamaSamplerLogitBias(
    val nVocab: Int,
    val logitBias: List<LlamaLogitBias>
) : LlamaSampler {

    override fun name(): String = "logit-bias"

    override fun apply(curP: LlamaTokenDataArray) {
        if (logitBias.isEmpty()) return

        val toSearch = mutableListOf<LlamaLogitBias>()
        for (lb in logitBias) {
            if (lb.token >= 0 && curP.size > lb.token && curP.data[lb.token].id == lb.token) {
                curP.data[lb.token].logit += lb.bias
            } else {
                toSearch.add(lb)
            }
        }
        if (toSearch.isEmpty()) return
        for (i in 0 until curP.size) {
            for (lb in toSearch) {
                if (curP.data[i].id == lb.token) {
                    curP.data[i].logit += lb.bias
                    break
                }
            }
        }
    }

    override fun clone(): LlamaSampler = LlamaSamplerLogitBias(nVocab, logitBias.toList())
}

// ---------------------------------------------------------------------------
// 15. Mirostat v1
// ---------------------------------------------------------------------------

/**
 * Mirostat v1: adaptive perplexity targeting via surprise-based feedback.
 *
 * @property nVocab Vocabulary size (needed for the Zipf estimator).
 * @property seed   RNG seed.
 * @property tau    Target surprise (perplexity = 2^tau).
 * @property eta    Learning rate for mu updates.
 * @property m      Number of top tokens used to estimate s_hat.
 */
class LlamaSamplerMirostat(
    val nVocab: Int,
    val seed: UInt = LLAMA_DEFAULT_SEED,
    val tau: Float = 5.0f,
    val eta: Float = 0.1f,
    val m: Int = 100
) : LlamaSampler {

    internal var seedCur: UInt = getRngSeed(seed).toUInt()
    private var rng = Random(seedCur.toLong())
    internal var mu: Float = 2.0f * tau

    override fun name(): String = "mirostat"

    override fun apply(curP: LlamaTokenDataArray) {
        softmaxImpl(curP, true)

        // estimate s_hat
        var sumTiBi = 0.0f; var sumTiSq = 0.0f
        val limit = minOf(m - 1, curP.size - 1)
        for (i in 0 until limit) {
            val ti = ln((i + 2).toFloat() / (i + 1).toFloat())
            val bi = ln(curP.data[i].p / curP.data[i + 1].p)
            sumTiBi += ti * bi
            sumTiSq += ti * ti
        }
        val sHat = if (sumTiSq != 0f) sumTiBi / sumTiSq else 1f

        val epsilonHat = sHat - 1f
        val k = ((epsilonHat * 2f.pow(mu)) / (1f - nVocab.toFloat().pow(-epsilonHat))).pow(1f / sHat)

        topKImpl(curP, maxOf(k.toInt(), 1))
        softmaxImpl(curP, true)

        val idx = sampleDist(curP, rng)
        curP.selected = idx

        val observedSurprise = -log2(curP.data[idx].p)
        mu -= eta * (observedSurprise - tau)
    }

    override fun reset() {
        mu = 2.0f * tau
        seedCur = getRngSeed(seed).toUInt()
        rng = Random(seedCur.toLong())
    }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerMirostat(nVocab, seed, tau, eta, m)
        c.mu = mu
        c.seedCur = seedCur
        c.rng = Random(seedCur.toLong())
        return c
    }
}

// ---------------------------------------------------------------------------
// 16. Mirostat v2
// ---------------------------------------------------------------------------

/**
 * Mirostat v2: simpler adaptive perplexity targeting that truncates tokens
 * whose surprise (−log₂ p) exceeds μ, then samples and updates μ.
 */
class LlamaSamplerMirostatV2(
    val seed: UInt = LLAMA_DEFAULT_SEED,
    val tau: Float = 5.0f,
    val eta: Float = 0.1f
) : LlamaSampler {

    internal var seedCur: UInt = getRngSeed(seed).toUInt()
    private var rng = Random(seedCur.toLong())
    internal var mu: Float = 2.0f * tau

    override fun name(): String = "mirostat-v2"

    override fun apply(curP: LlamaTokenDataArray) {
        softmaxImpl(curP, true)

        // truncate tokens with surprise > mu
        var newSize = curP.size
        for (i in 0 until curP.size) {
            if (-log2(curP.data[i].p) > mu) { newSize = i; break }
        }
        if (newSize == 0) newSize = 1
        curP.size = newSize

        softmaxImpl(curP, true)

        val idx = sampleDist(curP, rng)
        curP.selected = idx

        val observedSurprise = -log2(curP.data[idx].p)
        mu -= eta * (observedSurprise - tau)
    }

    override fun reset() {
        mu = 2.0f * tau
        seedCur = getRngSeed(seed).toUInt()
        rng = Random(seedCur.toLong())
    }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerMirostatV2(seed, tau, eta)
        c.mu = mu
        c.seedCur = seedCur
        c.rng = Random(seedCur.toLong())
        return c
    }
}

// ---------------------------------------------------------------------------
// 17. Grammar sampler (skeleton – integrates with Grammar.kt)
// ---------------------------------------------------------------------------

/**
 * Grammar-constrained sampling. Delegates to the GBNF grammar engine in
 * Grammar.kt to mask out tokens that would violate the grammar.
 *
 * Transliteration of `llama_sampler_grammar` from llama-sampler.cpp.
 */
class LlamaSamplerGrammar(
    val vocab: LlamaVocab?,
    val grammarStr: String = "",
    val grammarRoot: String = "root"
) : LlamaSampler {

    var grammar: LlamaGrammar? = if (vocab != null && grammarStr.isNotEmpty()) {
        LlamaGrammar.fromString(grammarStr, grammarRoot)
    } else {
        null
    }

    override fun name(): String = "grammar"

    override fun accept(token: LlamaToken) {
        val g = grammar ?: return
        val piece = vocab?.tokenToPiece(token) ?: return
        g.acceptImpl(token, piece) { vocab.isEog(it) }
    }

    override fun apply(curP: LlamaTokenDataArray) {
        val g = grammar ?: return
        val v = vocab ?: return
        // Build an Array<TokenData> wrapper for the grammar engine
        val candidates = Array(curP.size) { i ->
            TokenData(curP.data[i].id, curP.data[i].logit, curP.data[i].p)
        }
        g.apply(candidates, { id -> v.tokenToPiece(id) }, { id -> v.isEog(id) })
        // Write back any logit changes (grammar sets rejected logits to -Inf)
        for (i in 0 until curP.size) {
            curP.data[i].logit = candidates[i].logit
        }
    }

    override fun reset() {
        val v = vocab ?: return
        if (grammar == null) return
        grammar = LlamaGrammar.fromString(grammarStr, grammarRoot)
    }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerGrammar(vocab, "", "")
        if (grammar != null) {
            c.grammar = grammar!!.clone()
        }
        return c
    }
}

// ---------------------------------------------------------------------------
// 18. Infill sampler
// ---------------------------------------------------------------------------

/**
 * Special sampling logic for fill-in-the-middle (FIM) generation. Combines
 * tokens with common prefixes and biases towards EOG when appropriate.
 *
 * Requires a reference to [LlamaVocab] to test `isEog` and retrieve special
 * tokens (EOT, EOS).
 */
class LlamaSamplerInfill(val vocab: LlamaVocab) : LlamaSampler {

    override fun name(): String = "infill"

    override fun apply(curP: LlamaTokenDataArray) {
        softmaxImpl(curP, true)

        var pTxtSum = 0.0f
        var pEogSum = 0.0f
        for (i in 0 until curP.size) {
            if (vocab.isEog(curP.data[i].id)) pEogSum += curP.data[i].p
            else pTxtSum += curP.data[i].p
        }

        if (3 * pEogSum * curP.size > pTxtSum) {
            // keep only EOG tokens
            val sizeOrg = curP.size
            curP.size = 0
            var pSum = 0.0f
            for (i in 0 until sizeOrg) {
                if (vocab.isEog(curP.data[i].id)) {
                    pSum += curP.data[i].p
                    curP.data[curP.size] = curP.data[i]
                    curP.size++
                }
            }
            for (i in 0 until curP.size) curP.data[i].p /= pSum
            return
        }

        // combine tokens with common prefix
        for (i0 in 0 until curP.size) {
            for (i1 in 0 until curP.size) {
                if (curP.data[i0].logit == Float.NEGATIVE_INFINITY) break
                if (i0 == i1 || curP.data[i1].logit == Float.NEGATIVE_INFINITY) continue

                val piece0 = vocab.tokenToPiece(curP.data[i0].id, lstrip = 0, special = false)
                val piece1 = vocab.tokenToPiece(curP.data[i1].id, lstrip = 0, special = false)

                // token i0 is a prefix of token i1
                if (piece0.isNotEmpty() && piece0.length <= piece1.length && piece1.startsWith(piece0)) {
                    var dst = i0
                    var src = i1
                    if (curP.data[i1].p > curP.data[i0].p) {
                        dst = i1
                        src = i0
                    }
                    curP.data[dst].p += curP.data[src].p
                    curP.data[src].logit = Float.NEGATIVE_INFINITY
                    curP.data[src].p = 0.0f
                }
            }
        }

        // threshold pass
        val thold = 0.2f
        val sizeOrg = curP.size
        curP.size = 0
        var pSum = 0.0f
        var nNonEog = 0
        for (i in 0 until sizeOrg) {
            val isEog = vocab.isEog(curP.data[i].id)
            if (curP.data[i].p < thold && !isEog) continue
            if (!isEog) nNonEog++
            pSum += curP.data[i].p
            curP.data[curP.size] = curP.data[i]
            curP.size++
        }

        if (nNonEog == 0) {
            curP.size = 1
            var eotId = vocab.tokenEot()
            if (eotId == LLAMA_TOKEN_NULL) eotId = vocab.tokenEos()
            curP.data[0] = SamplerTokenData(eotId, 1.0f, 1.0f)
            return
        }

        for (i in 0 until curP.size) curP.data[i].p /= pSum

        // second threshold pass
        val sizeOrg2 = curP.size
        val thold2 = 1.0f / (nNonEog + 1)
        curP.size = 0
        var pSum2 = 0.0f
        for (i in 0 until sizeOrg2) {
            val isEog = vocab.isEog(curP.data[i].id)
            if (curP.data[i].p < thold2 && !isEog) continue
            pSum2 += curP.data[i].p
            curP.data[curP.size] = curP.data[i]
            curP.size++
        }
        for (i in 0 until curP.size) curP.data[i].p /= pSum2
    }

    override fun clone(): LlamaSampler = LlamaSamplerInfill(vocab)
}

// ---------------------------------------------------------------------------
// 19. Adaptive-P sampler
// ---------------------------------------------------------------------------

/**
 * Adaptive probability sampling with an exponential moving average (EMA) of
 * the selected token's original probability, used to steer towards a [target]
 * probability band.
 *
 * Reference: <https://github.com/ggml-org/llama.cpp/pull/17927>
 */
class LlamaSamplerAdaptiveP(
    val target: Float,
    val decay: Float,
    val seed: UInt = LLAMA_DEFAULT_SEED
) : LlamaSampler {

    internal var seedCur: UInt = getRngSeed(seed).toUInt()
    private var rng = Random(seedCur.toLong())

    private val clampedDecay = decay.coerceIn(0.0f, 0.99f)
    private var weightedSum = target / (1.0f - clampedDecay)
    private var totalWeight = 1.0f / (1.0f - clampedDecay)
    private var originalProbs = FloatArray(0)
    private var pendingTokenId: LlamaToken = LLAMA_TOKEN_NULL
    private var pendingTokenIdx: Int = -1

    override fun name(): String = "adaptive-p"

    override fun apply(curP: LlamaTokenDataArray) {
        softmaxImpl(curP, false)

        if (target < 0.0f) {
            curP.selected = sampleDist(curP, rng)
            return
        }

        // store original probabilities
        originalProbs = FloatArray(curP.size) { curP.data[it].p }

        val t = target.coerceIn(0.0f, 1.0f)
        val adaptedTarget = (if (totalWeight == 0.0f) t
            else 2.0f * t - (weightedSum / totalWeight)).coerceIn(0.0f, 1.0f)

        // adaptive probability transform
        val distributionWidth = 0.3f
        val peakLogitValue = 5.0f
        val sharpness = 10.0f
        val invWidth = 1.0f / distributionWidth

        for (i in 0 until curP.size) {
            if (curP.data[i].logit == Float.NEGATIVE_INFINITY) continue
            val dist = abs((curP.data[i].p - adaptedTarget) * invWidth)
            curP.data[i].logit = peakLogitValue - sharpness * dist * dist / (1.0f + dist)
        }

        softmaxImpl(curP, false)
        val idx = sampleDist(curP, rng)
        curP.selected = idx
        pendingTokenId = curP.data[idx].id
        pendingTokenIdx = idx
    }

    override fun accept(token: LlamaToken) {
        if (pendingTokenId == token && pendingTokenId != LLAMA_TOKEN_NULL && pendingTokenIdx >= 0) {
            weightedSum = originalProbs[pendingTokenIdx] + clampedDecay * weightedSum
            totalWeight = 1.0f + clampedDecay * totalWeight
        }
        pendingTokenId = LLAMA_TOKEN_NULL
        pendingTokenIdx = -1
    }

    override fun reset() {
        weightedSum = target / (1.0f - clampedDecay)
        totalWeight = 1.0f / (1.0f - clampedDecay)
        pendingTokenId = LLAMA_TOKEN_NULL
        pendingTokenIdx = -1
        seedCur = getRngSeed(seed).toUInt()
        rng = Random(seedCur.toLong())
    }

    override fun clone(): LlamaSampler {
        val c = LlamaSamplerAdaptiveP(target, decay, seed)
        c.weightedSum = weightedSum
        c.totalWeight = totalWeight
        c.pendingTokenId = pendingTokenId
        c.pendingTokenIdx = pendingTokenIdx
        c.seedCur = seedCur
        c.rng = Random(seedCur.toLong())
        return c
    }
}

// ---------------------------------------------------------------------------
// Factory functions (mirror llama_sampler_init_* from C++)
// ---------------------------------------------------------------------------

/** Create a greedy (argmax) sampler. */
fun llamaSamplerInitGreedy(): LlamaSampler = LlamaSamplerGreedy()

/** Create a distribution sampler with the given [seed]. */
fun llamaSamplerInitDist(seed: UInt = LLAMA_DEFAULT_SEED): LlamaSampler = LlamaSamplerDist(seed)

/** Create a top-k sampler (returns identity if [k] ≤ 0). */
fun llamaSamplerInitTopK(k: Int): LlamaSampler =
    if (k <= 0) LlamaSamplerEmpty("?top-k") else LlamaSamplerTopK(k)

/** Create a top-p (nucleus) sampler (returns identity if [p] ≥ 1). */
fun llamaSamplerInitTopP(p: Float, minKeep: Int = 1): LlamaSampler =
    if (p >= 1.0f) LlamaSamplerEmpty("?top-p") else LlamaSamplerTopP(p, minKeep)

/** Create a min-p sampler (returns identity if [p] ≤ 0). */
fun llamaSamplerInitMinP(p: Float, minKeep: Int = 1): LlamaSampler =
    if (p <= 0.0f) LlamaSamplerEmpty("?min-p") else LlamaSamplerMinP(p, minKeep)

/** Create a typical-p sampler (returns identity if [p] ≥ 1). */
fun llamaSamplerInitTypical(p: Float, minKeep: Int = 1): LlamaSampler =
    if (p >= 1.0f) LlamaSamplerEmpty("?typical") else LlamaSamplerTypical(p, minKeep)

/** Create a temperature sampler (returns identity if [temp] == 1). */
fun llamaSamplerInitTemp(temp: Float): LlamaSampler =
    if (temp == 1.0f) LlamaSamplerEmpty("?temp") else LlamaSamplerTemp(temp)

/** Create a dynamic-temperature sampler with smoothing. */
fun llamaSamplerInitTempExt(temp: Float, delta: Float, exponent: Float): LlamaSampler =
    if (temp == 1.0f && delta <= 0.0f) LlamaSamplerEmpty("?temp-ext")
    else LlamaSamplerTempExt(temp, delta, exponent)

/** Create an XTC sampler (returns identity if disabled). */
fun llamaSamplerInitXtc(p: Float, t: Float, minKeep: Int = 1, seed: UInt = LLAMA_DEFAULT_SEED): LlamaSampler =
    if (p <= 0.0f || t > 0.5f) LlamaSamplerEmpty("?xtc") else LlamaSamplerXtc(p, t, minKeep, seed)

/** Create a penalties sampler (returns identity if disabled). */
fun llamaSamplerInitPenalties(
    penaltyLastN: Int,
    penaltyRepeat: Float = 1.0f,
    penaltyFreq: Float = 0.0f,
    penaltyPresent: Float = 0.0f
): LlamaSampler {
    val n = maxOf(penaltyLastN, 0)
    if (n == 0 || (penaltyRepeat == 1.0f && penaltyFreq == 0.0f && penaltyPresent == 0.0f)) {
        return LlamaSamplerEmpty("?penalties")
    }
    return LlamaSamplerPenalties(n, penaltyRepeat, penaltyFreq, penaltyPresent)
}

/** Create a top-n-sigma sampler (returns identity if [n] ≤ 0). */
fun llamaSamplerInitTopNSigma(n: Float): LlamaSampler =
    if (n <= 0.0f) LlamaSamplerEmpty("?top-n-sigma") else LlamaSamplerTopNSigma(n)

/** Create a logit-bias sampler. */
fun llamaSamplerInitLogitBias(nVocab: Int, biases: List<LlamaLogitBias>): LlamaSampler =
    if (biases.isEmpty()) LlamaSamplerEmpty("?logit-bias") else LlamaSamplerLogitBias(nVocab, biases)

/** Create a Mirostat v1 sampler. */
fun llamaSamplerInitMirostat(
    nVocab: Int,
    seed: UInt = LLAMA_DEFAULT_SEED,
    tau: Float = 5.0f,
    eta: Float = 0.1f,
    m: Int = 100
): LlamaSampler = LlamaSamplerMirostat(nVocab, seed, tau, eta, m)

/** Create a Mirostat v2 sampler. */
fun llamaSamplerInitMirostatV2(
    seed: UInt = LLAMA_DEFAULT_SEED,
    tau: Float = 5.0f,
    eta: Float = 0.1f
): LlamaSampler = LlamaSamplerMirostatV2(seed, tau, eta)

/** Create a grammar-constrained sampler. */
fun llamaSamplerInitGrammar(vocab: LlamaVocab, grammarStr: String, grammarRoot: String = "root"): LlamaSampler =
    LlamaSamplerGrammar(vocab, grammarStr, grammarRoot)

/** Create an infill (FIM) sampler. */
fun llamaSamplerInitInfill(vocab: LlamaVocab): LlamaSampler = LlamaSamplerInfill(vocab)

/** Create an adaptive-p sampler. */
fun llamaSamplerInitAdaptiveP(
    target: Float,
    decay: Float,
    seed: UInt = LLAMA_DEFAULT_SEED
): LlamaSampler = LlamaSamplerAdaptiveP(target, decay, seed)

/** Create a DRY penalty sampler (with pre-processed breakers). */
fun llamaSamplerInitDry(
    totalContextSize: Int,
    dryMultiplier: Float,
    dryBase: Float,
    dryAllowedLength: Int,
    dryPenaltyLastN: Int,
    processedBreakers: HashMap<LlamaToken, MutableList<List<LlamaToken>>> = HashMap()
): LlamaSampler {
    if (dryMultiplier == 0.0f || dryBase < 1.0f || dryPenaltyLastN == 0) {
        return LlamaSamplerEmpty("?dry")
    }
    return LlamaSamplerDry(
        totalContextSize, dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN,
        processedBreakers
    )
}

/**
 * Ported from Koboldcpp: find all token sequences that overlap with a given
 * string [str] at the boundary. This populates [tokenSequences] with
 * (head_token -> tail_token_list) entries used by the DRY sampler.
 */
private fun getOverlappingTokenSequences(
    vocab: LlamaVocab,
    str: String,
    tokenSequences: HashMap<LlamaToken, MutableList<List<LlamaToken>>>,
    maxTailLen: Int = -1
) {
    for (tokenId in 0 until vocab.nTokens()) {
        val word = vocab.detokenize(listOf(tokenId), removeSpecial = false, unparseSpecial = true)
        if (word.contains(str)) {
            val existing = tokenSequences.getOrPut(tokenId) { mutableListOf() }
            if (existing.none { it.isEmpty() }) {
                existing.add(emptyList())
            }
        } else {
            val wordLen = word.length
            val strLen = str.length
            var pos = -1
            while (true) {
                pos = word.indexOf(str[0], pos + 1)
                if (pos == -1) break
                var match = true
                var i = 1
                while (i < strLen && i + pos < wordLen) {
                    if (word[pos + i] != str[i]) { match = false; break }
                    i++
                }
                if (match) {
                    var tokenization = vocab.tokenize(str.substring(i), addSpecial = false, parseSpecial = false)
                    if (maxTailLen >= 0 && tokenization.size > maxTailLen) {
                        tokenization = tokenization.subList(0, maxTailLen)
                    }
                    val existing = tokenSequences.getOrPut(tokenId) { mutableListOf() }
                    if (existing.none { it == tokenization }) {
                        existing.add(tokenization)
                    }
                }
            }
        }
    }
}

/** Create a DRY penalty sampler from raw sequence breaker strings (matches C++ API). */
fun llamaSamplerInitDry(
    vocab: LlamaVocab,
    nCtxTrain: Int,
    dryMultiplier: Float,
    dryBase: Float,
    dryAllowedLength: Int,
    dryPenaltyLastN: Int,
    seqBreakers: List<String>
): LlamaSampler {
    val dryEnabled = dryMultiplier != 0.0f && dryBase >= 1.0f && dryPenaltyLastN != 0
    if (!dryEnabled) return LlamaSamplerEmpty("?dry")

    val maxCharLen = 40
    val maxSeqLen = 20
    val processedBreakers = HashMap<LlamaToken, MutableList<List<LlamaToken>>>()

    for (breaker in seqBreakers) {
        if (breaker.isEmpty()) continue
        val seq = if (breaker.length > maxCharLen) breaker.substring(0, maxCharLen) else breaker
        getOverlappingTokenSequences(vocab, seq, processedBreakers, maxSeqLen)
    }

    return LlamaSamplerDry(
        nCtxTrain, dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN,
        processedBreakers
    )
}

/** Create a sampler chain with the given [params]. */
fun llamaSamplerChainInit(params: LlamaSamplerChainParams = LlamaSamplerChainParams()): LlamaSamplerChain =
    LlamaSamplerChain(params)

// ---------------------------------------------------------------------------
// Empty (identity) sampler
// ---------------------------------------------------------------------------

/**
 * A identity sampler used as a skeleton when a sampler is disabled
 * (e.g. top-k with k ≤ 0). Named with a leading `?` by convention to
 * indicate it is inactive.
 */
class LlamaSamplerEmpty(private val label: String) : LlamaSampler {
    override fun name(): String = label
    override fun apply(curP: LlamaTokenDataArray) { /* identity */ }
    override fun clone(): LlamaSampler = LlamaSamplerEmpty(label)
}

// llamaSamplerName/Accept/Apply/Reset/Clone are defined in LlamaApi.kt.

// ---------------------------------------------------------------------------
// Extended llama_sampler_sample variant (uses cached sampled tensors)
// ---------------------------------------------------------------------------

private fun llamaSamplerSampleExtended(smpl: LlamaSampler, ctx: LlamaContext, idx: Int): LlamaToken {
    error("llamaSamplerSampleExtended not yet ported — depends on LlamaContext.getSampled*Ith APIs")
}

// llamaSamplerGetSeed is defined in LlamaApi.kt.

// ---------------------------------------------------------------------------
// llama_sampler_backend_support
// ---------------------------------------------------------------------------

fun llamaSamplerBackendSupport(smpl: LlamaSampler, buft: GGMLBackendBufferType): Boolean {
    // Depends on ggmlNewTensor1d(ctx, type, n) and GGMLCGraph node iteration APIs not yet ported.
    return true
}

// llamaPerfSampler/Print/Reset are defined in LlamaApi.kt.

// ---------------------------------------------------------------------------
// llama_sampler_init_dry_testing (wrapper for tests)
// ---------------------------------------------------------------------------

fun llamaSamplerInitDryTesting(
    contextSize: Int,
    dryMultiplier: Float,
    dryBase: Float,
    dryAllowedLength: Int,
    dryPenaltyLastN: Int,
    seqBreakers: List<List<LlamaToken>>
): LlamaSampler {
    val processedBreakers = HashMap<LlamaToken, MutableList<List<LlamaToken>>>()
    for (breaker in seqBreakers) {
        if (breaker.isEmpty()) continue
        val headToken = breaker[0]
        val tailTokens = if (breaker.size > 1) breaker.subList(1, breaker.size) else emptyList()
        processedBreakers.getOrPut(headToken) { mutableListOf() }.add(tailTokens)
    }
    return llamaSamplerInitDry(
        contextSize, dryMultiplier, dryBase, dryAllowedLength, dryPenaltyLastN,
        processedBreakers
    )
}

// ---------------------------------------------------------------------------
// Grammar lazy init variants
// ---------------------------------------------------------------------------

fun llamaSamplerInitGrammarLazy(
    vocab: LlamaVocab,
    grammarStr: String,
    grammarRoot: String,
    triggerWords: List<String>,
    triggerTokens: List<LlamaToken>
): LlamaSampler = LlamaSamplerGrammar(vocab, grammarStr, grammarRoot)

fun llamaSamplerInitGrammarLazyPatterns(
    vocab: LlamaVocab,
    grammarStr: String,
    grammarRoot: String,
    triggerPatterns: List<String>,
    triggerTokens: List<LlamaToken>
): LlamaSampler = LlamaSamplerGrammar(vocab, grammarStr, grammarRoot)

// ---------------------------------------------------------------------------
// Backend temp sampling (shared by temp and temp-ext)
// ---------------------------------------------------------------------------

private fun backendTempSampling(
    ctx: GGMLContext,
    gf: GGMLCGraph,
    data: LlamaSamplerData,
    temp: Float
) {
    if (temp <= 0.0f) {
        val maxIdx = ggmlArgmax(ctx, data.logits!!)
        if (data.candidates != null) {
            val candidatesRows = ggmlReshape2d(ctx, data.candidates!!, 1, data.candidates!!.ne[0])
            data.candidates = ggmlGetRows(ctx, candidatesRows, maxIdx)
        } else {
            data.candidates = maxIdx
        }
        val logitsRows = ggmlReshape2d(ctx, data.logits!!, 1, data.logits!!.ne[0])
        data.logits = ggmlGetRows(ctx, logitsRows, maxIdx)
        return
    }
    data.logits = ggmlScale(ctx, data.logits!!, 1.0f / temp)
}
