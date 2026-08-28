// port-lint: source llama.cpp/src/llama-kv-cells.h
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.math.max

// =============================================================================
// LlamaKvCellExt – per-cell 2D spatial metadata (M-RoPE)
// Ported from: llama_kv_cell_ext in llama-kv-cells.h
// =============================================================================

/**
 * Extended cell metadata for 2D spatial positions, typically used for M-RoPE.
 *
 * @property x  Horizontal spatial position.
 * @property y  Vertical spatial position.
 */
data class LlamaKvCellExt(
    var x: LlamaPos = 0,
    var y: LlamaPos = 0,
) {
    /** Return `true` if this 2D position is greater than ([ox], [oy]). */
    fun is2dGt(ox: LlamaPos, oy: LlamaPos): Boolean =
        (y > oy) || (y == oy && x > ox)

    fun reset() {
        x = 0
        y = 0
    }
}

// =============================================================================
// LlamaKvCells – ring-buffer cell management with sequence tracking
// Ported from: llama_kv_cells in llama-kv-cells.h
// =============================================================================

/**
 * Manages metadata for KV cache cells that can belong to multiple sequences.
 *
 * This is a faithful port of `llama_kv_cells` from `llama-kv-cells.h`.
 * It tracks positions, sequence membership (via bitsets), accumulated shifts,
 * and maintains per-sequence position-count maps for O(1) min/max queries.
 *
 * The C++ implementation uses `std::bitset<LLAMA_MAX_SEQ>` for per-cell
 * sequence membership. Here we use `BooleanArray(LLAMA_MAX_SEQ)`.
 *
 * The C++ implementation uses `std::set<uint32_t>` for tracking used cell
 * indices. Here we use a `TreeSet` (sorted set) for the same O(log n)
 * insert/erase/min/max semantics.
 *
 * The C++ implementation uses `std::map<llama_pos, int>` for per-sequence
 * position reference counts (`seq_pos`). Here we use a `MutableMap` for the
 * same ordered-map semantics with O(log n) first/last key access.
 */
class LlamaKvCells {

    // -- storage arrays -------------------------------------------------------
    private var pos: IntArray = IntArray(0)
    private var ext: Array<LlamaKvCellExt> = emptyArray()
    private var shift: IntArray = IntArray(0)

    // per-cell sequence bitsets: seq[i][s] == true means cell i contains seq s
    private var seq: Array<BooleanArray> = emptyArray()

    // sorted set of used cell indices (mirrors std::set<uint32_t> in C++)
    private var used = mutableSetOf<Int>()

    // has_shift flag — set to true whenever pos_add or pos_div modifies shifts
    private var hasShift: Boolean = false

    // per-sequence position reference counts:
    // seqPos[s][p] == count of cells with position p for sequence s
    // uses sorted map so min/max can be computed
    private val seqPos: Array<MutableMap<Int, Int>> =
        Array(LLAMA_MAX_SEQ) { mutableMapOf() }

    // -- lifecycle ------------------------------------------------------------

    fun reset() {
        for (i in pos.indices) {
            pos[i] = -1
            ext[i].reset()
            shift[i] = 0
            seq[i].fill(false)
        }
        hasShift = false
        used.clear()
        for (s in 0 until LLAMA_MAX_SEQ) {
            seqPos[s].clear()
        }
    }

    fun resetShift() {
        hasShift = false
        shift.fill(0)
    }

    fun size(): Int = pos.size

    fun resize(n: Int) {
        pos = IntArray(n) { -1 }
        ext = Array(n) { LlamaKvCellExt() }
        shift = IntArray(n)
        seq = Array(n) { BooleanArray(LLAMA_MAX_SEQ) }
        reset()
    }

    // -- queries --------------------------------------------------------------

    fun isEmpty(i: Int): Boolean {
        check(i < pos.size)
        return pos[i] == -1
    }

    fun getUsed(): Int = used.size

    /** Index of the first used cell, or 0 if none. */
    fun usedMin(): Int = if (used.isEmpty()) 0 else used.first()

    /** One past the index of the last used cell, or 0 if none. */
    fun usedMaxP1(): Int = if (used.isEmpty()) 0 else used.last() + 1

    fun getHasShift(): Boolean = hasShift

    // -- copy / restore (for prepare/rollback) --------------------------------

    /**
     * Copy the state of cells `[i, i + n)` into a new [LlamaKvCells].
     * The caller must ensure `shift` is zero for all copied cells.
     */
    fun cp(i: Int, n: Int): LlamaKvCells {
        check(i + n <= pos.size)
        val res = LlamaKvCells()
        res.resize(n)
        for (j in 0 until n) {
            val idx = i + j
            res.pos[j] = pos[idx]
            res.ext[j] = ext[idx].copy()
            seq[idx].copyInto(res.seq[j])
            check(shift[idx] == 0)
        }
        // rebuild res.used and res.seqPos from copied data
        for (j in 0 until n) {
            if (res.pos[j] != -1) {
                res.used.add(j)
                res.seqPosAdd(j)
            }
        }
        return res
    }

    /**
     * Copy the state of cells at [idxs] into a new [LlamaKvCells].
     */
    fun cp(idxs: List<Int>): LlamaKvCells {
        val res = LlamaKvCells()
        res.resize(idxs.size)
        for (j in idxs.indices) {
            val idx = idxs[j]
            res.pos[j] = pos[idx]
            res.ext[j] = ext[idx].copy()
            seq[idx].copyInto(res.seq[j])
            check(shift[idx] == 0)
        }
        for (j in idxs.indices) {
            if (res.pos[j] != -1) {
                res.used.add(j)
                res.seqPosAdd(j)
            }
        }
        return res
    }

    /**
     * Restore the state of cells `[i, i + other.size())` from [other].
     */
    fun set(i: Int, other: LlamaKvCells) {
        check(i + other.pos.size <= pos.size)
        for (j in other.pos.indices) {
            val idx = i + j
            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.add(idx)
            }
            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.remove(idx)
            }
            if (pos[idx] != -1) {
                seqPosRm(idx)
            }
            pos[idx] = other.pos[j]
            ext[idx] = other.ext[j].copy()
            other.seq[j].copyInto(seq[idx])
            if (pos[idx] != -1) {
                seqPosAdd(idx)
            }
            check(shift[idx] == 0)
        }
    }

    /**
     * Restore the state of cells at positions [idxs] from [other].
     * `idxs.size` must equal `other.size()`.
     */
    fun set(idxs: List<Int>, other: LlamaKvCells) {
        check(idxs.size == other.pos.size)
        for (j in other.pos.indices) {
            val idx = idxs[j]
            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.add(idx)
            }
            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.remove(idx)
            }
            if (pos[idx] != -1) {
                seqPosRm(idx)
            }
            pos[idx] = other.pos[j]
            ext[idx] = other.ext[j].copy()
            other.seq[j].copyInto(seq[idx])
            if (pos[idx] != -1) {
                seqPosAdd(idx)
            }
            check(shift[idx] == 0)
        }
    }

    // -- cell removal ---------------------------------------------------------

    /** Remove a non-empty cell (clear all sequence membership + position). */
    fun rm(i: Int) {
        check(i < pos.size)
        check(pos[i] != -1)
        seqPosRm(i)
        seq[i].fill(false)
        pos[i] = -1
        ext[i].reset()
        shift[i] = 0
        used.remove(i)
    }

    // -- sequence operations --------------------------------------------------

    /**
     * Remove [seqId] from cell [i]. Returns `true` if the cell becomes empty.
     * Note: call only if the cell has [seqId].
     */
    fun seqRm(i: Int, seqId: LlamaSeqId): Boolean {
        check(i < pos.size)
        check(seq[i][seqId]) { "cell $i does not contain seq $seqId" }
        check(pos[i] != -1)
        check(seqId >= 0)

        seq[i][seqId] = false
        seqPosDec(seqId, pos[i])

        if (seq[i].none()) {
            pos[i] = -1
            ext[i].reset()
            shift[i] = 0
            used.remove(i)
            return true
        }
        return false
    }

    /**
     * Keep only [seqId] in cell [i], removing all other sequences.
     * Returns `true` if the cell becomes empty (i.e. it did not contain [seqId]).
     */
    fun seqKeep(i: Int, seqId: LlamaSeqId): Boolean {
        check(i < pos.size)

        if (seq[i][seqId]) {
            // cell has the target sequence — remove all others, keep only seqId
            seqPosRm(i)
            seq[i].fill(false)
            seq[i][seqId] = true
            seqPosInc(seqId, pos[i])
            return false
        }

        if (seq[i].any()) {
            // cell has sequences, but not the target — clear it
            seqPosRm(i)
            seq[i].fill(false)
            pos[i] = -1
            ext[i].reset()
            shift[i] = 0
            used.remove(i)
            return true
        }

        // cell is already empty
        check(pos[i] == -1)
        return false
    }

    /** Number of different sequences in cell [i]. */
    fun seqCount(i: Int): Int {
        check(i < pos.size)
        check(pos[i] != -1)
        return seq[i].count { it }
    }

    /** Check if cell [i] contains [seqId]. */
    fun seqHas(i: Int, seqId: LlamaSeqId): Boolean {
        check(i < pos.size)
        check(seqId >= 0)
        return seq[i][seqId]
    }

    /**
     * Add [seqId] to a non-empty cell [i].
     * Note: call only if the cell is not empty and does not already have [seqId].
     */
    fun seqAdd(i: Int, seqId: LlamaSeqId) {
        check(i < pos.size)
        check(pos[i] != -1)
        check(!seq[i][seqId]) { "cell $i already contains seq $seqId" }
        seq[i][seqId] = true
        seqPosInc(seqId, pos[i])
    }

    /**
     * Return the single sequence ID of cell [i].
     * Note: call only when the cell has exactly one sequence.
     */
    fun seqGet(i: Int): LlamaSeqId {
        check(seq[i].count { it } == 1)
        for (s in 0 until LLAMA_MAX_SEQ) {
            if (seq[i][s]) return s
        }
        return -1
    }

    // -- position queries via seq_pos maps ------------------------------------

    /**
     * Minimum position of [seqId] currently in any cell, or -1 if not present.
     */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos {
        check(seqId >= 0 && seqId < LLAMA_MAX_SEQ)
        val map = seqPos[seqId]
        if (map.isEmpty()) return -1
        val minKey = map.keys.min()
        check((map[minKey] ?: 0) > 0)
        return minKey
    }

    /**
     * Maximum position of [seqId] currently in any cell, or -1 if not present.
     */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos {
        check(seqId >= 0 && seqId < LLAMA_MAX_SEQ)
        val map = seqPos[seqId]
        if (map.isEmpty()) return -1
        val maxKey = map.keys.max()
        check((map[maxKey] ?: 0) > 0)
        return maxKey
    }

    // -- position accessors ---------------------------------------------------

    /** Get position of a non-empty cell. */
    fun posGet(i: Int): LlamaPos {
        check(i < pos.size)
        check(pos[i] != -1)
        return pos[i]
    }

    fun extGet(i: Int): LlamaKvCellExt {
        check(i < pos.size)
        check(pos[i] != -1)
        return ext[i]
    }

    /** Get accumulated shift of a non-empty cell. */
    fun getShift(i: Int): LlamaPos {
        check(i < pos.size)
        check(pos[i] != -1)
        return shift[i]
    }

    /** Check if cell [i] is non-empty and within `[p0, p1)`. */
    fun posIn(i: Int, p0: LlamaPos, p1: LlamaPos): Boolean {
        check(i < pos.size)
        return pos[i] >= p0 && pos[i] < p1
    }

    /**
     * Set the position of an empty cell. Does not modify [hasShift].
     * Note: call only if the cell is empty.
     */
    fun posSet(i: Int, p: LlamaPos) {
        check(i < pos.size)
        check(pos[i] == -1)
        check(seq[i].none())
        pos[i] = p
        used.add(i)
    }

    fun extSet(i: Int, e: LlamaKvCellExt) {
        check(i < ext.size)
        ext[i] = e.copy()
    }

    /**
     * Add [d] to the position of a non-empty cell.
     * Sets [hasShift] to `true`.
     * Returns `true` if the cell becomes empty (position went negative).
     */
    fun posAdd(i: Int, d: LlamaPos): Boolean {
        check(i < pos.size)
        check(pos[i] != -1)

        seqPosRm(i)

        pos[i] += d
        shift[i] += d
        hasShift = true

        if (pos[i] < 0) {
            seq[i].fill(false)
            pos[i] = -1
            shift[i] = 0
            used.remove(i)
            return true
        }

        seqPosAdd(i)
        return false
    }

    /**
     * Divide the position of a non-empty cell by [d].
     * Sets [hasShift] to `true`.
     */
    fun posDiv(i: Int, d: Int) {
        check(i < pos.size)
        check(pos[i] != -1)

        val pOld = pos[i]
        seqPosRm(i)

        pos[i] /= d
        shift[i] += pOld - pos[i]

        seqPosAdd(i)
        hasShift = true
    }

    // -- private helpers for seq_pos bookkeeping ------------------------------

    /** Decrement the count of position [p] for sequence [s]. Remove entry if zero. */
    private fun seqPosDec(s: LlamaSeqId, p: LlamaPos) {
        val map = seqPos[s]
        val count = map[p] ?: error("seqPos[$s] has no entry for position $p")
        if (count <= 1) {
            map.remove(p)
        } else {
            map[p] = count - 1
        }
    }

    /** Increment the count of position [p] for sequence [s]. */
    private fun seqPosInc(s: LlamaSeqId, p: LlamaPos) {
        val map = seqPos[s]
        map[p] = (map[p] ?: 0) + 1
    }

    /** Remove all sequence-position entries for cell [i]. */
    private fun seqPosRm(i: Int) {
        for (s in 0 until LLAMA_MAX_SEQ) {
            if (seq[i][s]) {
                seqPosDec(s, pos[i])
            }
        }
    }

    /** Add sequence-position entries for cell [i]. */
    private fun seqPosAdd(i: Int) {
        for (s in 0 until LLAMA_MAX_SEQ) {
            if (seq[i][s]) {
                seqPosInc(s, pos[i])
            }
        }
    }
}

// -- extension: BooleanArray.none() / .any() ----------------------------------
// These intentionally shadow the stdlib versions. The stdlib's parameterless
// none()/any() check array *emptiness*. These check whether any *value* is true.

/** Returns `true` if no element is `true`. */
private fun BooleanArray.none(): Boolean {
    for (v in this) if (v) return false
    return true
}

/** Returns `true` if any element is `true`. */
private fun BooleanArray.any(): Boolean {
    for (v in this) if (v) return true
    return false
}
