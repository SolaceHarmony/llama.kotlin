// port-lint: source llama.cpp/src/llama-memory-recurrent.h llama.cpp/src/llama-memory-recurrent.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// LlamaMemoryRecurrent – recurrent state cache for SSM / RWKV models
// Ported from: llama-memory-recurrent.h / llama-memory-recurrent.cpp
// =============================================================================

// LayerFilterCallback is defined in LlamaMemory.kt

// ---------------------------------------------------------------------------
// MemCell – per-slot metadata
// ---------------------------------------------------------------------------

/**
 * Metadata for a single recurrent-memory cell.
 *
 * Port of `llama_memory_recurrent::mem_cell`.
 *
 * @property pos   Token position, or -1 if empty.
 * @property src   Source cell index for state copying (general).
 * @property src0  Source cell index used only during input setting.
 * @property tail  Index of the tail cell in the sequence (for ring-buffer logic).
 */
data class RecurrentMemCell(
    var pos: LlamaPos = -1,
    var src: Int = -1,
    var src0: Int = -1,
    var tail: Int = -1,
) {
    /** Sequence ids that own this cell. */
    val seqId: MutableSet<LlamaSeqId> = mutableSetOf()

    /** `true` if [id] is among the occupying sequences. */
    fun hasSeqId(id: LlamaSeqId): Boolean = id in seqId

    /** `true` when the cell is unoccupied. */
    fun isEmpty(): Boolean = seqId.isEmpty()

    /** `true` when this cell has exactly the same sequence membership as [other]. */
    fun isSameSeq(other: RecurrentMemCell): Boolean = seqId == other.seqId
}

// ---------------------------------------------------------------------------
// LlamaMemoryRecurrent
// ---------------------------------------------------------------------------

/**
 * Recurrent-state memory manager for SSM and RWKV model families.
 *
 * This class is the Kotlin port of `class llama_memory_recurrent` from
 * `llama-memory-recurrent.h`. It manages a fixed pool of [RecurrentMemCell]
 * entries and per-layer r/s state tensors that are re-used across inference
 * steps.
 *
 * @param hparams    Model hyper-parameters (layer count, state sizes, etc.).
 * @param typeR      GGML data type for the "r" (recurrent) state tensors.
 * @param typeS      GGML data type for the "s" (SSM hidden) state tensors.
 * @param offload    Whether to offload state tensors to accelerator memory.
 * @param memSize    Total number of recurrent-memory cells to allocate.
 * @param nSeqMax    Maximum number of concurrent sequences.
 * @param filter     Per-layer filter deciding which layers get state buffers.
 */
class LlamaMemoryRecurrent(
    private val hparams: LlamaHparams,
    private val typeR: GGMLType = GGMLType.F32,
    private val typeS: GGMLType = GGMLType.F32,
    private val offload: Boolean = false,
    memSize: Int,
    private val nSeqMax: Int = 1,
    private val filter: LayerFilterCallback = { true },
) {
    /** Location where the next batch will be placed (see [findSlot]). */
    var head: Int = 0

    /** Total number of cells. */
    var size: Int = memSize

    /** Number of currently used cells (≥ 1 seq_id). */
    var used: Int = 0

    /** Number of active cells — computed before each graph build. */
    var n: Int = 0

    /** Index of the first zero-initialised state, or -1 if none. */
    var rsZ: Int = -1

    /** Per-slot metadata. */
    val cells: MutableList<RecurrentMemCell> =
        MutableList(memSize) { RecurrentMemCell() }

    /** Per-layer "r" state tensors (recurrent hidden state). */
    val rL: MutableList<GGMLTensor?> = MutableList(hparams.nLayer) { null }

    /** Per-layer "s" state tensors (SSM hidden state). */
    val sL: MutableList<GGMLTensor?> = MutableList(hparams.nLayer) { null }

    // -----------------------------------------------------------------------
    // Memory interface methods (mirror llama_memory_i)
    // -----------------------------------------------------------------------

    /**
     * Initialise a batch-processing context.
     *
     * Splits the batch into micro-batches (by sequence when [embdAll], otherwise
     * sequential-equal split), verifies they all fit via [prepare], then returns
     * a context for iterating over them.
     *
     * Port of `llama_memory_recurrent::init_batch()`.
     */
    fun initBatch(
        balloc: LlamaBatchAllocr,
        nUbatch: Int,
        embdAll: Boolean,
    ): LlamaMemoryRecurrentContext {
        do {
            balloc.splitReset()

            val ubatches = mutableListOf<LlamaUBatch>()
            while (true) {
                val ubatch: LlamaUBatchInternal = if (embdAll) {
                    balloc.splitSeq(nUbatch)
                } else {
                    balloc.splitEqual(nUbatch, true)
                }

                if (ubatch.nTokens == 0) break

                ubatches.add(ubatch.toPublic())
            }

            if (balloc.getNUsed() < balloc.getNTokens()) {
                break
            }

            if (!prepare(ubatches)) {
                break
            }

            return LlamaMemoryRecurrentContext.batch(this, ubatches)
        } while (false)

        return LlamaMemoryRecurrentContext.error(LlamaMemoryStatus.FAILED_PREPARE)
    }

    /**
     * Initialise a full-cache context (used for state save/load).
     *
     * Port of `llama_memory_recurrent::init_full()`.
     */
    fun initFull(): LlamaMemoryRecurrentContext {
        return LlamaMemoryRecurrentContext.full(this)
    }

    /**
     * Initialise an update context (defrag / optimise).
     *
     * Recurrent caches never perform updates; this always returns NO_UPDATE.
     *
     * Port of `llama_memory_recurrent::init_update()`.
     */
    fun initUpdate(
        lctx: LlamaContext,
        optimize: Boolean,
    ): LlamaMemoryRecurrentContext {
        return LlamaMemoryRecurrentContext.error(LlamaMemoryStatus.NO_UPDATE)
    }

    /**
     * Clear all cells (and optionally the underlying tensor data).
     *
     * Port of `llama_memory_recurrent::clear()`.
     */
    fun clear(data: Boolean) {
        for (i in 0 until size) {
            cells[i].pos = -1
            cells[i].src = -1
            cells[i].src0 = -1
            cells[i].tail = -1
            cells[i].seqId.clear()
        }
        head = 0
        used = 0
        n = 0
        rsZ = -1
        // LATER: if `data`, zero-fill the r_l / s_l tensor backing buffers via backend
    }

    // -- sequence operations ------------------------------------------------

    /**
     * Remove positions `[p0, p1)` for [seqId]. Pass `p0 < 0` to remove all.
     *
     * For recurrent state models (Mamba, RWKV), partial erasure at the end of
     * a sequence is invalid because state isn't preserved for earlier tokens.
     *
     * Port of `llama_memory_recurrent::seq_rm()`.
     *
     * @return `true` on success.
     */
    fun seqRm(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos): Boolean {
        var effP0 = if (p0 < 0) 0 else p0
        var effP1 = if (p1 < 0) Int.MAX_VALUE else p1

        var newHead = size

        if (seqId >= size) {
            return false
        }

        if (seqId >= 0) {
            val tailId = cells[seqId].tail
            if (tailId >= 0) {
                val cell = cells[tailId]
                // partial intersection is invalid if it includes the final pos
                if (effP0 > 0 && effP0 <= cell.pos && effP1 > cell.pos) {
                    return false
                }
                // invalidate tails which will be cleared
                if (effP0 <= cell.pos && cell.pos < effP1) {
                    cells[seqId].tail = -1
                }
            }
        } else {
            // seq_id is negative, then the range should include everything or nothing
            if (effP0 != effP1 && (effP0 != 0 || effP1 != Int.MAX_VALUE)) {
                return false
            }
        }

        for (i in 0 until size) {
            if (cells[i].pos >= effP0 && cells[i].pos < effP1) {
                if (seqId < 0) {
                    cells[i].seqId.clear()
                } else if (cells[i].hasSeqId(seqId)) {
                    cells[i].seqId.remove(seqId)
                } else {
                    continue
                }
                if (cells[i].isEmpty()) {
                    if (cells[i].pos >= 0) {
                        used--
                    }
                    cells[i].pos = -1
                    cells[i].src = -1
                    if (newHead == size) {
                        newHead = i
                    }
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (newHead != size && newHead < head) {
            head = newHead
        }

        return true
    }

    /**
     * Copy sequence [seqIdSrc] to [seqIdDst] for positions `[p0, p1)`.
     *
     * Port of `llama_memory_recurrent::seq_cp()`.
     */
    fun seqCp(seqIdSrc: LlamaSeqId, seqIdDst: LlamaSeqId, p0: LlamaPos, p1: LlamaPos) {
        if (seqIdSrc == seqIdDst) return

        if (seqIdDst < size && seqIdSrc < size) {
            val tailSrc = cells[seqIdSrc]
            val tailDst = cells[seqIdDst]
            if (tailDst.tail >= 0) {
                // clear destination seq_id if it wasn't empty
                val cellDst = cells[tailDst.tail]
                cellDst.seqId.remove(seqIdDst)
                tailDst.tail = -1
                if (cellDst.seqId.isEmpty()) {
                    cellDst.pos = -1
                    cellDst.src = -1
                    used -= 1
                }
            }
            if (tailSrc.tail >= 0) {
                val cellSrc = cells[tailSrc.tail]
                cellSrc.seqId.add(seqIdDst)
                tailDst.tail = tailSrc.tail
            }
        }
    }

    /**
     * Keep only [seqId]; remove all other sequences from every cell.
     *
     * Port of `llama_memory_recurrent::seq_keep()`.
     */
    fun seqKeep(seqId: LlamaSeqId) {
        var newHead = size

        for (i in 0 until size) {
            if (i != seqId) {
                cells[i].tail = -1
            }

            if (!cells[i].hasSeqId(seqId)) {
                if (cells[i].pos >= 0) {
                    used--
                }
                cells[i].pos = -1
                cells[i].src = -1
                cells[i].seqId.clear()

                if (newHead == size) {
                    newHead = i
                }
            } else {
                cells[i].seqId.clear()
                cells[i].seqId.add(seqId)
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (newHead != size && newHead < head) {
            head = newHead
        }
    }

    /**
     * Shift positions of [seqId] in range `[p0, p1)` by [shift].
     *
     * For Mamba-like or RWKV models, only the pos needs to be shifted.
     *
     * Port of `llama_memory_recurrent::seq_add()`.
     */
    fun seqAdd(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, shift: LlamaPos) {
        if (shift == 0) return

        val effP0 = if (p0 < 0) 0 else p0
        val effP1 = if (p1 < 0) Int.MAX_VALUE else p1

        if (effP0 == effP1) return

        if (seqId in 0 until size) {
            val tailId = cells[seqId].tail
            if (tailId >= 0) {
                val cell = cells[tailId]
                if (cell.hasSeqId(seqId) && effP0 <= cell.pos && cell.pos < effP1) {
                    cell.pos += shift
                }
            }
        }
    }

    /**
     * Integer-divide positions of [seqId] in range `[p0, p1)` by [d].
     *
     * Port of `llama_memory_recurrent::seq_div()`.
     */
    fun seqDiv(seqId: LlamaSeqId, p0: LlamaPos, p1: LlamaPos, d: Int) {
        if (d == 1) return

        val effP0 = if (p0 < 0) 0 else p0
        val effP1 = if (p1 < 0) Int.MAX_VALUE else p1

        if (effP0 == effP1) return

        if (seqId in 0 until size) {
            val tailId = cells[seqId].tail
            if (tailId >= 0) {
                val cell = cells[tailId]
                if (cell.hasSeqId(seqId) && effP0 <= cell.pos && cell.pos < effP1) {
                    cell.pos /= d
                }
            }
        }
    }

    /** Minimum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMin(seqId: LlamaSeqId): LlamaPos {
        var result = Int.MAX_VALUE
        for (i in 0 until size) {
            if (cells[i].hasSeqId(seqId)) {
                result = minOf(result, cells[i].pos)
            }
        }
        return if (result == Int.MAX_VALUE) -1 else result
    }

    /** Maximum position among cells belonging to [seqId], or -1 if none. */
    fun seqPosMax(seqId: LlamaSeqId): LlamaPos {
        var result = -1
        for (i in 0 until size) {
            if (cells[i].hasSeqId(seqId)) {
                result = maxOf(result, cells[i].pos)
            }
        }
        return result
    }

    // -- slot management ---------------------------------------------------

    /**
     * Prepare the memory for the given sequence of ubatches.
     *
     * Saves the current state, attempts to [findSlot] for each ubatch, then
     * restores the original state regardless of outcome.
     *
     * Port of `llama_memory_recurrent::prepare()`.
     *
     * @return `true` when all ubatches can be accommodated.
     */
    fun prepare(ubatches: List<LlamaUBatch>): Boolean {
        // snapshot the full state because it is very small for this type of cache
        val orgCells = cells.map { cell ->
            RecurrentMemCell(cell.pos, cell.src, cell.src0, cell.tail).also {
                it.seqId.addAll(cell.seqId)
            }
        }
        val orgUsed = used
        val orgHead = head

        var success = true
        for (ubatch in ubatches) {
            if (!findSlot(ubatch)) {
                success = false
                break
            }
        }

        // restore the original state
        for (i in 0 until size) {
            cells[i].pos = orgCells[i].pos
            cells[i].src = orgCells[i].src
            cells[i].src0 = orgCells[i].src0
            cells[i].tail = orgCells[i].tail
            cells[i].seqId.clear()
            cells[i].seqId.addAll(orgCells[i].seqId)
        }
        used = orgUsed
        head = orgHead

        return success
    }

    /**
     * Find a contiguous slot of memory cells and emplace the ubatch there.
     *
     * For recurrent state architectures (Mamba, RWKV), each cache cell stores
     * the state for a whole sequence. A slot must always be contiguous.
     *
     * Port of `llama_memory_recurrent::find_slot()`.
     *
     * @return `true` on success; `false` when no contiguous slot is available.
     */
    fun findSlot(ubatch: LlamaUBatch): Boolean {
        val nSeqTokens = ubatch.nSeqTokens
        val nSeqs = ubatch.nSeqs

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head > used + 2 * nSeqs) {
            head = 0
        }

        // can only process batches with an equal number of new tokens in each sequence
        require(ubatch.equalSeqs) { "Recurrent cache requires equal-length sequences in ubatch" }

        var min = size - 1
        var max = 0

        // everything should fit if all seq_ids are smaller than the max
        for (s in 0 until nSeqs) {
            val i = s * nSeqTokens // first token of sequence set s
            val nSeqIdForToken = ubatch.nSeqId?.get(i) ?: ubatch.seqId?.get(i)?.size ?: 0

            for (j in 0 until nSeqIdForToken) {
                val sid = ubatch.seqId?.get(i)?.get(j) ?: continue

                if (sid < 0 || sid >= size) {
                    llamaLogError("findSlot: seq_id=$sid >= n_seq_max=$nSeqMax Try using a bigger --parallel value\n")
                    return false
                }
                if (j > 0) {
                    val seq = cells[sid]
                    if (seq.tail >= 0) {
                        val cell = cells[seq.tail]
                        // clear cells from seq_ids that become shared
                        cell.seqId.remove(sid)
                        seq.tail = -1
                        if (cell.seqId.isEmpty()) {
                            cell.pos = -1
                            cell.src = -1
                            used -= 1
                        }
                    }
                }
            }
        }

        // find next empty cell
        var nextEmptyCell = head
        for (i in 0 until size) {
            if (nextEmptyCell >= size) { nextEmptyCell -= size }
            val cell = cells[nextEmptyCell]
            if (cell.isEmpty()) break
            nextEmptyCell += 1
        }

        // find usable cell range
        for (s in 0 until nSeqs) {
            val i = s * nSeqTokens
            val sid = ubatch.seqId?.get(i)?.get(0) ?: continue
            val seqMeta = cells[sid]
            var hasCell = false
            if (seqMeta.tail >= 0) {
                val cell = cells[seqMeta.tail]
                check(cell.hasSeqId(sid))
                // does this seq_id "own" the cell?
                if (cell.seqId.size == 1) { hasCell = true }
            }
            if (!hasCell) {
                val emptyCell = cells[nextEmptyCell]
                check(emptyCell.isEmpty()) { "Expected empty cell at $nextEmptyCell" }
                // copy old tail into the empty cell
                if (seqMeta.tail >= 0) {
                    val origCell = cells[seqMeta.tail]
                    emptyCell.pos = origCell.pos
                    emptyCell.src = origCell.src
                    origCell.seqId.remove(sid)
                    emptyCell.seqId.add(sid) // will be overwritten
                    check(!origCell.isEmpty()) { "Original cell should have at least one remaining seq_id" }
                }
                seqMeta.tail = nextEmptyCell
                // find next empty cell
                if (s + 1 < nSeqs) {
                    for (j in 0 until size) {
                        nextEmptyCell += 1
                        if (nextEmptyCell >= size) { nextEmptyCell -= size }
                        val cell = cells[nextEmptyCell]
                        if (cell.isEmpty()) break
                    }
                }
            }
            if (min > seqMeta.tail) { min = seqMeta.tail }
            if (max < seqMeta.tail) { max = seqMeta.tail }
        }

        // gather and re-order
        for (s in 0 until nSeqs) {
            val i = s * nSeqTokens
            val dstId = s + min
            val sid = ubatch.seqId?.get(i)?.get(0) ?: continue
            val srcId = cells[sid].tail
            if (dstId != srcId) {
                val dstCell = cells[dstId]
                val srcCell = cells[srcId]

                // swap pos, src, seq_id
                val tmpPos = dstCell.pos; dstCell.pos = srcCell.pos; srcCell.pos = tmpPos
                val tmpSrc = dstCell.src; dstCell.src = srcCell.src; srcCell.src = tmpSrc
                val tmpSeqId = dstCell.seqId.toMutableSet()
                dstCell.seqId.clear(); dstCell.seqId.addAll(srcCell.seqId)
                srcCell.seqId.clear(); srcCell.seqId.addAll(tmpSeqId)

                // swap tails
                for (j in 0 until size) {
                    val tail = cells[j].tail
                    if (tail == srcId) {
                        cells[j].tail = dstId
                    } else if (tail == dstId) {
                        cells[j].tail = srcId
                    }
                }
            }
        }

        // update the pos of the used seqs
        for (s in 0 until nSeqs) {
            val i = s * nSeqTokens
            val lastPos = ubatch.pos?.get(i + nSeqTokens - 1) ?: 0
            val cellId = s + min
            val cell = cells[cellId]

            if (cell.pos >= 0 && lastPos != cell.pos + nSeqTokens) {
                llamaLogWarn(
                    "findSlot: non-consecutive token position $lastPos after ${cell.pos} " +
                    "for sequence ${ubatch.seqId?.get(i)?.get(0)} with $nSeqTokens new tokens\n"
                )
            }
            cell.pos = lastPos
            cell.seqId.clear()
            val nSeqIdForToken = ubatch.nSeqId?.get(i) ?: ubatch.seqId?.get(i)?.size ?: 0
            for (j in 0 until nSeqIdForToken) {
                val sid = ubatch.seqId?.get(i)?.get(j) ?: continue
                cell.seqId.add(sid)
                cells[sid].tail = cellId
            }
        }

        // Find first cell without src refs, to use as the zero-ed state
        run {
            val refcounts = IntArray(size)
            for (i in 0 until size) {
                val src = cells[i].src
                if (src >= 0) {
                    refcounts[src] += 1
                }
            }

            rsZ = -1
            for (i in min..max) {
                if (refcounts[i] == 0) {
                    rsZ = i
                    break
                }
            }

            for (i in min..max) {
                if (cells[i].src < 0) {
                    check(rsZ >= 0) { "No zero-state cell found but one is needed" }
                    cells[i].src0 = rsZ
                } else {
                    // Stage the source ids for all used cells
                    cells[i].src0 = cells[i].src
                }
                cells[i].src = i // avoid moving or clearing twice
            }
        }

        // allow getting the range of used cells, from head to head + n
        head = min
        n = max - min + 1
        used = cells.count { !it.isEmpty() }

        // sanity check
        return n >= nSeqs
    }

    /**
     * Whether this memory implementation supports K-shift.
     *
     * Recurrent models support trivial pos shifting.
     *
     * Port of `llama_memory_recurrent::get_can_shift()`.
     */
    fun getCanShift(): Boolean = true

    // -- state serialisation -----------------------------------------------

    /**
     * Serialise memory state for the given sequence.
     *
     * Port of `llama_memory_recurrent::state_write()`.
     *
     * @param io    Writer to serialise into.
     * @param seqId Sequence to serialise, or -1 for all.
     */
    fun stateWrite(io: LlamaIoWrite, seqId: LlamaSeqId = -1) {
        // ranges: from inclusive, to exclusive
        val cellRanges = mutableListOf<Pair<Int, Int>>()
        var cellCount = 0

        var cellRangeBegin = size
        for (i in 0 until size) {
            val cell = cells[i]
            if ((seqId == -1 && !cell.isEmpty()) || cell.hasSeqId(seqId)) {
                cellCount++
                if (cellRangeBegin == size) {
                    cellRangeBegin = i
                }
            } else {
                if (cellRangeBegin != size) {
                    cellRanges.add(cellRangeBegin to i)
                    cellRangeBegin = size
                }
            }
        }
        if (cellRangeBegin != size) {
            cellRanges.add(cellRangeBegin to size)
        }

        io.writeI32(cellCount)

        stateWriteMeta(io, cellRanges, seqId)
        stateWriteData(io, cellRanges)
    }

    /**
     * Deserialise memory state for the given sequence.
     *
     * Port of `llama_memory_recurrent::state_read()`.
     *
     * @param io    Reader to deserialise from.
     * @param seqId Destination sequence, or -1 for all.
     */
    fun stateRead(io: LlamaIoRead, seqId: LlamaSeqId = -1) {
        val cellCount = io.readU32().toInt()

        var res = true
        res = res && stateReadMeta(io, cellCount, seqId)
        res = res && stateReadData(io, cellCount)

        if (!res) {
            if (seqId == -1) {
                clear(true)
            } else {
                seqRm(seqId, -1, -1)
            }
            error("failed to restore recurrent state cache")
        }
    }

    // -- private helpers ---------------------------------------------------

    /** Total bytes consumed by r + s state tensors. */
    private fun totalSize(): Long = sizeRBytes() + sizeSBytes()

    /** Total bytes of all "r" state tensors. */
    private fun sizeRBytes(): Long {
        var total = 0L
        for (r in rL) {
            if (r != null) {
                total += r.nBytes()
            }
        }
        return total
    }

    /** Total bytes of all "s" state tensors. */
    private fun sizeSBytes(): Long {
        var total = 0L
        for (s in sL) {
            if (s != null) {
                total += s.nBytes()
            }
        }
        return total
    }

    // -- state write helpers ------------------------------------------------

    private fun stateWriteMeta(
        io: LlamaIoWrite,
        cellRanges: List<Pair<Int, Int>>,
        seqId: LlamaSeqId,
    ) {
        for ((first, second) in cellRanges) {
            for (i in first until second) {
                val cell = cells[i]
                io.writeI32(cell.pos)
                val nSeqIdOut = if (seqId == -1) cell.seqId.size else 0
                io.writeI32(nSeqIdOut)
                if (nSeqIdOut > 0) {
                    for (sid in cell.seqId) {
                        io.writeI32(sid)
                    }
                }
            }
        }
    }

    private fun stateWriteData(io: LlamaIoWrite, cellRanges: List<Pair<Int, Int>>) {
        val sTrans = 0
        val nLayer = hparams.nLayer

        io.writeI32(sTrans)
        io.writeI32(nLayer)

        // Write R tensors
        for (il in 0 until nLayer) {
            val r = rL[il] ?: continue

            io.writeI32(r.type.ordinal)
            val rSizeRow = r.rowSizeBytes(hparams.nEmbdR())
            io.writeI32(rSizeRow.toInt()) // write as 64-bit in C++, but we use Int here
            // Write each range
            for ((first, second) in cellRanges) {
                val rangeSize = second - first
                val bufSize = rangeSize * rSizeRow.toInt()
                io.writeTensor(r, first * rSizeRow.toInt(), bufSize)
            }
        }

        // Write S tensors (non-transposed path)
        for (il in 0 until nLayer) {
            val s = sL[il] ?: continue

            io.writeI32(s.type.ordinal)
            val sSizeRow = s.rowSizeBytes(hparams.nEmbdS())
            io.writeI32(sSizeRow.toInt())
            for ((first, second) in cellRanges) {
                val rangeSize = second - first
                val bufSize = rangeSize * sSizeRow.toInt()
                io.writeTensor(s, first * sSizeRow.toInt(), bufSize)
            }
        }
    }

    // -- state read helpers -------------------------------------------------

    private fun stateReadMeta(io: LlamaIoRead, cellCount: Int, destSeqId: LlamaSeqId): Boolean {
        if (destSeqId != -1) {
            // single sequence restore
            seqRm(destSeqId, -1, -1)

            if (cellCount == 0) return true

            // Build a ubatch for find_slot
            val posArr = IntArray(cellCount)
            for (i in 0 until cellCount) {
                val pos = io.readI32()
                val nSid = io.readU32().toInt()
                if (nSid != 0) {
                    llamaLogError("stateReadMeta: invalid seq_id-agnostic cell\n")
                    return false
                }
                posArr[i] = pos
            }

            val seqIdArr = Array(cellCount) { intArrayOf(destSeqId) }
            val nSeqIdArr = IntArray(cellCount) { 1 }
            val ubatch = LlamaUBatch(
                nTokens = cellCount,
                nSeqTokens = cellCount,
                nSeqs = 1,
                pos = posArr,
                nSeqId = nSeqIdArr,
                seqId = seqIdArr,
                equalSeqs = true,
            )

            if (!findSlot(ubatch)) {
                llamaLogError("stateReadMeta: failed to find available cells in cache\n")
                return false
            }
        } else {
            // whole cache restore
            if (cellCount > size) {
                llamaLogError("stateReadMeta: not enough cells in cache\n")
                return false
            }

            clear(true)

            for (i in 0 until cellCount) {
                val cell = cells[i]
                val pos = io.readI32()
                val nSid = io.readU32().toInt()

                cell.pos = pos

                for (j in 0 until nSid) {
                    val sid = io.readI32()
                    if (sid < 0 || sid >= nSeqMax) {
                        llamaLogError("stateReadMeta: invalid seq_id $sid out of range [0, $nSeqMax)\n")
                        return false
                    }
                    cell.seqId.add(sid)

                    val tail = cells[sid].tail
                    if (tail != -1) {
                        llamaLogError("stateReadMeta: duplicate tail for seq_id $sid in cell $i and $tail\n")
                        return false
                    }
                    cells[sid].tail = i
                }
            }

            head = 0
            used = cellCount
        }

        // make sure the recurrent states will keep their restored state
        for (i in 0 until cellCount) {
            val cellId = head + i
            cells[cellId].src = cellId
        }

        return true
    }

    private fun stateReadData(io: LlamaIoRead, cellCount: Int): Boolean {
        val sTrans = io.readU32().toInt()
        val nLayer = io.readU32().toInt()

        if (nLayer != hparams.nLayer) {
            llamaLogError("stateReadData: mismatched layer count ($nLayer instead of ${hparams.nLayer})\n")
            return false
        }
        if (cellCount > size) {
            llamaLogError("stateReadData: not enough cells ($cellCount > $size)\n")
            return false
        }
        if (sTrans != 0) {
            llamaLogError("stateReadData: incompatible s transposition\n")
            return false
        }

        // Read R tensors
        for (il in 0 until nLayer) {
            val r = rL[il] ?: continue

            val rTypeRef = io.readI32()
            val rTypeI = r.type.ordinal
            if (rTypeI != rTypeRef) {
                llamaLogError("stateReadData: mismatched r type ($rTypeI != $rTypeRef, layer $il)\n")
                return false
            }

            val rSizeRowRef = io.readI32()
            val rSizeRow = r.rowSizeBytes(hparams.nEmbdR()).toInt()
            if (rSizeRow != rSizeRowRef) {
                llamaLogError("stateReadData: mismatched r row size ($rSizeRow != $rSizeRowRef, layer $il)\n")
                return false
            }

            if (cellCount > 0) {
                val data = io.read(cellCount * rSizeRow)
                r.setData(data, head * rSizeRow, cellCount * rSizeRow)
            }
        }

        // Read S tensors (non-transposed)
        for (il in 0 until nLayer) {
            val s = sL[il] ?: continue

            val sTypeRef = io.readI32()
            val sTypeI = s.type.ordinal
            if (sTypeI != sTypeRef) {
                llamaLogError("stateReadData: mismatched s type ($sTypeI != $sTypeRef, layer $il)\n")
                return false
            }

            val sSizeRowRef = io.readI32()
            val sSizeRow = s.rowSizeBytes(hparams.nEmbdS()).toInt()
            if (sSizeRow != sSizeRowRef) {
                llamaLogError("stateReadData: mismatched s row size ($sSizeRow != $sSizeRowRef, layer $il)\n")
                return false
            }

            if (cellCount > 0) {
                val data = io.read(cellCount * sSizeRow)
                s.setData(data, head * sSizeRow, cellCount * sSizeRow)
            }
        }

        return true
    }
}

// =============================================================================
// Extension: convert LlamaUBatchInternal → LlamaUBatch
// =============================================================================

/**
 * Convert an internal ubatch (with full bookkeeping) to the public type used
 * by memory contexts.
 */
fun LlamaUBatchInternal.toPublic(): LlamaUBatch = LlamaUBatch(
    nTokens = nTokens,
    nSeqTokens = nSeqTokens,
    nSeqs = nSeqs,
    tokens = token,
    pos = pos,
    nSeqId = nSeqId,
    seqId = seqId,
    output = output?.map { it != 0.toByte() }?.toBooleanArray(),
    equalSeqs = equalSeqs,
)

// =============================================================================
// LlamaMemoryRecurrentContext – batch/full/update context
// Ported from: llama-memory-recurrent.h  class llama_memory_recurrent_context
// =============================================================================

/**
 * Context object handed out by [LlamaMemoryRecurrent] to manage a batch
 * processing session, a full-state snapshot, or a defrag/optimise pass.
 *
 * Port of `class llama_memory_recurrent_context`.
 */
class LlamaMemoryRecurrentContext private constructor(
    val status: LlamaMemoryStatus,
    private val mem: LlamaMemoryRecurrent?,
    private val ubatches: List<LlamaUBatch>,
    private val isFull: Boolean,
) : LlamaMemoryContext {
    private var iNext: Int = 0

    // -- factory constructors matching C++ overloads -----------------------

    companion object {
        /** Error context. */
        fun error(status: LlamaMemoryStatus) =
            LlamaMemoryRecurrentContext(status, mem = null, ubatches = emptyList(), isFull = false)

        /** Full-cache or update context. */
        fun full(mem: LlamaMemoryRecurrent) =
            LlamaMemoryRecurrentContext(LlamaMemoryStatus.SUCCESS, mem, ubatches = emptyList(), isFull = true)

        /** Batch-processing context. */
        fun batch(mem: LlamaMemoryRecurrent, ubatches: List<LlamaUBatch>) =
            LlamaMemoryRecurrentContext(LlamaMemoryStatus.SUCCESS, mem, ubatches, isFull = false)
    }

    // -- iteration ---------------------------------------------------------

    /**
     * Advance to the next ubatch.
     *
     * Port of `llama_memory_recurrent_context::next()`.
     *
     * @return `true` while there is a ubatch to process; `false` when done.
     */
    override fun next(): Boolean {
        if (++iNext >= ubatches.size) {
            return false
        }
        return true
    }

    /**
     * Apply state changes for the current ubatch (find slot, etc.).
     *
     * Port of `llama_memory_recurrent_context::apply()`.
     */
    override fun apply(): Boolean {
        check(!status.isFail()) { "Cannot apply on a failed context" }

        // no ubatches -> this is an update context
        if (ubatches.isEmpty()) {
            // recurrent cache never performs updates
            check(status == LlamaMemoryStatus.NO_UPDATE)
            return true
        }

        mem!!.findSlot(ubatches[iNext])
        return true
    }

    /** Status of this context. */
    override fun getStatus(): LlamaMemoryStatus = status

    /** Currently active ubatch (valid after a successful [next] call). */
    override fun getUbatch(): LlamaUBatch {
        check(status == LlamaMemoryStatus.SUCCESS) { "getUbatch requires SUCCESS status" }
        return ubatches[iNext]
    }

    // -- recurrent-specific accessors --------------------------------------

    /** Number of active recurrent states. */
    fun getNRs(): Int {
        val m = mem ?: return 0
        return if (isFull) m.size else m.n
    }

    /** Head position in the cell ring. */
    fun getHead(): Int {
        val m = mem ?: return 0
        return if (isFull) 0 else m.head
    }

    /** Index of the first zero state. */
    fun getRsZ(): Int {
        val m = mem ?: return -1
        return if (isFull) 0 else m.rsZ
    }

    /** Total cell count. */
    fun getSize(): Int = mem?.size ?: 0

    /** Per-layer "r" state tensor, or `null`. */
    fun getRL(il: Int): GGMLTensor? = mem?.rL?.getOrNull(il)

    /** Per-layer "s" state tensor, or `null`. */
    fun getSL(il: Int): GGMLTensor? = mem?.sL?.getOrNull(il)

    /**
     * Source index for the i-th state copy.
     *
     * Port of `llama_memory_recurrent_context::s_copy()`.
     */
    fun sCopy(i: Int): Int {
        val m = mem ?: return -1
        return m.cells[i + m.head].src0
    }
}
