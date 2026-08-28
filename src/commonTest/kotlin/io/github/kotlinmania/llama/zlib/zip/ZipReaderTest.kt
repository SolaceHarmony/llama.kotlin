package io.github.kotlinmania.llama.lib.zip

import kotlin.test.Test
import kotlin.test.assertEquals
import io.github.kotlinmania.io.Buffer
import io.github.kotlinmania.io.buffered
import io.github.kotlinmania.io.files.FileSystem
import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.files.SystemFileSystem
import io.github.kotlinmania.io.snapshot
import io.github.kotlinmania.io.write

class ZipReaderTest {
    private fun makeStoredZipSingle(name: String, data: ByteArray): ByteArray {
        val nameBytes = name.encodeToByteArray()
        val nameLen = nameBytes.size
        val dataLen = data.size
        val localHeaderSize = 30 + nameLen + dataLen
        val centralDirSize = 46 + nameLen
        val eocdSize = 22
        val totalSize = localHeaderSize + centralDirSize + eocdSize
        val out = ByteArray(totalSize)
        var offset = 0

        fun writeShortLe(value: Int) {
            out[offset++] = (value and 0xFF).toByte()
            out[offset++] = ((value ushr 8) and 0xFF).toByte()
        }

        fun writeIntLe(value: Int) {
            out[offset++] = (value and 0xFF).toByte()
            out[offset++] = ((value ushr 8) and 0xFF).toByte()
            out[offset++] = ((value ushr 16) and 0xFF).toByte()
            out[offset++] = ((value ushr 24) and 0xFF).toByte()
        }

        // Local header
        writeIntLe(0x04034b50)
        writeShortLe(20)
        writeShortLe(0)
        writeShortLe(0)
        writeShortLe(0)
        writeShortLe(0)
        writeIntLe(0)
        writeIntLe(dataLen)
        writeIntLe(dataLen)
        writeShortLe(nameLen)
        writeShortLe(0)
        nameBytes.copyInto(out, offset)
        offset += nameLen
        data.copyInto(out, offset)
        offset += dataLen

        // Central directory
        val centralDirOffset = offset
        writeIntLe(0x02014b50)
        writeShortLe(20)
        writeShortLe(20)
        writeShortLe(0)
        writeShortLe(0)
        writeShortLe(0)
        writeShortLe(0)
        writeIntLe(0)
        writeIntLe(dataLen)
        writeIntLe(dataLen)
        writeShortLe(nameLen)
        writeShortLe(0)
        writeShortLe(0)
        writeShortLe(0)
        writeShortLe(0)
        writeIntLe(0)
        writeIntLe(0)
        nameBytes.copyInto(out, offset)
        offset += nameLen

        // EOCD
        val centralDirLength = offset - centralDirOffset
        writeIntLe(0x06054b50)
        writeShortLe(0)
        writeShortLe(0)
        writeShortLe(1)
        writeShortLe(1)
        writeIntLe(centralDirLength)
        writeIntLe(centralDirOffset)
        writeShortLe(0)

        return out
    }

    @Test
    fun testListAndExtractStored() {
        if (!supportsZipFileIO) {
            println("Skipping ZipReaderTest on platform without filesystem support")
            return
        }
        val data = "hello world".encodeToByteArray()
        val zipBytes = makeStoredZipSingle("test.txt", data)
        val fs: FileSystem = SystemFileSystem
        val tmp = Path("./build/tmp/zip-test.zip")
        tmp.parent?.let { fs.createDirectories(it, mustCreate = false) }
        fs.sink(tmp).buffered().useResource { sink -> sink.write(zipBytes) }

        val entries = ZipReader.list(fs, tmp)
        assertEquals(1, entries.size)
        assertEquals("test.txt", entries[0].name)
        val buf = Buffer()
        ZipReader.extract(fs, tmp, entries[0], buf)
        val result = ByteArray(data.size)
        var filled = 0
        while (!buf.exhausted() && filled < result.size) {
            val read = buf.readAtMostTo(result, filled, result.size)
            if (read == -1) break
            filled += read
        }
        val text = result.decodeToString(endIndex = filled)
        assertEquals("hello world", text)
    }
}

private inline fun <T : AutoCloseable, R> T.useResource(block: (T) -> R): R {
    try {
        return block(this)
    } finally {
        this.close()
    }
}
