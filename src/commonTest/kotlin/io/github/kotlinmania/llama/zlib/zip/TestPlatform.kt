package io.github.kotlinmania.llama.lib.zip

import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.files.SystemFileSystem

internal val supportsZipFileIO: Boolean by lazy {
    return@lazy try {
        val tmpDir = Path("./build/tmp/zip-capability-check")
        SystemFileSystem.createDirectories(tmpDir, mustCreate = false)
        true
    } catch (_: Throwable) {
        false
    }
}
