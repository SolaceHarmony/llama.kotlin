package io.github.kotlinmania.llama.lib.common

import kotlin.js.Date

internal var logBuffer = StringBuilder()

actual fun logToFile(line: String) {
    // JS environment has no filesystem; accumulate in-memory buffer for debugging or drop
    logBuffer.appendLine(line)
}

actual fun getEnv(name: String): String? = null

actual var LOG_FILE_PATH: String? = null

actual fun currentTimestamp(): String = Date().toISOString()
