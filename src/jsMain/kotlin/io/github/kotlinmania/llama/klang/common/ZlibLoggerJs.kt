
package io.github.kotlinmania.llama.lang.common

import kotlin.js.Date

actual var LOG_FILE_PATH: String? = null

actual fun logToFile(line: String) {
    // No-op on JS; logs can be routed to console if needed
    console.log(line)
}

actual fun getEnv(name: String): String? = null

actual fun currentTimestamp(): String = Date().toISOString()
