package io.github.kotlinmania.llama.lib.common

import io.github.kotlinmania.io.files.Path
import io.github.kotlinmania.io.path.appendText
import kotlin.io.path.appendText

actual var LOG_FILE_PATH: String? = null

actual fun logToFile(line: String) {
    val path = LOG_FILE_PATH ?: "zlib.log"
    Path(path).appendText(line)
}

actual fun getEnv(name: String): String? = System.getenv(name)

actual fun currentTimestamp(): String = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"))
