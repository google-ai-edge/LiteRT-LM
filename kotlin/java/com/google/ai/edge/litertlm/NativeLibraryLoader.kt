/*
 * Copyright 2025 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.litertlm

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

/** Helper class for loading the LiteRT-LM native library. */
object NativeLibraryLoader {
  private const val JNI_LIBNAME = "litertlm_jni"
  private val loadedLibraries = mutableSetOf<String>()
  private val DEBUG =
    System.getProperty("com.google.ai.edge.litertlm.NativeLibraryLoader.DEBUG") != null

  fun load(libName: String = JNI_LIBNAME) {
    synchronized(loadedLibraries) {
      if (libName in loadedLibraries) {
        log("Skip loading as $libName is loaded already.")
        return
      }

      // 0. Skip loading if loaded already (for default lib).
      if (libName == JNI_LIBNAME && isLoaded()) {
        log("Skip loading as the native library is loaded already.")
        loadedLibraries.add(libName)
        return
      }

      // 1. Try loading from library path. (e.g., for Android)
      if (tryLoadLibrary(libName)) {
        log("Loaded $libName from library path.")
        loadedLibraries.add(libName)
        return
      }

      // For simplicity, the native library extension is ".so" instead of the default ".dylib" on
      // MacOS since it is the default cc_binary output for MacOS.
      val jniLibName = System.mapLibraryName(libName).replace(".dylib", ".so")

      // 2. Try extracting from JAR (generic path). (e.g., for bazel)
      val genericResourcePath = "com/google/ai/edge/litertlm/jni/$jniLibName"
      if (tryExtractAndLoad(genericResourcePath, jniLibName)) {
        log("Loaded $libName from JAR: $genericResourcePath")
        loadedLibraries.add(libName)
        return
      }

      // 3. Try extracting from JAR (OS-Arch specific path). (e.g., for multi-platform Maven
      // packages)
      val osArchResourcePath =
        "com/google/ai/edge/litertlm/jni/${os()}-${architecture()}/$jniLibName"
      if (tryExtractAndLoad(osArchResourcePath, jniLibName)) {
        log("Loaded $libName from JAR: $osArchResourcePath")
        loadedLibraries.add(libName)
        return
      }

      throw UnsatisfiedLinkError(
        "Failed to load native library $libName. Tried system path, $genericResourcePath, and $osArchResourcePath"
      )
    }
  }

  private fun isLoaded(): Boolean =
    try {
      nativeCheckLoaded()
      true
    } catch (e: UnsatisfiedLinkError) {
      false
    }

  private fun tryLoadLibrary(libName: String): Boolean =
    try {
      System.loadLibrary(libName)
      true
    } catch (e: UnsatisfiedLinkError) {
      log("System.loadLibrary($libName) failed: ${e.message}")
      false
    }

  private fun tryExtractAndLoad(resourcePath: String, libName: String): Boolean {
    log("Attempting to extract from: $resourcePath")
    val jniResource = NativeLibraryLoader::class.java.classLoader?.getResourceAsStream(resourcePath)

    if (jniResource == null) {
      log("Resource not found: $resourcePath")
      return false
    }

    return try {
      val tempPath = createTemporaryDirectory()
      tempPath.deleteOnExit()
      val tempDirectory = tempPath.canonicalPath
      val extractedLibraryPath = extractResource(jniResource, libName, tempDirectory)
      System.load(extractedLibraryPath)
      true
    } catch (e: IOException) {
      log("Failed to extract $resourcePath: $e")
      false
    } catch (e: UnsatisfiedLinkError) {
      log("Failed to load extracted library from $resourcePath: $e")
      false
    }
  }

  private fun extractResource(
    resource: InputStream,
    resourceName: String,
    extractToDirectory: String,
  ): String {
    val dst = File(extractToDirectory, resourceName)
    dst.deleteOnExit()
    val dstPath = dst.toString()
    log("extracting native library to: $dstPath")
    val nbytes = copy(resource, dst)
    log("copied $nbytes bytes to $dstPath")
    return dstPath
  }

  private fun os(): String {
    val p = System.getProperty("os.name", "")!!.lowercase()
    return when {
      p.contains("linux") -> "linux"
      p.contains("os x") || p.contains("darwin") -> "darwin"
      p.contains("windows") -> "windows"
      else -> p.replace("\\s".toRegex(), "") // os.name with all whitespace removed.
    }
  }

  private fun architecture(): String {
    val arch = System.getProperty("os.arch", "")!!.lowercase()
    return if (arch == "amd64") "x86_64" else arch
  }

  private fun copy(src: InputStream, dstFile: File): Long {
    FileOutputStream(dstFile).use { dst ->
      val buffer = ByteArray(1 shl 20) // 1MB
      var ret = 0L
      var n = src.read(buffer)
      while (n >= 0) {
        dst.write(buffer, 0, n)
        ret += n
        n = src.read(buffer)
      }
      return ret
    }
  }

  private fun createTemporaryDirectory(): File {
    val baseDirectory = File(System.getProperty("java.io.tmpdir")!!)
    val directoryName = "litertlm_native_libraries-${System.currentTimeMillis()}-"
    for (attempt in 0 until 1000) {
      val temporaryDirectory = File(baseDirectory, directoryName + attempt)
      if (temporaryDirectory.mkdir()) {
        return temporaryDirectory
      }
    }
    throw IllegalStateException(
      "Could not create a temporary directory (tried to make $directoryName*) to extract LiteRT-LM native libraries."
    )
  }

  private fun log(msg: String) {
    if (DEBUG) {
      System.err.println("com.google.ai.edge.litertlm.NativeLibraryLoader: $msg")
    }
  }

  /** Native function to check if `JNI_LIBNAME` is loaded. Throws [UnsatisfiedLinkError] if not. */
  external fun nativeCheckLoaded()
}
