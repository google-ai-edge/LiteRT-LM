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
import java.io.InputStream

/** Helper class for loading the LiteRT-LM native library. */
abstract class NativeLibraryLoader {

  abstract fun load()

  protected abstract fun isLoaded(): Boolean

  protected abstract fun tryLoadLibrary(libName: String): Boolean

  protected abstract fun tryExtractAndLoad(resourcePath: String, libName: String): Boolean

  protected abstract fun extractResource(
    resource: InputStream,
    resourceName: String,
    extractToDirectory: String,
  ): String

  protected abstract fun os(): String

  protected abstract fun architecture(): String

  protected abstract fun copy(src: InputStream, dstFile: File): Long

  protected abstract fun createTemporaryDirectory(): File

  protected abstract fun log(msg: String)

  /** Native function to check if native lib is loaded. Throws [UnsatisfiedLinkError] if not. */
  abstract fun nativeCheckLoaded()
}
