/*
 * Copyright 2026 Google LLC
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
package com.google.ai.edge.litertlm.example

import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.benchmark

suspend fun main(args: Array<String>) {
  val modelPath =
    requireNotNull(args.getOrNull(0)) { "Model path must be provided as the first argument." }

  @OptIn(ExperimentalApi::class)
  val benchmarkInfo = benchmark(modelPath = modelPath, backend = Backend.CPU())
  println(YELLOW + "Benchmark result: $benchmarkInfo" + RESET)
}

// ANSI color codes
const val RESET = "\u001B[0m"
const val YELLOW = "\u001B[33m"
