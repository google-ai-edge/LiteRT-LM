/*
 * Copyright 2025 Google LLC
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

import com.google.gson.JsonArray

/**
 * Manages the lifecycle of a LiteRT-LM engine, providing an interface for interacting with the
 * underlying native library.
 *
 * Example usage:
 * ```
 * val config = EngineConfig(modelPath = "...")
 * val engine = Engine(config)
 * engine.initialize()
 * ...
 * engine.close()
 * ```
 *
 * @param engineConfig The configuration for the engine.
 */
fun Engine(engineConfig: EngineConfig): Engine {
    return EngineJni(engineConfig, NativeLibraryLoaderJni)
}