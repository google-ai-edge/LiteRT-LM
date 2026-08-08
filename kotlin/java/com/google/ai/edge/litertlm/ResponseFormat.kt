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
package com.google.ai.edge.litertlm

import com.google.gson.JsonParser

/**
 * Response format for constrained decoding.
 *
 * Currently supports JSON Schema and Regex.
 *
 * @property type The type of constraint (e.g. REGEX, JSON_OBJECT).
 * @property schemaOrPattern The schema (for JSON_OBJECT) or pattern (for REGEX) string.
 */
data class ResponseFormat(val type: Type, val schemaOrPattern: String) {
  enum class Type(val value: Int) {
    REGEX(1),
    JSON_OBJECT(2),
  }

  companion object {
    /**
     * Creates a JSON Schema response format.
     *
     * @param schema The JSON schema as a JSON string.
     */
    @JvmStatic
    fun json(schema: String): ResponseFormat {
      try {
        val unused = JsonParser.parseString(schema)
      } catch (e: Exception) {
        throw IllegalArgumentException("Invalid JSON schema string: ${e.message}", e)
      }
      return ResponseFormat(Type.JSON_OBJECT, schema)
    }

    /**
     * Creates a JSON Schema response format.
     *
     * @param schema The JSON schema as a Map.
     */
    @JvmStatic
    fun json(schema: Map<String, Any?>): ResponseFormat {
      return json(schema.toJsonObject().toString())
    }

    /**
     * Creates a Regex response format.
     *
     * @param pattern The regex pattern string.
     */
    @JvmStatic
    fun regex(pattern: String): ResponseFormat {
      return ResponseFormat(Type.REGEX, pattern)
    }
  }
}
