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

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonNull
import com.google.gson.JsonObject
import com.google.gson.JsonPrimitive

internal fun Map<String, Any?>.toJsonObject(): JsonObject {
  val obj = JsonObject()
  for ((key, value) in this) {
    obj.add(key, value.toJsonElement())
  }
  return obj
}

internal fun Any?.toJsonElement(): JsonElement {
  return when (this) {
    null -> JsonNull.INSTANCE
    is JsonElement -> this
    is Map<*, *> -> {
      val obj = JsonObject()
      for ((k, v) in this) {
        obj.add(k.toString(), v.toJsonElement())
      }
      obj
    }
    is List<*> -> {
      val arr = JsonArray()
      for (item in this) {
        arr.add(item.toJsonElement())
      }
      arr
    }
    is String -> JsonPrimitive(this)
    is Number -> JsonPrimitive(this)
    is Boolean -> JsonPrimitive(this)
    is kotlin.Unit -> JsonPrimitive("")
    else -> JsonPrimitive(this.toString())
  }
}

internal fun JsonObject.toMap(): Map<String, Any?> {
  val map = mutableMapOf<String, Any?>()
  for (entry in this.entrySet()) {
    map[entry.key] = entry.value.toKotlinValue()
  }
  return map
}

internal fun JsonElement.toKotlinValue(): Any? {
  return when {
    this.isJsonNull -> null
    this.isJsonObject -> this.asJsonObject.toMap()
    this.isJsonArray -> this.asJsonArray.map { it.toKotlinValue() }
    this.isJsonPrimitive -> {
      val primitive = this.asJsonPrimitive
      when {
        primitive.isBoolean -> primitive.asBoolean
        primitive.isNumber -> primitive.asNumber
        primitive.isString -> primitive.asString
        else -> this
      }
    }
    else -> this
  }
}
