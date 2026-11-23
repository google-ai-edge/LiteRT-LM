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

import com.google.gson.JsonElement
import com.google.gson.JsonObject

/**
 * Stats that an executor reported for a turn. See [Message.executorStats]. Metric and latency names
 * depend on the executor and may change.
 *
 * @property moduleName The name of the module that reported the stats, e.g. "LLM".
 * @property metrics The metrics by name, e.g. "plain_steps".
 * @property latenciesInMicroseconds The latencies by name, in microseconds.
 * @property substats The stats of the submodules, in the same shape.
 */
class ExecutorStats(
  val moduleName: String = "",
  val metrics: Map<String, Double> = emptyMap(),
  val latenciesInMicroseconds: Map<String, Long> = emptyMap(),
  val substats: List<ExecutorStats> = emptyList(),
) {
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is ExecutorStats) return false
    return moduleName == other.moduleName &&
      metrics == other.metrics &&
      latenciesInMicroseconds == other.latenciesInMicroseconds &&
      substats == other.substats
  }

  override fun hashCode(): Int {
    var result = moduleName.hashCode()
    result = 31 * result + metrics.hashCode()
    result = 31 * result + latenciesInMicroseconds.hashCode()
    result = 31 * result + substats.hashCode()
    return result
  }

  override fun toString(): String =
    "ExecutorStats(moduleName=$moduleName, metrics=$metrics, " +
      "latenciesInMicroseconds=$latenciesInMicroseconds, substats=$substats)"
}

/**
 * Parses the `executor_stats` JSON of a message. Missing fields become empty, and fields of the
 * wrong type are skipped, so stats from a newer runtime still parse.
 */
internal fun parseExecutorStats(json: JsonObject): ExecutorStats {
  val metrics = buildMap {
    json.objectOrNull("metrics")?.entrySet()?.forEach { (name, value) ->
      value.asDoubleOrNull()?.let { put(name, it) }
    }
  }
  val latenciesInMicroseconds = buildMap {
    json.objectOrNull("latencies_us")?.entrySet()?.forEach { (name, value) ->
      value.asLongOrNull()?.let { put(name, it) }
    }
  }
  val substats = buildList {
    val substatsJson = json.get("substats")
    if (substatsJson != null && substatsJson.isJsonArray) {
      for (substat in substatsJson.asJsonArray) {
        if (substat.isJsonObject) add(parseExecutorStats(substat.asJsonObject))
      }
    }
  }
  return ExecutorStats(
    moduleName = json.get("module_name")?.asStringOrNull() ?: "",
    metrics = metrics,
    latenciesInMicroseconds = latenciesInMicroseconds,
    substats = substats,
  )
}

internal fun JsonObject.objectOrNull(key: String): JsonObject? {
  val value = get(key)
  return if (value != null && value.isJsonObject) value.asJsonObject else null
}

internal fun JsonElement.asStringOrNull(): String? =
  if (isJsonPrimitive && asJsonPrimitive.isString) asString else null

/** Returns the value if it's a finite JSON number. Gson parses leniently, so it accepts NaN. */
private fun JsonElement.asDoubleOrNull(): Double? =
  if (isJsonPrimitive && asJsonPrimitive.isNumber) asDouble.takeIf { it.isFinite() } else null

/** Returns the value if it's an exact JSON integer that fits in a [Long]. */
private fun JsonElement.asLongOrNull(): Long? {
  if (!isJsonPrimitive || !asJsonPrimitive.isNumber) return null
  return try {
    asBigDecimal.longValueExact()
  } catch (_: ArithmeticException) {
    null
  } catch (_: NumberFormatException) {
    null
  }
}
