// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreFoundation
import Foundation

/// The executor stats of a turn, e.g. speculative decoding rounds and accepted draft tokens.
///
/// Metric and latency names depend on the executor and may change.
public struct ExecutorStats: Sendable, Equatable {
  /// The name of the module that reported the stats, e.g. "LLM".
  public let moduleName: String

  /// The metrics by name, e.g. "plain_steps" or "acceptance_rate".
  public let metrics: [String: Double]

  /// The latencies by name, in microseconds.
  public let latenciesInMicroseconds: [String: Int64]

  /// The stats of the submodules, in the same shape.
  public let substats: [ExecutorStats]

  /// Creates an `ExecutorStats` with the given module name, metrics, latencies, and substats.
  public init(
    moduleName: String = "",
    metrics: [String: Double] = [:],
    latenciesInMicroseconds: [String: Int64] = [:],
    substats: [ExecutorStats] = []
  ) {
    self.moduleName = moduleName
    self.metrics = metrics
    self.latenciesInMicroseconds = latenciesInMicroseconds
    self.substats = substats
  }

  /// Parses the stats from their JSON object. Missing or malformed fields are left empty.
  init(json: [String: Any]) {
    var metrics: [String: Double] = [:]
    for (name, value) in json["metrics"] as? [String: Any] ?? [:] {
      if let number = jsonNumber(value) { metrics[name] = number.doubleValue }
    }
    var latenciesInMicroseconds: [String: Int64] = [:]
    for (name, value) in json["latencies_us"] as? [String: Any] ?? [:] {
      if let latency = jsonInt64(value) { latenciesInMicroseconds[name] = latency }
    }
    self.init(
      moduleName: json["module_name"] as? String ?? "",
      metrics: metrics,
      latenciesInMicroseconds: latenciesInMicroseconds,
      substats: (json["substats"] as? [Any] ?? []).compactMap {
        ($0 as? [String: Any]).map { ExecutorStats(json: $0) }
      }
    )
  }

  /// Parses the stats from their JSON string. Missing or malformed fields are left empty.
  ///
  /// - Throws: A `LiteRTLMError` if the string is not a JSON object.
  init(jsonString: String) throws {
    guard let data = jsonString.data(using: .utf8),
      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else {
      throw LiteRTLMError.conversation(.invalidJson("Invalid executor stats JSON"))
    }
    self.init(json: json)
  }
}

/// Returns `value` if it's a JSON number. `JSONSerialization` also returns `true` and `false` as
/// `NSNumber`, so they're told apart by their CoreFoundation type.
private func jsonNumber(_ value: Any) -> NSNumber? {
  guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else {
    return nil
  }
  return number
}

/// Returns `value` if it's an integer JSON number representable as `Int64`.
private func jsonInt64(_ value: Any) -> Int64? {
  guard let number = jsonNumber(value), !CFNumberIsFloatType(number) else {
    return nil
  }
  return Int64(exactly: number)
}
