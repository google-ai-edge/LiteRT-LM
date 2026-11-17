// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/conversation/end_of_turn.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/conversation/io_types.h"
#include "runtime/executor/executor_stats.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kModuleNameKey = "module_name";
constexpr absl::string_view kMetricsKey = "metrics";
constexpr absl::string_view kLatenciesUsKey = "latencies_us";
constexpr absl::string_view kSubstatsKey = "substats";

// Returns `json[key]`, or nullptr if `json` has no `key` or it is null.
const nlohmann::ordered_json* FindField(const nlohmann::ordered_json& json,
                                        absl::string_view key) {
  auto it = json.find(key);
  if (it == json.end() || it->is_null()) {
    return nullptr;
  }
  return &*it;
}

absl::Status WrongTypeError(absl::string_view field, absl::string_view type) {
  return absl::InvalidArgumentError(
      absl::StrCat("Executor stats field \"", field, "\" must be ", type, "."));
}

}  // namespace

absl::string_view FinishReasonToString(FinishReason reason) {
  switch (reason) {
    case FinishReason::kStop:
      return "stop";
    case FinishReason::kMaxTokens:
      return "max_tokens";
    case FinishReason::kAppended:
      return "appended";
    case FinishReason::kCancelled:
      return "cancelled";
  }
  return "unknown";
}

bool IsEndOfTurnMessage(const Message& message) {
  return message.is_null() || (message.is_object() &&
                               FindField(message, kFinishReasonKey) != nullptr);
}

Message CreateEndOfTurnMessage(
    FinishReason reason, const std::optional<ExecutorStats>& executor_stats) {
  Message message = Message::object();
  message["role"] = "assistant";
  message[kFinishReasonKey] = FinishReasonToString(reason);
  message[kExecutorStatsKey] = executor_stats.has_value()
                                   ? ExecutorStatsToJson(*executor_stats)
                                   : nlohmann::ordered_json(nullptr);
  return message;
}

nlohmann::ordered_json ExecutorStatsToJson(const ExecutorStats& stats) {
  nlohmann::ordered_json metrics = nlohmann::ordered_json::object();
  for (const auto& [name, value] : stats.metrics) {
    // JSON has no NaN or infinity: `dump()` would write null, which
    // `ExecutorStatsFromJson()` rejects.
    if (const double* d = std::get_if<double>(&value);
        d != nullptr && !std::isfinite(*d)) {
      continue;
    }
    std::visit([&metrics, &name](auto v) { metrics[name] = v; }, value);
  }
  nlohmann::ordered_json latencies_us = nlohmann::ordered_json::object();
  for (const auto& [name, duration] : stats.latencies) {
    latencies_us[name] = absl::ToInt64Microseconds(duration);
  }
  nlohmann::ordered_json substats = nlohmann::ordered_json::array();
  for (const ExecutorStats& substat : stats.substats) {
    substats.push_back(ExecutorStatsToJson(substat));
  }

  nlohmann::ordered_json json = nlohmann::ordered_json::object();
  json[kModuleNameKey] = stats.module_name;
  json[kMetricsKey] = std::move(metrics);
  json[kLatenciesUsKey] = std::move(latencies_us);
  json[kSubstatsKey] = std::move(substats);
  return json;
}

absl::StatusOr<ExecutorStats> ExecutorStatsFromJson(
    const nlohmann::ordered_json& json) {
  if (!json.is_object()) {
    return absl::InvalidArgumentError("Executor stats must be a JSON object.");
  }
  ExecutorStats stats;

  if (const auto* module_name = FindField(json, kModuleNameKey)) {
    if (!module_name->is_string()) {
      return WrongTypeError(kModuleNameKey, "a string");
    }
    stats.module_name = module_name->get<std::string>();
  }

  if (const auto* metrics = FindField(json, kMetricsKey)) {
    if (!metrics->is_object()) {
      return WrongTypeError(kMetricsKey, "an object");
    }
    for (const auto& [name, value] : metrics->items()) {
      if (value.is_number_integer()) {
        stats.metrics.emplace_back(name, value.get<int64_t>());
      } else if (value.is_number_float()) {
        stats.metrics.emplace_back(name, value.get<double>());
      } else {
        return WrongTypeError(absl::StrCat(kMetricsKey, ".", name), "a number");
      }
    }
  }

  if (const auto* latencies_us = FindField(json, kLatenciesUsKey)) {
    if (!latencies_us->is_object()) {
      return WrongTypeError(kLatenciesUsKey, "an object");
    }
    for (const auto& [name, value] : latencies_us->items()) {
      if (!value.is_number_integer()) {
        return WrongTypeError(absl::StrCat(kLatenciesUsKey, ".", name),
                              "an integer");
      }
      stats.latencies.emplace_back(name,
                                   absl::Microseconds(value.get<int64_t>()));
    }
  }

  if (const auto* substats = FindField(json, kSubstatsKey)) {
    if (!substats->is_array()) {
      return WrongTypeError(kSubstatsKey, "an array");
    }
    for (const auto& substat_json : *substats) {
      absl::StatusOr<ExecutorStats> substat =
          ExecutorStatsFromJson(substat_json);
      if (!substat.ok()) {
        return substat.status();
      }
      stats.substats.push_back(*std::move(substat));
    }
  }
  return stats;
}

absl::StatusOr<std::optional<ExecutorStats>> GetExecutorStats(
    const Message& message) {
  if (!message.is_object()) {
    return std::nullopt;
  }
  const auto* executor_stats = FindField(message, kExecutorStatsKey);
  if (executor_stats == nullptr) {
    return std::nullopt;
  }
  absl::StatusOr<ExecutorStats> stats = ExecutorStatsFromJson(*executor_stats);
  if (!stats.ok()) {
    return stats.status();
  }
  return *std::move(stats);
}

}  // namespace litert::lm
