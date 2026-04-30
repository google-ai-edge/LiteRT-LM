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

#include "runtime/util/logging.h"

#include <cstdint>
#include <map>
#include <optional>

#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/log/globals.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"  // from @litert
#include "tflite/logger.h"  // from @litert
#include "tflite/minimal_logging.h"  // from @litert

namespace litert::lm {

namespace {

std::optional<LogSeverity> g_stored_min_litert_severity;

struct SeverityMapping {
  absl::LogSeverityAtLeast absl_severity;
  LiteRtLogSeverity litert_severity;
  tflite::LogSeverity tflite_severity;
};

static const absl::NoDestructor<std::map<LogSeverity, SeverityMapping>> mapping(
    {
        {LogSeverity::kVerbose,
         {absl::LogSeverityAtLeast::kInfo, kLiteRtLogSeverityVerbose,
          tflite::TFLITE_LOG_VERBOSE}},
        {LogSeverity::kDebug,
         {absl::LogSeverityAtLeast::kInfo, kLiteRtLogSeverityDebug,
          tflite::TFLITE_LOG_VERBOSE}},
        {LogSeverity::kInfo,
         {absl::LogSeverityAtLeast::kInfo, kLiteRtLogSeverityInfo,
          tflite::TFLITE_LOG_INFO}},
        {LogSeverity::kWarning,
         {absl::LogSeverityAtLeast::kWarning, kLiteRtLogSeverityWarning,
          tflite::TFLITE_LOG_WARNING}},
        {LogSeverity::kError,
         {absl::LogSeverityAtLeast::kError, kLiteRtLogSeverityError,
          tflite::TFLITE_LOG_ERROR}},
        {LogSeverity::kFatal,
         {absl::LogSeverityAtLeast::kFatal, kLiteRtLogSeverityError,
          tflite::TFLITE_LOG_ERROR}},
        {LogSeverity::kSilent,
         {absl::LogSeverityAtLeast::kInfinity, kLiteRtLogSeveritySilent,
          tflite::TFLITE_LOG_SILENT}},
    });

}  // namespace

void SetMinLogSeverity(LogSeverity log_severity) {
  auto mapping_it = mapping->find(log_severity);
  const SeverityMapping& severity_mapping =
      (mapping_it != mapping->end()) ? mapping_it->second
                                     : mapping->at(LogSeverity::kSilent);

  absl::SetMinLogLevel(severity_mapping.absl_severity);
  absl::SetStderrThreshold(severity_mapping.absl_severity);
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(),
                             severity_mapping.litert_severity);
  tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
      severity_mapping.tflite_severity);

  g_stored_min_litert_severity = log_severity;
}

std::optional<LogSeverity> GetMinLogSeverity() {
  return g_stored_min_litert_severity;
}

int8_t ToLiteRtLogSeverityInt8(LogSeverity log_severity) {
  auto mapping_it = mapping->find(log_severity);
  const SeverityMapping& severity_mapping =
      (mapping_it != mapping->end()) ? mapping_it->second
                                     : mapping->at(LogSeverity::kSilent);

  return static_cast<int8_t>(severity_mapping.litert_severity);
}

}  // namespace litert::lm
