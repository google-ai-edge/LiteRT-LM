#include "runtime/executor/litert/debug_utils.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/log_severity.h"  // from @com_google_absl
#include "absl/log/scoped_mock_log.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/convert_tensor_buffer.h"

namespace litert::lm {
namespace {

using ::testing::_;

TEST(DebugUtilsTest, LogValuesLogsShortSpan) {
  absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(_, _, _)).Times(testing::AnyNumber());
  EXPECT_CALL(log,
              Log(absl::LogSeverity::kInfo, _, "test(size=5): 1, 2, 3, 4, 5"))
      .Times(1);
  log.StartCapturingLogs();

  std::vector<float> values = {1.0, 2.0, 3.0, 4.0, 5.0};
  LogValues(absl::MakeSpan(values), 2, "test");
}

TEST(DebugUtilsTest, LogValuesLogsLongSpan) {
  absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(_, _, _)).Times(testing::AnyNumber());
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       "test(size=20): 1, 2 ... 10, 11 ... 19, 20"))
      .Times(1);
  log.StartCapturingLogs();

  std::vector<float> values;
  for (int i = 1; i <= 20; ++i) {
    values.push_back(i);
  }
  LogValues(absl::MakeSpan(values), 2, "test");
}

TEST(DebugUtilsTest, LogTensorLogsTensorBuffer) {
  absl::ScopedMockLog log;
  EXPECT_CALL(log, Log(_, _, _)).Times(testing::AnyNumber());
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _,
                       "test_tensor(size=3): 1.5, 2.5, 3.5"))
      .Times(1);
  log.StartCapturingLogs();

  std::vector<float> values = {1.5, 2.5, 3.5};
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tensor_buffer,
      CopyToTensorBuffer<float>(absl::MakeSpan(values), {3}));

  LogTensor(tensor_buffer, 2, "test_tensor");
}

}  // namespace
}  // namespace litert::lm
