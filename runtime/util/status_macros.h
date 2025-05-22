#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STATUS_MACROS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STATUS_MACROS_H_

#include <sstream>

#include "absl/status/status.h"  // from @com_google_absl

// Minimal implementations of status_macros.h and ret_check.h.

#if _LITERT_LM_REDEFINE_STATUS_MACROS || !defined(ASSIGN_OR_RETURN)
#define ASSIGN_OR_RETURN(DECL, EXPR) \
  _ASSIGN_OR_RETURN_IMPL(_CONCAT_NAME(_statusor_, __LINE__), DECL, EXPR)
#define _ASSIGN_OR_RETURN_IMPL(TMP_VAR, DECL, EXPR) \
  auto&& TMP_VAR = (EXPR);                          \
  if (!TMP_VAR.ok()) return TMP_VAR.status();       \
  DECL = std::move(*TMP_VAR)
#endif  // _LITERT_LM_REDEFINE_STATUS_MACROS || !defined(ASSIGN_OR_RETURN)

#if _LITERT_LM_REDEFINE_STATUS_MACROS || !defined(RETURN_IF_ERROR)
#define RETURN_IF_ERROR(EXPR) \
  if (auto s = (EXPR); !s.ok()) return s
#endif  // _LITERT_LM_REDEFINE_STATUS_MACROS || !defined(RETURN_IF_ERROR)

#if _LITERT_LM_REDEFINE_STATUS_MACROS || !defined(RET_CHECK)
#define RET_CHECK(cond) \
  if (!(cond)) return ::litert::lm::internal::StreamToStatusHelper(#cond)
#define RET_CHECK_EQ(lhs, rhs) RET_CHECK((lhs) == (rhs))
#define RET_CHECK_NE(lhs, rhs) RET_CHECK((lhs) != (rhs))
#define RET_CHECK_LE(lhs, rhs) RET_CHECK((lhs) <= (rhs))
#define RET_CHECK_LT(lhs, rhs) RET_CHECK((lhs) < (rhs))
#define RET_CHECK_GE(lhs, rhs) RET_CHECK((lhs) >= (rhs))
#define RET_CHECK_GT(lhs, rhs) RET_CHECK((lhs) > (rhs))
#endif  // _LITERT_LM_REDEFINE_STATUS_MACROS || !defined(RET_CHECK)

namespace litert::lm::internal {

class StreamToStatusHelper {
 public:
  explicit StreamToStatusHelper(const char* message) {
    stream_ << message << ": ";
  }

  StreamToStatusHelper& SetCode(absl::StatusCode code) {
    code_ = code;
    return *this;
  }

  template <typename T>
  StreamToStatusHelper& operator<<(const T& value) {
    stream_ << value;
    return *this;
  }

  operator absl::Status() const& {  // NOLINT: converts implicitly
    return absl::Status(code_, stream_.str());
  }

 private:
  absl::StatusCode code_ = absl::StatusCode::kInternal;
  std::stringstream stream_;
};

}  // namespace litert::lm::internal

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_STATUS_MACROS_H_
