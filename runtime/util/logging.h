#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_H_

#include <iostream>
#include <optional>
#include <vector>
#include <variant>
#include "absl/strings/str_join.h"  // from @com_google_absl

namespace litert::lm {

// Helper function to print a vector of elements.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& data) {
  os << "vector of " << data.size() << " elements: ["
     << absl::StrJoin(data, ", ") << "]";
  return os;
}

// Helper function to print a std::optional of data.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& data) {
  if (data.has_value()) {
    return os << *data;
  }
  return os << "Not set.";
}

// Helper function to print a std::variant of data.
template <typename... T>
std::ostream& operator<<(std::ostream& os, const std::variant<T...>& data) {
  std::visit([&os](const auto& arg) { os << arg; }, data);
  return os;
}

}  // namespace litert::lm


#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_H_
