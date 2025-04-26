#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_TENSOR_BUFFER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_TENSOR_BUFFER_H_

#include <iostream>

#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Helper function to print a ::litert::TensorBuffer.
std::ostream& operator<<(std::ostream& os,
                         const ::litert::TensorBuffer& tensor_buffer);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_LOGGING_TENSOR_BUFFER_H_
