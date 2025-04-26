#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_

#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

// Returns the number of dimensions that are greater than 1 in the given
// tensor buffer.
int NumSignificantDims(const litert::TensorBuffer& tensor_buffer);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_TENSOR_BUFFER_UTIL_H_
