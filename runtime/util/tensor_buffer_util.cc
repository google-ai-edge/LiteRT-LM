#include "runtime/util/tensor_buffer_util.h"

#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

int NumSignificantDims(const ::litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_ABORT(auto tensor_type, tensor_buffer.TensorType());
  int num_significant_dims = 0;
  for (int d : tensor_type.Layout().Dimensions()) {
    num_significant_dims += (d > 1);
  }
  return num_significant_dims;
}

}  // namespace litert::lm
