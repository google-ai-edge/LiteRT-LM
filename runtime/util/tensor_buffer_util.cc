#include "runtime/util/tensor_buffer_util.h"

#include "litert/litert/cc/litert_macros.h"  // from @litert
#include "litert/litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

int NumSignificantDims(const ::litert::TensorBuffer& tensor_buffer) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor_buffer.TensorType());
  auto dims = tensor_type.Layout().Dimensions();
  int num_significant_dims = 0;
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[i] > 1) {
      num_significant_dims++;
    }
  }
  return num_significant_dims;
}

}  // namespace litert::lm
