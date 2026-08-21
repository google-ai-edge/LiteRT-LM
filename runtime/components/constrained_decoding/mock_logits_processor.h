#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_MOCK_LOGITS_PROCESSOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_MOCK_LOGITS_PROCESSOR_H_

#include <gmock/gmock.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constrained_decoder.h"
#include "runtime/components/constrained_decoding/logits_processor.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {

class MockLogitsProcessor : public LogitsProcessor {
 public:
  MOCK_METHOD(absl::Status, ProcessLogits, (::litert::TensorBuffer&),
              (override));
  MOCK_METHOD(absl::Status, ProcessLogits,
              (absl::Span<float>, absl::Span<const ::litert::Layout::Dim>),
              (override));
  MOCK_METHOD(absl::Status, ProcessLogits,
              (absl::Span<tflite::half>,
               absl::Span<const ::litert::Layout::Dim>),
              (override));
  MOCK_METHOD(absl::Status, UpdateState, (const ::litert::TensorBuffer&),
              (override));
  MOCK_METHOD(absl::Status, UpdateState, (absl::Span<int>), (override));
  MOCK_METHOD(ConstrainedDecoder*, GetConstraintDecoder, (), (override));
  MOCK_METHOD(const ConstrainedDecoder*, GetConstraintDecoder, (),
              (const, override));
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_CONSTRAINED_DECODING_MOCK_LOGITS_PROCESSOR_H_
