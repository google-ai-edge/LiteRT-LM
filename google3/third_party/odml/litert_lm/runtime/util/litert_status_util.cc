#include "third_party/odml/litert_lm/runtime/util/litert_status_util.h"

#include "third_party/absl/status/status.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::lm {

absl::Status ToAbslStatus(const litert::Error& error) {
#define LITERT_ERROR_TO_ABSL_MESSAGE(error) (error).Message()
  switch (error.Status()) {
    case kLiteRtStatusOk:
      return absl::OkStatus();
    case kLiteRtStatusErrorInvalidArgument:
      return absl::InvalidArgumentError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorMemoryAllocationFailure:
      return absl::ResourceExhaustedError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorRuntimeFailure:
      return absl::InternalError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorMissingInputTensor:
      return absl::InvalidArgumentError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorUnsupported:
      return absl::UnimplementedError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorNotFound:
      return absl::NotFoundError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorTimeoutExpired:
      return absl::DeadlineExceededError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorWrongVersion:
      return absl::FailedPreconditionError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorUnknown:
      return absl::UnknownError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorFileIO:
      return absl::UnavailableError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorInvalidFlatbuffer:
      return absl::InvalidArgumentError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorDynamicLoading:
      return absl::UnavailableError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorSerialization:
      return absl::InternalError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorCompilation:
      return absl::InternalError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorIndexOOB:
      return absl::OutOfRangeError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorInvalidIrType:
      return absl::InvalidArgumentError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorInvalidGraphInvariant:
      return absl::InvalidArgumentError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorGraphModification:
      return absl::InternalError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorInvalidToolConfig:
      return absl::InvalidArgumentError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusLegalizeNoMatch:
      return absl::NotFoundError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    case kLiteRtStatusErrorInvalidLegalization:
      return absl::InvalidArgumentError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
    default:
      return absl::UnknownError(LITERT_ERROR_TO_ABSL_MESSAGE(error));
  }
#undef LITERT_ERROR_TO_ABSL_MESSAGE
}

absl::Status ToAbslStatus(litert::Expected<void> expected) {
  return expected ? absl::OkStatus() : ToAbslStatus(expected.Error());
}

}  // namespace litert::lm
