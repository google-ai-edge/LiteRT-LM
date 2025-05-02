#include "runtime/components/top_k_opencl_sampler.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "third_party/ml_drift/cl/opencl_wrapper.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {
namespace {

TEST(TopKOpenClSamplerTest, CanCreateSuccessfully) {
  ASSERT_OK(ml_drift::cl::LoadOpenCL());
  proto::SamplerParameters sampler_params;
  sampler_params.set_k(1);
  sampler_params.set_p(0.5);
  sampler_params.set_temperature(1.0);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<TopKOpenClSampler> sampler,
      TopKOpenClSampler::Create(
          /*env=*/nullptr, /*batch_size=*/1, /*cache_size=*/100,
          /*vocab_size=*/201,
          /*activation_data_type=*/std::nullopt, sampler_params));
  EXPECT_NE(sampler, nullptr);
}

}  // namespace
}  // namespace litert::lm
