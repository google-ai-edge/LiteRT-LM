// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "litert/c/litert_event_type.h"  // from @litert
#include "litert/cc/litert_any.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_event.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/util/tensor_buffer_util.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep
#include "runtime/util/webgpu_tensor_buffer_copy.h"

namespace litert::lm {
namespace {

using ::testing::ElementsAre;

std::atomic<int> g_gpu_copy_calls{0};

absl::StatusOr<bool> CountingRealWebGpuCopy(const Environment& env,
                                            TensorBuffer& dst,
                                            const TensorBuffer& src) {
  g_gpu_copy_calls.fetch_add(1, std::memory_order_relaxed);
  return CopyWebGpuTensorBuffer(env, dst, src);
}

struct NoopCustomEvent : public LiteRtCustomEventT {
  static void NoopRetain(LiteRtCustomEvent event) {}
  static void NoopRelease(LiteRtCustomEvent event) {}
  static void NoopWait(LiteRtCustomEvent event, int64_t timeout_in_ms) {}
  static int AlwaysSignaled(LiteRtCustomEvent event) { return 1; }
  static void* NullNative(LiteRtCustomEvent event) { return nullptr; }

  NoopCustomEvent() {
    Retain = &NoopRetain;
    Release = &NoopRelease;
    Wait = &NoopWait;
    IsSignaled = &AlwaysSignaled;
    GetNative = &NullNative;
  }
};

struct ScopedWebGpuCopyFnOverride {
  DeviceTensorBufferCopyFn previous;

  explicit ScopedWebGpuCopyFnOverride(DeviceTensorBufferCopyFn replacement)
      : previous(RegisterDeviceTensorBufferCopyFn(
            DeviceTensorBufferCopyBackend::kWebGpu, replacement)) {}

  ~ScopedWebGpuCopyFnOverride() {
    RegisterDeviceTensorBufferCopyFn(DeviceTensorBufferCopyBackend::kWebGpu,
                                     previous);
  }
};

TEST(WebGpuTensorBufferCopyGpuTest,
     CopiesWebGpuBuffersOnDeviceAndPreservesHandles) {
  ::ml_drift::webgpu::Environment wgpu_env;
  ASSERT_OK(wgpu_env.Initialize());

  const std::array<EnvironmentOptions::Option, 2> options = {
      EnvironmentOptions::Option(
          EnvironmentOptions::Tag::kWebGpuDevice,
          LiteRtVariant(static_cast<int64_t>(
              reinterpret_cast<intptr_t>(wgpu_env.device().Get())))),
      EnvironmentOptions::Option(
          EnvironmentOptions::Tag::kWebGpuQueue,
          LiteRtVariant(static_cast<int64_t>(
              reinterpret_cast<intptr_t>(wgpu_env.queue().Get())))),
  };
  LITERT_ASSERT_OK_AND_ASSIGN(
      Environment env,
      Environment::Create(EnvironmentOptions(absl::MakeConstSpan(options))));

  ScopedWebGpuCopyFnOverride override_fn(&CountingRealWebGpuCopy);
  EXPECT_EQ(override_fn.previous, &CopyWebGpuTensorBuffer);
  g_gpu_copy_calls.store(0, std::memory_order_relaxed);

  const RankedTensorType tensor_type = MakeRankedTensorType<float>({1, 2, 4});
  constexpr size_t kBufferBytes = 8 * sizeof(float);

  for (TensorBufferType buffer_type : {TensorBufferType::kWebGpuBufferPacked,
                                       TensorBufferType::kWebGpuBuffer}) {
    LITERT_ASSERT_OK_AND_ASSIGN(
        TensorBuffer src, TensorBuffer::CreateManaged(
                              env, buffer_type, tensor_type, kBufferBytes));
    const std::vector<float> initial = {1.5f, -2.0f, 3.25f, 4.0f,
                                        5.5f, -6.0f, 7.75f, 8.0f};
    LITERT_ASSERT_OK(src.Write<float>(initial));

    // Attach a custom event (matching WebGPU delegate output buffers) to verify
    // events do not force a fallback to host staging.
    NoopCustomEvent custom_event;
    LITERT_ASSERT_OK_AND_ASSIGN(
        Event src_event, Event::CreateManaged(env, Event::Type::kCustom));
    LITERT_ASSERT_OK(src_event.SetCustomEvent(&custom_event));
    LITERT_ASSERT_OK(src.SetEvent(std::move(src_event)));

    const int calls_before_clone =
        g_gpu_copy_calls.load(std::memory_order_relaxed);
    ASSERT_OK_AND_ASSIGN(TensorBuffer cloned, CopyTensorBuffer(env, src));
    EXPECT_EQ(g_gpu_copy_calls.load(std::memory_order_relaxed),
              calls_before_clone + 1);

    std::vector<float> cloned_readback(8, 0.0f);
    LITERT_ASSERT_OK(cloned.Read<float>(absl::MakeSpan(cloned_readback)));
    EXPECT_THAT(cloned_readback, ElementsAre(1.5f, -2.0f, 3.25f, 4.0f, 5.5f,
                                             -6.0f, 7.75f, 8.0f));

    // Mutate `src` and copy in place into `cloned`, verifying `cloned` keeps
    // both its LiteRtTensorBuffer and underlying SpatialTensor handle.
    const std::vector<float> updated = {10.0f, 20.0f, 30.0f, 40.0f,
                                        50.0f, 60.0f, 70.0f, 80.0f};
    LITERT_ASSERT_OK(src.Write<float>(updated));

    const auto* original_tb_handle = cloned.Get();
    LITERT_ASSERT_OK_AND_ASSIGN(void* original_wgpu_handle,
                                cloned.GetWebGpuBuffer());
    const int calls_before_restore =
        g_gpu_copy_calls.load(std::memory_order_relaxed);
    ASSERT_OK(CopyTensorBufferContents(env, cloned, src));
    EXPECT_EQ(g_gpu_copy_calls.load(std::memory_order_relaxed),
              calls_before_restore + 1);
    EXPECT_EQ(cloned.Get(), original_tb_handle);
    LITERT_ASSERT_OK_AND_ASSIGN(void* after_wgpu_handle,
                                cloned.GetWebGpuBuffer());
    EXPECT_EQ(after_wgpu_handle, original_wgpu_handle);

    std::vector<float> restored_readback(8, 0.0f);
    LITERT_ASSERT_OK(cloned.Read<float>(absl::MakeSpan(restored_readback)));
    EXPECT_THAT(restored_readback, ElementsAre(10.0f, 20.0f, 30.0f, 40.0f,
                                               50.0f, 60.0f, 70.0f, 80.0f));
  }
}

}  // namespace
}  // namespace litert::lm
