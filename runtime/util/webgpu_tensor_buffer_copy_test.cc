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

#include "runtime/util/webgpu_tensor_buffer_copy.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "ml_drift/webgpu/environment.h"  // from @ml_drift
#include "ml_drift/webgpu/webgpu_headers.h"  // from @ml_drift
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

namespace litert::lm {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

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

std::atomic<int> g_device_copy_call_count{0};

absl::StatusOr<bool> CountingWebGpuCopyWrapper(const Environment& env,
                                               TensorBuffer& dst,
                                               const TensorBuffer& src) {
  g_device_copy_call_count.fetch_add(1, std::memory_order_relaxed);
  return CopyWebGpuTensorBuffer(env, dst, src);
}

absl::StatusOr<bool> FailingWebGpuCopyWrapper(const Environment& env,
                                              TensorBuffer& dst,
                                              const TensorBuffer& src) {
  return absl::InternalError("Injected WebGPU copy failure");
}

struct CountingCustomEvent : public LiteRtCustomEventT {
  std::atomic<int>* wait_counter = nullptr;

  static void NoopRetain(LiteRtCustomEvent event) {}
  static void NoopRelease(LiteRtCustomEvent event) {}
  static void RecordWait(LiteRtCustomEvent event, int64_t timeout_in_ms) {
    auto* self = static_cast<CountingCustomEvent*>(event);
    self->wait_counter->fetch_add(1, std::memory_order_relaxed);
  }
  static int AlwaysSignaled(LiteRtCustomEvent event) { return 1; }
  static void* NullNative(LiteRtCustomEvent event) { return nullptr; }

  explicit CountingCustomEvent(std::atomic<int>* counter)
      : wait_counter(counter) {
    Retain = &NoopRetain;
    Release = &NoopRelease;
    Wait = &RecordWait;
    IsSignaled = &AlwaysSignaled;
    GetNative = &NullNative;
  }
};

class WebGpuTensorBufferCopyNullBackendTest : public ::testing::Test {
 protected:
  void SetUp() override {
    absl::Status init_status = wgpu_env_.Initialize();
    if (!init_status.ok()) {
      GTEST_SKIP() << "Dawn Null backend unavailable: " << init_status;
    }

    const std::array<EnvironmentOptions::Option, 2> options = {
        EnvironmentOptions::Option(
            EnvironmentOptions::Tag::kWebGpuDevice,
            LiteRtVariant(static_cast<int64_t>(
                reinterpret_cast<intptr_t>(wgpu_env_.device().Get())))),
        EnvironmentOptions::Option(
            EnvironmentOptions::Tag::kWebGpuQueue,
            LiteRtVariant(static_cast<int64_t>(
                reinterpret_cast<intptr_t>(wgpu_env_.queue().Get())))),
    };
    LITERT_ASSERT_OK_AND_ASSIGN(
        Environment env,
        Environment::Create(EnvironmentOptions(absl::MakeConstSpan(options))));
    env_.emplace(std::move(env));
  }

  ::ml_drift::webgpu::Environment wgpu_env_{wgpu::BackendType::Null};
  std::optional<Environment> env_;
};

TEST_F(WebGpuTensorBufferCopyNullBackendTest,
       AutoRegisteredAndUsedByCopyTensorBufferAndContents) {
  const RankedTensorType tensor_type = MakeRankedTensorType<float>({1, 4});
  constexpr size_t kBufferBytes = 4 * sizeof(float);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer src,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  tensor_type, kBufferBytes));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer dst,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  tensor_type, kBufferBytes));

  // Verify that static initialization registered `CopyWebGpuTensorBuffer`.
  ScopedWebGpuCopyFnOverride override_fn(&CountingWebGpuCopyWrapper);
  EXPECT_EQ(override_fn.previous, &CopyWebGpuTensorBuffer);

  g_device_copy_call_count.store(0, std::memory_order_relaxed);
  ASSERT_OK_AND_ASSIGN(TensorBuffer cloned, CopyTensorBuffer(*env_, src));
  EXPECT_EQ(g_device_copy_call_count.load(std::memory_order_relaxed), 1);
  EXPECT_NE(cloned.Get(), src.Get());

  const auto* original_dst_handle = dst.Get();
  ASSERT_OK(CopyTensorBufferContents(*env_, dst, src));
  EXPECT_EQ(g_device_copy_call_count.load(std::memory_order_relaxed), 2);
  EXPECT_EQ(dst.Get(), original_dst_handle);
}

TEST_F(WebGpuTensorBufferCopyNullBackendTest,
       WaitsOnAttachedEventsAndPropagatesEventErrors) {
  const RankedTensorType tensor_type = MakeRankedTensorType<float>({4});
  constexpr size_t kBufferBytes = 4 * sizeof(float);

  // Declare custom events before `src` and `dst` so they outlive the
  // `LiteRtEventT` handles released by `~TensorBuffer()`.
  std::atomic<int> wait_count{0};
  CountingCustomEvent src_custom_event(&wait_count);
  CountingCustomEvent dst_custom_event(&wait_count);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer src,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  tensor_type, kBufferBytes));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer dst,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  tensor_type, kBufferBytes));

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event src_event, Event::CreateManaged(*env_, Event::Type::kCustom));
  LITERT_ASSERT_OK(src_event.SetCustomEvent(&src_custom_event));
  LITERT_ASSERT_OK(src.SetEvent(std::move(src_event)));

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event dst_event, Event::CreateManaged(*env_, Event::Type::kCustom));
  LITERT_ASSERT_OK(dst_event.SetCustomEvent(&dst_custom_event));
  LITERT_ASSERT_OK(dst.SetEvent(std::move(dst_event)));

  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, dst, src), IsOkAndHolds(true));
  EXPECT_EQ(wait_count.load(std::memory_order_relaxed), 2);

  // An unset custom event fails on Wait(), and the failure must propagate.
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event unset_event, Event::CreateManaged(*env_, Event::Type::kCustom));
  LITERT_ASSERT_OK(src.SetEvent(std::move(unset_event)));
  EXPECT_FALSE(CopyWebGpuTensorBuffer(*env_, dst, src).ok());
}

TEST_F(WebGpuTensorBufferCopyNullBackendTest,
       HandlesFallbacksSameBufferAndErrorPropagation) {
  const RankedTensorType tensor_type = MakeRankedTensorType<float>({4});
  constexpr size_t kBufferBytes = 4 * sizeof(float);

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer src,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  tensor_type, kBufferBytes));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer dst,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  tensor_type, kBufferBytes));

  // Same TensorBuffer handle returns true immediately.
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, dst, dst), IsOkAndHolds(true));

  // Host memory buffer returns false so callers fall back to host lock copy.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer host_buf,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kHostMemory,
                                  tensor_type, kBufferBytes));
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, dst, host_buf),
              IsOkAndHolds(false));

  // Environment without kWebGpuDevice returns false (graceful fallback).
  LITERT_ASSERT_OK_AND_ASSIGN(Environment empty_env, Environment::Create({}));
  EXPECT_THAT(CopyWebGpuTensorBuffer(empty_env, dst, src), IsOkAndHolds(false));

  // Environment passing device/queue as `const void*` succeeds as well.
  const std::array<EnvironmentOptions::Option, 2> ptr_options = {
      EnvironmentOptions::Option(EnvironmentOptions::Tag::kWebGpuDevice,
                                 LiteRtVariant(reinterpret_cast<const void*>(
                                     wgpu_env_.device().Get()))),
      EnvironmentOptions::Option(EnvironmentOptions::Tag::kWebGpuQueue,
                                 LiteRtVariant(reinterpret_cast<const void*>(
                                     wgpu_env_.queue().Get()))),
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment ptr_env,
                              Environment::Create(EnvironmentOptions(
                                  absl::MakeConstSpan(ptr_options))));
  EXPECT_THAT(CopyWebGpuTensorBuffer(ptr_env, dst, src), IsOkAndHolds(true));

  // Different WebGPU buffer types return false.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer webgpu_fp16,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferFp16,
                                  tensor_type, kBufferBytes));
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, dst, webgpu_fp16),
              IsOkAndHolds(false));

  // Mismatched sizes return InvalidArgumentError.
  const RankedTensorType larger_type = MakeRankedTensorType<float>({8});
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer larger_buf,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  larger_type, 8 * sizeof(float)));
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, dst, larger_buf),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Two TensorBuffers importing the same underlying WGPUBuffer at offset 0 are
  // treated as a no-op without issuing an invalid self-copy on WebGPU.
  wgpu::BufferDescriptor shared_desc{
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
               wgpu::BufferUsage::CopyDst,
      .size = kBufferBytes,
  };
  wgpu::Buffer shared_wgpu_buf = wgpu_env_.device().CreateBuffer(&shared_desc);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer imported_a,
      TensorBuffer::CreateFromWebGpuBuffer(
          *env_, tensor_type, TensorBufferType::kWebGpuBufferPacked,
          shared_wgpu_buf.Get(), kBufferBytes));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer imported_b,
      TensorBuffer::CreateFromWebGpuBuffer(
          *env_, tensor_type, TensorBufferType::kWebGpuBufferPacked,
          shared_wgpu_buf.Get(), kBufferBytes));
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, imported_a, imported_b),
              IsOkAndHolds(true));

  // Two distinct imported packed `float[4]` buffers sized to the packed size
  // (16 bytes, smaller than the PHWC4-padded `GetMemorySizeInBytes()`) are
  // copied on the device instead of falling back to a host copy.
  wgpu::Buffer packed_wgpu_a = wgpu_env_.device().CreateBuffer(&shared_desc);
  wgpu::Buffer packed_wgpu_b = wgpu_env_.device().CreateBuffer(&shared_desc);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer imported_packed_a,
      TensorBuffer::CreateFromWebGpuBuffer(
          *env_, tensor_type, TensorBufferType::kWebGpuBufferPacked,
          packed_wgpu_a.Get(), kBufferBytes));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer imported_packed_b,
      TensorBuffer::CreateFromWebGpuBuffer(
          *env_, tensor_type, TensorBufferType::kWebGpuBufferPacked,
          packed_wgpu_b.Get(), kBufferBytes));
  EXPECT_THAT(
      CopyWebGpuTensorBuffer(*env_, imported_packed_a, imported_packed_b),
      IsOkAndHolds(true));

  // A managed packed destination whose reported `Size()` is larger than an
  // imported packed source with the same `PackedSize()` still copies on device
  // (matching the host path, which compares `PackedSize()`).
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer larger_size_packed_dst,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  tensor_type, kBufferBytes * 4));
  EXPECT_THAT(
      CopyWebGpuTensorBuffer(*env_, larger_size_packed_dst, imported_packed_b),
      IsOkAndHolds(true));
  ASSERT_OK(CopyTensorBufferContents(*env_, larger_size_packed_dst,
                                     imported_packed_b));

  // Same byte size but different element types return InvalidArgumentError.
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer int_buf,
      TensorBuffer::CreateManaged(*env_, TensorBufferType::kWebGpuBufferPacked,
                                  MakeRankedTensorType<int32_t>({4}),
                                  kBufferBytes));
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, dst, int_buf),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // Buffer missing CopySrc usage falls back cleanly (`false`).
  wgpu::BufferDescriptor no_copy_src_desc{
      .usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
      .size = kBufferBytes,
  };
  wgpu::Buffer no_copy_src_wgpu =
      wgpu_env_.device().CreateBuffer(&no_copy_src_desc);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer no_copy_src_tb,
      TensorBuffer::CreateFromWebGpuBuffer(
          *env_, tensor_type, TensorBufferType::kWebGpuBufferPacked,
          no_copy_src_wgpu.Get(), kBufferBytes));
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, dst, no_copy_src_tb),
              IsOkAndHolds(false));

  // Texture-backed WebGPU TensorBuffer falls back cleanly (`false`).
  wgpu::TextureDescriptor tex_desc{
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {.width = 1, .height = 1, .depthOrArrayLayers = 1},
      .format = wgpu::TextureFormat::RGBA32Float,
  };
  wgpu::Texture wgpu_tex = wgpu_env_.device().CreateTexture(&tex_desc);
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tex_tb,
      TensorBuffer::CreateFromWebGpuTexture(*env_, tensor_type, wgpu_tex.Get(),
                                            kBufferBytes));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer tex_tb_2,
      TensorBuffer::CreateFromWebGpuTexture(*env_, tensor_type, wgpu_tex.Get(),
                                            kBufferBytes));
  EXPECT_THAT(CopyWebGpuTensorBuffer(*env_, tex_tb, tex_tb_2),
              IsOkAndHolds(false));

  // Errors from the registered callback propagate through CopyTensorBuffer and
  // CopyTensorBufferContents.
  ScopedWebGpuCopyFnOverride failing_override(&FailingWebGpuCopyWrapper);
  EXPECT_THAT(CopyTensorBuffer(*env_, src),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(CopyTensorBufferContents(*env_, dst, src),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace litert::lm
