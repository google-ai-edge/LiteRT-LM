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

#include "runtime/executor/litert/state.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/cc/litert_common.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/test/matchers.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/components/model_resources_litert_lm.h"
#include "runtime/proto/executor_metadata.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/litert_lm_loader.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/test_utils.h"  // IWYU pragma: keep

namespace litert::lm {
namespace {

using ::testing::status::StatusIs;

constexpr char kTestStaticModelPath[] =
    "litert_lm/runtime/testdata/test_lm.litertlm";

// Test model with dynamic sequence and context length dimensions.
constexpr char kTestDynamicModelPath[] =
    "litert_lm/runtime/testdata/test_lm_dynamic.litertlm";

constexpr absl::string_view kDecodeSignatureRunner = "decode";
constexpr absl::string_view kPrefillSignatureRunner = "prefill";

class CompiledModelTest : public CompiledModel {
 public:
  using CompiledModel::CompiledModel;
  using CompiledModel::Create;
};

class LitertStateTest : public ::testing::Test {
 protected:
  void SetUpKV(const std::string& model_path,
               LitertState::AllocationPolicy allocation_policy, int batch_size,
               const proto::ExecutorMetadata* executor_metadata = nullptr,
               bool clear_kv_cache_before_prefill = true) {
    auto path = std::filesystem::path(::testing::SrcDir()) /
                std::filesystem::path(model_path);
    ASSERT_OK_AND_ASSIGN(auto scoped_file, ScopedFile::Open(path.string()));
    ASSERT_OK_AND_ASSIGN(auto loader,
                         LitertLmLoader::Create(std::move(scoped_file)));
    ASSERT_OK_AND_ASSIGN(resources_,
                         ModelResourcesLitertLm::Create(std::move(loader)));
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto env, Environment::Create(std::vector<Environment::Option>()));
    env_ = std::move(env);
    ASSERT_OK_AND_ASSIGN(
        litert_model_,
        resources_->GetTFLiteModel(proto::LlmMetadata::TF_LITE_PREFILL_DECODE));
    LITERT_ASSERT_OK_AND_ASSIGN(auto compilation_options, Options::Create());
    compilation_options.SetHardwareAccelerators(HwAccelerators::kCpu);
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto compiled_model,
        CompiledModelTest::Create(*env_, litert_model_->Get(),
                                  compilation_options));
    compiled_model_ = std::move(compiled_model);
    ASSERT_OK_AND_ASSIGN(
        kv_cache_,
        LitertState::Create(*env_, *compiled_model_, kDecodeSignatureRunner,
                            executor_metadata, allocation_policy, batch_size,
                            clear_kv_cache_before_prefill));
  }

  std::unique_ptr<ModelResources> resources_;
  const litert::Model* litert_model_;
  std::optional<Environment> env_;
  std::optional<CompiledModel> compiled_model_;
  std::unique_ptr<LitertState> kv_cache_;
};

#ifndef _WIN32
TEST_F(LitertStateTest, CanCreateKVWithDynamicModel) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestDynamicModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 1);
}
#endif  // !_WIN32

TEST_F(LitertStateTest, CanCreateKVWithStaticModelOutOfPlace) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 128);
}

TEST_F(LitertStateTest, CanCreateKVWithStaticModelInPlace) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 128);
}

TEST_F(LitertStateTest, CanCreateKVWithClearFlagFalse) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1,
                                  /*executor_metadata=*/nullptr,
                                  /*clear_kv_cache_before_prefill=*/false));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 128);
}

TEST_F(LitertStateTest, SerializeNotSupported) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Serialize(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(LitertStateTest, LoadNotSupported) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Load(""), StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(LitertStateTest, StaticKVNotResizeable) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Resize(*compiled_model_, kDecodeSignatureRunner, 100),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(LitertStateTest, InplaceDynamicKVResizeable) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestDynamicModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 1);
  EXPECT_OK(kv_cache_->Resize(*compiled_model_, kDecodeSignatureRunner, 100));
  EXPECT_EQ(kv_cache_->GetNumEntries(), 100);
}

TEST_F(LitertStateTest, OutOfPlaceDynamicKVNotResizeable) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestDynamicModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->Resize(*compiled_model_, kDecodeSignatureRunner, 100),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

void FillBuffer(TensorBuffer& buffer, uint8_t value) {
  LITERT_ASSERT_OK_AND_ASSIGN(size_t size, buffer.PackedSize());
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kWrite));
  std::memset(lock_and_addr.second, value, size);
}

void ExpectBufferFilledWith(TensorBuffer& buffer, uint8_t value) {
  LITERT_ASSERT_OK_AND_ASSIGN(size_t size, buffer.PackedSize());
  ASSERT_GT(size, 0u);
  LITERT_ASSERT_OK_AND_ASSIGN(
      auto lock_and_addr,
      TensorBufferScopedLock::Create(buffer, TensorBuffer::LockMode::kRead));
  const auto* data = static_cast<const uint8_t*>(lock_and_addr.second);
  for (size_t i = 0; i < size; ++i) {
    ASSERT_EQ(data[i], value) << "Mismatch at byte " << i;
  }
}

TEST_F(LitertStateTest, IsHostMemoryForCpuState) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  EXPECT_TRUE(kv_cache_->IsHostMemory());
}

TEST_F(LitertStateTest, CopyContentsFromKeepsExistingBuffers) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LitertState> source,
      LitertState::Create(*env_, *compiled_model_, kDecodeSignatureRunner,
                          /*executor_metadata=*/nullptr,
                          LitertState::AllocationPolicy::kInplace,
                          /*batch_size=*/1));
  {
    ASSERT_OK_AND_ASSIGN(
        auto source_buffers,
        source->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
    ASSERT_FALSE(source_buffers.input_buffers.empty());
    for (auto& [_, buffer] : source_buffers.input_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 7));
    }
  }
  ASSERT_OK_AND_ASSIGN(
      auto before,
      kv_cache_->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));

  ASSERT_OK(kv_cache_->CopyContentsFrom(*source));

  ASSERT_OK_AND_ASSIGN(
      auto after,
      kv_cache_->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
  ASSERT_EQ(after.input_buffers.size(), before.input_buffers.size());
  for (auto& [name, buffer] : after.input_buffers) {
    // The underlying buffers must not be replaced, only their contents.
    EXPECT_EQ(buffer.Get(), before.input_buffers.at(name).Get()) << name;
    ASSERT_NO_FATAL_FAILURE(ExpectBufferFilledWith(buffer, 7)) << name;
  }
}

TEST_F(LitertStateTest, CopyContentsFromPreservesPingPongInputBank) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LitertState> source,
      LitertState::Create(*env_, *compiled_model_, kDecodeSignatureRunner,
                          /*executor_metadata=*/nullptr,
                          LitertState::AllocationPolicy::kPingPong,
                          /*batch_size=*/1));
  // Run one "step" on the source so that its latest state lives in bank 2,
  // which will be the input bank of the next step.
  {
    ASSERT_OK_AND_ASSIGN(
        auto step,
        source->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
    ASSERT_FALSE(step.output_buffers.empty());
    for (auto& [_, buffer] : step.output_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 9));
    }
  }

  ASSERT_OK(kv_cache_->CopyContentsFrom(*source));

  ASSERT_OK_AND_ASSIGN(
      auto next,
      kv_cache_->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
  ASSERT_FALSE(next.input_buffers.empty());
  for (auto& [name, buffer] : next.input_buffers) {
    ASSERT_NO_FATAL_FAILURE(ExpectBufferFilledWith(buffer, 9)) << name;
  }
}

// GPU command buffers prepared in advance rely on the next runs binding the
// same buffers as before, so restoring contents must not flip which bank the
// destination reads next, even if the source's bank parity differs.
TEST_F(LitertStateTest, CopyContentsFromKeepsDestinationPingPongBankParity) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  // Advance the destination by one step, so that it reads bank 2 next. The
  // buffers written in this step are the ones it must read in the next step.
  absl::flat_hash_map<std::string, LiteRtTensorBuffer> expected_next_inputs;
  {
    ASSERT_OK_AND_ASSIGN(
        auto step,
        kv_cache_->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
    ASSERT_FALSE(step.output_buffers.empty());
    for (auto& [name, buffer] : step.output_buffers) {
      expected_next_inputs[name] = buffer.Get();
    }
  }

  // The source runs two steps, so it reads bank 1 next, i.e. its parity
  // differs from the destination's. Its next input bank (bank 1) holds 3 and
  // its next output bank (bank 2) holds 5.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LitertState> source,
      LitertState::Create(*env_, *compiled_model_, kDecodeSignatureRunner,
                          /*executor_metadata=*/nullptr,
                          LitertState::AllocationPolicy::kPingPong,
                          /*batch_size=*/1));
  ASSERT_TRUE(kv_cache_->CanCopyContentsFrom(*source));
  {
    ASSERT_OK_AND_ASSIGN(
        auto first_step,
        source->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
    for (auto& [_, buffer] : first_step.input_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 3));
    }
    for (auto& [_, buffer] : first_step.output_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 5));
    }
    ASSERT_OK(source->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner)
                  .status());
  }

  ASSERT_OK(kv_cache_->CopyContentsFrom(*source));

  ASSERT_OK_AND_ASSIGN(
      auto next,
      kv_cache_->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
  ASSERT_EQ(next.input_buffers.size(), expected_next_inputs.size());
  for (auto& [name, buffer] : next.input_buffers) {
    EXPECT_EQ(buffer.Get(), expected_next_inputs.at(name)) << name;
    ASSERT_NO_FATAL_FAILURE(ExpectBufferFilledWith(buffer, 3)) << name;
  }
  for (auto& [name, buffer] : next.output_buffers) {
    ASSERT_NO_FATAL_FAILURE(ExpectBufferFilledWith(buffer, 5)) << name;
  }
}

TEST_F(LitertStateTest, CopyContentsFromRejectsMismatchedAllocationPolicy) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LitertState> source,
      LitertState::Create(*env_, *compiled_model_, kDecodeSignatureRunner,
                          /*executor_metadata=*/nullptr,
                          LitertState::AllocationPolicy::kPingPong,
                          /*batch_size=*/1));
  EXPECT_THAT(kv_cache_->CopyContentsFrom(*source),
              StatusIs(absl::StatusCode::kInternal));
}

#ifndef _WIN32
TEST_F(LitertStateTest, CopyContentsFromRejectsMismatchedNumEntries) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestDynamicModelPath,
                                  LitertState::AllocationPolicy::kInplace,
                                  /*batch_size=*/1));
  ASSERT_OK_AND_ASSIGN(auto source_interface, kv_cache_->DeepCopy());
  auto* source = dynamic_cast<LitertState*>(source_interface.get());
  ASSERT_NE(source, nullptr);
  ASSERT_OK(source->Resize(*compiled_model_, kDecodeSignatureRunner, 16));
  EXPECT_THAT(kv_cache_->CopyContentsFrom(*source),
              StatusIs(absl::StatusCode::kInternal));
}
#endif  // !_WIN32

// `SetUpKV` compiles on CPU (`kHostMemory`), so this test verifies that
// `CloneInternal` preserves `kPingPong` bank ordering and buffer isolation;
// device-to-host snapshot conversion and in-place hardware restoration are
// covered on GPU in
// `RestoreHostStateSnapshotIntoGpuExecutorPreservesGpuBuffers`.
TEST_F(LitertStateTest, DeepCopyToHostMemoryPreservesPingPongBanksAndContents) {
  ASSERT_NO_FATAL_FAILURE(SetUpKV(kTestStaticModelPath,
                                  LitertState::AllocationPolicy::kPingPong,
                                  /*batch_size=*/1));
  {
    ASSERT_OK_AND_ASSIGN(
        auto step,
        kv_cache_->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
    ASSERT_FALSE(step.input_buffers.empty());
    ASSERT_FALSE(step.output_buffers.empty());
    for (auto& [_, buffer] : step.input_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 3));
    }
    for (auto& [_, buffer] : step.output_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 11));
    }
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<LitertState> host_copy,
                       kv_cache_->DeepCopyToHostMemoryForTesting());
  ASSERT_NE(host_copy, nullptr);
  EXPECT_TRUE(host_copy->IsHostMemory());
  EXPECT_EQ(host_copy->GetBatchSize(), kv_cache_->GetBatchSize());
  EXPECT_EQ(host_copy->GetNumEntries(), kv_cache_->GetNumEntries());

  // Mutate original buffers after snapshot to verify deep-copy isolation.
  {
    ASSERT_OK_AND_ASSIGN(
        auto mutated,
        kv_cache_->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
    for (auto& [_, buffer] : mutated.input_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 99));
    }
    for (auto& [_, buffer] : mutated.output_buffers) {
      ASSERT_NO_FATAL_FAILURE(FillBuffer(buffer, 99));
    }
  }

  ASSERT_OK_AND_ASSIGN(
      auto copy_step,
      host_copy->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
  for (auto& [name, buffer] : copy_step.input_buffers) {
    ASSERT_NO_FATAL_FAILURE(ExpectBufferFilledWith(buffer, 11)) << name;
  }
  for (auto& [name, buffer] : copy_step.output_buffers) {
    ASSERT_NO_FATAL_FAILURE(ExpectBufferFilledWith(buffer, 3)) << name;
  }
}

}  // namespace
}  // namespace litert::lm
