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

// Tests of the parts of the NVML reader that do not need a GPU. The live
// check is accelerator_memory_nvidia_test.

#include "runtime/util/memory_profiling/accelerator_memory_nvml.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace litert::lm {
namespace {

constexpr uint64_t kMb = 1024 * 1024;
constexpr unsigned int kPid = 317809;

TEST(NvmlProcessBytesTest, ProcessInBothListsIsCountedOnce) {
  // A "C+G" process (OpenCL plus EGL, as LiteRT-LM's GPU backend on Linux):
  // the driver puts its whole 10908 MiB in both lists, nvidia-smi shows
  // 10908 MiB once, and so must we rather than 21816 MiB.
  const std::vector<NvmlProcessEntry> compute = {
      {.pid = 13343, .used_bytes = 13 * kMb},
      {.pid = kPid, .used_bytes = 10908 * kMb}};
  const std::vector<NvmlProcessEntry> graphics = {
      {.pid = 6989, .used_bytes = 139 * kMb},
      {.pid = 6991, .used_bytes = 136 * kMb},
      {.pid = 13343, .used_bytes = 13 * kMb},
      {.pid = kPid, .used_bytes = 10908 * kMb}};

  EXPECT_EQ(NvmlProcessBytes(kPid, compute, graphics), 10908 * kMb);
}

TEST(NvmlProcessBytesTest, ProcessInOneListIsThatEntry) {
  const std::vector<NvmlProcessEntry> compute = {
      {.pid = kPid, .used_bytes = 772 * kMb}};
  const std::vector<NvmlProcessEntry> graphics = {
      {.pid = 6989, .used_bytes = 139 * kMb}};

  EXPECT_EQ(NvmlProcessBytes(kPid, compute, graphics), 772 * kMb);
  EXPECT_EQ(NvmlProcessBytes(kPid, {}, compute), 772 * kMb);
}

TEST(NvmlProcessBytesTest, AbsentProcessUsesNothing) {
  const std::vector<NvmlProcessEntry> graphics = {
      {.pid = 6989, .used_bytes = 139 * kMb}};

  EXPECT_EQ(NvmlProcessBytes(kPid, {}, graphics), 0);
  EXPECT_EQ(NvmlProcessBytes(kPid, {}, {}), 0);
}

TEST(NvmlProcessBytesTest, DifferingEntriesTakeTheLarger) {
  // Not observed from the driver, but if the two lists ever disagree the
  // reader must not add them up.
  const std::vector<NvmlProcessEntry> compute = {
      {.pid = kPid, .used_bytes = 100 * kMb}};
  const std::vector<NvmlProcessEntry> graphics = {
      {.pid = kPid, .used_bytes = 120 * kMb}};

  EXPECT_EQ(NvmlProcessBytes(kPid, compute, graphics), 120 * kMb);
}

}  // namespace
}  // namespace litert::lm
