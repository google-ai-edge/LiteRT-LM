// Copyright 2024 The ODML Authors.
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

#include "runtime/executor/gpu_artisan/ml_drift/tasks/rotqat_rotate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_replace.h"  // from @com_google_absl
#include "absl/strings/substitute.h"  // from @com_google_absl
#include "ml_drift/common/data_type.h"  // from @ml_drift
#include "ml_drift/common/gpu_info.h"  // from @ml_drift
#include "ml_drift/common/shape.h"  // from @ml_drift
#include "ml_drift/common/task/gpu_operation.h"  // from @ml_drift
#include "ml_drift/common/task/tensor_desc.h"  // from @ml_drift
#include "ml_drift/common/types.h"  // from @ml_drift
#include "runtime/util/status_macros.h"

namespace ml_drift {
namespace {

// A 7x7 orthogonal matrix used in a matrix multiplication during the first part
// of the ROTQAT operation. Dividing by 5 makes the matrix orthonormal.
constexpr int kOrtho7[7][7] = {
    {2, 4, 1, -1, -1, -1, -1},   {2, -1, 1, -1, 4, -1, -1},
    {-2, 1, -1, -4, 1, 1, 1},    {2, -1, 1, -1, -1, -1, 4},
    {1, -2, -2, -2, -2, -2, -2}, {-2, -1, 4, -1, -1, -1, -1},
    {-2, 1, -1, 1, 1, -4, 1}};

std::string GenerateMatVecMul7Code(std::array<std::string, 7> src_names,
                                   std::array<std::string, 7> dst_names) {
  std::string c;
  for (int i = 0; i < 7; ++i) {
    absl::StrAppend(&c, "  ", dst_names[i], " = ucl::Init<SType>(0);\n");
    for (int j = 0; j < 7; ++j) {
      absl::StrAppend(&c, "  ", dst_names[i], " += ", src_names[j], " * ",
                      kOrtho7[j][i], ";\n");
    }
  }
  return c;
}

std::string GenerateFusedCode(int reduce_size, int wg_size_z) {
  std::string c = absl::Substitute(R"(
  int DST_Y = ucl::GetGlobalId<1>();
  int DST_S = 7 * ucl::GetGlobalId<2>();

  int hadamard_batch = ucl::GetGlobalId<2>() / HADAMARD_DIM;
  int start_index =
    ucl::GetLocalId<2>() * 7 + hadamard_batch * HADAMARD_DIM * 7;

  Type src_val[$0];
  Type dst_val[$0];
)", 7 * reduce_size);

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < reduce_size; ++j) {
      // Scale by 1/5 to normalize `kOrtho7`.
      absl::StrAppend(&c, "  src_val[", j * 7 + i,
                      "] = args.src_tensor.Read(DST_X, DST_Y, start_index + ",
                      wg_size_z * 7 * j + i, ") / 5;\n");
    }
  }

  // multiply src_vals by `kOrtho7`
  for (int j = 0; j < reduce_size; ++j) {
    c += "{\n";
    int offset = j * 7;
    absl::StrAppend(
        &c,
        GenerateMatVecMul7Code(
            {"src_val[OFFSET0].x", "src_val[OFFSET0].y", "src_val[OFFSET0].z",
             "src_val[OFFSET0].w", "src_val[OFFSET1].x", "src_val[OFFSET1].y",
             "src_val[OFFSET1].z"},
            {"dst_val[OFFSET0].x", "dst_val[OFFSET0].y", "dst_val[OFFSET0].z",
             "dst_val[OFFSET0].w", "dst_val[OFFSET1].x", "dst_val[OFFSET1].y",
             "dst_val[OFFSET1].z"}),
        GenerateMatVecMul7Code(
            {"src_val[OFFSET1].w", "src_val[OFFSET2].x", "src_val[OFFSET2].y",
             "src_val[OFFSET2].z", "src_val[OFFSET2].w", "src_val[OFFSET3].x",
             "src_val[OFFSET3].y"},
            {"dst_val[OFFSET1].w", "dst_val[OFFSET2].x", "dst_val[OFFSET2].y",
             "dst_val[OFFSET2].z", "dst_val[OFFSET2].w", "dst_val[OFFSET3].x",
             "dst_val[OFFSET3].y"}),
        GenerateMatVecMul7Code(
            {"src_val[OFFSET3].z", "src_val[OFFSET3].w", "src_val[OFFSET4].x",
             "src_val[OFFSET4].y", "src_val[OFFSET4].z", "src_val[OFFSET4].w",
             "src_val[OFFSET5].x"},
            {"dst_val[OFFSET3].z", "dst_val[OFFSET3].w", "dst_val[OFFSET4].x",
             "dst_val[OFFSET4].y", "dst_val[OFFSET4].z", "dst_val[OFFSET4].w",
             "dst_val[OFFSET5].x"}),
        GenerateMatVecMul7Code(
            {"src_val[OFFSET5].y", "src_val[OFFSET5].z", "src_val[OFFSET5].w",
             "src_val[OFFSET6].x", "src_val[OFFSET6].y", "src_val[OFFSET6].z",
             "src_val[OFFSET6].w"},
            {"dst_val[OFFSET5].y", "dst_val[OFFSET5].z", "dst_val[OFFSET5].w",
             "dst_val[OFFSET6].x", "dst_val[OFFSET6].y", "dst_val[OFFSET6].z",
             "dst_val[OFFSET6].w"}));

    // Scale by 1/2 to normalize 4x4 Hadamard Transform.
    for (int i = 0; i < 7; ++i) {
      absl::StrAppend(&c, "  dst_val[", offset + i, "] /= 2;\n");
    }

    // Apply 4x4 Hadamard Transform.
    c += R"(
  dst_val[OFFSET0].x += dst_val[OFFSET1].w;
  dst_val[OFFSET1].w = dst_val[OFFSET0].x - 2 * dst_val[OFFSET1].w;
  dst_val[OFFSET0].y += dst_val[OFFSET2].x;
  dst_val[OFFSET2].x = dst_val[OFFSET0].y - 2 * dst_val[OFFSET2].x;
  dst_val[OFFSET0].z += dst_val[OFFSET2].y;
  dst_val[OFFSET2].y = dst_val[OFFSET0].z - 2 * dst_val[OFFSET2].y;
  dst_val[OFFSET0].w += dst_val[OFFSET2].z;
  dst_val[OFFSET2].z = dst_val[OFFSET0].w - 2 * dst_val[OFFSET2].z;
  dst_val[OFFSET1].x += dst_val[OFFSET2].w;
  dst_val[OFFSET2].w = dst_val[OFFSET1].x - 2 * dst_val[OFFSET2].w;
  dst_val[OFFSET1].y += dst_val[OFFSET3].x;
  dst_val[OFFSET3].x = dst_val[OFFSET1].y - 2 * dst_val[OFFSET3].x;
  dst_val[OFFSET1].z += dst_val[OFFSET3].y;
  dst_val[OFFSET3].y = dst_val[OFFSET1].z - 2 * dst_val[OFFSET3].y;

  dst_val[OFFSET3].z += dst_val[OFFSET5].y;
  dst_val[OFFSET5].y = dst_val[OFFSET3].z - 2 * dst_val[OFFSET5].y;
  dst_val[OFFSET3].w += dst_val[OFFSET5].z;
  dst_val[OFFSET5].z = dst_val[OFFSET3].w - 2 * dst_val[OFFSET5].z;
  dst_val[OFFSET4].x += dst_val[OFFSET5].w;
  dst_val[OFFSET5].w = dst_val[OFFSET4].x - 2 * dst_val[OFFSET5].w;
  dst_val[OFFSET4].y += dst_val[OFFSET6].x;
  dst_val[OFFSET6].x = dst_val[OFFSET4].y - 2 * dst_val[OFFSET6].x;
  dst_val[OFFSET4].z += dst_val[OFFSET6].y;
  dst_val[OFFSET6].y = dst_val[OFFSET4].z - 2 * dst_val[OFFSET6].y;
  dst_val[OFFSET4].w += dst_val[OFFSET6].z;
  dst_val[OFFSET6].z = dst_val[OFFSET4].w - 2 * dst_val[OFFSET6].z;
  dst_val[OFFSET5].x += dst_val[OFFSET6].w;
  dst_val[OFFSET6].w = dst_val[OFFSET5].x - 2 * dst_val[OFFSET6].w;

  dst_val[OFFSET0].x += dst_val[OFFSET3].z;
  dst_val[OFFSET3].z = dst_val[OFFSET0].x - 2 * dst_val[OFFSET3].z;
  dst_val[OFFSET0].y += dst_val[OFFSET3].w;
  dst_val[OFFSET3].w = dst_val[OFFSET0].y - 2 * dst_val[OFFSET3].w;
  dst_val[OFFSET0].z += dst_val[OFFSET4].x;
  dst_val[OFFSET4].x = dst_val[OFFSET0].z - 2 * dst_val[OFFSET4].x;
  dst_val[OFFSET0].w += dst_val[OFFSET4].y;
  dst_val[OFFSET4].y = dst_val[OFFSET0].w - 2 * dst_val[OFFSET4].y;
  dst_val[OFFSET1].x += dst_val[OFFSET4].z;
  dst_val[OFFSET4].z = dst_val[OFFSET1].x - 2 * dst_val[OFFSET4].z;
  dst_val[OFFSET1].y += dst_val[OFFSET4].w;
  dst_val[OFFSET4].w = dst_val[OFFSET1].y - 2 * dst_val[OFFSET4].w;
  dst_val[OFFSET1].z += dst_val[OFFSET5].x;
  dst_val[OFFSET5].x = dst_val[OFFSET1].z - 2 * dst_val[OFFSET5].x;

  dst_val[OFFSET1].w += dst_val[OFFSET5].y;
  dst_val[OFFSET5].y = dst_val[OFFSET1].w - 2 * dst_val[OFFSET5].y;
  dst_val[OFFSET2].x += dst_val[OFFSET5].z;
  dst_val[OFFSET5].z = dst_val[OFFSET2].x - 2 * dst_val[OFFSET5].z;
  dst_val[OFFSET2].y += dst_val[OFFSET5].w;
  dst_val[OFFSET5].w = dst_val[OFFSET2].y - 2 * dst_val[OFFSET5].w;
  dst_val[OFFSET2].z += dst_val[OFFSET6].x;
  dst_val[OFFSET6].x = dst_val[OFFSET2].z - 2 * dst_val[OFFSET6].x;
  dst_val[OFFSET2].w += dst_val[OFFSET6].y;
  dst_val[OFFSET6].y = dst_val[OFFSET2].w - 2 * dst_val[OFFSET6].y;
  dst_val[OFFSET3].x += dst_val[OFFSET6].z;
  dst_val[OFFSET6].z = dst_val[OFFSET3].x - 2 * dst_val[OFFSET6].z;
  dst_val[OFFSET3].y += dst_val[OFFSET6].w;
  dst_val[OFFSET6].w = dst_val[OFFSET3].y - 2 * dst_val[OFFSET6].w;
)";
    c += "}\n";
    std::vector<std::pair<std::string, std::string>> replacements(7);
    for (int i = 0; i < 7; ++i) {
      replacements.push_back(
          {absl::StrCat("OFFSET", i), std::to_string(offset + i)});
    }
    absl::StrReplaceAll(replacements, &c);
  }

  // Apply reduce_size x reduce_size Hadamard Transform across workgroups.
  c += R"(
  __local Type loc_mem[7][WG_SIZE_X][WG_SIZE_Z];
  int tid_x = ucl::GetLocalId<0>();
  int tid_z = ucl::GetLocalId<2>();
  int group_id = ucl::GetGroupId<2>();
  Type reduce = ucl::Init<Type>(0);
)";
  for (int i = 0; i < 7; ++i) {
    c += "  reduce = ucl::Init<Type>(0);\n";
    for (int j = 0; j < reduce_size; ++j) {
      // The (m, n)-th entry of a Hadamard Matrix is equal to
      // (-1)**{parity of m & n}.
      c += absl::Substitute(R"(
  {
    // Compute parity of number of 1s in bit representation of group_id & $0.
    int v = group_id & $0;
    v ^= v >> 1;
    v ^= v >> 2;
    v = (v & 0x11111111) * 0x11111111;
    v = (v >> 28) & 1;
    SType hadamard_coeff = ucl::Init<SType>(1 - 2 * v);
    SType hadamard_scale = ucl::Init<SType>($2);
    reduce += hadamard_coeff * dst_val[$1] / hadamard_scale;
  })",
                            j, j * 7 + i, std::sqrt(reduce_size));
    }
    absl::StrAppend(&c, "  loc_mem[", i, "][tid_x][tid_z] = reduce;\n");
  }
  c += "  ucl::SyncThreads<WorkGroup, Local>();\n";

  // Apply wg_size_z x wg_size_z Fast Walsh-Hadamard Transform.
  for (int stride = 1; stride < wg_size_z; stride *= 2) {
    std::string reduction_block = R"(
  if (tid_z < WG_SIZE_Z / 2) {
    int remainder = tid_z & (STRIDE - 1);
    int index = 2 * tid_z - remainder;
)";
    for (int i = 0; i < 7; ++i) {
      reduction_block += absl::Substitute(R"(
    {
      Type a = loc_mem[$0][tid_x][index] * ucl::Convert<SType>(RSQRT2);
      Type b = loc_mem[$0][tid_x][index + STRIDE] * ucl::Convert<SType>(RSQRT2);
      loc_mem[$0][tid_x][index] = a + b;
      loc_mem[$0][tid_x][index + STRIDE] = a - b;
    }
)",
                                          i);
    }
    absl::StrReplaceAll({{"STRIDE", std::to_string(stride)}}, &reduction_block);
    c += reduction_block;
    c += R"(
  }
  ucl::SyncThreads<WorkGroup, Local>();
)";
  }

  c += R"(
  if (DST_X >= args.dst_tensor.Width() || DST_Y >= args.dst_tensor.Height() ||
      DST_S >= args.dst_tensor.Slices()) {
    return;
  }
)";

  c += "  Type result;\n";
  for (int i = 0; i < 7; ++i) {
    c += absl::Substitute(R"(
  result = loc_mem[$0][tid_x][tid_z];
  args.dst_tensor.Write(result, DST_X, DST_Y, DST_S + $0);
)", i);
  }
  c += "}  // MAIN_FUNCTION\n";
  return c;
}

std::string GenerateOrtho7Hadamard4Code() {
  std::string c = R"(
  int DST_Y = ucl::GetGlobalId<1>();
  int DST_S = 7 * ucl::GetGlobalId<2>();
  if (DST_X >= args.dst_tensor.Width() || DST_Y >= args.dst_tensor.Height() ||
      DST_S >= args.dst_tensor.Slices()) {
    return;
  }

  Type src_val[7];
  Type dst_val[7];
)";

  for (int i = 0; i < 7; ++i) {
    // Scale by 1/5 to normalize `kOrtho7`.
    absl::StrAppend(&c, "  src_val[", i,
                    "] = args.src_tensor.Read(DST_X, DST_Y, DST_S + ", i,
                    ") / 5;\n");
  }

  // multiply src_vals by `kOrtho7`
  absl::StrAppend(
      &c,
      GenerateMatVecMul7Code(
          {"src_val[0].x", "src_val[0].y", "src_val[0].z", "src_val[0].w",
           "src_val[1].x", "src_val[1].y", "src_val[1].z"},
          {"dst_val[0].x", "dst_val[0].y", "dst_val[0].z", "dst_val[0].w",
           "dst_val[1].x", "dst_val[1].y", "dst_val[1].z"}),
      GenerateMatVecMul7Code(
          {"src_val[1].w", "src_val[2].x", "src_val[2].y", "src_val[2].z",
           "src_val[2].w", "src_val[3].x", "src_val[3].y"},
          {"dst_val[1].w", "dst_val[2].x", "dst_val[2].y", "dst_val[2].z",
           "dst_val[2].w", "dst_val[3].x", "dst_val[3].y"}),
      GenerateMatVecMul7Code(
          {"src_val[3].z", "src_val[3].w", "src_val[4].x", "src_val[4].y",
           "src_val[4].z", "src_val[4].w", "src_val[5].x"},
          {"dst_val[3].z", "dst_val[3].w", "dst_val[4].x", "dst_val[4].y",
           "dst_val[4].z", "dst_val[4].w", "dst_val[5].x"}),
      GenerateMatVecMul7Code(
          {"src_val[5].y", "src_val[5].z", "src_val[5].w", "src_val[6].x",
           "src_val[6].y", "src_val[6].z", "src_val[6].w"},
          {"dst_val[5].y", "dst_val[5].z", "dst_val[5].w", "dst_val[6].x",
           "dst_val[6].y", "dst_val[6].z", "dst_val[6].w"}));

  // Scale by 1/2 to normalize 4x4 Hadamard Transform.
  for (int i = 0; i < 7; ++i) {
    absl::StrAppend(&c, "  dst_val[", i, "] /= 2;\n");
  }

  // Apply 4x4 Hadamard Transform.
  c += R"(
  dst_val[0].x += dst_val[1].w;
  dst_val[1].w = dst_val[0].x - 2 * dst_val[1].w;
  dst_val[0].y += dst_val[2].x;
  dst_val[2].x = dst_val[0].y - 2 * dst_val[2].x;
  dst_val[0].z += dst_val[2].y;
  dst_val[2].y = dst_val[0].z - 2 * dst_val[2].y;
  dst_val[0].w += dst_val[2].z;
  dst_val[2].z = dst_val[0].w - 2 * dst_val[2].z;
  dst_val[1].x += dst_val[2].w;
  dst_val[2].w = dst_val[1].x - 2 * dst_val[2].w;
  dst_val[1].y += dst_val[3].x;
  dst_val[3].x = dst_val[1].y - 2 * dst_val[3].x;
  dst_val[1].z += dst_val[3].y;
  dst_val[3].y = dst_val[1].z - 2 * dst_val[3].y;

  dst_val[3].z += dst_val[5].y;
  dst_val[5].y = dst_val[3].z - 2 * dst_val[5].y;
  dst_val[3].w += dst_val[5].z;
  dst_val[5].z = dst_val[3].w - 2 * dst_val[5].z;
  dst_val[4].x += dst_val[5].w;
  dst_val[5].w = dst_val[4].x - 2 * dst_val[5].w;
  dst_val[4].y += dst_val[6].x;
  dst_val[6].x = dst_val[4].y - 2 * dst_val[6].x;
  dst_val[4].z += dst_val[6].y;
  dst_val[6].y = dst_val[4].z - 2 * dst_val[6].y;
  dst_val[4].w += dst_val[6].z;
  dst_val[6].z = dst_val[4].w - 2 * dst_val[6].z;
  dst_val[5].x += dst_val[6].w;
  dst_val[6].w = dst_val[5].x - 2 * dst_val[6].w;

  dst_val[0].x += dst_val[3].z;
  dst_val[3].z = dst_val[0].x - 2 * dst_val[3].z;
  dst_val[0].y += dst_val[3].w;
  dst_val[3].w = dst_val[0].y - 2 * dst_val[3].w;
  dst_val[0].z += dst_val[4].x;
  dst_val[4].x = dst_val[0].z - 2 * dst_val[4].x;
  dst_val[0].w += dst_val[4].y;
  dst_val[4].y = dst_val[0].w - 2 * dst_val[4].y;
  dst_val[1].x += dst_val[4].z;
  dst_val[4].z = dst_val[1].x - 2 * dst_val[4].z;
  dst_val[1].y += dst_val[4].w;
  dst_val[4].w = dst_val[1].y - 2 * dst_val[4].w;
  dst_val[1].z += dst_val[5].x;
  dst_val[5].x = dst_val[1].z - 2 * dst_val[5].x;

  dst_val[1].w += dst_val[5].y;
  dst_val[5].y = dst_val[1].w - 2 * dst_val[5].y;
  dst_val[2].x += dst_val[5].z;
  dst_val[5].z = dst_val[2].x - 2 * dst_val[5].z;
  dst_val[2].y += dst_val[5].w;
  dst_val[5].w = dst_val[2].y - 2 * dst_val[5].w;
  dst_val[2].z += dst_val[6].x;
  dst_val[6].x = dst_val[2].z - 2 * dst_val[6].x;
  dst_val[2].w += dst_val[6].y;
  dst_val[6].y = dst_val[2].w - 2 * dst_val[6].y;
  dst_val[3].x += dst_val[6].z;
  dst_val[6].z = dst_val[3].x - 2 * dst_val[6].z;
  dst_val[3].y += dst_val[6].w;
  dst_val[6].w = dst_val[3].y - 2 * dst_val[6].w;
)";

  for (int i = 0; i < 7; ++i) {
    absl::StrAppend(&c, "  args.dst_tensor.Write(dst_val[", i,
                    "], DST_X, DST_Y, DST_S + ", i, ");\n");
  }
  c += "}  // MAIN_FUNCTION\n";
  return c;
}

std::string GenerateHadamardOverSlicesCode(int reduce_size, int wg_size_z) {
  std::string c = R"(
  int DST_Y = ucl::GetGlobalId<1>();
  int tid_z = ucl::GetLocalId<2>();
  int group_row = ucl::GetGroupId<2>() / 7;
  int DST_S = 7 * WG_SIZE_Z * group_row + (ucl::GetGroupId<2>() % 7) + 7 * tid_z;

  int hadamard_batch = ucl::GetGlobalId<2>() / (7 * HADAMARD_DIM);
  int start_index = (ucl::GetGroupId<2>() % 7) + 7 * tid_z +
    hadamard_batch * 7 * HADAMARD_DIM;
  Type reduce = ucl::Init<Type>(0);
)";
  // reduce_size x reduce_size Hadamard Transform across workgroups.
  for (int i = 0; i < reduce_size; ++i) {
    // The (m, n)-th entry of a Hadamard Matrix is equal to
    // (-1)**{parity of m & n}.
    c += absl::Substitute(R"(
  {
    // Compute parity of number of 1s in bit representation of group_row & $0.
    int v = group_row & $0;
    v ^= v >> 1;
    v ^= v >> 2;
    v = (v & 0x11111111) * 0x11111111;
    v = (v >> 28) & 1;
    SType hadamard_coeff = ucl::Init<SType>(1 - 2 * v);
    SType hadamard_scale = ucl::Init<SType>($2);
    reduce += hadamard_coeff * args.src_tensor.Read(
        DST_X, DST_Y, start_index + $1) / hadamard_scale;
  })",
                          i, i * 7 * wg_size_z, std::sqrt(reduce_size));
  }

  // wg_size_z x wg_size_z Fast Walsh-Hadamard Transform within workgroups.
  c += R"(
  __local Type loc_mem[WG_SIZE_X][WG_SIZE_Z];
  int tid_x = ucl::GetLocalId<0>();
  loc_mem[tid_x][tid_z] = reduce;
  ucl::SyncThreads<WorkGroup, Local>();
)";
  for (int stride = 1; stride < wg_size_z; stride *= 2) {
    std::string reduction_block = R"(
  if (tid_z < WG_SIZE_Z / 2) {
    int remainder = tid_z & (STRIDE - 1);
    int index = 2 * tid_z - remainder;
    Type a = loc_mem[tid_x][index] * ucl::Convert<SType>(RSQRT2);
    Type b = loc_mem[tid_x][index + STRIDE] * ucl::Convert<SType>(RSQRT2);
    loc_mem[tid_x][index] = a + b;
    loc_mem[tid_x][index + STRIDE] = a - b;
  }
  ucl::SyncThreads<WorkGroup, Local>();
)";
    absl::StrReplaceAll({{"STRIDE", std::to_string(stride)}}, &reduction_block);
    c += reduction_block;
  }

  c += R"(
  if (DST_X >= args.dst_tensor.Width() || DST_Y >= args.dst_tensor.Height() ||
      DST_S >= args.dst_tensor.Slices()) {
    return;
  }
  Type result = loc_mem[tid_x][tid_z];
  args.dst_tensor.Write(result, DST_X, DST_Y, DST_S);
}  // MAIN_FUNCTION
)";
  return c;
}

std::string GenerateCode(const DataType& data_type, bool has_batch_axis,
                         int3 work_group_size, bool is_opencl, bool use_o7_h4,
                         std::optional<int> hadamard_dim) {
  std::string c;
  if (is_opencl) {
    c += "__attribute__((reqd_work_group_size(WG_SIZE_X, WG_SIZE_Y, "
         "WG_SIZE_Z)))\n";
  }
  c += "MAIN_FUNCTION($0) {\n";

  if (has_batch_axis) {
    c += R"(
  int linear_id = ucl::GetGlobalId<0>();
  int DST_X = linear_id / args.dst_tensor.Batch();
  int B = linear_id % args.dst_tensor.Batch();
  args.src_tensor.SetBatchRef(B);
  args.dst_tensor.SetBatchRef(B);
  )";
  } else {
    c += "  int DST_X = ucl::GetGlobalId<0>();\n";
  }

  // Divide `hadamard_dim` by 4 because we apply the Hadamard Transform over
  // slices in the second part of the kernel.
  const int hdim = hadamard_dim.has_value() ? *hadamard_dim / 4 : 0;
  // `reduce_size` should be a power of 2
  const int reduce_size = hdim / work_group_size.z;

  if (use_o7_h4 && hadamard_dim.has_value()) {
    c += GenerateFusedCode(reduce_size, work_group_size.z);
  } else if (use_o7_h4) {
    c += GenerateOrtho7Hadamard4Code();
  } else if (hadamard_dim.has_value()) {
    c += GenerateHadamardOverSlicesCode(reduce_size, work_group_size.z);
  }

  absl::StrReplaceAll({{"SType", ToUclDataType(data_type, 1)},
                       {"Type", ToUclDataType(data_type, 4)},
                       {"WG_SIZE_X", std::to_string(work_group_size.x)},
                       {"WG_SIZE_Y", std::to_string(work_group_size.y)},
                       {"WG_SIZE_Z", std::to_string(work_group_size.z)},
                       {"RSQRT2", "0.707106781186548f"}},  // 1 / sqrt(2)
                      &c);
  if (hadamard_dim.has_value()) {
    absl::StrReplaceAll({{"HADAMARD_DIM", std::to_string(hdim)}},
                        &c);
  }
  return c;
}

}  // namespace

absl::StatusOr<RotqatRotateOp> CreateRotqatRotateOp(
    const GpuInfo& gpu_info, const TensorDescriptor& src,
    const TensorDescriptor& dst, bool use_o7_h4,
    std::optional<int> hadamard_dim) {
  RET_CHECK(use_o7_h4 || hadamard_dim.has_value());
  const BHWC shape = src.GetBHWCShape();
  RET_CHECK(shape.c % 7 == 0);
  const int pow2 = shape.c / 7;
  RET_CHECK((pow2 & (pow2 - 1)) == 0)
      << "Number of channels should equal 7 times a power of 2.";
  RET_CHECK(!hadamard_dim.has_value() || pow2 % *hadamard_dim == 0)
      << "Number of channels should be a multiple of `hadamard_dim`.";

  RotqatRotateOp op(use_o7_h4, hadamard_dim);
  op.tensor_to_grid_ = TensorToGrid::kCustom;
  // TODO: b/384775474 - Tune work group size.
  op.work_group_size_ = {std::min(4, shape.c), 1, 64};
  op.code_ = GenerateCode(src.GetDataType(), src.HasAxis(Axis::kBatch),
                          op.work_group_size_, gpu_info.IsApiOpenCl(),
                          use_o7_h4, hadamard_dim);
  op.AddSrcTensor("src_tensor", src);
  op.AddDstTensor("dst_tensor", dst);
  return op;
}

}  // namespace ml_drift
