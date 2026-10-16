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

#include "omni/text2speech/kokoro/cjk_test_blobs.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl

namespace litert::omni::tts::kokoro {
namespace {

constexpr int kCategoryNameSize = 32;

// One node of the trie `BuildMecabDoubleArray` lays out.
struct TrieNode {
  absl::flat_hash_map<unsigned char, int> children;
  // The packed MeCab value, or -1 when no key ends here.
  int value = -1;
};

// How many array slots each trie node owns: one for its value and one for
// every byte that could follow it. Keeping nodes from sharing slots is the
// whole difference between this and a real Darts array.
constexpr int kNodeStride = 258;

}  // namespace

void AppendUint32(uint32_t value, std::string& out) {
  out.push_back(static_cast<char>(value & 0xFF));
  out.push_back(static_cast<char>((value >> 8) & 0xFF));
  out.push_back(static_cast<char>((value >> 16) & 0xFF));
  out.push_back(static_cast<char>((value >> 24) & 0xFF));
}

void AppendUint16(uint16_t value, std::string& out) {
  out.push_back(static_cast<char>(value & 0xFF));
  out.push_back(static_cast<char>((value >> 8) & 0xFF));
}

std::string BuildCjkBlob(
    const std::vector<std::pair<std::string, std::string>>& sections) {
  std::string header;
  header.reserve(16);
  header.append("LTTSCJK1");
  AppendUint32(1, header);
  AppendUint32(sections.size(), header);

  const size_t table_size = 16 * sections.size();
  size_t body_start = header.size() + table_size;
  body_start += (8 - body_start % 8) % 8;

  size_t total_body_size = 0;
  for (const auto& [name, payload] : sections) {
    const size_t padding = (8 - payload.size() % 8) % 8;
    total_body_size += payload.size() + padding;
  }

  std::string table;
  table.reserve(table_size);
  std::string body;
  body.reserve(total_body_size);
  size_t cursor = body_start;
  for (const auto& [name, payload] : sections) {
    std::string padded_name = name;
    padded_name.resize(8, '\0');
    table.append(padded_name);
    AppendUint32(cursor, table);
    AppendUint32(payload.size(), table);
    body.append(payload);
    const size_t padding = (8 - payload.size() % 8) % 8;
    body.append(padding, '\0');
    cursor += payload.size() + padding;
  }
  std::string blob;
  blob.reserve(body_start + total_body_size);
  blob.append(header);
  blob.append(table);
  blob.append(body_start - blob.size(), '\0');
  blob.append(body);
  return blob;
}

std::string BuildMecabDoubleArray(
    const std::vector<std::pair<std::string, int>>& entries) {
  std::vector<TrieNode> nodes(1);
  for (const auto& [key, value] : entries) {
    int node = 0;
    for (const char character : key) {
      const auto byte = static_cast<unsigned char>(character);
      const auto found = nodes[node].children.find(byte);
      if (found != nodes[node].children.end()) {
        node = found->second;
        continue;
      }
      nodes.push_back({});
      const int child = static_cast<int>(nodes.size()) - 1;
      nodes[node].children[byte] = child;
      node = child;
    }
    nodes[node].value = value;
  }

  const int units = (static_cast<int>(nodes.size()) + 1) * kNodeStride;
  std::vector<int32_t> base(units, 0);
  std::vector<uint32_t> check(units, 0xFFFFFFFFu);

  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    // Traversal carries a node's base *value*, so a node is identified by the
    // slot its stride starts at rather than by its index here.
    const int node_base = (i + 1) * kNodeStride;
    if (i == 0) base[0] = node_base;
    if (nodes[i].value >= 0) {
      // A key ending here hangs its value off the node's own slot.
      check[node_base] = node_base;
      base[node_base] = -nodes[i].value - 1;
    }
    for (const auto& [byte, child] : nodes[i].children) {
      check[node_base + byte + 1] = node_base;
      base[node_base + byte + 1] = (child + 1) * kNodeStride;
    }
  }

  std::string blob;
  blob.reserve(units * 8);
  for (int i = 0; i < units; ++i) {
    AppendUint32(static_cast<uint32_t>(base[i]), blob);
    AppendUint32(check[i], blob);
  }
  return blob;
}

std::string BuildMecabToken(int left_id, int right_id, int pos_id,
                            int word_cost, uint32_t feature_offset) {
  std::string row;
  row.reserve(16);
  AppendUint16(left_id, row);
  AppendUint16(right_id, row);
  AppendUint16(pos_id, row);
  AppendUint16(static_cast<uint16_t>(static_cast<int16_t>(word_cost)), row);
  AppendUint32(feature_offset, row);
  AppendUint32(0, row);  // The compound field, which nothing reads.
  return row;
}

std::string BuildMecabFeatures(const std::vector<std::string>& records,
                               std::vector<uint32_t>& offsets) {
  std::string blob;
  for (const std::string& record : records) {
    offsets.push_back(blob.size());
    blob += record;
    blob.push_back('\0');
  }
  return blob;
}

std::string BuildMecabConnectionMatrix(int right_context_count,
                                       int left_context_count,
                                       const std::vector<int>& costs) {
  std::string matrix;
  matrix.reserve(4 + 2 * static_cast<size_t>(left_context_count) *
                         right_context_count);
  AppendUint16(right_context_count, matrix);
  AppendUint16(left_context_count, matrix);
  for (int i = 0; i < left_context_count * right_context_count; ++i) {
    AppendUint16(static_cast<uint16_t>(static_cast<int16_t>(
                     i < static_cast<int>(costs.size()) ? costs[i] : 0)),
                 matrix);
  }
  return matrix;
}

uint32_t PackMecabCharInfo(uint32_t types, int default_type, int length,
                           bool group, bool invoke) {
  return (types & 0x3FFFF) | (static_cast<uint32_t>(default_type) << 18) |
         (static_cast<uint32_t>(length) << 26) |
         (static_cast<uint32_t>(group) << 30) |
         (static_cast<uint32_t>(invoke) << 31);
}

std::string BuildMecabCharProperty(const std::vector<std::string>& names,
                                   const std::vector<uint32_t>& records,
                                   int code_points) {
  std::string data;
  data.reserve(4 + names.size() * kCategoryNameSize +
               static_cast<size_t>(code_points) * 4);
  AppendUint32(names.size(), data);
  for (const std::string& name : names) {
    std::string padded = name;
    padded.resize(kCategoryNameSize, '\0');
    data += padded;
  }
  for (int i = 0; i < code_points; ++i) {
    AppendUint32(i < static_cast<int>(records.size()) ? records[i] : 0, data);
  }
  return data;
}

}  // namespace litert::omni::tts::kokoro
