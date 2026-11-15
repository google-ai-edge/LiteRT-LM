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

#ifndef THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_SEQUENCED_H_
#define THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_SEQUENCED_H_

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl

namespace litert::omni {

// Associates a sequence number `seqnum` with an item `T`.
//
// Sequence numbers are monotonically increasing, contiguous integers
// assigned by `FanOutStage` when inputs arrive, and used by `FanInStage`
// to restore the original temporal input order across asynchronous modality
// branches.
template <typename T>
struct Sequenced {
  // Contiguous sequence number assigned by FanOutStage (0, 1, 2, ...).
  uint64_t seqnum = 0;

  // Modality-specific payload.
  T item;
};

// Base interface for channels that accept sequenced input items from a
// FanOutStage.
template <typename InputT>
class SequencedChannel {
 public:
  virtual ~SequencedChannel() = default;

  // Returns true if this channel accepts `input`.
  virtual bool Accepts(const InputT& input) const = 0;

  // Attempts to accept an input with its assigned sequence number and append it
  // to the back of the channel.
  // Returns Ok if accepted and enqueued.
  // Returns Unimplemented if this channel does not handle this input type.
  // Returns other status on error.
  virtual absl::Status PushBack(uint64_t seqnum, InputT input) = 0;

  // Returns true if the channel currently has no queued items.
  virtual bool Empty() const = 0;

  // Clears all queued items in the channel.
  virtual void Reset() = 0;
};

}  // namespace litert::omni

#endif  // THIRD_PARTY_ODML_LITERT_LM_OMNI_BASE_SEQUENCED_H_
