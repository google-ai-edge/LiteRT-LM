// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import CLiteRTLM

/// Provides information about capabilities and features supported by a LiteRT-LM file.
public class Capabilities {
  private let handle: OpaquePointer?

  /// Loads a LiteRT-LM file from the given path.
  /// Returns nil if the file cannot be opened.
  public init?(modelPath: String) {
    guard let handle = litert_lm_loaded_file_create(modelPath) else {
      return nil
    }
    self.handle = handle
  }

  /// Checks if the loaded LiteRT-LM file supports speculative decoding.
  public func hasSpeculativeDecodingSupport() -> Bool {
    return litert_lm_loaded_file_has_speculative_decoding_support(handle)
  }

  deinit {
    if let handle = handle {
      litert_lm_loaded_file_delete(handle)
    }
  }
}
