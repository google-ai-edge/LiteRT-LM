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

#include <jni.h>

#include <gtest/gtest.h>

extern "C" {
JNIEXPORT void JNICALL
Java_com_google_ai_edge_litertlm_LiteRtLmJni_nativeDeleteEmbeddingEngine(
    JNIEnv* env, jclass thiz, jlong engine_pointer);
}

namespace {

TEST(LiteRtLmJniTest, NativeDeleteEmbeddingEngineNullPointerDoesNotCrash) {
  // Passing 0 as engine_pointer should safely do nothing without crashing.
  Java_com_google_ai_edge_litertlm_LiteRtLmJni_nativeDeleteEmbeddingEngine(
      nullptr, nullptr, 0);
}

}  // namespace
