/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITERTLM_JNI_JNI_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITERTLM_JNI_JNI_UTILS_H_

#include <jni.h>

#include <string>

namespace litert::lm::jni {

// RAII wrapper for JNI local references.
template <typename T>
class ScopedLocalRef {
 public:
  ScopedLocalRef(JNIEnv* env, T local_ref) : env_(env), local_ref_(local_ref) {}

  ~ScopedLocalRef() { reset(); }

  void reset(T ptr = nullptr) {
    if (ptr != local_ref_) {
      if (local_ref_ != nullptr) {
        env_->DeleteLocalRef(local_ref_);
      }
      local_ref_ = ptr;
    }
  }

  T release() {
    T ref = local_ref_;
    local_ref_ = nullptr;
    return ref;
  }

  T get() const { return local_ref_; }

  ScopedLocalRef(const ScopedLocalRef&) = delete;
  ScopedLocalRef& operator=(const ScopedLocalRef&) = delete;

 private:
  JNIEnv* const env_;
  T local_ref_;
};

// Converts a jstring to a standard std::string, handling null and freeing
// chars.
std::string JStringToString(JNIEnv* env, jstring jstr);

// Replacement of env->NewStringUTF(str.c_str()) to handle standard UTF-8.
jstring NewStringStandardUTF(JNIEnv* env, const std::string& standard_utf8_str);

// Helper to get JNIEnv and attach to the current thread if necessary.
JNIEnv* GetJniEnvAndAttach(JavaVM* jvm, bool* attached);

}  // namespace litert::lm::jni

#endif  // THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITERTLM_JNI_JNI_UTILS_H_
