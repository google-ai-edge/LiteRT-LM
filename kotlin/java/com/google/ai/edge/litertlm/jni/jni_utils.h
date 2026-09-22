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
#include <vector>

namespace litert::lm::jni {

// Helper to get JNIEnv and attach to the current thread if necessary.
JNIEnv* GetJniEnvAndAttach(JavaVM* jvm, bool* attached);

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

// RAII wrapper for JNI global references that safely attaches to JavaVM if
// destroyed on a native worker thread.
template <typename T>
class ScopedGlobalRef {
 public:
  ScopedGlobalRef(JavaVM* jvm, T global_ref)
      : jvm_(jvm), global_ref_(global_ref) {}

  ScopedGlobalRef(JNIEnv* env, T local_or_global_ref) {
    env->GetJavaVM(&jvm_);
    global_ref_ =
        local_or_global_ref != nullptr
            ? reinterpret_cast<T>(env->NewGlobalRef(local_or_global_ref))
            : nullptr;
  }

  ~ScopedGlobalRef() { reset(); }

  void reset(T ptr = nullptr) {
    if (ptr != global_ref_) {
      if (global_ref_ != nullptr && jvm_ != nullptr) {
        bool attached = false;
        JNIEnv* env = GetJniEnvAndAttach(jvm_, &attached);
        if (env != nullptr) {
          env->DeleteGlobalRef(global_ref_);
        }
        if (attached) {
          jvm_->DetachCurrentThread();
        }
      }
      global_ref_ = ptr;
    }
  }

  T release() {
    T ref = global_ref_;
    global_ref_ = nullptr;
    return ref;
  }

  T get() const { return global_ref_; }

  ScopedGlobalRef(ScopedGlobalRef&& other) noexcept
      : jvm_(other.jvm_), global_ref_(other.release()) {}

  ScopedGlobalRef& operator=(ScopedGlobalRef&& other) noexcept {
    if (this != &other) {
      reset();
      jvm_ = other.jvm_;
      global_ref_ = other.release();
    }
    return *this;
  }

  ScopedGlobalRef(const ScopedGlobalRef&) = delete;
  ScopedGlobalRef& operator=(const ScopedGlobalRef&) = delete;

 private:
  JavaVM* jvm_ = nullptr;
  T global_ref_ = nullptr;
};

// Converts a jstring to a standard std::string, handling null and freeing
// chars.
std::string JStringToString(JNIEnv* env, jstring jstr);

// Replacement of env->NewStringUTF(str.c_str()) to handle standard UTF-8.
jstring NewStringStandardUTF(JNIEnv* env, const std::string& standard_utf8_str);

// Converts a vector of standard UTF-8 strings to a Java String[] array.
jobjectArray ToJavaStringArray(JNIEnv* env,
                               const std::vector<std::string>& strings);

}  // namespace litert::lm::jni

#endif  // THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITERTLM_JNI_JNI_UTILS_H_
