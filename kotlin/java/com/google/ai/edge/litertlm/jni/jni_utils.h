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

#include "absl/log/absl_log.h"  // from @com_google_absl

namespace litert::jni {

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
inline std::string JStringToString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) return "";
  const char* chars = env->GetStringUTFChars(jstr, nullptr);
  if (chars == nullptr) return "";
  std::string result(chars);
  env->ReleaseStringUTFChars(jstr, chars);
  return result;
}

// Replacement of env->NewStringUTF(str.c_str()) to handle standard UTF-8.
inline jstring NewStringStandardUTF(JNIEnv* env,
                                    const std::string& standard_utf8_str) {
  jbyteArray bytes = env->NewByteArray(standard_utf8_str.length());
  if (bytes == nullptr) return nullptr;
  ScopedLocalRef<jbyteArray> scoped_bytes(env, bytes);

  env->SetByteArrayRegion(
      bytes, 0, standard_utf8_str.length(),
      reinterpret_cast<const jbyte*>(standard_utf8_str.c_str()));

  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) return nullptr;
  ScopedLocalRef<jclass> scoped_string_class(env, string_class);

  jmethodID string_ctor =
      env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
  if (string_ctor == nullptr) return nullptr;

  jstring charset_name = env->NewStringUTF("UTF-8");
  if (charset_name == nullptr) return nullptr;
  ScopedLocalRef<jstring> scoped_charset(env, charset_name);

  return reinterpret_cast<jstring>(
      env->NewObject(string_class, string_ctor, bytes, charset_name));
}

// Helper to get JNIEnv and attach to the current thread if necessary.
inline JNIEnv* GetJniEnvAndAttach(JavaVM* jvm, bool* attached) {
  JNIEnv* env = nullptr;
  *attached = false;
  int get_env_stat =
      jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
  if (get_env_stat == JNI_EDETACHED) {
#if defined(__ANDROID__)
    if (jvm->AttachCurrentThread(&env, nullptr) == 0) {
#else
    if (jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) ==
        0) {
#endif
      *attached = true;
      return env;
    } else {
      ABSL_LOG(ERROR) << "Failed to attach to JVM.";
      return nullptr;
    }
  } else if (get_env_stat == JNI_OK) {
    return env;
  } else {
    ABSL_LOG(ERROR) << "Failed to get JNIEnv: GetEnv returned " << get_env_stat;
    return nullptr;
  }
}

}  // namespace litert::jni

#endif  // THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITERTLM_JNI_JNI_UTILS_H_
