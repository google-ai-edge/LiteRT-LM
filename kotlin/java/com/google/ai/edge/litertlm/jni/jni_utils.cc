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

#include "kotlin/java/com/google/ai/edge/litertlm/jni/jni_utils.h"

#include <jni.h>

#include <string>

#include "absl/log/absl_log.h"  // from @com_google_absl

namespace litert::lm::jni {

std::string JStringToString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) return "";
  const char* chars = env->GetStringUTFChars(jstr, nullptr);
  if (chars == nullptr) return "";
  std::string result(chars);
  env->ReleaseStringUTFChars(jstr, chars);
  return result;
}

// Replacement of env->NewStringUTF(str.c_str()) to handle "Standard UTF-8".
//
// NewStringUTF() expects a "modified UTF-8" string. "Standard UTF-8" and
// "modified UTF-8" are mostly the same, but differ in the encoding of null
// characters and characters outside the Basic Multilingual Plane (BMP). Emojis
// often fall into this latter category. nlohmann::json::dump() also returns a
// "Standard UTF-8".
//
// https://developer.android.com/ndk/guides/jni-tips#utf-8-and-utf-16-strings
jstring NewStringStandardUTF(JNIEnv* env,
                             const std::string& standard_utf8_str) {
  // Create a jbyteArray from the UTF-8 string
  jbyteArray bytes = env->NewByteArray(standard_utf8_str.length());
  if (bytes == nullptr) return nullptr;
  env->SetByteArrayRegion(
      bytes, 0, standard_utf8_str.length(),
      reinterpret_cast<const jbyte*>(standard_utf8_str.c_str()));

  // Get the java.lang.String class
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    env->DeleteLocalRef(bytes);
    return nullptr;
  }

  // Get the constructor for String(byte[], String)
  jmethodID string_ctor =
      env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
  if (string_ctor == nullptr) {
    env->DeleteLocalRef(string_class);
    env->DeleteLocalRef(bytes);
    return nullptr;
  }

  // Create a jstring for the charset name "UTF-8"
  jstring charset_name = env->NewStringUTF("UTF-8");
  if (charset_name == nullptr) {
    env->DeleteLocalRef(string_class);
    env->DeleteLocalRef(bytes);
    return nullptr;
  }

  // Create the new String object
  jstring result =
      (jstring)env->NewObject(string_class, string_ctor, bytes, charset_name);

  // Clean up local references
  env->DeleteLocalRef(bytes);
  env->DeleteLocalRef(string_class);
  env->DeleteLocalRef(charset_name);

  return result;
}

JNIEnv* GetJniEnvAndAttach(JavaVM* jvm, bool* attached) {
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

}  // namespace litert::lm::jni
