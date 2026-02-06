/*
 * Copyright 2025 Google LLC
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
package com.google.ai.edge.litertlm

/** Simple test runner for config validation. */
object ConfigTest {

  private var testsPassed = 0
  private var testsFailed = 0

  private fun test(name: String, block: () -> Unit) {
    try {
      block()
      testsPassed++
      println("✓ $name")
    } catch (e: Throwable) {
      testsFailed++
      println("✗ $name: ${e.message}")
    }
  }

  private fun assertEquals(expected: Any?, actual: Any?) {
    if (expected != actual) {
      throw AssertionError("Expected <$expected> but was <$actual>")
    }
  }

  private fun assertNull(value: Any?) {
    if (value != null) {
      throw AssertionError("Expected null but was <$value>")
    }
  }

  private inline fun <reified T : Throwable> assertThrows(block: () -> Unit) {
    try {
      block()
      throw AssertionError("Expected ${T::class.simpleName} to be thrown")
    } catch (e: Throwable) {
      if (e !is T) {
        throw AssertionError("Expected ${T::class.simpleName} but got ${e::class.simpleName}: ${e.message}")
      }
    }
  }

  @JvmStatic
  fun main(args: Array<String>) {
    println("Running ConfigTest...")
    println()

    test("sessionConfig_defaultLoraIdIsNull") {
      val config = SessionConfig()
      assertNull(config.loraId)
    }

    test("sessionConfig_loraIdZeroIsValid") {
      val config = SessionConfig(loraId = 0)
      assertEquals(0, config.loraId)
    }

    test("sessionConfig_positiveLoraIdIsValid") {
      val config = SessionConfig(loraId = 42)
      assertEquals(42, config.loraId)
    }

    test("sessionConfig_negativeLoraIdThrows") {
      assertThrows<IllegalArgumentException> {
        SessionConfig(loraId = -1)
      }
    }

    test("sessionConfig_withSamplerConfigAndLoraId") {
      val samplerConfig = SamplerConfig(topK = 40, topP = 0.95, temperature = 0.7)
      val config = SessionConfig(samplerConfig = samplerConfig, loraId = 5)
      assertEquals(samplerConfig, config.samplerConfig)
      assertEquals(5, config.loraId)
    }

    test("conversationConfig_defaultLoraIdIsNull") {
      val config = ConversationConfig()
      assertNull(config.loraId)
    }

    test("conversationConfig_loraIdZeroIsValid") {
      val config = ConversationConfig(loraId = 0)
      assertEquals(0, config.loraId)
    }

    test("conversationConfig_positiveLoraIdIsValid") {
      val config = ConversationConfig(loraId = 99)
      assertEquals(99, config.loraId)
    }

    test("conversationConfig_negativeLoraIdThrows") {
      assertThrows<IllegalArgumentException> {
        ConversationConfig(loraId = -1)
      }
    }

    test("conversationConfig_withSamplerConfigAndLoraId") {
      val samplerConfig = SamplerConfig(topK = 10, topP = 0.5, temperature = 0.1)
      val config = ConversationConfig(samplerConfig = samplerConfig, loraId = 3)
      assertEquals(samplerConfig, config.samplerConfig)
      assertEquals(3, config.loraId)
    }

    test("sessionConfig_copyWithDifferentLoraId") {
      val original = SessionConfig(loraId = 1)
      val copied = original.copy(loraId = 2)
      assertEquals(1, original.loraId)
      assertEquals(2, copied.loraId)
    }

    test("conversationConfig_copyWithDifferentLoraId") {
      val original = ConversationConfig(loraId = 10)
      val copied = original.copy(loraId = 20)
      assertEquals(10, original.loraId)
      assertEquals(20, copied.loraId)
    }

    println()
    println("Results: $testsPassed passed, $testsFailed failed")

    if (testsFailed > 0) {
      System.exit(1)
    }
  }
}
