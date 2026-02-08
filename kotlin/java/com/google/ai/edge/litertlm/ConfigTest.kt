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

    // --- MeZoConfig Tests ---

    test("mezoConfig_defaultValues") {
      val config = MeZoConfig()
      assertEquals(1e-6f, config.learningRate)
      assertEquals(1e-3f, config.epsilon)
      assertEquals(0.0f, config.weightDecay)
      assertEquals(0L, config.seed)
      assertEquals(OptimizerMode.VANILLA_MEZO, config.optimizerMode)
      assertEquals(0.9f, config.momentumDecay)
      assertEquals(0.7854f, config.coneAngle)
      assertEquals(16, config.agzoSubspaceRank)
    }

    test("mezoConfig_customValues") {
      val config = MeZoConfig(
        learningRate = 1e-4f,
        epsilon = 2e-3f,
        weightDecay = 0.01f,
        seed = 42L,
        optimizerMode = OptimizerMode.CON_MEZO,
        momentumDecay = 0.95f,
        coneAngle = 0.5f,
      )
      assertEquals(1e-4f, config.learningRate)
      assertEquals(2e-3f, config.epsilon)
      assertEquals(0.01f, config.weightDecay)
      assertEquals(42L, config.seed)
      assertEquals(OptimizerMode.CON_MEZO, config.optimizerMode)
      assertEquals(0.95f, config.momentumDecay)
      assertEquals(0.5f, config.coneAngle)
    }

    test("mezoConfig_agzoMode") {
      val config = MeZoConfig(
        optimizerMode = OptimizerMode.AGZO,
        agzoSubspaceRank = 32,
      )
      assertEquals(OptimizerMode.AGZO, config.optimizerMode)
      assertEquals(32, config.agzoSubspaceRank)
    }

    test("mezoConfig_agzoInvalidRankThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoConfig(agzoSubspaceRank = 0)
      }
      assertThrows<IllegalArgumentException> {
        MeZoConfig(agzoSubspaceRank = -1)
      }
    }

    test("mezoConfig_optimizerModeNativeValues") {
      assertEquals(0, OptimizerMode.VANILLA_MEZO.nativeValue)
      assertEquals(1, OptimizerMode.CON_MEZO.nativeValue)
      assertEquals(2, OptimizerMode.AGZO.nativeValue)
    }

    test("mezoConfig_negativeLearningRateThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoConfig(learningRate = -1e-5f)
      }
    }

    test("mezoConfig_zeroEpsilonThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoConfig(epsilon = 0.0f)
      }
    }

    test("mezoConfig_negativeWeightDecayThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoConfig(weightDecay = -0.01f)
      }
    }

    test("mezoConfig_invalidMomentumDecayThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoConfig(momentumDecay = -0.1f)
      }
      assertThrows<IllegalArgumentException> {
        MeZoConfig(momentumDecay = 1.1f)
      }
    }

    test("mezoConfig_invalidConeAngleThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoConfig(coneAngle = -0.1f)
      }
      assertThrows<IllegalArgumentException> {
        MeZoConfig(coneAngle = 2.0f)
      }
    }

    test("mezoConfig_copyWithConMeZo") {
      val original = MeZoConfig()
      val copied = original.copy(optimizerMode = OptimizerMode.CON_MEZO, momentumDecay = 0.95f)
      assertEquals(OptimizerMode.VANILLA_MEZO, original.optimizerMode)
      assertEquals(OptimizerMode.CON_MEZO, copied.optimizerMode)
      assertEquals(0.95f, copied.momentumDecay)
    }

    test("mezoConfig_copyToAgzo") {
      val original = MeZoConfig(optimizerMode = OptimizerMode.CON_MEZO)
      val agzo = original.copy(optimizerMode = OptimizerMode.AGZO, agzoSubspaceRank = 8)
      assertEquals(OptimizerMode.CON_MEZO, original.optimizerMode)
      assertEquals(OptimizerMode.AGZO, agzo.optimizerMode)
      assertEquals(8, agzo.agzoSubspaceRank)
    }

    // --- MeZoParameter Tests ---

    test("mezoParameter_validCreation") {
      val param = MeZoParameter(name = "w", dataPointer = 12345L, numElements = 100L)
      assertEquals("w", param.name)
      assertEquals(12345L, param.dataPointer)
      assertEquals(100L, param.numElements)
      assertEquals(false, param.isBiasOrLayerNorm)
    }

    test("mezoParameter_nullPointerThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoParameter(name = "w", dataPointer = 0L, numElements = 100L)
      }
    }

    test("mezoParameter_zeroElementsThrows") {
      assertThrows<IllegalArgumentException> {
        MeZoParameter(name = "w", dataPointer = 1L, numElements = 0L)
      }
    }

    println()
    println("Results: $testsPassed passed, $testsFailed failed")

    if (testsFailed > 0) {
      System.exit(1)
    }
  }
}
