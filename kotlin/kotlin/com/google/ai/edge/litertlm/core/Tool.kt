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

import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.google.gson.JsonParseException
import com.google.gson.JsonParser
import com.google.gson.JsonPrimitive
import kotlin.reflect.KParameter
import kotlin.reflect.full.functions

/**
 * Example of how to define tools with `AutoToolSet`:
 * - Use `@Tool` to define a method as a tool.
 * - Use `@ToolParam` to add information to the param of a tool.
 * - The allowed parameter types are: String, Int, Boolean, Float, Double, and List of them.
 * - The return type of your tool can be any Kotlin type.
 *
 * ```kotlin
 * class MyToolSet: AutoToolSet() {
 *   @Tool(description = "Get the current weather")
 *   fun getCurrentWeather(
 *     @ToolParam(description = "The city and state, e.g. San Francisco, CA") location: String,
 *     @ToolParam(description = "The temperature unit to use") unit: String = "celsius",
 *   ): Map {
 *     return mapOf(
 *       "temperature" to 25,
 *       "unit" to "Celsius",
 *     )
 *   }
 * }
 * ```
 */

/**
 * Annotation to mark a function as a tool that can be used by the LiteRT-LM model.
 *
 * @property description A description of the tool.
 */
@Target(AnnotationTarget.FUNCTION) // This annotation can only be applied to functions
@Retention(AnnotationRetention.RUNTIME) // IMPORTANT: Makes the annotation available at runtime
annotation class Tool(val description: String)

/**
 * Annotation to provide a description for a tool parameter.
 *
 * @property description A description of the tool parameter.
 */
@Target(AnnotationTarget.VALUE_PARAMETER) // This annotation can only be applied to functions
@Retention(AnnotationRetention.RUNTIME) // IMPORTANT: Makes the annotation available at runtime
annotation class ToolParam(val description: String)

/**
 * Tool provider for Conversation.
 *
 * Use [tool] function to convert the supported tool types to this [ToolProvider].
 */
abstract class ToolProvider {

  /**
   * Provides a map of tools, where the key is the tool name and the value is the tool
   * implementation.
   *
   * This method uses reflection to find all functions annotated with @Tool in the current class. It
   * then wraps each function in a ReflectionTool instance and returns them in a map.
   */
  internal abstract fun provideTools(): Map<String, InternalJsonTool>
}

/**
 * An abstract class for defining collection of tools using Kotlin functions.
 * - Use `@Tool` to define a method as a tool.
 * - Use `@ToolParam` to add information to the param of a tool.
 * - The allowed parameter types are: String, Int, Boolean, Float, Double, and List of them.
 * - The return type of your tool can be any Kotlin type.
 *
 * ```kotlin
 * class MyToolSet: ToolSet {
 *   @Tool(description = "Get the current weather")
 *   fun getCurrentWeather(
 *     @ToolParam(description = "The city and state, e.g. San Francisco, CA") location: String,
 *     @ToolParam(description = "The temperature unit to use") unit: String = "celsius",
 *   ): Map {
 *     return mapOf(
 *       "temperature" to 25,
 *       "unit" to "Celsius",
 *     )
 *   }
 * }
 * ```
 */
interface ToolSet {}

/**
 * An abstract class for defining a tool using an Open API specification.
 *
 * This class provides a way to define a tool by providing a JSON string that conforms to the Open
 * API specification. The tool can then be executed by providing a JSON string containing the
 * parameters.
 *
 * The input and output are both JSON strings to provide flexibility in choosing a JSON library.
 * This avoids forcing a specific library on the user, allowing them to use the one that best suits
 * their project needs and dependencies.
 */
interface OpenApiTool {

  /**
   * Gets the tool description JSON string based on the Open API specification.
   *
   * The JSON string should be a valid JSON object with keys:
   * - name : Required. The name of the tool.
   * - description: Optional. A brief description of the function.
   * - parameters: Optional. Describes the parameters to this function.
   *
   * For example,
   * - {"name":"addition","description":"Add all
   *   numbers.","parameters":{"type":"object","properties":{"numbers":{"type":"array","items":{"type":"number"}},"description":"The
   *   list of numbers to sum."},"required":["numbers"]}}
   */
  fun getToolDescriptionJsonString(): String

  /**
   * Executes the tools with the paramsJsonString and return the result as string.
   *
   * The paramsJsonString is a JSONObject where the keys are the name of the parameters.
   *
   * For examples, {"numbers":[3.0,4.5,6.0]}
   *
   * The return value is the JSON string of the results. For examples,
   * - 13.5 // return a JSON primitive value
   * - {"sum": 13.5} // return as a JSON Object
   */
  fun execute(paramsJsonString: String): String
}

/** Returns a [ToolProvider] from a [ToolSet]. */
fun tool(toolSet: ToolSet): ToolProvider {
  return object : ToolProvider() {
    @OptIn(ExperimentalApi::class)
    override fun provideTools(): Map<String, InternalJsonTool> {
      val useSnakeCase = ExperimentalFlags.convertCamelToSnakeCaseInToolDescription

      val toolClass = toolSet.javaClass.kotlin
      return toolClass.functions
        .filter { function -> function.annotations.any { annotation -> annotation is Tool } }
        .map { function ->
          (if (useSnakeCase) function.name.camelToSnakeCase() else function.name) to
            ReflectionTool(toolSet, function, useSnakeCase) // Pass toolSet here
        }
        .toMap()
    }
  }
}

/** Returns a [ToolProvider] from an [OpenApiTool]. */
fun tool(openApiTool: OpenApiTool): ToolProvider {
  return object : ToolProvider() {
    override fun provideTools(): Map<String, InternalJsonTool> {
      val toolDescription: JsonObject =
        try {
          JsonParser.parseString(openApiTool.getToolDescriptionJsonString()).asJsonObject
        } catch (e: JsonParseException) {
          throw ToolException("Failed to parse JSON. ${e.message}", e)
        }

      val name: String =
        try {
          toolDescription.get("name").asString
        } catch (e: Throwable) {
          throw ToolException("Failed to parse field \"name\" as String. ${e.message}", e)
        }

      val jsonTool =
        object : InternalJsonTool {
          override fun getToolDescription(): JsonObject {
            return toolDescription
          }

          override fun execute(params: JsonObject): Any? {
            return openApiTool.execute(params.toString())
          }
        }

      return mapOf(name to jsonTool)
    }
  }
}

/**
 * Manages a collection of tools and provides methods to execute tools and get their specifications.
 *
 * @property tools A list of [ToolProvider].
 */
class ToolManager(val tools: List<ToolProvider> = emptyList()) {

  private val internalTools: Map<String, InternalJsonTool> =
    tools.fold(mapOf()) { acc, tool -> acc + tool.provideTools() }

  /**
   * Executes a tool function by its name with the given parameters.
   *
   * @param functionName The name of the tool function to execute.
   * @param params A JsonObject containing the parameter names and their values.
   * @return The result of the tool function execution as a string.
   * @throws IllegalArgumentException if the tool function is not found.
   */
  fun execute(functionName: String, params: JsonObject): JsonElement {
    try {
      val tool =
        internalTools[functionName]
          ?: throw IllegalArgumentException("Tool not found: ${functionName}")
      return tool.execute(params).toJsonElement()
    } catch (e: Exception) {
      return JsonPrimitive("Error occured. ${e.toString()}")
    }
  }

  /**
   * Gets the tools description for all registered tools in Open API format.
   *
   * @return A json array of OpenAPI tool description JSON as string.
   */
  fun getToolsDescription() =
    JsonArray().apply {
      for (tool in internalTools.values) {
        // Wrap the Open API spec in function object, expected by the native library.
        add(
          JsonObject().apply {
            addProperty("type", "function")
            add("function", tool.getToolDescription())
          }
        )
      }
    }
}

/** Internal use only. */
interface InternalJsonTool {

  fun getToolDescription(): JsonObject

  fun execute(params: JsonObject): Any?
}

/**
 * Represents a single tool, wrapping an instance and a specific Kotlin function.
 *
 * @property instance The instance of the class containing the tool function.
 * @property kFunction The Kotlin function to be executed as a tool.
 * @property useSnakeCase Whether to use snake case for function and param names for tool calling.
 */
internal class ReflectionTool(
  val instance: Any,
  val kFunction: kotlin.reflect.KFunction<*>,
  val useSnakeCase: Boolean,
) : InternalJsonTool {

  /**
   * Gets the tool description in Open API format.
   *
   * @return The tool description.
   */
  override fun getToolDescription(): JsonObject {
    val toolAnnotation = kFunction.annotations.find { it is Tool } as? Tool ?: return JsonObject()

    val description = toolAnnotation.description

    val openApiSpec =
      JsonObject().apply {
        val funcName = if (useSnakeCase) kFunction.name.camelToSnakeCase() else kFunction.name
        addProperty("name", funcName)
        addProperty("description", description)
      }

    val parameters = kFunction.parameters.drop(1) // Drop the instance parameter
    if (!parameters.isEmpty()) {
      val properties = JsonObject()
      for (param in parameters) {
        val paramAnnotation = param.annotations.find { it is ToolParam } as? ToolParam
        val paramJsonSchema = getTypeJsonSchema(param.type)
        // add "description" if provided
        paramAnnotation?.description?.let { paramJsonSchema.addProperty("description", it) }
        if (param.type.isMarkedNullable) paramJsonSchema.addProperty("nullable", true)
        properties.add(param.toModelParamName(), paramJsonSchema)
      }

      val requiredParams = JsonArray()
      for (param in parameters) {
        if (!param.isOptional) {
          requiredParams.add(param.toModelParamName())
        }
      }

      val schema =
        JsonObject().apply {
          addProperty("type", "object")
          add("properties", properties)
          if (!requiredParams.isEmpty) add("required", requiredParams)
        }

      openApiSpec.add("parameters", schema)
    }

    return openApiSpec
  }

  /**
   * Executes the tool function with the given parameters.
   *
   * @param params A JsonObject containing the parameter names and their values.
   * @return The result of the tool function execution as a Any?.
   * @throws IllegalArgumentException if any required parameters are missing.
   */
  override fun execute(params: JsonObject): Any? {
    val args =
      kFunction.parameters
        .associateWith { param ->
          when {
            param.index == 0 -> instance // First parameter is the instance
            param.name != null && params.has(param.toModelParamName()) -> {
              val value = params.get(param.toModelParamName())
              convertJsonValueToKotlinValue(value, param.type)
            }
            param.isOptional -> null // Should not be reached
            else -> throw IllegalArgumentException("Missing parameter: ${param.toModelParamName()}")
          }
        }
        .filterValues { it != null }

    return kFunction.callBy(args)
  }

  /**
   * Converts a JSON value to the expected Kotlin type.
   *
   * @param value The JSON value to convert.
   * @param type The target Kotlin type.
   * @return The converted value.
   * @throws IllegalArgumentException if the value cannot be converted to the target type.
   */
  private fun convertJsonValueToKotlinValue(value: JsonElement, type: kotlin.reflect.KType): Any {
    val classifier = type.classifier
    return when {
      classifier == List::class && value is JsonArray -> {
        val listTypeArgument = type.arguments.firstOrNull()?.type
        value.map { convertJsonValueToKotlinValue(it, listTypeArgument!!) }
      }
      classifier == Int::class && value is JsonPrimitive && value.isNumber -> value.asInt
      classifier == Float::class && value is JsonPrimitive && value.isNumber -> value.asFloat
      classifier == Double::class && value is JsonPrimitive && value.isNumber -> value.asDouble
      classifier == String::class && value is JsonPrimitive && value.isString -> value.asString
      classifier == Boolean::class && value is JsonPrimitive && value.isBoolean -> value.asBoolean
      // Add more conversions if needed
      else -> value
    }
  }

  /**
   * Generates a JSON schema for the given Kotlin type.
   *
   * @param type The Kotlin type to generate the schema for.
   * @return A JsonObject representing the JSON schema.
   * @throws IllegalArgumentException if the type is not supported.
   */
  private fun getTypeJsonSchema(type: kotlin.reflect.KType): JsonObject {
    val classifier = type.classifier
    val jsonType = javaTypeToJsonTypeString[classifier]

    if (jsonType == null) {
      throw IllegalArgumentException(
        "Unsupported type: ${classifier.toString()}. " +
          "Allowed types are: ${javaTypeToJsonTypeString.keys.joinToString { it.simpleName ?: "" }}"
      )
    }

    val schema = JsonObject()
    schema.addProperty("type", jsonType)
    if (classifier == List::class) {
      val listTypeArgument = type.arguments.firstOrNull()?.type
      if (listTypeArgument == null) {
        throw IllegalArgumentException("List type argument is missing.")
      }
      schema.add("items", getTypeJsonSchema(listTypeArgument))
    }
    return schema
  }

  private fun KParameter.toModelParamName(): String {
    return if (useSnakeCase) this.name!!.camelToSnakeCase() else this.name!!
  }

  companion object {
    private val javaTypeToJsonTypeString =
      mapOf(
        String::class to "string",
        Int::class to "integer",
        Boolean::class to "boolean",
        Float::class to "number",
        Double::class to "number",
        List::class to "array",
      )
  }
}

private fun String.camelToSnakeCase(): String {
  return this.replace(Regex("(?<=[a-zA-Z])[A-Z]")) { "_${it.value}" }.lowercase()
}

private fun String.snakeToCamelCase(): String {
  return Regex("_([a-z])").replace(this) { it.value.substring(1).uppercase() }
}

/** Exception related to tool calling. */
class ToolException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
