// Copyright 2026 Google LLC
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

import LiteRTLM
import XCTest

final class ToolTests: XCTestCase {

  override func setUp() {
    super.setUp()
    ExperimentalFlags.optIntoExperimentalAPIs()
    ExperimentalFlags.convertCamelToSnakeCaseInToolDescription = true
  }

  override func tearDown() {
    ExperimentalFlags.convertCamelToSnakeCaseInToolDescription = true
    super.tearDown()
  }

  func testToolsDescriptionConversion() throws {
    let tools: [Tool] = [
      AddIntegersTool(),
      DeterminantTool(),
      GetCurrentWeatherTool(),
      SumListTool(),
    ]

    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"add_integers","description":"Adds two integers and returns the result.","parameters":{"type":"object","properties":{"first_int":{"type":"integer","description":"The first integer."},"second_int":{"type":"integer","description":"The second integer."}},"required":["first_int","second_int"]}}},{"type":"function","function":{"name":"determinant","description":"Calculates the determinant of a 2x2 matrix.","parameters":{"type":"object","properties":{"matrix":{"type":"array","items":{"type":"array","items":{"type":"integer"}},"description":"2x2 matrix."}},"required":["matrix"]}}},{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city name, e.g., San Francisco"},"country":{"type":"string","description":"Optional country code, e.g., US","nullable":true},"unit":{"type":"string","description":"Temperature unit (celsius or fahrenheit). Default: celsius"}},"required":["city"]}}},{"type":"function","function":{"name":"sum_list","description":"Sums a list of integers.","parameters":{"type":"object","properties":{"numbers":{"type":"array","items":{"type":"integer"},"description":"List of integers to sum."}},"required":["numbers"]}}}]
      """

    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_useCamelCase() throws {
    ExperimentalFlags.convertCamelToSnakeCaseInToolDescription = false

    let tools: [Tool] = [
      AddIntegersTool(),
      DeterminantTool(),
      GetCurrentWeatherTool(),
      SumListTool(),
    ]

    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"addIntegers","description":"Adds two integers and returns the result.","parameters":{"type":"object","properties":{"firstInt":{"type":"integer","description":"The first integer."},"secondInt":{"type":"integer","description":"The second integer."}},"required":["firstInt","secondInt"]}}},{"type":"function","function":{"name":"determinant","description":"Calculates the determinant of a 2x2 matrix.","parameters":{"type":"object","properties":{"matrix":{"type":"array","items":{"type":"array","items":{"type":"integer"}},"description":"2x2 matrix."}},"required":["matrix"]}}},{"type":"function","function":{"name":"getCurrentWeather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city name, e.g., San Francisco"},"country":{"type":"string","description":"Optional country code, e.g., US","nullable":true},"unit":{"type":"string","description":"Temperature unit (celsius or fahrenheit). Default: celsius"}},"required":["city"]}}},{"type":"function","function":{"name":"sumList","description":"Sums a list of integers.","parameters":{"type":"object","properties":{"numbers":{"type":"array","items":{"type":"integer"},"description":"List of integers to sum."}},"required":["numbers"]}}}]
      """

    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_emptyToolSet() throws {
    let tools: [Tool] = []
    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = "[]"
    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_multiToolSets() throws {
    let tools: [Tool] = [
      AddIntegersTool(),
      DeterminantTool(),
      GetCurrentWeatherTool(),
      SumListTool(),
    ]

    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"add_integers","description":"Adds two integers and returns the result.","parameters":{"type":"object","properties":{"first_int":{"type":"integer","description":"The first integer."},"second_int":{"type":"integer","description":"The second integer."}},"required":["first_int","second_int"]}}},{"type":"function","function":{"name":"determinant","description":"Calculates the determinant of a 2x2 matrix.","parameters":{"type":"object","properties":{"matrix":{"type":"array","items":{"type":"array","items":{"type":"integer"}},"description":"2x2 matrix."}},"required":["matrix"]}}},{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"The city name, e.g., San Francisco"},"country":{"type":"string","description":"Optional country code, e.g., US","nullable":true},"unit":{"type":"string","description":"Temperature unit (celsius or fahrenheit). Default: celsius"}},"required":["city"]}}},{"type":"function","function":{"name":"sum_list","description":"Sums a list of integers.","parameters":{"type":"object","properties":{"numbers":{"type":"array","items":{"type":"integer"},"description":"List of integers to sum."}},"required":["numbers"]}}}]
      """

    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_allParamsOptional_noRequiredField() throws {
    let tools: [Tool] = [AllOptionalParamsTool()]
    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"get_sum","description":"All params are optional.","parameters":{"type":"object","properties":{"foo":{"type":"integer"},"bar":{"type":"integer"}}}}}]
      """
    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_snakeCaseToolSet_shouldStillWork() throws {
    let tools: [Tool] = [SnakeCaseTool()]
    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"another_tool","description":"Another tool.","parameters":{"type":"object","properties":{"another_param":{"type":"string","description":"Another param."}},"required":["another_param"]}}}]
      """
    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_snakeCaseToolSet_useCamelCaseFlag() throws {
    ExperimentalFlags.convertCamelToSnakeCaseInToolDescription = false
    let tools: [Tool] = [SnakeCaseTool()]
    let toolManager = ToolManager(tools: tools)

    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"another_tool","description":"Another tool.","parameters":{"type":"object","properties":{"another_param":{"type":"string","description":"Another param."}},"required":["another_param"]}}}]
      """
    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_missingToolParam() throws {
    let tools: [Tool] = [MissingToolParamTool()]
    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"tool_with_missing_param","description":"Tool with missing param description.","parameters":{"type":"object","properties":{"first_int":{"type":"integer"}},"required":["first_int"]}}}]
      """
    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolsDescriptionConversion_noParams() throws {
    let tools: [Tool] = [VoidReturnTool()]
    let toolManager = ToolManager(tools: tools)
    let actualJson = toolManager.toolsJsonDescription
    let expectedJson = """
      [{"type":"function","function":{"name":"do_nothing","description":"A tool that does nothing and returns no value."}}]
      """
    try assertJsonEqual(actual: actualJson, expected: expectedJson)
  }

  func testToolManagerExecute() async throws {
    let tools: [Tool] = [AddIntegersTool()]
    let toolManager = ToolManager(tools: tools)

    let args: [String: Any] = ["first_int": 10, "second_int": 20]

    let result = try await toolManager.execute(name: "add_integers", arguments: args)

    guard let dict = result as? [String: Int] else {
      XCTFail("Result is not a dictionary: \(result)")
      return
    }

    XCTAssertEqual(dict["sum"], 30)
  }

  func testToolManagerExecute_getWeather_withoutUnit() async throws {
    let tools: [Tool] = [GetCurrentWeatherTool()]
    let toolManager = ToolManager(tools: tools)
    let args: [String: Any] = ["city": "London"]
    let result = try await toolManager.execute(name: "get_current_weather", arguments: args)
    guard let dict = result as? [String: Any],
      let temp = dict["temperature"] as? Int,
      let unit = dict["unit"] as? String
    else {
      XCTFail("Result is not correct dictionary: \(result)")
      return
    }
    XCTAssertEqual(temp, 25)
    XCTAssertEqual(unit, "celsius")
  }

  func testToolManagerExecute_toolNotFound() async {
    let tools: [Tool] = [AddIntegersTool()]
    let toolManager = ToolManager(tools: tools)
    let args: [String: Any] = ["first_int": 10, "second_int": 20]
    do {
      _ = try await toolManager.execute(name: "non_existent_tool", arguments: args)
      XCTFail("Expected to throw error for tool not found, but it did not.")
    } catch LiteRTLMError.tool(.notFound) {
      // Expected error
    } catch {
      XCTFail("Expected LiteRTLMError.tool(.notFound), but caught \(error)")
    }
  }

  func testToolManagerExecute_missingRequiredArgument() async {
    let tools: [Tool] = [AddIntegersTool()]
    let toolManager = ToolManager(tools: tools)
    let args: [String: Any] = ["first_int": 10]  // Missing "second_int"
    do {
      _ = try await toolManager.execute(name: "add_integers", arguments: args)
      XCTFail("Expected to throw error for missing required argument, but it did not.")
    } catch {}
  }

  func testToolManagerExecute_wrongArgumentType() async {
    let tools: [Tool] = [AddIntegersTool()]
    let toolManager = ToolManager(tools: tools)
    let args: [String: Any] = ["first_int": 10, "second_int": "20"]  // "second_int" is string not int
    do {
      _ = try await toolManager.execute(name: "add_integers", arguments: args)
      XCTFail("Expected to throw error for wrong argument type, but it did not.")
    } catch {}
  }

  func testToolManagerExecute_listParam() async throws {
    let tools: [Tool] = [SumListTool()]
    let toolManager = ToolManager(tools: tools)
    let args: [String: Any] = ["numbers": [1, 2, 3]]
    let result = try await toolManager.execute(name: "sum_list", arguments: args)
    XCTAssertEqual(result as? Int, 6)
  }

  func testToolManagerExecute_2DListParam() async throws {
    let tools: [Tool] = [DeterminantTool()]
    let toolManager = ToolManager(tools: tools)
    let args: [String: Any] = ["matrix": [[1, 2], [3, 4]]]
    let result = try await toolManager.execute(name: "determinant", arguments: args)
    XCTAssertEqual(result as? Int, -2)
  }

  func testToolManagerExecute_voidReturn() async throws {
    let tools: [Tool] = [VoidReturnTool()]
    let toolManager = ToolManager(tools: tools)
    let result = try await toolManager.execute(name: "do_nothing", arguments: [:])
    XCTAssertEqual(result as? String, "")
  }

  func testToolManagerExecute_nullReturn() async throws {
    let tools: [Tool] = [NullReturnTool()]
    let toolManager = ToolManager(tools: tools)
    let result = try await toolManager.execute(name: "return_null", arguments: [:])
    XCTAssertEqual(result as? String, "nil")
  }

  func testToolManagerExecute_listReturn() async throws {
    let tools: [Tool] = [ListReturnTool()]
    let toolManager = ToolManager(tools: tools)
    let result = try await toolManager.execute(name: "return_list", arguments: [:])
    XCTAssertEqual(result as? [String], ["a", "b", "c"])
  }

  func testToolManagerExecute_mapReturn() async throws {
    let tools: [Tool] = [MapReturnTool()]
    let toolManager = ToolManager(tools: tools)
    let result = try await toolManager.execute(name: "return_map", arguments: [:])
    XCTAssertEqual(result as? [String: Int], ["a": 1, "b": 2])
  }

  // MARK: - Helpers

  private func assertJsonEqual(
    actual: String, expected: String, file: StaticString = #file, line: UInt = #line
  ) throws {
    guard let actualData = actual.data(using: .utf8),
      let expectedData = expected.data(using: .utf8)
    else {
      XCTFail("Failed to convert strings to data", file: file, line: line)
      return
    }
    let actualObj = try JSONSerialization.jsonObject(with: actualData) as? [Any]
    let expectedObj = try JSONSerialization.jsonObject(with: expectedData) as? [Any]
    guard let actualArr = actualObj, let expectedArr = expectedObj else {
      XCTFail("JSON root is not an array", file: file, line: line)
      return
    }
    XCTAssertEqual(
      actualArr.count, expectedArr.count, "Array count mismatch", file: file, line: line)
    for (index, item) in actualArr.enumerated() {
      guard let actualDict = item as? [String: Any],
        let expectedDict = expectedArr[index] as? [String: Any]
      else {
        XCTFail("Item at index \(index) is not a dictionary", file: file, line: line)
        continue
      }
      let actualNsDict = actualDict as NSDictionary
      let expectedNsDict = expectedDict as NSDictionary
      XCTAssertEqual(
        actualNsDict, expectedNsDict, "Mismatch at index \(index)", file: file, line: line)
    }
  }
}

// MARK: - Test Tools

struct AddIntegersTool: Tool {
  static let name = "addIntegers"
  static let description = "Adds two integers and returns the result."

  @ToolParam(description: "The first integer.")
  var firstInt: Int

  @ToolParam(description: "The second integer.")
  var secondInt: Int

  func run() async throws -> Any {
    return ["sum": firstInt + secondInt]
  }
}

struct DeterminantTool: Tool {
  static let name = "determinant"
  static let description = "Calculates the determinant of a 2x2 matrix."

  @ToolParam(description: "2x2 matrix.")
  var matrix: [[Int]]

  func run() async throws -> Any {
    guard matrix.count == 2, matrix[0].count == 2, matrix[1].count == 2 else {
      throw NSError(
        domain: "DeterminantTool", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Matrix must be 2x2"])
    }
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
  }
}

struct GetCurrentWeatherTool: Tool {
  static let name = "getCurrentWeather"
  static let description = "Get the current weather for a city"

  @ToolParam(description: "The city name, e.g., San Francisco")
  var city: String

  @ToolParam(description: "Optional country code, e.g., US")
  var country: String? = nil

  @ToolParam(description: "Temperature unit (celsius or fahrenheit). Default: celsius")
  var unit: String = "celsius"

  func run() async throws -> Any { return ["temperature": 25, "unit": unit] }
}

struct SumListTool: Tool {
  static let name = "sumList"
  static let description = "Sums a list of integers."

  @ToolParam(description: "List of integers to sum.")
  var numbers: [Int]

  func run() async throws -> Any { return numbers.reduce(0, +) }
}

struct SnakeCaseTool: Tool {
  static let name = "another_tool"
  static let description = "Another tool."

  @ToolParam(description: "Another param.")
  var another_param: String

  func run() async throws -> Any { [:] }
}

struct AllOptionalParamsTool: Tool {
  static let name = "getSum"
  static let description = "All params are optional."

  @ToolParam(description: "")
  var foo: Int = 1

  @ToolParam(description: "")
  var bar: Int = 2

  func run() async throws -> Any {
    return (foo) + (bar)
  }
}

struct MissingToolParamTool: Tool {
  static let name = "toolWithMissingParam"
  static let description = "Tool with missing param description."

  @ToolParam(description: "")
  var firstInt: Int

  func run() async throws -> Any {
    return "\(firstInt)"
  }
}

struct VoidReturnTool: Tool {
  static let name = "doNothing"
  static let description = "A tool that does nothing and returns no value."
  func run() async throws -> Any {
    return ()
  }
}

struct NullReturnTool: Tool {
  static let name = "returnNull"
  static let description = "A tool that always returns null."
  func run() async throws -> Any {
    let ret: String? = nil
    return ret as Any
  }
}

struct ListReturnTool: Tool {
  static let name = "returnList"
  static let description = "A tool that returns a list."
  func run() async throws -> Any {
    return ["a", "b", "c"]
  }
}

struct MapReturnTool: Tool {
  static let name = "returnMap"
  static let description = "A tool that returns a map."
  func run() async throws -> Any {
    return ["a": 1, "b": 2]
  }
}
