// Copyright 2025 The Google AI Edge Authors.
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

use super::*;

/// Returns the `value` argument of the single tool call in `result`.
fn single_value(result: &ffi::ToolCalls) -> Box<JsonValue> {
    assert!(result.is_ok, "parse failed: {}", result.error);
    assert_eq!(result.tool_calls.len(), 1);
    result.tool_calls[0].object_get("arguments").object_get("value")
}

/// Parses `call:test_tool{value:<literal>}` through the FC parser.
fn fc_value(literal: &str) -> Box<JsonValue> {
    single_value(&parse_fc_expression(&format!("call:test_tool{{value:{}}}", literal)))
}

#[test]
fn reports_integers_as_integers() {
    for (literal, expected) in [("1000", 1000), ("-678", -678), ("0", 0)] {
        let value = fc_value(literal);
        assert!(value.is_number(), "{} should be a number", literal);
        assert!(value.is_integer(), "{} should be an integer", literal);
        assert_eq!(value.get_integer(), expected);
    }
}

#[test]
fn does_not_report_floats_as_integers() {
    for literal in ["1000.0", "3.14", "1e3"] {
        let value = fc_value(literal);
        assert!(value.is_number(), "{} should be a number", literal);
        assert!(!value.is_integer(), "{} should not be an integer", literal);
    }
    assert_eq!(fc_value("1000.0").get_number(), 1000.0);
}

#[test]
fn integers_remain_readable_as_f64() {
    // is_number and get_number keep working for integers, so callers that only
    // need a double are unaffected.
    assert_eq!(fc_value("1000").get_number(), 1000.0);
}

#[test]
fn preserves_integers_beyond_f64_precision() {
    let value = fc_value("9007199254740993");
    assert!(value.is_integer());
    assert_eq!(value.get_integer(), 9007199254740993);
}

#[test]
fn non_numbers_are_not_integers() {
    let result =
        parse_fc_expression("call:t{s:<escape>hi<escape>,b:true,n:null,arr:[1],obj:{k:1}}");
    assert!(result.is_ok, "parse failed: {}", result.error);
    let args = result.tool_calls[0].object_get("arguments");
    for key in ["s", "b", "n", "arr", "obj"] {
        let value = args.object_get(key);
        assert!(!value.is_integer(), "{} should not be an integer", key);
        assert_eq!(value.get_integer(), 0);
    }
}

#[test]
fn json_and_python_parsers_also_preserve_integers() {
    // The accessor, not the FC parser, is the shared loss point: the JSON and
    // Python parsers always produced correct integers in Rust and lost them
    // when crossing the bridge.
    let json_value =
        single_value(&parse_json_expression(r#"{"name":"t","arguments":{"value":1000}}"#));
    assert!(json_value.is_integer());
    assert_eq!(json_value.get_integer(), 1000);

    let python_value = single_value(&parse_python_expression("t(value=1000)"));
    assert!(python_value.is_integer());
    assert_eq!(python_value.get_integer(), 1000);
}

#[test]
fn python_parser_still_distinguishes_floats() {
    let value = single_value(&parse_python_expression("t(value=3.14)"));
    assert!(value.is_number());
    assert!(!value.is_integer());
    assert_eq!(value.get_number(), 3.14);
}
