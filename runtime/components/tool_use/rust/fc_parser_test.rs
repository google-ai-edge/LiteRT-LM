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

/// Parses `call:test_tool{value:<literal>}` and returns the `value` argument.
fn parse_value_literal(literal: &str) -> Value {
    let text = format!("call:test_tool{{value:{}}}", literal);
    let tool_calls = parse_fc_expression(&text).expect("parse failed");
    assert_eq!(tool_calls.len(), 1);
    tool_calls[0]["arguments"]["value"].clone()
}

#[test]
fn parses_integers_as_integers() {
    // The grammar has a single NUMBER token for both integers and floats, so
    // the type is decided from the literal text. Parsing everything as f64
    // would turn 1000 into 1000.0, which a strict consumer rejects for a field
    // declared as an integer.
    for literal in ["1000", "-678", "0"] {
        let value = parse_value_literal(literal);
        assert!(value.is_i64(), "{} should be an integer, got {}", literal, value);
        assert_eq!(value.to_string(), literal);
    }
}

#[test]
fn parses_floats_as_floats() {
    for literal in ["1000.0", "3.14", "-4.5"] {
        let value = parse_value_literal(literal);
        assert!(value.is_f64(), "{} should be a float, got {}", literal, value);
    }
}

#[test]
fn keeps_exponent_notation_as_float() {
    // The writer chose exponent notation and there is no way to tell an
    // intended integer from an intended float, so stay conservative.
    let value = parse_value_literal("1e3");
    assert!(value.is_f64());
    assert_eq!(value.as_f64(), Some(1000.0));
}

#[test]
fn preserves_integers_beyond_f64_precision() {
    // Above 2^53 an f64 round trip silently changes the digits.
    let value = parse_value_literal("9007199254740993");
    assert_eq!(value.as_i64(), Some(9007199254740993));
    assert_eq!(value.to_string(), "9007199254740993");
}

#[test]
fn preserves_integers_nested_in_containers() {
    let tool_calls =
        parse_fc_expression("call:t{list:[10,20],obj:{inner:30}}").expect("parse failed");
    let args = &tool_calls[0]["arguments"];
    assert!(args["list"][0].is_i64());
    assert!(args["list"][1].is_i64());
    assert!(args["obj"]["inner"].is_i64());
}

#[test]
fn rejects_malformed_numbers() {
    // Falling through both integer and float parsing is still an error.
    assert!(parse_fc_expression("call:t{value:1.2.3}").is_err());
}
