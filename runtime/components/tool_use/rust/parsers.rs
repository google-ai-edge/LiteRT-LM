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

use indexmap::IndexMap;
use json_parser;
use python_parser;
use serde_json::{Number, Value};
use std::sync::Arc;

#[cxx::bridge(namespace = "litert::lm")]
mod ffi {
    struct ToolCalls {
        tool_calls: Vec<JsonValue>,
        is_ok: bool,
        error: String,
    }

    extern "Rust" {
        type JsonValue;

        fn is_null(self: &JsonValue) -> bool;
        fn is_bool(self: &JsonValue) -> bool;
        fn is_number(self: &JsonValue) -> bool;
        fn is_string(self: &JsonValue) -> bool;
        fn is_array(self: &JsonValue) -> bool;
        fn is_object(self: &JsonValue) -> bool;

        fn get_bool(self: &JsonValue) -> bool;
        fn get_number(self: &JsonValue) -> f64;
        fn is_integer(self: &JsonValue) -> bool;
        fn get_integer(self: &JsonValue) -> i64;
        fn get_string(self: &JsonValue) -> &str;

        fn array_len(self: &JsonValue) -> usize;
        fn array_get(self: &JsonValue) -> Vec<JsonValue>;

        fn object_has_key(self: &JsonValue, key: &str) -> bool;
        fn object_keys(self: &JsonValue) -> Vec<String>;
        fn object_get(self: &JsonValue, key: &str) -> Box<JsonValue>;

        fn parse_python_expression(text: &str) -> ToolCalls;
        fn parse_fc_expression(text: &str) -> ToolCalls;
        fn parse_json_expression(text: &str) -> ToolCalls;
    }
}

pub struct JsonValue {
    inner: Arc<InnerValue>,
}

enum InnerValue {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Arc<InnerValue>>),
    Object(IndexMap<String, Arc<InnerValue>>),
}

impl From<Value> for InnerValue {
    fn from(v: Value) -> Self {
        match v {
            Value::Null => InnerValue::Null,
            Value::Bool(b) => InnerValue::Bool(b),
            Value::Number(n) => InnerValue::Number(n),
            Value::String(s) => InnerValue::String(s),
            Value::Array(a) => {
                InnerValue::Array(a.into_iter().map(|x| Arc::new(InnerValue::from(x))).collect())
            }
            Value::Object(o) => InnerValue::Object(
                o.into_iter().map(|(k, v)| (k, Arc::new(InnerValue::from(v)))).collect(),
            ),
        }
    }
}

impl JsonValue {
    fn is_null(self: &JsonValue) -> bool {
        matches!(*self.inner, InnerValue::Null)
    }
    fn is_bool(self: &JsonValue) -> bool {
        matches!(*self.inner, InnerValue::Bool(_))
    }
    fn is_number(self: &JsonValue) -> bool {
        matches!(*self.inner, InnerValue::Number(_))
    }
    fn is_string(self: &JsonValue) -> bool {
        matches!(*self.inner, InnerValue::String(_))
    }
    fn is_array(self: &JsonValue) -> bool {
        matches!(*self.inner, InnerValue::Array(_))
    }
    fn is_object(self: &JsonValue) -> bool {
        matches!(*self.inner, InnerValue::Object(_))
    }

    fn get_bool(self: &JsonValue) -> bool {
        match &*self.inner {
            InnerValue::Bool(b) => *b,
            _ => false,
        }
    }
    fn get_number(self: &JsonValue) -> f64 {
        match &*self.inner {
            InnerValue::Number(n) => n.as_f64().unwrap_or(0.0),
            _ => 0.0,
        }
    }
    /// Returns true if this value is a number that was written as an integer.
    ///
    /// `serde_json::Number` already records whether the literal had a fraction
    /// or exponent; `get_number` discards that by widening everything to f64.
    /// Callers should consult this before `get_number` so that `1000` is not
    /// silently re-serialized as `1000.0`.
    ///
    /// Note this is deliberately gated on `is_i64` alone: values above
    /// `i64::MAX` keep falling through to the f64 path, which is exactly the
    /// pre-existing behavior, rather than trapping to zero in `get_integer`.
    fn is_integer(self: &JsonValue) -> bool {
        match &*self.inner {
            InnerValue::Number(n) => n.is_i64(),
            _ => false,
        }
    }
    fn get_integer(self: &JsonValue) -> i64 {
        match &*self.inner {
            InnerValue::Number(n) => n.as_i64().unwrap_or(0),
            _ => 0,
        }
    }
    fn get_string(self: &JsonValue) -> &str {
        match &*self.inner {
            InnerValue::String(s) => s,
            _ => "",
        }
    }

    fn array_len(self: &JsonValue) -> usize {
        match &*self.inner {
            InnerValue::Array(a) => a.len(),
            _ => 0,
        }
    }

    fn array_get(self: &JsonValue) -> Vec<JsonValue> {
        match &*self.inner {
            InnerValue::Array(arr) => arr.iter().map(|a| JsonValue { inner: a.clone() }).collect(),
            _ => Vec::new(),
        }
    }

    fn object_has_key(self: &JsonValue, key: &str) -> bool {
        match &*self.inner {
            InnerValue::Object(obj) => obj.contains_key(key),
            _ => false,
        }
    }

    fn object_keys(self: &JsonValue) -> Vec<String> {
        match &*self.inner {
            InnerValue::Object(obj) => obj.keys().cloned().collect(),
            _ => Vec::new(),
        }
    }

    fn object_get(self: &JsonValue, key: &str) -> Box<JsonValue> {
        match &*self.inner {
            InnerValue::Object(obj) => match obj.get(key) {
                Some(val) => Box::new(JsonValue { inner: val.clone() }),
                None => Box::new(JsonValue { inner: Arc::new(InnerValue::Null) }),
            },
            _ => Box::new(JsonValue { inner: Arc::new(InnerValue::Null) }),
        }
    }
}

pub fn parse_python_expression(text: &str) -> ffi::ToolCalls {
    match python_parser::parse_python_expression(text) {
        Ok(v) => ffi::ToolCalls {
            tool_calls: v
                .into_iter()
                .map(|v| JsonValue { inner: Arc::new(InnerValue::from(v)) })
                .collect(),
            is_ok: true,
            error: "".to_string(),
        },
        Err(e) => ffi::ToolCalls { tool_calls: Vec::new(), is_ok: false, error: e },
    }
}

pub fn parse_fc_expression(text: &str) -> ffi::ToolCalls {
    match fc_parser::parse_fc_expression(text) {
        Ok(v) => ffi::ToolCalls {
            tool_calls: v
                .into_iter()
                .map(|v| JsonValue { inner: Arc::new(InnerValue::from(v)) })
                .collect(),
            is_ok: true,
            error: "".to_string(),
        },
        Err(e) => ffi::ToolCalls { tool_calls: Vec::new(), is_ok: false, error: e },
    }
}

pub fn parse_json_expression(text: &str) -> ffi::ToolCalls {
    match json_parser::parse_json_expression(text) {
        Ok(v) => ffi::ToolCalls {
            tool_calls: v
                .into_iter()
                .map(|v| JsonValue { inner: Arc::new(InnerValue::from(v)) })
                .collect(),
            is_ok: true,
            error: "".to_string(),
        },
        Err(e) => ffi::ToolCalls { tool_calls: Vec::new(), is_ok: false, error: e },
    }
}

#[cfg(test)]
mod parsers_test;
