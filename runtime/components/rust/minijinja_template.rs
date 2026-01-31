// Copyright 2026 The Google AI Edge Authors.
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

use minijinja::Environment;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[cxx::bridge(namespace = "litert::lm")]
mod ffi {
    #[derive(Clone, Copy, Default)]
    struct ChatTemplateCapabilities {
        supports_tools: bool,
        supports_tool_calls: bool,
        supports_system_role: bool,
        supports_parallel_tool_calls: bool,
        supports_tool_call_id: bool,
        requires_typed_content: bool,
        supports_single_turn: bool,
    }

    struct ApplyResult {
        content: String,
        error: String,
        is_ok: bool,
    }

    extern "Rust" {
        type MinijinjaTemplate;

        fn new_minijinja_template(source: String) -> Box<MinijinjaTemplate>;
        fn apply(self: &MinijinjaTemplate, inputs_json: String) -> ApplyResult;
        fn source(self: &MinijinjaTemplate) -> &str;
        fn get_capabilities(self: &MinijinjaTemplate) -> ChatTemplateCapabilities;
        fn get_error(self: &MinijinjaTemplate) -> String;
        fn clone_template(self: &MinijinjaTemplate) -> Box<MinijinjaTemplate>;
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct ChatTemplateInputs {
    #[serde(default)]
    messages: Value,
    #[serde(default)]
    tools: Value,
    #[serde(default)]
    add_generation_prompt: bool,
    #[serde(default)]
    extra_context: Value,
    #[serde(default)]
    now: Option<i64>,
    #[serde(default)]
    bos_token: Value,
    #[serde(default)]
    eos_token: Value,
}

#[derive(Clone)]
pub struct MinijinjaTemplate {
    source: String,
    caps: ffi::ChatTemplateCapabilities,
    creation_error: String,
}

fn detect_capabilities(source: &str) -> ffi::ChatTemplateCapabilities {
    let mut caps = ffi::ChatTemplateCapabilities::default();
    let mut env = Environment::new();
    env.set_keep_trailing_newline(false);
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    if let Ok(tmpl) = env.template_from_str(source) {
        let undeclared = tmpl.undeclared_variables(true);
        if undeclared.contains("tools") {
            caps.supports_tools = true;
        }

        let test_content = "test content";
        let test_str_user_msg = json!({ "role": "user", "content": test_content });
        let test_typed_user_msg = json!({
            "role": "user",
            "content": [{ "type": "text", "text": test_content }]
        });

        let try_render = |msg: Value| -> bool {
            let ctx = json!({
                "messages": [msg],
                "add_generation_prompt": false,
                "tools": [],
                "extra_context": {}
            });
            tmpl.render(ctx).map(|s| s.contains(test_content)).unwrap_or(false)
        };

        let str_works = try_render(test_str_user_msg);
        let typed_works = try_render(test_typed_user_msg);

        if !str_works && typed_works {
            caps.requires_typed_content = true;
        }
    }

    if source.contains("tool_calls") {
        caps.supports_tool_calls = true;
    }
    if source.contains("tool_call_id") {
        caps.supports_tool_call_id = true;
    }
    if !caps.supports_tools && source.contains("tools") {
        caps.supports_tools = true;
    }

    if source.contains("system") {
        caps.supports_system_role = true;
    }

    if caps.supports_tool_calls && source.contains("for") && source.contains("tool_calls") {
        caps.supports_parallel_tool_calls = true;
    }

    if source.contains("is_appending_to_prefill") {
        caps.supports_single_turn = true;
    }

    caps
}

fn new_minijinja_template(source: String) -> Box<MinijinjaTemplate> {
    let env = Environment::new();
    let creation_error = match env.template_from_str(&source) {
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    };

    let caps = detect_capabilities(&source);
    Box::new(MinijinjaTemplate { source, caps, creation_error })
}

fn strftime_now(state: &minijinja::State, format: String) -> Result<String, minijinja::Error> {
    let now_val = state.lookup("now");
    let timestamp = if now_val.is_none() || now_val.as_ref().is_some_and(|v| v.is_undefined()) {
        chrono::Utc::now().timestamp()
    } else {
        let val = now_val.ok_or_else(|| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "now value is missing")
        })?;
        i64::try_from(val).map_err(|_| {
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "now must be an integer")
        })?
    };

    let dt = chrono::DateTime::from_timestamp(timestamp, 0).ok_or_else(|| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, "invalid timestamp")
    })?;
    Ok(dt.format(&format).to_string())
}

// A serde_json formatter that adds spaces after commas in arrays and object keys,
// and a space after the colon in object key-value pairs.
struct SpaceFormatter;

impl serde_json::ser::Formatter for SpaceFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

fn tojson(value: minijinja::Value) -> Result<String, minijinja::Error> {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, SpaceFormatter);
    value.serialize(&mut serializer).map_err(|err| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, err.to_string())
    })?;
    String::from_utf8(buf).map_err(|err| {
        minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, err.to_string())
    })
}

fn is_none(value: minijinja::Value) -> bool {
    value.is_undefined()
}

impl MinijinjaTemplate {
    fn apply(&self, inputs_json: String) -> ffi::ApplyResult {
        match self.apply_impl(inputs_json) {
            Ok(s) => ffi::ApplyResult { content: s, error: String::new(), is_ok: true },
            Err(e) => {
                ffi::ApplyResult { content: String::new(), error: e.to_string(), is_ok: false }
            }
        }
    }

    fn source(&self) -> &str {
        &self.source
    }

    fn get_capabilities(&self) -> ffi::ChatTemplateCapabilities {
        self.caps
    }

    fn get_error(&self) -> String {
        self.creation_error.clone()
    }

    fn clone_template(&self) -> Box<MinijinjaTemplate> {
        Box::new(self.clone())
    }

    fn apply_impl(&self, inputs_json: String) -> Result<String, Box<dyn std::error::Error>> {
        if !self.creation_error.is_empty() {
            return Err(format!("Template creation failed: {}", self.creation_error).into());
        }

        let inputs: ChatTemplateInputs = serde_json::from_str(&inputs_json)?;
        let actual_messages = match inputs.messages {
            Value::Array(arr) => arr,
            _ => vec![],
        };

        let mut env = Environment::new();
        env.set_keep_trailing_newline(false);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", tojson);
        env.add_test("none", is_none);
        env.add_template("main", &self.source)?;
        let tmpl = env.get_template("main")?;

        let mut ctx = serde_json::Map::new();
        ctx.insert("messages".to_string(), Value::Array(actual_messages));

        if !inputs.tools.is_null() {
            ctx.insert("tools".to_string(), inputs.tools.clone());
        }

        ctx.insert("add_generation_prompt".to_string(), Value::Bool(inputs.add_generation_prompt));

        if let Some(now) = inputs.now {
            ctx.insert("now".to_string(), Value::Number(serde_json::Number::from(now)));
        }

        ctx.insert("bos_token".to_string(), inputs.bos_token.clone());
        ctx.insert("eos_token".to_string(), inputs.eos_token.clone());

        if let Some(obj) = inputs.extra_context.as_object() {
            for (k, v) in obj {
                ctx.insert(k.clone(), v.clone());
            }
        }

        let res = tmpl.render(Value::Object(ctx))?;
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_render() {
        let source = "Hello {{ messages[0].content }}";
        let wrapper = new_minijinja_template(source.to_string());
        assert!(wrapper.creation_error.is_empty());

        let inputs = r#"{
            "messages": [{"role": "user", "content": "World"}],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {}
        }"#;

        let res = wrapper.apply(inputs.to_string());
        assert!(res.is_ok);
        assert_eq!(res.content, "Hello World");
    }

    #[test]
    fn test_requires_typed_content() {
        // Template that expects content to be a list of dicts.
        let source_requires_typed_content = "{% for m in messages %}{% for block in m.content %}{{ block.text }}{% endfor %}{% endfor %}";
        let wrapper_requires_typed_content =
            new_minijinja_template(source_requires_typed_content.to_string());
        assert!(wrapper_requires_typed_content.caps.requires_typed_content);

        // Template that works with string content.
        let source_any_content = "{{ messages[0].content }}";
        let wrapper_any_content = new_minijinja_template(source_any_content.to_string());
        assert!(!wrapper_any_content.caps.requires_typed_content);
    }

    #[test]
    fn test_clone() {
        let source = "Hello {{ messages[0].content }}";
        let wrapper = new_minijinja_template(source.to_string());
        let cloned = wrapper.clone_template();
        assert_eq!(wrapper.source, cloned.source);
    }

    #[test]
    fn test_strftime_now() {
        let source = "{{ strftime_now('%Y-%m-%d') }}";
        let wrapper = new_minijinja_template(source.to_string());

        // Test with specific time (2025-01-01)
        let inputs = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {},
            "now": 1735689600
        });

        let res = wrapper.apply(inputs.to_string());
        assert!(res.is_ok);
        assert_eq!(res.content, "2025-01-01");

        // Test fallback to current time when now is missing
        let inputs_no_now = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {}
        });
        let res_no_now = wrapper.apply(inputs_no_now.to_string());
        assert!(res_no_now.is_ok);
        // Basic check that it formatted something resembling a year at the start
        assert!(res_no_now.content.len() >= 4);
    }

    #[test]
    fn test_tojson_spacing() {
        let source = "{{ data|tojson }}";
        let wrapper = new_minijinja_template(source.to_string());

        // Test object spacing.
        let inputs_obj = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {"data": {"a": 1, "b": 2}}
        });
        let res_obj = wrapper.apply(inputs_obj.to_string());
        assert!(res_obj.is_ok);
        assert!(res_obj.content.contains("{\"a\": 1, \"b\": 2}"));

        // Test list spacing
        let inputs_list = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {"data": [1, 2]}
        });
        let res_list = wrapper.apply(inputs_list.to_string());
        assert!(res_list.is_ok);
        assert_eq!(res_list.content, "[1, 2]");
    }

    #[test]
    fn test_is_none() {
        let source =
            "{% if does_not_exist is none %}does_not_exist is none{% else %}FAIL{% endif %},
        {% if existing is not none %}existing is not none{% else %}FAIL{% endif %}";
        let wrapper = new_minijinja_template(source.to_string());

        let inputs = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {"existing": 123}
        });
        let res = wrapper.apply(inputs.to_string());
        assert!(res.is_ok);
        assert_eq!(res.content, "does_not_exist is none,\nexisting is not none");
    }
}
