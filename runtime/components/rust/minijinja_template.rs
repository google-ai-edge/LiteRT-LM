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
        supports_single_turn: bool,
    }

    struct ApplyResult {
        content: String,
        error: String,
        is_ok: bool,
    }

    struct UndeclaredVariablesResult {
        variables: Vec<String>,
        error: String,
        is_ok: bool,
    }

    extern "Rust" {
        type MinijinjaTemplate;

        fn new_minijinja_template(source: String) -> Box<MinijinjaTemplate>;
        fn apply(self: &MinijinjaTemplate, inputs_json: String) -> ApplyResult;
        fn source(self: &MinijinjaTemplate) -> &str;
        fn get_capabilities(self: &MinijinjaTemplate) -> ChatTemplateCapabilities;
        fn get_undeclared_variables(self: &MinijinjaTemplate) -> UndeclaredVariablesResult;
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
    ffi::ChatTemplateCapabilities {
        supports_single_turn: source.contains("is_appending_to_prefill"),
    }
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

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
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

// As in Jinja2, `is none` is true for null values. Undefined values also count
// as none, e.g. `tools is none` when no tools are passed.
fn is_none(value: minijinja::Value) -> bool {
    value.is_undefined() || value.is_none()
}

fn lstrip(s: std::borrow::Cow<'_, str>, chars: Option<std::borrow::Cow<'_, str>>) -> String {
    match chars {
        Some(chars) => {
            let chars = chars.chars().collect::<Vec<_>>();
            s.trim_start_matches(&chars[..]).to_string()
        }
        None => s.trim_start().to_string(),
    }
}

fn rstrip(s: std::borrow::Cow<'_, str>, chars: Option<std::borrow::Cow<'_, str>>) -> String {
    match chars {
        Some(chars) => {
            let chars = chars.chars().collect::<Vec<_>>();
            s.trim_end_matches(&chars[..]).to_string()
        }
        None => s.trim_end().to_string(),
    }
}

#[allow(unexpected_cfgs)]
fn create_env() -> Result<Environment<'static>, minijinja::Error> {
    let mut env = Environment::new();
    // Keep trailing newlines in the rendered prompt text (e.g., `<think>\n`
    // when prefilling thinking channels) so that prompt specifications and
    // model training formatting are preserved exactly without being stripped.
    // TODO(b/590227335): Remove minijinja_v2 compatibility once Chromium
    // upgrades to minijinja v3.
    #[cfg(minijinja_v2)]
    {
        env.set_keep_trailing_newline(true);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
    }
    #[cfg(not(minijinja_v2))]
    {
        use minijinja::syntax::SyntaxConfig;
        env.set_syntax(
            SyntaxConfig::builder()
                .keep_trailing_newline(true)
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()?,
        );
    }
    env.add_function("strftime_now", strftime_now);
    env.add_function("raise_exception", raise_exception);
    env.add_filter("tojson", tojson);
    env.add_filter("lstrip", lstrip);
    env.add_filter("rstrip", rstrip);
    env.add_test("none", is_none);
    Ok(env)
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

// Extracts bare identifiers from a Jinja expression/statement, skipping string
// literals (with backslash escapes) and attribute accesses (`.attr`).
fn collect_bare_idents<'a>(expr: &'a str, out: &mut std::collections::HashSet<&'a str>) {
    let mut chars = expr.char_indices().peekable();
    let mut prev_was_dot = false;
    while let Some((idx, ch)) = chars.next() {
        if ch == '\'' || ch == '"' {
            let quote = ch;
            while let Some((_, qch)) = chars.next() {
                if qch == '\\' {
                    let _ = chars.next();
                } else if qch == quote {
                    break;
                }
            }
            prev_was_dot = false;
        } else if ch == '.' {
            prev_was_dot = true;
        } else if is_ident_start(ch) {
            let mut end = idx + ch.len_utf8();
            while let Some(&(next_idx, next_ch)) = chars.peek() {
                if is_ident_char(next_ch) {
                    end = next_idx + next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            if !prev_was_dot {
                out.insert(&expr[idx..end]);
            }
            prev_was_dot = false;
        } else if !ch.is_whitespace() {
            prev_was_dot = false;
        }
    }
}

// Collects top-level macro names and `{% set name = ... %}` local variables
// defined in `source`. Minijinja's `undeclared_variables()` walks the AST
// top-down with lexical scoping on `{% if %}` blocks, so it reports
// forward-referenced macros and variables assigned inside `{% if %}`/`{% else %}`
// branches even though they are template-local at runtime.
fn collect_template_local_definitions<'a>(source: &'a str) -> std::collections::HashSet<&'a str> {
    let mut locals = std::collections::HashSet::new();
    let mut referenced_before_set = std::collections::HashSet::new();
    let mut macro_depth: usize = 0;
    let mut pos = 0;

    while pos < source.len() {
        let rest = &source[pos..];
        let next_open = rest.find('{');
        let Some(rel_open) = next_open else {
            break;
        };
        let tag_start = pos + rel_open;
        let after_brace = &source[tag_start + 1..];
        let Some(kind) = after_brace.chars().next() else {
            break;
        };

        if kind == '#' {
            // Skip Jinja comment `{# ... #}`.
            if let Some(rel_close) = source[tag_start + 2..].find("#}") {
                pos = tag_start + 2 + rel_close + 2;
            } else {
                break;
            }
        } else if kind == '{' {
            // Expression tag `{{ ... }}`.
            let Some(rel_close) = source[tag_start + 2..].find("}}") else {
                break;
            };
            let expr = source[tag_start + 2..tag_start + 2 + rel_close]
                .trim_matches(|c: char| c.is_whitespace() || c == '-');
            if macro_depth == 0 {
                collect_bare_idents(expr, &mut referenced_before_set);
            }
            pos = tag_start + 2 + rel_close + 2;
        } else if kind == '%' {
            // Statement tag `{% ... %}`.
            let Some(rel_close) = source[tag_start + 2..].find("%}") else {
                break;
            };
            let stmt = source[tag_start + 2..tag_start + 2 + rel_close]
                .trim_matches(|c: char| c.is_whitespace() || c == '-');
            if let Some(after_macro) =
                stmt.strip_prefix("macro").filter(|s| s.starts_with(char::is_whitespace))
            {
                if macro_depth == 0 {
                    let name = after_macro.trim_start().split(|c: char| !is_ident_char(c)).next();
                    if let Some(name) = name.filter(|n| !n.is_empty()) {
                        locals.insert(name);
                    }
                }
                macro_depth += 1;
            } else if stmt == "endmacro"
                || stmt.strip_prefix("endmacro").is_some_and(|s| s.starts_with(char::is_whitespace))
            {
                macro_depth = macro_depth.saturating_sub(1);
            } else if macro_depth == 0 {
                if let Some(after_set) =
                    stmt.strip_prefix("set").filter(|s| s.starts_with(char::is_whitespace))
                {
                    if let Some((lhs, rhs)) = after_set.split_once('=') {
                        let name = lhs.trim();
                        collect_bare_idents(rhs, &mut referenced_before_set);
                        // Ignore namespace attribute assignments (`ns.field = ...`)
                        // and variables already referenced before this `set`
                        // (including `{% if x is not defined %}{% set x = ... %}`
                        // and self-referential `{% set x = x | default(...) %}`).
                        if !name.is_empty()
                            && name.chars().all(is_ident_char)
                            && !referenced_before_set.contains(name)
                        {
                            locals.insert(name);
                        }
                    } else {
                        collect_bare_idents(after_set, &mut referenced_before_set);
                    }
                } else {
                    collect_bare_idents(stmt, &mut referenced_before_set);
                }
            }
            pos = tag_start + 2 + rel_close + 2;
        } else {
            pos = tag_start + 1;
        }
    }
    locals
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

    fn get_undeclared_variables(&self) -> ffi::UndeclaredVariablesResult {
        if !self.creation_error.is_empty() {
            return ffi::UndeclaredVariablesResult {
                variables: Vec::new(),
                error: format!("Template creation failed: {}", self.creation_error),
                is_ok: false,
            };
        }
        let mut env = match create_env() {
            Ok(env) => env,
            Err(e) => {
                return ffi::UndeclaredVariablesResult {
                    variables: Vec::new(),
                    error: e.to_string(),
                    is_ok: false,
                };
            }
        };
        if let Err(e) = env.add_template("template", &self.source) {
            return ffi::UndeclaredVariablesResult {
                variables: Vec::new(),
                error: e.to_string(),
                is_ok: false,
            };
        }
        let tmpl = match env.get_template("template") {
            Ok(tmpl) => tmpl,
            Err(e) => {
                return ffi::UndeclaredVariablesResult {
                    variables: Vec::new(),
                    error: e.to_string(),
                    is_ok: false,
                };
            }
        };
        let locals = collect_template_local_definitions(&self.source);
        let mut vars: Vec<String> = tmpl
            .undeclared_variables(false)
            .into_iter()
            .filter(|name| {
                !env.globals().any(|(g, _)| g == name.as_str()) && !locals.contains(name.as_str())
            })
            .collect();
        vars.sort();
        ffi::UndeclaredVariablesResult { variables: vars, error: String::new(), is_ok: true }
    }

    fn clone_template(&self) -> Box<MinijinjaTemplate> {
        Box::new(self.clone())
    }

    #[allow(unexpected_cfgs)]
    fn apply_impl(&self, inputs_json: String) -> Result<String, Box<dyn std::error::Error>> {
        if !self.creation_error.is_empty() {
            return Err(format!("Template creation failed: {}", self.creation_error).into());
        }

        let inputs: ChatTemplateInputs = serde_json::from_str(&inputs_json)?;
        let actual_messages = match inputs.messages {
            Value::Array(arr) => arr,
            _ => vec![],
        };

        let mut env = create_env()?;
        env.add_template("template", &self.source)?;
        let tmpl = env.get_template("template")?;

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

        #[cfg(minijinja_v2)]
        let res = tmpl.render(Value::Object(ctx))?;
        #[cfg(not(minijinja_v2))]
        let res = {
            use minijinja::value::Serde;
            tmpl.render(Serde(Value::Object(ctx)))?
        };
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

    #[test]
    fn test_is_none_null() {
        let source = "{% if null_val is none %}null is none{% else %}FAIL{% endif %},{% if null_val is not none %}FAIL{% else %}not none is false{% endif %}";
        let wrapper = new_minijinja_template(source.to_string());

        let inputs = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {"null_val": null}
        });
        let res = wrapper.apply(inputs.to_string());
        assert!(res.is_ok);
        assert_eq!(res.content, "null is none,not none is false");
    }

    #[test]
    fn test_raise_exception() {
        let source = "{{ raise_exception('Something went wrong') }}";
        let wrapper = new_minijinja_template(source.to_string());

        let inputs = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {}
        });
        let res = wrapper.apply(inputs.to_string());
        assert!(!res.is_ok);
        assert!(res.error.contains("Something went wrong"));
    }

    #[test]
    fn test_lstrip_rstrip() {
        let source = "  foo  |{{ '  bar  '|lstrip }}|{{ '  baz  '|rstrip }}|{{ '1212foo12'|lstrip('12') }}|{{ '12foo1212'|rstrip('12') }}";
        let wrapper = new_minijinja_template(source.to_string());
        let inputs = json!({
            "messages": [],
            "tools": null,
            "add_generation_prompt": false,
            "extra_context": {}
        });
        let res = wrapper.apply(inputs.to_string());
        assert!(res.is_ok);
        assert_eq!(res.content, "  foo  |bar  |  baz|foo12|12foo");
    }

    #[test]
    fn test_undeclared_variables() {
        let source = "{%- macro outer(x) -%}{{ inner(x) }}{%- endmacro -%}\
            {%- macro inner(x) -%}{{ x }}{%- endmacro -%}\
            {% set ns = namespace(x=None, flag=True) %}\
            {% set ns.x = 1 %}\
            {% if messages %}{% set local_var = 'a' %}{% else %}{% set local_var = 'b' %}{% endif %}\
            {{ local_var }}\
            {% set resolved = extra_input | default(false) %}\
            {% for m in messages %}{{ outer(m.content) }}{% endfor %}\
            {% if enable_thinking | default(False) %}{{ reasoning_effort }}{% endif %}\
            {{ strftime_now('%Y') }}\
            {% if false %}{{ raise_exception('err') }}{{ custom_var }}{% endif %}";
        let wrapper = new_minijinja_template(source.to_string());
        let res = wrapper.get_undeclared_variables();
        assert!(res.is_ok);
        assert_eq!(
            res.variables,
            vec!["custom_var", "enable_thinking", "extra_input", "messages", "reasoning_effort"]
        );
    }

    #[test]
    fn test_undeclared_variables_edge_cases() {
        // 1. `is not defined` default guard must NOT suppress `custom_flag`.
        // 2. Macro-local `set` must NOT suppress top-level `shared_name`.
        // 3. `role = message['role']` and `content = msg.content` inside `if` must be recognized as
        //    local (not false-positive).
        let source = "{#- comment with {{ ignored_in_comment }} -#}\
            {% macro helper() %}{% set shared_name = 1 %}{{ shared_name }}{% endmacro %}\
            {{ shared_name }}\
            {% if custom_flag is not defined %}{% set custom_flag = false %}{% endif %}\
            {{ custom_flag }}\
            {% for message in messages %}\
                {% if message['role'] == 'user' %}\
                    {% set role = message['role'] %}\
                    {% set content = message.content %}\
                {% else %}\
                    {% set role = 'other' %}\
                    {% set content = '' %}\
                {% endif %}\
                {{ role }}{{ content }}\
            {% endfor %}";
        let wrapper = new_minijinja_template(source.to_string());
        let res = wrapper.get_undeclared_variables();
        assert!(res.is_ok);
        assert_eq!(res.variables, vec!["custom_flag", "messages", "shared_name"]);

        // 4. Self-referential `set` must not be added to template locals.
        let locals = collect_template_local_definitions(
            "{% set self_ref = self_ref | default(false) %}{% set ok_local = 'self_ref' %}",
        );
        assert!(!locals.contains("self_ref"));
        assert!(locals.contains("ok_local"));
    }

    #[test]
    fn test_undeclared_variables_invalid_syntax() {
        let wrapper = new_minijinja_template("{% if messages %}".to_string());
        let res = wrapper.get_undeclared_variables();
        assert!(!res.is_ok);
        assert!(!res.error.is_empty());
    }
}
