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

use antlr4rust::common_token_stream::CommonTokenStream;
use antlr4rust::error_strategy::BailErrorStrategy;
use antlr4rust::tree::{ParseTree, ParseTreeListener};
use antlr4rust::InputStream;
use antlr_fc_tool_call_parser::{antlrfclexer, antlrfcparser, antlrfcparserlistener};
use antlrfclexer::AntlrFcLexer;
use antlrfcparser::{
    AntlrFcParser, AntlrFcParserContextType, AntlrFcParserTreeWalker, ArrayContext,
    ArrayContextAttrs, FunctionCallContext, FunctionCallContextAttrs, ObjectContext,
    ObjectContextAttrs, PairContextAttrs, ValueContext, ValueContextAttrs,
};
use antlrfcparserlistener::AntlrFcParserListener;
use serde_json::{json, Map, Value};
use std::collections::HashSet;

fn strip_escape_tokens(text: &str) -> &str {
    let mut s = text;

    const ESCAPE: &str = "<escape>";
    if s.starts_with(ESCAPE) {
        s = &s[ESCAPE.len()..];
    }
    if s.ends_with(ESCAPE) {
        s = &s[..(s.len() - ESCAPE.len())];
    }
    s
}

fn parse_value(value_ctx: &ValueContext) -> Result<Value, String> {
    if let Some(escaped_string_ctx) = value_ctx.ESCAPED_STRING() {
        Ok(json!(strip_escape_tokens(&escaped_string_ctx.get_text())))
    } else if let Some(number_ctx) = value_ctx.NUMBER() {
        let text = number_ctx.get_text();
        if let Ok(double_val) = text.parse::<f64>() {
            Ok(json!(double_val))
        } else {
            Err(format!("Failed to parse number: {}", text))
        }
    } else if let Some(object_ctx) = value_ctx.object() {
        parse_object(&object_ctx)
    } else if let Some(array_ctx) = value_ctx.array() {
        parse_array(&array_ctx)
    } else if let Some(boolean_ctx) = value_ctx.BOOLEAN() {
        Ok(json!(boolean_ctx.get_text() == "true"))
    } else if let Some(_null_literal_ctx) = value_ctx.NULL_LITERAL() {
        Ok(Value::Null)
    } else {
        Err(format!("Unhandled value type: {}", value_ctx.get_text()))
    }
}

fn parse_array(array_ctx: &ArrayContext) -> Result<Value, String> {
    let mut list = vec![];
    for value in array_ctx.value_all() {
        let parsed_value = parse_value(&value)?;
        list.push(parsed_value);
    }
    Ok(Value::Array(list))
}

fn parse_object(object_ctx: &ObjectContext) -> Result<Value, String> {
    let mut map = Map::new();
    let mut seen_keys = HashSet::new();

    for pair_ctx in object_ctx.pair_all() {
        let id_token =
            pair_ctx.ID().ok_or_else(|| "Invalid pair in object: ID missing".to_string())?;
        let value_ctx =
            pair_ctx.value().ok_or_else(|| "Invalid pair in object: Value missing".to_string())?;

        let key = id_token.get_text();
        if key.is_empty() {
            return Err("Object key is empty".to_string());
        }

        if seen_keys.contains(&key) {
            // Log duplicate key but don't treat it as an error.
            eprintln!("Ignoring duplicate key: {}", key);
            continue;
        }
        seen_keys.insert(key.clone());

        let parsed_value = parse_value(&value_ctx)
            .map_err(|e| format!("Error parsing value for key '{}': {}", key, e))?;

        map.insert(key, parsed_value);
    }
    Ok(Value::Object(map))
}

struct FcListener {
    tool_calls: Result<Vec<Value>, String>,
}

impl FcListener {
    fn new() -> Self {
        FcListener { tool_calls: Ok(Vec::new()) }
    }

    fn tool_calls(self) -> Result<Vec<Value>, String> {
        self.tool_calls
    }
}

impl<'input> ParseTreeListener<'input, AntlrFcParserContextType> for FcListener {}

impl<'input> AntlrFcParserListener<'input> for FcListener {
    fn enter_functionCall(&mut self, ctx: &FunctionCallContext<'input>) {
        if let Ok(tool_calls) = &mut self.tool_calls {
            let name =
                if let Some(id_token) = ctx.ID() { id_token.get_text() } else { "".to_string() };

            let mut tool_call_map = Map::new();
            tool_call_map.insert("name".to_string(), json!(name));

            if let Some(object_ctx) = ctx.object() {
                match parse_object(&object_ctx) {
                    Ok(args) => {
                        tool_call_map.insert("arguments".to_string(), args);
                    }
                    Err(e) => {
                        self.tool_calls = Err(e);
                        return;
                    }
                }
            } else {
                tool_call_map.insert("arguments".to_string(), json!({}));
            }

            tool_calls.push(Value::Object(tool_call_map));
        }
    }
}

pub fn parse_fc_expression(text: &str) -> Result<Vec<Value>, String> {
    if text.len() == 0 {
        return Ok(Vec::new());
    }
    let lexer = AntlrFcLexer::new(InputStream::new(text));
    let mut parser = AntlrFcParser::with_strategy(
        CommonTokenStream::new(lexer),
        Box::new(BailErrorStrategy::new()),
    );
    let start = match parser.start() {
        Ok(start) => start,
        Err(e) => return Err(e.to_string()),
    };
    match AntlrFcParserTreeWalker::walk(Box::new(FcListener::new()), start.as_ref()) {
        Ok(listener) => listener.tool_calls(),
        Err(e) => Err(e.to_string()),
    }
}
