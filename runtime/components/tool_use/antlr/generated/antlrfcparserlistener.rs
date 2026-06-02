#![allow(nonstandard_style)]
// Generated from AntlrFcParser.g4 by ANTLR 4.13.2

// Copyright 2026 The ODML Authors.
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

use super::antlrfcparser::*;
use antlr4rust::tree::ParseTreeListener;

pub trait AntlrFcParserListener<'input>:
    ParseTreeListener<'input, AntlrFcParserContextType>
{
    /**
     * Enter a parse tree produced by {@link AntlrFcParser#start}.
     * @param ctx the parse tree
     */
    fn enter_start(&mut self, _ctx: &StartContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link AntlrFcParser#start}.
     * @param ctx the parse tree
     */
    fn exit_start(&mut self, _ctx: &StartContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link AntlrFcParser#functionCall}.
     * @param ctx the parse tree
     */
    fn enter_functionCall(&mut self, _ctx: &FunctionCallContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link AntlrFcParser#functionCall}.
     * @param ctx the parse tree
     */
    fn exit_functionCall(&mut self, _ctx: &FunctionCallContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link AntlrFcParser#object}.
     * @param ctx the parse tree
     */
    fn enter_object(&mut self, _ctx: &ObjectContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link AntlrFcParser#object}.
     * @param ctx the parse tree
     */
    fn exit_object(&mut self, _ctx: &ObjectContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link AntlrFcParser#pair}.
     * @param ctx the parse tree
     */
    fn enter_pair(&mut self, _ctx: &PairContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link AntlrFcParser#pair}.
     * @param ctx the parse tree
     */
    fn exit_pair(&mut self, _ctx: &PairContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link AntlrFcParser#value}.
     * @param ctx the parse tree
     */
    fn enter_value(&mut self, _ctx: &ValueContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link AntlrFcParser#value}.
     * @param ctx the parse tree
     */
    fn exit_value(&mut self, _ctx: &ValueContext<'input>) {}
    /**
     * Enter a parse tree produced by {@link AntlrFcParser#array}.
     * @param ctx the parse tree
     */
    fn enter_array(&mut self, _ctx: &ArrayContext<'input>) {}
    /**
     * Exit a parse tree produced by {@link AntlrFcParser#array}.
     * @param ctx the parse tree
     */
    fn exit_array(&mut self, _ctx: &ArrayContext<'input>) {}
}

antlr4rust::coerce_from! { 'input : AntlrFcParserListener<'input> }
