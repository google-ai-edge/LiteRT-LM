// Generated from AntlrPythonParser.g4 by ANTLR 4.13.2

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

use super::antlrpythonparser::*;
use antlr4rust::tree::ParseTreeListener;

// A complete Visitor for a parse tree produced by AntlrPythonParser.

pub trait AntlrPythonParserBaseListener<'input>:
    ParseTreeListener<'input, AntlrPythonParserContextType>
{
    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_main(&mut self, _ctx: &MainContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_main(&mut self, _ctx: &MainContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_expr(&mut self, _ctx: &ExprContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_expr(&mut self, _ctx: &ExprContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_key(&mut self, _ctx: &KeyContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_key(&mut self, _ctx: &KeyContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_value(&mut self, _ctx: &ValueContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_value(&mut self, _ctx: &ValueContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_list(&mut self, _ctx: &ListContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_list(&mut self, _ctx: &ListContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_dict(&mut self, _ctx: &DictContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_dict(&mut self, _ctx: &DictContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_argval(&mut self, _ctx: &ArgValContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_argval(&mut self, _ctx: &ArgValContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_argvalexpr(&mut self, _ctx: &ArgValExprContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_argvalexpr(&mut self, _ctx: &ArgValExprContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_object(&mut self, _ctx: &ObjectContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_object(&mut self, _ctx: &ObjectContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_emptyfunctioncall(&mut self, _ctx: &EmptyFunctionCallContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_emptyfunctioncall(&mut self, _ctx: &EmptyFunctionCallContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_fullfunctioncall(&mut self, _ctx: &FullFunctionCallContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_fullfunctioncall(&mut self, _ctx: &FullFunctionCallContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_functioncall(&mut self, _ctx: &FunctionCallContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_functioncall(&mut self, _ctx: &FunctionCallContext<'input>) {}

    /**
     * Enter a parse tree produced by \{@link AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn enter_functioncalllist(&mut self, _ctx: &FunctionCallListContext<'input>) {}
    /**
     * Exit a parse tree produced by \{@link  AntlrPythonParserBaseParser#s}.
     * @param ctx the parse tree
     */
    fn exit_functioncalllist(&mut self, _ctx: &FunctionCallListContext<'input>) {}
}
