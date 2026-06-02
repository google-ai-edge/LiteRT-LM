// Generated from AntlrPythonParser.g4 by ANTLR 4.13.2
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(nonstandard_style)]
#![allow(unused_imports)]
#![allow(unused_mut)]
#![allow(unused_braces)]

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

use super::antlrpythonparserlistener::*;
use antlr4rust::atn::{ATN, INVALID_ALT};
use antlr4rust::atn_deserializer::ATNDeserializer;
use antlr4rust::dfa::DFA;
use antlr4rust::error_strategy::{DefaultErrorStrategy, ErrorStrategy};
use antlr4rust::errors::*;
use antlr4rust::int_stream::EOF;
use antlr4rust::lazy_static;
use antlr4rust::parser::{BaseParser, Parser, ParserNodeType, ParserRecog};
use antlr4rust::parser_atn_simulator::ParserATNSimulator;
use antlr4rust::parser_rule_context::{cast, cast_mut, BaseParserRuleContext, ParserRuleContext};
use antlr4rust::recognizer::{Actions, Recognizer};
use antlr4rust::rule_context::{BaseRuleContext, CustomRuleContext, RuleContext};
use antlr4rust::token::{OwningToken, Token, TOKEN_EOF};
use antlr4rust::token_factory::{CommonTokenFactory, TokenAware, TokenFactory};
use antlr4rust::token_stream::TokenStream;
use antlr4rust::tree::*;
use antlr4rust::vocabulary::{Vocabulary, VocabularyImpl};
use antlr4rust::PredictionContextCache;
use antlr4rust::TokenSource;
use antlr4rust::{TidAble, TidExt};

use std::any::{Any, TypeId};
use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

pub const AntlrPythonParser_EQ: i32 = 1;
pub const AntlrPythonParser_COLON: i32 = 2;
pub const AntlrPythonParser_SEP: i32 = 3;
pub const AntlrPythonParser_OPEN_PAR: i32 = 4;
pub const AntlrPythonParser_CLOSE_PAR: i32 = 5;
pub const AntlrPythonParser_OPEN_BRACE: i32 = 6;
pub const AntlrPythonParser_CLOSE_BRACE: i32 = 7;
pub const AntlrPythonParser_LIST_OPEN: i32 = 8;
pub const AntlrPythonParser_LIST_CLOSE: i32 = 9;
pub const AntlrPythonParser_BOOL: i32 = 10;
pub const AntlrPythonParser_INT: i32 = 11;
pub const AntlrPythonParser_FLOAT: i32 = 12;
pub const AntlrPythonParser_STRING: i32 = 13;
pub const AntlrPythonParser_NONE: i32 = 14;
pub const AntlrPythonParser_NAME: i32 = 15;
pub const AntlrPythonParser_WS: i32 = 16;
pub const AntlrPythonParser_EOF: i32 = EOF;
pub const RULE_main: usize = 0;
pub const RULE_expr: usize = 1;
pub const RULE_key: usize = 2;
pub const RULE_value: usize = 3;
pub const RULE_list: usize = 4;
pub const RULE_dict: usize = 5;
pub const RULE_argVal: usize = 6;
pub const RULE_argValExpr: usize = 7;
pub const RULE_object: usize = 8;
pub const RULE_emptyFunctionCall: usize = 9;
pub const RULE_fullFunctionCall: usize = 10;
pub const RULE_functionCall: usize = 11;
pub const RULE_functionCallList: usize = 12;
pub const ruleNames: [&'static str; 13] = [
    "main",
    "expr",
    "key",
    "value",
    "list",
    "dict",
    "argVal",
    "argValExpr",
    "object",
    "emptyFunctionCall",
    "fullFunctionCall",
    "functionCall",
    "functionCallList",
];

pub const _LITERAL_NAMES: [Option<&'static str>; 15] = [
    None,
    Some("'='"),
    Some("':'"),
    Some("','"),
    Some("'('"),
    Some("')'"),
    Some("'{'"),
    Some("'}'"),
    Some("'['"),
    Some("']'"),
    None,
    None,
    None,
    None,
    Some("'None'"),
];
pub const _SYMBOLIC_NAMES: [Option<&'static str>; 17] = [
    None,
    Some("EQ"),
    Some("COLON"),
    Some("SEP"),
    Some("OPEN_PAR"),
    Some("CLOSE_PAR"),
    Some("OPEN_BRACE"),
    Some("CLOSE_BRACE"),
    Some("LIST_OPEN"),
    Some("LIST_CLOSE"),
    Some("BOOL"),
    Some("INT"),
    Some("FLOAT"),
    Some("STRING"),
    Some("NONE"),
    Some("NAME"),
    Some("WS"),
];
lazy_static! {
    static ref _shared_context_cache: Arc<PredictionContextCache> =
        Arc::new(PredictionContextCache::new());
    static ref VOCABULARY: Box<dyn Vocabulary> =
        Box::new(VocabularyImpl::new(_LITERAL_NAMES.iter(), _SYMBOLIC_NAMES.iter(), None));
}

type BaseParserType<'input, I> = BaseParser<
    'input,
    AntlrPythonParserExt<'input>,
    I,
    AntlrPythonParserContextType,
    dyn AntlrPythonParserListener<'input> + 'input,
>;

type TokenType<'input> = <LocalTokenFactory<'input> as TokenFactory<'input>>::Tok;
pub type LocalTokenFactory<'input> = CommonTokenFactory;

pub type AntlrPythonParserTreeWalker<'input, 'a> = ParseTreeWalker<
    'input,
    'a,
    AntlrPythonParserContextType,
    dyn AntlrPythonParserListener<'input> + 'a,
>;

/// Parser for AntlrPythonParser grammar
pub struct AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    base: BaseParserType<'input, I>,
    interpreter: Arc<ParserATNSimulator>,
    _shared_context_cache: Box<PredictionContextCache>,
    pub err_handler: Box<dyn ErrorStrategy<'input, BaseParserType<'input, I>>>,
}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn set_error_strategy(
        &mut self,
        strategy: Box<dyn ErrorStrategy<'input, BaseParserType<'input, I>>>,
    ) {
        self.err_handler = strategy
    }

    pub fn with_strategy(
        input: I,
        strategy: Box<dyn ErrorStrategy<'input, BaseParserType<'input, I>>>,
    ) -> Self {
        antlr4rust::recognizer::check_version("0", "5");
        let interpreter = Arc::new(ParserATNSimulator::new(
            _ATN.clone(),
            _decision_to_DFA.clone(),
            _shared_context_cache.clone(),
        ));
        Self {
            base: BaseParser::new_base_parser(
                input,
                Arc::clone(&interpreter),
                AntlrPythonParserExt { _pd: Default::default() },
            ),
            interpreter,
            _shared_context_cache: Box::new(PredictionContextCache::new()),
            err_handler: strategy,
        }
    }
}

type DynStrategy<'input, I> = Box<dyn ErrorStrategy<'input, BaseParserType<'input, I>> + 'input>;

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn with_dyn_strategy(input: I) -> Self {
        Self::with_strategy(input, Box::new(DefaultErrorStrategy::new()))
    }
}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn new(input: I) -> Self {
        Self::with_strategy(input, Box::new(DefaultErrorStrategy::new()))
    }
}

/// Trait for monomorphized trait object that corresponds to the nodes of parse tree generated for AntlrPythonParser
pub trait AntlrPythonParserContext<'input>:
    for<'x> Listenable<dyn AntlrPythonParserListener<'input> + 'x>
    + ParserRuleContext<'input, TF = LocalTokenFactory<'input>, Ctx = AntlrPythonParserContextType>
{
}

antlr4rust::coerce_from! { 'input : AntlrPythonParserContext<'input> }

impl<'input> AntlrPythonParserContext<'input>
    for TerminalNode<'input, AntlrPythonParserContextType>
{
}
impl<'input> AntlrPythonParserContext<'input> for ErrorNode<'input, AntlrPythonParserContextType> {}

antlr4rust::tid! { impl<'input> TidAble<'input> for dyn AntlrPythonParserContext<'input> + 'input }

antlr4rust::tid! { impl<'input> TidAble<'input> for dyn AntlrPythonParserListener<'input> + 'input }

pub struct AntlrPythonParserContextType;
antlr4rust::tid! {AntlrPythonParserContextType}

impl<'input> ParserNodeType<'input> for AntlrPythonParserContextType {
    type TF = LocalTokenFactory<'input>;
    type Type = dyn AntlrPythonParserContext<'input> + 'input;
}

impl<'input, I> Deref for AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    type Target = BaseParserType<'input, I>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'input, I> DerefMut for AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

pub struct AntlrPythonParserExt<'input> {
    _pd: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserExt<'input> {}
antlr4rust::tid! { AntlrPythonParserExt<'a> }

impl<'input> TokenAware<'input> for AntlrPythonParserExt<'input> {
    type TF = LocalTokenFactory<'input>;
}

impl<'input, I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>>
    ParserRecog<'input, BaseParserType<'input, I>> for AntlrPythonParserExt<'input>
{
}

impl<'input, I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>>
    Actions<'input, BaseParserType<'input, I>> for AntlrPythonParserExt<'input>
{
    fn get_grammar_file_name(&self) -> &str {
        "AntlrPythonParser.g4"
    }

    fn get_rule_names(&self) -> &[&str] {
        &ruleNames
    }

    fn get_vocabulary(&self) -> &dyn Vocabulary {
        &**VOCABULARY
    }
}
//------------------- main ----------------
pub type MainContextAll<'input> = MainContext<'input>;

pub type MainContext<'input> = BaseParserRuleContext<'input, MainContextExt<'input>>;

#[derive(Clone)]
pub struct MainContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for MainContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for MainContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_main(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_main(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for MainContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_main
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_main }
}
antlr4rust::tid! {MainContextExt<'a>}

impl<'input> MainContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<MainContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            MainContextExt { ph: PhantomData },
        ))
    }
}

pub trait MainContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<MainContextExt<'input>>
{
    fn expr(&self) -> Option<Rc<ExprContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
    /// Retrieves first TerminalNode corresponding to token EOF
    /// Returns `None` if there is no child corresponding to token EOF
    fn EOF(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_EOF, 0)
    }
}

impl<'input> MainContextAttrs<'input> for MainContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn main(&mut self) -> Result<Rc<MainContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = MainContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 0, RULE_main);
        let mut _localctx: Rc<MainContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                /*InvokeRule expr*/
                recog.base.set_state(26);
                recog.expr()?;

                recog.base.set_state(27);
                recog.base.match_token(AntlrPythonParser_EOF, &mut recog.err_handler)?;
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- expr ----------------
pub type ExprContextAll<'input> = ExprContext<'input>;

pub type ExprContext<'input> = BaseParserRuleContext<'input, ExprContextExt<'input>>;

#[derive(Clone)]
pub struct ExprContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for ExprContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for ExprContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_expr(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_expr(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ExprContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_expr
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_expr }
}
antlr4rust::tid! {ExprContextExt<'a>}

impl<'input> ExprContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<ExprContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            ExprContextExt { ph: PhantomData },
        ))
    }
}

pub trait ExprContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<ExprContextExt<'input>>
{
    fn functionCall_all(&self) -> Vec<Rc<FunctionCallContextAll<'input>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    fn functionCall(&self, i: usize) -> Option<Rc<FunctionCallContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(i)
    }
    fn functionCallList(&self) -> Option<Rc<FunctionCallListContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> ExprContextAttrs<'input> for ExprContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn expr(&mut self) -> Result<Rc<ExprContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ExprContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 2, RULE_expr);
        let mut _localctx: Rc<ExprContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            recog.base.set_state(35);
            recog.err_handler.sync(&mut recog.base)?;
            match recog.base.input.la(1) {
                AntlrPythonParser_NAME => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
                    recog.base.enter_outer_alt(None, 1)?;
                    {
                        recog.base.set_state(30);
                        recog.err_handler.sync(&mut recog.base)?;
                        _la = recog.base.input.la(1);
                        loop {
                            {
                                {
                                    /*InvokeRule functionCall*/
                                    recog.base.set_state(29);
                                    recog.functionCall()?;
                                }
                            }
                            recog.base.set_state(32);
                            recog.err_handler.sync(&mut recog.base)?;
                            _la = recog.base.input.la(1);
                            if !(_la == AntlrPythonParser_NAME) {
                                break;
                            }
                        }
                    }
                }

                AntlrPythonParser_LIST_OPEN => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 2)?;
                    recog.base.enter_outer_alt(None, 2)?;
                    {
                        /*InvokeRule functionCallList*/
                        recog.base.set_state(34);
                        recog.functionCallList()?;
                    }
                }

                _ => Err(ANTLRError::NoAltError(NoViableAltError::new(&mut recog.base)))?,
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- key ----------------
pub type KeyContextAll<'input> = KeyContext<'input>;

pub type KeyContext<'input> = BaseParserRuleContext<'input, KeyContextExt<'input>>;

#[derive(Clone)]
pub struct KeyContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for KeyContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for KeyContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_key(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_key(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for KeyContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_key
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_key }
}
antlr4rust::tid! {KeyContextExt<'a>}

impl<'input> KeyContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<KeyContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            KeyContextExt { ph: PhantomData },
        ))
    }
}

pub trait KeyContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<KeyContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token STRING
    /// Returns `None` if there is no child corresponding to token STRING
    fn STRING(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_STRING, 0)
    }
}

impl<'input> KeyContextAttrs<'input> for KeyContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn key(&mut self) -> Result<Rc<KeyContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = KeyContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 4, RULE_key);
        let mut _localctx: Rc<KeyContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(37);
                recog.base.match_token(AntlrPythonParser_STRING, &mut recog.err_handler)?;
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- value ----------------
pub type ValueContextAll<'input> = ValueContext<'input>;

pub type ValueContext<'input> = BaseParserRuleContext<'input, ValueContextExt<'input>>;

#[derive(Clone)]
pub struct ValueContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for ValueContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for ValueContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_value(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_value(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ValueContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_value
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_value }
}
antlr4rust::tid! {ValueContextExt<'a>}

impl<'input> ValueContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<ValueContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            ValueContextExt { ph: PhantomData },
        ))
    }
}

pub trait ValueContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<ValueContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token INT
    /// Returns `None` if there is no child corresponding to token INT
    fn INT(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_INT, 0)
    }
    /// Retrieves first TerminalNode corresponding to token FLOAT
    /// Returns `None` if there is no child corresponding to token FLOAT
    fn FLOAT(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_FLOAT, 0)
    }
    /// Retrieves first TerminalNode corresponding to token BOOL
    /// Returns `None` if there is no child corresponding to token BOOL
    fn BOOL(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_BOOL, 0)
    }
    /// Retrieves first TerminalNode corresponding to token STRING
    /// Returns `None` if there is no child corresponding to token STRING
    fn STRING(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_STRING, 0)
    }
    /// Retrieves first TerminalNode corresponding to token NONE
    /// Returns `None` if there is no child corresponding to token NONE
    fn NONE(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_NONE, 0)
    }
    fn list(&self) -> Option<Rc<ListContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
    fn dict(&self) -> Option<Rc<DictContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
    fn object(&self) -> Option<Rc<ObjectContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> ValueContextAttrs<'input> for ValueContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn value(&mut self) -> Result<Rc<ValueContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ValueContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 6, RULE_value);
        let mut _localctx: Rc<ValueContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            recog.base.set_state(47);
            recog.err_handler.sync(&mut recog.base)?;
            match recog.base.input.la(1) {
                AntlrPythonParser_INT => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
                    recog.base.enter_outer_alt(None, 1)?;
                    {
                        recog.base.set_state(39);
                        recog.base.match_token(AntlrPythonParser_INT, &mut recog.err_handler)?;
                    }
                }

                AntlrPythonParser_FLOAT => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 2)?;
                    recog.base.enter_outer_alt(None, 2)?;
                    {
                        recog.base.set_state(40);
                        recog.base.match_token(AntlrPythonParser_FLOAT, &mut recog.err_handler)?;
                    }
                }

                AntlrPythonParser_BOOL => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 3)?;
                    recog.base.enter_outer_alt(None, 3)?;
                    {
                        recog.base.set_state(41);
                        recog.base.match_token(AntlrPythonParser_BOOL, &mut recog.err_handler)?;
                    }
                }

                AntlrPythonParser_STRING => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 4)?;
                    recog.base.enter_outer_alt(None, 4)?;
                    {
                        recog.base.set_state(42);
                        recog.base.match_token(AntlrPythonParser_STRING, &mut recog.err_handler)?;
                    }
                }

                AntlrPythonParser_NONE => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 5)?;
                    recog.base.enter_outer_alt(None, 5)?;
                    {
                        recog.base.set_state(43);
                        recog.base.match_token(AntlrPythonParser_NONE, &mut recog.err_handler)?;
                    }
                }

                AntlrPythonParser_LIST_OPEN => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 6)?;
                    recog.base.enter_outer_alt(None, 6)?;
                    {
                        /*InvokeRule list*/
                        recog.base.set_state(44);
                        recog.list()?;
                    }
                }

                AntlrPythonParser_OPEN_BRACE => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 7)?;
                    recog.base.enter_outer_alt(None, 7)?;
                    {
                        /*InvokeRule dict*/
                        recog.base.set_state(45);
                        recog.dict()?;
                    }
                }

                AntlrPythonParser_NAME => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 8)?;
                    recog.base.enter_outer_alt(None, 8)?;
                    {
                        /*InvokeRule object*/
                        recog.base.set_state(46);
                        recog.object()?;
                    }
                }

                _ => Err(ANTLRError::NoAltError(NoViableAltError::new(&mut recog.base)))?,
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- list ----------------
pub type ListContextAll<'input> = ListContext<'input>;

pub type ListContext<'input> = BaseParserRuleContext<'input, ListContextExt<'input>>;

#[derive(Clone)]
pub struct ListContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for ListContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for ListContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_list(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_list(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ListContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_list
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_list }
}
antlr4rust::tid! {ListContextExt<'a>}

impl<'input> ListContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<ListContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            ListContextExt { ph: PhantomData },
        ))
    }
}

pub trait ListContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<ListContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token LIST_OPEN
    /// Returns `None` if there is no child corresponding to token LIST_OPEN
    fn LIST_OPEN(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_LIST_OPEN, 0)
    }
    /// Retrieves first TerminalNode corresponding to token LIST_CLOSE
    /// Returns `None` if there is no child corresponding to token LIST_CLOSE
    fn LIST_CLOSE(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_LIST_CLOSE, 0)
    }
    fn value_all(&self) -> Vec<Rc<ValueContextAll<'input>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    fn value(&self, i: usize) -> Option<Rc<ValueContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(i)
    }
    /// Retrieves all `TerminalNode`s corresponding to token SEP in current rule
    fn SEP_all(&self) -> Vec<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    /// Retrieves 'i's TerminalNode corresponding to token SEP, starting from 0.
    /// Returns `None` if number of children corresponding to token SEP is less or equal than `i`.
    fn SEP(&self, i: usize) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_SEP, i)
    }
}

impl<'input> ListContextAttrs<'input> for ListContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn list(&mut self) -> Result<Rc<ListContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ListContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 8, RULE_list);
        let mut _localctx: Rc<ListContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            let mut _alt: i32;
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(49);
                recog.base.match_token(AntlrPythonParser_LIST_OPEN, &mut recog.err_handler)?;

                recog.base.set_state(61);
                recog.err_handler.sync(&mut recog.base)?;
                _la = recog.base.input.la(1);
                if (((_la) & !0x3f) == 0 && ((1usize << _la) & 64832) != 0) {
                    {
                        /*InvokeRule value*/
                        recog.base.set_state(50);
                        recog.value()?;

                        recog.base.set_state(55);
                        recog.err_handler.sync(&mut recog.base)?;
                        _alt = recog.interpreter.adaptive_predict(3, &mut recog.base)?;
                        while { _alt != 2 && _alt != INVALID_ALT } {
                            if _alt == 1 {
                                {
                                    {
                                        recog.base.set_state(51);
                                        recog.base.match_token(
                                            AntlrPythonParser_SEP,
                                            &mut recog.err_handler,
                                        )?;

                                        /*InvokeRule value*/
                                        recog.base.set_state(52);
                                        recog.value()?;
                                    }
                                }
                            }
                            recog.base.set_state(57);
                            recog.err_handler.sync(&mut recog.base)?;
                            _alt = recog.interpreter.adaptive_predict(3, &mut recog.base)?;
                        }
                        recog.base.set_state(59);
                        recog.err_handler.sync(&mut recog.base)?;
                        _la = recog.base.input.la(1);
                        if _la == AntlrPythonParser_SEP {
                            {
                                recog.base.set_state(58);
                                recog
                                    .base
                                    .match_token(AntlrPythonParser_SEP, &mut recog.err_handler)?;
                            }
                        }
                    }
                }

                recog.base.set_state(63);
                recog.base.match_token(AntlrPythonParser_LIST_CLOSE, &mut recog.err_handler)?;
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- dict ----------------
pub type DictContextAll<'input> = DictContext<'input>;

pub type DictContext<'input> = BaseParserRuleContext<'input, DictContextExt<'input>>;

#[derive(Clone)]
pub struct DictContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for DictContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for DictContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_dict(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_dict(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for DictContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_dict
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_dict }
}
antlr4rust::tid! {DictContextExt<'a>}

impl<'input> DictContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<DictContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            DictContextExt { ph: PhantomData },
        ))
    }
}

pub trait DictContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<DictContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token OPEN_BRACE
    /// Returns `None` if there is no child corresponding to token OPEN_BRACE
    fn OPEN_BRACE(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_OPEN_BRACE, 0)
    }
    /// Retrieves first TerminalNode corresponding to token CLOSE_BRACE
    /// Returns `None` if there is no child corresponding to token CLOSE_BRACE
    fn CLOSE_BRACE(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_CLOSE_BRACE, 0)
    }
    fn key_all(&self) -> Vec<Rc<KeyContextAll<'input>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    fn key(&self, i: usize) -> Option<Rc<KeyContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(i)
    }
    /// Retrieves all `TerminalNode`s corresponding to token COLON in current rule
    fn COLON_all(&self) -> Vec<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    /// Retrieves 'i's TerminalNode corresponding to token COLON, starting from 0.
    /// Returns `None` if number of children corresponding to token COLON is less or equal than `i`.
    fn COLON(&self, i: usize) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_COLON, i)
    }
    fn value_all(&self) -> Vec<Rc<ValueContextAll<'input>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    fn value(&self, i: usize) -> Option<Rc<ValueContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(i)
    }
    /// Retrieves all `TerminalNode`s corresponding to token SEP in current rule
    fn SEP_all(&self) -> Vec<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    /// Retrieves 'i's TerminalNode corresponding to token SEP, starting from 0.
    /// Returns `None` if number of children corresponding to token SEP is less or equal than `i`.
    fn SEP(&self, i: usize) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_SEP, i)
    }
}

impl<'input> DictContextAttrs<'input> for DictContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn dict(&mut self) -> Result<Rc<DictContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = DictContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 10, RULE_dict);
        let mut _localctx: Rc<DictContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            let mut _alt: i32;
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(65);
                recog.base.match_token(AntlrPythonParser_OPEN_BRACE, &mut recog.err_handler)?;

                recog.base.set_state(82);
                recog.err_handler.sync(&mut recog.base)?;
                _la = recog.base.input.la(1);
                if _la == AntlrPythonParser_STRING {
                    {
                        /*InvokeRule key*/
                        recog.base.set_state(66);
                        recog.key()?;

                        recog.base.set_state(67);
                        recog.base.match_token(AntlrPythonParser_COLON, &mut recog.err_handler)?;

                        /*InvokeRule value*/
                        recog.base.set_state(68);
                        recog.value()?;

                        recog.base.set_state(76);
                        recog.err_handler.sync(&mut recog.base)?;
                        _alt = recog.interpreter.adaptive_predict(6, &mut recog.base)?;
                        while { _alt != 2 && _alt != INVALID_ALT } {
                            if _alt == 1 {
                                {
                                    {
                                        recog.base.set_state(69);
                                        recog.base.match_token(
                                            AntlrPythonParser_SEP,
                                            &mut recog.err_handler,
                                        )?;

                                        /*InvokeRule key*/
                                        recog.base.set_state(70);
                                        recog.key()?;

                                        recog.base.set_state(71);
                                        recog.base.match_token(
                                            AntlrPythonParser_COLON,
                                            &mut recog.err_handler,
                                        )?;

                                        /*InvokeRule value*/
                                        recog.base.set_state(72);
                                        recog.value()?;
                                    }
                                }
                            }
                            recog.base.set_state(78);
                            recog.err_handler.sync(&mut recog.base)?;
                            _alt = recog.interpreter.adaptive_predict(6, &mut recog.base)?;
                        }
                        recog.base.set_state(80);
                        recog.err_handler.sync(&mut recog.base)?;
                        _la = recog.base.input.la(1);
                        if _la == AntlrPythonParser_SEP {
                            {
                                recog.base.set_state(79);
                                recog
                                    .base
                                    .match_token(AntlrPythonParser_SEP, &mut recog.err_handler)?;
                            }
                        }
                    }
                }

                recog.base.set_state(84);
                recog.base.match_token(AntlrPythonParser_CLOSE_BRACE, &mut recog.err_handler)?;
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- argVal ----------------
pub type ArgValContextAll<'input> = ArgValContext<'input>;

pub type ArgValContext<'input> = BaseParserRuleContext<'input, ArgValContextExt<'input>>;

#[derive(Clone)]
pub struct ArgValContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for ArgValContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for ArgValContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_argVal(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_argVal(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ArgValContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_argVal
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_argVal }
}
antlr4rust::tid! {ArgValContextExt<'a>}

impl<'input> ArgValContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<ArgValContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            ArgValContextExt { ph: PhantomData },
        ))
    }
}

pub trait ArgValContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<ArgValContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token NAME
    /// Returns `None` if there is no child corresponding to token NAME
    fn NAME(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_NAME, 0)
    }
    /// Retrieves first TerminalNode corresponding to token EQ
    /// Returns `None` if there is no child corresponding to token EQ
    fn EQ(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_EQ, 0)
    }
    fn value(&self) -> Option<Rc<ValueContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> ArgValContextAttrs<'input> for ArgValContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn argVal(&mut self) -> Result<Rc<ArgValContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ArgValContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 12, RULE_argVal);
        let mut _localctx: Rc<ArgValContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(86);
                recog.base.match_token(AntlrPythonParser_NAME, &mut recog.err_handler)?;

                recog.base.set_state(87);
                recog.base.match_token(AntlrPythonParser_EQ, &mut recog.err_handler)?;

                /*InvokeRule value*/
                recog.base.set_state(88);
                recog.value()?;
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- argValExpr ----------------
pub type ArgValExprContextAll<'input> = ArgValExprContext<'input>;

pub type ArgValExprContext<'input> = BaseParserRuleContext<'input, ArgValExprContextExt<'input>>;

#[derive(Clone)]
pub struct ArgValExprContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for ArgValExprContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a>
    for ArgValExprContext<'input>
{
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_argValExpr(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_argValExpr(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ArgValExprContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_argValExpr
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_argValExpr }
}
antlr4rust::tid! {ArgValExprContextExt<'a>}

impl<'input> ArgValExprContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<ArgValExprContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            ArgValExprContextExt { ph: PhantomData },
        ))
    }
}

pub trait ArgValExprContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<ArgValExprContextExt<'input>>
{
    fn argVal_all(&self) -> Vec<Rc<ArgValContextAll<'input>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    fn argVal(&self, i: usize) -> Option<Rc<ArgValContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(i)
    }
    /// Retrieves all `TerminalNode`s corresponding to token SEP in current rule
    fn SEP_all(&self) -> Vec<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    /// Retrieves 'i's TerminalNode corresponding to token SEP, starting from 0.
    /// Returns `None` if number of children corresponding to token SEP is less or equal than `i`.
    fn SEP(&self, i: usize) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_SEP, i)
    }
}

impl<'input> ArgValExprContextAttrs<'input> for ArgValExprContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn argValExpr(&mut self) -> Result<Rc<ArgValExprContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ArgValExprContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 14, RULE_argValExpr);
        let mut _localctx: Rc<ArgValExprContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            let mut _alt: i32;
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                /*InvokeRule argVal*/
                recog.base.set_state(90);
                recog.argVal()?;

                recog.base.set_state(95);
                recog.err_handler.sync(&mut recog.base)?;
                _alt = recog.interpreter.adaptive_predict(9, &mut recog.base)?;
                while { _alt != 2 && _alt != INVALID_ALT } {
                    if _alt == 1 {
                        {
                            {
                                recog.base.set_state(91);
                                recog
                                    .base
                                    .match_token(AntlrPythonParser_SEP, &mut recog.err_handler)?;

                                /*InvokeRule argVal*/
                                recog.base.set_state(92);
                                recog.argVal()?;
                            }
                        }
                    }
                    recog.base.set_state(97);
                    recog.err_handler.sync(&mut recog.base)?;
                    _alt = recog.interpreter.adaptive_predict(9, &mut recog.base)?;
                }
                recog.base.set_state(99);
                recog.err_handler.sync(&mut recog.base)?;
                _la = recog.base.input.la(1);
                if _la == AntlrPythonParser_SEP {
                    {
                        recog.base.set_state(98);
                        recog.base.match_token(AntlrPythonParser_SEP, &mut recog.err_handler)?;
                    }
                }
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- object ----------------
pub type ObjectContextAll<'input> = ObjectContext<'input>;

pub type ObjectContext<'input> = BaseParserRuleContext<'input, ObjectContextExt<'input>>;

#[derive(Clone)]
pub struct ObjectContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for ObjectContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a> for ObjectContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_object(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_object(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ObjectContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_object
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_object }
}
antlr4rust::tid! {ObjectContextExt<'a>}

impl<'input> ObjectContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<ObjectContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            ObjectContextExt { ph: PhantomData },
        ))
    }
}

pub trait ObjectContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<ObjectContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token NAME
    /// Returns `None` if there is no child corresponding to token NAME
    fn NAME(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_NAME, 0)
    }
    /// Retrieves first TerminalNode corresponding to token OPEN_PAR
    /// Returns `None` if there is no child corresponding to token OPEN_PAR
    fn OPEN_PAR(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_OPEN_PAR, 0)
    }
    /// Retrieves first TerminalNode corresponding to token CLOSE_PAR
    /// Returns `None` if there is no child corresponding to token CLOSE_PAR
    fn CLOSE_PAR(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_CLOSE_PAR, 0)
    }
    fn argValExpr(&self) -> Option<Rc<ArgValExprContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> ObjectContextAttrs<'input> for ObjectContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn object(&mut self) -> Result<Rc<ObjectContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ObjectContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 16, RULE_object);
        let mut _localctx: Rc<ObjectContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            recog.base.set_state(109);
            recog.err_handler.sync(&mut recog.base)?;
            match recog.interpreter.adaptive_predict(11, &mut recog.base)? {
                1 => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
                    recog.base.enter_outer_alt(None, 1)?;
                    {
                        recog.base.set_state(101);
                        recog.base.match_token(AntlrPythonParser_NAME, &mut recog.err_handler)?;

                        recog.base.set_state(102);
                        recog
                            .base
                            .match_token(AntlrPythonParser_OPEN_PAR, &mut recog.err_handler)?;

                        recog.base.set_state(103);
                        recog
                            .base
                            .match_token(AntlrPythonParser_CLOSE_PAR, &mut recog.err_handler)?;
                    }
                }
                2 => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 2)?;
                    recog.base.enter_outer_alt(None, 2)?;
                    {
                        recog.base.set_state(104);
                        recog.base.match_token(AntlrPythonParser_NAME, &mut recog.err_handler)?;

                        recog.base.set_state(105);
                        recog
                            .base
                            .match_token(AntlrPythonParser_OPEN_PAR, &mut recog.err_handler)?;

                        /*InvokeRule argValExpr*/
                        recog.base.set_state(106);
                        recog.argValExpr()?;

                        recog.base.set_state(107);
                        recog
                            .base
                            .match_token(AntlrPythonParser_CLOSE_PAR, &mut recog.err_handler)?;
                    }
                }

                _ => {}
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- emptyFunctionCall ----------------
pub type EmptyFunctionCallContextAll<'input> = EmptyFunctionCallContext<'input>;

pub type EmptyFunctionCallContext<'input> =
    BaseParserRuleContext<'input, EmptyFunctionCallContextExt<'input>>;

#[derive(Clone)]
pub struct EmptyFunctionCallContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for EmptyFunctionCallContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a>
    for EmptyFunctionCallContext<'input>
{
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_emptyFunctionCall(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_emptyFunctionCall(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for EmptyFunctionCallContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_emptyFunctionCall
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_emptyFunctionCall }
}
antlr4rust::tid! {EmptyFunctionCallContextExt<'a>}

impl<'input> EmptyFunctionCallContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<EmptyFunctionCallContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            EmptyFunctionCallContextExt { ph: PhantomData },
        ))
    }
}

pub trait EmptyFunctionCallContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<EmptyFunctionCallContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token NAME
    /// Returns `None` if there is no child corresponding to token NAME
    fn NAME(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_NAME, 0)
    }
    /// Retrieves first TerminalNode corresponding to token OPEN_PAR
    /// Returns `None` if there is no child corresponding to token OPEN_PAR
    fn OPEN_PAR(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_OPEN_PAR, 0)
    }
    /// Retrieves first TerminalNode corresponding to token CLOSE_PAR
    /// Returns `None` if there is no child corresponding to token CLOSE_PAR
    fn CLOSE_PAR(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_CLOSE_PAR, 0)
    }
}

impl<'input> EmptyFunctionCallContextAttrs<'input> for EmptyFunctionCallContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn emptyFunctionCall(
        &mut self,
    ) -> Result<Rc<EmptyFunctionCallContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx =
            EmptyFunctionCallContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 18, RULE_emptyFunctionCall);
        let mut _localctx: Rc<EmptyFunctionCallContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(111);
                recog.base.match_token(AntlrPythonParser_NAME, &mut recog.err_handler)?;

                recog.base.set_state(112);
                recog.base.match_token(AntlrPythonParser_OPEN_PAR, &mut recog.err_handler)?;

                recog.base.set_state(113);
                recog.base.match_token(AntlrPythonParser_CLOSE_PAR, &mut recog.err_handler)?;
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- fullFunctionCall ----------------
pub type FullFunctionCallContextAll<'input> = FullFunctionCallContext<'input>;

pub type FullFunctionCallContext<'input> =
    BaseParserRuleContext<'input, FullFunctionCallContextExt<'input>>;

#[derive(Clone)]
pub struct FullFunctionCallContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for FullFunctionCallContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a>
    for FullFunctionCallContext<'input>
{
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_fullFunctionCall(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_fullFunctionCall(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for FullFunctionCallContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_fullFunctionCall
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_fullFunctionCall }
}
antlr4rust::tid! {FullFunctionCallContextExt<'a>}

impl<'input> FullFunctionCallContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<FullFunctionCallContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            FullFunctionCallContextExt { ph: PhantomData },
        ))
    }
}

pub trait FullFunctionCallContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<FullFunctionCallContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token NAME
    /// Returns `None` if there is no child corresponding to token NAME
    fn NAME(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_NAME, 0)
    }
    /// Retrieves first TerminalNode corresponding to token OPEN_PAR
    /// Returns `None` if there is no child corresponding to token OPEN_PAR
    fn OPEN_PAR(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_OPEN_PAR, 0)
    }
    fn argValExpr(&self) -> Option<Rc<ArgValExprContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
    /// Retrieves first TerminalNode corresponding to token CLOSE_PAR
    /// Returns `None` if there is no child corresponding to token CLOSE_PAR
    fn CLOSE_PAR(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_CLOSE_PAR, 0)
    }
}

impl<'input> FullFunctionCallContextAttrs<'input> for FullFunctionCallContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn fullFunctionCall(
        &mut self,
    ) -> Result<Rc<FullFunctionCallContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx =
            FullFunctionCallContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 20, RULE_fullFunctionCall);
        let mut _localctx: Rc<FullFunctionCallContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(115);
                recog.base.match_token(AntlrPythonParser_NAME, &mut recog.err_handler)?;

                recog.base.set_state(116);
                recog.base.match_token(AntlrPythonParser_OPEN_PAR, &mut recog.err_handler)?;

                /*InvokeRule argValExpr*/
                recog.base.set_state(117);
                recog.argValExpr()?;

                recog.base.set_state(118);
                recog.base.match_token(AntlrPythonParser_CLOSE_PAR, &mut recog.err_handler)?;
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- functionCall ----------------
pub type FunctionCallContextAll<'input> = FunctionCallContext<'input>;

pub type FunctionCallContext<'input> =
    BaseParserRuleContext<'input, FunctionCallContextExt<'input>>;

#[derive(Clone)]
pub struct FunctionCallContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for FunctionCallContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a>
    for FunctionCallContext<'input>
{
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_functionCall(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_functionCall(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for FunctionCallContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_functionCall
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_functionCall }
}
antlr4rust::tid! {FunctionCallContextExt<'a>}

impl<'input> FunctionCallContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<FunctionCallContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            FunctionCallContextExt { ph: PhantomData },
        ))
    }
}

pub trait FunctionCallContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<FunctionCallContextExt<'input>>
{
    fn fullFunctionCall(&self) -> Option<Rc<FullFunctionCallContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
    fn emptyFunctionCall(&self) -> Option<Rc<EmptyFunctionCallContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> FunctionCallContextAttrs<'input> for FunctionCallContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn functionCall(&mut self) -> Result<Rc<FunctionCallContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = FunctionCallContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 22, RULE_functionCall);
        let mut _localctx: Rc<FunctionCallContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            recog.base.set_state(122);
            recog.err_handler.sync(&mut recog.base)?;
            match recog.interpreter.adaptive_predict(12, &mut recog.base)? {
                1 => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
                    recog.base.enter_outer_alt(None, 1)?;
                    {
                        /*InvokeRule fullFunctionCall*/
                        recog.base.set_state(120);
                        recog.fullFunctionCall()?;
                    }
                }
                2 => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 2)?;
                    recog.base.enter_outer_alt(None, 2)?;
                    {
                        /*InvokeRule emptyFunctionCall*/
                        recog.base.set_state(121);
                        recog.emptyFunctionCall()?;
                    }
                }

                _ => {}
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
//------------------- functionCallList ----------------
pub type FunctionCallListContextAll<'input> = FunctionCallListContext<'input>;

pub type FunctionCallListContext<'input> =
    BaseParserRuleContext<'input, FunctionCallListContextExt<'input>>;

#[derive(Clone)]
pub struct FunctionCallListContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrPythonParserContext<'input> for FunctionCallListContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrPythonParserListener<'input> + 'a>
    for FunctionCallListContext<'input>
{
    fn enter(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_functionCallList(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrPythonParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_functionCallList(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for FunctionCallListContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrPythonParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_functionCallList
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_functionCallList }
}
antlr4rust::tid! {FunctionCallListContextExt<'a>}

impl<'input> FunctionCallListContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrPythonParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<FunctionCallListContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            FunctionCallListContextExt { ph: PhantomData },
        ))
    }
}

pub trait FunctionCallListContextAttrs<'input>:
    AntlrPythonParserContext<'input> + BorrowMut<FunctionCallListContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token LIST_OPEN
    /// Returns `None` if there is no child corresponding to token LIST_OPEN
    fn LIST_OPEN(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_LIST_OPEN, 0)
    }
    fn functionCall_all(&self) -> Vec<Rc<FunctionCallContextAll<'input>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    fn functionCall(&self, i: usize) -> Option<Rc<FunctionCallContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(i)
    }
    /// Retrieves first TerminalNode corresponding to token LIST_CLOSE
    /// Returns `None` if there is no child corresponding to token LIST_CLOSE
    fn LIST_CLOSE(&self) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_LIST_CLOSE, 0)
    }
    /// Retrieves all `TerminalNode`s corresponding to token SEP in current rule
    fn SEP_all(&self) -> Vec<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    /// Retrieves 'i's TerminalNode corresponding to token SEP, starting from 0.
    /// Returns `None` if number of children corresponding to token SEP is less or equal than `i`.
    fn SEP(&self, i: usize) -> Option<Rc<TerminalNode<'input, AntlrPythonParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrPythonParser_SEP, i)
    }
}

impl<'input> FunctionCallListContextAttrs<'input> for FunctionCallListContext<'input> {}

impl<'input, I> AntlrPythonParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn functionCallList(
        &mut self,
    ) -> Result<Rc<FunctionCallListContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx =
            FunctionCallListContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 24, RULE_functionCallList);
        let mut _localctx: Rc<FunctionCallListContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            let mut _alt: i32;
            recog.base.set_state(140);
            recog.err_handler.sync(&mut recog.base)?;
            match recog.interpreter.adaptive_predict(15, &mut recog.base)? {
                1 => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
                    recog.base.enter_outer_alt(None, 1)?;
                    {
                        recog.base.set_state(124);
                        recog
                            .base
                            .match_token(AntlrPythonParser_LIST_OPEN, &mut recog.err_handler)?;

                        /*InvokeRule functionCall*/
                        recog.base.set_state(125);
                        recog.functionCall()?;

                        recog.base.set_state(130);
                        recog.err_handler.sync(&mut recog.base)?;
                        _alt = recog.interpreter.adaptive_predict(13, &mut recog.base)?;
                        while { _alt != 2 && _alt != INVALID_ALT } {
                            if _alt == 1 {
                                {
                                    {
                                        recog.base.set_state(126);
                                        recog.base.match_token(
                                            AntlrPythonParser_SEP,
                                            &mut recog.err_handler,
                                        )?;

                                        /*InvokeRule functionCall*/
                                        recog.base.set_state(127);
                                        recog.functionCall()?;
                                    }
                                }
                            }
                            recog.base.set_state(132);
                            recog.err_handler.sync(&mut recog.base)?;
                            _alt = recog.interpreter.adaptive_predict(13, &mut recog.base)?;
                        }
                        recog.base.set_state(134);
                        recog.err_handler.sync(&mut recog.base)?;
                        _la = recog.base.input.la(1);
                        if _la == AntlrPythonParser_SEP {
                            {
                                recog.base.set_state(133);
                                recog
                                    .base
                                    .match_token(AntlrPythonParser_SEP, &mut recog.err_handler)?;
                            }
                        }

                        recog.base.set_state(136);
                        recog
                            .base
                            .match_token(AntlrPythonParser_LIST_CLOSE, &mut recog.err_handler)?;
                    }
                }
                2 => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 2)?;
                    recog.base.enter_outer_alt(None, 2)?;
                    {
                        recog.base.set_state(138);
                        recog
                            .base
                            .match_token(AntlrPythonParser_LIST_OPEN, &mut recog.err_handler)?;

                        recog.base.set_state(139);
                        recog
                            .base
                            .match_token(AntlrPythonParser_LIST_CLOSE, &mut recog.err_handler)?;
                    }
                }

                _ => {}
            }
            Ok(())
        })();
        match result {
            Ok(_) => {}
            Err(e @ ANTLRError::FallThrough(_)) => return Err(e),
            Err(ref re) => {
                //_localctx.exception = re;
                recog.err_handler.report_error(&mut recog.base, re);
                recog.err_handler.recover(&mut recog.base, re)?;
            }
        }
        recog.base.exit_rule()?;

        Ok(_localctx)
    }
}
lazy_static! {
    static ref _ATN: Arc<ATN> =
        Arc::new(ATNDeserializer::new(None).deserialize(&mut _serializedATN.iter()));
    static ref _decision_to_DFA: Arc<Vec<antlr4rust::RwLock<DFA>>> = {
        let mut dfa = Vec::new();
        let size = _ATN.decision_to_state.len() as i32;
        for i in 0..size {
            dfa.push(DFA::new(_ATN.clone(), _ATN.get_decision_state(i), i).into())
        }
        Arc::new(dfa)
    };
    static ref _serializedATN: Vec<i32> = vec![
        4, 1, 16, 143, 2, 0, 7, 0, 2, 1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7, 4, 2, 5, 7, 5, 2,
        6, 7, 6, 2, 7, 7, 7, 2, 8, 7, 8, 2, 9, 7, 9, 2, 10, 7, 10, 2, 11, 7, 11, 2, 12, 7, 12, 1,
        0, 1, 0, 1, 0, 1, 1, 4, 1, 31, 8, 1, 11, 1, 12, 1, 32, 1, 1, 3, 1, 36, 8, 1, 1, 2, 1, 2, 1,
        3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 48, 8, 3, 1, 4, 1, 4, 1, 4, 1, 4, 5, 4,
        54, 8, 4, 10, 4, 12, 4, 57, 9, 4, 1, 4, 3, 4, 60, 8, 4, 3, 4, 62, 8, 4, 1, 4, 1, 4, 1, 5,
        1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 5, 5, 75, 8, 5, 10, 5, 12, 5, 78, 9, 5, 1,
        5, 3, 5, 81, 8, 5, 3, 5, 83, 8, 5, 1, 5, 1, 5, 1, 6, 1, 6, 1, 6, 1, 6, 1, 7, 1, 7, 1, 7, 5,
        7, 94, 8, 7, 10, 7, 12, 7, 97, 9, 7, 1, 7, 3, 7, 100, 8, 7, 1, 8, 1, 8, 1, 8, 1, 8, 1, 8,
        1, 8, 1, 8, 1, 8, 3, 8, 110, 8, 8, 1, 9, 1, 9, 1, 9, 1, 9, 1, 10, 1, 10, 1, 10, 1, 10, 1,
        10, 1, 11, 1, 11, 3, 11, 123, 8, 11, 1, 12, 1, 12, 1, 12, 1, 12, 5, 12, 129, 8, 12, 10, 12,
        12, 12, 132, 9, 12, 1, 12, 3, 12, 135, 8, 12, 1, 12, 1, 12, 1, 12, 1, 12, 3, 12, 141, 8,
        12, 1, 12, 0, 0, 13, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 0, 0, 151, 0, 26, 1, 0,
        0, 0, 2, 35, 1, 0, 0, 0, 4, 37, 1, 0, 0, 0, 6, 47, 1, 0, 0, 0, 8, 49, 1, 0, 0, 0, 10, 65,
        1, 0, 0, 0, 12, 86, 1, 0, 0, 0, 14, 90, 1, 0, 0, 0, 16, 109, 1, 0, 0, 0, 18, 111, 1, 0, 0,
        0, 20, 115, 1, 0, 0, 0, 22, 122, 1, 0, 0, 0, 24, 140, 1, 0, 0, 0, 26, 27, 3, 2, 1, 0, 27,
        28, 5, 0, 0, 1, 28, 1, 1, 0, 0, 0, 29, 31, 3, 22, 11, 0, 30, 29, 1, 0, 0, 0, 31, 32, 1, 0,
        0, 0, 32, 30, 1, 0, 0, 0, 32, 33, 1, 0, 0, 0, 33, 36, 1, 0, 0, 0, 34, 36, 3, 24, 12, 0, 35,
        30, 1, 0, 0, 0, 35, 34, 1, 0, 0, 0, 36, 3, 1, 0, 0, 0, 37, 38, 5, 13, 0, 0, 38, 5, 1, 0, 0,
        0, 39, 48, 5, 11, 0, 0, 40, 48, 5, 12, 0, 0, 41, 48, 5, 10, 0, 0, 42, 48, 5, 13, 0, 0, 43,
        48, 5, 14, 0, 0, 44, 48, 3, 8, 4, 0, 45, 48, 3, 10, 5, 0, 46, 48, 3, 16, 8, 0, 47, 39, 1,
        0, 0, 0, 47, 40, 1, 0, 0, 0, 47, 41, 1, 0, 0, 0, 47, 42, 1, 0, 0, 0, 47, 43, 1, 0, 0, 0,
        47, 44, 1, 0, 0, 0, 47, 45, 1, 0, 0, 0, 47, 46, 1, 0, 0, 0, 48, 7, 1, 0, 0, 0, 49, 61, 5,
        8, 0, 0, 50, 55, 3, 6, 3, 0, 51, 52, 5, 3, 0, 0, 52, 54, 3, 6, 3, 0, 53, 51, 1, 0, 0, 0,
        54, 57, 1, 0, 0, 0, 55, 53, 1, 0, 0, 0, 55, 56, 1, 0, 0, 0, 56, 59, 1, 0, 0, 0, 57, 55, 1,
        0, 0, 0, 58, 60, 5, 3, 0, 0, 59, 58, 1, 0, 0, 0, 59, 60, 1, 0, 0, 0, 60, 62, 1, 0, 0, 0,
        61, 50, 1, 0, 0, 0, 61, 62, 1, 0, 0, 0, 62, 63, 1, 0, 0, 0, 63, 64, 5, 9, 0, 0, 64, 9, 1,
        0, 0, 0, 65, 82, 5, 6, 0, 0, 66, 67, 3, 4, 2, 0, 67, 68, 5, 2, 0, 0, 68, 76, 3, 6, 3, 0,
        69, 70, 5, 3, 0, 0, 70, 71, 3, 4, 2, 0, 71, 72, 5, 2, 0, 0, 72, 73, 3, 6, 3, 0, 73, 75, 1,
        0, 0, 0, 74, 69, 1, 0, 0, 0, 75, 78, 1, 0, 0, 0, 76, 74, 1, 0, 0, 0, 76, 77, 1, 0, 0, 0,
        77, 80, 1, 0, 0, 0, 78, 76, 1, 0, 0, 0, 79, 81, 5, 3, 0, 0, 80, 79, 1, 0, 0, 0, 80, 81, 1,
        0, 0, 0, 81, 83, 1, 0, 0, 0, 82, 66, 1, 0, 0, 0, 82, 83, 1, 0, 0, 0, 83, 84, 1, 0, 0, 0,
        84, 85, 5, 7, 0, 0, 85, 11, 1, 0, 0, 0, 86, 87, 5, 15, 0, 0, 87, 88, 5, 1, 0, 0, 88, 89, 3,
        6, 3, 0, 89, 13, 1, 0, 0, 0, 90, 95, 3, 12, 6, 0, 91, 92, 5, 3, 0, 0, 92, 94, 3, 12, 6, 0,
        93, 91, 1, 0, 0, 0, 94, 97, 1, 0, 0, 0, 95, 93, 1, 0, 0, 0, 95, 96, 1, 0, 0, 0, 96, 99, 1,
        0, 0, 0, 97, 95, 1, 0, 0, 0, 98, 100, 5, 3, 0, 0, 99, 98, 1, 0, 0, 0, 99, 100, 1, 0, 0, 0,
        100, 15, 1, 0, 0, 0, 101, 102, 5, 15, 0, 0, 102, 103, 5, 4, 0, 0, 103, 110, 5, 5, 0, 0,
        104, 105, 5, 15, 0, 0, 105, 106, 5, 4, 0, 0, 106, 107, 3, 14, 7, 0, 107, 108, 5, 5, 0, 0,
        108, 110, 1, 0, 0, 0, 109, 101, 1, 0, 0, 0, 109, 104, 1, 0, 0, 0, 110, 17, 1, 0, 0, 0, 111,
        112, 5, 15, 0, 0, 112, 113, 5, 4, 0, 0, 113, 114, 5, 5, 0, 0, 114, 19, 1, 0, 0, 0, 115,
        116, 5, 15, 0, 0, 116, 117, 5, 4, 0, 0, 117, 118, 3, 14, 7, 0, 118, 119, 5, 5, 0, 0, 119,
        21, 1, 0, 0, 0, 120, 123, 3, 20, 10, 0, 121, 123, 3, 18, 9, 0, 122, 120, 1, 0, 0, 0, 122,
        121, 1, 0, 0, 0, 123, 23, 1, 0, 0, 0, 124, 125, 5, 8, 0, 0, 125, 130, 3, 22, 11, 0, 126,
        127, 5, 3, 0, 0, 127, 129, 3, 22, 11, 0, 128, 126, 1, 0, 0, 0, 129, 132, 1, 0, 0, 0, 130,
        128, 1, 0, 0, 0, 130, 131, 1, 0, 0, 0, 131, 134, 1, 0, 0, 0, 132, 130, 1, 0, 0, 0, 133,
        135, 5, 3, 0, 0, 134, 133, 1, 0, 0, 0, 134, 135, 1, 0, 0, 0, 135, 136, 1, 0, 0, 0, 136,
        137, 5, 9, 0, 0, 137, 141, 1, 0, 0, 0, 138, 139, 5, 8, 0, 0, 139, 141, 5, 9, 0, 0, 140,
        124, 1, 0, 0, 0, 140, 138, 1, 0, 0, 0, 141, 25, 1, 0, 0, 0, 16, 32, 35, 47, 55, 59, 61, 76,
        80, 82, 95, 99, 109, 122, 130, 134, 140
    ];
}
