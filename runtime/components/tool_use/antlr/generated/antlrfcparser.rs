// Generated from AntlrFcParser.g4 by ANTLR 4.13.2
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

use super::antlrfcparserlistener::*;
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

pub const AntlrFcParser_OPEN_BRACE: i32 = 1;
pub const AntlrFcParser_CLOSE_BRACE: i32 = 2;
pub const AntlrFcParser_OPEN_BRACKET: i32 = 3;
pub const AntlrFcParser_CLOSE_BRACKET: i32 = 4;
pub const AntlrFcParser_COMMA: i32 = 5;
pub const AntlrFcParser_COLON: i32 = 6;
pub const AntlrFcParser_ESCAPE: i32 = 7;
pub const AntlrFcParser_BOOLEAN: i32 = 8;
pub const AntlrFcParser_NULL_LITERAL: i32 = 9;
pub const AntlrFcParser_NUMBER: i32 = 10;
pub const AntlrFcParser_ESCAPED_STRING: i32 = 11;
pub const AntlrFcParser_CALL: i32 = 12;
pub const AntlrFcParser_ID: i32 = 13;
pub const AntlrFcParser_WS: i32 = 14;
pub const AntlrFcParser_EOF: i32 = EOF;
pub const RULE_start: usize = 0;
pub const RULE_functionCall: usize = 1;
pub const RULE_object: usize = 2;
pub const RULE_pair: usize = 3;
pub const RULE_value: usize = 4;
pub const RULE_array: usize = 5;
pub const ruleNames: [&'static str; 6] =
    ["start", "functionCall", "object", "pair", "value", "array"];

pub const _LITERAL_NAMES: [Option<&'static str>; 13] = [
    None,
    Some("'{'"),
    Some("'}'"),
    Some("'['"),
    Some("']'"),
    Some("','"),
    Some("':'"),
    None,
    None,
    Some("'null'"),
    None,
    None,
    Some("'call'"),
];
pub const _SYMBOLIC_NAMES: [Option<&'static str>; 15] = [
    None,
    Some("OPEN_BRACE"),
    Some("CLOSE_BRACE"),
    Some("OPEN_BRACKET"),
    Some("CLOSE_BRACKET"),
    Some("COMMA"),
    Some("COLON"),
    Some("ESCAPE"),
    Some("BOOLEAN"),
    Some("NULL_LITERAL"),
    Some("NUMBER"),
    Some("ESCAPED_STRING"),
    Some("CALL"),
    Some("ID"),
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
    AntlrFcParserExt<'input>,
    I,
    AntlrFcParserContextType,
    dyn AntlrFcParserListener<'input> + 'input,
>;

type TokenType<'input> = <LocalTokenFactory<'input> as TokenFactory<'input>>::Tok;
pub type LocalTokenFactory<'input> = CommonTokenFactory;

pub type AntlrFcParserTreeWalker<'input, 'a> =
    ParseTreeWalker<'input, 'a, AntlrFcParserContextType, dyn AntlrFcParserListener<'input> + 'a>;

/// Parser for AntlrFcParser grammar
pub struct AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    base: BaseParserType<'input, I>,
    interpreter: Arc<ParserATNSimulator>,
    _shared_context_cache: Box<PredictionContextCache>,
    pub err_handler: Box<dyn ErrorStrategy<'input, BaseParserType<'input, I>>>,
}

impl<'input, I> AntlrFcParser<'input, I>
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
                AntlrFcParserExt { _pd: Default::default() },
            ),
            interpreter,
            _shared_context_cache: Box::new(PredictionContextCache::new()),
            err_handler: strategy,
        }
    }
}

type DynStrategy<'input, I> = Box<dyn ErrorStrategy<'input, BaseParserType<'input, I>> + 'input>;

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn with_dyn_strategy(input: I) -> Self {
        Self::with_strategy(input, Box::new(DefaultErrorStrategy::new()))
    }
}

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn new(input: I) -> Self {
        Self::with_strategy(input, Box::new(DefaultErrorStrategy::new()))
    }
}

/// Trait for monomorphized trait object that corresponds to the nodes of parse tree generated for AntlrFcParser
pub trait AntlrFcParserContext<'input>:
    for<'x> Listenable<dyn AntlrFcParserListener<'input> + 'x>
    + ParserRuleContext<'input, TF = LocalTokenFactory<'input>, Ctx = AntlrFcParserContextType>
{
}

antlr4rust::coerce_from! { 'input : AntlrFcParserContext<'input> }

impl<'input> AntlrFcParserContext<'input> for TerminalNode<'input, AntlrFcParserContextType> {}
impl<'input> AntlrFcParserContext<'input> for ErrorNode<'input, AntlrFcParserContextType> {}

antlr4rust::tid! { impl<'input> TidAble<'input> for dyn AntlrFcParserContext<'input> + 'input }

antlr4rust::tid! { impl<'input> TidAble<'input> for dyn AntlrFcParserListener<'input> + 'input }

pub struct AntlrFcParserContextType;
antlr4rust::tid! {AntlrFcParserContextType}

impl<'input> ParserNodeType<'input> for AntlrFcParserContextType {
    type TF = LocalTokenFactory<'input>;
    type Type = dyn AntlrFcParserContext<'input> + 'input;
}

impl<'input, I> Deref for AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    type Target = BaseParserType<'input, I>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'input, I> DerefMut for AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

pub struct AntlrFcParserExt<'input> {
    _pd: PhantomData<&'input str>,
}

impl<'input> AntlrFcParserExt<'input> {}
antlr4rust::tid! { AntlrFcParserExt<'a> }

impl<'input> TokenAware<'input> for AntlrFcParserExt<'input> {
    type TF = LocalTokenFactory<'input>;
}

impl<'input, I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>>
    ParserRecog<'input, BaseParserType<'input, I>> for AntlrFcParserExt<'input>
{
}

impl<'input, I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>>
    Actions<'input, BaseParserType<'input, I>> for AntlrFcParserExt<'input>
{
    fn get_grammar_file_name(&self) -> &str {
        "AntlrFcParser.g4"
    }

    fn get_rule_names(&self) -> &[&str] {
        &ruleNames
    }

    fn get_vocabulary(&self) -> &dyn Vocabulary {
        &**VOCABULARY
    }
}
//------------------- start ----------------
pub type StartContextAll<'input> = StartContext<'input>;

pub type StartContext<'input> = BaseParserRuleContext<'input, StartContextExt<'input>>;

#[derive(Clone)]
pub struct StartContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrFcParserContext<'input> for StartContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrFcParserListener<'input> + 'a> for StartContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_start(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_start(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for StartContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrFcParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_start
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_start }
}
antlr4rust::tid! {StartContextExt<'a>}

impl<'input> StartContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrFcParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<StartContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            StartContextExt { ph: PhantomData },
        ))
    }
}

pub trait StartContextAttrs<'input>:
    AntlrFcParserContext<'input> + BorrowMut<StartContextExt<'input>>
{
    fn functionCall(&self) -> Option<Rc<FunctionCallContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
    /// Retrieves first TerminalNode corresponding to token EOF
    /// Returns `None` if there is no child corresponding to token EOF
    fn EOF(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_EOF, 0)
    }
}

impl<'input> StartContextAttrs<'input> for StartContext<'input> {}

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn start(&mut self) -> Result<Rc<StartContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = StartContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 0, RULE_start);
        let mut _localctx: Rc<StartContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                /*InvokeRule functionCall*/
                recog.base.set_state(12);
                recog.functionCall()?;

                recog.base.set_state(13);
                recog.base.match_token(AntlrFcParser_EOF, &mut recog.err_handler)?;
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

impl<'input> AntlrFcParserContext<'input> for FunctionCallContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrFcParserListener<'input> + 'a>
    for FunctionCallContext<'input>
{
    fn enter(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_functionCall(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_functionCall(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for FunctionCallContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrFcParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_functionCall
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_functionCall }
}
antlr4rust::tid! {FunctionCallContextExt<'a>}

impl<'input> FunctionCallContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrFcParserContext<'input> + 'input>>,
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
    AntlrFcParserContext<'input> + BorrowMut<FunctionCallContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token CALL
    /// Returns `None` if there is no child corresponding to token CALL
    fn CALL(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_CALL, 0)
    }
    /// Retrieves first TerminalNode corresponding to token COLON
    /// Returns `None` if there is no child corresponding to token COLON
    fn COLON(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_COLON, 0)
    }
    /// Retrieves first TerminalNode corresponding to token ID
    /// Returns `None` if there is no child corresponding to token ID
    fn ID(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_ID, 0)
    }
    fn object(&self) -> Option<Rc<ObjectContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> FunctionCallContextAttrs<'input> for FunctionCallContext<'input> {}

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn functionCall(&mut self) -> Result<Rc<FunctionCallContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = FunctionCallContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 2, RULE_functionCall);
        let mut _localctx: Rc<FunctionCallContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(15);
                recog.base.match_token(AntlrFcParser_CALL, &mut recog.err_handler)?;

                recog.base.set_state(16);
                recog.base.match_token(AntlrFcParser_COLON, &mut recog.err_handler)?;

                recog.base.set_state(17);
                recog.base.match_token(AntlrFcParser_ID, &mut recog.err_handler)?;

                recog.base.set_state(19);
                recog.err_handler.sync(&mut recog.base)?;
                _la = recog.base.input.la(1);
                if _la == AntlrFcParser_OPEN_BRACE {
                    {
                        /*InvokeRule object*/
                        recog.base.set_state(18);
                        recog.object()?;
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

impl<'input> AntlrFcParserContext<'input> for ObjectContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrFcParserListener<'input> + 'a> for ObjectContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_object(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_object(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ObjectContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrFcParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_object
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_object }
}
antlr4rust::tid! {ObjectContextExt<'a>}

impl<'input> ObjectContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrFcParserContext<'input> + 'input>>,
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
    AntlrFcParserContext<'input> + BorrowMut<ObjectContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token OPEN_BRACE
    /// Returns `None` if there is no child corresponding to token OPEN_BRACE
    fn OPEN_BRACE(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_OPEN_BRACE, 0)
    }
    /// Retrieves first TerminalNode corresponding to token CLOSE_BRACE
    /// Returns `None` if there is no child corresponding to token CLOSE_BRACE
    fn CLOSE_BRACE(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_CLOSE_BRACE, 0)
    }
    fn pair_all(&self) -> Vec<Rc<PairContextAll<'input>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    fn pair(&self, i: usize) -> Option<Rc<PairContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(i)
    }
    /// Retrieves all `TerminalNode`s corresponding to token COMMA in current rule
    fn COMMA_all(&self) -> Vec<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    /// Retrieves 'i's TerminalNode corresponding to token COMMA, starting from 0.
    /// Returns `None` if number of children corresponding to token COMMA is less or equal than `i`.
    fn COMMA(&self, i: usize) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_COMMA, i)
    }
}

impl<'input> ObjectContextAttrs<'input> for ObjectContext<'input> {}

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn object(&mut self) -> Result<Rc<ObjectContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ObjectContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 4, RULE_object);
        let mut _localctx: Rc<ObjectContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(21);
                recog.base.match_token(AntlrFcParser_OPEN_BRACE, &mut recog.err_handler)?;

                recog.base.set_state(30);
                recog.err_handler.sync(&mut recog.base)?;
                _la = recog.base.input.la(1);
                if _la == AntlrFcParser_ID {
                    {
                        /*InvokeRule pair*/
                        recog.base.set_state(22);
                        recog.pair()?;

                        recog.base.set_state(27);
                        recog.err_handler.sync(&mut recog.base)?;
                        _la = recog.base.input.la(1);
                        while _la == AntlrFcParser_COMMA {
                            {
                                {
                                    recog.base.set_state(23);
                                    recog
                                        .base
                                        .match_token(AntlrFcParser_COMMA, &mut recog.err_handler)?;

                                    /*InvokeRule pair*/
                                    recog.base.set_state(24);
                                    recog.pair()?;
                                }
                            }
                            recog.base.set_state(29);
                            recog.err_handler.sync(&mut recog.base)?;
                            _la = recog.base.input.la(1);
                        }
                    }
                }

                recog.base.set_state(32);
                recog.base.match_token(AntlrFcParser_CLOSE_BRACE, &mut recog.err_handler)?;
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
//------------------- pair ----------------
pub type PairContextAll<'input> = PairContext<'input>;

pub type PairContext<'input> = BaseParserRuleContext<'input, PairContextExt<'input>>;

#[derive(Clone)]
pub struct PairContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrFcParserContext<'input> for PairContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrFcParserListener<'input> + 'a> for PairContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_pair(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_pair(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for PairContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrFcParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_pair
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_pair }
}
antlr4rust::tid! {PairContextExt<'a>}

impl<'input> PairContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrFcParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<PairContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            PairContextExt { ph: PhantomData },
        ))
    }
}

pub trait PairContextAttrs<'input>:
    AntlrFcParserContext<'input> + BorrowMut<PairContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token ID
    /// Returns `None` if there is no child corresponding to token ID
    fn ID(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_ID, 0)
    }
    /// Retrieves first TerminalNode corresponding to token COLON
    /// Returns `None` if there is no child corresponding to token COLON
    fn COLON(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_COLON, 0)
    }
    fn value(&self) -> Option<Rc<ValueContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> PairContextAttrs<'input> for PairContext<'input> {}

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn pair(&mut self) -> Result<Rc<PairContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = PairContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 6, RULE_pair);
        let mut _localctx: Rc<PairContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(34);
                recog.base.match_token(AntlrFcParser_ID, &mut recog.err_handler)?;

                recog.base.set_state(35);
                recog.base.match_token(AntlrFcParser_COLON, &mut recog.err_handler)?;

                /*InvokeRule value*/
                recog.base.set_state(36);
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
//------------------- value ----------------
pub type ValueContextAll<'input> = ValueContext<'input>;

pub type ValueContext<'input> = BaseParserRuleContext<'input, ValueContextExt<'input>>;

#[derive(Clone)]
pub struct ValueContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrFcParserContext<'input> for ValueContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrFcParserListener<'input> + 'a> for ValueContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_value(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_value(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ValueContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrFcParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_value
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_value }
}
antlr4rust::tid! {ValueContextExt<'a>}

impl<'input> ValueContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrFcParserContext<'input> + 'input>>,
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
    AntlrFcParserContext<'input> + BorrowMut<ValueContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token ESCAPED_STRING
    /// Returns `None` if there is no child corresponding to token ESCAPED_STRING
    fn ESCAPED_STRING(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_ESCAPED_STRING, 0)
    }
    /// Retrieves first TerminalNode corresponding to token NUMBER
    /// Returns `None` if there is no child corresponding to token NUMBER
    fn NUMBER(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_NUMBER, 0)
    }
    /// Retrieves first TerminalNode corresponding to token BOOLEAN
    /// Returns `None` if there is no child corresponding to token BOOLEAN
    fn BOOLEAN(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_BOOLEAN, 0)
    }
    /// Retrieves first TerminalNode corresponding to token NULL_LITERAL
    /// Returns `None` if there is no child corresponding to token NULL_LITERAL
    fn NULL_LITERAL(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_NULL_LITERAL, 0)
    }
    fn object(&self) -> Option<Rc<ObjectContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
    fn array(&self) -> Option<Rc<ArrayContextAll<'input>>>
    where
        Self: Sized,
    {
        self.child_of_type(0)
    }
}

impl<'input> ValueContextAttrs<'input> for ValueContext<'input> {}

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn value(&mut self) -> Result<Rc<ValueContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ValueContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 8, RULE_value);
        let mut _localctx: Rc<ValueContextAll> = _localctx;
        let result: Result<(), ANTLRError> = (|| {
            recog.base.set_state(44);
            recog.err_handler.sync(&mut recog.base)?;
            match recog.base.input.la(1) {
                AntlrFcParser_ESCAPED_STRING => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
                    recog.base.enter_outer_alt(None, 1)?;
                    {
                        recog.base.set_state(38);
                        recog
                            .base
                            .match_token(AntlrFcParser_ESCAPED_STRING, &mut recog.err_handler)?;
                    }
                }

                AntlrFcParser_NUMBER => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 2)?;
                    recog.base.enter_outer_alt(None, 2)?;
                    {
                        recog.base.set_state(39);
                        recog.base.match_token(AntlrFcParser_NUMBER, &mut recog.err_handler)?;
                    }
                }

                AntlrFcParser_BOOLEAN => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 3)?;
                    recog.base.enter_outer_alt(None, 3)?;
                    {
                        recog.base.set_state(40);
                        recog.base.match_token(AntlrFcParser_BOOLEAN, &mut recog.err_handler)?;
                    }
                }

                AntlrFcParser_NULL_LITERAL => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 4)?;
                    recog.base.enter_outer_alt(None, 4)?;
                    {
                        recog.base.set_state(41);
                        recog
                            .base
                            .match_token(AntlrFcParser_NULL_LITERAL, &mut recog.err_handler)?;
                    }
                }

                AntlrFcParser_OPEN_BRACE => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 5)?;
                    recog.base.enter_outer_alt(None, 5)?;
                    {
                        /*InvokeRule object*/
                        recog.base.set_state(42);
                        recog.object()?;
                    }
                }

                AntlrFcParser_OPEN_BRACKET => {
                    //recog.base.enter_outer_alt(_localctx.clone(), 6)?;
                    recog.base.enter_outer_alt(None, 6)?;
                    {
                        /*InvokeRule array*/
                        recog.base.set_state(43);
                        recog.array()?;
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
//------------------- array ----------------
pub type ArrayContextAll<'input> = ArrayContext<'input>;

pub type ArrayContext<'input> = BaseParserRuleContext<'input, ArrayContextExt<'input>>;

#[derive(Clone)]
pub struct ArrayContextExt<'input> {
    ph: PhantomData<&'input str>,
}

impl<'input> AntlrFcParserContext<'input> for ArrayContext<'input> {}

impl<'input, 'a> Listenable<dyn AntlrFcParserListener<'input> + 'a> for ArrayContext<'input> {
    fn enter(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.enter_every_rule(self)?;
        listener.enter_array(self);
        Ok(())
    }
    fn exit(
        &self,
        listener: &mut (dyn AntlrFcParserListener<'input> + 'a),
    ) -> Result<(), ANTLRError> {
        listener.exit_array(self);
        listener.exit_every_rule(self)?;
        Ok(())
    }
}

impl<'input> CustomRuleContext<'input> for ArrayContextExt<'input> {
    type TF = LocalTokenFactory<'input>;
    type Ctx = AntlrFcParserContextType;
    fn get_rule_index(&self) -> usize {
        RULE_array
    }
    //fn type_rule_index() -> usize where Self: Sized { RULE_array }
}
antlr4rust::tid! {ArrayContextExt<'a>}

impl<'input> ArrayContextExt<'input> {
    fn new(
        parent: Option<Rc<dyn AntlrFcParserContext<'input> + 'input>>,
        invoking_state: i32,
    ) -> Rc<ArrayContextAll<'input>> {
        Rc::new(BaseParserRuleContext::new_parser_ctx(
            parent,
            invoking_state,
            ArrayContextExt { ph: PhantomData },
        ))
    }
}

pub trait ArrayContextAttrs<'input>:
    AntlrFcParserContext<'input> + BorrowMut<ArrayContextExt<'input>>
{
    /// Retrieves first TerminalNode corresponding to token OPEN_BRACKET
    /// Returns `None` if there is no child corresponding to token OPEN_BRACKET
    fn OPEN_BRACKET(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_OPEN_BRACKET, 0)
    }
    /// Retrieves first TerminalNode corresponding to token CLOSE_BRACKET
    /// Returns `None` if there is no child corresponding to token CLOSE_BRACKET
    fn CLOSE_BRACKET(&self) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_CLOSE_BRACKET, 0)
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
    /// Retrieves all `TerminalNode`s corresponding to token COMMA in current rule
    fn COMMA_all(&self) -> Vec<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.children_of_type()
    }
    /// Retrieves 'i's TerminalNode corresponding to token COMMA, starting from 0.
    /// Returns `None` if number of children corresponding to token COMMA is less or equal than `i`.
    fn COMMA(&self, i: usize) -> Option<Rc<TerminalNode<'input, AntlrFcParserContextType>>>
    where
        Self: Sized,
    {
        self.get_token(AntlrFcParser_COMMA, i)
    }
}

impl<'input> ArrayContextAttrs<'input> for ArrayContext<'input> {}

impl<'input, I> AntlrFcParser<'input, I>
where
    I: TokenStream<'input, TF = LocalTokenFactory<'input>> + TidAble<'input>,
{
    pub fn array(&mut self) -> Result<Rc<ArrayContextAll<'input>>, ANTLRError> {
        let mut recog = self;
        let _parentctx = recog.ctx.take();
        let mut _localctx = ArrayContextExt::new(_parentctx.clone(), recog.base.get_state());
        recog.base.enter_rule(_localctx.clone(), 10, RULE_array);
        let mut _localctx: Rc<ArrayContextAll> = _localctx;
        let mut _la: i32 = -1;
        let result: Result<(), ANTLRError> = (|| {
            //recog.base.enter_outer_alt(_localctx.clone(), 1)?;
            recog.base.enter_outer_alt(None, 1)?;
            {
                recog.base.set_state(46);
                recog.base.match_token(AntlrFcParser_OPEN_BRACKET, &mut recog.err_handler)?;

                recog.base.set_state(55);
                recog.err_handler.sync(&mut recog.base)?;
                _la = recog.base.input.la(1);
                if (((_la) & !0x3f) == 0 && ((1usize << _la) & 3850) != 0) {
                    {
                        /*InvokeRule value*/
                        recog.base.set_state(47);
                        recog.value()?;

                        recog.base.set_state(52);
                        recog.err_handler.sync(&mut recog.base)?;
                        _la = recog.base.input.la(1);
                        while _la == AntlrFcParser_COMMA {
                            {
                                {
                                    recog.base.set_state(48);
                                    recog
                                        .base
                                        .match_token(AntlrFcParser_COMMA, &mut recog.err_handler)?;

                                    /*InvokeRule value*/
                                    recog.base.set_state(49);
                                    recog.value()?;
                                }
                            }
                            recog.base.set_state(54);
                            recog.err_handler.sync(&mut recog.base)?;
                            _la = recog.base.input.la(1);
                        }
                    }
                }

                recog.base.set_state(57);
                recog.base.match_token(AntlrFcParser_CLOSE_BRACKET, &mut recog.err_handler)?;
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
        4, 1, 14, 60, 2, 0, 7, 0, 2, 1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7, 4, 2, 5, 7, 5, 1, 0,
        1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 20, 8, 1, 1, 2, 1, 2, 1, 2, 1, 2, 5, 2, 26, 8, 2,
        10, 2, 12, 2, 29, 9, 2, 3, 2, 31, 8, 2, 1, 2, 1, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 4, 1, 4, 1,
        4, 1, 4, 1, 4, 1, 4, 3, 4, 45, 8, 4, 1, 5, 1, 5, 1, 5, 1, 5, 5, 5, 51, 8, 5, 10, 5, 12, 5,
        54, 9, 5, 3, 5, 56, 8, 5, 1, 5, 1, 5, 1, 5, 0, 0, 6, 0, 2, 4, 6, 8, 10, 0, 0, 63, 0, 12, 1,
        0, 0, 0, 2, 15, 1, 0, 0, 0, 4, 21, 1, 0, 0, 0, 6, 34, 1, 0, 0, 0, 8, 44, 1, 0, 0, 0, 10,
        46, 1, 0, 0, 0, 12, 13, 3, 2, 1, 0, 13, 14, 5, 0, 0, 1, 14, 1, 1, 0, 0, 0, 15, 16, 5, 12,
        0, 0, 16, 17, 5, 6, 0, 0, 17, 19, 5, 13, 0, 0, 18, 20, 3, 4, 2, 0, 19, 18, 1, 0, 0, 0, 19,
        20, 1, 0, 0, 0, 20, 3, 1, 0, 0, 0, 21, 30, 5, 1, 0, 0, 22, 27, 3, 6, 3, 0, 23, 24, 5, 5, 0,
        0, 24, 26, 3, 6, 3, 0, 25, 23, 1, 0, 0, 0, 26, 29, 1, 0, 0, 0, 27, 25, 1, 0, 0, 0, 27, 28,
        1, 0, 0, 0, 28, 31, 1, 0, 0, 0, 29, 27, 1, 0, 0, 0, 30, 22, 1, 0, 0, 0, 30, 31, 1, 0, 0, 0,
        31, 32, 1, 0, 0, 0, 32, 33, 5, 2, 0, 0, 33, 5, 1, 0, 0, 0, 34, 35, 5, 13, 0, 0, 35, 36, 5,
        6, 0, 0, 36, 37, 3, 8, 4, 0, 37, 7, 1, 0, 0, 0, 38, 45, 5, 11, 0, 0, 39, 45, 5, 10, 0, 0,
        40, 45, 5, 8, 0, 0, 41, 45, 5, 9, 0, 0, 42, 45, 3, 4, 2, 0, 43, 45, 3, 10, 5, 0, 44, 38, 1,
        0, 0, 0, 44, 39, 1, 0, 0, 0, 44, 40, 1, 0, 0, 0, 44, 41, 1, 0, 0, 0, 44, 42, 1, 0, 0, 0,
        44, 43, 1, 0, 0, 0, 45, 9, 1, 0, 0, 0, 46, 55, 5, 3, 0, 0, 47, 52, 3, 8, 4, 0, 48, 49, 5,
        5, 0, 0, 49, 51, 3, 8, 4, 0, 50, 48, 1, 0, 0, 0, 51, 54, 1, 0, 0, 0, 52, 50, 1, 0, 0, 0,
        52, 53, 1, 0, 0, 0, 53, 56, 1, 0, 0, 0, 54, 52, 1, 0, 0, 0, 55, 47, 1, 0, 0, 0, 55, 56, 1,
        0, 0, 0, 56, 57, 1, 0, 0, 0, 57, 58, 5, 4, 0, 0, 58, 11, 1, 0, 0, 0, 6, 19, 27, 30, 44, 52,
        55
    ];
}
