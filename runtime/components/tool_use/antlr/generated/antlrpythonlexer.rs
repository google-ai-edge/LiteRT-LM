// Generated from AntlrPythonLexer.g4 by ANTLR 4.13.2
#![allow(dead_code)]
#![allow(nonstandard_style)]
#![allow(unused_imports)]
#![allow(unused_variables)]

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

use antlr4rust::atn::ATN;
use antlr4rust::atn_deserializer::ATNDeserializer;
use antlr4rust::char_stream::CharStream;
use antlr4rust::dfa::DFA;
use antlr4rust::error_listener::ErrorListener;
use antlr4rust::int_stream::IntStream;
use antlr4rust::lexer::{BaseLexer, Lexer, LexerRecog};
use antlr4rust::lexer_atn_simulator::{ILexerATNSimulator, LexerATNSimulator};
use antlr4rust::parser_rule_context::{cast, BaseParserRuleContext, ParserRuleContext};
use antlr4rust::recognizer::{Actions, Recognizer};
use antlr4rust::rule_context::{BaseRuleContext, EmptyContext, EmptyCustomRuleContext};
use antlr4rust::token::*;
use antlr4rust::token_factory::{CommonTokenFactory, TokenAware, TokenFactory};
use antlr4rust::tree::ParseTree;
use antlr4rust::vocabulary::{Vocabulary, VocabularyImpl};
use antlr4rust::PredictionContextCache;
use antlr4rust::TokenSource;

use antlr4rust::{lazy_static, Tid, TidAble, TidExt};

use std::cell::RefCell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

pub const EQ: i32 = 1;
pub const COLON: i32 = 2;
pub const SEP: i32 = 3;
pub const OPEN_PAR: i32 = 4;
pub const CLOSE_PAR: i32 = 5;
pub const OPEN_BRACE: i32 = 6;
pub const CLOSE_BRACE: i32 = 7;
pub const LIST_OPEN: i32 = 8;
pub const LIST_CLOSE: i32 = 9;
pub const BOOL: i32 = 10;
pub const INT: i32 = 11;
pub const FLOAT: i32 = 12;
pub const STRING: i32 = 13;
pub const NONE: i32 = 14;
pub const NAME: i32 = 15;
pub const WS: i32 = 16;
pub const channelNames: [&'static str; 0 + 2] = ["DEFAULT_TOKEN_CHANNEL", "HIDDEN"];

pub const modeNames: [&'static str; 1] = ["DEFAULT_MODE"];

pub const ruleNames: [&'static str; 16] = [
    "EQ",
    "COLON",
    "SEP",
    "OPEN_PAR",
    "CLOSE_PAR",
    "OPEN_BRACE",
    "CLOSE_BRACE",
    "LIST_OPEN",
    "LIST_CLOSE",
    "BOOL",
    "INT",
    "FLOAT",
    "STRING",
    "NONE",
    "NAME",
    "WS",
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

pub type LexerContext<'input> =
    BaseRuleContext<'input, EmptyCustomRuleContext<'input, LocalTokenFactory<'input>>>;
pub type LocalTokenFactory<'input> = CommonTokenFactory;

type From<'a> = <LocalTokenFactory<'a> as TokenFactory<'a>>::From;

pub struct AntlrPythonLexer<'input, Input: CharStream<From<'input>>> {
    base: BaseLexer<'input, AntlrPythonLexerActions, Input, LocalTokenFactory<'input>>,
}

antlr4rust::tid! { impl<'input,Input> TidAble<'input> for AntlrPythonLexer<'input,Input> where Input:CharStream<From<'input> > }

impl<'input, Input: CharStream<From<'input>>> Deref for AntlrPythonLexer<'input, Input> {
    type Target = BaseLexer<'input, AntlrPythonLexerActions, Input, LocalTokenFactory<'input>>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'input, Input: CharStream<From<'input>>> DerefMut for AntlrPythonLexer<'input, Input> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<'input, Input: CharStream<From<'input>>> AntlrPythonLexer<'input, Input> {
    fn get_rule_names(&self) -> &'static [&'static str] {
        &ruleNames
    }
    fn get_literal_names(&self) -> &[Option<&str>] {
        &_LITERAL_NAMES
    }

    fn get_symbolic_names(&self) -> &[Option<&str>] {
        &_SYMBOLIC_NAMES
    }

    fn get_grammar_file_name(&self) -> &'static str {
        "AntlrPythonLexer.g4"
    }

    pub fn new_with_token_factory(input: Input, tf: &'input LocalTokenFactory<'input>) -> Self {
        antlr4rust::recognizer::check_version("0", "5");
        Self {
            base: BaseLexer::new_base_lexer(
                input,
                LexerATNSimulator::new_lexer_atnsimulator(
                    _ATN.clone(),
                    _decision_to_DFA.clone(),
                    _shared_context_cache.clone(),
                ),
                AntlrPythonLexerActions {},
                tf,
            ),
        }
    }
}

impl<'input, Input: CharStream<From<'input>>> AntlrPythonLexer<'input, Input>
where
    &'input LocalTokenFactory<'input>: Default,
{
    pub fn new(input: Input) -> Self {
        AntlrPythonLexer::new_with_token_factory(
            input,
            <&LocalTokenFactory<'input> as Default>::default(),
        )
    }
}

pub struct AntlrPythonLexerActions {}

impl AntlrPythonLexerActions {}

impl<'input, Input: CharStream<From<'input>>>
    Actions<'input, BaseLexer<'input, AntlrPythonLexerActions, Input, LocalTokenFactory<'input>>>
    for AntlrPythonLexerActions
{
}

impl<'input, Input: CharStream<From<'input>>> AntlrPythonLexer<'input, Input> {}

impl<'input, Input: CharStream<From<'input>>>
    LexerRecog<'input, BaseLexer<'input, AntlrPythonLexerActions, Input, LocalTokenFactory<'input>>>
    for AntlrPythonLexerActions
{
}
impl<'input> TokenAware<'input> for AntlrPythonLexerActions {
    type TF = LocalTokenFactory<'input>;
}

impl<'input, Input: CharStream<From<'input>>> TokenSource<'input>
    for AntlrPythonLexer<'input, Input>
{
    type TF = LocalTokenFactory<'input>;

    fn next_token(&mut self) -> <Self::TF as TokenFactory<'input>>::Tok {
        self.base.next_token()
    }

    fn get_line(&self) -> isize {
        self.base.get_line()
    }

    fn get_char_position_in_line(&self) -> isize {
        self.base.get_char_position_in_line()
    }

    fn get_input_stream(&mut self) -> Option<&mut dyn IntStream> {
        self.base.get_input_stream()
    }

    fn get_source_name(&self) -> String {
        self.base.get_source_name()
    }

    fn get_token_factory(&self) -> &'input Self::TF {
        self.base.get_token_factory()
    }

    fn get_dfa_string(&self) -> String {
        self.base.get_dfa_string()
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
        4, 0, 16, 143, 6, -1, 2, 0, 7, 0, 2, 1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7, 4, 2, 5, 7,
        5, 2, 6, 7, 6, 2, 7, 7, 7, 2, 8, 7, 8, 2, 9, 7, 9, 2, 10, 7, 10, 2, 11, 7, 11, 2, 12, 7,
        12, 2, 13, 7, 13, 2, 14, 7, 14, 2, 15, 7, 15, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 3, 1,
        3, 1, 4, 1, 4, 1, 5, 1, 5, 1, 6, 1, 6, 1, 7, 1, 7, 1, 8, 1, 8, 1, 9, 1, 9, 1, 9, 1, 9, 1,
        9, 1, 9, 1, 9, 1, 9, 1, 9, 3, 9, 61, 8, 9, 1, 10, 3, 10, 64, 8, 10, 1, 10, 4, 10, 67, 8,
        10, 11, 10, 12, 10, 68, 1, 11, 3, 11, 72, 8, 11, 1, 11, 4, 11, 75, 8, 11, 11, 11, 12, 11,
        76, 1, 11, 1, 11, 5, 11, 81, 8, 11, 10, 11, 12, 11, 84, 9, 11, 1, 11, 3, 11, 87, 8, 11, 1,
        11, 5, 11, 90, 8, 11, 10, 11, 12, 11, 93, 9, 11, 1, 11, 1, 11, 4, 11, 97, 8, 11, 11, 11,
        12, 11, 98, 3, 11, 101, 8, 11, 1, 12, 1, 12, 1, 12, 1, 12, 5, 12, 107, 8, 12, 10, 12, 12,
        12, 110, 9, 12, 1, 12, 1, 12, 1, 12, 1, 12, 1, 12, 5, 12, 117, 8, 12, 10, 12, 12, 12, 120,
        9, 12, 1, 12, 3, 12, 123, 8, 12, 1, 13, 1, 13, 1, 13, 1, 13, 1, 13, 1, 14, 1, 14, 5, 14,
        132, 8, 14, 10, 14, 12, 14, 135, 9, 14, 1, 15, 4, 15, 138, 8, 15, 11, 15, 12, 15, 139, 1,
        15, 1, 15, 0, 0, 16, 1, 1, 3, 2, 5, 3, 7, 4, 9, 5, 11, 6, 13, 7, 15, 8, 17, 9, 19, 10, 21,
        11, 23, 12, 25, 13, 27, 14, 29, 15, 31, 16, 1, 0, 8, 1, 0, 48, 57, 1, 0, 46, 46, 2, 0, 34,
        34, 92, 92, 1, 0, 92, 92, 2, 0, 39, 39, 92, 92, 3, 0, 65, 90, 95, 95, 97, 122, 4, 0, 48,
        57, 65, 90, 95, 95, 97, 122, 3, 0, 9, 10, 13, 13, 32, 32, 159, 0, 1, 1, 0, 0, 0, 0, 3, 1,
        0, 0, 0, 0, 5, 1, 0, 0, 0, 0, 7, 1, 0, 0, 0, 0, 9, 1, 0, 0, 0, 0, 11, 1, 0, 0, 0, 0, 13, 1,
        0, 0, 0, 0, 15, 1, 0, 0, 0, 0, 17, 1, 0, 0, 0, 0, 19, 1, 0, 0, 0, 0, 21, 1, 0, 0, 0, 0, 23,
        1, 0, 0, 0, 0, 25, 1, 0, 0, 0, 0, 27, 1, 0, 0, 0, 0, 29, 1, 0, 0, 0, 0, 31, 1, 0, 0, 0, 1,
        33, 1, 0, 0, 0, 3, 35, 1, 0, 0, 0, 5, 37, 1, 0, 0, 0, 7, 39, 1, 0, 0, 0, 9, 41, 1, 0, 0, 0,
        11, 43, 1, 0, 0, 0, 13, 45, 1, 0, 0, 0, 15, 47, 1, 0, 0, 0, 17, 49, 1, 0, 0, 0, 19, 60, 1,
        0, 0, 0, 21, 63, 1, 0, 0, 0, 23, 100, 1, 0, 0, 0, 25, 122, 1, 0, 0, 0, 27, 124, 1, 0, 0, 0,
        29, 129, 1, 0, 0, 0, 31, 137, 1, 0, 0, 0, 33, 34, 5, 61, 0, 0, 34, 2, 1, 0, 0, 0, 35, 36,
        5, 58, 0, 0, 36, 4, 1, 0, 0, 0, 37, 38, 5, 44, 0, 0, 38, 6, 1, 0, 0, 0, 39, 40, 5, 40, 0,
        0, 40, 8, 1, 0, 0, 0, 41, 42, 5, 41, 0, 0, 42, 10, 1, 0, 0, 0, 43, 44, 5, 123, 0, 0, 44,
        12, 1, 0, 0, 0, 45, 46, 5, 125, 0, 0, 46, 14, 1, 0, 0, 0, 47, 48, 5, 91, 0, 0, 48, 16, 1,
        0, 0, 0, 49, 50, 5, 93, 0, 0, 50, 18, 1, 0, 0, 0, 51, 52, 5, 84, 0, 0, 52, 53, 5, 114, 0,
        0, 53, 54, 5, 117, 0, 0, 54, 61, 5, 101, 0, 0, 55, 56, 5, 70, 0, 0, 56, 57, 5, 97, 0, 0,
        57, 58, 5, 108, 0, 0, 58, 59, 5, 115, 0, 0, 59, 61, 5, 101, 0, 0, 60, 51, 1, 0, 0, 0, 60,
        55, 1, 0, 0, 0, 61, 20, 1, 0, 0, 0, 62, 64, 5, 45, 0, 0, 63, 62, 1, 0, 0, 0, 63, 64, 1, 0,
        0, 0, 64, 66, 1, 0, 0, 0, 65, 67, 7, 0, 0, 0, 66, 65, 1, 0, 0, 0, 67, 68, 1, 0, 0, 0, 68,
        66, 1, 0, 0, 0, 68, 69, 1, 0, 0, 0, 69, 22, 1, 0, 0, 0, 70, 72, 5, 45, 0, 0, 71, 70, 1, 0,
        0, 0, 71, 72, 1, 0, 0, 0, 72, 74, 1, 0, 0, 0, 73, 75, 7, 0, 0, 0, 74, 73, 1, 0, 0, 0, 75,
        76, 1, 0, 0, 0, 76, 74, 1, 0, 0, 0, 76, 77, 1, 0, 0, 0, 77, 78, 1, 0, 0, 0, 78, 82, 7, 1,
        0, 0, 79, 81, 7, 0, 0, 0, 80, 79, 1, 0, 0, 0, 81, 84, 1, 0, 0, 0, 82, 80, 1, 0, 0, 0, 82,
        83, 1, 0, 0, 0, 83, 101, 1, 0, 0, 0, 84, 82, 1, 0, 0, 0, 85, 87, 5, 45, 0, 0, 86, 85, 1, 0,
        0, 0, 86, 87, 1, 0, 0, 0, 87, 91, 1, 0, 0, 0, 88, 90, 7, 0, 0, 0, 89, 88, 1, 0, 0, 0, 90,
        93, 1, 0, 0, 0, 91, 89, 1, 0, 0, 0, 91, 92, 1, 0, 0, 0, 92, 94, 1, 0, 0, 0, 93, 91, 1, 0,
        0, 0, 94, 96, 7, 1, 0, 0, 95, 97, 7, 0, 0, 0, 96, 95, 1, 0, 0, 0, 97, 98, 1, 0, 0, 0, 98,
        96, 1, 0, 0, 0, 98, 99, 1, 0, 0, 0, 99, 101, 1, 0, 0, 0, 100, 71, 1, 0, 0, 0, 100, 86, 1,
        0, 0, 0, 101, 24, 1, 0, 0, 0, 102, 108, 5, 34, 0, 0, 103, 107, 8, 2, 0, 0, 104, 105, 7, 3,
        0, 0, 105, 107, 9, 0, 0, 0, 106, 103, 1, 0, 0, 0, 106, 104, 1, 0, 0, 0, 107, 110, 1, 0, 0,
        0, 108, 106, 1, 0, 0, 0, 108, 109, 1, 0, 0, 0, 109, 111, 1, 0, 0, 0, 110, 108, 1, 0, 0, 0,
        111, 123, 5, 34, 0, 0, 112, 118, 5, 39, 0, 0, 113, 117, 8, 4, 0, 0, 114, 115, 7, 3, 0, 0,
        115, 117, 9, 0, 0, 0, 116, 113, 1, 0, 0, 0, 116, 114, 1, 0, 0, 0, 117, 120, 1, 0, 0, 0,
        118, 116, 1, 0, 0, 0, 118, 119, 1, 0, 0, 0, 119, 121, 1, 0, 0, 0, 120, 118, 1, 0, 0, 0,
        121, 123, 5, 39, 0, 0, 122, 102, 1, 0, 0, 0, 122, 112, 1, 0, 0, 0, 123, 26, 1, 0, 0, 0,
        124, 125, 5, 78, 0, 0, 125, 126, 5, 111, 0, 0, 126, 127, 5, 110, 0, 0, 127, 128, 5, 101, 0,
        0, 128, 28, 1, 0, 0, 0, 129, 133, 7, 5, 0, 0, 130, 132, 7, 6, 0, 0, 131, 130, 1, 0, 0, 0,
        132, 135, 1, 0, 0, 0, 133, 131, 1, 0, 0, 0, 133, 134, 1, 0, 0, 0, 134, 30, 1, 0, 0, 0, 135,
        133, 1, 0, 0, 0, 136, 138, 7, 7, 0, 0, 137, 136, 1, 0, 0, 0, 138, 139, 1, 0, 0, 0, 139,
        137, 1, 0, 0, 0, 139, 140, 1, 0, 0, 0, 140, 141, 1, 0, 0, 0, 141, 142, 6, 15, 0, 0, 142,
        32, 1, 0, 0, 0, 18, 0, 60, 63, 68, 71, 76, 82, 86, 91, 98, 100, 106, 108, 116, 118, 122,
        133, 139, 1, 6, 0, 0
    ];
}
