parser grammar AntlrFcParser;

@header {
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
}

options {
    tokenVocab = AntlrFcLexer;
}

start : functionCall EOF;

functionCall: CALL COLON ID object?;

object : OPEN_BRACE ( pair (COMMA pair)* )? CLOSE_BRACE;

pair : ID COLON value;

value
    : ESCAPED_STRING
    | NUMBER
    | BOOLEAN
    | NULL_LITERAL
    | object
    | array
    ;

array: OPEN_BRACKET ( value (COMMA value)* )? CLOSE_BRACKET;
