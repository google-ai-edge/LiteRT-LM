parser grammar AntlrJsonParser;

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
    tokenVocab = AntlrJsonLexer; // Use the tokens defined in AntlrJsonLexer.g4
}

// The root of a JSON document can be either an object or an array.
json: functionCall | functionCallList;

// Represents any valid JSON value.
value
    : STRING
    | NUMBER
    | object // Recursively allows objects as values
    | array  // Recursively allows arrays as values
    | BOOLEAN
    | NONE
    ;

// Defines a JSON object: { stringKey : value, ... }
object
    : OPEN_BRACE pair (COMMA pair)* CLOSE_BRACE
    | OPEN_BRACE CLOSE_BRACE // Empty object
    ;

// Defines a key-value pair within an object.
pair: STRING COLON value;

// Defines a JSON array: [ value, ... ]
array
    : OPEN_BRACKET value (COMMA value)* CLOSE_BRACKET
    | OPEN_BRACKET CLOSE_BRACKET // Empty array
    ;

functionNamePair: FUNCTION_NAME COLON STRING;
functionArgsPair: FUNCTION_ARGUMENTS COLON object;

emptyFunctionCall: OPEN_BRACE CLOSE_BRACE;
fullFunctionCall: OPEN_BRACE functionNamePair COMMA functionArgsPair CLOSE_BRACE;

functionCall
  : fullFunctionCall
  | emptyFunctionCall
  ;

functionCallList
  : OPEN_BRACKET functionCall (COMMA functionCall)* CLOSE_BRACKET
  | OPEN_BRACKET CLOSE_BRACKET
  ;
