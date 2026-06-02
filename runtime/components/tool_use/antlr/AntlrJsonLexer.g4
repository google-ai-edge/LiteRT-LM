lexer grammar AntlrJsonLexer;

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

// Structural Characters
OPEN_BRACE : '{';
CLOSE_BRACE : '}';
OPEN_BRACKET : '[';
CLOSE_BRACKET : ']';
COMMA : ',';
COLON : ':';

// Literals
BOOLEAN : 'true' | 'false';
NONE : 'null';

// Literals function name key
FUNCTION_NAME: '"' 'name' '"' ;

// Literals arguments key
FUNCTION_ARGUMENTS: '"' ( 'args' | 'arguments' ) '"' ;

// String: Double-quoted with escape sequences
STRING : '"' ( ESC | ~["\\] )* '"';

fragment ESC : '\\' ( ["\\/bfnrt] | UNICODE );
fragment UNICODE : 'u' HEX HEX HEX HEX;
fragment HEX : [0-9a-fA-F];

// Number: Integer and floating-point, including exponents
NUMBER : '-'? INT ( FRAC | EXP )? | '-'? FRAC | '-'? EXP ;

fragment INT : '0' | [1-9] [0-9]*;
fragment FRAC : '.' [0-9]+;
fragment EXP : [eE] [+-]? [0-9]+;

// Whitespace: Skipped
WS : [ \t\n\r]+ -> skip;
