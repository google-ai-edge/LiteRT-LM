lexer grammar AntlrPythonLexer;

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

EQ: '=';
COLON: ':';
SEP: ',';
OPEN_PAR: '(';
CLOSE_PAR: ')';
OPEN_BRACE: '{';
CLOSE_BRACE: '}';
LIST_OPEN: '[';
LIST_CLOSE: ']';

BOOL: 'True' | 'False';
INT: '-'? [0-9]+;
FLOAT: '-'? [0-9]+[.][0-9]* | '-'? [0-9]*[.][0-9]+;
STRING : '"' ( ~["\\] | [\\]. )* '"'
       | '\'' ( ~['\\] | [\\]. )* '\''
       ;
NONE: 'None';

NAME: [a-zA-Z_][a-zA-Z0-9_]*;

WS: [ \t\n\r]+ -> skip;
