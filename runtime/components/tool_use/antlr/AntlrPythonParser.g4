parser grammar AntlrPythonParser;

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
    tokenVocab = AntlrPythonLexer;
}

main: expr EOF;
expr: functionCall+ | functionCallList;

key: STRING;
value: INT | FLOAT | BOOL | STRING | NONE | list | dict | object;

list: LIST_OPEN (value (SEP value)* SEP?)? LIST_CLOSE;

dict
  : OPEN_BRACE (key COLON value (SEP key COLON value)* SEP?)? CLOSE_BRACE
  ;

argVal: NAME EQ value;
argValExpr: argVal (SEP argVal)* SEP?;
object
  : NAME OPEN_PAR CLOSE_PAR
  | NAME OPEN_PAR argValExpr CLOSE_PAR
  ;

emptyFunctionCall: NAME OPEN_PAR CLOSE_PAR;
fullFunctionCall: NAME OPEN_PAR argValExpr CLOSE_PAR;

functionCall
  : fullFunctionCall
  | emptyFunctionCall
  ;

functionCallList
  : LIST_OPEN functionCall (SEP functionCall)* SEP? LIST_CLOSE
  | LIST_OPEN LIST_CLOSE;