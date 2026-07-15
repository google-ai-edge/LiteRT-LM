# uv

[![npm version](https://img.shields.io/npm/v/uv.svg?logo=npm&style=flat-square)](https://www.npmjs.com/package/uv)
![npm downloads](https://img.shields.io/npm/dm/uv.svg?style=flat-square)
![npm types](https://img.shields.io/npm/types/uv.svg?style=flat-square)
![node version](https://img.shields.io/node/v/uv.svg?style=flat-square)
![license](https://img.shields.io/npm/l/uv.svg?style=flat-square)

Ultrafast UTF-8 data validation

## Installation

`npm i uv`

## Usage

`uv(buffer: Buffer | Uint8Array, start?: number, end?: number): boolean`

`uv` exports a function that accepts a `buffer` as an argument and validate it
for UTF-8 data, optionally it accepts two more arguments for slice validation as
`start` and `end` indexes, which are zero-based values that cannot be negative
numbers.

```js
const uv = require('uv');

const someBuffer = Buffer.from('Some UTF-8 data');

uv(someBuffer); // => true

uv(Buffer.from([0xFF, 0x00, 0x00, 0x00, 0xFF])); // => false

//               0     1     2     3     4
uv(Buffer.from([0xFF, 0x00, 0x00, 0x00, 0xFF]), 1, 4); // => true
//                   |                 |        |  |
//                   +------start------|--------+  |
//                                     +----end----+
```

## Comparison with other UTF-8 validation packages

This module is a pure JavaScript implementation of UTF-8 validation, in most
cases it is faster than other alternatives, performance may vary based on OS,
node version, CPU or input size, please run the
[`bench.js`](https://github.com/micnic/uv/blob/master/bench.js) file from the
github repo for a benchmark on your machine to compare it with other UTF-8
validation implementations.