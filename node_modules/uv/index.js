'use strict';

// Below is the deterministic finite automaton used for UTF-8 data validation,
// it is added here for easier understanding of the algorithm used in the
// exported function. The validation process should start in the state "a" and
// should end in the same state to have a valid UTF-8 sequence, in any other
// case an invalid bytes sequence is received.
//
// +---------+
// |         |-----------+
// |         |           | 00..7F
// |         |<----------+
// |         |                        C2..DF                         +---------+
// |         |------------------------------------------------------>|         |
// |         |                                                       |         |
// |         |           +-------------------------------+           |         |
// |         |    E0     |                               |  A0..BF   |         |
// |         |---------->|               c               |---------->|         |
// |         |           |                               |           |         |
// |         |           +-------------------------------+           |         |
// |         |                                                       |         |
// |         |         E1..EC, EE, EF          +---------+           |         |
// |         |-------------------------------->|         |           |         |
// |         |                                 |         |           |         |
// |         |           +---------+           |         |           |         |
// |         |    F0     |         |  90..BF   |         |           |         |
// |         |---------->|    f    |---------->|         |           |         |
// |         |           |         |           |         |           |         |
// |         |           +---------+           |         |           |         |
// |         |                                 |         |           |         |
// |         |           +---------+           |         |           |         |
// |         |  F1..F3   |         |  80..BF   |         |  80..BF   |         |
// |    a    |---------->|    g    |---------->|    d    |---------->|    b    |
// |         |           |         |           |         |           |         |
// |         |           +---------+           |         |           |         |
// |         |                                 |         |           |         |
// |         |           +---------+           |         |           |         |
// |         |    F4     |         |  80..8F   |         |           |         |
// |         |---------->|    h    |---------->|         |           |         |
// |         |           |         |           |         |           |         |
// |         |           +---------+           +---------+           |         |
// |         |                                                       |         |
// |         |           +-------------------------------+           |         |
// |         |    ED     |                               |  80..9F   |         |
// |         |---------->|               e               |---------->|         |
// |         |           |                               |           |         |
// |         |           +-------------------------------+           |         |
// |         |                                                       |         |
// |         |                        80..BF                         |         |
// |         |<------------------------------------------------------|         |
// +---------+                                                       +---------+

/**
 * @typedef {Buffer | Uint8Array} BufferData
 * @typedef {(byte: number) => boolean} SingleByteValidation
 * @typedef {(buffer: BufferData, index: number) => boolean} MultiByteValidation
 */

/**
 * Validate 00..7F bytes
 * @type {SingleByteValidation}
 */
const aa = (byte) => (byte < 0x80);

/**
 * Validate C2..DF bytes
 * @type {SingleByteValidation}
 */
const ab = (byte) => (byte > 0xC1 && byte < 0xE0);

/**
 * Validate E0 byte
 * @type {SingleByteValidation}
 */
const ac = (byte) => (byte === 0xE0);

/**
 * Validate E1..EC, EE, EF bytes
 * @type {SingleByteValidation}
 */
const ad = (byte) => ((byte & 0xF0) === 0xE0 && !ac(byte) && !ae(byte));

/**
 * Validate ED byte
 * @type {SingleByteValidation}
 */
const ae = (byte) => (byte === 0xED);

/**
 * Validate F0 byte
 * @type {SingleByteValidation}
 */
const af = (byte) => (byte === 0xF0);

/**
 * Validate F1..F3 bytes
 * @type {SingleByteValidation}
 */
const ag = (byte) => (byte > 0xF0 && byte < 0xF4);

/**
 * Validate F4 byte
 * @type {SingleByteValidation}
 */
const ah = (byte) => (byte === 0xF4);

/**
 * Validate 80..BF bytes
 * @type {SingleByteValidation}
 */
const ba = (byte) => ((byte & 0xC0) === 0x80);

/**
 * Validate A0..BF bytes
 * @type {SingleByteValidation}
 */
const cb = (byte) => ((byte & 0xE0) === 0xA0);

/**
 * Validate 80..9F bytes
 * @type {SingleByteValidation}
 */
const eb = (byte) => ((byte & 0xE0) === 0x80);

/**
 * Validate 90..BF bytes
 * @type {SingleByteValidation}
 */
const fd = (byte) => (byte > 0x8F && byte < 0xC0);

/**
 * Validate 80..8F bytes
 * @type {SingleByteValidation}
 */
const hd = (byte) => ((byte & 0xF0) === 0x80);

/**
 * Validate A0..BF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const ca = (b, i) => (cb(b[i]) && ba(b[i + 1]));

/**
 * Validate 80..BF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const da = (b, i) => (ba(b[i]) && ba(b[i + 1]));

/**
 * Validate 80..9F -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const ea = (b, i) => (eb(b[i]) && ba(b[i + 1]));

/**
 * Validate 90..BF -> 80..BF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const fa = (b, i) => (fd(b[i]) && da(b, i + 1));

/**
 * Validate 80..BF -> 80..BF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const ga = (b, i) => (ba(b[i]) && da(b, i + 1));

/**
 * Validate 80..8F -> 80..BF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const ha = (b, i) => (hd(b[i]) && da(b, i + 1));

/**
 * Validate C2..DF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const aba = (b, i) => (ab(b[i]) && ba(b[i + 1]));

/**
 * Validate E0 -> A0..BF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const aca = (b, i) => (ac(b[i]) && ca(b, i + 1));

/**
 * Validate E1..EC, EE, EF -> 80..BF -> 80..BF byte sequence
 * @type {MultiByteValidation}
 */
const ada = (b, i) => (ad(b[i]) && da(b, i + 1));

/**
 * Validate a 00..7F sequence of 4 bytes
 * @type {MultiByteValidation}
 */
const aam = (b, i) => aa(b[i] | b[i + 1] | b[i + 2] | b[i + 3]);

/**
 * Validate two C2..DF -> 80..BF sequences
 * @type {MultiByteValidation}
 */
const abm = (b, i) => (aba(b, i) && aba(b, i + 2));

/**
 * Validate trailing bytes in buffer
 * @param {BufferData} buffer
 * @param {number} index
 * @param {number} length
 * @returns {boolean}
 */
const vt = (buffer, index, length) => {

	// Check for one byte to validate
	if (length === 1) {

		// a -> a
		return aa(buffer[index]);
	}

	// Check for two bytes to validate
	if (length === 2) {

		// a -> a -> a
		if (aa(buffer[index] | buffer[index + 1])) {
			return true;
		}

		// a -> b -> a
		return aba(buffer, index);
	}

	// a -> a
	if (aa(buffer[index])) {

		// a -> a -> a -> a
		if (aa(buffer[index + 1] | buffer[index + 2])) {
			return true;
		}

		// a -> a -> b -> a
		return aba(buffer, index + 1);
	}

	// a -> b
	if (ab(buffer[index])) {

		// a -> b -> a -> a
		return ba(buffer[index + 1]) && aa(buffer[index + 2]);
	}

	// a -> c
	if (ac(buffer[index])) {

		// a -> c -> b -> a
		return ca(buffer, index + 1);
	}

	// a -> d
	if (ad(buffer[index])) {

		// a -> d -> b -> a
		return da(buffer, index + 1);
	}

	// a -> e -> b -> a
	return ae(buffer[index]) && ea(buffer, index + 1);
};

/**
 * Validate UTF-8 data in the buffer
 * @param {BufferData} buffer
 * @param {number} [start]
 * @param {number} [end]
 * @returns {boolean}
 */
module.exports = (buffer, start = 0, end = buffer.length) => {

	const stop = end - 3;

	let index = start;

	// Loop through all buffer bytes
	while (index < stop) {

		// Cache first byte value
		const byte = buffer[index];

		// a -> a
		if (aa(byte)) {
			index++;

			// Optimize for 4 consecutive ASCII bytes
			while (index < stop && aam(buffer, index)) {
				index += 4;
			}

		// a -> b
		} else if (ab(byte)) {

			// a -> b -> x
			if (!ba(buffer[index + 1])) {
				return false;
			}

			// a -> b -> a
			index += 2;

			// Optimize for repeated 2 bytes sequence
			while (index < stop && abm(buffer, index)) {
				index += 4;
			}

		// a -> c
		} else if (ac(byte)) {

			// a -> c -> x
			if (!ca(buffer, index + 1)) {
				return false;
			}

			// a -> c -> b -> a
			index += 3;

			// Optimize for repeated 3 bytes sequence (a -> c -> ...)
			while (index < stop && aca(buffer, index)) {
				index += 3;
			}

		// a -> d
		} else if (ad(byte)) {

			// a -> d -> x
			if (!da(buffer, index + 1)) {
				return false;
			}

			// a -> d -> b -> a
			index += 3;

			// Optimize for repeated 3 bytes sequence (a -> d -> ...)
			while (index < stop && ada(buffer, index)) {
				index += 3;
			}

		// a -> e
		} else if (ae(byte)) {

			// a -> e -> x
			if (!ea(buffer, index + 1)) {
				return false;
			}

			// a -> e -> b -> a
			index += 3;

		// a -> f
		} else if (af(byte)) {

			// a -> f -> x
			if (!fa(buffer, index + 1)) {
				return false;
			}

			// a -> f -> d -> b -> a
			index += 4;

		// a -> g
		} else if (ag(byte)) {

			// a -> g -> x
			if (!ga(buffer, index + 1)) {
				return false;
			}

			// a -> g -> d -> b -> a
			index += 4;

		// a -> h
		} else if (ah(buffer[index]) && ha(buffer, index + 1)) {

			// a -> h -> d -> b -> a
			index += 4;

		// x
		} else {
			return false;
		}
	}

	// Check for trailing bytes that were not covered in the previous loop
	if (index < end) {
		return vt(buffer, index, end - index);
	}

	return true;
};