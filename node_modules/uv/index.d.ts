/**
 * Validate UTF-8 data in the buffer
 */
declare function uv(buffer: Buffer | Uint8Array, start?: number, end?: number): boolean;

export = uv;