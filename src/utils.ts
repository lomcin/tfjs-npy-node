export function assert(cond: boolean, msg?: string): asserts cond {
  if (!cond) {
    throw Error(msg || "assert failed");
  }
}

/**
 * Converts a Node.js `Buffer` to a JS `ArrayBuffer`.
 *
 * Note that this should always be used over `Buffer.buffer`, because the
 * `ArrayBuffer` underlying a `Buffer` can have a byte offset and byte length,
 * which would not be captured by just reading the `ArrayBuffer`.
 */
export function bufferToArrayBuffer(b: Buffer): ArrayBuffer {
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
