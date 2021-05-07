export function assert(cond: boolean, msg?: string): asserts cond {
  if (!cond) {
    throw Error(msg || "assert failed");
  }
}

export function bufferToArrayBuffer(b: Buffer): ArrayBuffer {
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
