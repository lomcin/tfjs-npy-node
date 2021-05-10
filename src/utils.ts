export function assert(cond: boolean, msg?: string): asserts cond {
  if (!cond) {
    throw Error(msg || "assert failed");
  }
}
