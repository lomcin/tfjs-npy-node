import { assert } from "chai";
import * as tf from "@tensorflow/tfjs-node";
import { npz } from "../src";
import { readFileSync } from "fs";
import { bufferToArrayBuffer } from "../src/utils";

describe("npz.parse", () => {
  async function loadz(fn: string): Promise<tf.Tensor[]> {
    const b = readFileSync(__dirname + "/data/" + fn, null);
    const ab = bufferToArrayBuffer(b);
    return npz.parse(ab);
  }

  it("parses 1.npz correctly", async () => {
    const ts = await loadz("1.npz");
    assert.strictEqual(ts.length, 2);
    assert.deepStrictEqual(ts[0].shape, [2]);
    assert.deepStrictEqual(ts[0].arraySync(), [1.5, 2.5]);
    assert.deepStrictEqual(ts[1].shape, [2]);
    assert.deepStrictEqual(ts[1].arraySync(), [3.5, 4.5]);
  });
});

describe("npz.serialize", () => {
  it("serializes to a parseable representation", async () => {
    const tensors = [tf.tensor([1.5, 2.5]), tf.tensor([3.5, 4.5])];
    const ab = await npz.serialize(tensors);
    const tt = npz.parse(ab);
    assert.strictEqual(tt.length, 2);
    assert.deepStrictEqual(tt[0].arraySync(), tensors[0].arraySync());
    assert.deepStrictEqual(tt[1].arraySync(), tensors[1].arraySync());
  });
});
