import { assert } from "chai";
import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as path from "path";
import { npz } from "../src";

describe("npz.load", () => {
  async function loadz(fn: string): Promise<tf.Tensor[]> {
    return npz.load(path.join(__dirname, "data", fn));
  }

  it("parses 1.npz correctly", async () => {
    const tensors = await loadz("1.npz");
    assert.strictEqual(tensors.length, 2);
    assert.deepStrictEqual(tensors[0].shape, [2]);
    assert.deepStrictEqual(tensors[0].arraySync(), [1.5, 2.5]);
    assert.deepStrictEqual(tensors[1].shape, [2]);
    assert.deepStrictEqual(tensors[1].arraySync(), [3.5, 4.5]);
  });
});

describe("npz.parse", () => {
  async function load(filename: string): Promise<Buffer> {
    return fs.promises.readFile(path.join(__dirname, "data", filename));
  }

  it("parses from an ArrayBuffer", async () => {
    const buf = await load("1.npz");
    const ab: ArrayBuffer = buf.buffer.slice(
      buf.byteOffset,
      buf.byteOffset + buf.byteLength,
    );
    const tensors = npz.parse(ab);
    assert.strictEqual(tensors.length, 2);
    assert.deepStrictEqual(tensors[0].shape, [2]);
    assert.deepStrictEqual(tensors[0].arraySync(), [1.5, 2.5]);
    assert.deepStrictEqual(tensors[1].shape, [2]);
    assert.deepStrictEqual(tensors[1].arraySync(), [3.5, 4.5]);
  });

  it("parses from a Buffer", async () => {
    const buf = await load("1.npz");
    const tensors = npz.parse(buf);
    assert.strictEqual(tensors.length, 2);
    assert.deepStrictEqual(tensors[0].shape, [2]);
    assert.deepStrictEqual(tensors[0].arraySync(), [1.5, 2.5]);
    assert.deepStrictEqual(tensors[1].shape, [2]);
    assert.deepStrictEqual(tensors[1].arraySync(), [3.5, 4.5]);
  });

  it("parses from a Uint8Array", async () => {
    const buf = await load("1.npz");
    const array = Uint8Array.from(buf);
    const tensors = npz.parse(array);
    assert.strictEqual(tensors.length, 2);
    assert.deepStrictEqual(tensors[0].shape, [2]);
    assert.deepStrictEqual(tensors[0].arraySync(), [1.5, 2.5]);
    assert.deepStrictEqual(tensors[1].shape, [2]);
    assert.deepStrictEqual(tensors[1].arraySync(), [3.5, 4.5]);
  });
});

describe("npz.parseToNpzData", () => {
  async function load(filename: string): Promise<Buffer> {
    return fs.promises.readFile(path.join(__dirname, "data", filename));
  }

  it("parses from an ArrayBuffer", async () => {
    const buf = await load("1.npz");
    const ab: ArrayBuffer = buf.buffer.slice(
      buf.byteOffset,
      buf.byteOffset + buf.byteLength,
    );
    const data = npz.parseToNpzData(ab);
    assert.strictEqual(data.length, 2);
    assert.deepStrictEqual(data[0].shape, [2]);
    assert.deepStrictEqual(data[1].shape, [2]);
  });

  it("parses from a Buffer", async () => {
    const buf = await load("1.npz");
    const data = npz.parseToNpzData(buf);
    assert.strictEqual(data.length, 2);
    assert.deepStrictEqual(data[0].shape, [2]);
    assert.deepStrictEqual(data[1].shape, [2]);
  });

  it("parses from a Uint8Array", async () => {
    const buf = await load("1.npz");
    const array = Uint8Array.from(buf);
    const data = npz.parseToNpzData(array);
    assert.strictEqual(data.length, 2);
    assert.deepStrictEqual(data[0].shape, [2]);
    assert.deepStrictEqual(data[1].shape, [2]);
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
