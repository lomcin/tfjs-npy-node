/*!
Copyright 2018 Propel http://propel.site/.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import { assert } from "chai";
import * as tf from "@tensorflow/tfjs-node";
import * as npy from "../src/npy";
import { readFileSync } from "fs";
const { expectArraysClose } = tf.test_util;

function bufferToArrayBuffer(b: Buffer): ArrayBuffer {
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

describe("parse", () => {
  async function load(fn: string): Promise<tf.Tensor> {
    const b = readFileSync(__dirname + "/data/" + fn, null);
    const ab = bufferToArrayBuffer(b);
    return npy.parse(ab);
  }

  it("parses 1.npy correctly", async () => {
    const t = await load("1.npy");
    assert.deepStrictEqual(await t.array(), [1.5, 2.5]);
    assert.deepStrictEqual(t.shape, [2]);
    assert.strictEqual(t.dtype, "float32");
  });

  it("parses 2.npy correctly", async () => {
    const t = await load("2.npy");
    assert.deepStrictEqual(await t.array(), [
      [1.5, 43],
      [13, 2.5],
    ]);
    assert.deepStrictEqual(t.shape, [2, 2]);
    assert.strictEqual(t.dtype, "float32");
  });

  it("parses 3.npy correctly", async () => {
    const t = await load("3.npy");
    assert.deepStrictEqual(await t.array(), [
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
    ]);
    assert.deepStrictEqual(t.shape, [1, 2, 3]);
    assert.strictEqual(t.dtype, "int32");
  });

  it("parses 4.npy correctly", async () => {
    const t = await load("4.npy");
    expectArraysClose(t.dataSync(), new Float32Array([0.1, 0.2]));
    assert.deepStrictEqual(t.shape, [2]);
    assert.strictEqual(t.dtype, "float32");
  });

  it("parses uint8.npy correctly", async () => {
    const t = await load("uint8.npy");
    expectArraysClose(t.dataSync(), new Int32Array([0, 127]));
    assert.deepStrictEqual(t.shape, [2]);
    assert.strictEqual(t.dtype, "int32"); // TODO uint8
  });
});

describe("parzeNpz", () => {
  async function loadz(fn: string): Promise<tf.Tensor[]> {
    const b = readFileSync(__dirname + "/data/" + fn, null);
    const ab = bufferToArrayBuffer(b);
    return npy.parseNpz(ab);
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

describe("serialize", () => {
  it("serializes to a parseable representation", async () => {
    const t = tf.tensor([1.5, 2.5]);
    const ab = await npy.serialize(t);
    // Now try to parse it.
    const tt = npy.parse(ab);
    const [actual, expected] = await Promise.all([t.data(), tt.data()]);
    expectArraysClose(actual, expected);
  });
});

describe("serializeNpz", () => {
  it("serializes to a parseable representation", async () => {
    const tensors = [tf.tensor([1.5, 2.5]), tf.tensor([3.5, 4.5])];
    const ab = await npy.serializeNpz(tensors);
    const tt = npy.parseNpz(ab);
    assert.strictEqual(tt.length, 2);
    assert.deepStrictEqual(tt[0].arraySync(), tensors[0].arraySync());
    assert.deepStrictEqual(tt[1].arraySync(), tensors[1].arraySync());
  });
});
