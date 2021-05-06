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

// This module saves and loads from the numpy format.
// https://docs.scipy.org/doc/numpy/neps/npy-format.html

import * as tf from "@tensorflow/tfjs-core";
import type { TypedArray } from "@tensorflow/tfjs-core";

/** Serializes a tensor into a npy file contents. */
export async function serialize(tensor: tf.Tensor): Promise<ArrayBuffer> {
  return doSerialize(tensor, await tensor.data());
}

/** Serializes a tensor into npy file contents synchronously. */
export function serializeSync(tensor: tf.Tensor): ArrayBuffer {
  return doSerialize(tensor, tensor.dataSync());
}

const MAGIC_STRING: string = "\x93NUMPY" as const;

interface DescrInfo {
  bytes: number;
  dtype: tf.DataType;
  /** Function for creating a typed array. */
  createArray: (buf: ArrayBuffer) => tf.TypedArray;
  /** Function for writing into a view. Undefined if serialization is not supported. */
  write?: (view: DataView, pos: number, byte: number) => void;
}

type SupportedDescr = "<f8" | "<f4" | "<i8" | "<i4" | "|u1" | "|b1";

const numpyDescrInfo: Readonly<Record<SupportedDescr, DescrInfo>> = {
  "<f8": {
    bytes: 8,
    dtype: "float32", // downcast to float32
    createArray: (buf) => new Float32Array(new Float64Array(buf)),
  },
  "<f4": {
    bytes: 4,
    dtype: "float32",
    createArray: (buf) => new Float32Array(buf),
    write: (view, pos, byte) => view.setFloat32(pos, byte, true),
  },
  "<i8": {
    bytes: 8,
    dtype: "int32", // downcast to int32
    createArray: (buf) => new Int32Array(buf).filter((val, i) => i % 2 === 0),
  },
  "<i4": {
    bytes: 4,
    dtype: "int32",
    createArray: (buf) => new Int32Array(buf),
    write: (view, pos, byte) => view.setInt32(pos, byte, true),
  },
  "|b1": {
    bytes: 1,
    dtype: "bool",
    createArray: (buf) => new Uint8Array(buf),
    write: (view, pos, byte) => view.setUint8(pos, byte),
  },
  "|u1": {
    bytes: 1,
    dtype: "int32", // FIXME: should be uint8
    createArray: (buf) => new Uint8Array(buf),
    write: (view, pos, byte) => view.setUint8(pos, byte),
  },
};

const tfDtypeToNumpyDescr: ReadonlyMap<tf.DataType, SupportedDescr> = new Map([
  ["float32", "<f4"],
  ["int32", "<i4"],
  ["bool", "|b1"],
]);

function doSerialize(tensor: tf.Tensor, data: TypedArray): ArrayBuffer {
  // Generate header
  const descr = tfDtypeToNumpyDescr.get(tensor.dtype);
  assert(
    descr !== undefined,
    `Tensors of dtype '${tensor.dtype}' not supported yet.`,
  );
  const versionStr = "\x01\x00"; // version 1.0
  const shapeStr = tensor.shape.join(",") + ",";
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${shapeStr}), }`;

  // Figure out how long the file is going to be so we can create the
  // output ArrayBuffer.
  const unpaddedLength =
    MAGIC_STRING.length + versionStr.length + 2 + header.length;
  // Spaces to 16-bit align.
  const padding = " ".repeat((16 - (unpaddedLength % 16)) % 16);
  header += padding;
  assertEqual((unpaddedLength + padding.length) % 16, 0);
  // Number of bytes is in the Numpy descr
  const bytesPerElement = Number.parseInt(descr[2]);
  assert(new Set([1, 2, 4, 8]).has(bytesPerElement));
  const dataLen = bytesPerElement * numEls(tensor.shape);
  const totalSize = unpaddedLength + padding.length + dataLen;

  const ab = new ArrayBuffer(totalSize);
  const view = new DataView(ab);
  let pos = 0;

  // Write magic string and version.
  pos = writeStrToDataView(view, MAGIC_STRING + versionStr, pos);

  // Write header length and header.
  view.setUint16(pos, header.length, true);
  pos += 2;
  pos = writeStrToDataView(view, header, pos);

  // Write data
  const write = numpyDescrInfo[descr].write;
  assert(write !== undefined, `dtype ${tensor.dtype} not yet supported.`);
  for (let i = 0; i < data.length; i++) {
    write(view, pos, data[i]);
    pos += bytesPerElement;
  }
  return ab;
}

/** Parses an ArrayBuffer containing a npy file. Returns a tensor. */
export function parse(ab: ArrayBuffer): tf.Tensor {
  assert(ab.byteLength > MAGIC_STRING.length);
  const view = new DataView(ab);
  let pos = 0;

  // First parse the magic string.
  const magicStr = dataViewToAscii(new DataView(ab, pos, MAGIC_STRING.length));
  if (magicStr !== MAGIC_STRING) {
    throw Error("Not a numpy file.");
  }
  pos += MAGIC_STRING.length;

  // Parse the version
  const version = [view.getUint8(pos++), view.getUint8(pos++)].join(".");
  if (version !== "1.0") {
    throw Error(`Unsupported npy version ${version}.`);
  }

  // Parse the header length.
  const headerLen = view.getUint16(pos, true);
  pos += 2;

  // Parse the header.
  // Header is almost json, so we just manipulated it until it is.
  // Example: {'descr': '<f8', 'fortran_order': False, 'shape': (1, 2), }
  const headerPy = dataViewToAscii(new DataView(ab, pos, headerLen));
  pos += headerLen;
  const bytesLeft = view.byteLength - pos;
  const headerJson = headerPy
    .replace("True", "true")
    .replace("False", "false")
    .replace(/'/g, `"`)
    .replace(/,\s*}/, " }")
    .replace(/,?\)/, "]")
    .replace("(", "[");
  const header = JSON.parse(headerJson);
  if (header.fortran_order) {
    throw Error("NPY parse error. Implement me.");
  }

  // Parse shape
  const shape = header.shape;
  assert(Array.isArray(shape));
  assert(shape.every((el) => typeof el === "number"));
  const size = numEls(shape);

  // Parse descr
  const descr = header.descr;
  assert(typeof descr === "string");
  const info = numpyDescrInfo[descr as SupportedDescr];
  assert(info !== undefined, `Unknown dtype "${descr}". Implement me.`);
  const bytesPerElement = info.bytes;
  assert(
    bytesLeft === size * bytesPerElement,
    `Expected there to be ${
      size * bytesPerElement
    } bytes left for npy file of dtype descr ${descr}, but there were ${bytesLeft} bytes left`,
  );

  // Finally parse the actual data.
  const slice = ab.slice(pos, pos + size * bytesPerElement);
  const typedArray = info.createArray(slice);
  return tf.tensor(typedArray, shape, info.dtype);
}

function numEls(shape: number[]): number {
  return shape.reduce((a: number, b: number) => a * b, 1);
}

function writeStrToDataView(view: DataView, str: string, pos: number) {
  for (let i = 0; i < str.length; i++) {
    view.setInt8(pos + i, str.charCodeAt(i));
  }
  return pos + str.length;
}

function assertEqual(actual: number, expected: number) {
  assert(
    actual === expected,
    `actual ${actual} not equal to expected ${expected}`,
  );
}

function assert(cond: boolean, msg?: string): asserts cond {
  if (!cond) {
    throw Error(msg || "assert failed");
  }
}

function dataViewToAscii(dv: DataView): string {
  let out = "";
  for (let i = 0; i < dv.byteLength; i++) {
    const val = dv.getUint8(i);
    if (val === 0) {
      break;
    }
    out += String.fromCharCode(val);
  }
  return out;
}
