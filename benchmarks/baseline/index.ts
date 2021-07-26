import * as tf from "@tensorflow/tfjs-node";
import { gzipSync, unzipSync } from "zlib";

export interface SerializedTensor {
  readonly dtype: tf.DataType;
  readonly shape: number[];
  readonly data: number[];
}

export async function serialize(
  tensor: tf.Tensor,
  compress: boolean,
): Promise<Buffer> {
  return doSerialize(tensor, await tensor.data(), compress);
}

export async function serializeArray(
  tensors: tf.Tensor[],
  compress: boolean,
): Promise<Buffer> {
  return toBuffer(
    await Promise.all(tensors.map(async (t) => toJson(t, await t.data()))),
    compress,
  );
}

export function serializeSync(tensor: tf.Tensor, compress: boolean): Buffer {
  return doSerialize(tensor, tensor.dataSync(), compress);
}

export function serializeArraySync(
  tensors: tf.Tensor[],
  compress: boolean,
): Buffer {
  return toBuffer(
    tensors.map((t) => toJson(t, t.dataSync())),
    compress,
  );
}

function doSerialize(
  tensor: tf.Tensor,
  data: tf.TypedArray,
  compress: boolean,
): Buffer {
  return toBuffer(toJson(tensor, data), compress);
}

function toBuffer(
  serialized: SerializedTensor | SerializedTensor[],
  compress: boolean,
): Buffer {
  const json = JSON.stringify(serialized);
  return compress ? gzipSync(json) : Buffer.from(json);
}

function toJson(tensor: tf.Tensor, data: tf.TypedArray): SerializedTensor {
  return {
    dtype: tensor.dtype,
    shape: tensor.shape,
    data: Array.from(data),
  };
}

export function deserialize(tensor: Buffer): tf.Tensor {
  const json = unzipSync(tensor).toString("utf-8");
  const serialized = JSON.parse(json) as SerializedTensor;
  return tf.tensor(
    serializedToArray(serialized),
    serialized.shape,
    serialized.dtype,
  );
}

function serializedToArray(serialized: SerializedTensor): tf.TypedArray {
  const { dtype, shape, data } = serialized;
  const numberElements = shape.reduce((x, y) => x * y, 1);
  if (data.length !== numberElements) {
    throw new Error(
      `Expected to get ${numberElements} elements, but got ${data.length}`,
    );
  }
  switch (dtype) {
    case "float32":
      return Float32Array.from(data);
    case "int32":
      return Int32Array.from(data);
    case "bool":
      return Uint8Array.from(data);
    default:
      throw new Error(
        `Deserialization of dtype "${dtype}" is not implemented yet`,
      );
  }
}
