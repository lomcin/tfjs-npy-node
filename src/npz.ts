import * as tf from "@tensorflow/tfjs-core";
import * as AdmZip from "adm-zip";
import { promisify } from "util";
import * as npy from "./npy";
import { assert, bufferToArrayBuffer } from "./utils";

/** Serializes multiple tensors into npz file contents. */
export async function serialize(tensors: tf.Tensor[]): Promise<ArrayBuffer> {
  const zip = new AdmZip();
  for (let i = 0; i < tensors.length; ++i) {
    const buffer = await npy.serialize(tensors[i]);
    zip.addFile(`arr_${i}`, Buffer.from(buffer));
  }
  return bufferToArrayBuffer(zip.toBuffer());
}

/** Serializes multiple tensors into npz file contents, synchronously. */
export function serializeSync(tensors: tf.Tensor[]): ArrayBuffer {
  return bufferToArrayBuffer(doSerialize(tensors).toBuffer());
}

/** Save a .npz file to disk */
export function save(filepath: string, tensors: tf.Tensor[]): Promise<void> {
  assert(filepath.endsWith(".npz"));
  const zip = doSerialize(tensors);
  return promisify(zip.writeZip)(filepath);
}

function doSerialize(tensors: tf.Tensor[]): AdmZip {
  const zip = new AdmZip();
  for (let i = 0; i < tensors.length; ++i) {
    const buffer = npy.serializeSync(tensors[i]);
    zip.addFile(`arr_${i}`, Buffer.from(buffer));
  }
  return zip;
}

/** Load a .npz file from disk. */
export function load(filepath: string): tf.Tensor[] {
  assert(
    filepath.endsWith(".npz"),
    `Expected provided filepath (${filepath}) to have file extension .npz`,
  );
  try {
    return doParse(filepath);
  } catch (err) {
    throw new Error(`Could not load ${filepath}: ` + (err as Error).message);
  }
}

/** Parse the contents of an npz file. */
export function parse(buf: Buffer | ArrayBuffer | tf.TypedArray): tf.Tensor[] {
  return doParse(buf instanceof Buffer ? buf : Buffer.from(buf));
}

export function parseToNpzData(
  buf: Buffer | ArrayBuffer | tf.TypedArray,
): npy.NpyData[] {
  const zip = new AdmZip(buf instanceof Buffer ? buf : Buffer.from(buf));
  return zip.getEntries().map((entry) => npy.parseToNpyData(entry.getData()));
}

function doParse(filenameOrData: string | Buffer): tf.Tensor[] {
  const zip = new AdmZip(filenameOrData);
  return zip.getEntries().map((entry) => npy.parse(entry.getData()));
}
