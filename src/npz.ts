import * as tf from "@tensorflow/tfjs-core";
import * as AdmZip from "adm-zip";
import { promisify } from "util";
import * as npy from "./npy";
import { assert, bufferToArrayBuffer } from "./utils";

/** Serializes multiple tensors into npz file contents. */
export async function serializeNpz(tensors: tf.Tensor[]): Promise<ArrayBuffer> {
  const zip = new AdmZip();
  for (let i = 0; i < tensors.length; ++i) {
    const buffer = await npy.serialize(tensors[i]);
    zip.addFile(`arr_${i}`, Buffer.from(buffer));
  }
  return zip.toBuffer().buffer;
}

/** Serializes multiple tensors into npz file contents, synchronously. */
export function serializeNpzSync(tensors: tf.Tensor[]): ArrayBuffer {
  return doSerializeNpz(tensors).toBuffer().buffer;
}

/** Save a .npz file to disk */
export function savez(filepath: string, tensors: tf.Tensor[]): Promise<void> {
  assert(filepath.endsWith(".npz"));
  const zip = doSerializeNpz(tensors);
  return promisify(zip.writeZip)(filepath);
}

function doSerializeNpz(tensors: tf.Tensor[]): AdmZip {
  const zip = new AdmZip();
  for (let i = 0; i < tensors.length; ++i) {
    const buffer = npy.serializeSync(tensors[i]);
    zip.addFile(`arr_${i}`, Buffer.from(buffer));
  }
  return zip;
}

/** Load a .npz file from disk. */
export function loadz(filepath: string): tf.Tensor[] {
  assert(filepath.endsWith(".npz"));
  return doParseNpz(filepath);
}

/** Parse the contents of an npz file. */
export function parseNpz(ab: ArrayBuffer): tf.Tensor[] {
  return doParseNpz(Buffer.from(ab));
}

function doParseNpz(filenameOrData: string | Buffer): tf.Tensor[] {
  const zip = new AdmZip(filenameOrData);
  return zip
    .getEntries()
    .map((entry) => npy.parse(bufferToArrayBuffer(entry.getData())));
}
