import * as tf from "@tensorflow/tfjs-node";
import { add, complete, cycle, suite } from "benny";
import * as baseline from "./baseline";
import { npy } from "../src";
import * as prettyBytes from "pretty-bytes";
import * as seedrandom from "seedrandom";
import * as chalk from "chalk";
import { printCompleteStats } from "./utils";

const rng = seedrandom("");
const tensor = tf.rand([100, 100], () => rng());

function sizes() {
  const jsonGzipBytes = baseline.serializeSync(tensor, true).byteLength;
  const jsonBytes = baseline.serializeSync(tensor, false).byteLength;
  const npyBytes = npy.serializeSync(tensor).byteLength;

  const printSize = (name: string, bytes: number) =>
    console.log(chalk.cyan(name) + ": " + prettyBytes(bytes));

  console.log();
  console.log(chalk.bold("Serialization sizes:"));
  printSize("baseline.serializeSync compressed", jsonGzipBytes);
  printSize("baseline.serializeSync uncompressed", jsonBytes);
  printSize("npy.serializeSync", npyBytes);
  console.log();
}

sizes();

suite(
  "NPY Serialization",

  add("baseline.serialize compressed", async () => {
    await baseline.serialize(tensor, true);
  }),

  add("baseline.serializeSync compressed", () => {
    baseline.serializeSync(tensor, true);
  }),

  add("baseline.serialize uncompressed", async () => {
    await baseline.serialize(tensor, false);
  }),

  add("baseline.serializeSync uncompressed", () => {
    baseline.serializeSync(tensor, false);
  }),

  add("npy.serialize", async () => {
    await npy.serialize(tensor);
  }),

  add("npy.serializeSync", () => {
    npy.serializeSync(tensor);
  }),

  cycle(),
  complete(printCompleteStats),
);
