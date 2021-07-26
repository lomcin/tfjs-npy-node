import * as tf from "@tensorflow/tfjs-node";
import { add, complete, cycle, suite } from "benny";
import * as baseline from "./baseline";
import { npz } from "../src";
import * as prettyBytes from "pretty-bytes";
import * as seedrandom from "seedrandom";
import * as chalk from "chalk";
import { printCompleteStats } from "./utils";

const rng = seedrandom("");
const tensors = [
  tf.rand([784, 200], () => rng()),
  tf.rand([200], () => rng()),
  tf.rand([200, 200], () => rng()),
  tf.rand([200], () => rng()),
  tf.rand([200, 10], () => rng()),
  tf.rand([10], () => rng()),
];

function sizes() {
  const jsonGzipBytes = baseline.serializeArraySync(tensors, true).byteLength;
  const jsonBytes = baseline.serializeArraySync(tensors, false).byteLength;
  const npzBytes = npz.serializeSync(tensors).byteLength;

  const printSize = (name: string, bytes: number) =>
    console.log(chalk.cyan(name) + ": " + prettyBytes(bytes));

  console.log();
  console.log(chalk.bold("Serialization sizes:"));
  printSize("baseline.serializeArraySync compressed", jsonGzipBytes);
  printSize("baseline.serializeArraySync uncompressed", jsonBytes);
  printSize("npz.serializeSync", npzBytes);
  console.log();
}

sizes();

suite(
  "NPZ Serialization",

  add("baseline.serializeArray compressed", async () => {
    await baseline.serializeArray(tensors, true);
  }),

  add("baseline.serializeArraySync compressed", () => {
    baseline.serializeArraySync(tensors, true);
  }),

  add("baseline.serializeArray uncompressed", async () => {
    await baseline.serializeArray(tensors, false);
  }),

  add("baseline.serializeArraySync uncompressed", () => {
    baseline.serializeArraySync(tensors, false);
  }),

  add("npz.serialize", async () => {
    await npz.serialize(tensors);
  }),

  add("npz.serializeSync", () => {
    npz.serializeSync(tensors);
  }),

  cycle(),
  complete(printCompleteStats),
);
