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
  const baselineBytes = baseline.serializeArraySync(tensors).byteLength;
  const npzBytes = npz.serializeSync(tensors).byteLength;

  console.log();
  console.log(chalk.bold("Serialization sizes:"));
  console.log(
    chalk.cyan("baseline.serializeArraySync: ".padEnd(29, " ")) +
      prettyBytes(baselineBytes),
  );
  console.log(
    chalk.cyan("npz.serializeSync: ".padEnd(29, " ")) + prettyBytes(npzBytes),
  );
  console.log();
}

sizes();

suite(
  "NPZ Serialization",

  add("baseline.serializeArray", async () => {
    await baseline.serializeArray(tensors);
  }),

  add("baseline.serializeArraySync", () => {
    baseline.serializeArraySync(tensors);
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
