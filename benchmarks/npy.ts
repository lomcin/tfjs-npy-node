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
  const baselineBytes = baseline.serializeSync(tensor).byteLength;
  const npyBytes = npy.serializeSync(tensor).byteLength;

  console.log();
  console.log(chalk.bold("Serialization sizes:"));
  console.log(
    chalk.cyan("baseline.serializeSync: ".padEnd(24, " ")) +
      prettyBytes(baselineBytes),
  );
  console.log(
    chalk.cyan("npy.serializeSync: ".padEnd(24, " ")) + prettyBytes(npyBytes),
  );
  console.log();
}

sizes();

suite(
  "NPY Serialization",

  add("baseline.serialize", async () => {
    await baseline.serialize(tensor);
  }),

  add("baseline.serializeSync", () => {
    baseline.serializeSync(tensor);
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
