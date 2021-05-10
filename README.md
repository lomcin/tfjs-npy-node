# NumPy file parsing and serialization for TensorFlow.js

[![test](https://github.com/MaximeKjaer/tfjs-npy-node/actions/workflows/test.yml/badge.svg)](https://github.com/MaximeKjaer/tfjs-npy-node/actions/workflows/test.yml)
[![npm](https://img.shields.io/npm/v/tfjs-npy-node)](https://www.npmjs.com/package/tfjs-npy-node)

This is a fork of [tfjs-npy](https://github.com/propelml/tfjs-npy), which adds:

- New synchronous APIs
- `.npz` parsing and serialization
- Loading and saving files from disk
- Using `Buffer` instead of `ArrayBuffer`

Note that because of support for `.npz` (which uses zlib for zipping), and for loading and saving files, this library is meant to be used in Node.js, not in the browser. If you want to convert to npy in the browser, consider using the original [tfjs-npy](https://github.com/propelml/tfjs-npy).

See [the Numpy docs](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html) for more information about the file format.

## Installation

```bash
# Using npm
$ npm install tfjs-npy-node

# Using yarn
$ yarn add tfjs-npy-node
```

## API

```ts
import * as tf from "@tensorflow/tfjs-core";
import { npy, npz } from "tfjs-npy-node";

////////////////
// .npy files //
////////////////

// Load a Tensor from a .npy file:
const tensor1: tf.Tensor = await npy.load("file.npy");

// Parse a Tensor from an Buffer containing the bytes of a .npy file:
const npyBuffer: Buffer = getBufferFromSomewhere();
const tensor2: tf.Tensor = npy.parse(npyBuffer);

// Save a tensor to a .npy file:
await npy.save("file2.npy", tensor2);

// Serialize a tensor to an Buffer containing the bytes of the .npy file:
const npyBuffer2: Buffer = await npy.serialize(tensor1);
const npyBuffer3: Buffer = npy.serializeSync(tensor2);

////////////////
// .npz files //
////////////////

// Load a Tensor from a .npy file:
const tensors1: tf.Tensor[] = await npz.load("file.npz");

// Parse a Tensor from an Buffer containing the bytes of a .npy file:
const npzBuffer: Buffer = getBufferFromSomewhere();
const tensors2: tf.Tensor[] = npz.parse(npyBuffer);

// Save a tensor to a .npy file:
await npz.save("file2.npz", tensors2);

// Serialize a tensor to an Buffer containing the bytes of the .npy file:
const npzBuffer2: Buffer = await npz.serialize(tensors1);
const npzBuffer3: Buffer = npz.serializeSync(tensors2);
```

## Contributing

### Getting started

Clone the repo, install dependencies, and run the tests:

```bash
$ git clone git@github.com:MaximeKjaer/tfjs-npy-node.git
$ cd tfjs-npy-node
$ npm install
$ npm run build
$ npm run test
```

### Releasing a new version

If you have write access to the main branch of the repo, use [`npm version`](https://docs.npmjs.com/cli/v7/commands/npm-version).

The `package.json` defines some tasks that are automatically executed when bumping the package version. You will need to have the [GitHub CLI](https://cli.github.com/) installed if you want the automatic release creation to work.
