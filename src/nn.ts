// Common functions for neural network libraries, mirroring `jax.nn` in JAX.

import { fudgeArray } from "./frontend/array";
import { absolute, Array, ArrayLike, clip, maximum } from "./numpy";

/**
 * Rectified Linear Unit (ReLU) activation function:
 * `relu(x) = max(x, 0)`.
 */
export function relu(x: ArrayLike): Array {
  return maximum(x, 0);
}

/**
 * Rectified Linear Unit 6 (ReLU6) activation function:
 * `relu6(x) = min(max(x, 0), 6)`.
 */
export function relu6(x: ArrayLike): Array {
  return clip(x, 0, 6);
}

/**
 * Soft-sign activation function, computed element-wise:
 * `softsign(x) = x / (|x| + 1)`.
 */
export function softsign(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.div(absolute(x).add(1));
}
