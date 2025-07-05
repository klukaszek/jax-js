// Port of the `jax.random` module.

import { type Device } from "./backend";
import { Array, DType, float32, full } from "./numpy";
import { rep } from "./utils";

/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
export function key(seed: number): Array {
  if (seed !== seed << 0) {
    throw new Error("Random seed must be a 32-bit integer");
  }
  return undefined as any; // TODO
}

/** Splits a PRNG key into `num` new keys by adding a leading axis. */
export function split(key: Array, num: number = 2): Array[] {
  void num;
  return rep(num, undefined as any); // TODO
}

/** Sample uniform random values in [minval, maxval) with given shape/dtype. */
export function uniform(
  key: Array,
  shape: number[] = [],
  {
    minval = 0,
    maxval = 1,
    dtype = float32,
    device,
  }: { minval?: number; maxval?: number; dtype?: DType; device?: Device } = {},
): Array {
  void key;
  void maxval;
  return full(shape, minval, { dtype, device });
}
