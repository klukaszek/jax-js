import { numpy as np, random } from "@jax-js/jax";
import { expect, test } from "vitest";

test("generate uniform random", () => {
  const key = random.key(0);
  const [a, b, c] = random.split(key, 3);

  const x = random.uniform(a);
  expect(x.js()).toBeWithinRange(0, 1);

  const y = random.uniform(b, [0]);
  expect(y.js()).toEqual([]);

  const z = random.uniform(c, [2, 3]);
  expect(z.shape).toEqual([2, 3]);
  expect(z.dtype).toEqual(np.float32);
  const zx = z.js() as number[][];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 3; j++) {
      expect(zx[i][j]).toBeWithinRange(0, 1);
    }
  }
});
