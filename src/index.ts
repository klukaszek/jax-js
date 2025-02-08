import * as core from "./core";
import * as numpy from "./numpy";
import { Array, ArrayLike } from "./numpy";
import * as tree from "./tree";
import type { JsTree } from "./tree";

export { numpy, tree };

// Convert a subtype of JsTree<A> into a JsTree<B>, with the same structure.
type MapJsTree<T, A, B> = T extends A
  ? B
  : T extends globalThis.Array<infer U>
    ? MapJsTree<U, A, B>[]
    : { [K in keyof T]: MapJsTree<T[K], A, B> };

// Assert that a function's arguments are a subtype of the given type.
type WithArgsSubtype<F extends (args: any[]) => any, T> =
  Parameters<F> extends T ? F : never;

export const jvp = core.jvp as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<Array>>,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  tangents: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [ReturnType<F>, ReturnType<F>];

export const vmap = core.vmap as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<Array>>,
  inAxes: MapJsTree<Parameters<F>, Array, number>
) => F;

export const deriv = core.deriv as unknown as (
  f: (x: ArrayLike) => ArrayLike
) => (x: ArrayLike) => Array;
