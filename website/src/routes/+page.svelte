<script lang="ts">
  import { resolve } from "$app/paths";

  import { ArrowUpRight } from "lucide-svelte";

  import logo from "$lib/assets/logo.svg";
  import EmbeddedRepl from "$lib/repl/EmbeddedRepl.svelte";

  const installText = {
    npm: `npm install @jax-js/jax`,
    web:
      `<` +
      String.raw`script type="module">
  import * as jax from "https://esm.sh/@jax-js/jax";
</script` +
      `>`,
  };

  let installMode = $state<"npm" | "web">("npm");
</script>

<svelte:head>
  <title>jax-js – ML for the web</title>
</svelte:head>

<!-- Header -->
<header
  class="px-6 py-4 flex items-center justify-between max-w-screen-xl mx-auto font-tiktok"
>
  <div class="flex items-center gap-3">
    <a href={resolve("/")}>
      <img src={logo} alt="jax-js logo" class="h-8" />
    </a>
  </div>
  <nav class="flex items-center gap-6">
    <a
      href="https://github.com/ekzhang/jax-js"
      target="_blank"
      class="hover:text-primary">GitHub</a
    >
    <a href={resolve("/repl")} class="hover:text-primary">REPL</a>
    <a
      rel="external"
      target="_blank"
      href="https://www.ekzhang.com/jax-js/docs/"
      class="hover:text-primary transition-colors">API Reference</a
    >
  </nav>
</header>

<main class="font-tiktok">
  <!-- Hero section -->
  <section class="px-6 py-12 md:py-16 max-w-screen-xl mx-auto">
    <div class="grid md:grid-cols-[5fr_3fr] gap-12 items-center">
      <div>
        <h1 class="text-3xl sm:text-4xl mb-6 leading-tight max-w-2xl">
          jax-js is a machine learning library and compiler for the web
        </h1>
        <p class="text-lg text-gray-700 leading-snug mb-8 max-w-2xl">
          High-performance WebAssembly and WebGPU kernels in JavaScript. Run AI
          training and inference, image algorithms, simulations, and numerical
          code on arrays, all JIT compiled in your browser.
        </p>

        <!-- Add to project box -->
        <div class="bg-primary/5 rounded-lg p-4">
          <h2 class="text-xl font-medium mb-1.5">Add jax-js to your project</h2>
          <p class="text-gray-600 text-sm mb-4">
            Zero dependencies. All major browsers, with <button
              class="enabled:underline"
              onclick={() => (installMode = "npm")}
              disabled={installMode === "npm"}>bundlers</button
            >
            and in
            <button
              class="enabled:underline"
              onclick={() => (installMode = "web")}
              disabled={installMode === "web"}>JS modules</button
            >.
          </p>
          <div
            class="bg-primary/5 border-1 border-primary rounded-lg px-3 py-2 font-mono whitespace-pre-wrap"
          >
            <span
              class="text-primary/50 select-none"
              class:hidden={installMode === "web"}>&gt;&nbsp;</span
            >{installText[installMode]}
          </div>
        </div>
      </div>

      <!-- Performance Chart -->
      <section class="flex flex-col justify-center items-center text-center">
        <h3 class="text-lg mb-1">Matrix multiplication</h3>
        <p class="text-gray-700 text-sm mb-8 max-w-[32ch]">
          Billions of floating point operations (GFLOPs) per second
        </p>
        <!-- Bar chart -->
        <div
          class="h-60 w-80 border bg-red-50 p-8 flex justify-center items-center text-red-500 text-sm"
        >
          widget / perf demo under construction
        </div>
      </section>
    </div>
  </section>

  <!-- Explainer section -->
  <section class="mx-auto max-w-screen-xl my-8 sm:px-6">
    <div class="sm:rounded-xl bg-primary/5 px-6 py-8">
      <div class="mx-auto max-w-2xl">
        <h2 class="text-xl font-medium text-center mb-6">
          Like JAX and PyTorch, but running in your browser
        </h2>

        <p class="mb-6">
          jax-js is a end-to-end ML library inspired by JAX, written in pure
          JavaScript:
        </p>

        <ul
          class="space-y-2 pl-4 mb-8 list-disc list-inside marker:text-gray-400"
        >
          <li>Runs completely client-side (Chrome, Firefox, iOS, Android).</li>
          <li>
            Has close <a
              href="https://github.com/ekzhang/jax-js/blob/main/FEATURES.md"
              target="_blank"
              class="underline hover:text-primary">API compatibility</a
            > with NumPy/JAX.
          </li>
          <li>Is written from scratch, with zero external dependencies.</li>
        </ul>

        <p class="mb-6">
          jax-js is likely the most portable ML framework, since it runs
          anywhere a browser can run. It's also simple but optimized, including
          a lightweight compiler and GPU kernel scheduler inspired by <a
            href="https://github.com/tinygrad/tinygrad"
            target="_blank"
            class="underline hover:text-primary">tinygrad</a
          >.
        </p>

        <p class="mb-6">
          Having an ML compiler on the web makes it possible to build a fully
          featured library while keeping good performance. We take a different
          approach compared to most runtimes by translating array programs into
          WebAssembly and WebGPU kernels.
        </p>

        <p>
          The goal of jax-js is to make numerical code accessible and deployable
          to everyone on the web, so compute-intensive apps can run fast,
          locally on consumer hardware.
        </p>
      </div>
    </div>
  </section>

  <!-- Live Editor section -->
  <section class="px-6 py-12 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-2">
      Try it out!
      <span class="text-red-500">(under construction)</span>
    </h2>

    <p class="mb-4 text-sm text-gray-600">
      This is a live editor, the code is running in your browser.
    </p>

    <div class="h-80">
      <EmbeddedRepl
        initialText={String.raw`import { numpy as np } from "@jax-js/jax";

const x = np.linspace(-10, 10, 200).reshape([10, 20]);
const y = np.exp(x).sub(1);

console.log(y.js());
`}
      />
    </div>
  </section>

  <!-- Learn More section -->
  <section class="px-6 py-16 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-6">Learn more</h2>

    <div class="grid sm:grid-cols-3 gap-x-6 md:gap-x-8 gap-y-4">
      <a
        href="https://github.com/ekzhang/jax-js"
        class="bg-primary/5 hover:bg-primary/15 transition-colors p-4 rounded-lg"
      >
        <h3 class="mb-2">
          GitHub Repository <ArrowUpRight
            size={18}
            class="inline-block text-gray-400 mb-px"
          />
        </h3>
        <p class="text-sm text-gray-600">
          Get started with jax-js and check out the code.
        </p>
      </a>

      <a
        href="https://github.com/ekzhang/jax-js?tab=readme-ov-file#examples"
        class="bg-primary/5 hover:bg-primary/15 transition-colors p-4 rounded-lg"
      >
        <h3 class="mb-2">
          Examples <ArrowUpRight
            size={18}
            class="inline-block text-gray-400 mb-px"
          />
        </h3>
        <p class="text-sm text-gray-600">
          Featured apps and demos built with jax-js.
        </p>
      </a>

      <a
        href={resolve("/repl")}
        class="bg-primary/5 hover:bg-primary/15 transition-colors p-4 rounded-lg"
      >
        <h3 class="mb-2">
          REPL <ArrowUpRight
            size={18}
            class="inline-block text-gray-400 mb-px"
          />
        </h3>
        <p class="text-sm text-gray-600">
          Try out the library in your browser.
        </p>
      </a>
    </div>
  </section>
</main>
