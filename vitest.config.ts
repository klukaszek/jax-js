import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

export default defineConfig({
  esbuild: {
    supported: {
      using: false, // Needed to lower 'using' statements in tests.
    },
  },
  test: {
    setupFiles: ["test/setup.ts"],
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: playwright(),
      // https://vitest.dev/config/browser/playwright.html
      instances: [{ browser: "chromium" }],
    },
  },
});
