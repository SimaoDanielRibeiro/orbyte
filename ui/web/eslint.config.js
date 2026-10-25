// O "ruff" do frontend: as regras estritas do typescript-eslint, com os tipos (como o mypy em
// modo strict do lado do Python), e as regras dos hooks do React.
import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import reactHooks from "eslint-plugin-react-hooks";
import { reactRefresh } from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores([
    "dist",
    "src/api/generated",
    "src/routeTree.gen.ts",
    "playwright-report",
    "test-results",
  ]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.strictTypeChecked,
      tseslint.configs.stylisticTypeChecked,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite(),
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Os números entram em textos a toda a hora (`passo ${step}`).
      "@typescript-eslint/restrict-template-expressions": ["error", { allowNumber: true }],
    },
  },
  {
    // As rotas do TanStack Router exportam a `Route` ao lado do componente; os componentes do
    // shadcn exportam as variantes ao lado do componente.
    files: ["src/routes/**/*.tsx", "src/components/ui/**/*.tsx"],
    rules: { "react-refresh/only-export-components": "off" },
  },
  {
    files: ["*.{js,ts}", "e2e/**/*.ts"],
    languageOptions: { globals: globals.node },
  },
]);
