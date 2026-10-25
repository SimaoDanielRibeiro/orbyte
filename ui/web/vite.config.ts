/**
 * O servidor de desenvolvimento e o build da interface.
 *
 * Os pedidos a `/api` passam para a API (os WebSockets do treino incluídos). O endereço vem
 * das mesmas variáveis que a API usa (`ML_LAB_API_HOST` e `ML_LAB_API_PORT`, no `.env` da raiz
 * do projeto ou no ambiente): muda-se num sítio e as duas pontas ficam de acordo.
 */
import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

const PROJECT_ROOT = fileURLToPath(new URL("../..", import.meta.url));

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, PROJECT_ROOT, "ML_LAB_API_");
  const api = `http://${env.ML_LAB_API_HOST ?? "127.0.0.1"}:${env.ML_LAB_API_PORT ?? "8000"}`;

  return {
    plugins: [tanstackRouter({ target: "react", autoCodeSplitting: true }), react(), tailwindcss()],
    resolve: {
      alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
    },
    server: {
      port: 5173,
      strictPort: true,
      // As rotas carregam-se aos bocados (uma por ecrã): prepará-las ao arrancar deixa o Vite
      // descobrir logo as dependências, em vez de recarregar a página na primeira visita.
      warmup: { clientFiles: ["./src/main.tsx", "./src/routes/**/*.tsx"] },
      proxy: {
        "/api": {
          target: api,
          ws: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
    build: {
      rolldownOptions: {
        output: {
          // As bibliotecas mudam pouco: em pedaços próprios, o browser guarda-as entre versões
          // da interface, e nenhum pedaço passa dos 500 kB.
          codeSplitting: {
            groups: [
              { name: "react", test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
              { name: "tanstack", test: /node_modules[\\/]@tanstack[\\/]/ },
              {
                name: "ui",
                test: /node_modules[\\/](radix-ui|@radix-ui|sonner|lucide-react|cn|class-variance-authority)[\\/]/,
              },
            ],
          },
        },
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      restoreMocks: true,
    },
  };
});
