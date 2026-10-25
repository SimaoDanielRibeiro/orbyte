import type { CreateClientConfig } from "./generated/client.gen";

/**
 * A configuração inicial do cliente gerado.
 *
 * Os pedidos vão para `/api`, no próprio servidor da interface: em desenvolvimento, o Vite
 * passa-os à API (ver `vite.config.ts`). Assim a interface e a API ficam na mesma origem, sem
 * CORS, e o endereço da API só se escolhe num sítio.
 */
export const createClientConfig: CreateClientConfig = (config) => ({
  ...config,
  baseUrl: API_BASE,
});

/** O prefixo de todos os pedidos à API. */
export const API_BASE = "/api";
