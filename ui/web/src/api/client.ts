/**
 * O cliente da API, pronto a usar.
 *
 * Importa-se uma vez, ao arrancar (`main.tsx`): regista a tradução dos erros, para que todos os
 * pedidos falhem com um `ApiRequestError` (com o `code` e a `message` da API) e nunca com o
 * corpo cru da resposta.
 */
import { toApiRequestError } from "./errors";
import { client } from "./generated/client.gen";

client.interceptors.error.use((error, response) => toApiRequestError(error, response?.status));

export { client };
