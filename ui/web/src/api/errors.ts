import type { ApiError } from "./generated";

/** O código de um erro sem resposta da API: ela não está a correr, ou não se chega a ela. */
export const API_UNREACHABLE = "api_unreachable";

/**
 * Um pedido à API que falhou, com o corpo do `ApiError` que ela mandou.
 *
 * A `message` vem em português e já diz o que fazer: mostra-se como está. O `code` é estável e
 * é com ele que a interface decide o resto (por exemplo, reabrir um run pelo histórico).
 */
export class ApiRequestError extends Error {
  override readonly name = "ApiRequestError";
  /** O código estável do erro (`run_not_found`, `training_in_progress`…). */
  readonly code: string;
  /** O código HTTP da resposta, ou 0 se não houve resposta. */
  readonly status: number;
  /** Num pedido inválido, uma frase por problema. */
  readonly problems: readonly string[];

  constructor(message: string, code: string, status: number, problems: readonly string[] = []) {
    super(message);
    this.code = code;
    this.status = status;
    this.problems = problems;
  }

  /** Se o erro quer dizer que a API não está a responder. */
  get unreachable(): boolean {
    return this.code === API_UNREACHABLE;
  }
}

function isApiError(value: unknown): value is ApiError {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Record<string, unknown>;
  return typeof candidate.code === "string" && typeof candidate.message === "string";
}

/**
 * Transforma o que o cliente gerado lançou num `ApiRequestError`.
 *
 * Há três casos: a API respondeu com um `ApiError` (o normal); não houve resposta, porque a
 * API não está a correr (o `fetch` falha); ou houve uma resposta que não é da API (o servidor
 * do Vite responde 5xx quando não consegue passar o pedido).
 */
export function toApiRequestError(error: unknown, status: number | undefined): ApiRequestError {
  if (error instanceof ApiRequestError) {
    return error;
  }
  if (isApiError(error)) {
    return new ApiRequestError(error.message, error.code, status ?? 0, error.problems);
  }
  if (status === undefined || status === 0 || status >= 500) {
    return new ApiRequestError(
      "The API isn't responding. Start it with `uv run python -m ml_lab.api` and try again.",
      API_UNREACHABLE,
      status ?? 0,
    );
  }
  return new ApiRequestError("The API answered in an unexpected way.", "unexpected", status);
}
