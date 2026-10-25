import { describe, expect, it } from "vitest";

import { API_UNREACHABLE, ApiRequestError, toApiRequestError } from "./errors";

describe("toApiRequestError", () => {
  it("guarda o código, a mensagem e os problemas de um ApiError", () => {
    const error = toApiRequestError(
      { code: "invalid_request", message: "O pedido tem campos a mais.", problems: ["campo x"] },
      422,
    );

    expect(error).toBeInstanceOf(ApiRequestError);
    expect(error.code).toBe("invalid_request");
    expect(error.message).toBe("O pedido tem campos a mais.");
    expect(error.status).toBe(422);
    expect(error.problems).toEqual(["campo x"]);
    expect(error.unreachable).toBe(false);
  });

  it("sem resposta, a API não está a correr", () => {
    const error = toApiRequestError(new TypeError("Failed to fetch"), undefined);

    expect(error.code).toBe(API_UNREACHABLE);
    expect(error.unreachable).toBe(true);
    expect(error.message).toContain("uv run python -m ml_lab.api");
  });

  it("uma resposta 5xx que não é da API também quer dizer que ela não responde", () => {
    // É o que o servidor do Vite responde quando não consegue passar o pedido.
    expect(toApiRequestError("", 502).unreachable).toBe(true);
  });

  it("um 4xx sem ApiError é uma resposta inesperada", () => {
    const error = toApiRequestError("<html>", 404);

    expect(error.code).toBe("unexpected");
    expect(error.status).toBe(404);
  });
});
