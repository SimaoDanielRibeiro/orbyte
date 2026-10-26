/**
 * O acompanhamento de um trabalho em segundo plano (um treino, uma procura dos agentes) pelo
 * WebSocket da API.
 *
 * A API envia, de 0,25 em 0,25 s, só o que mudou desde a última mensagem (`{"kind": "update",
 * "data": …}`), e fecha a ligação quando o trabalho acaba (ADR 0022). Se a ligação cair antes
 * do fim, volta a ligar: a API recomeça do princípio, por isso quem junta as mensagens tem de
 * ignorar o que já tem (os passos e as mensagens têm números).
 */
import { useEffect, useRef, useState } from "react";

import { API_BASE } from "./client-config";
import { ApiRequestError } from "./errors";
import type { ApiError, JobState } from "./generated";

const RETRY_MS = 1000;
const MAX_RETRIES = 5;
const DONE: ReadonlySet<JobState["status"]> = new Set(["finished", "failed"]);

type Message<U> = { kind: "update"; data: U } | { kind: "error"; data: ApiError };

export interface StreamStatus {
  /** Se a ligação está aberta. */
  connected: boolean;
  /** O erro que a API mandou (por exemplo, o run não existe). */
  error: ApiRequestError | null;
}

/**
 * Liga-se ao WebSocket `path` (por exemplo, `/experiments/run-1a2b/stream`) e chama
 * `onUpdate` com cada atualização, até o trabalho acabar.
 */
// O tipo da atualização (a do treino ou a dos agentes) só aparece no `onUpdate`, mas é ele que
// deixa quem chama receber o tipo certo, em vez de só `{ state }`.
// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters
export function useJobStream<U extends { state: JobState }>(
  path: string | null,
  onUpdate: (update: U) => void,
): StreamStatus {
  const handler = useRef(onUpdate);
  const [status, setStatus] = useState<StreamStatus>({ connected: false, error: null });

  useEffect(() => {
    handler.current = onUpdate;
  });

  useEffect(() => {
    if (path === null) {
      return;
    }
    let socket: WebSocket | null = null;
    let retries = 0;
    let finished = false;
    let timer: number | undefined;

    const connect = () => {
      const url = new URL(`${API_BASE}${path}`, window.location.href);
      url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(url);
      socket.onopen = () => {
        setStatus({ connected: true, error: null });
      };
      socket.onmessage = (event: MessageEvent<string>) => {
        const message = JSON.parse(event.data) as Message<U>;
        if (message.kind === "error") {
          finished = true;
          setStatus({
            connected: false,
            error: new ApiRequestError(
              message.data.message,
              message.data.code,
              404,
              message.data.problems,
            ),
          });
          return;
        }
        retries = 0;
        if (DONE.has(message.data.state.status)) {
          finished = true;
        }
        handler.current(message.data);
      };
      socket.onclose = () => {
        setStatus((previous) => ({ ...previous, connected: false }));
        if (!finished && retries < MAX_RETRIES) {
          retries += 1;
          timer = window.setTimeout(connect, RETRY_MS);
        }
      };
    };

    connect();
    return () => {
      finished = true;
      window.clearTimeout(timer);
      socket?.close();
    };
  }, [path]);

  return status;
}
