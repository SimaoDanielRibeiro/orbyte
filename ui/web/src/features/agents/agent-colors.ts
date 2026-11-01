/**
 * A cor de cada agente, a mesma no fluxograma, na conversa e no painel de cada um: cada família
 * tem a sua cor, fixa, a mesma do ecrã Modelos (`lib/families.ts`); o agente de dados é branco e
 * o coordenador cinzento, porque não são famílias de modelos (e têm o seu ícone). O agente de IA
 * tem um lilás que nenhuma família usa: vê-se logo que é outro tipo de agente (ADR 0042).
 */
import type { AgentMessage } from "@/api/generated";
import { familyColor } from "@/lib/families";

export { familyColor };

export const DATA_AGENT_COLOR = "#fafafa";
export const COORDINATOR_COLOR = "#a0a0a0";
export const AI_AGENT_COLOR = "#c4b5fd";

/** A cor de quem escreveu uma mensagem da conversa. */
export function messageColor(message: AgentMessage): string {
  if (message.role === "data") {
    return DATA_AGENT_COLOR;
  }
  if (message.role === "ai") {
    return AI_AGENT_COLOR;
  }
  if (message.role === "coordinator" || message.family === null) {
    return COORDINATOR_COLOR;
  }
  return familyColor(message.family);
}
