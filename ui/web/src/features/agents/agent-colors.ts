/**
 * A cor de cada agente, a mesma no fluxograma, na conversa e no painel de cada um: cada família
 * tem a cor da sua posição no plano (a da paleta dos dados); o agente de dados é branco e o
 * coordenador cinzento, porque não são famílias de modelos (e têm o seu ícone).
 */
import type { AgentMessage, ModelFamily } from "@/api/generated";
import { categoryColor } from "@/lib/colors";

export const DATA_AGENT_COLOR = "#fafafa";
export const COORDINATOR_COLOR = "#a0a0a0";

/** A cor de uma família, pela ordem dos agentes no plano. */
export function familyColor(families: readonly ModelFamily[], family: ModelFamily): string {
  const index = families.indexOf(family);
  return categoryColor(index < 0 ? 0 : index);
}

/** A cor de quem escreveu uma mensagem da conversa. */
export function messageColor(message: AgentMessage, families: readonly ModelFamily[]): string {
  if (message.role === "data") {
    return DATA_AGENT_COLOR;
  }
  if (message.role === "coordinator" || message.family === null) {
    return COORDINATOR_COLOR;
  }
  return familyColor(families, message.family);
}
