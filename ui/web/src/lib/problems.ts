/**
 * A cor de cada problema (ADR 0048): o que um modelo resolve, no ecrã Modelos e na página de
 * cada modelo.
 *
 * Antes, cada família tinha uma cor, mas eram onze famílias e nada ligava a cor ao que o modelo
 * faz: lia-se como cores ao acaso. Agora a cor lê-se em dois tempos:
 *
 * - **a temperatura é o paradigma:** os problemas supervisionados (aprendem com um alvo) são
 *   frios, os não supervisionados (procuram estrutura sozinhos) são quentes, e o controlo do
 *   reinforcement learning, que não é nenhum dos dois, fica no verde, a meio;
 * - **o tom é o problema**, dentro do paradigma: a classificação azul, a regressão ciano…
 *
 * As cores são as da paleta dos dados (`colors.ts`); o nome do problema está sempre escrito ao
 * lado, e a cor nunca é a única pista. Os agentes continuam com a cor da família (`families.ts`):
 * um agente é uma família, não um problema.
 */
import type { Problem } from "@/api/generated";

import { CATEGORICAL, OTHER } from "./colors";

const [BLUE, YELLOW, PINK, GREEN, , ORANGE, CYAN] = CATEGORICAL;

export const PROBLEM_COLORS: Readonly<Record<Problem, string>> = {
  classification: BLUE,
  regression: CYAN,
  clustering: YELLOW,
  dimensionality_reduction: PINK,
  anomaly_detection: ORANGE,
  control: GREEN,
};

/** A cor de um problema (cinzenta, se o backend trouxer um problema que ainda não tem cor). */
export function problemColor(problem: Problem): string {
  return (PROBLEM_COLORS as Partial<Record<string, string>>)[problem] ?? OTHER;
}
