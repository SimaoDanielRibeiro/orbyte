/**
 * Os passos do analista arrumados para se lerem (ADR 0057): pelo que ele fez, e não pela ordem
 * em que pensou e agiu.
 *
 * Cada vez que o modelo de linguagem responde, pensa e depois pede ferramentas, ou escreve a
 * resposta (`Analyst.answer`, no backend): o rasto é pensar, ferramenta, pensar, ferramenta,
 * pensar, resposta. Aqui junta-se cada pensamento ao que ele fez a seguir, numa volta: o ecrã
 * mostra as ferramentas (e, no fim, a resposta), com o pensamento que as pediu fechado por baixo.
 * Os pensamentos de um modelo pequeno começam quase todos da mesma maneira ("Okay, let's
 * tackle…"), e o que se quer ler primeiro é o que ele fez.
 */
import type { AnalystStep, ToolUse } from "@/api/generated";

/** Uma ferramenta usada, com o número do passo (a chave na lista). */
export interface ToolCall {
  number: number;
  tool: ToolUse;
}

/** Uma volta do analista: o que pensou e as ferramentas que pediu a seguir (nenhuma, no fim). */
export interface Round {
  /** O número do primeiro passo da volta: a chave na lista. */
  key: number;
  thought: AnalystStep | null;
  tools: ToolCall[];
}

/**
 * Os passos em voltas, pela ordem. Um pensamento começa uma volta, e as ferramentas que vêm a
 * seguir são dela (o modelo pode pedir mais do que uma de cada vez); uma ferramenta sem
 * pensamento antes (um modelo que não pensa em voz alta) fica numa volta sem pensamento. A
 * resposta não entra: o ecrã mostra-a à parte.
 */
export function rounds(steps: readonly AnalystStep[]): Round[] {
  const result: Round[] = [];
  for (const step of steps) {
    if (step.kind === "thought") {
      result.push({ key: step.number, thought: step, tools: [] });
      continue;
    }
    if (step.kind !== "tool" || step.tool === null) {
      continue;
    }
    const call = { number: step.number, tool: step.tool };
    const current = result.at(-1);
    if (current === undefined) {
      result.push({ key: step.number, thought: null, tools: [call] });
    } else {
      current.tools.push(call);
    }
  }
  return result;
}

/** Quantas palavras tem um texto: o tamanho de um pensamento, antes de o abrir. */
export function wordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/u).length;
}
