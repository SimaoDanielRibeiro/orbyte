import { describe, expect, it } from "vitest";

import type { AnalystStep } from "@/api/generated";

import { rounds, wordCount, type Round } from "./analyst-steps";

function thought(number: number, text: string): AnalystStep {
  return { number, kind: "thought", text, tool: null };
}

function used(number: number, name: string): AnalystStep {
  const summary = `What ${name} returned.`;
  return {
    number,
    kind: "tool",
    text: summary,
    tool: { name, label: name, arguments: {}, summary, result: "{}", error: null },
  };
}

function answer(number: number, text: string): AnalystStep {
  return { number, kind: "answer", text, tool: null };
}

/** Uma volta em poucas palavras: a chave, o pensamento e os nomes das ferramentas. */
function brief(round: Round) {
  return {
    key: round.key,
    thought: round.thought?.text ?? null,
    tools: round.tools.map((call) => call.tool.name),
  };
}

describe("rounds", () => {
  it("junta cada pensamento às ferramentas que pediu a seguir", () => {
    const steps = [
      thought(1, "The board first."),
      used(2, "list_candidates"),
      thought(3, "Now the winner against the second."),
      used(4, "compare_candidates"),
      thought(5, "I know enough."),
    ];

    expect(rounds(steps).map(brief)).toEqual([
      { key: 1, thought: "The board first.", tools: ["list_candidates"] },
      { key: 3, thought: "Now the winner against the second.", tools: ["compare_candidates"] },
      { key: 5, thought: "I know enough.", tools: [] },
    ]);
  });

  it("põe na mesma volta as ferramentas pedidas de uma vez", () => {
    const steps = [thought(1, "Two at once."), used(2, "get_candidate"), used(3, "get_candidate")];

    expect(rounds(steps).map(brief)).toEqual([
      { key: 1, thought: "Two at once.", tools: ["get_candidate", "get_candidate"] },
    ]);
    // Cada ferramenta guarda o número do seu passo: é a chave na lista.
    expect(rounds(steps)[0]?.tools.map((call) => call.number)).toEqual([2, 3]);
  });

  it("uma ferramenta sem pensamento antes fica numa volta sem pensamento", () => {
    const steps = [used(1, "list_candidates"), thought(2, "Now I know.")];

    expect(rounds(steps).map(brief)).toEqual([
      { key: 1, thought: null, tools: ["list_candidates"] },
      { key: 2, thought: "Now I know.", tools: [] },
    ]);
  });

  it("deixa a resposta de fora: o ecrã mostra-a à parte", () => {
    const steps = [thought(1, "Easy."), answer(2, "#4 won.")];

    expect(rounds(steps).map(brief)).toEqual([{ key: 1, thought: "Easy.", tools: [] }]);
    expect(rounds([])).toEqual([]);
  });
});

describe("wordCount", () => {
  it("conta as palavras, com os espaços e as mudanças de linha que houver", () => {
    expect(wordCount("  Okay, let's tackle\n\nthe question.  ")).toBe(5);
    expect(wordCount("one")).toBe(1);
  });

  it("um texto vazio, ou só com espaços, não tem palavras", () => {
    expect(wordCount("")).toBe(0);
    expect(wordCount(" \n\t ")).toBe(0);
  });
});
