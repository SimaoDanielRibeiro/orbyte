import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Notice, Vocabulary } from "@/api/generated";
import { vocabularyQuery } from "@/api/vocabulary";

import { NoticeList } from "./notice-list";

const VOCABULARY: Vocabulary = {
  labels: { NoticeLevel: { info: "Nota", warning: "Aviso", error: "Erro" } },
  tasks: [],
  step_metrics: [],
};

function renderWithVocabulary(notices: Notice[]) {
  const client = new QueryClient();
  client.setQueryData(vocabularyQuery.queryKey, VOCABULARY);
  return render(
    <QueryClientProvider client={client}>
      <NoticeList notices={notices} />
    </QueryClientProvider>,
  );
}

describe("NoticeList", () => {
  it("mostra cada aviso com o nome do nível para os leitores de ecrã", () => {
    renderWithVocabulary([
      { level: "error", code: "no_rows", message: "A tabela não tem linhas.", subject: null },
      { level: "warning", code: "few_rows", message: "Há poucas linhas.", subject: null },
    ]);

    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("Erro: A tabela não tem linhas.");
    expect(items[1]).toHaveTextContent("Aviso: Há poucas linhas.");
  });

  it("sem avisos, não mostra nada", () => {
    const { container } = renderWithVocabulary([]);

    expect(container).toBeEmptyDOMElement();
  });
});
