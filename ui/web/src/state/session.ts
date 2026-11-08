/**
 * O pouco estado que é só do cliente: o caminho escolhido, a barra lateral, e os
 * identificadores do que está aberto (o dataset, a preparação, o run, a procura dos agentes).
 *
 * Tudo o resto vem da API e vive na cache do TanStack Query: aqui guardam-se só os ids (e a
 * configuração da preparação, que é o que o utilizador escolheu). Fica no `localStorage`, por
 * isso sobrevive a recarregar a página; se a API recomeçar, os ids deixam de existir e os
 * ecrãs dizem que é preciso carregar outra vez.
 */
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import type { PreparationConfig, TaskType } from "@/api/generated";
import type { PrepareChoices } from "@/features/prepare/choices";

/** Os dois caminhos do painel inicial: testar um modelo à mão, ou deixar os agentes procurar. */
export type WorkPath = "manual" | "agents";

/** A preparação feita no ecrã Preparar: é com ela que se treina. */
export interface Preparation {
  datasetId: string;
  config: PreparationConfig;
  seed: number;
  taskType: TaskType;
}

/** O que se escolheu no ecrã Preparar (com que ele volta a abrir), mesmo que ainda não sirva. */
export interface PreparationDraft {
  datasetId: string;
  choices: PrepareChoices;
}

/** Uma pergunta ao analista a decorrer (fase 13): volta-se a ela depois de sair do ecrã. */
export interface AnalystQuestion {
  searchId: string;
  questionId: string;
  question: string;
}

interface SessionState {
  /** O caminho escolhido: a barra lateral mostra os passos dele. */
  path: WorkPath;
  /** Se a barra lateral está aberta (no computador; no telemóvel é uma gaveta). */
  sidebarOpen: boolean;
  /** O dataset aberto (o id na memória da API). */
  datasetId: string | null;
  /** A última preparação, se for do dataset aberto. */
  preparation: Preparation | null;
  /** As escolhas do ecrã Preparar para o dataset aberto. */
  draft: PreparationDraft | null;
  /** O último run (o que "Treinar e avaliar" e "Explicar" mostram). */
  runId: string | null;
  /** O nome do modelo do último run começado aqui: o título do ecrã enquanto ele treina. */
  runTitle: string | null;
  /**
   * A procura dos agentes aberta: enquanto houver uma, o ecrã Agentes mostra-a (volta-se a ela
   * tal como estava). "Nova procura" esquece-a, e abrir outro dataset também.
   */
  searchId: string | null;
  /**
   * A pergunta ao analista a decorrer: sair do ecrã (ou recarregar) e voltar segue-a outra vez.
   * Quando a resposta chega, fica nas respondidas da procura e isto esquece-se.
   */
  analystQuestion: AnalystQuestion | null;
  /** O último treino de um agente de reinforcement learning. */
  agentRunId: string | null;
  choosePath: (path: WorkPath) => void;
  setSidebarOpen: (open: boolean) => void;
  /** Abre um dataset; a preparação e a procura dos agentes de outro dataset deixam de valer. */
  openDataset: (datasetId: string | null) => void;
  setPreparation: (preparation: Preparation | null) => void;
  setDraft: (draft: PreparationDraft | null) => void;
  /** Abre um run; `title` é o nome do modelo, quando se acaba de o começar. */
  openRun: (runId: string | null, title?: string | null) => void;
  openSearch: (searchId: string | null) => void;
  openAnalystQuestion: (question: AnalystQuestion | null) => void;
  openAgentRun: (runId: string | null) => void;
}

export const useSession = create<SessionState>()(
  persist(
    (set) => ({
      path: "manual",
      sidebarOpen: true,
      datasetId: null,
      preparation: null,
      draft: null,
      runId: null,
      runTitle: null,
      searchId: null,
      analystQuestion: null,
      agentRunId: null,
      choosePath: (path) => {
        set({ path });
      },
      setSidebarOpen: (sidebarOpen) => {
        set({ sidebarOpen });
      },
      openDataset: (datasetId) => {
        set((state) =>
          state.datasetId === datasetId
            ? {}
            : { datasetId, preparation: null, draft: null, searchId: null },
        );
      },
      setPreparation: (preparation) => {
        set({ preparation });
      },
      setDraft: (draft) => {
        set({ draft });
      },
      openRun: (runId, title = null) => {
        set((state) => ({
          runId,
          runTitle: title ?? (state.runId === runId ? state.runTitle : null),
        }));
      },
      openSearch: (searchId) => {
        set({ searchId });
      },
      openAnalystQuestion: (analystQuestion) => {
        set({ analystQuestion });
      },
      openAgentRun: (agentRunId) => {
        set({ agentRunId });
      },
    }),
    {
      name: "ml-lab:session",
      version: 5,
      storage: createJSONStorage(() => localStorage),
      partialize: ({
        path,
        sidebarOpen,
        datasetId,
        preparation,
        draft,
        runId,
        runTitle,
        searchId,
        analystQuestion,
        agentRunId,
      }) => ({
        path,
        sidebarOpen,
        datasetId,
        preparation,
        draft,
        runId,
        runTitle,
        searchId,
        analystQuestion,
        agentRunId,
      }),
      // As versões anteriores não tinham alguns campos (a 1 só tinha o caminho e a barra; a 2,
      // o rascunho da preparação; a 3, o treino de RL; a 4, a pergunta ao analista): os que
      // faltam começam vazios.
      migrate: (persisted) => ({
        agentRunId: null,
        analystQuestion: null,
        ...(persisted as Partial<SessionState>),
      }),
    },
  ),
);
