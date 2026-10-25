/**
 * O pouco estado que é só do cliente: o caminho escolhido e se a barra lateral está aberta.
 *
 * Tudo o resto (os datasets, os runs, as procuras) vem da API e vive na cache do TanStack
 * Query. Guarda-se no `localStorage`, por isso sobrevive a recarregar a página.
 */
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

/** Os dois caminhos do painel inicial: testar um modelo à mão, ou deixar os agentes procurar. */
export type WorkPath = "manual" | "agents";

interface SessionState {
  /** O caminho escolhido: a barra lateral mostra os passos dele. */
  path: WorkPath;
  /** Se a barra lateral está aberta (no computador; no telemóvel é uma gaveta). */
  sidebarOpen: boolean;
  choosePath: (path: WorkPath) => void;
  setSidebarOpen: (open: boolean) => void;
}

export const useSession = create<SessionState>()(
  persist(
    (set) => ({
      path: "manual",
      sidebarOpen: true,
      choosePath: (path) => {
        set({ path });
      },
      setSidebarOpen: (sidebarOpen) => {
        set({ sidebarOpen });
      },
    }),
    {
      name: "ml-lab:session",
      version: 1,
      storage: createJSONStorage(() => localStorage),
      partialize: ({ path, sidebarOpen }) => ({ path, sidebarOpen }),
    },
  ),
);
