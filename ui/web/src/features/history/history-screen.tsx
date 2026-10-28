/**
 * O ecrã Experiências: o histórico dos runs (cada treino que termina fica registado sozinho),
 * com os filtros, a comparação de dois a cinco runs e a gaveta de cada um (abrir, reproduzir,
 * guardar, mudar o nome, apagar).
 *
 * Os filtros ficam no endereço (`?tarefa=…&dataset=…&modelo=…`), e a API filtra; as opções de
 * cada filtro são os valores que o histórico tem. Os runs a comparar escolhem-se na tabela.
 */
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { cn } from "cn";
import { Check } from "lucide-react";
import { useState } from "react";

import type { RunEntry, TaskType } from "@/api/generated";
import { historyRunsOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { TopicText } from "@/components/markdown/topic-text";
import { ErrorNote, NeedsStep } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { formatMoment, formatQuantity } from "@/lib/format";

import { ComparisonView } from "./comparison-view";
import { RunSheet } from "./run-sheet";
import { MAX_COMPARED, primaryValue } from "./values";

/** O valor de um filtro que não filtra nada. */
const ALL = "todos";

export function HistoryScreen() {
  const search = useSearch({ from: "/experiencias" });
  const navigate = useNavigate();
  const task = search.tarefa === undefined ? undefined : (search.tarefa as TaskType);
  const everything = useQuery({ ...historyRunsOptions(), staleTime: 0 });
  const filtered = useQuery({
    ...historyRunsOptions({
      query: { task_type: task, dataset_name: search.dataset, model_name: search.modelo },
    }),
    staleTime: 0,
  });
  const [chosen, setChosen] = useState<string[]>([]);
  const [opened, setOpened] = useState<string | null>(null);
  const entries = filtered.data ?? [];
  const openedEntry =
    opened === null
      ? null
      : ((everything.data ?? []).find((entry) => entry.run_id === opened) ?? null);

  const setFilter = (key: "tarefa" | "dataset" | "modelo", value: string) => {
    void navigate({
      to: "/experiencias",
      search: { ...search, [key]: value === ALL ? undefined : value },
    });
  };
  // Os runs escolhidos ficam pela ordem da tabela (do mais recente para o mais antigo).
  const order = (everything.data ?? []).map((entry) => entry.run_id);
  const toggle = (runId: string, on: boolean) => {
    setChosen((previous) =>
      (on ? [...previous, runId] : previous.filter((id) => id !== runId)).sort(
        (a, b) => order.indexOf(a) - order.indexOf(b),
      ),
    );
  };

  return (
    <Page>
      <PageHeader title={SCREENS.history.title} description={SCREENS.history.description}>
        <WhyButton topics={["experiment_tracking", "comparing_runs", "reproducibility"]} />
      </PageHeader>
      <ErrorNote error={everything.error ?? filtered.error} className="mt-8" />
      {everything.data === undefined ? (
        everything.error === null && <Skeleton className="mt-10 h-64 w-full" />
      ) : everything.data.length === 0 ? (
        <NeedsStep
          message="Ainda não há runs no histórico: cada treino que termina fica registado aqui sozinho."
          to="/modelos"
          action="Escolher um modelo"
        />
      ) : (
        <>
          <Filters
            all={everything.data}
            search={search}
            count={filtered.data?.length ?? null}
            onChange={setFilter}
          />
          <RunsTable
            entries={entries}
            chosen={chosen}
            onToggle={toggle}
            onOpen={setOpened}
            loading={filtered.data === undefined}
          />
          <TopicText topic="run_history" className="mt-4 max-w-[68ch]" />

          <section aria-labelledby="comparacao-titulo" className="mt-16">
            <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
              <h2 id="comparacao-titulo" className="text-2xl">
                Comparar runs
              </h2>
              {chosen.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setChosen([]);
                  }}
                >
                  Limpar a escolha ({chosen.length})
                </Button>
              )}
            </div>
            <ComparisonView runIds={chosen} />
          </section>
        </>
      )}
      <RunSheet
        entry={openedEntry}
        onClose={() => {
          setOpened(null);
        }}
        onDeleted={(runId) => {
          setChosen((previous) => previous.filter((id) => id !== runId));
        }}
      />
    </Page>
  );
}

interface HistorySearch {
  tarefa?: string;
  dataset?: string;
  modelo?: string;
}

/** Os filtros: as opções são os valores que o histórico tem. */
function Filters({
  all,
  search,
  count,
  onChange,
}: {
  all: readonly RunEntry[];
  search: HistorySearch;
  count: number | null;
  onChange: (key: "tarefa" | "dataset" | "modelo", value: string) => void;
}) {
  const label = useLabel();
  const tasks = [...new Set(all.map((entry) => entry.task_type))];
  const datasets = [...new Set(all.map((entry) => entry.dataset_name))].sort((a, b) =>
    a.localeCompare(b, "pt"),
  );
  const models = new Map(all.map((entry) => [entry.model_name, entry.model_display_name]));
  return (
    <div className="mt-10 flex flex-wrap items-end justify-between gap-4 border-b pb-5">
      <div className="flex flex-wrap gap-4">
        <Filter
          id="historico-tarefa"
          label="Tarefa"
          value={search.tarefa ?? ALL}
          options={tasks.map((task) => [task, label("TaskType", task)])}
          onChange={(value) => {
            onChange("tarefa", value);
          }}
        />
        <Filter
          id="historico-dataset"
          label="Dataset ou ambiente"
          value={search.dataset ?? ALL}
          options={datasets.map((name) => [name, name])}
          onChange={(value) => {
            onChange("dataset", value);
          }}
        />
        <Filter
          id="historico-modelo"
          label="Modelo ou agente"
          value={search.modelo ?? ALL}
          options={[...models].sort((a, b) => a[1].localeCompare(b[1], "pt"))}
          onChange={(value) => {
            onChange("modelo", value);
          }}
        />
      </div>
      {count !== null && (
        <p role="status" className="text-sm text-muted-foreground">
          {formatQuantity(count, "run", "runs")}
        </p>
      )}
    </div>
  );
}

function Filter({
  id,
  label,
  value,
  options,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  options: readonly (readonly [string, string])[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={id} className="text-xs text-muted-foreground">
        {label}
      </label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id={id} className="min-w-48">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>Todos</SelectItem>
          {options.map(([option, name]) => (
            <SelectItem key={option} value={option}>
              {name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

/** A tabela dos runs: o nome abre a gaveta; a caixa escolhe o run para comparar. */
function RunsTable({
  entries,
  chosen,
  onToggle,
  onOpen,
  loading,
}: {
  entries: readonly RunEntry[];
  chosen: readonly string[];
  onToggle: (runId: string, on: boolean) => void;
  onOpen: (runId: string) => void;
  loading: boolean;
}) {
  const label = useLabel();
  if (loading) {
    return <Skeleton className="mt-6 h-48 w-full" />;
  }
  if (entries.length === 0) {
    return (
      <p className="mt-6 font-serif text-reading text-muted-foreground">
        Nenhum run corresponde a estes filtros.
      </p>
    );
  }
  const full = chosen.length >= MAX_COMPARED;
  return (
    // `relative`: os textos só para leitores de ecrã das células (posição absoluta) ficam presos
    // aqui; sem ele, saíam da caixa e a página toda deslizava para o lado no telemóvel.
    <div className="relative mt-6 max-w-full overflow-x-auto rounded-md border">
      <table className="w-full text-sm">
        <caption className="sr-only">
          Os runs do histórico, do mais recente para o mais antigo
        </caption>
        <thead>
          <tr className="border-b text-left">
            <th scope="col" className="w-10 px-3 py-2 font-medium">
              <span className="sr-only">Comparar</span>
            </th>
            <th scope="col" className="px-3 py-2 font-medium whitespace-nowrap">
              Quando
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Nome
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Dataset
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Tarefa
            </th>
            <th scope="col" className="px-3 py-2 text-right font-medium whitespace-nowrap">
              Métrica principal
            </th>
            <th scope="col" className="px-3 py-2 text-right font-medium">
              Seed
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Guardado
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => {
            const on = chosen.includes(entry.run_id);
            return (
              <tr
                key={entry.run_id}
                className={cn(
                  "border-b border-border/50 last:border-0 hover:bg-accent/40",
                  on && "bg-accent/60",
                )}
              >
                <td className="px-3 py-2">
                  <Checkbox
                    checked={on}
                    disabled={!on && full}
                    aria-label={`Comparar «${entry.name}»`}
                    onCheckedChange={(checked) => {
                      onToggle(entry.run_id, checked === true);
                    }}
                  />
                </td>
                <td className="px-3 py-2 whitespace-nowrap text-muted-foreground tabular-nums">
                  {formatMoment(entry.created_at)}
                </td>
                <td className="px-3 py-2">
                  <button
                    type="button"
                    onClick={() => {
                      onOpen(entry.run_id);
                    }}
                    className="rounded-sm text-left font-medium underline-offset-4 outline-none hover:underline focus-visible:ring-3 focus-visible:ring-ring/50"
                  >
                    {entry.name}
                  </button>
                </td>
                <td className="px-3 py-2 text-muted-foreground">{entry.dataset_name}</td>
                <td className="px-3 py-2 whitespace-nowrap text-muted-foreground">
                  {label("TaskType", entry.task_type)}
                </td>
                <td className="px-3 py-2 text-right whitespace-nowrap tabular-nums">
                  {primaryValue(entry)}
                  {entry.primary_label !== null && (
                    <span className="ml-1.5 text-xs text-muted-foreground">
                      {entry.primary_label}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-right text-muted-foreground tabular-nums">
                  {entry.seed}
                </td>
                <td className="px-3 py-2">
                  {entry.saved ? (
                    <>
                      <Check aria-hidden="true" className="size-4" />
                      <span className="sr-only">Sim</span>
                    </>
                  ) : (
                    <span className="text-muted-foreground">
                      <span aria-hidden="true">—</span>
                      <span className="sr-only">Não</span>
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
