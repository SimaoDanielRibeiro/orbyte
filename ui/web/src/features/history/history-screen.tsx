/**
 * O ecrã Histórico: o histórico dos runs (cada treino que termina fica registado sozinho),
 * com os filtros, a comparação de dois a cinco runs e a gaveta de cada um (abrir, reproduzir,
 * guardar, mudar o nome, apagar).
 *
 * Apagar está também no próprio ecrã (ADR 0054): o caixote do lixo ao fim de cada linha apaga
 * esse run, e "Delete the selected" (ao lado de "Clear the selection") apaga os escolhidos. As
 * duas pedem confirmação, porque não se desfazem.
 *
 * Os filtros ficam no endereço (`?tarefa=…&dataset=…&modelo=…`), e a API filtra; as opções de
 * cada filtro são os valores que o histórico tem. Os runs a comparar escolhem-se na tabela.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { cn } from "cn";
import { Check, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import type { RunEntry, TaskType } from "@/api/generated";
import { historyRunsOptions } from "@/api/generated/@tanstack/react-query.gen";
import { deleteRuns } from "@/api/history";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { ErrorNote, NeedsStep, Section } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
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

import { invalidateHistory } from "./cache";
import { ComparisonView } from "./comparison-view";
import { RunSheet } from "./run-sheet";
import { MAX_COMPARED, primaryValue } from "./values";

/** O valor de um filtro que não filtra nada. */
const ALL = "all";

export function HistoryScreen() {
  const search = useSearch({ from: "/history" });
  const navigate = useNavigate();
  const task = search.task === undefined ? undefined : (search.task as TaskType);
  const everything = useQuery({ ...historyRunsOptions(), staleTime: 0 });
  const filtered = useQuery({
    ...historyRunsOptions({
      query: { task_type: task, dataset_name: search.dataset, model_name: search.model },
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

  const setFilter = (key: "task" | "dataset" | "model", value: string) => {
    void navigate({
      to: "/history",
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
  // Um run apagado sai da comparação e, se estava aberto, a gaveta fecha-se.
  const forget = (runIds: readonly string[]) => {
    setChosen((previous) => previous.filter((id) => !runIds.includes(id)));
    setOpened((previous) => (previous !== null && runIds.includes(previous) ? null : previous));
  };
  const chosenEntries = (everything.data ?? []).filter((entry) => chosen.includes(entry.run_id));

  return (
    <Page>
      <PageHeader
        title={SCREENS.history.title}
        description={SCREENS.history.description}
        topics={["experiment_tracking", "comparing_runs", "reproducibility"]}
      />
      <ErrorNote error={everything.error ?? filtered.error} className="mt-8" />
      {everything.data === undefined ? (
        everything.error === null && <Skeleton className="mt-10 h-64 w-full" />
      ) : everything.data.length === 0 ? (
        <NeedsStep
          message="No runs in the history yet: every training that finishes is recorded here on its own."
          to="/models"
          action="Choose a model"
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
            onDeleted={forget}
            loading={filtered.data === undefined}
          />

          <Section
            title="Compare runs"
            topic="comparing_runs"
            actions={
              chosen.length > 0 && (
                // As ações sobre os escolhidos ficam juntas, aqui: na linha dos filtros, mudavam a
                // altura da página a cada caixa marcada.
                <div className="flex flex-wrap items-start gap-2">
                  <DeleteRuns entries={chosenEntries} onDeleted={forget} />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setChosen([]);
                    }}
                  >
                    Clear the selection ({chosen.length})
                  </Button>
                </div>
              )
            }
          >
            <ComparisonView runIds={chosen} />
          </Section>
        </>
      )}
      <RunSheet
        entry={openedEntry}
        onClose={() => {
          setOpened(null);
        }}
        onDeleted={(runId) => {
          forget([runId]);
        }}
      />
    </Page>
  );
}

/**
 * Apagar os runs escolhidos na tabela, de uma vez, depois de os ver na confirmação. Só aparece
 * com algum escolhido.
 */
function DeleteRuns({
  entries,
  onDeleted,
}: {
  entries: readonly RunEntry[];
  onDeleted: (runIds: readonly string[]) => void;
}) {
  const queryClient = useQueryClient();
  const remove = useMutation({
    mutationFn: deleteRuns,
    onSettled: () => {
      invalidateHistory(queryClient);
    },
    onSuccess: (_, runIds) => {
      onDeleted(runIds);
      toast.success(`${formatQuantity(runIds.length, "run", "runs")} deleted from the history.`);
    },
  });
  if (entries.length === 0) {
    return null;
  }
  const runIds = entries.map((entry) => entry.run_id);
  return (
    <div className="flex flex-col items-end gap-2">
      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button variant="destructive" size="sm" disabled={remove.isPending}>
            <Trash2 aria-hidden="true" />
            Delete the selected ({entries.length})
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Delete {formatQuantity(entries.length, "run", "runs")}?
            </AlertDialogTitle>
            <AlertDialogDescription>
              They leave the history with their files (the model and the step-by-step training, if
              they were saved). This can&apos;t be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <ul className="flex flex-col gap-1 text-sm text-strong">
            {entries.map((entry) => (
              <li key={entry.run_id} className="truncate">
                {entry.name}
              </li>
            ))}
          </ul>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                remove.mutate(runIds);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <ErrorNote error={remove.error} />
    </div>
  );
}

/** O caixote do lixo de uma linha: apaga esse run, depois de confirmar. */
function DeleteRunButton({
  entry,
  onDeleted,
}: {
  entry: RunEntry;
  onDeleted: (runIds: readonly string[]) => void;
}) {
  const queryClient = useQueryClient();
  const remove = useMutation({
    mutationFn: deleteRuns,
    onSettled: () => {
      invalidateHistory(queryClient);
    },
    onSuccess: (_, runIds) => {
      onDeleted(runIds);
      toast.success("Run deleted from the history.");
    },
    onError: (error) => {
      toast.error(error.message);
    },
  });
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={`Delete “${entry.name}”`}
          disabled={remove.isPending}
          className="text-dim hover:bg-destructive/10 hover:text-destructive"
        >
          <Trash2 aria-hidden="true" />
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete “{entry.name}”?</AlertDialogTitle>
          <AlertDialogDescription>
            The run leaves the history with its files (the model and the step-by-step training, if
            it was saved). This can&apos;t be undone.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={() => {
              remove.mutate([entry.run_id]);
            }}
          >
            Delete
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

interface HistorySearch {
  task?: string;
  dataset?: string;
  model?: string;
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
  onChange: (key: "task" | "dataset" | "model", value: string) => void;
}) {
  const label = useLabel();
  const tasks = [...new Set(all.map((entry) => entry.task_type))];
  const datasets = [...new Set(all.map((entry) => entry.dataset_name))].sort((a, b) =>
    a.localeCompare(b, "en"),
  );
  const models = new Map(all.map((entry) => [entry.model_name, entry.model_display_name]));
  return (
    <div className="mt-8 flex flex-wrap items-end justify-between gap-4">
      <div className="flex flex-wrap gap-4">
        <Filter
          id="historico-tarefa"
          label="Task"
          value={search.task ?? ALL}
          options={tasks.map((task) => [task, label("TaskType", task)])}
          onChange={(value) => {
            onChange("task", value);
          }}
        />
        <Filter
          id="historico-dataset"
          label="Dataset or environment"
          value={search.dataset ?? ALL}
          options={datasets.map((name) => [name, name])}
          onChange={(value) => {
            onChange("dataset", value);
          }}
        />
        <Filter
          id="historico-modelo"
          label="Model or agent"
          value={search.model ?? ALL}
          options={[...models].sort((a, b) => a[1].localeCompare(b[1], "en"))}
          onChange={(value) => {
            onChange("model", value);
          }}
        />
      </div>
      {count !== null && (
        <div className="flex items-center gap-1">
          <p role="status" className="text-xs text-dim tabular-nums">
            {formatQuantity(count, "run", "runs")}
          </p>
          <Explain topic="run_history" />
        </div>
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
      <label htmlFor={id} className="small-label">
        {label}
      </label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger id={id} className="min-w-48">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={ALL}>All</SelectItem>
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

/**
 * A tabela dos runs: o nome abre a gaveta; a caixa escolhe o run para comparar.
 *
 * Cada célula tem uma linha principal e, por baixo, o pormenor em letra pequena (a tarefa por
 * baixo do dataset, o nome da métrica por baixo do valor): assim cabe num portátil sem o nome
 * partir em três linhas nem a tabela deslizar para o lado. A seed só aparece num ecrã largo
 * (está também na gaveta do run). A última coluna apaga o run.
 */
function RunsTable({
  entries,
  chosen,
  onToggle,
  onOpen,
  onDeleted,
  loading,
}: {
  entries: readonly RunEntry[];
  chosen: readonly string[];
  onToggle: (runId: string, on: boolean) => void;
  onOpen: (runId: string) => void;
  onDeleted: (runIds: readonly string[]) => void;
  loading: boolean;
}) {
  const label = useLabel();
  if (loading) {
    return <Skeleton className="mt-4 h-48 w-full" />;
  }
  if (entries.length === 0) {
    return <p className="mt-4 text-sm text-soft">No run matches these filters.</p>;
  }
  const full = chosen.length >= MAX_COMPARED;
  const head = "px-3 py-2.5 text-xs font-medium text-dim";
  return (
    // `relative`: os textos só para leitores de ecrã das células (posição absoluta) ficam presos
    // aqui; sem ele, saíam da caixa e a página toda deslizava para o lado no telemóvel.
    <div className="relative mt-4 max-w-full overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">
          The runs in the history, from the newest to the oldest
        </caption>
        <thead>
          <tr className="border-b text-left">
            <th scope="col" className={cn("w-10", head)}>
              <span className="sr-only">Compare</span>
            </th>
            <th scope="col" className={cn("whitespace-nowrap", head)}>
              When
            </th>
            <th scope="col" className={head}>
              Name
            </th>
            <th scope="col" className={head}>
              Dataset and task
            </th>
            <th scope="col" className={cn("text-right whitespace-nowrap", head)}>
              Main metric
            </th>
            <th scope="col" className={cn("hidden text-right @7xl/main:table-cell", head)}>
              Seed
            </th>
            <th scope="col" className={head}>
              Saved
            </th>
            <th scope="col" className={cn("sticky right-0 w-12 bg-background", head)}>
              <span className="sr-only">Delete</span>
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
                  "group/row border-b border-border/60 transition-colors last:border-0 hover:bg-surface-high/50",
                  on && "bg-surface-high hover:bg-surface-high",
                )}
              >
                <td className="px-3 py-2">
                  <Checkbox
                    checked={on}
                    disabled={!on && full}
                    aria-label={`Compare “${entry.name}”`}
                    onCheckedChange={(checked) => {
                      onToggle(entry.run_id, checked === true);
                    }}
                  />
                </td>
                <td className="px-3 py-2.5 font-mono text-xs whitespace-nowrap text-dim tabular-nums">
                  {formatMoment(entry.created_at)}
                </td>
                <td className="min-w-56 px-3 py-3">
                  <button
                    type="button"
                    onClick={() => {
                      onOpen(entry.run_id);
                    }}
                    className="rounded-sm text-left font-medium text-strong underline-offset-4 outline-none hover:underline focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {entry.name}
                  </button>
                </td>
                <td className="px-3 py-3">
                  <span className="block text-soft">{entry.dataset_name}</span>
                  <span className="block text-xs whitespace-nowrap text-dim">
                    {label("TaskType", entry.task_type)}
                  </span>
                </td>
                <td className="px-3 py-3 text-right whitespace-nowrap">
                  <span className="block text-strong tabular-nums">{primaryValue(entry)}</span>
                  {entry.primary_label !== null && (
                    <span className="block text-xs text-dim">{entry.primary_label}</span>
                  )}
                </td>
                <td className="hidden px-3 py-3 text-right font-mono text-xs text-dim tabular-nums @7xl/main:table-cell">
                  {entry.seed}
                </td>
                <td className="px-3 py-2.5">
                  {entry.saved ? (
                    <>
                      <Check aria-hidden="true" className="size-4 text-positive" />
                      <span className="sr-only">Yes</span>
                    </>
                  ) : (
                    <span className="text-dim">
                      <span aria-hidden="true">—</span>
                      <span className="sr-only">No</span>
                    </span>
                  )}
                </td>
                {/* Presa à direita: num portátil, a tabela desliza para o lado dentro da caixa, e
                    o caixote do lixo fica à vista. O fundo opaco é o da linha (o mesmo cinzento
                    por baixo do rato, ou o de escolhida), para o que desliza não se ver por baixo. */}
                <td
                  className={cn(
                    "sticky right-0 px-2 py-2 text-right transition-colors",
                    on ? "bg-surface-high" : "bg-background group-hover/row:bg-[#1b1b1b]",
                  )}
                >
                  <DeleteRunButton entry={entry} onDeleted={onDeleted} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
