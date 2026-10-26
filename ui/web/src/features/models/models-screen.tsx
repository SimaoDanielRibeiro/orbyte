/**
 * O ecrã Modelos: todos os modelos do laboratório, agrupados por família, só com o nome e uma
 * seta. O nome é o que se leva para a página do modelo: ao abri-la, ele desliza para o título
 * (uma transição partilhada do Motion, com o mesmo `layoutId` nos dois sítios).
 *
 * Com uma preparação feita, a lista mostra só os modelos da tarefa dela (o filtro fica no
 * endereço, `?tarefa=…`, e pode mudar-se para outra tarefa ou para todas).
 */
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { ChevronRight } from "lucide-react";
import { motion } from "motion/react";

import type { ModelCard, TaskType } from "@/api/generated";
import {
  catalogModelsOptions,
  catalogSuggestedModelOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { ErrorNote } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { formatQuantity } from "@/lib/format";
import { useSession } from "@/state/session";

import { ALL_TASKS, nameLayoutId } from "./names";

export function ModelsScreen() {
  const label = useLabel();
  const vocabulary = useVocabulary();
  const navigate = useNavigate();
  const search = useSearch({ from: "/modelos/" });
  const preparation = useSession((state) => state.preparation);
  const task: TaskType | null =
    search.tarefa === ALL_TASKS
      ? null
      : ((search.tarefa as TaskType | undefined) ?? preparation?.taskType ?? null);
  const models = useQuery({
    ...catalogModelsOptions({ query: task === null ? undefined : { task_type: task } }),
    staleTime: "static",
  });
  const suggested = useQuery({
    ...catalogSuggestedModelOptions({ query: { task_type: task ?? "binary_classification" } }),
    enabled: task !== null,
    staleTime: "static",
  });

  return (
    <Page>
      <PageHeader title={SCREENS.models.title} description={SCREENS.models.description}>
        <WhyButton topics={["model_selection", "bias_variance"]} />
      </PageHeader>
      <div className="mt-10 flex flex-wrap items-end justify-between gap-4 border-b pb-5">
        <div className="flex flex-col gap-1.5">
          <label htmlFor="modelos-tarefa" className="text-xs text-muted-foreground">
            Tarefa
          </label>
          <Select
            value={task ?? ALL_TASKS}
            onValueChange={(value) => {
              void navigate({ to: "/modelos", search: { tarefa: value } });
            }}
          >
            <SelectTrigger id="modelos-tarefa" className="min-w-64">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_TASKS}>Todas as tarefas</SelectItem>
              {vocabulary.tasks.map((one) => (
                <SelectItem key={one.task_type} value={one.task_type}>
                  {label("TaskType", one.task_type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {models.data !== undefined && (
          <p className="text-sm text-muted-foreground" role="status">
            {formatQuantity(models.data.length, "modelo", "modelos")}
            {preparation !== null && task === preparation.taskType
              ? ", os da tarefa que preparaste"
              : ""}
          </p>
        )}
      </div>
      <ErrorNote error={models.error} className="mt-8" />
      {models.data === undefined ? (
        models.error === null && <ListSkeleton />
      ) : (
        <Families models={models.data} suggested={suggested.data ?? null} />
      )}
    </Page>
  );
}

/** Os modelos por família, pela ordem do catálogo; cada família não se parte entre colunas. */
function Families({
  models,
  suggested,
}: {
  models: readonly ModelCard[];
  suggested: string | null;
}) {
  const label = useLabel();
  const families: { family: ModelCard["family"]; members: ModelCard[] }[] = [];
  for (const model of models) {
    const last = families.at(-1);
    if (last?.family === model.family) {
      last.members.push(model);
    } else {
      families.push({ family: model.family, members: [model] });
    }
  }
  return (
    <div className="mt-8 gap-x-16 @4xl/main:columns-2">
      {families.map(({ family, members }) => (
        <section
          key={family}
          aria-labelledby={`familia-${family}`}
          className="mb-10 break-inside-avoid"
        >
          <h2
            id={`familia-${family}`}
            className="flex items-baseline justify-between border-b pb-2 font-sans text-sm text-muted-foreground"
          >
            {label("ModelFamily", family)}
            <span className="tabular-nums">{members.length}</span>
          </h2>
          <ul>
            {members.map((model) => (
              <li key={model.name}>
                <Link
                  to="/modelos/$name"
                  params={{ name: model.name }}
                  className="group -mx-3 flex items-center justify-between gap-4 rounded-md px-3 py-2.5 outline-none hover:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50"
                >
                  <motion.span
                    layoutId={nameLayoutId(model.name)}
                    className="inline-block font-serif text-xl"
                  >
                    {model.display_name}
                  </motion.span>
                  <span className="flex shrink-0 items-center gap-3">
                    {model.name === suggested && (
                      <span className="text-xs text-muted-foreground">sugerido para começar</span>
                    )}
                    <ChevronRight
                      aria-hidden="true"
                      className="size-5 text-muted-foreground transition-transform group-hover:translate-x-1 group-hover:text-foreground motion-reduce:transition-none"
                    />
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

function ListSkeleton() {
  return (
    <div
      className="mt-8 flex flex-col gap-4"
      role="status"
      aria-busy="true"
      aria-label="A carregar"
    >
      {[0, 1, 2, 3].map((index) => (
        <Skeleton key={index} className="h-10 w-full max-w-md" />
      ))}
    </div>
  );
}
