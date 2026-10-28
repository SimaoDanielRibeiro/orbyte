/**
 * O ecrã Métricas: o glossário das métricas do laboratório. Cada uma com o que mede, a fórmula
 * (com o significado de cada símbolo), como se lê e para que lado é melhor.
 *
 * As métricas filtram-se pela tarefa (`?tarefa=…`, como nos Modelos), e cada uma tem a sua
 * âncora (`#metrica-accuracy`): a gaveta de uma métrica no ecrã Treinar e avaliar liga-se aqui.
 */
import { useQuery } from "@tanstack/react-query";
import { Link, useLocation, useNavigate, useSearch } from "@tanstack/react-router";
import { useEffect } from "react";

import type { MetricCard, TaskType } from "@/api/generated";
import { catalogMetricsOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { Prose } from "@/components/markdown/prose";
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
import { formatList, formatQuantity } from "@/lib/format";

import { directionText } from "./direction";
import { FormulaBlock } from "./formula-block";
import { metricAnchor } from "./names";

/** O valor do filtro que mostra todas as tarefas. */
const ALL_TASKS = "todas";

export function MetricsScreen() {
  const label = useLabel();
  const vocabulary = useVocabulary();
  const navigate = useNavigate();
  const search = useSearch({ from: "/metricas" });
  const hash = useLocation({ select: (location) => location.hash });
  const task = (search.tarefa as TaskType | undefined) ?? null;
  const metrics = useQuery({
    ...catalogMetricsOptions({ query: task === null ? undefined : { task_type: task } }),
    staleTime: "static",
  });

  // Uma ligação para uma métrica (`#metrica-f1`) leva até ela quando o glossário chega.
  useEffect(() => {
    const anchor = hash.replace(/^#/, "");
    if (metrics.data !== undefined && anchor !== "") {
      document.getElementById(anchor)?.scrollIntoView({ block: "start" });
    }
  }, [metrics.data, hash]);

  return (
    <Page>
      <PageHeader title={SCREENS.metrics.title} description={SCREENS.metrics.description}>
        <WhyButton topics={["train_vs_test", "cross_validation"]} />
      </PageHeader>
      <div className="mt-10 flex flex-wrap items-end justify-between gap-4 border-b pb-5">
        <div className="flex flex-col gap-1.5">
          <label htmlFor="metricas-tarefa" className="text-xs text-muted-foreground">
            Tarefa
          </label>
          <Select
            value={task ?? ALL_TASKS}
            onValueChange={(value) => {
              void navigate({
                to: "/metricas",
                search: value === ALL_TASKS ? {} : { tarefa: value },
              });
            }}
          >
            <SelectTrigger id="metricas-tarefa" className="min-w-64">
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
        {metrics.data !== undefined && (
          <p role="status" className="text-sm text-muted-foreground">
            {formatQuantity(metrics.data.length, "métrica", "métricas")}
          </p>
        )}
      </div>
      <ErrorNote error={metrics.error} className="mt-8" />
      {metrics.data === undefined ? (
        metrics.error === null && <Skeleton className="mt-8 h-96 w-full" />
      ) : (
        <>
          <nav aria-label="As métricas" className="mt-8">
            <ul className="columns-2 gap-x-8 text-sm @3xl/main:columns-3 @5xl/main:columns-4">
              {metrics.data.map((card) => (
                <li key={card.name} className="break-inside-avoid py-1">
                  <Link
                    to="/metricas"
                    search={search}
                    hash={metricAnchor(card.name)}
                    className="text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
                  >
                    {card.display_name}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
          <div className="mt-10">
            {metrics.data.map((card) => (
              <MetricArticle key={card.name} card={card} />
            ))}
          </div>
        </>
      )}
    </Page>
  );
}

function MetricArticle({ card }: { card: MetricCard }) {
  const label = useLabel();
  const tasks = card.task_types.map((one) => label("TaskType", one));
  return (
    <article
      id={metricAnchor(card.name)}
      aria-labelledby={`${metricAnchor(card.name)}-titulo`}
      className="grid scroll-mt-6 gap-8 border-t py-10 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]"
    >
      <div className="flex max-w-[68ch] min-w-0 flex-col gap-5">
        <header className="flex flex-col gap-2">
          <h2 id={`${metricAnchor(card.name)}-titulo`} className="text-3xl">
            {card.display_name}
          </h2>
          <p className="text-sm text-muted-foreground">
            {formatList(tasks)}. {directionText(card.higher_is_better)}
            {card.as_percent ? " Escreve-se em percentagem." : ""}
            {card.needs_proba ? " Precisa das probabilidades do modelo." : ""}
          </p>
        </header>
        <section className="flex flex-col gap-1.5">
          <h3 className="text-lg">O que mede</h3>
          <Prose text={card.what} />
        </section>
        <section className="flex flex-col gap-1.5">
          <h3 className="text-lg">Como se lê</h3>
          <Prose text={card.how_to_read} />
        </section>
      </div>
      <FormulaBlock card={card} />
    </article>
  );
}
