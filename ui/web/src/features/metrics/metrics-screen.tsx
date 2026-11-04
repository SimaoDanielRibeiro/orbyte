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
import { Explain } from "@/components/explain/explain";
import { Prose } from "@/components/markdown/prose";
import { ErrorNote } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
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
const ALL_TASKS = "all";

export function MetricsScreen() {
  const label = useLabel();
  const vocabulary = useVocabulary();
  const navigate = useNavigate();
  const search = useSearch({ from: "/metrics" });
  const hash = useLocation({ select: (location) => location.hash });
  const task = (search.task as TaskType | undefined) ?? null;
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
      <PageHeader
        title={SCREENS.metrics.title}
        description={SCREENS.metrics.description}
        topics={["train_vs_test", "cross_validation"]}
      />
      <div className="mt-8 flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <label htmlFor="metricas-tarefa" className="small-label">
            Task
          </label>
          <Select
            value={task ?? ALL_TASKS}
            onValueChange={(value) => {
              void navigate({
                to: "/metrics",
                search: value === ALL_TASKS ? {} : { task: value },
              });
            }}
          >
            <SelectTrigger id="metricas-tarefa" className="min-w-64">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_TASKS}>All tasks</SelectItem>
              {vocabulary.tasks.map((one) => (
                <SelectItem key={one.task_type} value={one.task_type}>
                  {label("TaskType", one.task_type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {metrics.data !== undefined && (
          <p role="status" className="text-xs text-dim tabular-nums">
            {formatQuantity(metrics.data.length, "metric", "metrics")}
          </p>
        )}
      </div>
      <ErrorNote error={metrics.error} className="mt-8" />
      {metrics.data === undefined ? (
        metrics.error === null && <Skeleton className="mt-8 h-96 w-full" />
      ) : (
        <>
          <nav aria-label="The metrics" className="mt-6">
            <ul className="flex flex-wrap gap-2">
              {metrics.data.map((card) => (
                <li key={card.name}>
                  <Link
                    to="/metrics"
                    search={search}
                    hash={metricAnchor(card.name)}
                    className="block rounded-full border border-[#2a2a2a] bg-surface px-3 py-1 text-xs text-soft outline-none hover:border-[#3a3a3a] hover:text-strong focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {card.display_name}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
          <div className="mt-6 grid gap-x-12 gap-y-10 @5xl/main:grid-cols-2">
            {metrics.data.map((card) => (
              <MetricArticle key={card.name} card={card} />
            ))}
          </div>
        </>
      )}
    </Page>
  );
}

/**
 * Uma métrica num cartão: o nome e a fórmula à vista; o que mede, como se lê e para que tarefas
 * serve, no "?" (ADR 0038).
 */
function MetricArticle({ card }: { card: MetricCard }) {
  const label = useLabel();
  const tasks = card.task_types.map((one) => label("TaskType", one));
  return (
    <article
      id={metricAnchor(card.name)}
      aria-labelledby={`${metricAnchor(card.name)}-titulo`}
      className="flex min-w-0 scroll-mt-6 flex-col gap-5"
    >
      <header className="flex items-center gap-1">
        <h2 id={`${metricAnchor(card.name)}-titulo`} className="text-lg">
          {card.display_name}
        </h2>
        <Explain title={card.display_name}>
          <p className="font-medium text-strong">What it measures</p>
          <Prose text={card.what} size="small" className="mt-1" />
          <p className="mt-3 font-medium text-strong">How to read it</p>
          <Prose text={card.how_to_read} size="small" className="mt-1" />
          <p className="mt-3 font-medium text-strong">Where it's used</p>
          <p className="mt-1">
            {formatList(tasks)}. {directionText(card.higher_is_better)}
            {card.as_percent ? " As a percentage." : ""}
            {card.needs_proba ? " Needs the probabilities." : ""}
          </p>
        </Explain>
      </header>
      <FormulaBlock card={card} />
    </article>
  );
}
