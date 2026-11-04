/**
 * Os runs lado a lado, de cima para baixo pela ordem em que se lê uma comparação:
 *
 * 1. quem se compara: um cartão por run, com a cor, o modelo, os dados e a métrica principal;
 * 2. a leitura da comparação (do backend);
 * 3. as métricas, com o melhor marcado. Quando os runs não têm as mesmas métricas (tarefas
 *    diferentes), as linhas arrumam-se por quem as tem ("Em comum", "Só A"…), em vez de uma
 *    tabela cheia de traços;
 * 4. o que muda entre os runs, arrumado pelos grupos que o backend dá (o run, a preparação, os
 *    hiperparâmetros…), e as curvas da loss.
 *
 * Cada run tem uma cor, pela letra que o backend lhe dá (A, B, C…): a mesma nos cartões, nas
 * tabelas e nos gráficos. As métricas em percentagem que mais de um run tem desenham-se como
 * pontos numa linha de 0 a 100% (uma linha por métrica), para se ver a distância entre os runs.
 */
import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { Star } from "lucide-react";
import type { ReactNode } from "react";

import { comparisonQuery } from "@/api/comparison";
import type {
  ChoiceGroup,
  ConfigDifference,
  MetricComparison,
  RunComparison,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { DotPlot } from "@/components/charts/dot-plot";
import { LineChart } from "@/components/charts/line-chart";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Panel } from "@/components/shell/blocks";
import { Skeleton } from "@/components/ui/skeleton";
import { categoryColor } from "@/lib/colors";
import { formatPercent, formatStatistic } from "@/lib/format";

import { comparedValue, primaryValue } from "./values";

/** O cabeçalho de uma coluna das tabelas da comparação. */
const HEAD = "px-3 py-2.5 text-xs font-medium text-dim";
/** A linha que abre um grupo de linhas numa tabela (as métricas de um run, um grupo de escolhas). */
const GROUP = "px-3 pt-5 pb-2 text-left text-xs font-medium text-dim";

export function ComparisonView({ runIds }: { runIds: readonly string[] }) {
  const comparison = useQuery(comparisonQuery(runIds));
  if (runIds.length < 2) {
    return (
      <p className="text-sm text-soft">
        Choose two to five runs in the table to see them side by side.
      </p>
    );
  }
  if (comparison.error !== null) {
    return <ErrorNote error={comparison.error} />;
  }
  if (comparison.data === undefined) {
    return <Skeleton className="h-72 w-full" />;
  }
  return <Compared comparison={comparison.data} />;
}

/** A letra de um run ("A") e o resto do nome, a partir do nome curto que o backend dá. */
function splitLabel(label: string): { letter: string; name: string } {
  const [letter = label, ...rest] = label.split(" · ");
  return { letter, name: rest.join(" · ") };
}

function Compared({ comparison }: { comparison: RunComparison }) {
  const label = useLabel();
  // Só as métricas em percentagem que mais de um run tem: com um só ponto não há distância a ver.
  const percent = comparison.metrics.filter(
    (metric) => metric.as_percent && metric.values.filter((value) => value !== null).length > 1,
  );
  const spread = (() => {
    const losses = comparison.losses.flatMap((series) => series.losses).filter((v) => v > 0);
    return losses.length === 0 ? 1 : Math.max(...losses) / Math.min(...losses);
  })();
  const withoutLoss = comparison.labels.filter(
    (name) => !comparison.losses.some((series) => series.label === name),
  );

  return (
    <div className="flex flex-col gap-8">
      <RunCards comparison={comparison} />

      {comparison.notices.length > 0 && <NoticeList notices={comparison.notices} />}

      <Panel title={`The metrics on the ${label("Partition", comparison.partition).toLowerCase()}`}>
        <div
          className={cn(
            "grid gap-x-12 gap-y-8",
            percent.length > 0 &&
              "@5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] @5xl/main:items-start",
          )}
        >
          <MetricsTable comparison={comparison} />
          {percent.length > 0 && (
            <DotPlot
              rows={percent.map((metric) => ({
                key: metric.name,
                label: metric.display_name,
                min: 0,
                max: 1,
                values: metric.values,
              }))}
              series={comparison.labels.map((name, index) => ({
                key: name,
                label: splitLabel(name).letter,
                color: categoryColor(index),
              }))}
              format={(value) => formatPercent(value)}
              description="The percentage metrics the runs share, on a 0 to 100% line per metric."
            />
          )}
        </div>
      </Panel>

      <div
        className={cn(
          "grid gap-x-12 gap-y-10",
          // A tabela das escolhas tem uma coluna por run: fica com mais espaço do que a curva.
          comparison.losses.length > 0 && "@5xl/main:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]",
        )}
      >
        <Panel title="What changes between the runs">
          {comparison.differences.length === 0 ? (
            <p className="text-sm text-soft">
              The runs have exactly the same configuration and the same seed.
            </p>
          ) : (
            <DifferencesTable comparison={comparison} />
          )}
        </Panel>

        {comparison.losses.length > 0 && (
          <Panel title="The loss over the training">
            <LineChart
              series={comparison.losses.map((series) => {
                const index = comparison.labels.indexOf(series.label);
                return {
                  key: series.label,
                  label: series.label,
                  color: categoryColor(index < 0 ? 0 : index),
                  points: series.steps.map((step, i) => ({ x: step, y: series.losses[i] ?? null })),
                };
              })}
              // Losses de ordens de grandeza diferentes só se comparam em escala logarítmica.
              logY={spread > 100}
              xLabel="Training step"
              yLabel="Loss"
              formatY={formatStatistic}
              description="Each run's loss over the training."
            />
            {withoutLoss.length > 0 && (
              <p className="mt-3 text-xs text-dim">
                No curve: {withoutLoss.map((name) => splitLabel(name).letter).join(", ")}{" "}
                (didn&apos;t train in steps).
              </p>
            )}
          </Panel>
        )}
      </div>
    </div>
  );
}

/** Um cartão por run: a letra na cor dele, o nome, o modelo, os dados e a métrica principal. */
function RunCards({ comparison }: { comparison: RunComparison }) {
  const label = useLabel();
  return (
    <ul
      aria-label="The runs compared"
      className="grid gap-3 @2xl/main:grid-cols-2 @6xl/main:grid-cols-[repeat(auto-fit,minmax(14rem,1fr))]"
    >
      {comparison.runs.map((run, index) => {
        const { letter, name } = splitLabel(comparison.labels[index] ?? run.name);
        const color = categoryColor(index);
        return (
          <li
            key={run.run_id}
            className="relative flex min-w-0 flex-col gap-3 overflow-hidden panel p-4"
          >
            {/* A cor do run, num filete em cima: a mesma das tabelas e dos gráficos. */}
            <span
              aria-hidden="true"
              className="absolute inset-x-0 top-0 h-0.5"
              style={{ background: color }}
            />
            <div className="flex min-w-0 items-center gap-2.5">
              <Letter letter={letter} color={color} />
              <span className="truncate font-medium text-strong" title={name}>
                {name}
              </span>
            </div>
            <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1 text-sm">
              <Fact term="Model">{run.model_display_name}</Fact>
              <Fact term="Data">{run.dataset_name}</Fact>
              <Fact term="Task">{label("TaskType", run.task_type)}</Fact>
            </dl>
            {run.primary_label !== null && (
              <div className="mt-auto flex items-baseline justify-between gap-3 border-t pt-3">
                <span className="truncate text-xs text-dim">{run.primary_label}</span>
                <span className="font-mono text-lg text-strong tabular-nums">
                  {primaryValue(run)}
                </span>
              </div>
            )}
          </li>
        );
      })}
    </ul>
  );
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <>
      <dt className="text-dim">{term}</dt>
      <dd className="truncate text-soft">{children}</dd>
    </>
  );
}

/** A letra de um run num círculo com a cor dele. */
function Letter({ letter, color }: { letter: string; color: string }) {
  return (
    <span
      className="inline-flex size-6 shrink-0 items-center justify-center rounded-full border font-mono text-xs text-strong"
      style={{ borderColor: color, background: `color-mix(in oklab, ${color} 18%, transparent)` }}
    >
      {letter}
    </span>
  );
}

/** O cabeçalho das colunas dos runs: a letra de cada um, na cor dele. */
function RunHeads({ labels, align }: { labels: readonly string[]; align: "left" | "right" }) {
  return labels.map((name, index) => (
    <th key={name} scope="col" className={cn(align === "right" ? "text-right" : "text-left", HEAD)}>
      <span className="sr-only">{name}</span>
      <span aria-hidden="true">
        <Letter letter={splitLabel(name).letter} color={categoryColor(index)} />
      </span>
    </th>
  ));
}

/** Um grupo de linhas: o nome por cima e as linhas por baixo. */
interface RowGroup<T> {
  key: string;
  title: string | null;
  rows: T[];
}

/**
 * As métricas por quem as tem: primeiro as de todos, depois as de cada combinação de runs, pela
 * ordem em que aparecem. Com todas em comum, há um só grupo (sem título).
 */
function metricGroups(comparison: RunComparison): RowGroup<MetricComparison>[] {
  const letters = comparison.labels.map((name) => splitLabel(name).letter);
  const groups = new Map<string, RowGroup<MetricComparison>>();
  for (const metric of comparison.metrics) {
    const owners = metric.values.flatMap((value, index) => (value === null ? [] : [index]));
    const key = owners.join(",");
    const all = owners.length === letters.length;
    const title = all
      ? "Shared"
      : owners.length === 0
        ? "No value"
        : `Only ${owners.map((index) => letters[index]).join(" and ")}`;
    const group = groups.get(key) ?? { key, title, rows: [] };
    group.rows.push(metric);
    groups.set(key, group);
  }
  const ordered = [...groups.values()].sort(
    (a, b) => b.key.split(",").length - a.key.split(",").length,
  );
  if (ordered.length === 1 && ordered[0] !== undefined) {
    ordered[0].title = null;
  }
  return ordered;
}

/** As métricas de cada run, com o melhor de cada uma marcado com uma estrela. */
function MetricsTable({ comparison }: { comparison: RunComparison }) {
  const columns = comparison.labels.length + 1;
  return (
    <div className="relative max-w-full overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">Each run's metrics, with the best one marked</caption>
        <thead>
          <tr className="border-b">
            <th scope="col" className={cn("text-left", HEAD)}>
              Metric
            </th>
            <RunHeads labels={comparison.labels} align="right" />
          </tr>
        </thead>
        {metricGroups(comparison).map((group) => (
          <tbody key={group.key}>
            {group.title !== null && (
              <tr>
                <th scope="colgroup" colSpan={columns} className={GROUP}>
                  {group.title}
                </th>
              </tr>
            )}
            {group.rows.map((metric) => (
              <tr key={metric.name} className="border-b border-border/60 last:border-0">
                <th scope="row" className="px-3 py-2 text-left font-normal text-soft">
                  {metric.display_name}
                </th>
                {metric.values.map((value, index) => {
                  const best = metric.best === index;
                  return (
                    <td
                      key={index}
                      className={cn(
                        "px-3 py-2 text-right tabular-nums",
                        best
                          ? "font-medium text-strong"
                          : value === null
                            ? "text-dim"
                            : "text-soft",
                      )}
                    >
                      <span className="inline-flex items-center justify-end gap-1">
                        {best && (
                          <>
                            <Star aria-hidden="true" className="size-3 fill-current" />
                            <span className="sr-only">The best: </span>
                          </>
                        )}
                        {comparedValue(value, metric.as_percent)}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        ))}
      </table>
    </div>
  );
}

/** As escolhas que mudam, arrumadas pelo grupo que o backend dá (o run, a preparação…). */
function DifferencesTable({ comparison }: { comparison: RunComparison }) {
  const label = useLabel();
  const columns = comparison.labels.length + 1;
  const groups: RowGroup<ConfigDifference>[] = [];
  for (const difference of comparison.differences) {
    const last = groups.at(-1);
    if (last?.key === difference.group) {
      last.rows.push(difference);
    } else {
      groups.push({
        key: difference.group,
        title: label("ChoiceGroup", difference.group satisfies ChoiceGroup),
        rows: [difference],
      });
    }
  }
  return (
    <div className="relative max-w-full overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">The choices that change from one run to another</caption>
        <thead>
          <tr className="border-b">
            <th scope="col" className={cn("text-left", HEAD)}>
              Choice
            </th>
            <RunHeads labels={comparison.labels} align="left" />
          </tr>
        </thead>
        {groups.map((group) => (
          <tbody key={group.key}>
            <tr>
              <th scope="colgroup" colSpan={columns} className={GROUP}>
                {group.title}
              </th>
            </tr>
            {group.rows.map((difference) => (
              <tr key={difference.label} className="border-b border-border/60 last:border-0">
                <th scope="row" className="px-3 py-2 text-left font-normal text-soft">
                  {difference.name}
                </th>
                {difference.values.map((value, index) => (
                  <td
                    key={index}
                    className={cn(
                      "px-3 py-2 tabular-nums",
                      value === "—" ? "text-dim" : "text-strong",
                    )}
                  >
                    {value}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        ))}
      </table>
    </div>
  );
}
