/**
 * Os runs lado a lado: a leitura da comparação (do backend), as métricas (com o melhor
 * marcado), as escolhas que mudam de um run para outro e as curvas da loss.
 *
 * Cada run tem uma cor, pela letra que o backend lhe dá (A, B, C…): a mesma no gráfico das
 * métricas e no das curvas. As métricas em percentagem desenham-se como pontos numa linha de 0 a
 * 100% (uma linha por métrica), para se ver a distância entre os runs sem empilhar barras.
 */
import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { Star } from "lucide-react";

import { comparisonQuery } from "@/api/comparison";
import type { RunComparison } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { DotPlot } from "@/components/charts/dot-plot";
import { LineChart } from "@/components/charts/line-chart";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote } from "@/components/shell/blocks";
import { Skeleton } from "@/components/ui/skeleton";
import { categoryColor } from "@/lib/colors";
import { formatPercent, formatStatistic } from "@/lib/format";

import { comparedValue } from "./values";

export function ComparisonView({ runIds }: { runIds: readonly string[] }) {
  const comparison = useQuery(comparisonQuery(runIds));
  if (runIds.length < 2) {
    return (
      <div className="flex max-w-[68ch] flex-col gap-3">
        <p className="font-serif text-reading text-muted-foreground">
          Escolhe dois a cinco runs na tabela para os ver lado a lado.
        </p>
        <TopicText topic="comparing_runs" />
      </div>
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

function Compared({ comparison }: { comparison: RunComparison }) {
  const label = useLabel();
  const percent = comparison.metrics.filter((metric) => metric.as_percent);
  const spread = (() => {
    const losses = comparison.losses.flatMap((series) => series.losses).filter((v) => v > 0);
    return losses.length === 0 ? 1 : Math.max(...losses) / Math.min(...losses);
  })();

  return (
    <div className="flex flex-col gap-10">
      <ul className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
        {comparison.labels.map((name, index) => (
          <li key={name} className="flex items-center gap-2">
            <span
              aria-hidden="true"
              className="size-2.5 rounded-full"
              style={{ background: categoryColor(index) }}
            />
            {name}
          </li>
        ))}
      </ul>

      {comparison.notices.length > 0 && (
        <section aria-labelledby="leitura-titulo" className="flex flex-col gap-3">
          <h3 id="leitura-titulo" className="text-xl">
            A leitura da comparação
          </h3>
          <NoticeList notices={comparison.notices} />
        </section>
      )}

      <section aria-labelledby="comparadas-titulo" className="flex flex-col gap-4">
        <h3 id="comparadas-titulo" className="text-xl">
          As métricas no {label("Partition", comparison.partition).toLowerCase()}
        </h3>
        <div className="grid gap-8 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
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
                label: name,
                color: categoryColor(index),
              }))}
              format={(value) => formatPercent(value)}
              description="As métricas em percentagem de cada run, numa linha de 0 a 100% por métrica."
            />
          )}
        </div>
      </section>

      <section aria-labelledby="mudancas-titulo" className="flex flex-col gap-4">
        <h3 id="mudancas-titulo" className="text-xl">
          O que muda entre os runs
        </h3>
        {comparison.differences.length === 0 ? (
          <p className="font-serif text-reading text-muted-foreground">
            Os runs têm exatamente a mesma configuração e a mesma seed.
          </p>
        ) : (
          <div className="relative max-w-full overflow-x-auto rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th scope="col" className="px-3 py-2 text-left font-medium">
                    Escolha
                  </th>
                  {comparison.labels.map((name) => (
                    <th key={name} scope="col" className="px-3 py-2 text-left font-medium">
                      {name.split(" · ")[0]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison.differences.map((difference) => (
                  <tr key={difference.label} className="border-b border-border/50 last:border-0">
                    <th scope="row" className="px-3 py-1.5 text-left font-normal">
                      {difference.label}
                    </th>
                    {difference.values.map((value, index) => (
                      <td key={index} className="px-3 py-1.5 tabular-nums">
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {comparison.losses.length > 0 && (
        <section aria-labelledby="curvas-titulo" className="flex max-w-4xl flex-col gap-4">
          <h3 id="curvas-titulo" className="text-xl">
            A loss ao longo do treino
          </h3>
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
            xLabel="Passo do treino"
            yLabel="Loss"
            formatY={formatStatistic}
            description="A loss de cada run ao longo do treino."
          />
        </section>
      )}
    </div>
  );
}

/** As métricas de cada run, com o melhor de cada uma marcado com uma estrela. */
function MetricsTable({ comparison }: { comparison: RunComparison }) {
  return (
    <div className="relative max-w-full overflow-x-auto rounded-md border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b">
            <th scope="col" className="px-3 py-2 text-left font-medium">
              Métrica
            </th>
            {comparison.labels.map((name, index) => (
              <th key={name} scope="col" className="px-3 py-2 text-right font-medium">
                <span className="inline-flex items-center gap-1.5">
                  <span
                    aria-hidden="true"
                    className="size-2 rounded-full"
                    style={{ background: categoryColor(index) }}
                  />
                  {name.split(" · ")[0]}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {comparison.metrics.map((metric) => (
            <tr key={metric.name} className="border-b border-border/50 last:border-0">
              <th scope="row" className="px-3 py-1.5 text-left font-normal">
                {metric.display_name}
              </th>
              {metric.values.map((value, index) => {
                const best = metric.best === index;
                return (
                  <td
                    key={index}
                    className={cn(
                      "px-3 py-1.5 text-right tabular-nums",
                      best ? "font-semibold text-foreground" : "text-muted-foreground",
                    )}
                  >
                    <span className="inline-flex items-center justify-end gap-1">
                      {best && (
                        <>
                          <Star aria-hidden="true" className="size-3 fill-current" />
                          <span className="sr-only">O melhor: </span>
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
      </table>
    </div>
  );
}
