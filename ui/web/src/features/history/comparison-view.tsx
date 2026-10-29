/**
 * Os runs lado a lado: a leitura da comparação (do backend), as métricas (com o melhor
 * marcado), as escolhas que mudam de um run para outro e as curvas da loss, cada uma no seu
 * cartão.
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
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Panel } from "@/components/shell/blocks";
import { Skeleton } from "@/components/ui/skeleton";
import { categoryColor } from "@/lib/colors";
import { formatPercent, formatStatistic } from "@/lib/format";

import { comparedValue } from "./values";

/** O cabeçalho de uma coluna das tabelas da comparação. */
const HEAD = "px-3 py-2.5 font-mono text-xs font-normal text-dim";

export function ComparisonView({ runIds }: { runIds: readonly string[] }) {
  const comparison = useQuery(comparisonQuery(runIds));
  if (runIds.length < 2) {
    return (
      <p className="text-sm text-soft">
        Escolhe dois a cinco runs na tabela para os ver lado a lado.
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

function Compared({ comparison }: { comparison: RunComparison }) {
  const label = useLabel();
  const percent = comparison.metrics.filter((metric) => metric.as_percent);
  const spread = (() => {
    const losses = comparison.losses.flatMap((series) => series.losses).filter((v) => v > 0);
    return losses.length === 0 ? 1 : Math.max(...losses) / Math.min(...losses);
  })();

  return (
    <div className="flex flex-col gap-4">
      <ul aria-label="Os runs comparados" className="flex flex-wrap gap-2">
        {comparison.labels.map((name, index) => (
          <li
            key={name}
            // A cor de cada run (a dos gráficos) é o contorno da etiqueta.
            className="flex items-center rounded-full border bg-surface px-3 py-1 text-xs text-soft"
            style={{ borderColor: categoryColor(index) }}
          >
            {name}
          </li>
        ))}
      </ul>

      {comparison.notices.length > 0 && (
        <Panel title="A leitura da comparação">
          <NoticeList notices={comparison.notices} />
        </Panel>
      )}

      <Panel title={`As métricas no ${label("Partition", comparison.partition).toLowerCase()}`}>
        <div className="grid gap-6 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] @5xl/main:items-start">
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
      </Panel>

      <div
        className={cn(
          "grid gap-x-12 gap-y-10",
          comparison.losses.length > 0 && "@5xl/main:grid-cols-2",
        )}
      >
        <Panel title="O que muda entre os runs">
          {comparison.differences.length === 0 ? (
            <p className="text-sm text-soft">
              Os runs têm exatamente a mesma configuração e a mesma seed.
            </p>
          ) : (
            <div className="relative max-w-full overflow-x-auto">
              <table className="w-full text-sm">
                <caption className="sr-only">As escolhas que mudam de um run para outro</caption>
                <thead>
                  <tr className="border-b">
                    <th scope="col" className={cn("text-left", HEAD)}>
                      Escolha
                    </th>
                    {comparison.labels.map((name) => (
                      <th key={name} scope="col" className={cn("text-left", HEAD)}>
                        {name.split(" · ")[0]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {comparison.differences.map((difference) => (
                    <tr key={difference.label} className="border-b border-border/60 last:border-0">
                      <th scope="row" className="px-3 py-2 text-left font-normal text-soft">
                        {difference.label}
                      </th>
                      {difference.values.map((value, index) => (
                        <td key={index} className="px-3 py-2 text-strong tabular-nums">
                          {value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        {comparison.losses.length > 0 && (
          <Panel title="A loss ao longo do treino">
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
          </Panel>
        )}
      </div>
    </div>
  );
}

/** As métricas de cada run, com o melhor de cada uma marcado com uma estrela. */
function MetricsTable({ comparison }: { comparison: RunComparison }) {
  return (
    <div className="relative max-w-full overflow-x-auto">
      <table className="w-full text-sm">
        <caption className="sr-only">As métricas de cada run, com a melhor marcada</caption>
        <thead>
          <tr className="border-b">
            <th scope="col" className={cn("text-left", HEAD)}>
              Métrica
            </th>
            {comparison.labels.map((name, index) => (
              <th key={name} scope="col" className={cn("text-right", HEAD)}>
                <span
                  className="inline-flex items-center rounded-full border px-2 py-0.5"
                  style={{ borderColor: categoryColor(index) }}
                >
                  {name.split(" · ")[0]}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {comparison.metrics.map((metric) => (
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
                      best ? "font-medium text-strong" : "text-soft",
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
