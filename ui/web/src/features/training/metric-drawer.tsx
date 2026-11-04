/**
 * A gaveta de uma métrica: o valor neste run, o que mede, a fórmula (KaTeX) com o significado
 * de cada símbolo, a conta deste run e como se lê.
 *
 * A conta desenha os termos da fórmula com os valores deste run (`MetricResult.terms`), cada um
 * com uma barra à escala do maior, a aparecer um a um, e no fim o resultado: vê-se de que é
 * feito o número. As métricas que não se decompõem em poucos termos (a AUC, a silhueta…)
 * mostram o gráfico de onde o valor sai. Com "reduzir movimento", aparece tudo de uma vez.
 */
import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { RotateCcw } from "lucide-react";
import { motion, useReducedMotion } from "motion/react";
import { useState } from "react";

import type {
  EvaluationCharts,
  EvaluationReport,
  MetricResult,
  MetricTerm,
  Partition,
} from "@/api/generated";
import { catalogMetricsOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { BarChart } from "@/components/charts/bar-chart";
import { LineChart } from "@/components/charts/line-chart";
import { SilhouetteChart } from "@/components/charts/silhouette-chart";
import { Formula, Prose } from "@/components/markdown/prose";
import { TopicText } from "@/components/markdown/topic-text";
import { Facts } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { directionText } from "@/features/metrics/direction";
import { FormulaBlock } from "@/features/metrics/formula-block";
import { metricAnchor } from "@/features/metrics/names";
import { classColors } from "@/lib/classes";
import { CATEGORICAL } from "@/lib/colors";
import { formatMetric, formatNumber, formatPercent, formatStatistic } from "@/lib/format";

/** O tempo entre um termo e o seguinte, na conta. */
const STAGGER = 0.35;

interface MetricDrawerProps {
  /** A métrica aberta (`null`: a gaveta fechada). */
  metric: MetricResult | null;
  reports: Partial<Record<Partition, EvaluationReport>>;
  mainPartition: Partition;
  classLabels: readonly string[];
  charts: EvaluationCharts | undefined;
  onClose: () => void;
}

export function MetricDrawer({
  metric,
  reports,
  mainPartition,
  classLabels,
  charts,
  onClose,
}: MetricDrawerProps) {
  const label = useLabel();
  const cards = useQuery({ ...catalogMetricsOptions(), staleTime: "static" });
  const card = cards.data?.find((one) => one.name === metric?.name);
  const others = (Object.entries(reports) as [Partition, EvaluationReport][]).filter(
    ([partition]) => partition !== mainPartition,
  );

  return (
    <Sheet
      open={metric !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <SheetContent
        side="right"
        className="w-full gap-0 overflow-y-auto data-[side=right]:sm:max-w-xl"
      >
        {metric !== null && (
          <>
            <SheetHeader className="gap-2 border-b p-6">
              <SheetTitle className="text-xl font-medium text-strong">
                {metric.display_name}
              </SheetTitle>
              <SheetDescription className="text-sm text-dim">
                {directionText(metric.higher_is_better)}
              </SheetDescription>
            </SheetHeader>
            <div className="flex flex-col gap-8 p-6 pb-10">
              <Facts
                facts={[
                  {
                    label: `On the ${label("Partition", mainPartition).toLowerCase()}`,
                    value: <span className="text-3xl">{formatMetric(metric)}</span>,
                  },
                  ...others.map(([partition, report]) => {
                    const found = report.metrics.find((one) => one.name === metric.name);
                    return {
                      label: `On the ${label("Partition", partition).toLowerCase()}`,
                      value: found === undefined ? "—" : formatMetric(found),
                    };
                  }),
                ]}
              />
              <Prose text={metric.what} />
              {card !== undefined && <FormulaBlock card={card} />}
              <section className="flex flex-col gap-3">
                <h3 className="text-base">The calculation in this run</h3>
                <RunAccount
                  key={metric.name}
                  metric={metric}
                  classLabels={classLabels}
                  charts={charts}
                />
              </section>
              <section className="flex flex-col gap-2">
                <h3 className="text-base">How to read it</h3>
                <Prose text={metric.how_to_read} />
              </section>
              {card !== undefined && (
                <Link
                  to="/metrics"
                  hash={metricAnchor(metric.name)}
                  className="text-sm text-soft underline-offset-4 hover:text-strong hover:underline"
                >
                  See it in the metrics glossary
                </Link>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

interface RunAccountProps {
  metric: MetricResult;
  classLabels: readonly string[];
  charts: EvaluationCharts | undefined;
}

/** A conta deste run: os termos à escala, ou o gráfico de onde o valor sai. */
function RunAccount({ metric, classLabels, charts }: RunAccountProps) {
  if (metric.terms.length > 0) {
    return <Terms metric={metric} classLabels={classLabels} />;
  }
  const curves = charts?.roc ?? [];
  if ((metric.name === "roc_auc" || metric.name === "anomaly_roc_auc") && curves.length > 0) {
    return (
      <LineChart
        series={curves.map((curve, index) => ({
          key: curve.label,
          label: `${curve.label} (area ${formatNumber(curve.area, 3)})`,
          color: CATEGORICAL[index] ?? CATEGORICAL[0],
          points: curve.x.map((x, position) => ({ x, y: curve.y[position] ?? null })),
        }))}
        fillFirst
        diagonal
        height={240}
        xDomain={[0, 1]}
        yDomain={[0, 1]}
        xLabel="False positive rate"
        yLabel="True positive rate"
        formatX={(value) => formatPercent(value, 0)}
        formatY={(value) => formatPercent(value, 0)}
        description="The ROC curve: the area under it is the metric's value."
        caption={<TopicText topic="roc_curve" />}
      />
    );
  }
  if (metric.name === "silhouette" && charts?.silhouette) {
    return (
      <SilhouetteChart
        silhouette={charts.silhouette}
        description="Each point's silhouette, by cluster: the mean is the metric's value."
      />
    );
  }
  if (metric.name === "n_clusters" && charts && charts.cluster_sizes.length > 0) {
    return (
      <BarChart
        items={charts.cluster_sizes.map(([group, size], index) => ({
          key: group,
          label: group,
          value: size,
          color: CATEGORICAL[index] ?? CATEGORICAL[0],
        }))}
        format={formatStatistic}
        valueLabel="points"
        description="The clusters found and how many points each one has."
      />
    );
  }
  return (
    <p className="text-sm text-soft">
      This metric has no terms to show: the value comes from the formula above, with all the points.
    </p>
  );
}

/** Os termos, um a um, com uma barra à escala do maior; no fim, o resultado. */
function Terms({ metric, classLabels }: { metric: MetricResult; classLabels: readonly string[] }) {
  const reduced = useReducedMotion() ?? false;
  const [round, setRound] = useState(0);
  const largest = Math.max(1e-12, ...metric.terms.map((term) => Math.abs(term.value)));
  const color = classColors(classLabels);
  const delay = (index: number) => (reduced ? 0 : index * STAGGER);

  return (
    <div className="flex flex-col gap-4">
      <ol key={round} className="flex flex-col gap-3">
        {metric.terms.map((term, index) => (
          <motion.li
            key={`${term.symbol}-${String(term.group)}-${String(index)}`}
            initial={reduced ? false : { opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: delay(index), duration: reduced ? 0 : 0.3 }}
            className="flex flex-col gap-1.5"
          >
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="flex items-baseline gap-2">
                <Formula tex={term.symbol} display={false} />
                <span className="text-muted-foreground">{termLabel(term, classLabels)}</span>
              </span>
              <span className="tabular-nums">{formatStatistic(term.value)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted" aria-hidden="true">
              <motion.div
                className="h-full rounded-full"
                style={{
                  background:
                    term.group === null
                      ? "var(--foreground)"
                      : color(classLabels[term.group] ?? ""),
                  opacity: term.group === null ? 0.7 : 1,
                }}
                initial={reduced ? false : { width: 0 }}
                animate={{ width: `${String((Math.abs(term.value) / largest) * 100)}%` }}
                transition={{
                  delay: delay(index) + (reduced ? 0 : 0.15),
                  duration: reduced ? 0 : 0.6,
                }}
              />
            </div>
          </motion.li>
        ))}
        <motion.li
          initial={reduced ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: delay(metric.terms.length) + (reduced ? 0 : 0.4) }}
          className="flex items-baseline justify-between border-t pt-3"
        >
          <span className="text-sm text-muted-foreground">{metric.display_name}</span>
          <span className="text-2xl tabular-nums">{formatMetric(metric)}</span>
        </motion.li>
      </ol>
      {!reduced && (
        <Button
          variant="ghost"
          size="sm"
          className="self-start"
          onClick={() => {
            setRound(round + 1);
          }}
        >
          <RotateCcw aria-hidden="true" />
          See the calculation again
        </Button>
      )}
    </div>
  );
}

function termLabel(term: MetricTerm, classLabels: readonly string[]): string {
  return term.group === null ? term.label : `${term.label} ${classLabels[term.group] ?? ""}`;
}
