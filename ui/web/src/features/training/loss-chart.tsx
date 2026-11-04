/**
 * A loss ao longo do treino e, à parte, as métricas que o modelo reporta em cada passo.
 *
 * Duas escalas diferentes (a loss e as percentagens) nunca partilham um gráfico com dois eixos:
 * cada uma tem o seu (uma regra do skill de visualização de dados). A loss na validação, quando
 * a há, vai junto da loss do treino, na mesma escala, a tracejado.
 */
import type { StepRecord } from "@/api/generated";
import { useVocabulary } from "@/api/vocabulary";
import { LineChart, type LineSeries } from "@/components/charts/line-chart";
import { TopicText } from "@/components/markdown/topic-text";
import { CATEGORICAL } from "@/lib/colors";
import { formatPercent, formatStatistic } from "@/lib/format";

interface LossChartsProps {
  steps: readonly StepRecord[];
  height?: number;
}

export function LossCharts({ steps, height = 240 }: LossChartsProps) {
  const vocabulary = useVocabulary();
  if (steps.length === 0) {
    return null;
  }
  const reported = new Set(steps.flatMap((record) => Object.keys(record.metrics)));
  const known = vocabulary.step_metrics.filter((metric) => reported.has(metric.name));
  const losses = known.filter((metric) => metric.is_loss);
  // O gráfico de baixo é em percentagem (de 0 a 100%): só as métricas que o são.
  const scores = known.filter((metric) => !metric.is_loss && metric.as_percent);

  const lossSeries: LineSeries[] = [
    {
      key: "loss",
      label: "Training loss",
      color: CATEGORICAL[0],
      points: steps.map((record) => ({ x: record.step, y: record.loss })),
    },
    ...losses.map((metric, index) => ({
      key: metric.name,
      label: metric.label,
      color: CATEGORICAL[index + 1] ?? CATEGORICAL[1],
      dashed: true,
      points: steps.map((record) => ({ x: record.step, y: record.metrics[metric.name] ?? null })),
    })),
  ];

  return (
    <div className="flex min-w-0 flex-col gap-8">
      <LineChart
        series={lossSeries}
        height={height}
        xLabel="Training step"
        yLabel="Loss"
        formatY={formatStatistic}
        description="The loss at each training step."
        caption={<TopicText topic="loss" />}
      />
      {scores.length > 0 && (
        <LineChart
          series={scores.map((metric, index) => ({
            key: metric.name,
            label: metric.label,
            color: CATEGORICAL[index + 2] ?? CATEGORICAL[2],
            points: steps.map((record) => ({
              x: record.step,
              y: record.metrics[metric.name] ?? null,
            })),
          }))}
          height={200}
          xLabel="Training step"
          yLabel={scores.length === 1 ? scores[0]?.label : undefined}
          yDomain={[0, 1]}
          formatY={(value) => formatPercent(value, 0)}
          description="The metrics the model measures at each training step."
        />
      )}
    </div>
  );
}
