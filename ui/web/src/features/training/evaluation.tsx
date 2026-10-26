/**
 * A avaliação de um run terminado: as métricas no treino e no teste (cada uma abre a sua
 * gaveta, com a fórmula e a conta deste run), os gráficos da tarefa e, a pedido, a validação
 * cruzada e a curva de aprendizagem (voltam a treinar o modelo: podem demorar).
 */
import { useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { LoaderCircle, Sigma } from "lucide-react";
import { useState, type ReactNode } from "react";

import type {
  ConfusionMatrix,
  Distribution,
  EvaluationCharts,
  ExperimentResult,
  Partition,
  PointCloud,
} from "@/api/generated";
import {
  analysisChartsOptions,
  analysisCrossValidationOptions,
  analysisLearningCurveOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useTaskInfo } from "@/api/vocabulary";
import { BarChart, Histogram } from "@/components/charts/bar-chart";
import { DendrogramChart } from "@/components/charts/dendrogram-chart";
import { Heatmap } from "@/components/charts/heatmap";
import { LineChart } from "@/components/charts/line-chart";
import { Scatter } from "@/components/charts/scatter";
import { SilhouetteChart } from "@/components/charts/silhouette-chart";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Facts } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { classColors, classOrder } from "@/lib/classes";
import { CATEGORICAL, categoryColor, sequential } from "@/lib/colors";
import {
  formatCount,
  formatMetric,
  formatNumber,
  formatPercent,
  formatStatistic,
} from "@/lib/format";

import { MetricDrawer } from "./metric-drawer";

const PARTITIONS: readonly Partition[] = ["train", "validation", "test"];

export function Evaluation({ result }: { result: ExperimentResult }) {
  const [open, setOpen] = useState<string | null>(null);
  const charts = useQuery({
    ...analysisChartsOptions({ path: { run_id: result.run_id } }),
    staleTime: "static",
  });
  const main = result.reports[result.main_partition];
  const metric = main?.metrics.find((one) => one.name === open) ?? null;
  // `notices` tem todos os avisos; os do treino já se mostram em cima, junto do treino.
  const shown = new Set(result.training_notices.map((notice) => notice.message));
  const diagnostics = result.notices.filter((notice) => !shown.has(notice.message));
  return (
    <div className="flex flex-col gap-12">
      <MetricsTable result={result} onOpen={setOpen} />
      <NoticeList notices={[...diagnostics, ...(main?.notices ?? [])]} />
      {charts.error !== null ? (
        <p className="font-serif text-reading text-muted-foreground">
          Os gráficos precisam do modelo, e este run já não está na memória da API (só ficam os
          últimos). Treina-o outra vez para os ver.
        </p>
      ) : charts.data === undefined ? (
        <Skeleton className="h-64 w-full" />
      ) : (
        <TaskCharts result={result} charts={charts.data} />
      )}
      <Validation result={result} />
      <MetricDrawer
        metric={metric}
        reports={result.reports}
        mainPartition={result.main_partition}
        classLabels={result.class_labels}
        charts={charts.data}
        onClose={() => {
          setOpen(null);
        }}
      />
    </div>
  );
}

/** As métricas em cada parte dos dados: cada nome abre a explicação da métrica. */
function MetricsTable({
  result,
  onOpen,
}: {
  result: ExperimentResult;
  onOpen: (name: string) => void;
}) {
  const label = useLabel();
  const main = result.reports[result.main_partition];
  const partitions = PARTITIONS.filter((partition) => result.reports[partition] !== undefined);
  if (main === undefined) {
    return null;
  }
  return (
    <section aria-labelledby="metricas-titulo" className="flex flex-col gap-4">
      <h3 id="metricas-titulo" className="text-xl">
        As métricas
      </h3>
      <TopicText topic="train_vs_test" />
      <div className="max-w-3xl overflow-x-auto rounded-md border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th scope="col" className="px-3 py-2 text-left font-medium">
                Métrica
              </th>
              {partitions.map((partition) => (
                <th
                  key={partition}
                  scope="col"
                  className={cn(
                    "px-3 py-2 text-right font-medium",
                    partition !== result.main_partition && "text-muted-foreground",
                  )}
                >
                  {label("Partition", partition)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {main.metrics.map((metric) => (
              <tr key={metric.name} className="border-b border-border/50 last:border-0">
                <th scope="row" className="px-3 py-1.5 text-left font-normal">
                  <button
                    type="button"
                    onClick={() => {
                      onOpen(metric.name);
                    }}
                    className="group inline-flex items-center gap-1.5 rounded-sm text-left underline decoration-input decoration-dotted underline-offset-4 outline-none hover:decoration-foreground focus-visible:ring-3 focus-visible:ring-ring/50"
                  >
                    {metric.display_name}
                    <Sigma
                      aria-hidden="true"
                      className="size-3.5 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100"
                    />
                  </button>
                </th>
                {partitions.map((partition) => {
                  const found = result.reports[partition]?.metrics.find(
                    (one) => one.name === metric.name,
                  );
                  return (
                    <td
                      key={partition}
                      className={cn(
                        "px-3 py-1.5 text-right tabular-nums",
                        partition !== result.main_partition && "text-muted-foreground",
                      )}
                    >
                      {found === undefined ? "—" : formatMetric(found)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted-foreground">
        Carrega no nome de uma métrica para ver a fórmula e a conta deste run.
      </p>
    </section>
  );
}

function Block({
  title,
  children,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <section className={cn("flex min-w-0 flex-col gap-3", wide && "@5xl/main:col-span-2")}>
      <h3 className="text-xl">{title}</h3>
      {children}
    </section>
  );
}

/** Os gráficos da tarefa do run. */
function TaskCharts({ result, charts }: { result: ExperimentResult; charts: EvaluationCharts }) {
  const task = result.task_type;
  const classification = task === "binary_classification" || task === "multiclass_classification";
  return (
    <div className="grid gap-12 @5xl/main:grid-cols-2">
      {classification && (
        <>
          <Confusion
            result={result}
            title="Matriz de confusão"
            rowTitle="Real"
            columnTitle="Previsto"
            topic="confusion_matrix"
          />
          <Block title="Curvas ROC e de precisão-recall" wide>
            {charts.roc.length === 0 ? (
              <p className="font-serif text-reading text-muted-foreground">
                Este modelo não dá probabilidades (ou falta uma classe no teste): sem elas, não há
                curvas.
              </p>
            ) : (
              <Curves charts={charts} />
            )}
          </Block>
          <PointsMap charts={charts} title="Onde o modelo erra" />
        </>
      )}
      {task === "regression" && <Residuals charts={charts} />}
      {task === "regression" && <PointsMap charts={charts} title="Onde o modelo erra" />}
      {task === "clustering" && (
        <>
          {charts.cluster_sizes.length > 0 && (
            <Block title="O tamanho dos grupos">
              <BarChart
                items={charts.cluster_sizes.map(([group, size], index) => ({
                  key: group,
                  label: group,
                  value: size,
                  color: categoryColor(index),
                }))}
                format={(value) => formatCount(value)}
                valueLabel="pontos de teste"
                description="Quantos pontos de teste ficaram em cada grupo."
                caption={<TopicText topic="cluster_sizes" />}
              />
            </Block>
          )}
          <PointsMap charts={charts} title="Os grupos no plano" />
          {charts.silhouette !== null && (
            <Block title="Silhueta" wide>
              <SilhouetteChart
                silhouette={charts.silhouette}
                description="A silhueta de cada ponto de teste, por grupo."
                caption={<TopicText topic="silhouette" />}
              />
            </Block>
          )}
          {charts.dendrogram !== null && (
            <Block title="Dendrograma" wide>
              <DendrogramChart
                dendrogram={charts.dendrogram}
                description="As junções dos grupos, à altura da distância a que se deram."
                caption={<TopicText topic="dendrogram" />}
              />
            </Block>
          )}
          <Confusion
            result={result}
            title="Grupos e referência"
            rowTitle="Referência"
            columnTitle="Grupo"
            topic="reference_matrix"
          />
        </>
      )}
      {task === "dimensionality_reduction" && charts.explained_variance.length > 0 && (
        <Variance ratios={charts.explained_variance} />
      )}
      {task === "anomaly_detection" && (
        <>
          {charts.scores !== null && (
            <Block title="As pontuações de anomalia" wide>
              <Histogram
                edges={charts.scores.edges}
                series={distributionSeries(charts.scores)}
                formatEdge={formatStatistic}
                xLabel="Pontuação de anomalia"
                marker={
                  charts.threshold === null
                    ? undefined
                    : { value: charts.threshold, label: "limiar" }
                }
                description="A distribuição das pontuações de anomalia dos pontos de teste."
                caption={<TopicText topic="anomaly_scores" />}
              />
            </Block>
          )}
          {charts.roc.length > 0 && (
            <Block title="A ROC da pontuação">
              <RocChart charts={charts} caption={<TopicText topic="anomaly_roc" />} />
            </Block>
          )}
          <PointsMap charts={charts} title="As anomalias no plano" />
          <Confusion
            result={result}
            title="Anomalias e referência"
            rowTitle="Real"
            columnTitle="Detetor"
            topic="anomaly_matrix"
          />
        </>
      )}
    </div>
  );
}

interface ConfusionProps {
  result: ExperimentResult;
  title: string;
  rowTitle: string;
  columnTitle: string;
  topic: string;
}

/** Uma matriz de confusão, numa das partes dos dados que a têm. */
function Confusion({ result, title, rowTitle, columnTitle, topic }: ConfusionProps) {
  const label = useLabel();
  const parts = PARTITIONS.filter((partition) => {
    const report = result.reports[partition];
    return report !== undefined && report.confusion_matrix !== null;
  });
  const [chosen, setChosen] = useState<Partition>(
    parts.includes(result.main_partition) ? result.main_partition : (parts[0] ?? "test"),
  );
  const matrix: ConfusionMatrix | null | undefined = result.reports[chosen]?.confusion_matrix;
  if (matrix === null || matrix === undefined) {
    return null;
  }
  const largest = Math.max(1, ...matrix.counts.flat());
  // Sem `column_labels`, as colunas são as mesmas classes das linhas (e a diagonal os acertos).
  const sameClasses = matrix.column_labels.length === 0;
  const columns = sameClasses ? matrix.labels : matrix.column_labels;
  return (
    <Block title={title}>
      {parts.length > 1 && (
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          value={chosen}
          onValueChange={(value) => {
            if (value !== "") {
              setChosen(value as Partition);
            }
          }}
          aria-label="Parte dos dados"
          className="justify-start"
        >
          {parts.map((partition) => (
            <ToggleGroupItem key={partition} value={partition} className="px-3">
              {label("Partition", partition)}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      )}
      <div style={{ maxWidth: 140 + columns.length * 84 }}>
        <Heatmap
          rows={matrix.labels}
          columns={columns}
          values={matrix.counts}
          color={(value) => sequential(value / largest)}
          format={formatCount}
          rowTitle={rowTitle}
          columnTitle={columnTitle}
          highlightDiagonal={sameClasses}
          description={`${title}: quantos pontos de cada classe real ficaram em cada classe prevista.`}
          caption={<TopicText topic={topic} />}
        />
      </div>
    </Block>
  );
}

function RocChart({ charts, caption }: { charts: EvaluationCharts; caption?: ReactNode }) {
  return (
    <LineChart
      series={charts.roc.map((curve, index) => ({
        key: curve.label,
        label: `${curve.label} (AUC ${formatNumber(curve.area, 3)})`,
        color: CATEGORICAL[index] ?? CATEGORICAL[0],
        points: curve.x.map((x, position) => ({ x, y: curve.y[position] ?? null })),
      }))}
      diagonal
      height={260}
      xDomain={[0, 1]}
      yDomain={[0, 1]}
      xLabel="Taxa de falsos positivos"
      yLabel="Taxa de verdadeiros positivos"
      formatX={(value) => formatPercent(value, 0)}
      formatY={(value) => formatPercent(value, 0)}
      description="A curva ROC de cada classe."
      caption={caption}
    />
  );
}

function Curves({ charts }: { charts: EvaluationCharts }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid gap-8 @3xl/main:grid-cols-2">
        <RocChart charts={charts} />
        <LineChart
          series={charts.pr.map((curve, index) => ({
            key: curve.label,
            label: `${curve.label} (área ${formatNumber(curve.area, 3)})`,
            color: CATEGORICAL[index] ?? CATEGORICAL[0],
            points: curve.x.map((x, position) => ({ x, y: curve.y[position] ?? null })),
          }))}
          height={260}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xLabel="Recall"
          yLabel="Precisão"
          formatX={(value) => formatPercent(value, 0)}
          formatY={(value) => formatPercent(value, 0)}
          description="A curva de precisão-recall de cada classe."
        />
      </div>
      <div className="max-w-[68ch] font-serif text-sm leading-relaxed text-muted-foreground">
        <TopicText topic="roc_pr_chart" />
      </div>
    </div>
  );
}

/** Os pontos de teste no plano da projeção, pintados com o que o backend lhes chama. */
function PointsMap({ charts, title }: { charts: EvaluationCharts; title: string }) {
  const points: PointCloud | null = charts.test_points;
  if (points === null) {
    return null;
  }
  const classes = classOrder([...new Set(points.labels)]);
  const color = classColors(classes);
  const low = Math.min(...points.values);
  const high = Math.max(...points.values);
  return (
    <Block title={title}>
      <Scatter
        height={300}
        xLabel={charts.axis_labels[0]}
        yLabel={charts.axis_labels[1]}
        points={points.x.map((x, index) => {
          const name = points.labels[index];
          const value = points.values[index];
          return {
            x,
            y: points.y[index] ?? 0,
            color:
              name !== undefined
                ? color(name)
                : value !== undefined
                  ? sequential(high > low ? (value - low) / (high - low) : 0.5)
                  : CATEGORICAL[0],
            label: name ?? (value === undefined ? undefined : formatStatistic(value)),
          };
        })}
        legend={classes.map((name) => ({ label: name, color: color(name), kind: "dot" as const }))}
        description={`${title}: os pontos de teste no plano da projeção 2D.`}
        caption={<TopicText topic="errors_map" />}
      />
    </Block>
  );
}

function Residuals({ charts }: { charts: EvaluationCharts }) {
  const residuals = charts.residuals;
  if (residuals === null) {
    return null;
  }
  return (
    <Block title="Os resíduos" wide>
      <div className="grid gap-8 @3xl/main:grid-cols-2">
        <Scatter
          height={260}
          zeroLine
          xLabel="Previsto"
          yLabel="Erro (real − previsto)"
          points={residuals.predicted.map((x, index) => ({
            x,
            y: residuals.errors[index] ?? 0,
            color: CATEGORICAL[0],
          }))}
          description="O erro de cada ponto de teste contra o valor previsto."
        />
        <Scatter
          height={260}
          identity
          xLabel="Real"
          yLabel="Previsto"
          points={residuals.actual.map((x, index) => ({
            x,
            y: residuals.predicted[index] ?? 0,
            color: CATEGORICAL[0],
          }))}
          description="O valor previsto contra o real, em cada ponto de teste: na diagonal, acerta."
        />
      </div>
      <Histogram
        edges={residuals.histogram.edges}
        series={distributionSeries(residuals.histogram)}
        formatEdge={formatStatistic}
        xLabel="Erro (real − previsto)"
        description="Quantos pontos de teste têm cada tamanho de erro."
        caption={<TopicText topic="residuals" />}
      />
    </Block>
  );
}

function Variance({ ratios }: { ratios: readonly number[] }) {
  const total = ratios.reduce((sum, ratio) => sum + ratio, 0);
  const items = ratios.map((ratio, index) => ({
    key: String(index),
    label: `Componente ${String(index + 1)}`,
    value: ratio,
  }));
  return (
    <Block title="Variância explicada" wide>
      <BarChart
        items={items}
        format={(value) => formatPercent(value)}
        max={1}
        valueLabel="da variância"
        description="A parte da variância dos dados que cada componente guarda."
        caption={
          <>
            <p className="mb-1">
              As componentes guardadas explicam {formatPercent(total)} da variância.
            </p>
            <TopicText topic="explained_variance" />
          </>
        }
      />
    </Block>
  );
}

/** A validação cruzada e a curva de aprendizagem: voltam a treinar, por isso só a pedido. */
function Validation({ result }: { result: ExperimentResult }) {
  const task = useTaskInfo(result.task_type);
  const [crossValidation, setCrossValidation] = useState(false);
  const [learningCurve, setLearningCurve] = useState(false);
  const path = { run_id: result.run_id };
  const folds = useQuery({
    ...analysisCrossValidationOptions({ path }),
    enabled: crossValidation,
    staleTime: "static",
  });
  const curve = useQuery({
    ...analysisLearningCurveOptions({ path }),
    enabled: learningCurve,
    staleTime: "static",
  });
  if (task?.requires_target !== true) {
    return null;
  }
  const percent = (asPercent: boolean) => (value: number) =>
    asPercent ? formatPercent(value) : formatNumber(value, 3);

  return (
    <div className="grid gap-12 @5xl/main:grid-cols-2">
      <Block title="Validação cruzada">
        <TopicText topic="cross_validation" />
        {!crossValidation ? (
          <Button
            variant="outline"
            className="self-start"
            onClick={() => {
              setCrossValidation(true);
            }}
          >
            Correr a validação cruzada
          </Button>
        ) : folds.data === undefined ? (
          folds.error === null ? (
            <Pending text="A treinar o modelo em cada fold" />
          ) : (
            <ErrorNote error={folds.error} />
          )
        ) : (
          <>
            <Facts
              size="sm"
              facts={[
                {
                  label: `${folds.data.metric_label}, média`,
                  value:
                    folds.data.mean === null
                      ? "—"
                      : percent(folds.data.as_percent)(folds.data.mean),
                },
                {
                  label: "Desvio padrão",
                  value:
                    folds.data.std === null ? "—" : percent(folds.data.as_percent)(folds.data.std),
                },
              ]}
            />
            <LineChart
              series={[
                {
                  key: "treino",
                  label: "No treino de cada fold",
                  color: CATEGORICAL[0],
                  points: folds.data.folds.map((fold) => ({ x: fold.fold, y: fold.train })),
                },
                {
                  key: "fora",
                  label: "No fold de fora",
                  color: CATEGORICAL[1],
                  points: folds.data.folds.map((fold) => ({ x: fold.fold, y: fold.test })),
                },
              ]}
              height={220}
              xLabel="Fold"
              xTicks={folds.data.folds.map((fold) => fold.fold)}
              formatX={(value) => formatCount(value)}
              formatY={percent(folds.data.as_percent)}
              reference={
                folds.data.mean === null ? undefined : { y: folds.data.mean, label: "média fora" }
              }
              description={`${folds.data.metric_label} em cada fold, no treino e no fold de fora.`}
            />
            <NoticeList notices={folds.data.notices} />
          </>
        )}
      </Block>
      <Block title="Curva de aprendizagem">
        <TopicText topic="learning_curve" />
        {!learningCurve ? (
          <Button
            variant="outline"
            className="self-start"
            onClick={() => {
              setLearningCurve(true);
            }}
          >
            Desenhar a curva de aprendizagem
          </Button>
        ) : curve.data === undefined ? (
          curve.error === null ? (
            <Pending text="A treinar com cada vez mais linhas" />
          ) : (
            <ErrorNote error={curve.error} />
          )
        ) : (
          <>
            <LineChart
              series={[
                {
                  key: "treino",
                  label: "No treino",
                  color: CATEGORICAL[0],
                  points: curve.data.points.map((point) => ({
                    x: point.size,
                    y: point.train_mean,
                  })),
                  band: curve.data.points.map((point) => ({
                    x: point.size,
                    lower: point.train_mean - point.train_std,
                    upper: point.train_mean + point.train_std,
                  })),
                },
                {
                  key: "avaliacao",
                  label: "Na avaliação",
                  color: CATEGORICAL[1],
                  points: curve.data.points.map((point) => ({
                    x: point.size,
                    y: point.test_mean,
                  })),
                  band: curve.data.points.map((point) => ({
                    x: point.size,
                    lower: point.test_mean - point.test_std,
                    upper: point.test_mean + point.test_std,
                  })),
                },
              ]}
              height={240}
              xLabel="Linhas de treino"
              formatX={(value) => formatCount(value)}
              formatY={percent(curve.data.as_percent)}
              description={`${curve.data.metric_label} com cada vez mais linhas de treino.`}
            />
            <NoticeList notices={curve.data.notices} />
          </>
        )}
      </Block>
    </div>
  );
}

/** As séries de um histograma do backend, cada uma com a cor da sua posição. */
function distributionSeries(distribution: Distribution) {
  return distribution.series.map((name, index) => ({
    label: name,
    color: categoryColor(index),
    counts: distribution.counts[index] ?? [],
  }));
}

function Pending({ text }: { text: string }) {
  return (
    <p role="status" className="flex items-center gap-2 text-sm text-muted-foreground">
      <LoaderCircle aria-hidden="true" className="size-4 animate-spin" />
      {text}
    </p>
  );
}
