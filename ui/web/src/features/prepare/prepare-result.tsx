/**
 * O que sai da preparação: as partes da divisão, a linhagem, as classes em cada parte, os dados
 * antes e depois, e a projeção 2D, cada parte no seu cartão (as explicações ficam nos "?").
 */
import { cn } from "cn";
import { LoaderCircle } from "lucide-react";

import type { LineageStep, Partition, PreparationSummary } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Heatmap } from "@/components/charts/heatmap";
import { Scatter, type ScatterPoint } from "@/components/charts/scatter";
import type { PointShape } from "@/components/charts/shapes";
import { StackedBars } from "@/components/charts/stacked-bars";
import { TablePreviewView } from "@/components/data/table-preview";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Facts, Panel } from "@/components/shell/blocks";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { classColors, classOrder } from "@/lib/classes";
import { CATEGORICAL, diverging, sequential } from "@/lib/colors";
import {
  formatCount,
  formatList,
  formatNumber,
  formatPercent,
  formatQuantity,
  formatStatistic,
} from "@/lib/format";

/** As partes da divisão em cinzentos (as cores ficam para as classes). */
const PARTITION_TONES: Readonly<Record<Partition, string>> = {
  train: "#d4d4d4",
  validation: "#8f8f8f",
  test: "#5c5c5c",
};

/** Na projeção, a parte de cada ponto é a forma (a cor é a classe). */
const PARTITION_SHAPES: Readonly<Record<Partition, PointShape>> = {
  train: "circle",
  validation: "diamond",
  test: "cross",
};

/** Quantas colunas de uma operação se escrevem antes de "e mais N". */
const LISTED_COLUMNS = 6;

interface PrepareResultProps {
  summary: PreparationSummary | undefined;
  /** Se há uma preparação nova a caminho (a que está à vista é a anterior). */
  pending: boolean;
  error: unknown;
}

export function PrepareResult({ summary, pending, error }: PrepareResultProps) {
  const label = useLabel();
  return (
    <section aria-labelledby="preparar-resultado" aria-busy={pending}>
      <div className="flex min-h-8 flex-wrap items-center justify-between gap-3">
        <h2 id="preparar-resultado" className="text-xl">
          O resultado
        </h2>
        {pending && (
          <span className="flex items-center gap-2 font-mono text-xs text-dim" role="status">
            <LoaderCircle aria-hidden="true" className="size-3.5 animate-spin" />A preparar
          </span>
        )}
      </div>
      <ErrorNote error={error} className="mt-4" />
      {summary !== undefined && error === null && (
        <div className={cn("mt-4 flex flex-col gap-4", pending && "stale")}>
          <div className="flex flex-col gap-5 border-t pt-5">
            <StackedBars
              rows={[
                {
                  key: "partes",
                  label: "Linhas",
                  segments: summary.partitions.map((part) => ({
                    key: part.partition,
                    label: label("Partition", part.partition),
                    value: part.n_rows,
                    color: PARTITION_TONES[part.partition],
                    detail: formatQuantity(part.n_rows, "linha", "linhas"),
                  })),
                },
              ]}
              format={(share) => formatPercent(share, 0)}
              legend={summary.partitions.map((part) => ({
                label: `${label("Partition", part.partition)}: ${formatCount(part.n_rows)}`,
                color: PARTITION_TONES[part.partition],
              }))}
              description="Como as linhas se dividem entre o treino, a validação e o teste."
              labelWidth={56}
            />
            <Facts
              size="sm"
              facts={[
                { label: "Tarefa", value: label("TaskType", summary.task_type) },
                { label: "Features", value: formatCount(summary.feature_names.length) },
                ...(summary.class_labels.length > 0
                  ? [{ label: "Classes", value: formatCount(summary.class_labels.length) }]
                  : []),
              ]}
            />
          </div>
          {summary.notices.length > 0 && (
            <div className="border-t pt-5">
              <NoticeList notices={summary.notices} />
            </div>
          )}
          <Tabs defaultValue="lineage" className="mt-2">
            <TabsList
              aria-label="O resultado por partes"
              className="flex-wrap group-data-horizontal/tabs:h-auto"
            >
              <TabsTrigger value="lineage">Linhagem</TabsTrigger>
              {summary.class_labels.length > 0 && (
                <TabsTrigger value="classes">Classes em cada parte</TabsTrigger>
              )}
              <TabsTrigger value="tables">Antes e depois</TabsTrigger>
              <TabsTrigger value="projection">Projeção 2D</TabsTrigger>
            </TabsList>
            <TabsContent value="lineage" className="mt-3">
              <Panel title="As operações, pela ordem em que correram" topic="lineage">
                <Lineage steps={summary.lineage} />
              </Panel>
            </TabsContent>
            {summary.class_labels.length > 0 && (
              <TabsContent value="classes" className="mt-3">
                <div className="border-t pt-5">
                  <ClassesPerPartition summary={summary} />
                </div>
              </TabsContent>
            )}
            <TabsContent value="tables" className="mt-3 flex flex-col gap-4">
              <Panel title="Depois da limpeza">
                <TablePreviewView
                  preview={summary.cleaned_preview}
                  label="Os dados depois da limpeza"
                />
              </Panel>
              <Panel
                title="O treino depois das transformações"
                explain="É isto que o modelo recebe: as colunas já preenchidas, codificadas e na mesma escala."
              >
                <TablePreviewView
                  preview={summary.transformed_preview}
                  label="O treino depois das transformações"
                />
              </Panel>
            </TabsContent>
            <TabsContent value="projection" className="mt-3">
              <Projection summary={summary} />
            </TabsContent>
          </Tabs>
        </div>
      )}
    </section>
  );
}

/** A linhagem: cada operação, pela ordem em que correu, com o que mudou. */
function Lineage({ steps }: { steps: readonly LineageStep[] }) {
  const label = useLabel();
  return (
    <div className="flex flex-col gap-4">
      <ol className="flex flex-col">
        {steps.map((step, index) => {
          const newStage = index === 0 || steps[index - 1]?.stage !== step.stage;
          const rows = rowsText(step);
          return (
            <li
              key={`${step.operation}-${String(index)}`}
              className={cn(
                "grid grid-cols-[7rem_minmax(0,1fr)] gap-x-4 py-3",
                newStage && index > 0 && "border-t",
              )}
            >
              <span className="pt-0.5 mono-label">
                {newStage ? label("Stage", step.stage) : ""}
              </span>
              <div>
                <p className="text-sm text-strong">{step.description}</p>
                <dl className="mt-1.5 flex flex-wrap gap-x-6 gap-y-1 font-mono text-xs text-dim">
                  {rows !== null && <Change title="Linhas" value={rows} />}
                  {step.columns_added.length > 0 && (
                    <Change title="Novas" value={columnsText(step.columns_added)} />
                  )}
                  {step.columns_removed.length > 0 && (
                    <Change title="Removidas" value={columnsText(step.columns_removed)} />
                  )}
                  {step.columns_changed.length > 0 && (
                    <Change title="Mudaram" value={columnsText(step.columns_changed)} />
                  )}
                </dl>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function Change({ title, value }: { title: string; value: string }) {
  return (
    <div className="flex gap-1.5">
      <dt>{title}</dt>
      <dd className="text-soft tabular-nums">{value}</dd>
    </div>
  );
}

function rowsText(step: LineageStep): string | null {
  if (step.rows_before !== null && step.rows_after !== null) {
    return step.rows_before === step.rows_after
      ? formatCount(step.rows_after)
      : `${formatCount(step.rows_before)} → ${formatCount(step.rows_after)}`;
  }
  return step.rows_after === null ? null : formatCount(step.rows_after);
}

function columnsText(columns: readonly string[]): string {
  if (columns.length <= LISTED_COLUMNS) {
    return formatList(columns);
  }
  const rest = columns.length - LISTED_COLUMNS;
  return `${columns.slice(0, LISTED_COLUMNS).join(", ")} e mais ${formatCount(rest)}`;
}

/** A proporção de cada classe em cada parte: na divisão estratificada, as partes alinham. */
function ClassesPerPartition({ summary }: { summary: PreparationSummary }) {
  const label = useLabel();
  const color = classColors(summary.class_labels);
  return (
    <StackedBars
      rows={summary.partitions.map((part) => {
        const counts = new Map(part.classes.map((value) => [value.value, value.count]));
        return {
          key: part.partition,
          label: label("Partition", part.partition),
          segments: summary.class_labels.map((name) => ({
            key: name,
            label: name,
            value: counts.get(name) ?? 0,
            color: color(name),
            detail: formatQuantity(counts.get(name) ?? 0, "linha", "linhas"),
          })),
        };
      })}
      format={(share) => formatPercent(share)}
      legend={summary.class_labels.map((name) => ({ label: name, color: color(name) }))}
      description="A proporção de cada classe no treino, na validação e no teste."
      caption={<TopicText topic="split_classes" />}
    />
  );
}

/** A projeção 2D: a cor é a classe (ou o valor do alvo) e a forma é a parte da divisão. */
function Projection({ summary }: { summary: PreparationSummary }) {
  const label = useLabel();
  const points = summary.projection;
  if (points === null) {
    return (
      <p className="text-sm text-soft">
        Com menos de duas features, não há projeção 2D para mostrar.
      </p>
    );
  }
  const classes =
    summary.class_labels.length > 0
      ? summary.class_labels
      : classOrder([...new Set(points.labels)]);
  const color = classColors(classes);
  const low = Math.min(...points.values);
  const high = Math.max(...points.values);
  const ratio = points.explained_variance_ratio;
  const axis = (index: 0 | 1) =>
    ratio === null
      ? points.axis_labels[index]
      : `${points.axis_labels[index]} (${formatPercent(ratio[index])} da variância)`;
  const partitions = [...new Set(points.partitions)];

  const scatter: ScatterPoint[] = points.x.map((x, index) => {
    const partition = points.partitions[index] ?? "train";
    const name = points.labels[index];
    const value = points.values[index];
    const where = label("Partition", partition).toLowerCase();
    return {
      x,
      y: points.y[index] ?? 0,
      color:
        name !== undefined
          ? color(name)
          : value !== undefined
            ? sequential(high > low ? (value - low) / (high - low) : 0.5)
            : CATEGORICAL[0],
      shape: PARTITION_SHAPES[partition],
      label:
        name !== undefined
          ? `${name}, ${where}`
          : value !== undefined
            ? `${formatStatistic(value)}, ${where}`
            : where,
    };
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="border-t pt-5">
        <Scatter
          points={scatter}
          height={360}
          xLabel={axis(0)}
          yLabel={axis(1)}
          legend={[
            ...(points.labels.length > 0
              ? classes.map((name) => ({ label: name, color: color(name), kind: "dot" as const }))
              : []),
            ...partitions.map((partition) => ({
              label: label("Partition", partition),
              color: "var(--muted-foreground)",
              shape: PARTITION_SHAPES[partition],
            })),
          ]}
          description="Os dados projetados em duas dimensões, pintados pela classe, com a forma da parte da divisão."
          caption={
            <>
              {points.labels.length === 0 && points.values.length > 0 && (
                <p className="mb-1 flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className="inline-block h-2 w-20 rounded-sm"
                    style={{
                      background: `linear-gradient(to right, ${sequential(0)}, ${sequential(0.5)}, ${sequential(1)})`,
                    }}
                  />
                  O alvo, de {formatStatistic(low)} a {formatStatistic(high)}.
                </p>
              )}
              <TopicText topic="projection" />
            </>
          }
        />
      </div>
      {points.loadings !== null && <Loadings summary={summary} />}
    </div>
  );
}

/** O peso de cada feature nas duas componentes da PCA. */
function Loadings({ summary }: { summary: PreparationSummary }) {
  const points = summary.projection;
  const loadings = points?.loadings ?? null;
  if (points === null || loadings === null) {
    return null;
  }
  const strongest = Math.max(...loadings.flat().map(Math.abs), 1e-9);
  return (
    <Panel title="O peso de cada feature nas componentes">
      <div className="max-w-md">
        <Heatmap
          rows={points.feature_names}
          columns={points.axis_labels}
          values={points.feature_names.map((_, feature) =>
            loadings.map((component) => component[feature] ?? null),
          )}
          color={(value) => diverging(value / strongest)}
          format={(value) => formatNumber(value, 2)}
          description="O peso de cada feature em cada componente principal."
          caption={<TopicText topic="pca_loadings" />}
          labelWidth={130}
          cellHeight={30}
        />
      </div>
    </Panel>
  );
}
