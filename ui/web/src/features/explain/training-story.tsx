/**
 * A história do treino, no topo do ecrã Explicar: o que aconteceu, em capítulos curtos, cada
 * um com o texto e um visual (ADR 0047).
 *
 * O backend escreve tudo (o texto, os números e que visual cabe em cada capítulo, em
 * `HistoryService.story`); aqui só se desenha: a divisão das linhas numa barra, a curva da
 * loss, umas barras ou a matriz de confusão. Os capítulos são uma sequência (dos dados ao que
 * fazer a seguir), por isso vão numerados.
 *
 * Cada capítulo é uma linha entre filetes (ADR 0048): num ecrã largo, o texto à esquerda e o
 * visual à direita, alinhados pelo título; num estreito, um por baixo do outro.
 */
import { useQuery } from "@tanstack/react-query";
import { Fragment, type ReactNode } from "react";

import type {
  BarTone,
  Partition,
  SplitPart,
  StoryBars,
  StoryChapter,
  StoryCurve,
  TrainingStory,
} from "@/api/generated";
import { historyStoryOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { BarChart } from "@/components/charts/bar-chart";
import { Heatmap } from "@/components/charts/heatmap";
import { LineChart } from "@/components/charts/line-chart";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote } from "@/components/shell/blocks";
import { Skeleton } from "@/components/ui/skeleton";
import { CATEGORICAL, OTHER, sequential } from "@/lib/colors";
import { formatCount, formatPercent, formatStatistic } from "@/lib/format";

/** A cor de cada papel de barra: as partes dos dados têm as cores de sempre (ADR 0047). */
const TONES: Readonly<Record<BarTone, string>> = {
  baseline: OTHER,
  train: CATEGORICAL[0],
  validation: CATEGORICAL[1],
  test: CATEGORICAL[3],
  up: CATEGORICAL[0],
  down: CATEGORICAL[7],
  plain: CATEGORICAL[0],
};

const PARTITION_TONES: Readonly<Record<Partition, string>> = {
  train: TONES.train,
  validation: TONES.validation,
  test: TONES.test,
};

/** A história de um run: pede-a ao backend e desenha-a. */
export function TrainingStoryView({ runId }: { runId: string }) {
  const story = useQuery({
    ...historyStoryOptions({ path: { run_id: runId } }),
    staleTime: "static",
  });
  return (
    <section aria-labelledby="historia-titulo" className="flex flex-col">
      <h2 id="historia-titulo" className="text-2xl">
        What happened in the training
      </h2>
      {story.error !== null ? (
        <ErrorNote error={story.error} className="mt-6" />
      ) : story.data === undefined ? (
        <StorySkeleton />
      ) : (
        <Story story={story.data} />
      )}
    </section>
  );
}

function Story({ story }: { story: TrainingStory }) {
  return (
    <>
      <p className="mt-4 max-w-[60ch] text-xl leading-snug text-pretty text-soft">
        {story.headline}
      </p>
      <ol className="mt-10 flex flex-col">
        {story.chapters.map((chapter, index) => (
          <Chapter key={chapter.key} chapter={chapter} number={index + 1} />
        ))}
      </ol>
    </>
  );
}

interface ChapterProps {
  chapter: StoryChapter;
  number: number;
}

/**
 * Um capítulo é uma linha da página, entre filetes (ADR 0048): o número, o título e o texto
 * numa coluna à esquerda, sempre da mesma largura, e o visual à direita, com o topo à altura do
 * título. Lê-se a história de cima para baixo pela coluna do texto, e cada visual fica ao lado
 * do parágrafo que explica; um capítulo sem visual deixa a direita vazia, e não puxa o seguinte.
 */
function Chapter({ chapter, number }: ChapterProps) {
  const id = `historia-${chapter.key}`;
  const visual = visualOf(chapter);
  return (
    <li
      aria-labelledby={id}
      className="grid gap-x-12 gap-y-8 border-t border-[#202020] py-10 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]"
    >
      <div className="flex min-w-0 flex-col gap-3">
        <span aria-hidden="true" className="font-mono text-sm text-dim tabular-nums">
          {String(number).padStart(2, "0")}
        </span>
        <h3 id={id} className="text-lg font-medium text-strong">
          {chapter.title}
        </h3>
        <p className="max-w-[58ch] leading-relaxed text-pretty text-soft">
          <Emphasis text={chapter.text} />
        </p>
        {chapter.points.length > 0 && (
          <ul className="flex max-w-[58ch] flex-col gap-1.5 text-sm leading-relaxed text-dim">
            {chapter.points.map((point) => (
              <li key={point} className="flex gap-2.5">
                <span aria-hidden="true" className="mt-2.5 size-1 shrink-0 rounded-full bg-dim" />
                <span>
                  <Emphasis text={point} />
                </span>
              </li>
            ))}
          </ul>
        )}
        {chapter.notices.length > 0 && <NoticeList notices={chapter.notices} className="mt-1" />}
      </div>
      {/* O topo do visual à altura do título (o número tem 24 px, mais 12 de intervalo). */}
      {visual !== null && <div className="min-w-0 @5xl/main:pt-9">{visual}</div>}
    </li>
  );
}

/** O visual do capítulo (no máximo um: o backend escolhe o que ajuda a ler o texto). */
function visualOf(chapter: StoryChapter): ReactNode {
  if (chapter.split.length > 0) {
    return <SplitBar parts={chapter.split} />;
  }
  if (chapter.curve !== null) {
    return <Curve curve={chapter.curve} />;
  }
  if (chapter.bars !== null) {
    return <Bars bars={chapter.bars} wholeScale={chapter.key === "result"} />;
  }
  if (chapter.matrix !== null) {
    return <Matrix chapter={chapter} />;
  }
  return null;
}

/** As linhas do run numa barra: cada parte com o seu tamanho, e para que serve. */
function SplitBar({ parts }: { parts: readonly SplitPart[] }) {
  const label = useLabel();
  const total = parts.reduce((sum, part) => sum + part.rows, 0);
  return (
    <figure className="flex min-w-0 flex-col gap-4 self-start pt-1">
      <div
        role="img"
        aria-label={`The ${formatCount(total)} rows: ${parts
          .map(
            (part) =>
              `${formatCount(part.rows)} for ${label("Partition", part.partition).toLowerCase()}`,
          )
          .join(", ")}.`}
        className="flex h-3 w-full gap-0.5 overflow-hidden rounded-full"
      >
        {parts.map((part) => (
          <span
            key={part.partition}
            className="h-full first:rounded-l-full last:rounded-r-full"
            style={{
              width: `${String(part.share * 100)}%`,
              backgroundColor: PARTITION_TONES[part.partition],
            }}
          />
        ))}
      </div>
      <figcaption className="grid grid-cols-[repeat(auto-fit,minmax(9rem,1fr))] gap-x-6 gap-y-3">
        {parts.map((part) => (
          <div key={part.partition} className="flex gap-2.5">
            <span
              aria-hidden="true"
              className="mt-1.5 size-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: PARTITION_TONES[part.partition] }}
            />
            <div className="flex flex-col">
              <span className="text-sm text-strong tabular-nums">
                {formatCount(part.rows)} rows ({formatPercent(part.share, 0)})
              </span>
              <span className="text-sm text-dim">
                {label("Partition", part.partition)}, {part.role}
              </span>
            </div>
          </div>
        ))}
      </figcaption>
    </figure>
  );
}

/** A loss ao longo do treino (e na validação, a tracejado, com o melhor passo marcado). */
function Curve({ curve }: { curve: StoryCurve }) {
  const best = curve.best_step;
  return (
    <div className="min-w-0">
      <LineChart
        series={[
          {
            key: "loss",
            label: "Training loss",
            color: TONES.train,
            points: curve.steps.map((step, index) => ({ x: step, y: curve.loss[index] ?? null })),
          },
          ...(curve.validation.length > 0
            ? [
                {
                  key: "validation",
                  label: "Validation loss",
                  color: TONES.validation,
                  dashed: true,
                  points: curve.steps.map((step, index) => ({
                    x: step,
                    y: curve.validation[index] ?? null,
                  })),
                },
              ]
            : []),
        ]}
        height={230}
        xLabel={curve.step_label}
        yLabel="Loss"
        formatY={formatStatistic}
        markerX={best === null ? undefined : { x: best, label: `Best on validation` }}
        description={`The loss at each ${curve.step_label.toLowerCase()} of the training.`}
      />
    </div>
  );
}

/** Umas barras: as métricas comparadas, os pesos, a importância ou a variância. */
function Bars({ bars, wholeScale }: { bars: StoryBars; wholeScale: boolean }) {
  const signed = bars.bars.some((bar) => bar.tone === "up" || bar.tone === "down");
  const longest = Math.max(...bars.bars.map((bar) => bar.label.length));
  const write = (value: number) =>
    bars.as_percent ? formatPercent(value) : formatStatistic(value);
  return (
    <div className="min-w-0 self-start">
      <BarChart
        items={bars.bars.map((bar) => ({
          key: bar.label,
          label: bar.label,
          value: bar.value,
          color: TONES[bar.tone],
        }))}
        format={write}
        valueLabel={bars.value_label}
        // Os nomes compridos (as features do one-hot) precisam de espaço; o resto, de menos.
        labelWidth={Math.min(240, Math.max(100, longest * 8))}
        // A exatidão, de 0 a 100%: as barras comparam-se com o todo, e não só entre si.
        max={wholeScale && bars.as_percent ? 1 : undefined}
        legend={
          signed
            ? [
                { label: "Pushes it up", color: TONES.up, kind: "rect" },
                { label: "Pushes it down", color: TONES.down, kind: "rect" },
              ]
            : undefined
        }
        description={`${bars.value_label}: ${bars.bars.map((bar) => `${bar.label} ${write(bar.value)}`).join(", ")}.`}
      />
    </div>
  );
}

/** A matriz de confusão, com a diagonal (os acertos) à vista quando as classes são as mesmas. */
function Matrix({ chapter }: { chapter: StoryChapter }) {
  const matrix = chapter.matrix;
  if (matrix === null) {
    return null;
  }
  const sameClasses = matrix.column_labels.length === 0;
  const columns = sameClasses ? matrix.labels : matrix.column_labels;
  const largest = Math.max(1, ...matrix.counts.flat());
  return (
    <div className="min-w-0 self-start">
      <div style={{ maxWidth: 150 + columns.length * 84 }}>
        <Heatmap
          rows={matrix.labels}
          columns={columns}
          values={matrix.counts}
          color={(value) => sequential(value / largest)}
          format={formatCount}
          rowTitle={chapter.matrix_rows}
          columnTitle={chapter.matrix_columns}
          highlightDiagonal={sameClasses}
          description={`${chapter.title}: how many rows of each ${chapter.matrix_rows.toLowerCase()} ended up in each ${chapter.matrix_columns.toLowerCase()}.`}
        />
      </div>
      {/* O título das linhas só aparece na dica do Heatmap: aqui fica escrito, à vista (fora da
          largura da matriz, para não partir numa coluna estreita). */}
      <p className="mt-3 text-sm text-dim">
        Rows: {chapter.matrix_rows.toLowerCase()}. Columns: {chapter.matrix_columns.toLowerCase()}.
      </p>
    </div>
  );
}

/** O texto do backend com os seus negritos (`**…**`): o único formato que a história usa. */
function Emphasis({ text }: { text: string }) {
  const parts = text.split(/\*\*(.+?)\*\*/g);
  return (
    <>
      {parts.map((part, index) =>
        // Os índices ímpares são o que estava entre os asteriscos.
        index % 2 === 1 ? (
          <strong key={index} className="font-medium text-strong">
            {part}
          </strong>
        ) : (
          <Fragment key={index}>{part}</Fragment>
        ),
      )}
    </>
  );
}

function StorySkeleton() {
  return (
    <div
      role="status"
      aria-label="Loading the story of the training"
      className="mt-4 flex flex-col gap-10"
    >
      <Skeleton className="h-8 w-full max-w-2xl" />
      {[0, 1, 2].map((row) => (
        <div key={row} className="grid gap-12 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      ))}
    </div>
  );
}
