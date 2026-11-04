/**
 * O ecrã Explicar de um run: o que aconteceu no treino e o que pesa nas previsões do modelo.
 *
 * Abre com a **história do treino** (ADR 0047): em capítulos curtos, com texto e um visual,
 * com que dados aprendeu, como aprendeu, o que aprendeu, como se saiu, onde se engana e o que
 * fazer a seguir. Conta-se com o resultado registado, por isso aparece em qualquer run.
 *
 * Depois, o que pesa nas previsões, em três partes, pela ordem em que se pergunta:
 *
 * 1. **o que pesa no conjunto**: a importância por permutação (a pedido, porque volta a prever o
 *    teste muitas vezes);
 * 2. **como a previsão muda com uma feature**: a dependência parcial, com as curvas individuais
 *    e os traços onde há dados;
 * 3. **uma previsão de perto**: o contributo de cada feature para a previsão de uma linha do
 *    teste, com o "como se lê" que vem com ela (depende do método).
 *
 * Estas três precisam do modelo na memória da API: um run aberto do histórico só o tem se tiver
 * sido guardado.
 *
 * O cabeçalho é só o título: o modelo está na ligação de volta, e os dados do run ficam no "?"
 * (ADR 0038).
 */
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { ChevronLeft, ChevronRight, LoaderCircle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { ApiRequestError } from "@/api/errors";
import type {
  ExperimentResult,
  PartialDependence,
  PermutationImportance,
  RowExplanation,
} from "@/api/generated";
import {
  analysisExplainRowOptions,
  analysisPartialDependenceOptions,
  analysisPermutationImportanceOptions,
  analysisTestRowsOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useTaskInfo } from "@/api/vocabulary";
import { BarChart } from "@/components/charts/bar-chart";
import { LineChart, type LineSeries } from "@/components/charts/line-chart";
import { Explain } from "@/components/explain/explain";
import { TopicText } from "@/components/markdown/topic-text";
import { ErrorNote, Stat } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { PageSkeleton } from "@/features/data/require-dataset";
import { TrainingStoryView } from "@/features/explain/training-story";
import { useRunResult } from "@/features/training/use-run-result";
import { classColors, classOrder } from "@/lib/classes";
import { CATEGORICAL, categoryColor, OTHER } from "@/lib/colors";
import { formatNumber, formatPercent, formatStatistic } from "@/lib/format";
import { useSession } from "@/state/session";

/** Os contributos mostrados, no máximo (os mais fortes, em valor absoluto). */
const TOP_CONTRIBUTIONS = 12;

export function ExplainScreen() {
  const { runId } = useParams({ from: "/explain/$runId" });
  return (
    <Page>
      <ExplainRun key={runId} runId={runId} />
    </Page>
  );
}

function ExplainRun({ runId }: { runId: string }) {
  const { result } = useRunResult(runId);
  const openRun = useSession((state) => state.openRun);
  const data = result.data;

  // O run explicado passa a ser o último (o ecrã Treinar e avaliar volta a ele).
  useEffect(() => {
    openRun(runId);
  }, [runId, openRun]);

  if (result.error !== null) {
    return (
      <>
        <PageHeader title={SCREENS.explain.title} description={SCREENS.explain.description} />
        <ErrorNote error={result.error} className="mt-10" />
        <Actions runId={runId} />
      </>
    );
  }
  if (data === undefined) {
    return <PageSkeleton />;
  }
  return <Explained result={data} />;
}

function Explained({ result }: { result: ExperimentResult }) {
  const label = useLabel();
  const task = useTaskInfo(result.task_type);
  // Um pedido barato diz se o modelo está na memória da API (é dele que tudo depende).
  const rows = useQuery({
    ...analysisTestRowsOptions({ path: { run_id: result.run_id } }),
    enabled: task?.requires_target === true,
    staleTime: "static",
  });
  const noModel = rows.error instanceof ApiRequestError && rows.error.code === "run_not_in_memory";

  return (
    <>
      <Link
        to="/train/$runId"
        params={{ runId: result.run_id }}
        className="-ml-1 inline-flex items-center gap-1 text-sm text-soft hover:text-strong"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        {result.model_display_name}
      </Link>
      {/* O passo seguinte fica no cabeçalho, à direita, como em todos os ecrãs (ADR 0036). */}
      <header className="mt-4 flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <div className="flex items-center gap-2">
          <h1 className="text-title">{SCREENS.explain.title}</h1>
          <WhyButton
            title={SCREENS.explain.title}
            description={`${result.model_display_name} on ${result.dataset_name}, ${label("TaskType", result.task_type).toLowerCase()}. ${SCREENS.explain.description}`}
            topics={["permutation_importance", "partial_dependence", "local_explanation"]}
          />
        </div>
        <Button asChild variant="proceed" size="proceed">
          <Link to="/save/$runId" params={{ runId: result.run_id }}>
            Save the model
          </Link>
        </Button>
      </header>

      <div className="mt-14 flex flex-col gap-24">
        {/* Primeiro, o que aconteceu no treino: conta-se com o resultado, mesmo sem o modelo. */}
        <TrainingStoryView runId={result.run_id} />
        <section aria-labelledby="previsoes-titulo" className="flex flex-col gap-10">
          <div className="flex items-center gap-1.5">
            <h2 id="previsoes-titulo" className="text-2xl">
              What drives the predictions
            </h2>
            <Explain title="What drives the predictions">
              The model, put to the test: which features weigh the most, how the prediction follows
              each one, and why it decided one row the way it did.
            </Explain>
          </div>
          {task?.requires_target !== true ? (
            <Blocked
              message={`Explaining predictions needs a target, and ${label("TaskType", result.task_type).toLowerCase()} doesn't predict one. What the model found is in the story above, and in full in Train & evaluate.`}
              runId={result.run_id}
            />
          ) : noModel ? (
            <Blocked
              message="This part needs the model, and the API no longer has it: it only keeps the latest runs, and a run from the history only comes back with its model if it was saved. Replay this run in History to explain its predictions."
              runId={result.run_id}
              history
            />
          ) : rows.data === undefined ? (
            rows.error === null ? (
              <PageSkeleton />
            ) : (
              <ErrorNote error={rows.error} />
            )
          ) : (
            <div className="flex flex-col">
              <Part id="importance" title="What weighs overall" topic="permutation_importance">
                <Importance runId={result.run_id} />
              </Part>
              <Part
                id="dependence"
                title="How the prediction changes with a feature"
                topic="partial_dependence"
              >
                <Dependence result={result} />
              </Part>
              <Part id="prediction" title="One prediction up close" topic="local_explanation">
                <OneRow result={result} rows={rows.data} />
              </Part>
            </div>
          )}
        </section>
      </div>
    </>
  );
}

/** A importância por permutação: a pedido, porque volta a prever o teste muitas vezes. */
function Importance({ runId }: { runId: string }) {
  const [asked, setAsked] = useState(false);
  const importance = useQuery({
    ...analysisPermutationImportanceOptions({ path: { run_id: runId } }),
    enabled: asked,
    staleTime: "static",
  });
  if (!asked) {
    return (
      <div className="flex flex-wrap items-center gap-1.5">
        <Button
          onClick={() => {
            setAsked(true);
          }}
        >
          Compute the importance
        </Button>
        <Explain title="How long it takes">
          Shuffles each feature, one at a time, and predicts the test again: with many features, it
          takes a while.
        </Explain>
      </div>
    );
  }
  if (importance.error !== null) {
    return <ErrorNote error={importance.error} />;
  }
  if (importance.data === undefined) {
    return (
      <div role="status" className="flex items-center gap-2 text-sm text-dim">
        <LoaderCircle
          aria-hidden="true"
          className="size-4 animate-spin motion-reduce:animate-none"
        />
        Shuffling each feature…
      </div>
    );
  }
  return <ImportanceChart importance={importance.data} />;
}

function ImportanceChart({ importance }: { importance: PermutationImportance }) {
  const write = (value: number) =>
    importance.as_percent ? `${formatNumber(value * 100, 1)} pp` : formatNumber(value, 3);
  return (
    <div className="flex max-w-3xl flex-col gap-6">
      <Stat
        label={`${importance.metric_label} on the test, unshuffled`}
        value={
          importance.as_percent
            ? formatPercent(importance.baseline)
            : formatNumber(importance.baseline, 3)
        }
      />
      <BarChart
        items={importance.importances.map((item) => ({
          key: item.feature,
          label: item.feature,
          value: item.mean,
          error: item.std,
          // O que não pesa (perto de zero, ou negativo) fica cinzento: não é uma descoberta.
          color: item.mean > 0 ? CATEGORICAL[0] : OTHER,
        }))}
        format={(value) => write(value)}
        valueLabel={`How much it worsens: ${importance.metric_label}${importance.as_percent ? ", in percentage points" : ""}`}
        labelWidth={170}
        description={`How much ${importance.metric_label} worsens on the test when each feature is shuffled, from the one that weighs the most to the least.`}
        caption={<TopicText topic="importance_bars" />}
      />
    </div>
  );
}

/** A dependência parcial de uma feature, com as curvas individuais (ICE). */
function Dependence({ result }: { result: ExperimentResult }) {
  const [feature, setFeature] = useState(result.feature_names[0] ?? "");
  const dependence = useQuery({
    ...analysisPartialDependenceOptions({ path: { run_id: result.run_id }, query: { feature } }),
    enabled: feature !== "",
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  return (
    <div className="flex flex-col gap-6">
      <div className="flex max-w-sm flex-col gap-1.5">
        <label htmlFor="explicar-feature" className="small-label">
          Feature
        </label>
        <Select value={feature} onValueChange={setFeature}>
          <SelectTrigger id="explicar-feature" className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {result.feature_names.map((name) => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <ErrorNote error={dependence.error} />
      {dependence.data === undefined ? (
        dependence.error === null && <Skeleton className="h-80 w-full max-w-4xl" />
      ) : (
        <div className={dependence.isPlaceholderData ? "stale" : undefined}>
          <DependenceChart dependence={dependence.data} classLabels={result.class_labels} />
        </div>
      )}
    </div>
  );
}

function DependenceChart({
  dependence,
  classLabels,
}: {
  dependence: PartialDependence;
  classLabels: readonly string[];
}) {
  // Na classificação, as saídas são probabilidades (ou partes) de cada classe; a cor é a da
  // classe. Numa binária com probabilidades, só a da segunda classe vem (a outra é o resto).
  const classification = classLabels.length > 0;
  const offset = Math.max(0, classLabels.length - dependence.output_labels.length);
  const series: LineSeries[] = [
    ...dependence.individual.map((curve, index) => ({
      key: `ice-${String(index)}`,
      label: "Individual rows (ICE)",
      color: OTHER,
      faint: true,
      points: curve.map((y, i) => ({ x: dependence.grid[i] ?? 0, y })),
    })),
    ...dependence.average.map((curve, index) => ({
      key: `media-${String(index)}`,
      label: dependence.output_labels[index] ?? "Prediction",
      color: categoryColor(index + offset),
      width: 3,
      points: curve.map((y, i) => ({ x: dependence.grid[i] ?? 0, y })),
    })),
  ];
  return (
    <div className="max-w-4xl">
      <LineChart
        series={series}
        height={340}
        rug={dependence.values}
        xLabel={`${dependence.feature} (after the preparation)`}
        yLabel={dependence.output_labels.length === 1 ? dependence.output_labels[0] : "Prediction"}
        formatY={classification ? (value) => formatPercent(value, 0) : formatStatistic}
        formatX={formatStatistic}
        description={`How the mean prediction changes when ${dependence.feature} changes, with the curves of some test rows.`}
        caption={<TopicText topic="dependence_curves" />}
      />
    </div>
  );
}

/** Uma previsão do teste, explicada feature a feature. */
function OneRow({ result, rows }: { result: ExperimentResult; rows: number }) {
  const [row, setRow] = useState(0);
  const explanation = useQuery({
    ...analysisExplainRowOptions({ path: { run_id: result.run_id, row } }),
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  const go = (next: number) => {
    setRow(Math.max(0, Math.min(rows - 1, next)));
  };
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1.5">
          <label htmlFor="explicar-linha" className="small-label">
            Test row (1 to {rows})
          </label>
          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="icon"
              aria-label="Previous row"
              disabled={row === 0}
              onClick={() => {
                go(row - 1);
              }}
            >
              <ChevronLeft aria-hidden="true" />
            </Button>
            <Input
              id="explicar-linha"
              name="linha"
              autoComplete="off"
              type="number"
              inputMode="numeric"
              min={1}
              max={rows}
              value={row + 1}
              onChange={(event) => {
                const value = Number(event.target.value);
                if (Number.isFinite(value) && value >= 1) {
                  go(Math.round(value) - 1);
                }
              }}
              className="w-24 tabular-nums"
            />
            <Button
              variant="outline"
              size="icon"
              aria-label="Next row"
              disabled={row >= rows - 1}
              onClick={() => {
                go(row + 1);
              }}
            >
              <ChevronRight aria-hidden="true" />
            </Button>
          </div>
        </div>
      </div>
      <ErrorNote error={explanation.error} />
      {explanation.data === undefined ? (
        explanation.error === null && <Skeleton className="h-72 w-full max-w-4xl" />
      ) : (
        <div className={explanation.isPlaceholderData ? "stale" : undefined}>
          <RowView explanation={explanation.data} classLabels={result.class_labels} />
        </div>
      )}
    </div>
  );
}

function RowView({
  explanation,
  classLabels,
}: {
  explanation: RowExplanation;
  classLabels: readonly string[];
}) {
  const colorOf = classColors(classOrder(classLabels));
  const best = explanation.probabilities.reduce<[string, number] | null>(
    (top, pair) => (top === null || pair[1] > top[1] ? pair : top),
    null,
  );
  const shown = explanation.contributions.slice(0, TOP_CONTRIBUTIONS);
  return (
    <div className="grid gap-8 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
      <div className="flex min-w-0 flex-col gap-6">
        <div className="flex flex-wrap gap-x-10 gap-y-4">
          <Stat label="Prediction" value={explanation.prediction} />
          <Stat label="Actual value" value={explanation.actual ?? "—"} />
          {best !== null && (
            <Stat label={`Probability of “${best[0]}”`} value={formatPercent(best[1])} />
          )}
          {explanation.base !== null && (
            <Stat label="Intercept" value={formatStatistic(explanation.base)} />
          )}
        </div>
        {explanation.probabilities.length > 0 && (
          <BarChart
            items={explanation.probabilities.map(([name, probability]) => ({
              key: name,
              label: name,
              value: probability,
              color: colorOf(name),
            }))}
            max={1}
            format={(value) => formatPercent(value)}
            valueLabel="Probability"
            labelWidth={110}
            description="The probability the model gives each class on this row."
          />
        )}
      </div>
      <div className="flex min-w-0 flex-col gap-3">
        <h3 className="text-sm font-normal text-soft">{explanation.effect_label}</h3>
        <BarChart
          items={shown.map((item) => ({
            key: item.feature,
            label: `${item.feature} = ${formatStatistic(item.value)}`,
            value: item.effect,
            color: item.effect >= 0 ? CATEGORICAL[0] : CATEGORICAL[7],
          }))}
          format={(value) => formatStatistic(value)}
          valueLabel={explanation.effect_label}
          labelWidth={190}
          legend={[
            { label: "Pushes up", color: CATEGORICAL[0], kind: "rect" },
            { label: "Pushes down", color: CATEGORICAL[7], kind: "rect" },
          ]}
          description={`Each feature's contribution to this row's prediction (${explanation.effect_label.toLowerCase()}).`}
          caption={explanation.how_to_read}
        />
        {explanation.contributions.length > TOP_CONTRIBUTIONS && (
          <p className="small-label">
            The {TOP_CONTRIBUTIONS} features that weigh the most, out of{" "}
            {explanation.contributions.length}.
          </p>
        )}
      </div>
    </div>
  );
}

/** O ecrã não se pode usar para este run: diz porquê e leva onde se resolve. */
function Blocked({
  message,
  runId,
  history = false,
}: {
  message: string;
  runId: string;
  history?: boolean;
}) {
  return (
    <div className="flex max-w-2xl flex-col items-start gap-2 panel p-6">
      <p className="text-sm text-soft">{message}</p>
      <Actions runId={runId} history={history} />
    </div>
  );
}

function Actions({ runId, history = true }: { runId: string; history?: boolean }) {
  return (
    <div className="mt-4 flex flex-wrap gap-2">
      <Button asChild variant="outline">
        <Link to="/train/$runId" params={{ runId }}>
          Open the training
        </Link>
      </Button>
      {history && (
        <Button asChild variant="outline">
          <Link to="/history">Open the history</Link>
        </Button>
      )}
    </div>
  );
}

interface PartProps {
  id: string;
  title: string;
  /** O tema que explica a parte (no "?"). */
  topic: string;
  children: ReactNode;
}

/**
 * Uma das três perguntas: uma linha da página entre filetes, como os capítulos da história
 * (ADR 0048), com o título por cima e o que responde por baixo, na largura toda.
 */
function Part({ id, title, topic, children }: PartProps) {
  return (
    <section
      id={id}
      aria-labelledby={`${id}-titulo`}
      className="flex scroll-mt-6 flex-col gap-6 border-t border-[#202020] py-10"
    >
      <div className="flex items-center gap-1">
        <h3 id={`${id}-titulo`} className="text-lg font-medium text-strong">
          {title}
        </h3>
        <Explain topic={topic} />
      </div>
      {children}
    </section>
  );
}
