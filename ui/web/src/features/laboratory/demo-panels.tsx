/**
 * As seis demonstrações do Laboratório, cada uma com o seu controlo e os seus gráficos, cada
 * gráfico com o seu nome.
 *
 * Os valores de cada controlo e o de partida vêm do cartão da demonstração (o backend decide-os:
 * são uma escolha didática). As cores seguem as do resto da interface: o treino a azul, o teste
 * a amarelo, e o que o modelo desenha (a curva, o caminho dos pesos) a branco. Onde há duas
 * classes e não há treino e teste (o limiar), as classes têm as cores das classes.
 */
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { DemoCard, DemoControl } from "@/api/generated";
import {
  laboratoryDataSizeOptions,
  laboratoryKnnOptions,
  laboratoryLearningRateOptions,
  laboratoryOverfittingOptions,
  laboratoryRegularizationOptions,
  laboratoryThresholdOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { ContourChart } from "@/components/charts/contour-chart";
import { DotPlot } from "@/components/charts/dot-plot";
import { FieldView } from "@/components/charts/field";
import { Heatmap } from "@/components/charts/heatmap";
import { LineChart } from "@/components/charts/line-chart";
import { OptionSlider } from "@/components/forms/option-slider";
import { Panel, StatStrip } from "@/components/shell/blocks";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useDebounced } from "@/hooks/use-debounced";
import { CATEGORICAL, LIGHT_INK, OTHER, sequential } from "@/lib/colors";
import {
  formatCount,
  formatNumber,
  formatPercent,
  formatStatistic,
  formatValue,
} from "@/lib/format";

import { DemoFrame } from "./demo-frame";

const TRAIN = CATEGORICAL[0];
const TEST = CATEGORICAL[1];
/** O tempo que um controlo fica parado antes de se pedir a demonstração outra vez. */
const SETTLE_MS = 150;
const HEIGHT = 340;

/** O controlo de uma demonstração sem versões (todas menos a regularização). */
function controlOf(card: DemoCard): DemoControl {
  return card.control ?? { label: "", values: [], default: 0 };
}

export function OverfittingDemo({ card }: { card: DemoCard }) {
  const control = controlOf(card);
  const [degree, setDegree] = useState(control.default);
  const asked = useDebounced(degree, SETTLE_MS);
  const demo = useQuery({
    ...laboratoryOverfittingOptions({ query: { degree: asked } }),
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  const data = demo.data;
  return (
    <DemoFrame
      card={card}
      controls={
        <OptionSlider
          label={control.label}
          values={control.values}
          value={degree}
          onChange={setDegree}
        />
      }
      notices={data?.notices}
      stale={demo.isPlaceholderData || asked !== degree}
      error={demo.error}
    >
      {data === undefined ? null : (
        <>
          <Panel title="The fitted curve">
            <LineChart
              height={HEIGHT}
              series={[
                {
                  key: "truth",
                  label: "The truth (the sine)",
                  color: OTHER,
                  dashed: true,
                  points: data.truth.x.map((x, i) => ({ x, y: data.truth.y[i] ?? null })),
                },
                {
                  key: "curve",
                  label: `The degree ${String(data.degree)} polynomial`,
                  color: LIGHT_INK,
                  width: 2.5,
                  points: data.curve.x.map((x, i) => ({ x, y: data.curve.y[i] ?? null })),
                },
              ]}
              dots={[
                {
                  key: "test",
                  label: "Test",
                  color: TEST,
                  radius: 2.5,
                  opacity: 0.5,
                  below: true,
                  points: data.test.x.map((x, i) => ({ x, y: data.test.y[i] ?? 0 })),
                },
                {
                  key: "train",
                  label: "Train",
                  color: TRAIN,
                  radius: 4.5,
                  points: data.train.x.map((x, i) => ({ x, y: data.train.y[i] ?? 0 })),
                },
              ]}
              // Os polinómios de grau alto disparam entre os pontos: a vista fica no seno.
              yDomain={[-2.5, 2.5]}
              xLabel="x"
              yLabel="y"
              formatY={formatStatistic}
              formatX={formatStatistic}
              description={`The training and test points, the true sine and the degree ${String(data.degree)} polynomial.`}
            />
          </Panel>
          <Panel title="The error for each degree" topic="bias_variance">
            <LineChart
              height={HEIGHT}
              series={[
                {
                  key: "train",
                  label: "Training error",
                  color: TRAIN,
                  markers: true,
                  points: data.degrees.map((x, i) => ({ x, y: data.train_errors[i] ?? null })),
                },
                {
                  key: "test",
                  label: "Test error",
                  color: TEST,
                  markers: true,
                  points: data.degrees.map((x, i) => ({ x, y: data.test_errors[i] ?? null })),
                },
              ]}
              logY
              markerX={{ x: data.degree, label: `degree ${String(data.degree)}` }}
              xTicks={data.degrees.filter((x) => x % 2 === 1)}
              xLabel="Polynomial degree"
              yLabel="Error (RMSE)"
              formatX={(value) => formatNumber(value, 0)}
              formatY={formatStatistic}
              description="The training and test error for each polynomial degree."
            />
          </Panel>
        </>
      )}
    </DemoFrame>
  );
}

export function LearningRateDemo({ card }: { card: DemoCard }) {
  const control = controlOf(card);
  const [rate, setRate] = useState(control.default);
  const asked = useDebounced(rate, SETTLE_MS);
  const demo = useQuery({
    ...laboratoryLearningRateOptions({ query: { rate: asked } }),
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  const data = demo.data;
  return (
    <DemoFrame
      card={card}
      controls={
        <OptionSlider
          label={control.label}
          values={control.values}
          value={rate}
          onChange={setRate}
        />
      }
      notices={data?.notices}
      stale={demo.isPlaceholderData || asked !== rate}
      error={demo.error}
    >
      {data === undefined ? null : (
        <>
          <Panel title="The path down the error bowl">
            <ContourChart
              height={HEIGHT + 40}
              xs={data.surface_x}
              ys={data.surface_y}
              values={data.surface}
              valueLabel="Error (MSE)"
              path={{
                x: data.path.x,
                y: data.path.y,
                label: `The weights' path (rate ${formatValue(data.rate)})`,
              }}
              target={{ x: data.optimum[0], y: data.optimum[1], label: "The bottom of the bowl" }}
              xLabel="Weight w₁"
              yLabel="Weight w₂"
              description={`The error bowl as contour lines, with the weights' path epoch by epoch at rate ${formatValue(data.rate)}.`}
            />
          </Panel>
          <Panel title="The loss, epoch by epoch">
            <LineChart
              height={HEIGHT + 40}
              series={data.curves.map((curve) => {
                const chosen = curve.rate === data.rate;
                const name = `${formatValue(curve.rate)}${curve.diverged ? " (blew up)" : ""}`;
                return {
                  key: String(curve.rate),
                  label: chosen ? `Rate ${name}` : "Other rates",
                  color: chosen ? LIGHT_INK : OTHER,
                  width: chosen ? 3 : 1.5,
                  faint: !chosen,
                  endLabel: name,
                  points: curve.losses.map((loss, epoch) => ({ x: epoch + 1, y: loss })),
                };
              })}
              logY
              xLabel="Epoch"
              yLabel="Loss (MSE)"
              formatX={(value) => formatNumber(value, 0)}
              formatY={formatStatistic}
              description="The loss epoch by epoch with the chosen rate and with others, to compare."
            />
          </Panel>
        </>
      )}
    </DemoFrame>
  );
}

export function KnnDemo({ card }: { card: DemoCard }) {
  const control = controlOf(card);
  const [k, setK] = useState(control.default);
  const asked = useDebounced(k, SETTLE_MS);
  const demo = useQuery({
    ...laboratoryKnnOptions({ query: { k: asked } }),
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  const data = demo.data;
  return (
    <DemoFrame
      card={card}
      controls={
        <OptionSlider label={control.label} values={control.values} value={k} onChange={setK} />
      }
      notices={data?.notices}
      stale={demo.isPlaceholderData || asked !== k}
      error={demo.error}
    >
      {data === undefined ? null : (
        <>
          <Panel title="The boundary">
            {data.boundary === null ? (
              <p className="text-sm text-soft">No boundary to draw with this k.</p>
            ) : (
              <FieldView
                animation={data.boundary}
                height={HEIGHT + 40}
                description={`The KNN boundary with k = ${String(data.k)} on the moons.`}
                caption={data.boundary.how_to_read}
              />
            )}
          </Panel>
          <Panel title="The accuracy for each k">
            <LineChart
              height={HEIGHT}
              series={[
                {
                  key: "train",
                  label: "Training accuracy",
                  color: TRAIN,
                  markers: true,
                  points: data.ks.map((x, i) => ({ x, y: data.train_accuracy[i] ?? null })),
                },
                {
                  key: "test",
                  label: "Test accuracy",
                  color: TEST,
                  markers: true,
                  points: data.ks.map((x, i) => ({ x, y: data.test_accuracy[i] ?? null })),
                },
              ]}
              logX
              markerX={{ x: data.k, label: `k = ${String(data.k)}` }}
              xLabel="Number of neighbors (k)"
              yLabel="Accuracy"
              formatY={(value) => formatPercent(value, 0)}
              formatX={(value) => formatNumber(value, 0)}
              description="The training and test accuracy for each number of neighbors."
            />
          </Panel>
        </>
      )}
    </DemoFrame>
  );
}

export function RegularizationDemo({ card }: { card: DemoCard }) {
  const [kind, setKind] = useState(card.default_variant ?? card.variants[0]?.name ?? "lasso");
  const variant = card.variants.find((one) => one.name === kind) ?? card.variants[0];
  // Cada versão anda numa gama de alphas diferente: cada uma guarda o seu valor.
  const [alphas, setAlphas] = useState<Record<string, number>>(() =>
    Object.fromEntries(card.variants.map((one) => [one.name, one.control.default])),
  );
  const alpha = alphas[kind] ?? variant?.control.default ?? 1;
  // Espera-se por um texto (e não por um objeto novo a cada desenho, que nunca assentava).
  const settled = useDebounced(`${kind}|${String(alpha)}`, SETTLE_MS);
  const [askedKind, askedAlpha] = settled.split("|");
  const asked = {
    kind: askedKind === "ridge" ? "ridge" : "lasso",
    alpha: Number(askedAlpha),
  } as const;
  const demo = useQuery({
    ...laboratoryRegularizationOptions({ query: asked }),
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  const data = demo.data;
  const reach =
    data === undefined
      ? 1
      : Math.max(
          ...data.true_coefficients.map((value) => Math.abs(value)),
          ...data.chosen.map((value) => Math.abs(value)),
          0.5,
        );
  return (
    <DemoFrame
      card={card}
      controls={
        <>
          <div className="flex flex-col gap-2">
            <span id="regularizacao-modelo" className="text-sm font-medium">
              {card.variant_label ?? "Model"}
            </span>
            <ToggleGroup
              aria-labelledby="regularizacao-modelo"
              type="single"
              variant="outline"
              size="sm"
              value={kind}
              onValueChange={(chosen) => {
                if (chosen !== "") {
                  setKind(chosen);
                }
              }}
              className="justify-start"
            >
              {card.variants.map((one) => (
                <ToggleGroupItem key={one.name} value={one.name} className="px-3">
                  {one.label}
                </ToggleGroupItem>
              ))}
            </ToggleGroup>
          </div>
          {variant !== undefined && (
            <OptionSlider
              key={variant.name}
              label={variant.control.label}
              values={variant.control.values}
              value={alpha}
              onChange={(value) => {
                setAlphas((previous) => ({ ...previous, [kind]: value }));
              }}
            />
          )}
        </>
      }
      notices={data?.notices}
      stale={demo.isPlaceholderData || asked.kind !== kind || asked.alpha !== alpha}
      error={demo.error}
    >
      {data === undefined ? null : (
        <>
          <Panel title="Each weight as alpha grows">
            <LineChart
              height={HEIGHT + 20}
              series={data.feature_names.map((name, index) => {
                const counts = (data.true_coefficients[index] ?? 0) !== 0;
                return {
                  key: name,
                  label: counts ? `${name} (counts)` : name,
                  color: CATEGORICAL[index] ?? OTHER,
                  width: counts ? 2.5 : 1.5,
                  dotted: !counts,
                  points: data.alphas.map((x, i) => ({ x, y: data.paths[index]?.[i] ?? null })),
                };
              })}
              logX
              reference={{ y: 0, label: "" }}
              markerX={{ x: data.alpha, label: `alpha ${formatValue(data.alpha)}` }}
              xLabel="Alpha"
              yLabel="Weight"
              formatY={formatStatistic}
              description="Each feature's weight as alpha grows; the three that count are drawn solid."
            />
          </Panel>
          <Panel title="The R² for each alpha">
            <LineChart
              height={HEIGHT + 20}
              series={[
                {
                  key: "train",
                  label: "Training R²",
                  color: TRAIN,
                  points: data.alphas.map((x, i) => ({ x, y: data.train_r2[i] ?? null })),
                },
                {
                  key: "test",
                  label: "Test R²",
                  color: TEST,
                  points: data.alphas.map((x, i) => ({ x, y: data.test_r2[i] ?? null })),
                },
              ]}
              logX
              markerX={{ x: data.alpha, label: `alpha ${formatValue(data.alpha)}` }}
              yDomain={[Math.max(Math.min(...data.train_r2, ...data.test_r2, 0), -1) - 0.05, 1]}
              xLabel="Alpha"
              yLabel="R²"
              formatY={(value) => formatNumber(value, 1)}
              description="The training and test R² for each alpha."
            />
          </Panel>
          <Panel title="The weights with this alpha" className="max-w-3xl @5xl/main:col-span-2">
            <DotPlot
              rows={data.feature_names.map((name, index) => ({
                key: name,
                label: name,
                min: -reach,
                max: reach,
                values: [data.chosen[index] ?? null, data.true_coefficients[index] ?? null],
              }))}
              series={[
                {
                  key: "chosen",
                  label: `Weight with alpha ${formatValue(data.alpha)}`,
                  color: LIGHT_INK,
                },
                { key: "true", label: "True weight", color: OTHER },
              ]}
              format={formatStatistic}
              labelWidth={60}
              description="The weights with the chosen alpha, next to the true ones (only three features count)."
            />
          </Panel>
        </>
      )}
    </DemoFrame>
  );
}

const percent = (value: number) => formatPercent(value, 0);

/**
 * O desvio vertical do ponto número `index` dentro da sua faixa (até ±0,3), para os pontos com a
 * mesma probabilidade não se taparem. Sempre o mesmo para o mesmo ponto (a sequência da razão de
 * ouro espalha-os por igual): mexer no limiar não os faz saltar.
 */
function jitter(index: number): number {
  return (((index * 0.618_033_988_75) % 1) - 0.5) * 0.6;
}

export function ThresholdDemo({ card }: { card: DemoCard }) {
  const control = controlOf(card);
  const [threshold, setThreshold] = useState(control.default);
  const asked = useDebounced(threshold, SETTLE_MS);
  const demo = useQuery({
    ...laboratoryThresholdOptions({ query: { threshold: asked } }),
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  const data = demo.data;
  const [negative = "Negative", positive = "Positive"] = data?.class_labels ?? [];
  const [[tn = 0, fp = 0] = [], [fn = 0, tp = 0] = []] = data?.confusion ?? [];
  const marker =
    data === undefined
      ? undefined
      : { x: data.threshold, label: `threshold ${percent(data.threshold)}` };
  return (
    <DemoFrame
      card={card}
      controls={
        <OptionSlider
          label={control.label}
          values={control.values}
          value={threshold}
          onChange={setThreshold}
          format={percent}
        />
      }
      notices={data?.notices}
      stale={demo.isPlaceholderData || asked !== threshold}
      error={demo.error}
    >
      {data === undefined ? null : (
        <>
          <Panel title="How the model scores each row">
            {/* Uma faixa por classe, um ponto por linha do teste: à direita do limiar, o modelo
                diz "positivo". Os pontos da faixa de cima à esquerda da linha são os casos
                perdidos; os da faixa de baixo à direita, os alarmes falsos. */}
            <LineChart
              height={HEIGHT}
              series={[]}
              dots={[
                {
                  key: "negative",
                  label: `${negative} rows`,
                  color: CATEGORICAL[0],
                  radius: 2.5,
                  opacity: 0.55,
                  points: data.negatives.map((x, i) => ({ x, y: jitter(i) })),
                },
                {
                  key: "positive",
                  label: `${positive} rows`,
                  color: CATEGORICAL[1],
                  radius: 3,
                  opacity: 0.85,
                  points: data.positives.map((x, i) => ({ x, y: 1 + jitter(i) })),
                },
              ]}
              markerX={marker}
              xDomain={[0, 1]}
              yDomain={[-0.5, 1.5]}
              xLabel={`Predicted probability of “${positive}”`}
              formatX={percent}
              formatY={(value) => (value === 0 ? negative : value === 1 ? positive : "")}
              description={`Each test row as a dot at the probability the model gives it, one band per class, with the threshold at ${percent(data.threshold)}: to its right the model says “${positive}”.`}
            />
          </Panel>
          <Panel title="The decisions with this threshold" topic="confusion_matrix">
            <div className="flex flex-col gap-8">
              <div className="max-w-sm">
                <Heatmap
                  rows={data.class_labels}
                  columns={data.class_labels}
                  values={data.confusion}
                  color={(value) => sequential(value / Math.max(1, tn, fp, fn, tp))}
                  format={formatCount}
                  rowTitle="Actual"
                  columnTitle="Predicted"
                  cellHeight={52}
                  labelWidth={96}
                  highlightDiagonal
                  description={`The confusion matrix with the threshold at ${percent(data.threshold)}: how many rows of each actual class the model decided were each class.`}
                />
              </div>
              <StatStrip
                compact
                items={[
                  {
                    key: "found",
                    label: `${positive} found`,
                    value: formatCount(tp),
                    detail: `of ${formatCount(tp + fn)}`,
                  },
                  { key: "missed", label: "Missed", value: formatCount(fn) },
                  { key: "alarms", label: "False alarms", value: formatCount(fp) },
                  {
                    key: "flagged",
                    label: "Flagged in all",
                    value: formatCount(tp + fp),
                    detail: `of ${formatCount(tn + fp + fn + tp)}`,
                  },
                ]}
              />
            </div>
          </Panel>
          <Panel title="The scores for each threshold" className="@5xl/main:col-span-2">
            <LineChart
              height={HEIGHT}
              series={[
                {
                  key: "precision",
                  label: "Precision",
                  color: CATEGORICAL[2],
                  markers: true,
                  points: data.thresholds.map((x, i) => ({ x, y: data.precision[i] ?? null })),
                },
                {
                  key: "recall",
                  label: "Recall",
                  color: CATEGORICAL[3],
                  markers: true,
                  points: data.thresholds.map((x, i) => ({ x, y: data.recall[i] ?? null })),
                },
                {
                  key: "f1",
                  label: "F1",
                  color: LIGHT_INK,
                  width: 2.5,
                  points: data.thresholds.map((x, i) => ({ x, y: data.f1[i] ?? null })),
                },
                {
                  key: "accuracy",
                  label: "Accuracy",
                  color: OTHER,
                  dashed: true,
                  points: data.thresholds.map((x, i) => ({ x, y: data.accuracy[i] ?? null })),
                },
              ]}
              reference={{
                y: data.baseline_accuracy,
                label: `always “${negative}”: ${percent(data.baseline_accuracy)}`,
              }}
              markerX={marker}
              yDomain={[0, 1]}
              xLabel="Decision threshold"
              yLabel="Score"
              formatX={percent}
              formatY={percent}
              description="Precision, recall, F1 and accuracy for each threshold, with the accuracy of always answering the larger class."
            />
          </Panel>
        </>
      )}
    </DemoFrame>
  );
}

export function DataSizeDemo({ card }: { card: DemoCard }) {
  const control = controlOf(card);
  const [size, setSize] = useState(control.default);
  const asked = useDebounced(size, SETTLE_MS);
  const demo = useQuery({
    ...laboratoryDataSizeOptions({ query: { size: asked } }),
    staleTime: "static",
    placeholderData: (previous) => previous,
  });
  const data = demo.data;
  const band = (means: readonly number[], spreads: readonly number[]) =>
    (data?.sizes ?? []).map((x, i) => ({
      x,
      lower: (means[i] ?? 0) - (spreads[i] ?? 0),
      upper: (means[i] ?? 0) + (spreads[i] ?? 0),
    }));
  return (
    <DemoFrame
      card={card}
      controls={
        <OptionSlider
          label={control.label}
          values={control.values}
          value={size}
          onChange={setSize}
          format={(value) => formatCount(value)}
        />
      }
      notices={data?.notices}
      stale={demo.isPlaceholderData || asked !== size}
      error={demo.error}
    >
      {data === undefined ? null : (
        <>
          <Panel title={`The tree's boundary with ${formatCount(data.size)} rows`}>
            {data.boundary === null ? (
              <p className="text-sm text-soft">No boundary to draw with this amount.</p>
            ) : (
              <FieldView
                animation={data.boundary}
                height={HEIGHT + 40}
                description={`The boundary of a decision tree trained on ${formatCount(data.size)} rows of the moons.`}
                caption={data.boundary.how_to_read}
              />
            )}
          </Panel>
          <Panel title="The accuracy as the data grows" topic="learning_curve">
            <LineChart
              height={HEIGHT + 40}
              series={[
                {
                  key: "tree-train",
                  label: "Tree, training",
                  color: TRAIN,
                  points: data.sizes.map((x, i) => ({ x, y: data.tree_train[i] ?? null })),
                },
                {
                  key: "tree-test",
                  label: "Tree, test",
                  color: TEST,
                  width: 2.5,
                  band: band(data.tree_test, data.tree_spread),
                  points: data.sizes.map((x, i) => ({ x, y: data.tree_test[i] ?? null })),
                },
                {
                  key: "logistic-train",
                  label: "Logistic regression, training",
                  color: TRAIN,
                  dashed: true,
                  points: data.sizes.map((x, i) => ({ x, y: data.logistic_train[i] ?? null })),
                },
                {
                  key: "logistic-test",
                  label: "Logistic regression, test",
                  color: TEST,
                  dashed: true,
                  band: band(data.logistic_test, data.logistic_spread),
                  points: data.sizes.map((x, i) => ({ x, y: data.logistic_test[i] ?? null })),
                },
              ]}
              logX
              markerX={{ x: data.size, label: `${formatCount(data.size)} rows` }}
              xLabel="Training rows"
              yLabel="Accuracy"
              formatX={(value) => formatNumber(value, 0)}
              formatY={percent}
              description={`The training and test accuracy of the tree and of the logistic regression for each amount of training rows, always tested on the same ${formatCount(data.test_rows)} rows.`}
            />
          </Panel>
        </>
      )}
    </DemoFrame>
  );
}
