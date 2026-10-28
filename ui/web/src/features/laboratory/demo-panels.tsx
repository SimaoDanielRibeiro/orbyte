/**
 * As quatro demonstrações do Laboratório, cada uma com o seu controlo e os seus gráficos.
 *
 * Os valores de cada controlo e o de partida vêm do cartão da demonstração (o backend decide-os:
 * são uma escolha didática). As cores seguem as do resto da interface: o treino a azul, o teste
 * a laranja, e o que o modelo desenha (a curva, o caminho dos pesos) a giz.
 */
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { DemoCard, DemoControl } from "@/api/generated";
import {
  laboratoryKnnOptions,
  laboratoryLearningRateOptions,
  laboratoryOverfittingOptions,
  laboratoryRegularizationOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { ContourChart } from "@/components/charts/contour-chart";
import { DotPlot } from "@/components/charts/dot-plot";
import { FieldView } from "@/components/charts/field";
import { LineChart } from "@/components/charts/line-chart";
import { OptionSlider } from "@/components/forms/option-slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useDebounced } from "@/hooks/use-debounced";
import { CATEGORICAL, OTHER } from "@/lib/colors";
import { formatNumber, formatPercent, formatStatistic, formatValue } from "@/lib/format";

import { DemoFrame } from "./demo-frame";

const TRAIN = CATEGORICAL[0];
const TEST = CATEGORICAL[1];
const CHALK = "#ece7dd";
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
          <LineChart
            height={HEIGHT}
            series={[
              {
                key: "verdade",
                label: "A verdade (o seno)",
                color: OTHER,
                dashed: true,
                points: data.truth.x.map((x, i) => ({ x, y: data.truth.y[i] ?? null })),
              },
              {
                key: "curva",
                label: `O polinómio de grau ${String(data.degree)}`,
                color: CHALK,
                width: 2.5,
                points: data.curve.x.map((x, i) => ({ x, y: data.curve.y[i] ?? null })),
              },
            ]}
            dots={[
              {
                key: "teste",
                label: "Teste",
                color: TEST,
                radius: 2.5,
                opacity: 0.5,
                below: true,
                points: data.test.x.map((x, i) => ({ x, y: data.test.y[i] ?? 0 })),
              },
              {
                key: "treino",
                label: "Treino",
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
            description={`Os pontos de treino e de teste, o seno verdadeiro e o polinómio de grau ${String(data.degree)}.`}
          />
          <LineChart
            height={HEIGHT}
            series={[
              {
                key: "treino",
                label: "Erro no treino",
                color: TRAIN,
                markers: true,
                points: data.degrees.map((x, i) => ({ x, y: data.train_errors[i] ?? null })),
              },
              {
                key: "teste",
                label: "Erro no teste",
                color: TEST,
                markers: true,
                points: data.degrees.map((x, i) => ({ x, y: data.test_errors[i] ?? null })),
              },
            ]}
            logY
            markerX={{ x: data.degree, label: `grau ${String(data.degree)}` }}
            xTicks={data.degrees.filter((x) => x % 2 === 1)}
            xLabel="Grau do polinómio"
            yLabel="Erro (RMSE)"
            formatX={(value) => formatNumber(value, 0)}
            formatY={formatStatistic}
            description="O erro no treino e no teste para cada grau do polinómio."
          />
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
          <ContourChart
            height={HEIGHT + 40}
            xs={data.surface_x}
            ys={data.surface_y}
            values={data.surface}
            valueLabel="Erro (EQM)"
            path={{
              x: data.path.x,
              y: data.path.y,
              label: `O caminho dos pesos (taxa ${formatValue(data.rate)})`,
            }}
            target={{ x: data.optimum[0], y: data.optimum[1], label: "O fundo da taça" }}
            xLabel="Peso w₁"
            yLabel="Peso w₂"
            description={`A taça do erro em curvas de nível, com o caminho dos pesos época a época com a taxa ${formatValue(data.rate)}.`}
          />
          <LineChart
            height={HEIGHT + 40}
            series={data.curves.map((curve) => {
              const chosen = curve.rate === data.rate;
              const name = `${formatValue(curve.rate)}${curve.diverged ? " (explodiu)" : ""}`;
              return {
                key: String(curve.rate),
                label: chosen ? `Taxa ${name}` : "Outras taxas",
                color: chosen ? CHALK : OTHER,
                width: chosen ? 3 : 1.5,
                faint: !chosen,
                endLabel: name,
                points: curve.losses.map((loss, epoch) => ({ x: epoch + 1, y: loss })),
              };
            })}
            logY
            xLabel="Época"
            yLabel="Loss (EQM)"
            formatX={(value) => formatNumber(value, 0)}
            formatY={formatStatistic}
            description="A loss época a época com a taxa escolhida e com outras, para comparar."
          />
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
          {data.boundary === null ? (
            <p className="font-serif text-reading text-muted-foreground">
              Sem fronteira para desenhar com este k.
            </p>
          ) : (
            <FieldView
              animation={data.boundary}
              height={HEIGHT + 40}
              description={`A fronteira do KNN com k = ${String(data.k)} nas luas.`}
              caption={data.boundary.how_to_read}
            />
          )}
          <LineChart
            height={HEIGHT}
            series={[
              {
                key: "treino",
                label: "Exatidão no treino",
                color: TRAIN,
                markers: true,
                points: data.ks.map((x, i) => ({ x, y: data.train_accuracy[i] ?? null })),
              },
              {
                key: "teste",
                label: "Exatidão no teste",
                color: TEST,
                markers: true,
                points: data.ks.map((x, i) => ({ x, y: data.test_accuracy[i] ?? null })),
              },
            ]}
            logX
            markerX={{ x: data.k, label: `k = ${String(data.k)}` }}
            xLabel="Número de vizinhos (k)"
            yLabel="Exatidão"
            formatY={(value) => formatPercent(value, 0)}
            formatX={(value) => formatNumber(value, 0)}
            description="A exatidão no treino e no teste para cada número de vizinhos."
          />
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
              {card.variant_label ?? "Modelo"}
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
          <LineChart
            height={HEIGHT + 20}
            series={data.feature_names.map((name, index) => {
              const counts = (data.true_coefficients[index] ?? 0) !== 0;
              return {
                key: name,
                label: counts ? `${name} (conta)` : name,
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
            yLabel="Peso"
            formatY={formatStatistic}
            description="O peso de cada feature à medida que o alpha cresce; as três que contam a cheio."
          />
          <LineChart
            height={HEIGHT + 20}
            series={[
              {
                key: "treino",
                label: "R² no treino",
                color: TRAIN,
                points: data.alphas.map((x, i) => ({ x, y: data.train_r2[i] ?? null })),
              },
              {
                key: "teste",
                label: "R² no teste",
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
            description="O R² no treino e no teste para cada alpha."
          />
          <div className="max-w-3xl @5xl/main:col-span-2">
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
                  key: "escolhido",
                  label: `Peso com alpha ${formatValue(data.alpha)}`,
                  color: CHALK,
                },
                { key: "verdadeiro", label: "Peso verdadeiro", color: OTHER },
              ]}
              format={formatStatistic}
              labelWidth={60}
              description="Os pesos com o alpha escolhido, ao lado dos verdadeiros (só três features contam)."
            />
          </div>
        </>
      )}
    </DemoFrame>
  );
}
