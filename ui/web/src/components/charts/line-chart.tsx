/**
 * Linhas ao longo de um eixo: a loss por passo, as curvas ROC, a curva de aprendizagem, as
 * curvas das demonstrações do Laboratório.
 *
 * Uma linha vertical segue o rato e encaixa no valor de x mais próximo; a dica mostra o valor de
 * todas as séries nesse sítio. Uma série com `band` ganha a faixa da média ± desvio; uma série
 * `faint` (as curvas individuais de uma dependência parcial) fica fina e translúcida, fora da
 * dica, e tem uma só chave na legenda. Por baixo das linhas, ou por cima, podem ir nuvens de
 * pontos (`dots`). Tudo se recorta à área do desenho: com um domínio fixo, o que sai dele não
 * pinta os eixos.
 */
import { area, extent, line, scaleLinear, scaleLog } from "d3";
import { useId, useState, type ReactNode } from "react";

import { formatNumber, formatScientific, formatValue } from "@/lib/format";

import {
  AxisBottom,
  AxisLeft,
  Chart,
  ChartTooltip,
  GlowFilter,
  Legend,
  TooltipRow,
  type LegendItem,
} from "./chart";
import { leftMargin } from "./margins";
import { stepDecimals } from "./ticks";

export interface LinePoint {
  x: number;
  y: number | null;
}

export interface LineSeries {
  key: string;
  label: string;
  color: string;
  points: readonly LinePoint[];
  /** A faixa à volta da linha (por exemplo, a média ± o desvio padrão). */
  band?: readonly { x: number; lower: number; upper: number }[];
  dashed?: boolean;
  /** Pontilhada: uma série que serve de referência (os pesos das features que não contam). */
  dotted?: boolean;
  /** A espessura, em px (por omissão, 2). */
  width?: number;
  /** Um ponto em cada valor (curvas com poucos valores, como o erro de cada grau). */
  markers?: boolean;
  /** Fina e translúcida, fora da dica: as curvas individuais, que só se leem em conjunto. */
  faint?: boolean;
  /** O nome escrito no fim da linha, para a identificar sem procurar na legenda (também nas
   * ténues: é o que distingue umas das outras). */
  endLabel?: string;
}

/** Uma nuvem de pontos desenhada com as linhas (os pontos de treino e de teste, por exemplo). */
export interface DotCloud {
  key: string;
  label: string;
  color: string;
  points: readonly { x: number; y: number }[];
  /** O raio, em px (por omissão, 3,5). */
  radius?: number;
  opacity?: number;
  /** Por baixo das linhas, em vez de por cima. */
  below?: boolean;
}

interface LineChartProps {
  series: readonly LineSeries[];
  description: string;
  caption?: ReactNode;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  /** Como escrever os valores de y (na dica e no eixo). */
  formatY?: (value: number) => string;
  formatX?: (value: number) => string;
  /** A diagonal do acaso (nas curvas ROC). */
  diagonal?: boolean;
  /** Uma linha horizontal de referência. */
  reference?: { y: number; label: string };
  /** Uma linha vertical num valor de x (o valor escolhido num controlo). */
  markerX?: { x: number; label: string };
  /** Pinta a área por baixo da primeira série (a AUC). */
  fillFirst?: boolean;
  /** O eixo de y em escala logarítmica (valores que variam em ordens de grandeza). */
  logY?: boolean;
  /** O eixo de x em escala logarítmica (o k do KNN, o alpha da regularização). */
  logX?: boolean;
  /** As marcas do eixo de x, quando só alguns valores fazem sentido (os números dos folds). */
  xTicks?: readonly number[];
  /** Nuvens de pontos, por baixo ou por cima das linhas. */
  dots?: readonly DotCloud[];
  /** Os valores de x onde há dados: um traço curto em cada um, junto ao eixo. */
  rug?: readonly number[];
}

/** O mínimo de uma escala logarítmica (os zeros e os negativos não têm lugar nela). */
const LOG_FLOOR = 1e-12;

export function LineChart({
  series,
  description,
  caption,
  height = 260,
  xLabel,
  yLabel,
  xDomain,
  yDomain,
  formatY,
  formatX,
  diagonal = false,
  reference,
  markerX,
  fillFirst = false,
  logY = false,
  logX = false,
  xTicks: fixedTicks,
  dots = [],
  rug = [],
}: LineChartProps) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const clipId = `linhas-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const solid = series.filter((one) => one.faint !== true);
  const faint = series.filter((one) => one.faint === true);
  const all = series.flatMap((one) => one.points);
  const defined = all.filter(
    (point): point is { x: number; y: number } => point.y !== null && (!logY || point.y > 0),
  );
  const dotPoints = dots.flatMap((cloud) => cloud.points);
  const xs = [...new Set(solid.flatMap((one) => one.points.map((point) => point.x)))].sort(
    (a, b) => a - b,
  );
  const [xMin = 0, xMax = 1] =
    xDomain ?? extent([...all.map((point) => point.x), ...dotPoints.map((point) => point.x)]);
  const bandValues = series.flatMap((one) => (one.band ?? []).flatMap((b) => [b.lower, b.upper]));
  const [yMin = 0, yMax = 1] =
    yDomain ??
    extent(
      [
        ...defined.map((point) => point.y),
        ...dotPoints.map((point) => point.y),
        ...bandValues,
        reference?.y ?? NaN,
      ].filter(Number.isFinite),
    );

  /** A escala de x numa largura: linear, ou logarítmica (só com valores positivos). */
  const xScale = (plotWidth: number) =>
    logX
      ? scaleLog()
          .domain([Math.max(xMin, LOG_FLOOR), Math.max(xMax, Math.max(xMin, LOG_FLOOR) * 10)])
          .range([0, plotWidth])
      : scaleLinear().domain([xMin, xMax]).range([0, plotWidth]);
  /** A escala de y numa altura: linear, ou logarítmica (só com valores positivos). */
  const yScale = (plotHeight: number) =>
    logY
      ? scaleLog()
          .domain([Math.max(yMin, LOG_FLOOR), Math.max(yMax, Math.max(yMin, LOG_FLOOR) * 10)])
          .nice()
          .range([plotHeight, 0])
      : scaleLinear().domain([yMin, yMax]).nice().range([plotHeight, 0]);

  // As marcas de y dependem só do domínio, não da altura: calculam-se já, para a margem da
  // esquerda caber o número mais comprido.
  const yTicks = logY
    ? yScale(1)
        .ticks()
        .filter((tick) => Math.abs(Math.log10(tick) % 1) < 1e-9)
    : yScale(1).ticks(5);
  const yDecimals = stepDecimals(yTicks);
  const writeTick =
    formatY ??
    (logY
      ? (value: number) => formatScientific(value, 1)
      : (value: number) => formatNumber(value, yDecimals));

  const legend: LegendItem[] = [
    ...solid.map((one) => ({ label: one.label, color: one.color, kind: "line" as const })),
    ...faint
      .slice(0, 1)
      .map((one) => ({ label: one.label, color: one.color, kind: "line" as const })),
    ...dots.map((cloud) => ({ label: cloud.label, color: cloud.color, kind: "dot" as const })),
  ];
  const endLabels = series.some((one) => one.endLabel !== undefined);

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{
        bottom: xLabel === undefined ? 24 : 40,
        left: leftMargin(yTicks.map(writeTick), yLabel !== undefined),
        right: endLabels ? 52 : 16,
      }}
      legend={<Legend items={legend} />}
      overlay={(plot) => {
        if (hoverX === null) {
          return null;
        }
        const x = xScale(plot.width);
        const y = yScale(plot.height);
        const rows = solid
          .map((one) => ({ one, point: nearest(one.points, hoverX) }))
          .filter(
            (row): row is { one: LineSeries; point: { x: number; y: number } } =>
              row.point !== null,
          );
        const first = rows[0];
        if (first === undefined) {
          return null;
        }
        const writeX =
          formatX ??
          (logX
            ? (value: number) => formatValue(value)
            : (value: number) => formatNumber(value, 0));
        const writeY = formatY ?? ((value: number) => formatNumber(value, 3));
        return (
          <ChartTooltip x={x(hoverX)} y={y(first.point.y)} plot={plot}>
            {xLabel !== undefined && (
              <div className="mb-1 text-muted-foreground">
                {xLabel} {writeX(hoverX)}
              </div>
            )}
            {rows.map(({ one, point }) => (
              <TooltipRow
                key={one.key}
                value={writeY(point.y)}
                label={one.label}
                color={one.color}
              />
            ))}
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const x = xScale(plot.width);
        const y = yScale(plot.height);
        // Nas escalas logarítmicas, uma marca por potência de 10 (as do meio enchiam o eixo).
        const xTicks =
          fixedTicks ??
          (logX
            ? x.ticks().filter((tick) => Math.abs(Math.log10(tick) % 1) < 1e-9)
            : x.ticks(Math.max(2, Math.floor(plot.width / 90))));
        const xDecimals = stepDecimals(xTicks);
        const draw = line<LinePoint>()
          .defined((point) => point.y !== null && (!logY || point.y > 0))
          .x((point) => x(point.x))
          .y((point) => y(point.y ?? 0));
        const firstSeries = solid[0];
        const cloud = (one: DotCloud) => (
          <g key={one.key} pointerEvents="none">
            {one.points.map((point, index) => (
              <circle
                key={index}
                cx={x(point.x)}
                cy={y(point.y)}
                r={one.radius ?? 3.5}
                fill={one.color}
                fillOpacity={one.opacity ?? 1}
                stroke="var(--card)"
                strokeWidth={1.5}
              />
            ))}
          </g>
        );
        return (
          <>
            <defs>
              <clipPath id={clipId}>
                <rect x={-6} y={-6} width={plot.width + 12} height={plot.height + 12} />
              </clipPath>
              <GlowFilter id={`${clipId}-brilho`} />
            </defs>
            <AxisLeft ticks={yTicks} position={y} format={writeTick} plot={plot} label={yLabel} />
            <AxisBottom
              ticks={xTicks}
              position={x}
              format={
                formatX ??
                (logX ? (value) => formatValue(value) : (value) => formatNumber(value, xDecimals))
              }
              plot={plot}
              label={xLabel}
            />
            <g clipPath={`url(#${clipId})`}>
              {diagonal && (
                <line
                  x1={x(0)}
                  y1={y(0)}
                  x2={x(1)}
                  y2={y(1)}
                  stroke="var(--input)"
                  strokeDasharray="4 4"
                />
              )}
              {reference !== undefined && (
                <g>
                  <line
                    x1={0}
                    x2={plot.width}
                    y1={y(reference.y)}
                    y2={y(reference.y)}
                    stroke="var(--input)"
                    strokeDasharray="4 4"
                  />
                  <text
                    x={plot.width}
                    y={y(reference.y) - 6}
                    textAnchor="end"
                    className="fill-dim text-[13px]"
                  >
                    {reference.label}
                  </text>
                </g>
              )}
              {dots.filter((one) => one.below === true).map(cloud)}
              {fillFirst && firstSeries !== undefined && (
                <path
                  d={
                    area<LinePoint>()
                      .defined((point) => point.y !== null)
                      .x((point) => x(point.x))
                      .y0(plot.height)
                      .y1((point) => y(point.y ?? 0))(firstSeries.points as LinePoint[]) ?? ""
                  }
                  fill={firstSeries.color}
                  fillOpacity={0.12}
                />
              )}
              {series.map((one) =>
                one.band === undefined ? null : (
                  <path
                    key={`${one.key}-band`}
                    d={
                      area<{ x: number; lower: number; upper: number }>()
                        .x((point) => x(point.x))
                        .y0((point) => y(point.lower))
                        .y1((point) => y(point.upper))(
                        one.band as { x: number; lower: number; upper: number }[],
                      ) ?? ""
                    }
                    fill={one.color}
                    fillOpacity={0.1}
                  />
                ),
              )}
              {faint.map((one) => (
                <path
                  key={one.key}
                  d={draw(one.points as LinePoint[]) ?? ""}
                  fill="none"
                  stroke={one.color}
                  strokeWidth={one.width ?? 1}
                  strokeOpacity={0.35}
                  strokeLinejoin="round"
                />
              ))}
              {/* O brilho: uma cópia larga e desfocada por baixo de cada linha cheia. */}
              {solid
                .filter((one) => one.dotted !== true && one.dashed !== true)
                .map((one) => (
                  <path
                    key={`${one.key}-brilho`}
                    d={draw(one.points as LinePoint[]) ?? ""}
                    fill="none"
                    stroke={one.color}
                    strokeWidth={(one.width ?? 2) * 2.5}
                    strokeOpacity={0.4}
                    strokeLinejoin="round"
                    filter={`url(#${clipId}-brilho)`}
                    pointerEvents="none"
                  />
                ))}
              {solid.map((one) => (
                <path
                  key={one.key}
                  d={draw(one.points as LinePoint[]) ?? ""}
                  fill="none"
                  stroke={one.color}
                  strokeWidth={one.width ?? 2}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                  strokeDasharray={
                    one.dotted === true ? "1 4" : one.dashed === true ? "6 4" : undefined
                  }
                />
              ))}
              {solid
                .filter((one) => one.markers === true)
                .map((one) => (
                  <g key={`${one.key}-markers`} pointerEvents="none">
                    {one.points.map((point) =>
                      point.y === null || (logY && point.y <= 0) ? null : (
                        <circle
                          key={point.x}
                          cx={x(point.x)}
                          cy={y(point.y)}
                          r={3}
                          fill={one.color}
                          stroke="var(--card)"
                          strokeWidth={1.5}
                        />
                      ),
                    )}
                  </g>
                ))}
              {dots.filter((one) => one.below !== true).map(cloud)}
            </g>
            {series.map((one) => {
              const last = [...one.points]
                .reverse()
                .find((point): point is { x: number; y: number } => point.y !== null);
              return one.endLabel === undefined || last === undefined ? null : (
                <text
                  key={`${one.key}-end`}
                  x={x(last.x) + 6}
                  y={y(last.y)}
                  dy="0.32em"
                  className="fill-soft text-[13px] tabular-nums"
                  aria-hidden="true"
                >
                  {one.endLabel}
                </text>
              );
            })}
            {rug.length > 0 && (
              <g aria-hidden="true" pointerEvents="none">
                {rug.map((value, index) => (
                  <line
                    key={index}
                    x1={x(value)}
                    x2={x(value)}
                    y1={plot.height - 7}
                    y2={plot.height}
                    stroke="var(--muted-foreground)"
                    strokeOpacity={0.5}
                  />
                ))}
              </g>
            )}
            {markerX !== undefined && (
              <g aria-hidden="true" pointerEvents="none">
                <line
                  x1={x(markerX.x)}
                  x2={x(markerX.x)}
                  y1={0}
                  y2={plot.height}
                  stroke="var(--foreground)"
                  strokeDasharray="4 3"
                />
                <text
                  x={x(markerX.x) + (x(markerX.x) > plot.width * 0.75 ? -4 : 4)}
                  y={10}
                  textAnchor={x(markerX.x) > plot.width * 0.75 ? "end" : "start"}
                  className="fill-foreground text-[13px]"
                >
                  {markerX.label}
                </text>
              </g>
            )}
            {hoverX !== null && (
              <line x1={x(hoverX)} x2={x(hoverX)} y1={0} y2={plot.height} stroke="var(--input)" />
            )}
            {hoverX !== null &&
              solid.map((one) => {
                const point = nearest(one.points, hoverX);
                return point === null ? null : (
                  <circle
                    key={`${one.key}-dot`}
                    cx={x(point.x)}
                    cy={y(point.y)}
                    r={4}
                    fill={one.color}
                    stroke="var(--card)"
                    strokeWidth={2}
                  />
                );
              })}
            <rect
              width={plot.width}
              height={plot.height}
              fill="transparent"
              onPointerMove={(event) => {
                const box = event.currentTarget.getBoundingClientRect();
                setHoverX(closest(xs, (value) => x(value), event.clientX - box.left));
              }}
              onPointerLeave={() => {
                setHoverX(null);
              }}
            />
          </>
        );
      }}
    </Chart>
  );
}

/** O valor de `xs` cuja posição no desenho fica mais perto de `pixel` (em qualquer escala). */
function closest(
  xs: readonly number[],
  position: (value: number) => number,
  pixel: number,
): number | null {
  let best: number | null = null;
  for (const value of xs) {
    if (best === null || Math.abs(position(value) - pixel) < Math.abs(position(best) - pixel)) {
      best = value;
    }
  }
  return best;
}

/** O ponto definido de `points` mais perto de `x`. */
function nearest(points: readonly LinePoint[], x: number): { x: number; y: number } | null {
  let best: { x: number; y: number } | null = null;
  for (const point of points) {
    if (point.y !== null && (best === null || Math.abs(point.x - x) < Math.abs(best.x - x))) {
      best = { x: point.x, y: point.y };
    }
  }
  return best;
}
