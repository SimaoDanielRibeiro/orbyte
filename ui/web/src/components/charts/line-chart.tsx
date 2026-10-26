/**
 * Linhas ao longo de um eixo: a loss por passo, as curvas ROC, a curva de aprendizagem.
 *
 * Uma linha vertical segue o rato e encaixa no ponto mais próximo; a dica mostra o valor de
 * todas as séries nesse sítio. Uma série com `band` ganha a faixa da média ± desvio.
 */
import { area, bisector, extent, line, scaleLinear, scaleLog } from "d3";
import { useState, type ReactNode } from "react";

import { formatNumber, formatScientific } from "@/lib/format";

import { AxisBottom, AxisLeft, Chart, ChartTooltip, Legend, TooltipRow } from "./chart";
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
  /** Pinta a área por baixo da primeira série (a AUC). */
  fillFirst?: boolean;
  /** O eixo de y em escala logarítmica (valores que variam em ordens de grandeza). */
  logY?: boolean;
  /** As marcas do eixo de x, quando só alguns valores fazem sentido (os números dos folds). */
  xTicks?: readonly number[];
}

const center = bisector<number, number>((value) => value);

/** A posição, em `xs` (por ordem), do valor mais perto de `x`. */
function byX(xs: readonly number[], x: number): number {
  return center.center(xs, x);
}

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
  fillFirst = false,
  logY = false,
  xTicks: fixedTicks,
}: LineChartProps) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const all = series.flatMap((one) => one.points);
  const defined = all.filter(
    (point): point is { x: number; y: number } => point.y !== null && (!logY || point.y > 0),
  );
  const xs = [...new Set(all.map((point) => point.x))].sort((a, b) => a - b);
  const [xMin = 0, xMax = 1] = xDomain ?? extent(xs);
  const bandValues = series.flatMap((one) => (one.band ?? []).flatMap((b) => [b.lower, b.upper]));
  const [yMin = 0, yMax = 1] =
    yDomain ??
    extent(
      [...defined.map((point) => point.y), ...bandValues, reference?.y ?? NaN].filter(
        Number.isFinite,
      ),
    );

  /** A escala de y numa altura: linear, ou logarítmica (só com valores positivos). */
  const yScale = (plotHeight: number) =>
    logY
      ? scaleLog()
          .domain([Math.max(yMin, 1e-12), Math.max(yMax, Math.max(yMin, 1e-12) * 10)])
          .nice()
          .range([plotHeight, 0])
      : scaleLinear().domain([yMin, yMax]).nice().range([plotHeight, 0]);

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ bottom: xLabel === undefined ? 24 : 40, left: yLabel === undefined ? 48 : 60 }}
      legend={
        <Legend
          items={series.map((one) => ({
            label: one.label,
            color: one.color,
            kind: "line" as const,
          }))}
        />
      }
      overlay={(plot) => {
        if (hoverX === null) {
          return null;
        }
        const x = scaleLinear().domain([xMin, xMax]).range([0, plot.width]);
        const y = yScale(plot.height);
        const rows = series
          .map((one) => ({ one, point: nearest(one.points, hoverX) }))
          .filter(
            (row): row is { one: LineSeries; point: { x: number; y: number } } =>
              row.point !== null,
          );
        const first = rows[0];
        if (first === undefined) {
          return null;
        }
        const writeX = formatX ?? ((value: number) => formatNumber(value, 0));
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
        const x = scaleLinear().domain([xMin, xMax]).range([0, plot.width]);
        const y = yScale(plot.height);
        const xTicks = fixedTicks ?? x.ticks(Math.max(2, Math.floor(plot.width / 90)));
        // Na escala logarítmica, uma marca por potência de 10 (as do meio enchiam o eixo).
        const yTicks = logY
          ? y.ticks().filter((tick) => Math.abs(Math.log10(tick) % 1) < 1e-9)
          : y.ticks(5);
        const xDecimals = stepDecimals(xTicks);
        const yDecimals = stepDecimals(yTicks);
        const draw = line<LinePoint>()
          .defined((point) => point.y !== null && (!logY || point.y > 0))
          .x((point) => x(point.x))
          .y((point) => y(point.y ?? 0));
        const firstSeries = series[0];
        return (
          <>
            <AxisLeft
              ticks={yTicks}
              position={y}
              format={
                formatY ??
                (logY
                  ? (value) => formatScientific(value, 1)
                  : (value) => formatNumber(value, yDecimals))
              }
              plot={plot}
              label={yLabel}
            />
            <AxisBottom
              ticks={xTicks}
              position={x}
              format={formatX ?? ((value) => formatNumber(value, xDecimals))}
              plot={plot}
              label={xLabel}
            />
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
                  className="fill-muted-foreground text-[11px]"
                >
                  {reference.label}
                </text>
              </g>
            )}
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
            {series.map((one) => (
              <path
                key={one.key}
                d={draw(one.points as LinePoint[]) ?? ""}
                fill="none"
                stroke={one.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
                strokeDasharray={one.dashed === true ? "6 4" : undefined}
              />
            ))}
            {hoverX !== null && (
              <line x1={x(hoverX)} x2={x(hoverX)} y1={0} y2={plot.height} stroke="var(--input)" />
            )}
            {hoverX !== null &&
              series.map((one) => {
                const point = nearest(one.points, hoverX);
                return point === null ? null : (
                  <circle
                    key={`${one.key}-dot`}
                    cx={x(point.x)}
                    cy={y(point.y)}
                    r={4}
                    fill={one.color}
                    stroke="var(--background)"
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
                const value = x.invert(event.clientX - box.left);
                const index = byX(xs, value);
                setHoverX(xs[index] ?? null);
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
