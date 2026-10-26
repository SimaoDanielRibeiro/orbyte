/**
 * Barras horizontais: uma por categoria (a importância de cada feature, a parte de nulos de
 * cada coluna, o tamanho de cada classe, os pesos de um modelo linear).
 *
 * Barras finas (no máximo 24 px), a ponta dos dados arredondada e a base direita, o valor na
 * ponta. Com valores dos dois lados do zero (os pesos), as barras crescem a partir do zero.
 */
import { scaleBand, scaleLinear } from "d3";
import { useState, type ReactNode } from "react";

import { CATEGORICAL } from "@/lib/colors";
import { formatCount } from "@/lib/format";

import { Chart, ChartTooltip, TooltipRow } from "./chart";

export interface BarItem {
  key: string;
  label: string;
  value: number;
  color?: string;
}

interface BarChartProps {
  items: readonly BarItem[];
  description: string;
  /** O valor escrito na ponta da barra e na dica (com o item, para juntar uma percentagem). */
  format: (value: number, item: BarItem) => string;
  caption?: ReactNode;
  /** O que o valor é (na dica). */
  valueLabel?: string;
  /** O espaço para os nomes, à esquerda. */
  labelWidth?: number;
  /** O máximo da escala (por omissão, o maior valor). */
  max?: number;
}

const ROW = 30;
const THICKNESS = 18;

export function BarChart({
  items,
  description,
  format,
  caption,
  valueLabel = "",
  labelWidth = 140,
  max,
}: BarChartProps) {
  const [hover, setHover] = useState<string | null>(null);
  const values = items.map((item) => item.value);
  const signed = values.some((value) => value < 0);
  const top = max ?? Math.max(0, ...values);
  const bottom = signed ? Math.min(0, ...values) : 0;
  const height = items.length * ROW + 8;

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ top: 4, bottom: 4, left: labelWidth, right: 64 }}
      overlay={(plot) => {
        const item = items.find((one) => one.key === hover);
        if (item === undefined) {
          return null;
        }
        const y = scaleBand()
          .domain(items.map((one) => one.key))
          .range([0, plot.height]);
        const x = scaleLinear()
          .domain([bottom, top || 1])
          .range([0, plot.width]);
        return (
          <ChartTooltip
            x={x(Math.max(0, item.value))}
            y={(y(item.key) ?? 0) + y.bandwidth() / 2}
            plot={plot}
          >
            <TooltipRow value={format(item.value, item)} label={item.label} />
            {valueLabel !== "" && <div className="text-muted-foreground">{valueLabel}</div>}
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const y = scaleBand()
          .domain(items.map((one) => one.key))
          .range([0, plot.height]);
        const x = scaleLinear()
          .domain([bottom, top || 1])
          .range([0, plot.width]);
        const zero = x(0);
        return (
          <>
            {signed && <line x1={zero} x2={zero} y1={0} y2={plot.height} stroke="var(--input)" />}
            {items.map((item) => {
              const center = (y(item.key) ?? 0) + y.bandwidth() / 2;
              const end = x(item.value);
              const width = Math.abs(end - zero);
              const left = Math.min(end, zero);
              const color = item.color ?? CATEGORICAL[0];
              return (
                <g
                  key={item.key}
                  onPointerEnter={() => {
                    setHover(item.key);
                  }}
                  onPointerLeave={() => {
                    setHover(null);
                  }}
                >
                  <rect
                    x={-plot.margin.left}
                    y={center - ROW / 2}
                    width={plot.width + plot.margin.left + plot.margin.right}
                    height={ROW}
                    fill={hover === item.key ? "var(--accent)" : "transparent"}
                    opacity={0.5}
                  />
                  <text
                    x={-10}
                    y={center}
                    dy="0.32em"
                    textAnchor="end"
                    className="fill-foreground text-xs"
                  >
                    {truncate(item.label, Math.floor(plot.margin.left / 7))}
                  </text>
                  <path
                    d={bar(left, center - THICKNESS / 2, width, THICKNESS, item.value >= 0)}
                    fill={color}
                  />
                  <text
                    x={item.value >= 0 ? end + 6 : left - 6}
                    y={center}
                    dy="0.32em"
                    textAnchor={item.value >= 0 ? "start" : "end"}
                    className="fill-muted-foreground text-xs tabular-nums"
                  >
                    {format(item.value, item)}
                  </text>
                </g>
              );
            })}
          </>
        );
      }}
    </Chart>
  );
}

/** Uma barra com a ponta dos dados arredondada (4 px) e a base direita. */
function bar(x: number, y: number, width: number, height: number, rightEnd: boolean): string {
  const r = Math.min(4, width, height / 2);
  if (width <= 0) {
    return "";
  }
  if (rightEnd) {
    return `M${String(x)},${String(y)}h${String(width - r)}a${String(r)},${String(r)} 0 0 1 ${String(r)},${String(r)}v${String(height - 2 * r)}a${String(r)},${String(r)} 0 0 1 ${String(-r)},${String(r)}h${String(-(width - r))}z`;
  }
  return `M${String(x + width)},${String(y)}v${String(height)}h${String(-(width - r))}a${String(r)},${String(r)} 0 0 1 ${String(-r)},${String(-r)}v${String(-(height - 2 * r))}a${String(r)},${String(r)} 0 0 1 ${String(r)},${String(-r)}z`;
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, Math.max(1, length - 1))}…` : text;
}

export interface HistogramSeries {
  label: string;
  color: string;
  counts: readonly number[];
}

interface HistogramProps {
  /** Os limites dos intervalos: um a mais do que as contagens. */
  edges: readonly number[];
  series: readonly HistogramSeries[];
  description: string;
  formatEdge: (value: number) => string;
  caption?: ReactNode;
  height?: number;
  xLabel?: string;
  /** Uma linha vertical (o limiar de anomalia). */
  marker?: { value: number; label: string };
}

/** Um histograma em colunas: as séries lado a lado em cada intervalo, com 2 px de folga. */
export function Histogram({
  edges,
  series,
  description,
  formatEdge,
  caption,
  height = 220,
  xLabel,
  marker,
}: HistogramProps) {
  const [hover, setHover] = useState<number | null>(null);
  const bins = Math.max(0, edges.length - 1);
  const peak = Math.max(1, ...series.flatMap((one) => one.counts));
  const first = edges[0] ?? 0;
  const last = edges[edges.length - 1] ?? 1;

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ left: 40, bottom: xLabel === undefined ? 24 : 40 }}
      legend={<ChartLegend series={series} />}
      overlay={(plot) => {
        if (hover === null) {
          return null;
        }
        const x = scaleLinear().domain([first, last]).range([0, plot.width]);
        const y = scaleLinear().domain([0, peak]).nice().range([plot.height, 0]);
        const low = edges[hover] ?? 0;
        const high = edges[hover + 1] ?? 0;
        const tallest = Math.max(...series.map((one) => one.counts[hover] ?? 0));
        return (
          <ChartTooltip x={x((low + high) / 2)} y={y(tallest)} plot={plot}>
            <div className="mb-1 text-muted-foreground">
              {formatEdge(low)} a {formatEdge(high)}
            </div>
            {series.map((one) => (
              <TooltipRow
                key={one.label}
                value={formatCount(one.counts[hover] ?? 0)}
                label={one.label}
                color={one.color}
                kind="rect"
              />
            ))}
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const x = scaleLinear().domain([first, last]).range([0, plot.width]);
        const y = scaleLinear().domain([0, peak]).nice().range([plot.height, 0]);
        const yTicks = y.ticks(4).filter(Number.isInteger);
        const xTicks = x.ticks(Math.max(2, Math.floor(plot.width / 90)));
        return (
          <>
            {yTicks.map((tick) => (
              <g key={tick} aria-hidden="true">
                <line x1={0} x2={plot.width} y1={y(tick)} y2={y(tick)} stroke="var(--border)" />
                <text
                  x={-8}
                  y={y(tick)}
                  dy="0.32em"
                  textAnchor="end"
                  className="fill-muted-foreground text-[11px] tabular-nums"
                >
                  {formatCount(tick)}
                </text>
              </g>
            ))}
            <line x1={0} x2={plot.width} y1={plot.height} y2={plot.height} stroke="var(--input)" />
            {xTicks.map((tick) => (
              <text
                key={tick}
                x={x(tick)}
                y={plot.height + 16}
                textAnchor="middle"
                className="fill-muted-foreground text-[11px] tabular-nums"
              >
                {formatEdge(tick)}
              </text>
            ))}
            {xLabel !== undefined && (
              <text
                x={plot.width / 2}
                y={plot.height + 32}
                textAnchor="middle"
                className="fill-muted-foreground text-[11px]"
              >
                {xLabel}
              </text>
            )}
            {Array.from({ length: bins }, (_, bin) => {
              const left = x(edges[bin] ?? 0);
              const right = x(edges[bin + 1] ?? 0);
              const slot = (right - left - 2) / Math.max(1, series.length);
              return (
                <g
                  key={bin}
                  onPointerEnter={() => {
                    setHover(bin);
                  }}
                  onPointerLeave={() => {
                    setHover(null);
                  }}
                >
                  <rect
                    x={left}
                    y={0}
                    width={right - left}
                    height={plot.height}
                    fill={hover === bin ? "var(--accent)" : "transparent"}
                    opacity={0.5}
                  />
                  {series.map((one, index) => {
                    const count = one.counts[bin] ?? 0;
                    const top = y(count);
                    const columnHeight = plot.height - top;
                    return columnHeight <= 0 ? null : (
                      <path
                        key={one.label}
                        d={column(
                          left + 1 + index * slot,
                          top,
                          Math.max(1, slot - (series.length > 1 ? 1 : 0)),
                          columnHeight,
                        )}
                        fill={one.color}
                      />
                    );
                  })}
                </g>
              );
            })}
            {marker !== undefined && (
              <g>
                <line
                  x1={x(marker.value)}
                  x2={x(marker.value)}
                  y1={0}
                  y2={plot.height}
                  stroke="var(--foreground)"
                  strokeDasharray="4 3"
                />
                <text x={x(marker.value) + 4} y={10} className="fill-foreground text-[11px]">
                  {marker.label}
                </text>
              </g>
            )}
          </>
        );
      }}
    </Chart>
  );
}

function ChartLegend({ series }: { series: readonly HistogramSeries[] }) {
  if (series.length < 2) {
    return null;
  }
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {series.map((one) => (
        <li key={one.label} className="flex items-center gap-1.5">
          <span
            aria-hidden="true"
            className="size-2.5 rounded-[2px]"
            style={{ background: one.color }}
          />
          {one.label}
        </li>
      ))}
    </ul>
  );
}

/** Uma coluna com o topo arredondado (4 px) e a base direita, no eixo. */
function column(x: number, y: number, width: number, height: number): string {
  const r = Math.min(4, width / 2, height);
  return `M${String(x)},${String(y + height)}v${String(-(height - r))}a${String(r)},${String(r)} 0 0 1 ${String(r)},${String(-r)}h${String(width - 2 * r)}a${String(r)},${String(r)} 0 0 1 ${String(r)},${String(r)}v${String(height - r)}z`;
}
