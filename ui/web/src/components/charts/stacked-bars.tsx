/**
 * Barras de 100%, uma por linha, cortadas nas partes de um todo: a proporção de cada classe no
 * treino, na validação e no teste. Com as partes alinhadas de linha para linha, vê-se logo se
 * as proporções se mantêm (é o que a divisão estratificada promete).
 *
 * Entre as partes fica uma folga de 2 px da cor do fundo; a percentagem escreve-se dentro da
 * parte quando cabe, e a dica diz o resto.
 */
import { scaleBand } from "d3";
import { useState, type ReactNode } from "react";

import { inkOn } from "@/lib/colors";

import { Chart, ChartTooltip, Legend, TooltipRow, type LegendItem } from "./chart";

export interface StackSegment {
  key: string;
  label: string;
  value: number;
  color: string;
  /** Uma linha a mais na dica (a contagem, por exemplo). */
  detail?: string;
}

export interface StackRow {
  key: string;
  label: string;
  segments: readonly StackSegment[];
}

interface StackedBarsProps {
  rows: readonly StackRow[];
  description: string;
  /** Escreve a parte de um segmento (de 0 a 1). */
  format: (share: number) => string;
  caption?: ReactNode;
  legend?: readonly LegendItem[];
  labelWidth?: number;
}

const ROW = 36;
const THICKNESS = 22;
const GAP = 2;

export function StackedBars({
  rows,
  description,
  format,
  caption,
  legend,
  labelWidth = 96,
}: StackedBarsProps) {
  const [hover, setHover] = useState<[string, string] | null>(null);
  return (
    <Chart
      height={rows.length * ROW + 4}
      description={description}
      caption={caption}
      legend={legend === undefined ? undefined : <Legend items={legend} />}
      margin={{ top: 2, bottom: 2, left: labelWidth, right: 8 }}
      overlay={(plot) => {
        if (hover === null) {
          return null;
        }
        const [rowKey, segmentKey] = hover;
        const index = rows.findIndex((row) => row.key === rowKey);
        const row = rows[index];
        const parts = row === undefined ? [] : layout(row, plot.width);
        const part = parts.find((one) => one.segment.key === segmentKey);
        if (row === undefined || part === undefined) {
          return null;
        }
        return (
          <ChartTooltip x={part.x + part.width / 2} y={index * ROW + ROW / 2} plot={plot}>
            <TooltipRow
              value={format(part.share)}
              label={part.segment.label}
              color={part.segment.color}
              kind="rect"
            />
            {part.segment.detail !== undefined && (
              <div className="text-muted-foreground">{part.segment.detail}</div>
            )}
            <div className="text-muted-foreground">{row.label}</div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const y = scaleBand()
          .domain(rows.map((row) => row.key))
          .range([0, plot.height]);
        return rows.map((row) => {
          const top = (y(row.key) ?? 0) + (y.bandwidth() - THICKNESS) / 2;
          const parts = layout(row, plot.width);
          return (
            <g key={row.key}>
              <text
                x={-12}
                y={top + THICKNESS / 2}
                dy="0.32em"
                textAnchor="end"
                className="fill-foreground text-[14px]"
              >
                {row.label}
              </text>
              {parts.map((part, index) => {
                const last = index === parts.length - 1;
                const width = Math.max(0, part.width - (last ? 0 : GAP));
                const isHover = hover?.[0] === row.key && hover[1] === part.segment.key;
                return (
                  <g
                    key={part.segment.key}
                    onPointerEnter={() => {
                      setHover([row.key, part.segment.key]);
                    }}
                    onPointerLeave={() => {
                      setHover(null);
                    }}
                  >
                    <path
                      d={segmentPath(part.x, top, width, THICKNESS, last)}
                      fill={part.segment.color}
                      stroke={isHover ? "var(--foreground)" : "none"}
                      strokeWidth={1.5}
                    />
                    {width >= 44 && (
                      <text
                        x={part.x + width / 2}
                        y={top + THICKNESS / 2}
                        dy="0.32em"
                        textAnchor="middle"
                        className="pointer-events-none text-[13px] tabular-nums"
                        fill={inkOn(part.segment.color)}
                      >
                        {format(part.share)}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          );
        });
      }}
    </Chart>
  );
}

interface Part {
  segment: StackSegment;
  share: number;
  x: number;
  width: number;
}

function layout(row: StackRow, width: number): Part[] {
  const total = row.segments.reduce((sum, segment) => sum + Math.max(0, segment.value), 0);
  let x = 0;
  return row.segments.map((segment) => {
    const share = total === 0 ? 0 : Math.max(0, segment.value) / total;
    const part = { segment, share, x, width: share * width };
    x += part.width;
    return part;
  });
}

/** Um segmento; o último tem a ponta dos dados arredondada (4 px). */
function segmentPath(x: number, y: number, width: number, height: number, end: boolean): string {
  if (width <= 0) {
    return "";
  }
  const r = end ? Math.min(4, width, height / 2) : 0;
  return `M${String(x)},${String(y)}h${String(width - r)}a${String(r)},${String(r)} 0 0 1 ${String(r)},${String(r)}v${String(height - 2 * r)}a${String(r)},${String(r)} 0 0 1 ${String(-r)},${String(r)}h${String(-(width - r))}z`;
}
