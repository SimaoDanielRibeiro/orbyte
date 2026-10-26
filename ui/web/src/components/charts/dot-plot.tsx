/**
 * Um ponto por série em cada linha, cada linha com a sua própria escala: a média de cada
 * feature em cada classe do alvo, entre o mínimo e o máximo dessa feature.
 *
 * Cada feature tem as suas unidades (anos, euros, contagens), por isso uma escala comum não
 * serviria: cada linha vai do mínimo ao máximo da sua coluna, escritos nas pontas. O que se lê é
 * a distância entre os pontos de uma linha, comparada com a largura da coluna toda.
 */
import { scaleLinear } from "d3";
import { useState, type ReactNode } from "react";

import { Chart, ChartTooltip, Legend, TooltipRow } from "./chart";

export interface DotSeries {
  key: string;
  label: string;
  color: string;
}

export interface DotRow {
  key: string;
  label: string;
  min: number;
  max: number;
  /** Um valor por série, pela ordem das séries (`null`: sem valor). */
  values: readonly (number | null)[];
}

interface DotPlotProps {
  rows: readonly DotRow[];
  series: readonly DotSeries[];
  format: (value: number) => string;
  description: string;
  caption?: ReactNode;
  labelWidth?: number;
}

const ROW = 40;

export function DotPlot({
  rows,
  series,
  format,
  description,
  caption,
  labelWidth = 140,
}: DotPlotProps) {
  const [hover, setHover] = useState<string | null>(null);
  return (
    <Chart
      height={rows.length * ROW + 8}
      description={description}
      caption={caption}
      margin={{ top: 4, bottom: 4, left: labelWidth, right: 24 }}
      legend={
        <Legend
          items={series.map((one) => ({ label: one.label, color: one.color, kind: "dot" }))}
        />
      }
      overlay={(plot) => {
        const index = rows.findIndex((row) => row.key === hover);
        const row = rows[index];
        if (row === undefined) {
          return null;
        }
        const x = scaleLinear().domain(domain(row)).range([0, plot.width]);
        const present = row.values.filter((value): value is number => value !== null);
        return (
          <ChartTooltip x={x(Math.max(...present, row.min))} y={index * ROW + ROW / 2} plot={plot}>
            {series.map((one, position) => {
              const value = row.values[position] ?? null;
              return (
                <TooltipRow
                  key={one.key}
                  value={value === null ? "—" : format(value)}
                  label={one.label}
                  color={one.color}
                  kind="dot"
                />
              );
            })}
            <div className="mt-1 text-muted-foreground">{row.label}</div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) =>
        rows.map((row, index) => {
          const x = scaleLinear().domain(domain(row)).range([0, plot.width]);
          const center = index * ROW + ROW / 2 - 4;
          return (
            <g
              key={row.key}
              onPointerEnter={() => {
                setHover(row.key);
              }}
              onPointerLeave={() => {
                setHover(null);
              }}
            >
              <rect
                x={-plot.margin.left}
                y={index * ROW}
                width={plot.width + plot.margin.left + plot.margin.right}
                height={ROW}
                fill={hover === row.key ? "var(--accent)" : "transparent"}
                opacity={0.5}
              />
              <text
                x={-12}
                y={center}
                dy="0.32em"
                textAnchor="end"
                className="fill-foreground text-xs"
              >
                {row.label}
              </text>
              <line x1={0} x2={plot.width} y1={center} y2={center} stroke="var(--input)" />
              <text
                x={0}
                y={center + 15}
                className="fill-muted-foreground text-[10px] tabular-nums"
                aria-hidden="true"
              >
                {format(row.min)}
              </text>
              <text
                x={plot.width}
                y={center + 15}
                textAnchor="end"
                className="fill-muted-foreground text-[10px] tabular-nums"
                aria-hidden="true"
              >
                {format(row.max)}
              </text>
              {row.values.map((value, position) => {
                const one = series[position];
                return value === null || one === undefined ? null : (
                  <circle
                    key={one.key}
                    cx={x(value)}
                    cy={center}
                    r={5}
                    fill={one.color}
                    stroke="var(--background)"
                    strokeWidth={2}
                  />
                );
              })}
            </g>
          );
        })
      }
    </Chart>
  );
}

/** A escala de uma linha; uma coluna constante ganha uma largura para os pontos ficarem ao meio. */
function domain(row: DotRow): [number, number] {
  return row.max > row.min ? [row.min, row.max] : [row.min - 1, row.max + 1];
}
