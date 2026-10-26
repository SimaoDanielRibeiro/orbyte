/**
 * O perfil da silhueta: uma barra fina por ponto, agrupada pelo grupo do ponto e ordenada da
 * maior para a menor. Perto de 1, o ponto está bem no seu grupo; negativa, estaria melhor
 * noutro. A linha tracejada é a média.
 */
import { scaleLinear } from "d3";
import { useState, type ReactNode } from "react";

import type { Silhouette } from "@/api/generated";
import { categoryColor } from "@/lib/colors";
import { formatCount, formatNumber } from "@/lib/format";

import { AxisBottom, Chart, ChartTooltip, Legend, TooltipRow } from "./chart";

interface SilhouetteChartProps {
  silhouette: Silhouette;
  description: string;
  caption?: ReactNode;
}

const BAR = 3;
const GROUP_GAP = 10;

export function SilhouetteChart({ silhouette, description, caption }: SilhouetteChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const groups = silhouette.groups.map((group, index) => ({
    ...group,
    color: categoryColor(index),
    sorted: [...group.values].sort((a, b) => b - a),
  }));
  const offsets: number[] = [];
  let total = 0;
  for (const group of groups) {
    offsets.push(total);
    total += group.sorted.length * BAR + GROUP_GAP;
  }
  const low = Math.min(0, ...groups.flatMap((group) => group.sorted));

  return (
    <Chart
      height={total + 40}
      description={description}
      caption={caption}
      margin={{ top: 4, bottom: 36, left: 72, right: 16 }}
      legend={
        <Legend items={groups.map((group) => ({ label: group.label, color: group.color }))} />
      }
      overlay={(plot) => {
        const group = hover === null ? undefined : groups[hover];
        if (hover === null || group === undefined) {
          return null;
        }
        const mean = group.values.reduce((sum, value) => sum + value, 0) / group.values.length;
        const x = scaleLinear().domain([low, 1]).range([0, plot.width]);
        return (
          <ChartTooltip
            x={x(mean)}
            y={(offsets[hover] ?? 0) + (group.sorted.length * BAR) / 2}
            plot={plot}
          >
            <TooltipRow
              value={formatNumber(mean, 2)}
              label="silhueta média"
              color={group.color}
              kind="rect"
            />
            <div className="text-muted-foreground">
              {group.label}, {formatCount(group.values.length)} pontos
            </div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const x = scaleLinear().domain([low, 1]).range([0, plot.width]);
        const ticks = x.ticks(5);
        return (
          <>
            <AxisBottom
              ticks={ticks}
              position={x}
              format={(value) => formatNumber(value, 1)}
              plot={plot}
              label="Silhueta de cada ponto"
            />
            <line x1={x(0)} x2={x(0)} y1={0} y2={plot.height} stroke="var(--input)" />
            {groups.map((group, index) => (
              <g
                key={group.label}
                onPointerEnter={() => {
                  setHover(index);
                }}
                onPointerLeave={() => {
                  setHover(null);
                }}
              >
                <rect
                  x={-plot.margin.left}
                  y={offsets[index] ?? 0}
                  width={plot.width + plot.margin.left}
                  height={group.sorted.length * BAR}
                  fill={hover === index ? "var(--accent)" : "transparent"}
                  opacity={0.5}
                />
                <text
                  x={-10}
                  y={(offsets[index] ?? 0) + (group.sorted.length * BAR) / 2}
                  dy="0.32em"
                  textAnchor="end"
                  className="fill-foreground text-xs"
                >
                  {group.label}
                </text>
                {group.sorted.map((value, position) => (
                  <rect
                    key={position}
                    x={Math.min(x(0), x(value))}
                    y={(offsets[index] ?? 0) + position * BAR}
                    width={Math.abs(x(value) - x(0))}
                    height={BAR - 0.5}
                    fill={group.color}
                  />
                ))}
              </g>
            ))}
            <line
              x1={x(silhouette.average)}
              x2={x(silhouette.average)}
              y1={0}
              y2={plot.height}
              stroke="var(--foreground)"
              strokeDasharray="4 3"
            />
            <text
              x={x(silhouette.average) + 4}
              y={10}
              className="fill-foreground text-[11px] tabular-nums"
            >
              média {formatNumber(silhouette.average, 2)}
            </text>
          </>
        );
      }}
    </Chart>
  );
}
