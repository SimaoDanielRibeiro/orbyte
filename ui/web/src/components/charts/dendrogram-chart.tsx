/**
 * Um dendrograma: cada "U" é a junção de dois grupos, à altura da distância a que se deu. A
 * linha tracejada é o corte que dá os grupos escolhidos.
 */
import { scaleLinear } from "d3";
import type { ReactNode } from "react";

import type { Dendrogram } from "@/api/generated";
import { formatStatistic } from "@/lib/format";

import { AxisLeft, Chart } from "./chart";

interface DendrogramChartProps {
  dendrogram: Dendrogram;
  description: string;
  caption?: ReactNode;
  height?: number;
}

/** Com mais folhas do que isto, os nomes por baixo não cabem: ficam de fora. */
const MAX_LABELS = 30;

export function DendrogramChart({
  dendrogram,
  description,
  caption,
  height = 300,
}: DendrogramChartProps) {
  const points = dendrogram.segments.flat();
  const xs = [...points.map(([x]) => x), ...dendrogram.leaf_positions];
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMax = Math.max(...points.map(([, y]) => y), dendrogram.cut_height ?? 0);
  const labelled = dendrogram.leaf_labels.length <= MAX_LABELS;

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ top: 12, bottom: labelled ? 40 : 12, left: 56, right: 16 }}
    >
      {(plot) => {
        const x = scaleLinear()
          .domain([xMin, xMax === xMin ? xMin + 1 : xMax])
          .range([8, plot.width - 8]);
        const y = scaleLinear().domain([0, yMax]).nice().range([plot.height, 0]);
        return (
          <>
            <AxisLeft
              ticks={y.ticks(4)}
              position={y}
              format={formatStatistic}
              plot={plot}
              label="Distância"
            />
            {dendrogram.segments.map((segment, index) => (
              <path
                key={index}
                d={`M${segment.map(([px, py]) => `${String(x(px))},${String(y(py))}`).join("L")}`}
                fill="none"
                stroke="var(--foreground)"
                strokeOpacity={0.75}
                strokeWidth={1.5}
              />
            ))}
            {dendrogram.cut_height !== null && (
              <g>
                <line
                  x1={0}
                  x2={plot.width}
                  y1={y(dendrogram.cut_height)}
                  y2={y(dendrogram.cut_height)}
                  stroke="var(--muted-foreground)"
                  strokeDasharray="5 4"
                />
                <text
                  x={plot.width}
                  y={y(dendrogram.cut_height) - 6}
                  textAnchor="end"
                  className="fill-foreground text-[12px]"
                >
                  o corte
                </text>
              </g>
            )}
            {labelled &&
              dendrogram.leaf_positions.map((position, index) => (
                <text
                  key={`${String(position)}-${String(index)}`}
                  transform={`translate(${String(x(position))},${String(plot.height + 10)}) rotate(45)`}
                  className="fill-muted-foreground text-[11px]"
                >
                  {dendrogram.leaf_labels[index]}
                </text>
              ))}
          </>
        );
      }}
    </Chart>
  );
}
