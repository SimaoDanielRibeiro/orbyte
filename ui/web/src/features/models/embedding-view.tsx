/**
 * O mapa de uma redução de dimensionalidade: os pontos nas duas coordenadas novas, pintados
 * com a coluna de referência (se houver). Com mais de três classes, a forma também muda.
 */
import type { ReactNode } from "react";

import type { Embedding } from "@/api/generated";
import { Scatter } from "@/components/charts/scatter";
import { shapeOf } from "@/components/charts/shapes";
import { classColors, classOrder } from "@/lib/classes";
import { CATEGORICAL, COLOR_ONLY_LIMIT, sequential } from "@/lib/colors";

interface EmbeddingViewProps {
  embedding: Embedding;
  description: string;
  caption?: ReactNode;
  height?: number;
}

export function EmbeddingView({
  embedding,
  description,
  caption,
  height = 360,
}: EmbeddingViewProps) {
  const { points, axis_labels: axis } = embedding;
  const classes = classOrder([...new Set(points.labels)]);
  const color = classColors(classes);
  // Só três cores se distinguem com segurança numa dispersão: com mais classes, a forma também.
  const shaped = classes.length > COLOR_ONLY_LIMIT;
  const shape = (name: string | undefined) =>
    shaped && name !== undefined ? shapeOf(classes.indexOf(name)) : undefined;
  const low = Math.min(...points.values);
  const high = Math.max(...points.values);
  return (
    <Scatter
      height={height}
      xLabel={axis[0]}
      yLabel={axis[1]}
      points={points.x.map((x, index) => {
        const name = points.labels[index];
        const value = points.values[index];
        return {
          x,
          y: points.y[index] ?? 0,
          color:
            name !== undefined
              ? color(name)
              : value !== undefined
                ? sequential(high > low ? (value - low) / (high - low) : 0.5)
                : CATEGORICAL[0],
          shape: shape(name),
          label: name,
        };
      })}
      legend={classes.map((name) => ({
        label: name,
        color: color(name),
        kind: "dot" as const,
        shape: shape(name),
      }))}
      description={description}
      caption={caption}
    />
  );
}
