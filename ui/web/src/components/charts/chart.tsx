/**
 * A moldura de todos os gráficos: mede a largura que tem, dá a área do desenho e junta a
 * legenda, a dica e a legenda de leitura (a `figcaption`).
 *
 * As especificações vêm do skill de visualização de dados: grelha e eixos numa linha fina e
 * discreta, texto sempre nas cores do texto (nunca na cor dos dados), uma dica ao passar o rato
 * em que o valor vem primeiro.
 */
import { cn } from "cn";
import type { ReactNode } from "react";

import { useSize } from "@/hooks/use-size";

import { SHAPE_PATHS, type PointShape } from "./shapes";

export interface Margin {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/** A área onde se desenha (sem as margens dos eixos). */
export interface Plot {
  width: number;
  height: number;
  margin: Margin;
}

const DEFAULT_MARGIN: Margin = { top: 12, right: 16, bottom: 36, left: 48 };

interface ChartProps {
  /** A altura total, com as margens. */
  height: number;
  margin?: Partial<Margin>;
  /** O que o gráfico mostra, numa frase, para quem usa um leitor de ecrã. */
  description: string;
  /** Como se lê o gráfico (mostra-se por baixo dele). */
  caption?: ReactNode;
  legend?: ReactNode;
  /** O desenho em SVG, dentro da área. */
  children: (plot: Plot) => ReactNode;
  /** O que fica por cima do SVG, em HTML (a dica). */
  overlay?: (plot: Plot) => ReactNode;
  /** O que fica por baixo do SVG (um canvas com o campo de decisão). */
  underlay?: (plot: Plot & { outerWidth: number }) => ReactNode;
  className?: string;
}

export function Chart({
  height,
  margin,
  description,
  caption,
  legend,
  children,
  overlay,
  underlay,
  className,
}: ChartProps) {
  const { ref, width } = useSize<HTMLDivElement>();
  const full = { ...DEFAULT_MARGIN, ...margin };
  const plot: Plot = {
    width: Math.max(0, width - full.left - full.right),
    height: Math.max(0, height - full.top - full.bottom),
    margin: full,
  };
  return (
    <figure className={cn("flex min-w-0 flex-col gap-2", className)}>
      {legend}
      <div ref={ref} className="relative w-full" style={{ height }}>
        {width > 0 && (
          <>
            {underlay?.({ ...plot, outerWidth: width })}
            <svg
              width={width}
              height={height}
              role="img"
              aria-label={description}
              className="absolute inset-0 block overflow-visible"
            >
              <g transform={`translate(${String(full.left)},${String(full.top)})`}>
                {children(plot)}
              </g>
            </svg>
            {overlay?.(plot)}
          </>
        )}
      </div>
      {caption !== undefined && (
        <figcaption className="max-w-[68ch] font-serif text-sm leading-relaxed text-muted-foreground">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

interface AxisProps {
  ticks: readonly number[];
  position: (value: number) => number;
  format: (value: number) => string;
  plot: Plot;
  label?: string;
}

/** O eixo de baixo: a linha de base, as marcas e o nome. */
export function AxisBottom({ ticks, position, format, plot, label }: AxisProps) {
  return (
    <g aria-hidden="true" className="text-[11px]">
      <line x1={0} x2={plot.width} y1={plot.height} y2={plot.height} stroke="var(--input)" />
      {ticks.map((tick) => (
        <text
          key={tick}
          x={position(tick)}
          y={plot.height + 16}
          textAnchor="middle"
          className="fill-muted-foreground tabular-nums"
        >
          {format(tick)}
        </text>
      ))}
      {label !== undefined && (
        <text
          x={plot.width / 2}
          y={plot.height + 32}
          textAnchor="middle"
          className="fill-muted-foreground"
        >
          {label}
        </text>
      )}
    </g>
  );
}

/** O eixo da esquerda, com a grelha horizontal: uma linha fina por marca. */
export function AxisLeft({ ticks, position, format, plot, label }: AxisProps) {
  return (
    <g aria-hidden="true" className="text-[11px]">
      {ticks.map((tick) => (
        <g key={tick}>
          <line
            x1={0}
            x2={plot.width}
            y1={position(tick)}
            y2={position(tick)}
            stroke="var(--border)"
          />
          <text
            x={-8}
            y={position(tick)}
            dy="0.32em"
            textAnchor="end"
            className="fill-muted-foreground tabular-nums"
          >
            {format(tick)}
          </text>
        </g>
      ))}
      {label !== undefined && (
        <text
          transform={`translate(${String(-plot.margin.left + 12)},${String(plot.height / 2)}) rotate(-90)`}
          textAnchor="middle"
          className="fill-muted-foreground"
        >
          {label}
        </text>
      )}
    </g>
  );
}

export interface LegendItem {
  label: string;
  color: string;
  /** A forma da chave: a da marca (linha nas curvas, retângulo nas barras, ponto nos pontos). */
  kind?: "line" | "rect" | "dot";
  /** A forma do ponto, quando é ela que distingue (a parte da divisão, numa projeção). */
  shape?: PointShape;
}

/** A legenda: sempre que há duas séries ou mais. O texto fica nas cores do texto. */
export function Legend({ items }: { items: readonly LegendItem[] }) {
  if (items.length < 2) {
    return null;
  }
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {items.map((item) => (
        <li key={item.label} className="flex items-center gap-1.5">
          {item.shape === undefined ? (
            <Swatch color={item.color} kind={item.kind ?? "rect"} />
          ) : (
            <ShapeSwatch color={item.color} shape={item.shape} />
          )}
          {item.label}
        </li>
      ))}
    </ul>
  );
}

/** A chave com a forma de um ponto. */
function ShapeSwatch({ color, shape }: { color: string; shape: PointShape }) {
  return (
    <svg aria-hidden="true" width={12} height={12} viewBox="-6 -6 12 12" className="shrink-0">
      <path d={SHAPE_PATHS[shape]} fill={color} />
    </svg>
  );
}

export function Swatch({ color, kind }: { color: string; kind: "line" | "rect" | "dot" }) {
  if (kind === "line") {
    return (
      <span aria-hidden="true" className="h-0.5 w-3 rounded-full" style={{ background: color }} />
    );
  }
  if (kind === "dot") {
    return (
      <span aria-hidden="true" className="size-2 rounded-full" style={{ background: color }} />
    );
  }
  return (
    <span aria-hidden="true" className="size-2.5 rounded-[2px]" style={{ background: color }} />
  );
}

interface TooltipProps {
  /** A posição do ponto, dentro da área do desenho. */
  x: number;
  y: number;
  plot: Plot;
  children: ReactNode;
}

/** A dica: perto do ponto, do lado onde cabe. */
export function ChartTooltip({ x, y, plot, children }: TooltipProps) {
  const left = x + plot.margin.left;
  const top = y + plot.margin.top;
  const flip = x > plot.width * 0.6;
  return (
    <div
      role="status"
      className="pointer-events-none absolute z-10 min-w-28 rounded-md border bg-popover px-2.5 py-1.5 text-xs text-popover-foreground"
      style={{
        left,
        top,
        transform: `translate(${flip ? "calc(-100% - 12px)" : "12px"}, -50%)`,
      }}
    >
      {children}
    </div>
  );
}

/** Uma linha da dica: o valor primeiro (forte), o nome a seguir. */
export function TooltipRow({
  value,
  label,
  color,
  kind = "line",
}: {
  value: string;
  label: string;
  color?: string;
  kind?: "line" | "rect" | "dot";
}) {
  return (
    <div className="flex items-center gap-2 whitespace-nowrap">
      {color !== undefined && <Swatch color={color} kind={kind} />}
      <span className="font-medium tabular-nums">{value}</span>
      <span className="text-muted-foreground">{label}</span>
    </div>
  );
}
