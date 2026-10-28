/**
 * Um ambiente em grelha, visto de cima: o que o agente sabe de cada célula.
 *
 * A cor de cada célula é um valor de −1 (espera perder) a 1 (espera ganhar), na escala
 * divergente (laranja, cinzento neutro, azul), sempre a mesma, para os episódios se poderem
 * comparar. As células onde não há nada a escolher (as paredes, o objetivo e as armadilhas) têm
 * um desenho e um símbolo próprios, e o nome na dica: não dependem só da cor. Por cima, as setas
 * da política e, a branco, o caminho a partir da partida.
 */
import { cn } from "cn";
import { useState, type ReactNode } from "react";

import type { CellKind, GridLayout } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { useSize } from "@/hooks/use-size";
import { CATEGORICAL, diverging, inkOn, LIGHT_INK, SURFACE } from "@/lib/colors";
import { formatNumber } from "@/lib/format";

import { ChartTooltip, TooltipRow } from "./chart";

const WALL = "#3a3a3a";
const EMPTY_KIND = "#1e1e1e";
const TRAP = CATEGORICAL[7];

interface GridWorldProps {
  layout: GridLayout;
  /** Um valor por célula, linha a linha (`null` onde não há escolha). */
  values: readonly (number | null)[];
  /** A ação que a política escolhe em cada célula (as setas). */
  actions?: readonly (number | null)[];
  /** As células do caminho a partir da partida. */
  path?: readonly number[];
  description: string;
  caption?: ReactNode;
  /** O que o valor é (na dica): "Valor", "Q de cima"… */
  valueLabel?: string;
  /** O lado máximo de uma célula, em px. */
  maxCell?: number;
  /** A legenda da escala e das células especiais. */
  legend?: boolean;
  className?: string;
}

export function GridWorld({
  layout,
  values,
  actions,
  path = [],
  description,
  caption,
  valueLabel = "Valor",
  maxCell = 64,
  legend = true,
  className,
}: GridWorldProps) {
  const label = useLabel();
  const { ref, width } = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const cell = Math.max(16, Math.min(maxCell, Math.floor(width / layout.columns)));
  const gridWidth = cell * layout.columns;
  const gridHeight = cell * layout.rows;
  const center = (state: number): [number, number] => [
    (state % layout.columns) * cell + cell / 2,
    Math.floor(state / layout.columns) * cell + cell / 2,
  ];
  const walk = path.map((state) => center(state).join(",")).join("L");
  const plot = {
    width: gridWidth,
    height: gridHeight,
    margin: { top: 0, right: 0, bottom: 0, left: 0 },
  };

  const hasCaption = caption !== undefined && caption !== null && caption !== false;

  return (
    <figure className={cn("flex min-w-0 flex-col gap-3", className)}>
      {(legend || hasCaption) && (
        <div className="flex min-h-6 items-start justify-between gap-3">
          <div className="min-w-0">{legend && <GridLegend />}</div>
          {hasCaption && (
            <Explain title="Como se lê" label="Como se lê este gráfico" align="end">
              {caption}
            </Explain>
          )}
        </div>
      )}
      <div ref={ref} className="relative w-full" style={{ height: width > 0 ? gridHeight : 0 }}>
        {width > 0 && (
          <svg
            width={gridWidth}
            height={gridHeight}
            role="img"
            aria-label={description}
            className="absolute top-0 left-0 block"
          >
            {layout.cells.map((kind, state) => {
              const value = values[state] ?? null;
              const [cx, cy] = center(state);
              const x = cx - cell / 2;
              const y = cy - cell / 2;
              const fill = cellFill(kind, value);
              const action = actions?.[state] ?? null;
              const move = action === null ? undefined : layout.action_moves[action];
              return (
                <g
                  key={state}
                  onPointerEnter={() => {
                    setHover(state);
                  }}
                  onPointerLeave={() => {
                    setHover(null);
                  }}
                >
                  <rect
                    x={x + 1}
                    y={y + 1}
                    width={cell - 2}
                    height={cell - 2}
                    rx={3}
                    fill={fill}
                    stroke={hover === state || kind === "goal" ? LIGHT_INK : "none"}
                    strokeOpacity={hover === state ? 1 : 0.7}
                    strokeWidth={hover === state ? 1.5 : 1}
                  />
                  <CellSymbol kind={kind} cx={cx} cy={cy} size={cell} />
                  {move !== undefined && (
                    <Arrow cx={cx} cy={cy} size={cell} move={move} ink={inkOn(fill)} />
                  )}
                </g>
              );
            })}
            {path.length > 1 && (
              <g pointerEvents="none">
                <path
                  d={`M${walk}`}
                  fill="none"
                  stroke={SURFACE}
                  strokeWidth={5}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                <path
                  d={`M${walk}`}
                  fill="none"
                  stroke={LIGHT_INK}
                  strokeWidth={2.5}
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              </g>
            )}
          </svg>
        )}
        {hover !== null && width > 0 && (
          <ChartTooltip x={center(hover)[0]} y={center(hover)[1]} plot={plot}>
            <CellTip
              layout={layout}
              state={hover}
              value={values[hover] ?? null}
              action={actions?.[hover] ?? null}
              valueLabel={valueLabel}
              kindLabel={label("CellKind", layout.cells[hover] ?? "empty")}
            />
          </ChartTooltip>
        )}
      </div>
    </figure>
  );
}

/** A cor de uma célula: o valor, ou o desenho próprio das células sem escolha. */
function cellFill(kind: CellKind, value: number | null): string {
  if (kind === "wall") {
    return WALL;
  }
  if (value === null) {
    return EMPTY_KIND;
  }
  return diverging(value);
}

/** O símbolo das células especiais: a bandeira do objetivo, a cruz da armadilha, a partida. */
function CellSymbol({
  kind,
  cx,
  cy,
  size,
}: {
  kind: CellKind;
  cx: number;
  cy: number;
  size: number;
}) {
  const s = size * 0.2;
  if (kind === "goal") {
    return (
      <g aria-hidden="true" stroke={LIGHT_INK} strokeWidth={1.75} strokeLinecap="round" fill="none">
        <path d={`M${String(cx - s * 0.6)},${String(cy + s)}V${String(cy - s)}`} />
        <path
          d={`M${String(cx - s * 0.6)},${String(cy - s)}L${String(cx + s)},${String(cy - s * 0.45)}L${String(cx - s * 0.6)},${String(cy + s * 0.1)}`}
          fill={LIGHT_INK}
          fillOpacity={0.85}
        />
      </g>
    );
  }
  if (kind === "trap") {
    return (
      <path
        aria-hidden="true"
        d={`M${String(cx - s)},${String(cy - s)}L${String(cx + s)},${String(cy + s)}M${String(cx - s)},${String(cy + s)}L${String(cx + s)},${String(cy - s)}`}
        stroke={TRAP}
        strokeWidth={2.5}
        strokeLinecap="round"
      />
    );
  }
  if (kind === "start") {
    return (
      <circle
        aria-hidden="true"
        cx={cx}
        cy={cy}
        r={size * 0.3}
        fill="none"
        stroke={LIGHT_INK}
        strokeWidth={1.25}
        strokeDasharray="3 3"
      />
    );
  }
  return null;
}

/** A seta da ação que a política escolhe numa célula. */
function Arrow({
  cx,
  cy,
  size,
  move,
  ink,
}: {
  cx: number;
  cy: number;
  size: number;
  /** Para onde a ação leva, em (linhas, colunas). */
  move: readonly number[];
  ink: string;
}) {
  const [rows = 0, columns = 0] = move;
  const length = size * 0.26;
  const tipX = cx + columns * length;
  const tipY = cy + rows * length;
  const tailX = cx - columns * length;
  const tailY = cy - rows * length;
  // A ponta: um triângulo pequeno, perpendicular à direção.
  const head = size * 0.1;
  const left: [number, number] = [
    tipX - columns * head - rows * head,
    tipY - rows * head + columns * head,
  ];
  const right: [number, number] = [
    tipX - columns * head + rows * head,
    tipY - rows * head - columns * head,
  ];
  return (
    <g aria-hidden="true" pointerEvents="none" stroke={ink} fill={ink} strokeLinecap="round">
      <line x1={tailX} y1={tailY} x2={tipX} y2={tipY} strokeWidth={1.5} strokeOpacity={0.8} />
      <path
        d={`M${String(tipX)},${String(tipY)}L${left.join(",")}L${right.join(",")}Z`}
        strokeWidth={1}
        fillOpacity={0.9}
      />
    </g>
  );
}

function CellTip({
  layout,
  state,
  value,
  action,
  valueLabel,
  kindLabel,
}: {
  layout: GridLayout;
  state: number;
  value: number | null;
  action: number | null;
  valueLabel: string;
  kindLabel: string;
}) {
  const kind = layout.cells[state] ?? "empty";
  return (
    <>
      {value !== null && <TooltipRow value={formatNumber(value, 3)} label={valueLabel} />}
      {action !== null && (
        <div className="text-muted-foreground">Escolhe {layout.action_labels[action]}</div>
      )}
      <div className="text-muted-foreground">
        {kind === "empty" ? "" : `${kindLabel} · `}linha {Math.floor(state / layout.columns) + 1},
        coluna {(state % layout.columns) + 1}
      </div>
    </>
  );
}

/** A escala do valor e as células especiais. */
function GridLegend() {
  const label = useLabel();
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <li className="flex items-center gap-1.5">
        <span className="tabular-nums">−1</span>
        <span
          aria-hidden="true"
          className="h-2 w-20 rounded-[2px]"
          style={{
            background: `linear-gradient(to right, ${diverging(-1)}, ${diverging(0)}, ${diverging(1)})`,
          }}
        />
        <span className="tabular-nums">1</span>
        <span>valor</span>
      </li>
      {(["goal", "trap", "wall"] as const).map((kind) => (
        <li key={kind} className="flex items-center gap-1.5">
          <svg aria-hidden="true" width={14} height={14} viewBox="0 0 28 28" className="shrink-0">
            <rect x={1} y={1} width={26} height={26} rx={3} fill={cellFill(kind, null)} />
            <CellSymbol kind={kind} cx={14} cy={14} size={28} />
          </svg>
          {label("CellKind", kind)}
        </li>
      ))}
      <li className="flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className="h-0.5 w-3 rounded-full"
          style={{ background: LIGHT_INK }}
        />
        Caminho a partir da partida
      </li>
    </ul>
  );
}
