/**
 * Uma tabela de números em cores: as correlações, a matriz de confusão, o mapa dos nulos.
 *
 * Cada célula é a sua própria área de dica. O número dentro da célula (quando cabe) fica a
 * branco ou a escuro, conforme a cor da célula, para se ler sempre.
 */
import { scaleBand } from "d3";
import { useState, type ReactNode } from "react";

import { inkOn } from "@/lib/colors";

import { Chart, ChartTooltip, TooltipRow } from "./chart";

interface HeatmapProps {
  rows: readonly string[];
  columns: readonly string[];
  /** `values[i][j]` é a célula da linha `i` e da coluna `j` (`null`: sem valor). */
  values: readonly (readonly (number | null)[])[];
  /** A cor de uma célula (com a linha e a coluna, para escalas diferentes por coluna). */
  color: (value: number, row: number, column: number) => string;
  format: (value: number) => string;
  description: string;
  caption?: ReactNode;
  /** Escrever o número dentro de cada célula. */
  showValues?: boolean;
  /** O que as linhas e as colunas são (na dica e nos eixos). */
  rowTitle?: string;
  columnTitle?: string;
  cellHeight?: number;
  labelWidth?: number;
  /** Não escrever os nomes das colunas (muitas colunas estreitas, como no mapa dos nulos). */
  hideColumnLabels?: boolean;
  /** Destacar a diagonal (os acertos de uma matriz de confusão). */
  highlightDiagonal?: boolean;
}

export function Heatmap({
  rows,
  columns,
  values,
  color,
  format,
  description,
  caption,
  showValues = true,
  rowTitle,
  columnTitle,
  cellHeight = 34,
  labelWidth = 120,
  hideColumnLabels = false,
  highlightDiagonal = false,
}: HeatmapProps) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const top = hideColumnLabels ? 8 : 64;
  const height = rows.length * cellHeight + top + (columnTitle === undefined ? 8 : 28);

  return (
    <Chart
      height={height}
      description={description}
      caption={caption}
      margin={{ top, left: labelWidth, bottom: columnTitle === undefined ? 8 : 28, right: 8 }}
      overlay={(plot) => {
        if (hover === null) {
          return null;
        }
        const [row, column] = hover;
        const value = values[row]?.[column] ?? null;
        const x = scaleBand<number>()
          .domain(columns.map((_, index) => index))
          .range([0, plot.width]);
        const y = scaleBand<number>()
          .domain(rows.map((_, index) => index))
          .range([0, plot.height]);
        return (
          <ChartTooltip
            x={(x(column) ?? 0) + x.bandwidth()}
            y={(y(row) ?? 0) + y.bandwidth() / 2}
            plot={plot}
          >
            <TooltipRow value={value === null ? "—" : format(value)} label="" />
            <div className="text-muted-foreground">
              {rowTitle === undefined ? "" : `${rowTitle}: `}
              {rows[row]}
            </div>
            <div className="text-muted-foreground">
              {columnTitle === undefined ? "" : `${columnTitle}: `}
              {columns[column]}
            </div>
          </ChartTooltip>
        );
      }}
    >
      {(plot) => {
        const x = scaleBand<number>()
          .domain(columns.map((_, index) => index))
          .range([0, plot.width]);
        const y = scaleBand<number>()
          .domain(rows.map((_, index) => index))
          .range([0, plot.height]);
        const fits = x.bandwidth() >= 36;
        return (
          <>
            {!hideColumnLabels &&
              columns.map((name, index) => (
                <text
                  key={name}
                  transform={`translate(${String((x(index) ?? 0) + x.bandwidth() / 2)},-8) rotate(-40)`}
                  className="fill-muted-foreground text-[12px]"
                >
                  {name.length > 14 ? `${name.slice(0, 13)}…` : name}
                </text>
              ))}
            {rows.map((name, index) => (
              <text
                key={name}
                x={-8}
                y={(y(index) ?? 0) + y.bandwidth() / 2}
                dy="0.32em"
                textAnchor="end"
                className="fill-foreground text-xs"
              >
                {name.length > 18 ? `${name.slice(0, 17)}…` : name}
              </text>
            ))}
            {values.map((row, rowIndex) =>
              row.map((value, columnIndex) => {
                const fill = value === null ? "var(--muted)" : color(value, rowIndex, columnIndex);
                const isHover = hover?.[0] === rowIndex && hover[1] === columnIndex;
                return (
                  <g
                    key={`${String(rowIndex)}-${String(columnIndex)}`}
                    onPointerEnter={() => {
                      setHover([rowIndex, columnIndex]);
                    }}
                    onPointerLeave={() => {
                      setHover(null);
                    }}
                  >
                    <rect
                      x={(x(columnIndex) ?? 0) + 1}
                      y={(y(rowIndex) ?? 0) + 1}
                      width={Math.max(0, x.bandwidth() - 2)}
                      height={Math.max(0, y.bandwidth() - 2)}
                      rx={2}
                      fill={fill}
                      stroke={
                        isHover || (highlightDiagonal && rowIndex === columnIndex)
                          ? "var(--foreground)"
                          : "none"
                      }
                      strokeWidth={isHover ? 1.5 : 1}
                      strokeOpacity={isHover ? 1 : 0.5}
                    />
                    {showValues && fits && value !== null && (
                      <text
                        x={(x(columnIndex) ?? 0) + x.bandwidth() / 2}
                        y={(y(rowIndex) ?? 0) + y.bandwidth() / 2}
                        dy="0.32em"
                        textAnchor="middle"
                        className="text-xs tabular-nums"
                        fill={inkOn(fill)}
                      >
                        {format(value)}
                      </text>
                    )}
                  </g>
                );
              }),
            )}
            {columnTitle !== undefined && (
              <text
                x={plot.width / 2}
                y={plot.height + 20}
                textAnchor="middle"
                className="fill-muted-foreground text-[12px]"
              >
                {columnTitle}
              </text>
            )}
          </>
        );
      }}
    </Chart>
  );
}
