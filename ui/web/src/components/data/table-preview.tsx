/**
 * Uma tabela de dados tal como o backend a manda (`TablePreview`): as primeiras linhas de um
 * dataset, o antes e o depois da preparação.
 *
 * Os números escrevem-se à portuguesa e alinham à direita, com algarismos da mesma largura, para
 * se compararem de cima a baixo; um valor em falta é um traço discreto. A tabela rola na
 * horizontal dentro do seu espaço, e o cabeçalho fica à vista ao rolar na vertical.
 */
import { cn } from "cn";

import type { TablePreview } from "@/api/generated";
import { formatCount, formatQuantity, formatValue } from "@/lib/format";

interface TablePreviewViewProps {
  preview: TablePreview;
  /** O que a tabela mostra, para os leitores de ecrã. */
  label: string;
  /** Mostrar o tipo do pandas de cada coluna por baixo do nome. */
  showTypes?: boolean;
  className?: string;
}

export function TablePreviewView({
  preview,
  label,
  showTypes = true,
  className,
}: TablePreviewViewProps) {
  const numeric = preview.dtypes.map(isNumericType);
  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div
        className="max-h-[28rem] overflow-auto rounded-xl border"
        role="region"
        aria-label={label}
        tabIndex={0}
      >
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-card">
            <tr className="border-b">
              {preview.columns.map((column, index) => (
                <th
                  key={column}
                  scope="col"
                  className={cn(
                    "px-4 py-2.5 align-bottom font-mono text-xs font-normal whitespace-nowrap text-dim",
                    numeric[index] === true ? "text-right" : "text-left",
                  )}
                >
                  {column}
                  {showTypes && (
                    <span className="block text-[0.6875rem] text-faint">
                      {preview.dtypes[index]}
                    </span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="border-b border-border/60 transition-colors last:border-0 hover:bg-surface-high/50"
              >
                {row.map((value, index) => (
                  <td
                    key={preview.columns[index] ?? index}
                    className={cn(
                      "max-w-[22rem] truncate px-4 py-2.5 whitespace-nowrap",
                      numeric[index] === true && "text-right tabular-nums",
                    )}
                    title={typeof value === "string" && value.length > 40 ? value : undefined}
                  >
                    <Cell value={value} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mono-label">
        {preview.rows.length < preview.n_rows
          ? `As primeiras ${formatCount(preview.rows.length)} de ${formatQuantity(preview.n_rows, "linha", "linhas")}.`
          : formatQuantity(preview.n_rows, "linha", "linhas")}
      </p>
    </div>
  );
}

function Cell({ value }: { value: unknown }) {
  if (value === null || value === undefined || (typeof value === "number" && Number.isNaN(value))) {
    return (
      <span className="text-muted-foreground/60" aria-label="em falta">
        —
      </span>
    );
  }
  return <>{formatValue(value, 4)}</>;
}

/** Os tipos do pandas que guardam números (alinham à direita). */
function isNumericType(dtype: string): boolean {
  return /^(u?int|float|Int|UInt|Float)/.test(dtype);
}
