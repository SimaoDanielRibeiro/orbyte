/**
 * Como se mostram as partes de texto de um pacote: os JSON numa árvore que abre e fecha, e o
 * CSV numa tabela.
 *
 * Os documentos podem ser grandes (o treino passo a passo tem milhares de números): cada ramo
 * só desenha o que tem dentro quando se abre, e uma lista longa mostra os primeiros elementos.
 */
import { cn } from "cn";
import { useState } from "react";

import { TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatCount } from "@/lib/format";

import { parseCsv } from "./csv";

/** Quantos elementos de uma lista ou de um objeto se mostram, no máximo. */
const SHOWN = 100;

/** Até que profundidade a árvore abre sozinha (a raiz e os seus filhos). */
const OPEN_DEPTH = 1;

/** Um JSON numa árvore; se não for JSON válido (por exemplo, cortado), o texto tal como está. */
export function JsonTree({ text }: { text: string }) {
  const parsed = parseJson(text);
  if (!parsed.ok) {
    return <pre className="font-mono text-xs whitespace-pre-wrap text-soft">{text}</pre>;
  }
  return (
    <div className="font-mono text-xs leading-relaxed">
      <JsonNode name={null} value={parsed.value} depth={0} />
    </div>
  );
}

function parseJson(text: string): { ok: true; value: unknown } | { ok: false } {
  try {
    return { ok: true, value: JSON.parse(text) as unknown };
  } catch {
    return { ok: false };
  }
}

function JsonNode({ name, value, depth }: { name: string | null; value: unknown; depth: number }) {
  const [open, setOpen] = useState(depth <= OPEN_DEPTH);
  if (value === null || typeof value !== "object") {
    return (
      <div className="flex gap-1.5">
        {name !== null && <span className="text-dim">{name}:</span>}
        <Primitive value={value} />
      </div>
    );
  }
  const entries: [string, unknown][] = Array.isArray(value)
    ? value.map((item, index) => [String(index), item])
    : Object.entries(value);
  const brackets = Array.isArray(value) ? ["[", "]"] : ["{", "}"];
  const hint = Array.isArray(value) ? null : nameOf(value);
  if (entries.length === 0) {
    return (
      <div className="flex gap-1.5">
        {name !== null && <span className="text-dim">{name}:</span>}
        <span className="text-soft">{brackets.join("")}</span>
      </div>
    );
  }
  return (
    <details
      open={open}
      onToggle={(event) => {
        setOpen(event.currentTarget.open);
      }}
    >
      <summary className="cursor-pointer text-soft outline-none select-none hover:text-strong focus-visible:ring-2 focus-visible:ring-ring">
        {name !== null && <span className="text-dim">{name}: </span>}
        <span>
          {brackets[0]}
          {open ? "" : ` ${formatCount(entries.length)} `}
          {open ? "" : brackets[1]}
        </span>
        {/* Fechado, um elemento de uma lista diz o que é (a coluna, o passo da preparação). */}
        {!open && hint !== null && <span className="ml-2 text-foreground">{hint}</span>}
      </summary>
      {/* Só se desenha o que está dentro quando se abre: os documentos grandes ficam leves. */}
      {open && (
        <div className="ml-1.5 border-l border-[#2a2a2a] pl-3">
          {entries.slice(0, SHOWN).map(([key, item]) => (
            <JsonNode key={key} name={key} value={item} depth={depth + 1} />
          ))}
          {entries.length > SHOWN && (
            <p className="text-dim">… and {formatCount(entries.length - SHOWN)} more</p>
          )}
          <span className="text-soft">{brackets[1]}</span>
        </div>
      )}
    </details>
  );
}

/** As chaves que dizem o que é um objeto: o nome de uma coluna, a operação de um passo. */
const NAME_KEYS = ["name", "operation", "path", "title"] as const;

/** O nome de um objeto, para o mostrar quando está fechado (ou `null`, se não tiver). */
function nameOf(value: object): string | null {
  for (const key of NAME_KEYS) {
    const found: unknown = (value as Record<string, unknown>)[key];
    if (typeof found === "string" && found !== "") {
      return found;
    }
  }
  return null;
}

function Primitive({ value }: { value: unknown }) {
  if (typeof value === "string") {
    return <span className="break-all text-foreground">&quot;{value}&quot;</span>;
  }
  return (
    <span className={cn("tabular-nums", value === null ? "text-dim" : "text-strong")}>
      {String(value)}
    </span>
  );
}

/** Um CSV (com cabeçalho) numa tabela. */
export function CsvTable({ text, label }: { text: string; label: string }) {
  const [header = [], ...rows] = parseCsv(text);
  // Um só contentor desliza (o `Table` traria outro, por dentro), e recebe o foco: com muitas
  // colunas, as setas do teclado deslizam-no para o lado.
  return (
    <div
      className="overflow-x-auto rounded-lg border outline-none focus-visible:ring-2 focus-visible:ring-ring"
      role="region"
      aria-label={`${label} (scrolls sideways)`}
      tabIndex={0}
    >
      <table aria-label={label} className="w-full caption-bottom text-sm">
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            {header.map((name, index) => (
              <TableHead key={`${name}-${String(index)}`} className="px-3 font-mono text-xs">
                {name}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, index) => (
            <TableRow key={index}>
              {row.map((cell, column) => (
                <TableCell key={column} className="px-3 font-mono text-xs tabular-nums">
                  {cell}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </table>
    </div>
  );
}
