/**
 * O leitor de um pacote: as partes numa lista, à esquerda, e o conteúdo da escolhida ao lado.
 *
 * Serve a pré-visualização (o ficheiro antes de existir) e um pacote da biblioteca: os dois
 * chegam do backend no mesmo formato (`PackagePart`). Cada parte mostra-se como é: a ficha
 * desenhada, os JSON numa árvore que abre e fecha, o CSV numa tabela; os binários (o modelo, o
 * ONNX) só dizem o tamanho e o que são. Uma parte que entra ou sai da lista (ao mudar uma
 * opção) anima-se.
 *
 * As duas peças (`PartList` e `PartContent`) também se usam soltas: o ecrã Guardar põe a lista
 * por baixo das opções e o conteúdo numa coluna só dele, à direita.
 */
import { cn } from "cn";
import {
  Binary,
  Braces,
  FileKey2,
  FileText,
  Sheet as SheetIcon,
  type LucideIcon,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useState } from "react";

import type { PackageFileKind, PackagePart } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { Prose } from "@/components/markdown/prose";
import { formatBytes } from "@/lib/format";

import { CsvTable, JsonTree } from "./content-views";

const ICONS: Readonly<Record<PackageFileKind, LucideIcon>> = {
  markdown: FileText,
  json: Braces,
  csv: SheetIcon,
  text: FileKey2,
  binary: Binary,
};

/** O tema do backend que explica cada binário (o modelo é um pickle; o ONNX é um grafo). */
const BINARY_TOPICS: Readonly<Record<string, { topic: string }>> = {
  "model.joblib": { topic: "pickle_safety" },
  "model.onnx": { topic: "onnx_format" },
};

interface PackageViewerProps {
  parts: readonly PackagePart[];
  /** O nome do leitor, para quem usa um leitor de ecrã (há dois no ecrã). */
  label: string;
  className?: string;
}

export function PackageViewer({ parts, label, className }: PackageViewerProps) {
  const [chosen, setChosen] = useState(parts[0]?.path ?? "");
  const current = parts.find((part) => part.path === chosen) ?? parts[0];
  return (
    // O contentor fica num elemento à volta da grelha: uma consulta de contentor não se aplica
    // ao próprio elemento que o define (antes, a lista e o conteúdo ficavam sempre empilhados).
    <div className={cn("@container/viewer", className)}>
      <div className="grid gap-4 @2xl/viewer:grid-cols-[15rem_minmax(0,1fr)]">
        <PartList parts={parts} chosen={current?.path ?? null} onChoose={setChosen} label={label} />
        {current !== undefined && <PartContent part={current} />}
      </div>
    </div>
  );
}

interface PartListProps {
  parts: readonly PackagePart[];
  /** O caminho da parte aberta. */
  chosen: string | null;
  onChoose: (path: string) => void;
  /** O nome da lista, para quem usa um leitor de ecrã (há duas no ecrã). */
  label: string;
}

/** As partes do pacote, uma por linha: o ícone do tipo, o caminho e o tamanho. */
export function PartList({ parts, chosen, onChoose, label }: PartListProps) {
  const reduced = useReducedMotion() ?? false;
  return (
    <ul aria-label={label} className="flex flex-col gap-1">
      <AnimatePresence initial={false}>
        {parts.map((part) => {
          const Icon = ICONS[part.kind];
          const on = part.path === chosen;
          return (
            <motion.li
              key={part.path}
              layout={!reduced}
              initial={reduced ? false : { opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={reduced ? { opacity: 0 } : { opacity: 0, x: -8 }}
              transition={{ duration: 0.2 }}
            >
              <button
                type="button"
                aria-pressed={on}
                onClick={() => {
                  onChoose(part.path);
                }}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-lg border px-3 py-2 text-left transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  on
                    ? "border-[#3a3a3a] bg-surface-high text-strong"
                    : "border-transparent text-soft hover:bg-surface hover:text-strong",
                )}
              >
                <Icon aria-hidden="true" className="size-4 shrink-0 text-dim" />
                <span className="min-w-0 flex-1 truncate font-mono text-sm">{part.path}</span>
                <span className="shrink-0 font-mono text-[0.75rem] text-dim tabular-nums">
                  {formatBytes(part.size)}
                </span>
              </button>
            </motion.li>
          );
        })}
      </AnimatePresence>
    </ul>
  );
}

interface PartContentProps {
  part: PackagePart;
  className?: string;
  /** A altura da área que rola (por omissão, 34rem). */
  scrollClassName?: string;
}

/** O conteúdo de uma parte, como ela é. */
export function PartContent({ part, className, scrollClassName }: PartContentProps) {
  const label = useLabel();
  return (
    <section
      aria-label={`The content of ${part.path}`}
      className={cn("flex min-w-0 flex-col gap-3 rounded-xl border bg-ground p-4", className)}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <div className="flex min-w-0 items-center gap-1">
          <h3 className="text-base font-medium tracking-normal text-strong">{part.title}</h3>
          <Explain title={part.title}>{part.description}</Explain>
        </div>
        <span className="font-mono text-xs text-dim">
          {part.path} · {label("PackageFileKind", part.kind)} · {formatBytes(part.size)}
        </span>
      </div>
      {/* Desliza por dentro: com o foco, as setas do teclado também a deslizam. */}
      <div
        className={cn(
          "max-h-[34rem] min-w-0 overflow-auto rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring",
          scrollClassName,
        )}
        role="region"
        aria-label={`The text of ${part.path}`}
        tabIndex={0}
      >
        <Body part={part} />
      </div>
      {part.truncated && (
        <p className="text-xs text-dim">Only the beginning is shown; the file has the rest.</p>
      )}
    </section>
  );
}

function Body({ part }: { part: PackagePart }) {
  const text = part.content ?? "";
  switch (part.kind) {
    case "markdown":
      return <Prose text={text} gfm />;
    case "json":
      return <JsonTree text={text} />;
    case "csv":
      return <CsvTable text={text} label={`The rows of ${part.path}`} />;
    case "text":
      return (
        <pre className="font-mono text-xs break-all whitespace-pre-wrap text-soft">{text}</pre>
      );
    case "binary":
      return (
        <div className="flex items-start gap-3 text-sm text-soft">
          <Binary aria-hidden="true" className="mt-0.5 size-5 shrink-0 text-dim" />
          <p className="max-w-[60ch]">
            A binary file of {formatBytes(part.size)}: it can&apos;t be read as text.{" "}
            {part.description}
          </p>
          <Explain topic={BINARY_TOPICS[part.path]?.topic} />
        </div>
      );
  }
}
