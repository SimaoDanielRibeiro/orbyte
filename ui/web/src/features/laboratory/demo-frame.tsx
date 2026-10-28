/**
 * A moldura de uma demonstração do Laboratório: o controlo, a leitura do resultado (os avisos,
 * do backend) e os gráficos, cada um no seu cartão. O resumo do que se vê fica no "?".
 *
 * Mexer no controlo volta a pedir a demonstração; enquanto a nova não chega, a anterior fica à
 * vista, esbatida: o gráfico nunca desaparece, e vê-se a passagem de um valor para o outro.
 */
import { cn } from "cn";
import type { ReactNode } from "react";

import type { DemoCard, Notice } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote } from "@/components/shell/blocks";
import { Skeleton } from "@/components/ui/skeleton";

interface DemoFrameProps {
  card: DemoCard;
  /** Os controlos da demonstração. */
  controls: ReactNode;
  notices: readonly Notice[] | undefined;
  /** Se a demonstração mostrada ainda é a do valor anterior. */
  stale: boolean;
  error: unknown;
  /** Os gráficos (`null` enquanto a primeira resposta não chega). */
  children: ReactNode;
}

export function DemoFrame({ card, controls, notices, stale, error, children }: DemoFrameProps) {
  const label = useLabel();
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-x-12 gap-y-10 @4xl/main:grid-cols-[minmax(0,1fr)_minmax(0,24rem)] @4xl/main:items-stretch">
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-1">
            <h2 className="text-[0.9375rem] font-normal tracking-normal text-soft">O que se vê</h2>
            <Explain title={label("LaboratoryDemo", card.name)}>{card.summary}</Explain>
          </div>
          <ErrorNote error={error} />
          <div aria-live="polite">{notices !== undefined && <NoticeList notices={notices} />}</div>
        </div>
        <div className="flex flex-col gap-5">{controls}</div>
      </div>
      {children === null ? (
        <Skeleton className="h-80 w-full" />
      ) : (
        // Cada gráfico (os filhos diretos) fica no seu cartão.
        <div
          aria-busy={stale}
          className={cn("grid gap-x-12 gap-y-10 *:min-w-0 @5xl/main:grid-cols-2", stale && "stale")}
        >
          {children}
        </div>
      )}
    </div>
  );
}
