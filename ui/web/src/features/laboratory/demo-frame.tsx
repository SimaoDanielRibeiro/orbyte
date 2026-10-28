/**
 * A moldura de uma demonstração do Laboratório: o que se vê (o resumo, do backend), o controlo,
 * a leitura do resultado (os avisos, também do backend) e os gráficos.
 *
 * Mexer no controlo volta a pedir a demonstração; enquanto a nova não chega, a anterior fica à
 * vista, esbatida: o gráfico nunca desaparece, e vê-se a passagem de um valor para o outro.
 */
import { cn } from "cn";
import type { ReactNode } from "react";

import type { DemoCard, Notice } from "@/api/generated";
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
  return (
    <div className="flex flex-col gap-8">
      <div className="grid gap-6 @4xl/main:grid-cols-[minmax(0,1fr)_minmax(0,22rem)] @4xl/main:items-start">
        <div className="flex max-w-[68ch] flex-col gap-4">
          <p className="font-serif text-reading">{card.summary}</p>
          <ErrorNote error={error} />
          <div aria-live="polite">{notices !== undefined && <NoticeList notices={notices} />}</div>
        </div>
        <div className="flex flex-col gap-5 rounded-md border p-4">{controls}</div>
      </div>
      {children === null ? (
        <Skeleton className="h-80 w-full" />
      ) : (
        <div
          aria-busy={stale}
          className={cn("grid gap-10 @5xl/main:grid-cols-2", stale && "stale")}
        >
          {children}
        </div>
      )}
    </div>
  );
}
