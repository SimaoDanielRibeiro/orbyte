/**
 * A moldura de uma demonstração do Laboratório, pela ordem em que se aprende: a ideia, o que se
 * vê, o controlo com o que experimentar, a leitura do resultado (os avisos, do backend), os
 * gráficos, e no fim o que fica.
 *
 * No resto da aplicação as explicações ficam no "?" (ADR 0032); aqui ficam à vista, porque no
 * Laboratório a explicação é o conteúdo (ADR 0051). Os textos vêm todos do guia do cartão.
 *
 * Mexer no controlo volta a pedir a demonstração; enquanto a nova não chega, a anterior fica à
 * vista, esbatida: o gráfico nunca desaparece, e vê-se a passagem de um valor para o outro.
 */
import { cn } from "cn";
import { useId, type ReactNode } from "react";

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
  const { guide } = card;
  return (
    <div className="flex flex-col gap-12">
      <div className="grid gap-x-14 gap-y-10 @4xl/main:grid-cols-[minmax(0,1fr)_minmax(0,24rem)]">
        <div className="flex max-w-3xl min-w-0 flex-col gap-8">
          <p className="text-xl leading-relaxed text-balance">{guide.idea}</p>
          <Part title="What you see">
            <p className="leading-relaxed text-soft">{card.summary}</p>
          </Part>
          <ErrorNote error={error} />
          {/* A leitura muda com o controlo: quem usa um leitor de ecrã ouve-a mudar. */}
          <Part title="What it shows now">
            <div aria-live="polite">
              {notices === undefined ? (
                <Skeleton className="h-16 w-full" />
              ) : (
                <NoticeList notices={notices} />
              )}
            </div>
          </Part>
        </div>
        <div className="flex min-w-0 flex-col gap-10">
          <div className="flex flex-col gap-5">{controls}</div>
          <Part title="Try this">
            <ol className="flex flex-col gap-3.5">
              {guide.steps.map((step, index) => (
                <li key={step} className="flex gap-3 text-sm leading-relaxed">
                  <span
                    aria-hidden="true"
                    className="mt-px grid size-5 shrink-0 place-items-center rounded-full border border-border font-mono text-[0.6875rem] text-soft tabular-nums"
                  >
                    {index + 1}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </Part>
        </div>
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
      <div className="border-t border-border pt-8">
        <Part title="What to take away">
          <p className="max-w-3xl text-lg leading-relaxed text-balance">{guide.takeaway}</p>
        </Part>
      </div>
    </div>
  );
}

/** Uma parte do guia: o nome pequeno por cima, e o conteúdo. */
function Part({ title, children }: { title: string; children: ReactNode }) {
  const id = useId();
  return (
    <section aria-labelledby={id} className="flex flex-col gap-3">
      <h2 id={id} className="text-sm font-medium tracking-normal text-soft">
        {title}
      </h2>
      {children}
    </section>
  );
}
