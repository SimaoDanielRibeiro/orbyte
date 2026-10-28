import { cn } from "cn";
import type { ReactNode } from "react";

import { PageExplain } from "@/components/explain/explain";

interface PageProps {
  children: ReactNode;
  /** `reading` para as páginas de texto (até 68 caracteres por linha); `wide` para o resto. */
  width?: "wide" | "reading";
  className?: string;
}

/**
 * A área de uma página. As largas ocupam o ecrã todo, com 2 cm de margem de cada lado (só 16 px
 * no telemóvel); as de leitura ficam numa coluna centrada.
 */
export function Page({ children, width = "wide", className }: PageProps) {
  return (
    <div
      className={cn(
        "w-full px-4 py-8 md:px-[2cm] md:py-10",
        width === "reading" && "mx-auto max-w-3xl",
        className,
      )}
    >
      {children}
    </div>
  );
}

interface PageHeaderProps {
  title: string;
  /** Uma linha curta à vista, por baixo do título (o contexto: um erro, o dataset aberto). */
  subtitle?: ReactNode;
  /** O que se faz na página: fica no "?" ao lado do título, não à vista. */
  description?: string;
  /** As ideias por trás da página (os temas do "Porquê?"), no mesmo "?". */
  topics?: readonly string[];
  /** As ações da página (os botões), à direita quando há espaço. */
  children?: ReactNode;
}

/** O título de uma página, com o "?" que explica o que se faz nela. */
export function PageHeader({ title, subtitle, description, topics, children }: PageHeaderProps) {
  const explained = description !== undefined || (topics !== undefined && topics.length > 0);
  return (
    <header className="flex flex-col gap-4 @3xl/main:flex-row @3xl/main:items-end @3xl/main:justify-between">
      <div className="max-w-[68ch] min-w-0">
        <div className="flex items-center gap-2">
          <h1 className="text-[2rem] leading-tight tracking-[-0.025em] text-balance">{title}</h1>
          {explained && <PageExplain title={title} description={description} topics={topics} />}
        </div>
        {subtitle !== undefined && <p className="mt-2 text-sm text-soft">{subtitle}</p>}
      </div>
      {children !== undefined && <div className="flex shrink-0 flex-wrap gap-2">{children}</div>}
    </header>
  );
}
