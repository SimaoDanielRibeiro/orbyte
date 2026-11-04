import { cn } from "cn";
import { lazy, Suspense, type ReactNode } from "react";

/**
 * O "?" da página traz o Markdown e o KaTeX (os textos do "Porquê?"): carrega-se à parte. O
 * cabeçalho também serve os ecrãs de erro da raiz, e com a importação direta a entrada
 * descarregava 700 kB de fórmulas antes de mostrar a primeira página.
 */
const PageExplain = lazy(() =>
  import("@/components/explain/explain").then((module) => ({ default: module.PageExplain })),
);

interface PageProps {
  children: ReactNode;
  /** `reading` para as páginas de texto (até 68 caracteres por linha); `wide` para o resto. */
  width?: "wide" | "reading";
  className?: string;
}

/**
 * A área de uma página. As largas ocupam o ecrã todo, com 3 cm de margem de cada lado num
 * computador (a partir de 1280 px), 32 px no tablet e 16 px no telemóvel, como a entrada; as de
 * leitura ficam numa coluna centrada. Abaixo de 1280 px, os 3 cm e a barra comiam metade do ecrã.
 *
 * Ao abrir, os blocos de cima entram em cascata (`page-in-stagger`, ADR 0048).
 */
export function Page({ children, width = "wide", className }: PageProps) {
  return (
    <div
      className={cn(
        "w-full page-in-stagger px-4 py-8 md:px-8 md:pt-14 md:pb-20 xl:px-[3cm]",
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
          <h1 className="text-title text-balance">{title}</h1>
          {explained && (
            // Enquanto chega, guarda o lugar do botão: o título não salta.
            <Suspense fallback={<span aria-hidden="true" className="size-7 shrink-0" />}>
              <PageExplain title={title} description={description} topics={topics} />
            </Suspense>
          )}
        </div>
        {subtitle !== undefined && <p className="mt-3 text-base text-soft">{subtitle}</p>}
      </div>
      {children !== undefined && <div className="flex shrink-0 flex-wrap gap-2">{children}</div>}
    </header>
  );
}
