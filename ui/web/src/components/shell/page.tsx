import { cn } from "cn";
import type { ReactNode } from "react";

interface PageProps {
  children: ReactNode;
  /** `reading` para as páginas de texto (até 68 caracteres por linha); `wide` para o resto. */
  width?: "wide" | "reading";
  className?: string;
}

/** A área de uma página, com as margens de todas. */
export function Page({ children, width = "wide", className }: PageProps) {
  return (
    <div
      className={cn(
        "mx-auto w-full px-4 py-8 md:px-8 md:py-12",
        width === "reading" ? "max-w-3xl" : "max-w-[90rem]",
        className,
      )}
    >
      {children}
    </div>
  );
}

interface PageHeaderProps {
  title: string;
  description?: string;
  /** As ações da página (os botões), à direita quando há espaço. */
  children?: ReactNode;
}

/** O título de uma página e o que se faz nela. */
export function PageHeader({ title, description, children }: PageHeaderProps) {
  return (
    <header className="flex flex-col gap-4 @3xl/main:flex-row @3xl/main:items-end @3xl/main:justify-between">
      <div className="max-w-[68ch]">
        <h1 className="text-4xl leading-tight">{title}</h1>
        {description !== undefined && (
          <p className="mt-3 font-serif text-reading text-muted-foreground">{description}</p>
        )}
      </div>
      {children !== undefined && <div className="flex shrink-0 gap-2">{children}</div>}
    </header>
  );
}
