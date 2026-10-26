/**
 * Os blocos com que as páginas se montam: uma secção com o seu título, uma fila de factos
 * (linhas, colunas…), o aviso do passo que falta e a nota de um erro da API.
 */
import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import { CircleAlert } from "lucide-react";
import type { ReactNode } from "react";

import { ApiRequestError } from "@/api/errors";
import { Button } from "@/components/ui/button";

import type { ScreenPath } from "./screens";

interface SectionProps {
  title: string;
  description?: ReactNode;
  /** As ações da secção, à direita do título. */
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Uma parte de uma página, com o título (serifado: é o que se lê). */
export function Section({ title, description, actions, children, className }: SectionProps) {
  return (
    <section className={cn("mt-12 first:mt-0", className)}>
      <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
        <div className="max-w-[68ch]">
          <h2 className="text-2xl">{title}</h2>
          {description !== undefined && (
            <p className="mt-1.5 font-serif text-reading text-muted-foreground">{description}</p>
          )}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export interface Fact {
  label: string;
  value: ReactNode;
}

interface FactsProps {
  facts: readonly Fact[];
  /** `md` para os números de uma página; `sm` para os de um item numa lista. */
  size?: "md" | "sm";
  className?: string;
}

/** Uma fila de factos curtos: o nome pequeno por cima, o valor por baixo. */
export function Facts({ facts, size = "md", className }: FactsProps) {
  return (
    <dl
      className={cn(
        "flex flex-wrap",
        size === "md" ? "gap-x-10 gap-y-4" : "gap-x-6 gap-y-2",
        className,
      )}
    >
      {facts.map((fact) => (
        <div key={fact.label} className="min-w-0">
          <dt className="text-xs text-muted-foreground">{fact.label}</dt>
          <dd className={cn("mt-0.5 tabular-nums", size === "md" ? "text-lg" : "text-sm")}>
            {fact.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

interface NeedsStepProps {
  /** O que falta, numa frase. */
  message: string;
  /** O ecrã onde se resolve. */
  to: ScreenPath;
  action: string;
}

/** Um ecrã a que falta um passo anterior: diz o quê e leva lá. */
export function NeedsStep({ message, to, action }: NeedsStepProps) {
  return (
    <div className="mt-10 flex max-w-[68ch] flex-col items-start gap-4 border-t pt-6">
      <p className="font-serif text-reading text-muted-foreground">{message}</p>
      <Button asChild variant="outline">
        <Link to={to}>{action}</Link>
      </Button>
    </div>
  );
}

/** O erro de um pedido: a mensagem da API (que já diz o que fazer) e os problemas, se houver. */
export function ErrorNote({ error, className }: { error: unknown; className?: string }) {
  if (error === null || error === undefined) {
    return null;
  }
  const message = error instanceof Error ? error.message : "Aconteceu um erro inesperado.";
  const problems = error instanceof ApiRequestError ? error.problems : [];
  return (
    <div role="alert" className={cn("flex max-w-[68ch] gap-3 text-destructive", className)}>
      <CircleAlert aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
      <div className="font-serif text-reading">
        <p>{message}</p>
        {problems.length > 0 && (
          <ul className="mt-1 list-disc pl-5">
            {problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
