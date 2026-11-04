/**
 * Os blocos com que as páginas se montam: uma secção com o seu título, um cartão (o "instrumento"
 * da referência: um gráfico, um número, uma tabela), um número grande, uma fila de factos, o
 * aviso do passo que falta e a nota de um erro da API.
 *
 * As explicações não ficam à vista: cada título tem o seu "?" (ADR 0032).
 */
import { Link } from "@tanstack/react-router";
import { cn } from "cn";
import { CircleAlert } from "lucide-react";
import type { ReactNode } from "react";

import { ApiRequestError } from "@/api/errors";
import { Explain } from "@/components/explain/explain";
import { Button } from "@/components/ui/button";

import type { ScreenPath } from "./screens";

interface SectionProps {
  title: string;
  /** O tema do backend que explica a secção (no "?" ao lado do título). */
  topic?: string;
  /** O que a secção mostra, numa frase: também vai para o "?", não fica à vista. */
  description?: ReactNode;
  /** As ações da secção, à direita do título. */
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Uma parte de uma página, com o título e o "?" que a explica. */
export function Section({ title, topic, description, actions, children, className }: SectionProps) {
  return (
    <section className={cn("mt-20 first:mt-0", className)}>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <h2 className="text-xl">{title}</h2>
          <Explain topic={topic} title={topic === undefined ? title : undefined}>
            {description}
          </Explain>
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

interface PanelProps {
  /** O nome do cartão, pequeno e cinzento, como na referência. */
  title?: ReactNode;
  /** O tema do backend que o explica (no "?" ao lado do nome). */
  topic?: string;
  /** Uma explicação montada aqui, quando não há tema (também no "?"). */
  explain?: ReactNode;
  /** As ações do cartão, à direita do nome. */
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  /** O nível do título, para a ordem dos títulos da página (por omissão, um `h3`). */
  as?: "h2" | "h3" | "h4";
  /** Uma âncora, para se chegar ao cartão por uma ligação. */
  id?: string;
}

/**
 * Uma secção aberta: um gráfico ou uma tabela, com o nome em cima e o "?" ao lado. Não é uma
 * caixa: um filete por cima separa-a da que está acima, e o espaço entre colunas faz o resto
 * (ADR 0034). As caixas (a classe `panel`) ficam para o que se escolhe ou se arrasta.
 */
export function Panel({
  title,
  topic,
  explain,
  actions,
  children,
  className,
  as: Heading = "h3",
  id,
}: PanelProps) {
  const label = typeof title === "string" ? title : undefined;
  return (
    <section id={id} className={cn("flex min-w-0 flex-col gap-4", className)}>
      {(title !== undefined || actions !== undefined) && (
        <div className="flex min-h-6 flex-wrap items-center justify-between gap-x-3 gap-y-2">
          <div className="flex min-w-0 items-center gap-1">
            {title !== undefined && (
              <Heading className="text-base font-medium tracking-normal text-foreground">
                {title}
              </Heading>
            )}
            <Explain topic={topic} title={topic === undefined ? label : undefined}>
              {explain}
            </Explain>
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

interface StatProps {
  label: ReactNode;
  value: ReactNode;
  /** O que acompanha o valor, em mono e pequeno (a comparação, a parte, a unidade). */
  detail?: ReactNode;
  /** A cor do detalhe: verde quando é bom, vermelho quando é mau, cinzento sem juízo. */
  tone?: "positive" | "negative" | "neutral";
  className?: string;
}

/** Um número grande com o nome por cima (o "Paid this month" da referência). */
export function Stat({ label, value, detail, tone = "neutral", className }: StatProps) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      <span className="text-base text-soft">{label}</span>
      <span className="text-[1.875rem] leading-tight font-semibold tracking-[-0.02em] text-strong tabular-nums">
        {value}
      </span>
      {detail !== undefined && (
        <span
          className={cn(
            "text-xs tabular-nums",
            tone === "positive"
              ? "text-positive"
              : tone === "negative"
                ? "text-negative"
                : "text-dim",
          )}
        >
          {detail}
        </span>
      )}
    </div>
  );
}

export interface StatItem extends Omit<StatProps, "className"> {
  key: string;
}

/**
 * Uma faixa de números grandes, lado a lado, separados por filetes (como o mostrador de um
 * instrumento), em vez de uma caixa por número. No telemóvel ficam dois por linha.
 */
export function StatStrip({
  items,
  compact = false,
  className,
}: {
  items: readonly StatItem[];
  /** Sempre dois por linha: para uma coluna estreita ou um painel lateral. */
  compact?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-2 gap-y-5",
        !compact && "@3xl/main:auto-cols-fr @3xl/main:grid-flow-col @3xl/main:grid-cols-none",
        className,
      )}
    >
      {items.map(({ key, ...stat }) => (
        <Stat
          key={key}
          {...stat}
          className={
            compact
              ? "pr-6 even:border-l even:pl-6"
              : "pr-6 @3xl/main:border-l @3xl/main:pl-6 @3xl/main:first:border-l-0 @3xl/main:first:pl-0"
          }
        />
      ))}
    </div>
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
          <dt className={size === "md" ? "text-xs text-soft" : "small-label"}>{fact.label}</dt>
          <dd
            className={cn(
              "mt-1 text-strong tabular-nums",
              size === "md" ? "text-xl font-medium tracking-[-0.01em]" : "text-sm",
            )}
          >
            {fact.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

interface ProceedProps {
  /** O ecrã do passo seguinte. */
  to: ScreenPath;
  children: ReactNode;
  disabled?: boolean;
}

/**
 * Prosseguir para o passo seguinte: o botão mais saliente da página, sempre igual (branco e maior)
 * e num só sítio, no topo à direita (ADR 0044). Sem seta: o nome já diz para onde se vai, e a
 * seta acrescentada a cada botão era uma marca de interface feita em série (ADR 0045).
 */
export function Proceed({ to, children, disabled = false }: ProceedProps) {
  if (disabled) {
    return (
      <Button variant="proceed" size="proceed" disabled>
        {children}
      </Button>
    );
  }
  return (
    <Button asChild variant="proceed" size="proceed">
      <Link to={to}>{children}</Link>
    </Button>
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
    <div className="mt-10 flex max-w-2xl flex-col items-start gap-4 panel p-6">
      <p className="text-reading text-soft">{message}</p>
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
  const message = error instanceof Error ? error.message : "Something unexpected went wrong.";
  const problems = error instanceof ApiRequestError ? error.problems : [];
  return (
    <div role="alert" className={cn("flex max-w-[68ch] gap-3 text-destructive", className)}>
      <CircleAlert aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
      <div className="text-sm leading-relaxed">
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
