/**
 * As explicações, atrás de um botão discreto (o "?"): o texto didático deixa de estar à vista em
 * todo o lado, mas fica sempre a um clique do que explica (ADR 0032).
 *
 * Os textos são os temas do backend (regra 5 do `CONTRIBUTING.md`), pedidos pelo nome escrito por
 * inteiro (`topic="confusion_matrix"`): o teste `tests/contract/test_topics_in_use.py` procura
 * esses nomes no código e falha se algum deixar de existir.
 */
import { cn } from "cn";
import { CircleHelp } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";

import { useTopic, useTopics } from "@/api/topics";
import { Prose } from "@/components/markdown/prose";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

interface TriggerProps extends ComponentProps<"button"> {
  /** O nome do botão para os leitores de ecrã: o ícone sozinho não diz o que explica. */
  label: string;
}

/** O "?": pequeno, cinzento, e só acende ao passar o rato ou com o foco. */
export function ExplainTrigger({ label, className, ...props }: TriggerProps) {
  return (
    <button
      type="button"
      aria-label={label}
      className={cn(
        "inline-flex size-6 shrink-0 items-center justify-center rounded-full text-dim transition-colors outline-none hover:bg-surface-high hover:text-strong focus-visible:ring-2 focus-visible:ring-ring data-[state=open]:bg-surface-high data-[state=open]:text-strong",
        className,
      )}
      {...props}
    >
      <CircleHelp aria-hidden="true" className="size-[0.9375rem]" />
    </button>
  );
}

interface ExplainProps {
  /** O tema do backend (`"confusion_matrix"`): o título e o texto vêm dele. */
  topic?: string;
  /** O título, quando não há tema (ou para o trocar). */
  title?: string;
  /** O que se explica sem tema (uma legenda montada aqui), ou o que se junta ao tema. */
  children?: ReactNode;
  /** O nome do botão (por omissão, "Explicar: <título>"). */
  label?: string;
  side?: "top" | "right" | "bottom" | "left";
  align?: "start" | "center" | "end";
  className?: string;
}

/** Uma explicação curta (o "como se lê" de um gráfico, uma secção), numa caixa ao lado do "?". */
export function Explain({
  topic,
  title,
  children,
  label,
  side = "bottom",
  align = "start",
  className,
}: ExplainProps) {
  const explanation = useTopic(topic ?? "");
  if (explanation === undefined && (children === undefined || children === null)) {
    return null;
  }
  const heading = title ?? explanation?.title;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <ExplainTrigger
          label={label ?? (heading === undefined ? "Explain" : `Explain: ${heading}`)}
          className={className}
        />
      </PopoverTrigger>
      <PopoverContent
        side={side}
        align={align}
        className="w-[23rem] max-w-[calc(100vw-2rem)] gap-2.5 p-5"
      >
        {heading !== undefined && <p className="text-sm font-medium text-strong">{heading}</p>}
        {explanation !== undefined && <Prose text={explanation.text} size="small" />}
        {children !== undefined && children !== null && (
          <div className="text-sm leading-relaxed text-soft">{children}</div>
        )}
      </PopoverContent>
    </Popover>
  );
}

interface PageExplainProps {
  /** O nome da página. */
  title: string;
  /** O que se faz nela, numa frase. */
  description?: string;
  /** As ideias por trás dela: os temas do painel "Porquê?". */
  topics?: readonly string[];
}

/** O "?" do cabeçalho de uma página: abre, ao lado, o que se faz nela e as ideias por trás. */
export function PageExplain({ title, description, topics = [] }: PageExplainProps) {
  const explanations = useTopics();
  return (
    <Sheet>
      <SheetTrigger asChild>
        <ExplainTrigger label={`About the ${title} screen`} className="size-7" />
      </SheetTrigger>
      <SheetContent
        side="right"
        className="w-full gap-0 overflow-y-auto data-[side=right]:sm:max-w-lg"
      >
        <SheetHeader className="gap-2 border-b p-6">
          <SheetTitle className="text-xl font-medium text-strong">{title}</SheetTitle>
          {description !== undefined && (
            <SheetDescription className="text-sm leading-relaxed text-soft">
              {description}
            </SheetDescription>
          )}
        </SheetHeader>
        <div className="flex flex-col gap-8 p-6">
          {topics.map((topic) => {
            const explanation = explanations[topic];
            return explanation === undefined ? null : (
              <section key={topic}>
                <h3 className="text-base">{explanation.title}</h3>
                <Prose text={explanation.text} className="mt-2" />
              </section>
            );
          })}
        </div>
      </SheetContent>
    </Sheet>
  );
}
