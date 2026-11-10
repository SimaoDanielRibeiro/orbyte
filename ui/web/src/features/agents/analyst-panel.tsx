/**
 * O analista (fase 13, ADR 0053, 0054 e 0057): depois de a procura acabar, conversa-se com ele
 * sobre ela.
 *
 * O painel é uma conversa, como num chat: um cartão ao centro da página, com a altura fixa, em que
 * só a conversa desliza (a página fica onde está). Em cima, quem é, o modelo de linguagem e o botão
 * "Expand", que abre a mesma conversa em ecrã inteiro, como a conversa dos agentes: a conversa ao
 * centro e a caixa da pergunta em baixo. No meio, as perguntas (à direita) e as respostas do
 * analista (à esquerda); em baixo, a caixa da pergunta, com as perguntas sugeridas que ainda não se
 * fizeram. Sem perguntas, o meio dá as boas-vindas: as perguntas sugeridas e as ferramentas que ele
 * usa.
 *
 * O analista é um agente de IA com ferramentas: pensa, pede uma ferramenta (o quadro, um
 * candidato, dois lado a lado, o plano dos dados), lê o que ela devolveu e volta a pensar, até
 * responder. Enquanto trabalha, os passos chegam à vista, com o raciocínio a crescer, e a conversa
 * segue-os (quem sobe para ler fica onde está, com um botão para voltar ao fim). Quando responde,
 * os passos fecham-se numa linha com as ferramentas que usou, e abrem-se pelo que ele fez (cada
 * ferramenta e, no fim, a resposta), cada um com o pensamento que o levou lá fechado por baixo
 * (`analyst-steps.ts`). A resposta fica em texto corrido, sem caixa, como num chat; por baixo, o
 * botão para a copiar, o selo dos números verificados (a leitura do backend), o modelo e o tempo.
 *
 * Sem o Ollama, o painel mostra o estado de `GET /llm/status` e o que fazer; as respostas que já
 * houver continuam à vista. As perguntas sugeridas, os nomes das ferramentas e dos passos e os
 * avisos vêm do backend; aqui fica só a apresentação.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "cn";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  BadgeCheck,
  Check,
  ChevronRight,
  Copy,
  LoaderCircle,
  Maximize2,
  MessageSquareText,
  Minimize2,
  Sparkles,
  Square,
  Wrench,
} from "lucide-react";
import { useReducedMotion } from "motion/react";
import { Dialog } from "radix-ui";
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type SyntheticEvent,
} from "react";
import { toast } from "sonner";

import type {
  AnalystAnswer,
  AnalystOptions,
  AnalystStep,
  LiveTurn,
  Notice,
  ToolUse,
} from "@/api/generated";
import {
  automlAnalystOptionsOptions,
  automlAskMutation,
  automlCancelQuestionMutation,
  automlQuestionsOptions,
  automlQuestionsQueryKey,
  llmStatusOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { Prose } from "@/components/markdown/prose";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatCount, formatNumber, formatQuantity } from "@/lib/format";
import { useSession, type AnalystQuestion } from "@/state/session";

import { AI_AGENT_COLOR } from "./agent-colors";
import { rounds, wordCount, type Round } from "./analyst-steps";
import { useQuestion } from "./use-question";

/** A que distância do fim (em píxeis) a conversa ainda conta como "no fim", para seguir as novas. */
const BOTTOM_SLACK = 64;

/** A altura máxima da caixa da pergunta, em píxeis: a partir daí, desliza ela. */
const COMPOSER_MAX_HEIGHT = 160;

/** Quanto tempo o botão de copiar diz que copiou, em milissegundos. */
const COPIED_MS = 2000;

/**
 * O aviso do backend que diz que cada número da resposta está no que as ferramentas devolveram
 * (`answer_notices`, ADR 0053). Mostra-se como um selo pequeno por baixo da resposta, com a
 * mensagem inteira na dica e para os leitores de ecrã; os outros avisos ficam por extenso (um
 * número que nenhuma ferramenta devolveu tem de se ver).
 */
const NUMBERS_CHECKED = "numbers_checked";

const TITLE = "Ask the analyst";
const DESCRIPTION = "An AI agent that reads what this search measured, step by step";
const TITLE_CLASS = "truncate text-base font-medium tracking-normal text-strong";
const DESCRIPTION_CLASS = "truncate text-xs text-dim";

export function AnalystPanel({ searchId }: { searchId: string }) {
  const chat = useAnalystChat(searchId);
  // A conversa em ecrã inteiro (ADR 0057). Mostra-a um sítio de cada vez: o cartão fica na página,
  // com a mesma altura, para nada saltar ao fechar.
  const [expanded, setExpanded] = useState(false);
  return (
    <Dialog.Root open={expanded} onOpenChange={setExpanded}>
      <section
        aria-labelledby="analyst-title"
        className="mx-auto flex h-[clamp(32rem,78vh,52rem)] w-full max-w-4xl min-w-0 flex-col overflow-hidden rounded-3xl border border-[#262626] bg-[#0f0f0f] shadow-2xl shadow-black/40"
      >
        <PanelHeader
          className="py-3"
          title={
            <h2 id="analyst-title" className={TITLE_CLASS}>
              {TITLE}
            </h2>
          }
          description={<p className={DESCRIPTION_CLASS}>{DESCRIPTION}</p>}
          model={chat.model}
          ready={chat.ready}
          action={
            <Dialog.Trigger asChild>
              <Button variant="outline" size="sm" className="shrink-0">
                <Maximize2 aria-hidden="true" />
                Expand
              </Button>
            </Dialog.Trigger>
          }
        />
        {expanded ? <OpenInFullScreen /> : <Chat chat={chat} layout="card" />}
      </section>
      <AnalystRoom chat={chat} />
    </Dialog.Root>
  );
}

/** O que a conversa precisa, partilhado pelo cartão e pelo ecrã inteiro. */
interface AnalystChat {
  options: AnalystOptions | undefined;
  /** O modelo de linguagem configurado ("qwen3:8b"), se houver um. */
  model: string | null;
  /** Se o modelo de linguagem está pronto a responder. */
  ready: boolean;
  /** O que diz `GET /llm/status` quando o modelo não está pronto (o que fazer). */
  llmNotices: readonly Notice[] | null;
  answers: readonly AnalystAnswer[];
  /** A pergunta a decorrer, enquanto não está nas respondidas. */
  pending: AnalystQuestion | null;
  /** O analista ocupado: a pergunta a seguir para a API, ou ele a trabalhar nela. */
  busy: boolean;
  /** A pergunta a seguir para a API. */
  sending: boolean;
  /** A pergunta que se está a escrever. */
  question: string;
  setQuestion: (value: string) => void;
  /** Pergunta, e diz se a pergunta seguiu (não segue vazia, nem com o analista ocupado). */
  send: (text: string) => boolean;
  /** As perguntas sugeridas que ainda não se fizeram. */
  remaining: readonly string[];
  /** Pára a pergunta a decorrer (só com o analista a trabalhar). */
  stop: (() => void) | null;
  stopping: boolean;
  error: unknown;
  /** A resposta chegou: a lista das respondidas volta a pedir-se, e passa a mostrá-la. */
  onAnswered: () => Promise<void>;
  /** A pergunta falhou, ou a API já não a tem: pode fazer-se outra. */
  onFailed: (questionId: string) => void;
}

/**
 * A conversa com o analista: o que já respondeu, a pergunta a decorrer, a caixa da pergunta e o
 * estado do modelo de linguagem. Vive no painel, por cima dos dois sítios que a mostram: passar
 * do cartão para o ecrã inteiro (e voltar) não perde nada, nem o que se está a escrever.
 */
function useAnalystChat(searchId: string): AnalystChat {
  const path = { search_id: searchId };
  const queryClient = useQueryClient();
  const options = useQuery(automlAnalystOptionsOptions({ path }));
  const asked = useQuery(automlQuestionsOptions({ path }));
  // Se o Ollama está pronto: pergunta-se outra vez de vez em quando, para o painel ligar sozinho
  // quando se abre o Ollama com o ecrã aberto (como o interruptor do pedido aos agentes).
  const llm = useQuery({ ...llmStatusOptions(), refetchInterval: 10_000 });
  const ready = llm.data?.ready === true;
  const [question, setQuestion] = useState("");
  // A pergunta a decorrer fica na sessão: sair do ecrã (ou recarregar) e voltar segue-a outra vez.
  const stored = useSession((state) => state.analystQuestion);
  const openAnalystQuestion = useSession((state) => state.openAnalystQuestion);
  const current = stored?.searchId === searchId ? stored : null;
  const [failed, setFailed] = useState<string | null>(null);
  const ask = useMutation({
    ...automlAskMutation(),
    onSuccess: (job, variables) => {
      openAnalystQuestion({ searchId, questionId: job.run_id, question: variables.body.question });
      setQuestion("");
    },
  });
  const cancel = useMutation(automlCancelQuestionMutation());
  const answers = asked.data ?? [];
  // Quando a resposta entra nas respondidas, a vez ao vivo dá lugar a ela.
  const answered =
    current !== null && answers.some((one) => one.question_id === current.questionId);
  const working = current !== null && current.questionId !== failed && !answered;
  const busy = ask.isPending || working;

  // Estáveis: a pergunta a decorrer chama-as num efeito, que não pode correr a cada desenho.
  const onAnswered = useCallback(async () => {
    // Primeiro chega a lista com a resposta, e só depois se esquece a pergunta: a resposta nunca
    // desaparece do ecrã entre as duas coisas.
    await queryClient.invalidateQueries({
      queryKey: automlQuestionsQueryKey({ path: { search_id: searchId } }),
    });
    openAnalystQuestion(null);
  }, [queryClient, searchId, openAnalystQuestion]);
  const onFailed = useCallback((questionId: string) => {
    setFailed(questionId);
  }, []);

  const send = (text: string): boolean => {
    const trimmed = text.trim();
    if (trimmed === "" || busy || !ready) {
      return false;
    }
    ask.mutate({ path, body: { question: trimmed } });
    return true;
  };

  // As sugestões que ainda não se perguntaram ficam por cima da caixa da pergunta.
  const askedAlready = new Set(
    [...answers.map((answer) => answer.question), current?.question ?? ""].map(normalized),
  );
  const suggestions = options.data?.suggestions ?? [];

  return {
    options: options.data,
    model: options.data?.model ?? null,
    ready,
    llmNotices: llm.data !== undefined && !ready ? llm.data.notices : null,
    answers,
    pending: current !== null && !answered ? current : null,
    busy,
    sending: ask.isPending,
    question,
    setQuestion,
    send,
    remaining: suggestions.filter((one) => !askedAlready.has(normalized(one))),
    stop: working
      ? () => {
          cancel.mutate({ path: { question_id: current.questionId } });
        }
      : null,
    stopping: cancel.isPending,
    error: ask.error ?? cancel.error,
    onAnswered,
    onFailed,
  };
}

/** As perguntas comparam-se sem maiúsculas nem espaços à volta. */
function normalized(text: string): string {
  return text.trim().toLowerCase();
}

/**
 * A conversa em ecrã inteiro (ADR 0057), como a dos agentes: o mesmo cabeçalho, a conversa ao
 * centro e a caixa da pergunta em baixo, como num chat. Abre-se no "Expand" e fecha-se no
 * "Collapse" ou com o Esc; a pergunta a meio de escrever e a que está a decorrer continuam.
 */
function AnalystRoom({ chat }: { chat: AnalystChat }) {
  const content = useRef<HTMLDivElement>(null);
  return (
    <Dialog.Portal>
      <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 duration-150 supports-backdrop-filter:backdrop-blur-sm data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
      <Dialog.Content
        ref={content}
        // Abre-se para perguntar: o foco vai para a caixa da pergunta, se der para perguntar.
        onOpenAutoFocus={(event) => {
          const box = content.current?.querySelector<HTMLTextAreaElement>("textarea:enabled");
          if (box !== null && box !== undefined) {
            event.preventDefault();
            box.focus();
          }
        }}
        className="fixed inset-0 z-50 flex flex-col bg-[#0f0f0f] duration-200 outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-[0.985] data-closed:animate-out data-closed:fade-out-0"
      >
        <PanelHeader
          className="h-16"
          title={<Dialog.Title className={TITLE_CLASS}>{TITLE}</Dialog.Title>}
          description={
            <Dialog.Description className={DESCRIPTION_CLASS}>{DESCRIPTION}</Dialog.Description>
          }
          model={chat.model}
          ready={chat.ready}
          action={
            <Dialog.Close asChild>
              <Button variant="outline" size="sm" className="shrink-0">
                <Minimize2 aria-hidden="true" />
                Collapse
              </Button>
            </Dialog.Close>
          }
        />
        <Chat chat={chat} layout="room" />
      </Dialog.Content>
    </Dialog.Portal>
  );
}

/** O cartão, enquanto a conversa está em ecrã inteiro: fica no mesmo sítio, por baixo. */
function OpenInFullScreen() {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 px-6 text-center">
      <Maximize2 aria-hidden="true" className="size-5 text-dim" />
      <p className="text-sm text-soft">The conversation is open in full screen.</p>
    </div>
  );
}

interface PanelHeaderProps {
  /** O título: um `h2` no cartão, o título do diálogo em ecrã inteiro. */
  title: ReactNode;
  description: ReactNode;
  model: string | null;
  ready: boolean;
  /** À direita: abrir em ecrã inteiro, ou voltar ao cartão. */
  action: ReactNode;
  className?: string;
}

/** Em cima: quem é, o que é, o modelo de linguagem com o seu estado e o botão de mudar de sítio. */
function PanelHeader({ title, description, model, ready, action, className }: PanelHeaderProps) {
  return (
    <header
      className={cn(
        "flex shrink-0 items-center gap-3 border-b border-[#222] px-4 sm:px-5",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className="flex size-9 shrink-0 items-center justify-center rounded-xl border bg-[#161616]"
        style={{ borderColor: AI_AGENT_COLOR }}
      >
        <Sparkles className="size-4" style={{ color: AI_AGENT_COLOR }} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1">
          {title}
          <Explain topic="analyst" />
        </div>
        {description}
      </div>
      {model !== null && (
        <span className="hidden shrink-0 items-center gap-2 rounded-full border border-[#2a2a2a] px-2.5 py-1 font-mono text-xs text-soft sm:flex">
          <span
            aria-hidden="true"
            className={cn("size-1.5 rounded-full", ready ? "bg-positive" : "bg-[#6b6b6b]")}
          />
          {model}
          <span className="sr-only">{ready ? ", ready" : ", not running"}</span>
        </span>
      )}
      {action}
    </header>
  );
}

interface ChatProps {
  chat: AnalystChat;
  /** No cartão da página, ou em ecrã inteiro (com mais espaço à volta, como num chat). */
  layout: "card" | "room";
}

/** A conversa e a caixa da pergunta: o mesmo no cartão e em ecrã inteiro. */
function Chat({ chat, layout }: ChatProps) {
  const { scroller, content, detached, onScroll, follow } = useStickToBottom(chat.busy);
  const empty = chat.answers.length === 0 && chat.pending === null;
  const room = layout === "room";

  // Ao abrir, a conversa começa no fim (a última pergunta), como num chat.
  const placed = useRef(false);
  useLayoutEffect(() => {
    if (!placed.current && chat.answers.length > 0) {
      placed.current = true;
      follow(false);
    }
  }, [chat.answers.length, follow]);

  const ask = (text: string) => {
    if (chat.send(text)) {
      follow(true);
    }
  };

  return (
    <>
      <div className="relative min-h-0 flex-1">
        {/* `relative`: o que lá dentro tem posição absoluta (os textos só para os leitores de
            ecrã) fica na conversa e desliza com ela; senão escapava-lhe, abaixo do cartão, e o
            cartão (com `overflow-hidden`) deslizava por dentro quando o foco lá chegava. */}
        <div
          ref={scroller}
          onScroll={onScroll}
          role="region"
          aria-label="The conversation with the analyst"
          tabIndex={0}
          className="relative h-full overflow-y-auto overscroll-contain outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
        >
          <div
            ref={content}
            className={cn(
              "mx-auto flex max-w-3xl flex-col gap-10 px-4 py-8 sm:px-8",
              empty && "min-h-full justify-center",
            )}
          >
            {empty ? (
              <Welcome options={chat.options} disabled={!chat.ready || chat.busy} onAsk={ask} />
            ) : (
              <>
                {chat.answers.map((answer) => (
                  <AnswerTurn key={answer.question_id} answer={answer} />
                ))}
                {chat.pending !== null && (
                  <LiveQuestionTurn
                    key={chat.pending.questionId}
                    questionId={chat.pending.questionId}
                    question={chat.pending.question}
                    onAnswered={chat.onAnswered}
                    onFailed={chat.onFailed}
                  />
                )}
              </>
            )}
          </div>
        </div>
        {/* Em ecrã inteiro não há linha por cima da caixa da pergunta: a conversa esbate-se. */}
        {room && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute inset-x-0 bottom-0 h-8 bg-linear-to-b from-transparent to-[#0f0f0f]"
          />
        )}
        {detached && !empty && (
          <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center">
            <Button
              size="sm"
              className="pointer-events-auto rounded-full shadow-lg shadow-black/60"
              onClick={() => {
                follow(true);
              }}
            >
              <ArrowDown aria-hidden="true" />
              Latest
            </Button>
          </div>
        )}
      </div>

      <footer className={cn("shrink-0 px-4 pt-3 pb-4 sm:px-8", !room && "border-t border-[#222]")}>
        {/* Com a largura das mensagens: os 48rem da conversa, menos os 2rem de cada lado. */}
        <div className="mx-auto flex max-w-[44rem] flex-col gap-3">
          {chat.llmNotices !== null && (
            <div role="status" aria-label="The analyst's language model">
              <NoticeList notices={chat.llmNotices} />
            </div>
          )}
          {!empty && chat.remaining.length > 0 && (
            <SuggestionRow
              suggestions={chat.remaining}
              disabled={!chat.ready || chat.busy}
              onAsk={ask}
            />
          )}
          <Composer
            value={chat.question}
            onChange={chat.setQuestion}
            onSend={() => {
              ask(chat.question);
            }}
            disabled={!chat.ready || chat.busy}
            canSend={chat.ready && !chat.busy && chat.question.trim() !== ""}
            sending={chat.sending}
            onStop={chat.stop}
            stopping={chat.stopping}
          />
          {/* Num telemóvel não há Shift+Enter: a dica fica para os ecrãs com teclado. */}
          {room && (
            <p className="hidden text-center text-xs text-dim sm:block">
              Enter to ask · Shift+Enter for a new line
            </p>
          )}
          <ErrorNote error={chat.error} />
        </div>
      </footer>
    </>
  );
}

/**
 * Deslizar a conversa, como num chat: enquanto o analista trabalha, o que cresce (os passos, o
 * raciocínio, a resposta a chegar) segue-se, se se estava no fim. Quem sobe para ler deixa de
 * seguir, e chegar outra vez ao fim volta a seguir. Só a conversa desliza: nunca se pede à página
 * que mexa (nada de `scrollIntoView`), e o `overscroll-contain` não passa o deslizar à página
 * quando a conversa chega a uma ponta.
 */
function useStickToBottom(working: boolean) {
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const lastTop = useRef(0);
  const workingNow = useRef(working);
  const [detached, setDetached] = useState(false);
  const reduced = useReducedMotion() ?? false;

  useEffect(() => {
    workingNow.current = working;
  }, [working]);

  const toBottom = useCallback(
    (smooth: boolean) => {
      const element = scroller.current;
      element?.scrollTo({
        top: element.scrollHeight,
        behavior: smooth && !reduced ? "smooth" : "auto",
      });
    },
    [reduced],
  );

  const onScroll = useCallback(() => {
    const element = scroller.current;
    if (element === null) {
      return;
    }
    // Chegar ao fim volta a seguir; só subir deixa de seguir. O deslizar suave até ao fim só
    // desce: a meio dele ainda não se está no fim, mas não se deixa de seguir.
    const bottom = element.scrollHeight - element.scrollTop - element.clientHeight < BOTTOM_SLACK;
    if (bottom) {
      following.current = true;
    } else if (element.scrollTop < lastTop.current - 1) {
      following.current = false;
    }
    lastTop.current = element.scrollTop;
    setDetached(!following.current);
  }, []);

  useEffect(() => {
    const element = content.current;
    if (element === null || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      // Só enquanto trabalha: abrir um passo de uma resposta antiga não salta para o fim.
      if (following.current && workingNow.current) {
        toBottom(false);
      }
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
    };
  }, [toBottom]);

  /** Vai ao fim e volta a seguir (ao perguntar, ao abrir, e no botão "Latest"). */
  const follow = useCallback(
    (smooth: boolean) => {
      following.current = true;
      setDetached(false);
      toBottom(smooth);
    },
    [toBottom],
  );

  return { scroller, content, detached, onScroll, follow };
}

/** Quem é o analista: o ícone dele, com o contorno lilás do agente de IA (ADR 0042). */
function AnalystAvatar({ size = "md", dashed = false }: { size?: "md" | "lg"; dashed?: boolean }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex shrink-0 items-center justify-center border bg-[#181818]",
        size === "lg" ? "size-12 rounded-2xl" : "size-8 rounded-xl",
        dashed && "border-dashed",
      )}
      style={{ borderColor: AI_AGENT_COLOR }}
    >
      <Sparkles
        className={cn(size === "lg" ? "size-5" : "size-4", dashed && "animate-pulse")}
        style={{ color: AI_AGENT_COLOR }}
      />
    </span>
  );
}

/** A conversa vazia: o que perguntar e as ferramentas que o analista usa para responder. */
function Welcome({
  options,
  disabled,
  onAsk,
}: {
  options: AnalystOptions | undefined;
  disabled: boolean;
  onAsk: (question: string) => void;
}) {
  return (
    <div className="flex flex-col items-center gap-8 py-4 text-center">
      <div className="flex flex-col items-center gap-3">
        <AnalystAvatar size="lg" />
        <p className="text-xl font-medium text-strong">Ask about this search</p>
        <p className="max-w-[46ch] text-sm leading-relaxed text-soft">
          Pick a question or write your own. You&apos;ll see each tool the analyst uses before it
          answers.
        </p>
      </div>
      {options !== undefined && options.suggestions.length > 0 && (
        <ul aria-label="Suggested questions" className="grid w-full gap-2 sm:grid-cols-2">
          {options.suggestions.map((suggestion) => (
            <li key={suggestion} className="flex">
              <button
                type="button"
                disabled={disabled}
                onClick={() => {
                  onAsk(suggestion);
                }}
                className="group flex w-full items-center justify-between gap-3 rounded-2xl border border-[#262626] bg-[#151515] px-4 py-3 text-left text-sm text-foreground/90 transition-colors outline-none hover:border-[#3a3a3a] hover:bg-[#1a1a1a] focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50"
              >
                {suggestion}
                <ArrowUpRight
                  aria-hidden="true"
                  className="size-4 shrink-0 text-dim transition-colors group-hover:text-strong"
                />
              </button>
            </li>
          ))}
        </ul>
      )}
      {options !== undefined && options.tools.length > 0 && (
        <div className="w-full text-left">
          <p className="small-label">Its tools</p>
          <ul aria-label="The analyst's tools" className="mt-3 grid gap-x-6 gap-y-3 sm:grid-cols-2">
            {options.tools.map((tool) => (
              <li key={tool.name} className="flex gap-2.5 text-xs">
                <Wrench aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-dim" />
                <span>
                  <span className="font-medium text-strong">{tool.label}</span>
                  <span className="text-soft"> · {tool.purpose}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * As perguntas sugeridas que faltam, por cima da caixa da pergunta: uma fila que desliza para o
 * lado, sem barra à vista. A roda do rato também a desliza (até à ponta; daí em diante, a roda
 * volta à conversa), e a borda direita esbate-se enquanto houver mais sugestões para lá dela.
 */
function SuggestionRow({
  suggestions,
  disabled,
  onAsk,
}: {
  suggestions: readonly string[];
  disabled: boolean;
  onAsk: (question: string) => void;
}) {
  const row = useRef<HTMLUListElement>(null);
  const [more, setMore] = useState(false);
  const measure = useCallback(() => {
    const element = row.current;
    if (element !== null) {
      setMore(element.scrollLeft + element.clientWidth < element.scrollWidth - 1);
    }
  }, []);
  useEffect(() => {
    const element = row.current;
    if (element === null) {
      return;
    }
    measure();
    const onWheel = (event: WheelEvent) => {
      if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) {
        return;
      }
      const room = element.scrollWidth - element.clientWidth;
      const next = Math.min(room, Math.max(0, element.scrollLeft + event.deltaY));
      if (room <= 0 || next === element.scrollLeft) {
        return;
      }
      event.preventDefault();
      element.scrollLeft = next;
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(measure);
    observer?.observe(element);
    return () => {
      element.removeEventListener("wheel", onWheel);
      observer?.disconnect();
    };
  }, [measure, suggestions.length]);
  return (
    <ul
      ref={row}
      aria-label="Suggested questions"
      onScroll={measure}
      className={cn(
        "-mx-1 flex [scrollbar-width:none] gap-2 overflow-x-auto px-1 [&::-webkit-scrollbar]:hidden",
        more && "[mask-image:linear-gradient(to_right,black_calc(100%-3rem),transparent)]",
      )}
    >
      {suggestions.map((suggestion) => (
        <li key={suggestion} className="shrink-0">
          <button
            type="button"
            disabled={disabled}
            onClick={() => {
              onAsk(suggestion);
            }}
            className="rounded-full border border-[#2a2a2a] bg-[#151515] px-3 py-1.5 text-xs text-soft transition-colors outline-none hover:border-[#3a3a3a] hover:text-strong focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset disabled:pointer-events-none disabled:opacity-50"
          >
            {suggestion}
          </button>
        </li>
      ))}
    </ul>
  );
}

interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled: boolean;
  canSend: boolean;
  /** A pergunta a seguir para a API. */
  sending: boolean;
  /** Com o analista a trabalhar, o botão de parar toma o lugar do de perguntar. */
  onStop: (() => void) | null;
  stopping: boolean;
}

/**
 * A caixa da pergunta, em baixo: cresce com o texto (até `COMPOSER_MAX_HEIGHT`), Enter pergunta e
 * Shift+Enter muda de linha.
 */
function Composer({
  value,
  onChange,
  onSend,
  disabled,
  canSend,
  sending,
  onStop,
  stopping,
}: ComposerProps) {
  const box = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const element = box.current;
    if (element === null) {
      return;
    }
    element.style.height = "auto";
    element.style.height = `${String(Math.min(element.scrollHeight, COMPOSER_MAX_HEIGHT))}px`;
  }, [value]);
  const onSubmit = (event: SyntheticEvent) => {
    event.preventDefault();
    onSend();
  };
  return (
    <form
      onSubmit={onSubmit}
      className={cn(
        "flex items-end gap-2 rounded-2xl border border-[#2c2c2c] bg-[#161616] py-2 pr-2 pl-4 transition-colors focus-within:border-[#555]",
        disabled && "opacity-80",
      )}
    >
      <textarea
        ref={box}
        rows={1}
        aria-label="Your question to the analyst"
        placeholder="Ask about this search…"
        value={value}
        maxLength={500}
        disabled={disabled}
        onChange={(event) => {
          onChange(event.target.value);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            onSend();
          }
        }}
        className="min-h-10 flex-1 resize-none bg-transparent py-2 text-sm leading-relaxed text-strong outline-none placeholder:text-dim disabled:cursor-not-allowed"
      />
      {onStop !== null ? (
        <Button
          type="button"
          variant="outline"
          className="rounded-xl"
          disabled={stopping}
          onClick={onStop}
        >
          <Square aria-hidden="true" className="fill-current" />
          Stop
        </Button>
      ) : (
        <Button
          type="submit"
          size="icon-lg"
          aria-label="Ask"
          disabled={!canSend}
          className="rounded-xl"
        >
          {sending ? (
            <LoaderCircle aria-hidden="true" className="animate-spin" />
          ) : (
            <ArrowUp aria-hidden="true" />
          )}
        </Button>
      )}
    </form>
  );
}

/** Uma pergunta já respondida: a pergunta, os passos (fechados), a resposta e o que se leu dela. */
function AnswerTurn({ answer }: { answer: AnalystAnswer }) {
  const work = answer.trace.filter((step) => step.kind !== "answer");
  const written = answer.answer !== "";
  const checked = answer.notices.find((notice) => notice.code === NUMBERS_CHECKED) ?? null;
  const notices = answer.notices.filter((notice) => notice.code !== NUMBERS_CHECKED);
  return (
    <Turn question={answer.question}>
      {work.length > 0 && <Work steps={work} answered={written} />}
      {written && <AnswerText text={answer.answer} />}
      {notices.length > 0 && <NoticeList notices={notices} className="gap-2" />}
      <AnswerFooter answer={answer} checked={checked} />
    </Turn>
  );
}

interface LiveQuestionTurnProps {
  questionId: string;
  question: string;
  /** A resposta chegou: a lista das respondidas volta a pedir-se, e passa a mostrá-la. */
  onAnswered: () => Promise<void>;
  /** A pergunta falhou, ou a API já não a tem: pode fazer-se outra. */
  onFailed: (questionId: string) => void;
}

/** A pergunta a decorrer: os passos a chegar, o raciocínio a crescer e a resposta a ser escrita. */
function LiveQuestionTurn({ questionId, question, onAnswered, onFailed }: LiveQuestionTurnProps) {
  const { live, status, result } = useQuestion(questionId);
  const arrived = result.data !== undefined;
  // A API já não tem a pergunta (recomeçou, ou a pergunta saiu da memória): não há o que seguir.
  const lost = result.error !== null;
  useEffect(() => {
    if (arrived) {
      void onAnswered();
    }
  }, [arrived, onAnswered]);
  useEffect(() => {
    if (status === "failed" || lost) {
      onFailed(questionId);
    }
  }, [status, lost, questionId, onFailed]);
  const running = !lost && (status === null || status === "pending" || status === "running");
  const work = live.trace.filter((step) => step.kind !== "answer");
  const final = live.trace.find((step) => step.kind === "answer");
  const now = running ? live.live : null;
  const writing = final?.text ?? (now !== null && now.answer !== "" ? now.answer : null);
  return (
    <Turn
      question={question}
      status={
        running ? (
          <span role="status" className="flex items-center gap-1.5 text-xs text-dim">
            <LoaderCircle aria-hidden="true" className="size-3.5 animate-spin" />
            The analyst is working
          </span>
        ) : null
      }
    >
      {(work.length > 0 || now !== null) && (
        <Work steps={work} answered={final !== undefined} live={now} open />
      )}
      {writing !== null && <AnswerText text={writing} streaming={final === undefined} />}
      {status === "failed" && (
        <ErrorNote
          error={new Error(live.state?.error ?? "The analyst failed without saying why.")}
        />
      )}
      <ErrorNote error={result.error} />
    </Turn>
  );
}

interface TurnProps {
  question: string;
  /** Ao lado do nome do analista: se ainda está a trabalhar. */
  status?: ReactNode;
  children?: ReactNode;
}

/** Uma vez da conversa: a pergunta, à direita, e o que o analista fez com ela, à esquerda. */
function Turn({ question, status = null, children }: TurnProps) {
  return (
    <article aria-label={`Question: ${question}`} className="flex flex-col gap-5">
      <p className="max-w-[85%] self-end rounded-2xl rounded-br-md bg-[#262626] px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-strong">
        {question}
      </p>
      <div className="flex gap-3">
        <AnalystAvatar dashed={status !== null} />
        <div className="flex min-w-0 flex-1 flex-col gap-4">
          <p className="flex flex-wrap items-center gap-x-3 gap-y-1 pt-1">
            <span className="text-sm font-medium" style={{ color: AI_AGENT_COLOR }}>
              Analyst
            </span>
            {status}
          </p>
          {children}
        </div>
      </div>
    </article>
  );
}

/** A resposta, em texto corrido, como num chat; a ser escrita, com o cursor a piscar. */
function AnswerText({ text, streaming = false }: { text: string; streaming?: boolean }) {
  return (
    <div className="min-w-0">
      <Prose text={text} className="max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0" />
      {streaming && (
        <span
          aria-hidden="true"
          className="mt-1 inline-block h-3.5 w-1.5 animate-pulse rounded-sm"
          style={{ background: AI_AGENT_COLOR }}
        />
      )}
    </div>
  );
}

/**
 * Por baixo da resposta: o botão para a copiar, o selo dos números verificados e quem respondeu,
 * com quantas ferramentas e em quanto tempo.
 */
function AnswerFooter({ answer, checked }: { answer: AnalystAnswer; checked: Notice | null }) {
  const tools = answer.tools_used;
  return (
    <div className="-mt-1 flex flex-wrap items-center gap-x-3 gap-y-2">
      {answer.answer !== "" && <CopyButton text={answer.answer} />}
      {checked !== null && <CheckedBadge notice={checked} />}
      <p className="font-mono text-xs text-dim sm:ml-auto">
        {answer.model} · {formatCount(tools)} {tools === 1 ? "tool" : "tools"} ·{" "}
        {formatNumber(answer.duration_seconds, 1)} s
      </p>
    </div>
  );
}

/**
 * Copia a resposta tal como o analista a escreveu (em Markdown). Diz que copiou durante uns
 * segundos, no botão e para os leitores de ecrã.
 */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) {
      return;
    }
    const timer = window.setTimeout(() => {
      setCopied(false);
    }, COPIED_MS);
    return () => {
      window.clearTimeout(timer);
    };
  }, [copied]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
    } catch {
      toast.error("The browser didn't let the answer be copied.");
    }
  };
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="xs"
        className="-ml-2 text-dim"
        onClick={() => {
          void copy();
        }}
      >
        {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
        {copied ? "Copied" : "Copy"}
      </Button>
      <span role="status" className="sr-only">
        {copied ? "The answer was copied." : ""}
      </span>
    </>
  );
}

/**
 * O selo dos números verificados: o backend procurou cada número da resposta no que as
 * ferramentas devolveram, e encontrou-os todos. A mensagem dele fica na dica e para os leitores de
 * ecrã.
 */
function CheckedBadge({ notice }: { notice: Notice }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex cursor-default items-center gap-1 rounded-full border border-positive/25 bg-positive/10 px-2 py-0.5 text-xs font-medium text-positive">
          <BadgeCheck aria-hidden="true" className="size-3.5" />
          Numbers checked
          <span className="sr-only">: {notice.message}</span>
        </span>
      </TooltipTrigger>
      <TooltipContent>{notice.message}</TooltipContent>
    </Tooltip>
  );
}

/**
 * Os passos do analista até à resposta, numa linha que abre: as ferramentas que usou à vista e,
 * aberta, o que ele fez, pela ordem (cada ferramenta com o que devolveu e, no fim, a resposta),
 * cada passo com o pensamento que o levou lá fechado por baixo. Ao vivo, vem aberta, com o modelo
 * a pensar no fim.
 */
function Work({
  steps,
  answered,
  live = null,
  open = false,
}: {
  steps: readonly AnalystStep[];
  /** Se a resposta já está escrita: o último pensamento foi o que a preparou. */
  answered: boolean;
  live?: LiveTurn | null;
  open?: boolean;
}) {
  const tools = steps.flatMap((step) => (step.tool === null ? [] : [step.tool]));
  const groups = rounds(steps);
  return (
    <details open={open} className="group/work rounded-2xl border border-[#222] bg-[#121212]">
      <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-2 gap-y-1.5 rounded-2xl px-3.5 py-2.5 text-xs text-soft outline-none hover:text-strong focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
        <ChevronRight
          aria-hidden="true"
          className="size-3.5 shrink-0 transition-transform group-open/work:rotate-90"
        />
        <span className="font-medium">{live === null ? "How it got there" : "Working it out"}</span>
        {tools.map((tool, index) => (
          <span
            key={`${tool.name}-${String(index)}`}
            className={cn(
              "inline-flex items-center gap-1 rounded-full border border-[#2a2a2a] bg-[#181818] px-2 py-0.5 text-dim",
              tool.error !== null && "text-warning",
            )}
          >
            <Wrench aria-hidden="true" className="size-3" />
            {tool.label}
          </span>
        ))}
      </summary>
      <ol
        aria-label="What the analyst did"
        className="relative flex flex-col gap-5 px-3.5 pt-1 pb-4 before:absolute before:top-3 before:bottom-6 before:left-[1.625rem] before:w-px before:bg-[#262626]"
      >
        {groups.map((round, index) => (
          <RoundSteps
            key={round.key}
            round={round}
            answer={answered && index === groups.length - 1}
          />
        ))}
        {live !== null && <LiveStep live={live} />}
      </ol>
    </details>
  );
}

const STEP_ROW = "relative grid grid-cols-[1.5rem_minmax(0,1fr)] gap-3";

/** A que distância do fim (em píxeis) a caixa do raciocínio ainda conta como "no fim". */
const FOLLOW_MARGIN = 24;

/** O raciocínio, na letra do resto, com a entrelinha de ler (as quebras de linha do modelo ficam). */
const THOUGHT_TEXT = "text-sm leading-relaxed whitespace-pre-wrap text-soft";

/**
 * As caixas que deslizam (o raciocínio, o que uma ferramenta devolveu): alcançam-se pelo teclado,
 * com o anel de foco, para se poderem ler sem o rato.
 */
const SCROLLS =
  "overflow-y-auto overscroll-contain outline-none focus-visible:ring-2 focus-visible:ring-ring";

function StepIcon({
  icon: Icon,
  color = AI_AGENT_COLOR,
  dashed = false,
}: {
  icon: typeof Sparkles;
  color?: string;
  dashed?: boolean;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "relative mt-0.5 flex size-6 items-center justify-center rounded-lg border bg-[#141414]",
        dashed && "border-dashed",
      )}
      style={{ borderColor: color }}
    >
      <Icon className={cn("size-3", dashed && "animate-pulse")} style={{ color }} />
    </span>
  );
}

/**
 * Uma volta, pelo que o analista fez: cada ferramenta que pediu (o pensamento vai com a primeira)
 * ou, sem ferramentas, a resposta (ou só o pensamento, se não chegou a responder).
 */
function RoundSteps({ round, answer }: { round: Round; answer: boolean }) {
  if (round.tools.length > 0) {
    return round.tools.map((call, index) => (
      <ToolStep key={call.number} tool={call.tool} thought={index === 0 ? round.thought : null} />
    ));
  }
  if (round.thought === null) {
    return null;
  }
  return answer ? <AnswerStep thought={round.thought} /> : <ThoughtStep thought={round.thought} />;
}

/** Um botão que abre e fecha o que está por baixo de um passo (o pensamento, o que devolveu). */
function Toggle({
  open,
  onToggle,
  controls,
  children,
}: {
  open: boolean;
  onToggle: () => void;
  controls: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-expanded={open}
      aria-controls={controls}
      onClick={onToggle}
      className="inline-flex cursor-pointer items-center gap-1 rounded-sm text-xs text-dim outline-none hover:text-soft focus-visible:ring-2 focus-visible:ring-ring aria-expanded:text-soft"
    >
      <ChevronRight
        aria-hidden="true"
        className={cn("size-3 shrink-0 transition-transform", open && "rotate-90")}
      />
      {children}
    </button>
  );
}

/** O nome de um pensamento fechado: "Thought · 52 words" (o tamanho, em vez do princípio). */
function useThoughtTitle(): (thought: AnalystStep) => string {
  const label = useLabel();
  return (thought) =>
    `${label("AnalystStepKind", "thought")} · ${formatQuantity(wordCount(thought.text), "word", "words")}`;
}

/** O pensamento, aberto: o texto inteiro do modelo, numa caixa que desliza se for comprido. */
function ThoughtText({
  id,
  text,
  name,
  hidden,
}: {
  id: string;
  text: string;
  /** O nome da caixa para os leitores de ecrã. */
  name: string;
  hidden: boolean;
}) {
  return (
    <div
      id={id}
      hidden={hidden}
      role="region"
      aria-label={name}
      tabIndex={0}
      className={cn("mt-2 max-h-72 border-l pl-3", THOUGHT_TEXT, SCROLLS)}
      style={{ borderColor: AI_AGENT_COLOR }}
    >
      {text}
    </div>
  );
}

/** Os argumentos de uma ferramenta, como numa chamada de função: `number=3, first=10`. */
function argumentsText(tool: ToolUse): string {
  return Object.entries(tool.arguments)
    .map(([name, value]) => `${name}=${JSON.stringify(value)}`)
    .join(", ");
}

/**
 * Uma ferramenta usada: o nome, o pedido e o que devolveu numa frase; por baixo, fechados, o
 * pensamento que a pediu e o que o modelo leu (o JSON).
 */
function ToolStep({ tool, thought }: { tool: ToolUse; thought: AnalystStep | null }) {
  const id = useId();
  const thoughtTitle = useThoughtTitle();
  const [showThought, setShowThought] = useState(false);
  const [showResult, setShowResult] = useState(false);
  const refused = tool.error !== null;
  return (
    <li className={STEP_ROW}>
      <StepIcon icon={Wrench} color={refused ? "var(--warning)" : AI_AGENT_COLOR} />
      <div className="min-w-0">
        <p className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-sm">
          <span className="font-medium text-strong">{tool.label}</span>
          <code className="font-mono text-xs break-all text-dim">
            {tool.name}({argumentsText(tool)})
          </code>
        </p>
        <p className={cn("mt-0.5 text-sm", refused ? "text-warning" : "text-foreground/85")}>
          {refused ? `Refused: ${tool.summary}` : tool.summary}
        </p>
        <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1">
          {thought !== null && (
            <Toggle
              open={showThought}
              controls={`${id}-thought`}
              onToggle={() => {
                setShowThought(!showThought);
              }}
            >
              {thoughtTitle(thought)}
            </Toggle>
          )}
          <Toggle
            open={showResult}
            controls={`${id}-result`}
            onToggle={() => {
              setShowResult(!showResult);
            }}
          >
            What it returned
          </Toggle>
        </div>
        {thought !== null && (
          <ThoughtText
            id={`${id}-thought`}
            text={thought.text}
            name={`What the analyst thought before ${tool.label}`}
            hidden={!showThought}
          />
        )}
        <pre
          id={`${id}-result`}
          hidden={!showResult}
          role="region"
          aria-label={`What ${tool.label} returned`}
          tabIndex={0}
          className={cn(
            "mt-2 max-h-80 rounded-xl border border-[#262626] bg-[#0b0b0b] p-3 font-mono text-xs leading-relaxed [overflow-wrap:anywhere] whitespace-pre-wrap text-soft",
            SCROLLS,
          )}
        >
          {tool.result}
        </pre>
      </div>
    </li>
  );
}

/** O último passo: a resposta (escrita por baixo dos passos), com o pensamento que a preparou. */
function AnswerStep({ thought }: { thought: AnalystStep }) {
  const id = useId();
  const label = useLabel();
  const thoughtTitle = useThoughtTitle();
  const [open, setOpen] = useState(false);
  return (
    <li className={STEP_ROW}>
      <StepIcon icon={MessageSquareText} />
      <div className="min-w-0">
        <p className="text-sm font-medium text-strong">{label("AnalystStepKind", "answer")}</p>
        <div className="mt-1.5">
          <Toggle
            open={open}
            controls={id}
            onToggle={() => {
              setOpen(!open);
            }}
          >
            {thoughtTitle(thought)}
          </Toggle>
        </div>
        <ThoughtText
          id={id}
          text={thought.text}
          name="What the analyst thought before answering"
          hidden={!open}
        />
      </div>
    </li>
  );
}

/** Um pensamento sem nada a seguir (o analista parou, ou não chegou a responder). */
function ThoughtStep({ thought }: { thought: AnalystStep }) {
  const id = useId();
  const thoughtTitle = useThoughtTitle();
  const [open, setOpen] = useState(false);
  return (
    <li className={STEP_ROW}>
      <StepIcon icon={Sparkles} />
      <div className="min-w-0 pt-0.5">
        <Toggle
          open={open}
          controls={id}
          onToggle={() => {
            setOpen(!open);
          }}
        >
          {thoughtTitle(thought)}
        </Toggle>
        <ThoughtText id={id} text={thought.text} name="What the analyst thought" hidden={!open} />
      </div>
    </li>
  );
}

/**
 * O modelo a trabalhar, agora: o raciocínio a crescer (a resposta, quando a escreve, aparece por
 * baixo dos passos).
 *
 * A caixa do raciocínio segue o fim do texto enquanto ele cresce, como a conversa dos agentes; quem
 * sobe para ler o princípio deixa de a fazer seguir, até voltar ao fim.
 */
function LiveStep({ live }: { live: LiveTurn }) {
  const writing = live.answer !== "";
  const box = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const length = live.thinking.length;
  useEffect(() => {
    const element = box.current;
    if (element !== null && following.current) {
      element.scrollTop = element.scrollHeight;
    }
  }, [length]);
  return (
    <li className={STEP_ROW} aria-live="polite" aria-busy="true">
      <StepIcon icon={Sparkles} dashed />
      <div className="min-w-0">
        <p className="pt-0.5 text-sm text-soft">{writing ? "Writing the answer" : "Thinking"}</p>
        <div
          ref={box}
          role="region"
          aria-label="What the analyst is thinking"
          tabIndex={0}
          onScroll={(event) => {
            const element = event.currentTarget;
            following.current =
              element.scrollHeight - element.scrollTop - element.clientHeight < FOLLOW_MARGIN;
          }}
          className={cn("mt-1.5 max-h-52 border-l pl-3", THOUGHT_TEXT, SCROLLS)}
          style={{ borderColor: AI_AGENT_COLOR }}
        >
          {live.thinking === "" ? "…" : live.thinking}
          {!writing && (
            <span
              aria-hidden="true"
              className="ml-0.5 inline-block h-3 w-1.5 animate-pulse align-middle"
              style={{ background: AI_AGENT_COLOR }}
            />
          )}
        </div>
      </div>
    </li>
  );
}
