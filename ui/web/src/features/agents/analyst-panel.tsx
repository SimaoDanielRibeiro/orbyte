/**
 * O analista (fase 13, ADR 0053 e 0054): depois de a procura acabar, conversa-se com ele sobre ela.
 *
 * O painel é uma conversa, como num chat: um cartão ao centro da página, com a altura fixa, em que
 * só a conversa desliza (a página fica onde está). Em cima, quem é e o modelo de linguagem; no
 * meio, as perguntas (à direita) e as respostas do analista (à esquerda); em baixo, a caixa da
 * pergunta, com as perguntas sugeridas que ainda não se fizeram. Sem perguntas, o meio dá as
 * boas-vindas: as perguntas sugeridas e as ferramentas que ele usa.
 *
 * O analista é um agente de IA com ferramentas: pensa, pede uma ferramenta (o quadro, um
 * candidato, dois lado a lado, o plano dos dados), lê o que ela devolveu e volta a pensar, até
 * responder. Enquanto trabalha, os passos chegam à vista, com o raciocínio a crescer, e a conversa
 * segue-os (quem sobe para ler fica onde está, com um botão para voltar ao fim). Quando responde,
 * os passos fecham-se numa linha com as ferramentas que usou, e abrem-se para ver cada um, com o
 * JSON que o modelo leu; a resposta fica numa bolha, com a leitura do backend por baixo (os
 * números verificados).
 *
 * Sem o Ollama, o painel mostra o estado de `GET /llm/status` e o que fazer; as respostas que já
 * houver continuam à vista. As perguntas sugeridas, os nomes das ferramentas e os avisos vêm do
 * backend; aqui fica só a apresentação.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { cn } from "cn";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
  LoaderCircle,
  Sparkles,
  Square,
  Wrench,
} from "lucide-react";
import { useReducedMotion } from "motion/react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
  type SyntheticEvent,
} from "react";

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
import { Explain } from "@/components/explain/explain";
import { Prose } from "@/components/markdown/prose";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import { formatCount, formatNumber } from "@/lib/format";
import { useSession } from "@/state/session";

import { AI_AGENT_COLOR } from "./agent-colors";
import { useQuestion } from "./use-question";

/** A que distância do fim (em píxeis) a conversa ainda conta como "no fim", para seguir as novas. */
const BOTTOM_SLACK = 64;

/** A altura máxima da caixa da pergunta, em píxeis: a partir daí, desliza ela. */
const COMPOSER_MAX_HEIGHT = 160;

export function AnalystPanel({ searchId }: { searchId: string }) {
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
  const { scroller, content, detached, onScroll, follow } = useStickToBottom(busy);
  const empty = answers.length === 0 && current === null;

  // Ao abrir, a conversa começa no fim (a última pergunta), como num chat.
  const placed = useRef(false);
  useLayoutEffect(() => {
    if (!placed.current && answers.length > 0) {
      placed.current = true;
      follow(false);
    }
  }, [answers.length, follow]);

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

  const send = (text: string) => {
    const trimmed = text.trim();
    if (trimmed === "" || busy || !ready) {
      return;
    }
    follow(true);
    ask.mutate({ path, body: { question: trimmed } });
  };

  // As sugestões que ainda não se perguntaram ficam por cima da caixa da pergunta.
  const askedAlready = new Set(
    [...answers.map((answer) => answer.question), current?.question ?? ""].map(normalized),
  );
  const suggestions = options.data?.suggestions ?? [];
  const remaining = suggestions.filter((one) => !askedAlready.has(normalized(one)));

  return (
    <section
      aria-labelledby="analyst-title"
      className="mx-auto flex h-[clamp(32rem,78vh,52rem)] w-full max-w-4xl min-w-0 flex-col overflow-hidden rounded-3xl border border-[#262626] bg-[#0f0f0f] shadow-2xl shadow-black/40"
    >
      <PanelHeader model={options.data?.model ?? null} ready={ready} />

      <div className="relative min-h-0 flex-1">
        <div
          ref={scroller}
          onScroll={onScroll}
          role="region"
          aria-label="The conversation with the analyst"
          tabIndex={0}
          className="h-full overflow-y-auto overscroll-contain outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset"
        >
          <div ref={content} className="mx-auto flex max-w-3xl flex-col gap-10 px-4 py-8 sm:px-8">
            {empty ? (
              <Welcome options={options.data} disabled={!ready || busy} onAsk={send} />
            ) : (
              <>
                {answers.map((answer) => (
                  <AnswerTurn key={answer.question_id} answer={answer} />
                ))}
                {current !== null && !answered && (
                  <LiveQuestionTurn
                    key={current.questionId}
                    questionId={current.questionId}
                    question={current.question}
                    onAnswered={onAnswered}
                    onFailed={onFailed}
                  />
                )}
              </>
            )}
          </div>
        </div>
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

      <footer className="shrink-0 border-t border-[#222] px-4 pt-3 pb-4 sm:px-8">
        <div className="mx-auto flex max-w-3xl flex-col gap-3">
          {llm.data !== undefined && !ready && (
            <div role="status" aria-label="The analyst's language model">
              <NoticeList notices={llm.data.notices} />
            </div>
          )}
          {!empty && remaining.length > 0 && (
            <SuggestionRow suggestions={remaining} disabled={!ready || busy} onAsk={send} />
          )}
          <Composer
            value={question}
            onChange={setQuestion}
            onSend={() => {
              send(question);
            }}
            disabled={!ready || busy}
            canSend={ready && !busy && question.trim() !== ""}
            sending={ask.isPending}
            stop={
              working ? (
                <Button
                  type="button"
                  variant="outline"
                  className="rounded-xl"
                  disabled={cancel.isPending}
                  onClick={() => {
                    cancel.mutate({ path: { question_id: current.questionId } });
                  }}
                >
                  <Square aria-hidden="true" className="fill-current" />
                  Stop
                </Button>
              ) : null
            }
          />
          <ErrorNote error={ask.error ?? cancel.error} />
        </div>
      </footer>
    </section>
  );
}

/** As perguntas comparam-se sem maiúsculas nem espaços à volta. */
function normalized(text: string): string {
  return text.trim().toLowerCase();
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

/** Em cima: o nome do painel, o que é, e o modelo de linguagem com o seu estado. */
function PanelHeader({ model, ready }: { model: string | null; ready: boolean }) {
  return (
    <header className="flex shrink-0 items-center gap-3 border-b border-[#222] px-4 py-3 sm:px-5">
      <span
        aria-hidden="true"
        className="flex size-9 shrink-0 items-center justify-center rounded-xl border bg-[#161616]"
        style={{ borderColor: AI_AGENT_COLOR }}
      >
        <Sparkles className="size-4" style={{ color: AI_AGENT_COLOR }} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1">
          <h2
            id="analyst-title"
            className="truncate text-base font-medium tracking-normal text-strong"
          >
            Ask the analyst
          </h2>
          <Explain topic="analyst" />
        </div>
        <p className="truncate text-xs text-dim">
          An AI agent that reads what this search measured, step by step
        </p>
      </div>
      {model !== null && (
        <span className="flex shrink-0 items-center gap-2 rounded-full border border-[#2a2a2a] px-2.5 py-1 font-mono text-xs text-soft">
          <span
            aria-hidden="true"
            className={cn("size-1.5 rounded-full", ready ? "bg-positive" : "bg-[#6b6b6b]")}
          />
          {model}
          <span className="sr-only">{ready ? ", ready" : ", not running"}</span>
        </span>
      )}
    </header>
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
  stop: ReactNode;
}

/**
 * A caixa da pergunta, em baixo: cresce com o texto (até `COMPOSER_MAX_HEIGHT`), Enter pergunta e
 * Shift+Enter muda de linha.
 */
function Composer({ value, onChange, onSend, disabled, canSend, sending, stop }: ComposerProps) {
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
      {stop ?? (
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

/** Uma pergunta já respondida: a pergunta, os passos (fechados), a resposta e a leitura dela. */
function AnswerTurn({ answer }: { answer: AnalystAnswer }) {
  const tools = answer.tools_used;
  const work = answer.trace.filter((step) => step.kind !== "answer");
  return (
    <Turn question={answer.question}>
      {work.length > 0 && <Work steps={work} />}
      {answer.answer !== "" && <AnswerBubble text={answer.answer} />}
      <Reading notices={answer.notices} />
      <p className="font-mono text-xs text-dim">
        {answer.model} · {formatCount(tools)} {tools === 1 ? "tool" : "tools"} ·{" "}
        {formatNumber(answer.duration_seconds, 1)} s
      </p>
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
      {(work.length > 0 || now !== null) && <Work steps={work} live={now} open />}
      {writing !== null && <AnswerBubble text={writing} streaming={final === undefined} />}
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
        <div className="flex min-w-0 flex-1 flex-col gap-3">
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

/** A resposta, numa bolha; a ser escrita, com o cursor a piscar. */
function AnswerBubble({ text, streaming = false }: { text: string; streaming?: boolean }) {
  return (
    <div className="rounded-2xl rounded-tl-md border border-[#262626] bg-[#171717] px-4 py-3 sm:px-5">
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

/** A leitura do backend por baixo da resposta (os números verificados, uma resposta forçada…). */
function Reading({ notices }: { notices: readonly Notice[] }) {
  if (notices.length === 0) {
    return null;
  }
  return <NoticeList notices={notices} className="gap-2 px-1" />;
}

/**
 * Os passos do analista até à resposta, numa linha que abre: as ferramentas que usou à vista e,
 * aberta, cada passo pela ordem (o raciocínio, cada ferramenta com o que devolveu). Ao vivo, vem
 * aberta, com o modelo a pensar no fim.
 */
function Work({
  steps,
  live = null,
  open = false,
}: {
  steps: readonly AnalystStep[];
  live?: LiveTurn | null;
  open?: boolean;
}) {
  const tools = steps.flatMap((step) => (step.tool === null ? [] : [step.tool]));
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
        className="relative flex flex-col gap-4 px-3.5 pt-1 pb-4 before:absolute before:top-3 before:bottom-6 before:left-[1.625rem] before:w-px before:bg-[#262626]"
      >
        {steps.map((step) => (
          <Step key={step.number} step={step} />
        ))}
        {live !== null && <LiveStep live={live} />}
      </ol>
    </details>
  );
}

const STEP_ROW = "relative grid grid-cols-[1.5rem_minmax(0,1fr)] gap-3";

/** A que distância do fim (em píxeis) a caixa do raciocínio ainda conta como "no fim". */
const FOLLOW_MARGIN = 24;

/** O raciocínio, em letra de máquina, como na conversa dos agentes. */
const THOUGHT_TEXT = "font-mono text-xs leading-relaxed whitespace-pre-wrap text-soft";

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

/** Um passo: o raciocínio (fechado, com o princípio à vista) ou uma ferramenta. */
function Step({ step }: { step: AnalystStep }) {
  if (step.kind === "tool" && step.tool !== null) {
    return <ToolStep tool={step.tool} />;
  }
  return (
    <li className={STEP_ROW}>
      <StepIcon icon={Sparkles} />
      <details className="group min-w-0">
        <summary className="flex cursor-pointer list-none items-center gap-2 pt-1 font-mono text-xs text-dim hover:text-soft [&::-webkit-details-marker]:hidden">
          <span className="min-w-0 flex-1 truncate">
            <span className="text-soft">Thought</span> · {step.text}
          </span>
          <ChevronDown
            aria-hidden="true"
            className="size-3.5 shrink-0 transition-transform group-open:rotate-180"
          />
        </summary>
        <div
          role="region"
          aria-label="The analyst's thought"
          tabIndex={0}
          className={cn("mt-2 max-h-72 border-l pl-3", THOUGHT_TEXT, SCROLLS)}
          style={{ borderColor: AI_AGENT_COLOR }}
        >
          {step.text}
        </div>
      </details>
    </li>
  );
}

/** Os argumentos de uma ferramenta, como numa chamada de função: `number=3, first=10`. */
function argumentsText(tool: ToolUse): string {
  return Object.entries(tool.arguments)
    .map(([name, value]) => `${name}=${JSON.stringify(value)}`)
    .join(", ");
}

/** Uma ferramenta usada: o nome, o pedido, o que devolveu numa frase e, aberto, o que o modelo leu. */
function ToolStep({ tool }: { tool: ToolUse }) {
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
        <details className="group mt-1.5">
          <summary className="inline-flex cursor-pointer list-none items-center gap-1 rounded-sm text-xs text-dim outline-none hover:text-soft focus-visible:ring-2 focus-visible:ring-ring [&::-webkit-details-marker]:hidden">
            <ChevronRight
              aria-hidden="true"
              className="size-3 shrink-0 transition-transform group-open:rotate-90"
            />
            What it returned
          </summary>
          <pre
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
        </details>
      </div>
    </li>
  );
}

/**
 * O modelo a trabalhar, agora: o raciocínio a crescer (a resposta, quando a escreve, vai para a
 * bolha da resposta).
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
        <p className="pt-0.5 text-xs text-soft">{writing ? "Writing the answer" : "Thinking"}</p>
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
