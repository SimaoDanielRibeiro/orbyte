/**
 * A conversa dos agentes em ecrã inteiro, como um chat: o que cada agente disse, a quem e por
 * que ordem, com o resultado de cada candidato dentro da resposta de quem o mediu. Serve para se
 * perceber como a procura aconteceu, de uma ponta à outra.
 *
 * - O coordenador fala à direita (é ele que dá a vez e decide); os outros agentes à esquerda,
 *   cada um com a cor e o ícone do seu nó no fluxograma. Mensagens seguidas do mesmo agente
 *   juntam-se, como num chat.
 * - Os pedidos curtos do coordenador ("é a tua vez") ficam numa linha pequena, e podem
 *   esconder-se; o raciocínio do agente de IA abre e fecha.
 * - À esquerda, a linha do tempo das trocas: uma marca por mensagem, na cor de quem a disse,
 *   como a do histórico das trocas, mas na vertical. Mostra as fases, a janela da parte que se
 *   está a ler, e leva a qualquer momento (clicar, arrastar ou as setas). Por baixo, quem falou.
 * - Ao vivo, segue a última mensagem enquanto se está no fim; quem sobe para ler fica onde está,
 *   com um botão para voltar ao fim.
 *
 * É só apresentação: as mensagens, os candidatos e as regras (quem liderava, quem decide) vêm do
 * backend, como no resto do ecrã.
 */
import { cn } from "cn";
import {
  ArrowDown,
  ChevronDown,
  Maximize,
  MessagesSquare,
  Minimize,
  Sparkles,
  X,
} from "lucide-react";
import { useReducedMotion } from "motion/react";
import { Dialog } from "radix-ui";
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type RefObject,
} from "react";

import type {
  AgentMessage,
  Candidate,
  MessageKind,
  MetricResult,
  ObjectiveOption,
  SearchPhase,
  Thought,
} from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Prose } from "@/components/markdown/prose";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { formatCount, formatMetric } from "@/lib/format";

import { AI_AGENT_COLOR, messageColor, ROLE_ICONS } from "./agent-colors";
import { AiMark } from "./search-parts";
import { changesText, leaderNumbers, objectiveValue, recipientsText } from "./values";

/** O comprimento de cada marca na linha do tempo: o que decide mais, mais comprido. */
const TICK_WIDTH: Readonly<Record<MessageKind, string>> = {
  report: "55%",
  announcement: "75%",
  request: "35%",
  reply: "55%",
  decision: "95%",
  test: "95%",
  thought: "45%",
};

/** A que distância do fim (em píxeis) ainda se conta como "no fim", para seguir as novas. */
const BOTTOM_SLACK = 72;
/** Com mais mensagens do que isto, as marcas da linha do tempo encostam-se (sem espaço entre elas). */
const SPACED_TICKS = 160;

interface ConversationRoomProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** O nome da procura ("Predict Survived in Titanic"). */
  title: string;
  messages: readonly AgentMessage[];
  candidates: readonly Candidate[];
  objective: ObjectiveOption | null;
  /** O que o agente de IA está a pensar agora (só ao vivo). */
  thought: Thought | null;
  /** Se a procura ainda está a correr. */
  live: boolean;
  /** No fim: a nota do vencedor no teste. */
  test: MetricResult | null;
}

export function ConversationRoom(props: ConversationRoomProps) {
  const { open, onOpenChange } = props;
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/70 duration-150 supports-backdrop-filter:backdrop-blur-sm data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        <Dialog.Content className="fixed inset-0 z-50 flex flex-col bg-[#0f0f0f] text-sm duration-200 outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-[0.985] data-closed:animate-out data-closed:fade-out-0">
          <Room {...props} />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

interface Indexed {
  message: AgentMessage;
  /** A posição na conversa toda (com os pedidos), a mesma da linha do tempo. */
  index: number;
}

interface PhaseGroup {
  key: string;
  phase: SearchPhase;
  items: Indexed[];
}

function Room({
  title,
  messages,
  candidates,
  objective,
  thought,
  live,
  test,
}: ConversationRoomProps) {
  const label = useLabel();
  const reduced = useReducedMotion() ?? false;
  const switchId = useId();
  const frame = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const [showRequests, setShowRequests] = useState(true);
  const [visible, setVisible] = useState<readonly [number, number]>([0, 0]);
  const [flash, setFlash] = useState<number | null>(null);
  /** A última mensagem para onde se saltou: é daí que as setas e "quem falou" continuam. */
  const [target, setTarget] = useState<number | null>(null);
  /** Se se está a seguir as mensagens novas (ao vivo); só quem lê, ao subir, deixa de seguir. */
  const following = useRef(live);
  const [detached, setDetached] = useState(!live);
  const lastTop = useRef(0);
  const fullscreen = useFullscreen(frame);

  const leaders = useMemo(() => leaderNumbers(candidates, objective), [candidates, objective]);
  const byNumber = useMemo(
    () => new Map(candidates.map((candidate) => [candidate.number, candidate])),
    [candidates],
  );
  const groups = useMemo(() => {
    const result: PhaseGroup[] = [];
    messages.forEach((message, index) => {
      if (!showRequests && message.kind === "request") {
        return;
      }
      const last = result.at(-1);
      if (last?.phase === message.phase) {
        last.items.push({ message, index });
      } else {
        result.push({
          key: `${message.phase}-${String(index)}`,
          phase: message.phase,
          items: [{ message, index }],
        });
      }
    });
    return result;
  }, [messages, showRequests]);

  // Que mensagens estão à vista (para a janela da linha do tempo) e se se está no fim.
  const measure = useCallback(() => {
    const element = scroller.current;
    if (element === null) {
      return;
    }
    const box = element.getBoundingClientRect();
    let first = -1;
    let last = -1;
    for (const item of element.querySelectorAll<HTMLElement>("[data-index]")) {
      const rect = item.getBoundingClientRect();
      if (rect.bottom < box.top + 8 || rect.top > box.bottom - 8) {
        continue;
      }
      const index = Number(item.dataset.index);
      first = first === -1 ? index : Math.min(first, index);
      last = Math.max(last, index);
    }
    if (first !== -1) {
      setVisible((previous) =>
        previous[0] === first && previous[1] === last ? previous : [first, last],
      );
    }
    // Chegar ao fim volta a seguir; só subir deixa de seguir. O deslizar do próprio ecrã até ao
    // fim só desce: a meio dele ainda não se está no fim, mas não se deixa de seguir (senão uma
    // mensagem que chegasse nesse instante ficava por seguir).
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
    const element = scroller.current;
    if (element === null) {
      return;
    }
    let request = 0;
    const onScroll = () => {
      cancelAnimationFrame(request);
      request = requestAnimationFrame(measure);
    };
    element.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      cancelAnimationFrame(request);
      element.removeEventListener("scroll", onScroll);
    };
  }, [measure]);

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

  // Ao abrir: ao vivo, no fim (a seguir as novas); no fim da procura, no princípio, para a ler.
  // Só uma vez: depois, quem decide onde se está é quem lê.
  const placed = useRef(false);
  useEffect(() => {
    if (placed.current) {
      return;
    }
    placed.current = true;
    if (live) {
      toBottom(false);
    }
    measure();
  }, [live, toBottom, measure]);

  // Ao vivo: as mensagens novas (e o pensamento a crescer) seguem-se, se se estava no fim.
  const thinking = thought?.text.length ?? -1;
  useEffect(() => {
    if (following.current) {
      toBottom(true);
    }
    measure();
  }, [messages.length, thinking, showRequests, toBottom, measure]);

  // Ir a uma mensagem (da linha do tempo, ou de quem falou): fica a um terço do topo, e acende-se.
  const jump = useCallback(
    (target: number) => {
      const element = scroller.current;
      if (element === null || messages.length === 0) {
        return;
      }
      const index = Math.max(0, Math.min(target, messages.length - 1));
      // Um pedido escondido não está na conversa: vai-se ao mais perto que está.
      let item: HTMLElement | null = null;
      for (let step = 0; item === null && step < messages.length; step += 1) {
        item =
          element.querySelector<HTMLElement>(`[data-index="${String(index + step)}"]`) ??
          element.querySelector<HTMLElement>(`[data-index="${String(index - step)}"]`);
      }
      if (item === null) {
        return;
      }
      const top =
        element.scrollTop +
        item.getBoundingClientRect().top -
        element.getBoundingClientRect().top -
        element.clientHeight / 3;
      following.current = false;
      element.scrollTo({ top, behavior: reduced ? "auto" : "smooth" });
      setFlash(Number(item.dataset.index));
      setTarget(Number(item.dataset.index));
    },
    [messages.length, reduced],
  );

  useEffect(() => {
    if (flash === null) {
      return;
    }
    const timer = window.setTimeout(() => {
      setFlash(null);
    }, 1400);
    return () => {
      window.clearTimeout(timer);
    };
  }, [flash]);

  const [first, last] = visible;
  // Onde se está a ler: a mensagem para onde se saltou, enquanto estiver à vista; senão, a
  // primeira à vista. O salto deixa a mensagem a um terço do topo, com outras por cima: contar a
  // partir da primeira à vista fazia a seta para baixo andar para trás.
  const at = target !== null && target >= first && target <= last ? target : first;
  return (
    <div ref={frame} className="flex h-full min-h-0 flex-col bg-[#0f0f0f]">
      <header className="flex h-16 shrink-0 items-center gap-4 border-b border-[#222] px-5">
        <span
          aria-hidden="true"
          className="flex size-9 items-center justify-center rounded-xl border border-[#2a2a2a] bg-[#161616]"
        >
          <MessagesSquare className="size-[1.125rem] text-soft" />
        </span>
        <div className="min-w-0">
          <Dialog.Title className="truncate text-base font-medium tracking-normal text-strong">
            Agent conversation
          </Dialog.Title>
          <Dialog.Description className="truncate text-xs text-dim">
            {title} · {formatCount(messages.length)} messages
          </Dialog.Description>
        </div>
        <Status live={live} />
        <div className="ml-auto flex items-center gap-2 sm:gap-4">
          <div className="hidden items-center gap-2 sm:flex">
            <Switch id={switchId} checked={showRequests} onCheckedChange={setShowRequests} />
            <Label htmlFor={switchId} className="text-sm font-normal text-soft">
              Turn requests
            </Label>
          </div>
          {fullscreen.supported && (
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={fullscreen.active ? "Exit full screen" : "Full screen"}
              onClick={fullscreen.toggle}
            >
              {fullscreen.active ? (
                <Minimize aria-hidden="true" />
              ) : (
                <Maximize aria-hidden="true" />
              )}
            </Button>
          )}
          <Dialog.Close asChild>
            <Button variant="outline" size="icon-sm" aria-label="Close the conversation">
              <X aria-hidden="true" />
            </Button>
          </Dialog.Close>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside
          aria-label="Timeline"
          className="hidden w-[19rem] shrink-0 flex-col gap-6 border-r border-[#222] px-5 py-6 md:flex"
        >
          <div>
            <p className="small-label">Timeline</p>
            <p className="mt-1.5 font-mono text-xs text-soft tabular-nums">
              {messages.length === 0
                ? "No messages yet"
                : `Messages ${formatCount(first + 1)}–${formatCount(last + 1)} of ${formatCount(messages.length)}`}
            </p>
          </div>
          {messages.length > 0 && (
            <TimelineRail
              messages={messages}
              visible={visible}
              at={at}
              onJump={jump}
              label={label}
            />
          )}
          <Speakers messages={messages} from={at} onJump={jump} />
        </aside>

        <div ref={scroller} className="relative min-h-0 flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <p className="mt-24 text-center text-sm text-dim">
              The agents haven&apos;t said anything yet.
            </p>
          ) : (
            <ol
              aria-label="Messages"
              className="mx-auto flex max-w-3xl flex-col px-4 pt-2 pb-28 sm:px-8"
            >
              {groups.map((group) => (
                <li key={group.key} className="flex flex-col">
                  {/* A fase: fica presa em cima enquanto se lê dentro dela. */}
                  <div className="sticky top-0 z-10 -mx-4 flex justify-center bg-[#0f0f0f]/85 px-4 py-3 backdrop-blur sm:-mx-8">
                    <span className="rounded-full border border-[#2a2a2a] bg-[#151515] px-3 py-1 text-xs font-medium text-soft">
                      {label("SearchPhase", group.phase)}
                    </span>
                  </div>
                  <ol className="flex flex-col">
                    {group.items.map((item, position) => (
                      <Entry
                        key={item.message.number}
                        item={item}
                        previous={group.items[position - 1]?.message ?? null}
                        candidate={
                          item.message.kind === "reply" && item.message.candidate !== null
                            ? (byNumber.get(item.message.candidate) ?? null)
                            : null
                        }
                        leader={
                          item.message.candidate !== null && leaders.has(item.message.candidate)
                        }
                        objective={objective}
                        test={test}
                        flash={flash === item.index}
                        label={label}
                      />
                    ))}
                  </ol>
                </li>
              ))}
              {thought !== null && <Thinking thought={thought} />}
            </ol>
          )}
          {live && detached && messages.length > 0 && (
            <div className="pointer-events-none sticky bottom-6 flex justify-center">
              <Button
                size="sm"
                className="pointer-events-auto shadow-lg shadow-black/50"
                onClick={() => {
                  following.current = true;
                  toBottom(true);
                }}
              >
                <ArrowDown aria-hidden="true" />
                Latest message
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Se a procura ainda está a correr (com um ponto a pulsar) ou se já acabou. */
function Status({ live }: { live: boolean }) {
  return (
    <span className="hidden items-center gap-2 rounded-full border border-[#2a2a2a] px-2.5 py-1 text-xs font-medium text-soft sm:flex">
      <span
        aria-hidden="true"
        className={cn("size-1.5 rounded-full", live ? "animate-pulse bg-positive" : "bg-[#6b6b6b]")}
      />
      {live ? "Live" : "Finished"}
    </span>
  );
}

interface EntryProps {
  item: Indexed;
  /** A mensagem de antes, na mesma fase (para juntar as do mesmo agente). */
  previous: AgentMessage | null;
  /** O candidato que esta resposta mediu, se for uma resposta. */
  candidate: Candidate | null;
  /** Se esse candidato passou a ser o melhor até ali. */
  leader: boolean;
  objective: ObjectiveOption | null;
  test: MetricResult | null;
  /** Se acabou de se saltar para aqui (acende-se um momento). */
  flash: boolean;
  label: ReturnType<typeof useLabel>;
}

/** Uma mensagem da conversa, na forma do seu tipo. */
function Entry({ item, previous, candidate, leader, objective, test, flash, label }: EntryProps) {
  const { message, index } = item;
  const color = messageColor(message);
  const recipients = recipientsText(message.recipients, label);
  const ring = flash ? "ring-2 ring-white/40" : undefined;

  if (message.kind === "request") {
    // O pedido diz a quem e porquê (ADR 0049), e por isso lê-se inteiro, em vez de se cortar
    // numa linha. O texto já começa pelo nome de quem recebe: o "→" basta.
    return (
      <li data-index={index} className="flex justify-end py-1 lazy-paint">
        <p
          className={cn(
            "max-w-[80%] rounded-2xl border border-[#232323] bg-[#131313] px-3.5 py-2 text-xs leading-snug text-soft transition-shadow",
            ring,
          )}
        >
          <span aria-hidden="true" className="mr-1.5 text-dim">
            →
          </span>
          {message.text}
        </p>
      </li>
    );
  }

  if (message.kind === "test") {
    return (
      <li
        data-index={index}
        className={cn(
          "my-6 w-full max-w-md self-center rounded-2xl border border-[#3a3a3a] bg-[#171717] px-6 py-5 text-center transition-shadow",
          ring,
        )}
      >
        <p className="text-sm font-medium text-soft">The test</p>
        {test !== null && (
          <>
            <p className="mt-2 text-4xl font-semibold tracking-[-0.03em] text-strong tabular-nums">
              {formatMetric(test)}
            </p>
            <p className="mt-1 text-xs text-dim">{test.display_name} on data no agent saw</p>
          </>
        )}
        <Prose text={message.text} size="small" className="mt-3 text-left text-soft" />
      </li>
    );
  }

  // Junta-se à de antes se for do mesmo agente, sem pedidos nem pensamentos pelo meio.
  const joined =
    previous !== null &&
    previous.author === message.author &&
    previous.kind !== "request" &&
    previous.kind !== "thought" &&
    message.kind !== "decision";
  const right = message.role === "coordinator";
  const Icon = ROLE_ICONS[message.role];

  if (message.kind === "thought") {
    const firstLine = message.text.split("\n").find((line) => line.trim() !== "") ?? "";
    return (
      <li
        data-index={index}
        className={cn("flex max-w-[88%] gap-3 self-start lazy-paint", joined ? "mt-1.5" : "mt-5")}
      >
        <Avatar icon={Sparkles} color={AI_AGENT_COLOR} dashed hidden={joined} offset={false} />
        <details
          className={cn(
            "group min-w-0 flex-1 rounded-2xl rounded-tl-md border border-dashed bg-[#141414] px-4 py-2.5 transition-shadow",
            ring,
          )}
          style={{ borderColor: `${AI_AGENT_COLOR}66` }}
        >
          <summary className="flex cursor-pointer list-none items-center gap-2 font-mono text-xs">
            <span style={{ color: AI_AGENT_COLOR }}>{label("MessageKind", "thought")}</span>
            <span className="min-w-0 flex-1 truncate text-dim">{firstLine}</span>
            <ChevronDown
              aria-hidden="true"
              className="size-3.5 shrink-0 text-dim transition-transform group-open:rotate-180"
            />
          </summary>
          <p className="mt-3 max-h-[28rem] overflow-y-auto font-mono text-xs leading-relaxed whitespace-pre-wrap text-soft">
            {message.text}
          </p>
        </details>
      </li>
    );
  }

  const decision = message.kind === "decision";
  return (
    <li
      data-index={index}
      className={cn(
        "flex max-w-[88%] gap-3 lazy-paint",
        right ? "flex-row-reverse self-end" : "self-start",
        joined ? "mt-1.5" : "mt-5",
      )}
    >
      <Avatar icon={Icon} color={color} hidden={joined} />
      <div className={cn("flex min-w-0 flex-col gap-1.5", right && "items-end")}>
        {!joined && (
          <p
            className={cn("flex flex-wrap items-baseline gap-x-2 text-xs", right && "justify-end")}
          >
            <span className="text-sm font-medium" style={{ color }}>
              {message.author}
            </span>
            {recipients !== null && <span className="text-dim">to {recipients}</span>}
          </p>
        )}
        <div
          className={cn(
            "rounded-2xl border px-4 py-2.5 transition-shadow",
            right
              ? "rounded-tr-md border-[#2c2c2c] bg-[#1b1b1b]"
              : "rounded-tl-md border-[#242424] bg-[#161616]",
            decision && "border-[#5a5a5a] bg-[#212121]",
            ring,
          )}
        >
          {decision && (
            <p className="mb-1.5 text-xs font-medium text-strong">
              {label("MessageKind", "decision")}
            </p>
          )}
          <Prose text={message.text} size="small" className="text-foreground/90" />
          {candidate !== null && (
            <ResultChip candidate={candidate} objective={objective} leader={leader} />
          )}
        </div>
        <span className="font-mono text-[12px] text-dim tabular-nums">
          #{formatCount(message.number)} · {label("MessageKind", message.kind)}
        </span>
      </div>
    </li>
  );
}

/** Quem fala: o ícone numa caixa com o contorno da cor dele (a do nó no fluxograma). */
function Avatar({
  icon: Icon,
  color,
  dashed = false,
  hidden,
  offset = true,
}: {
  icon: (typeof ROLE_ICONS)[keyof typeof ROLE_ICONS];
  color: string;
  dashed?: boolean;
  /** Numa mensagem seguida do mesmo agente, fica só o espaço (como num chat). */
  hidden: boolean;
  /** Se há a linha do nome por cima da bolha: o ícone desce para ficar ao lado da bolha. */
  offset?: boolean;
}) {
  if (hidden) {
    return <span aria-hidden="true" className="w-8 shrink-0" />;
  }
  return (
    <span
      aria-hidden="true"
      className={cn(
        "flex size-8 shrink-0 items-center justify-center rounded-xl border bg-[#181818]",
        offset && "mt-[1.375rem]",
        dashed && "border-dashed",
      )}
      style={{ borderColor: color }}
    >
      <Icon className="size-4" style={{ color }} />
    </span>
  );
}

/** O resultado do candidato que uma resposta mediu: o modelo, o que mudou e as notas. */
function ResultChip({
  candidate,
  objective,
  leader,
}: {
  candidate: Candidate;
  objective: ObjectiveOption | null;
  leader: boolean;
}) {
  return (
    <div className="mt-2.5 flex flex-col gap-1 rounded-xl border border-[#262626] bg-[#111] px-3 py-2 font-mono text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="text-soft">
          #{formatCount(candidate.number)} {candidate.model_display_name}
        </span>
        {candidate.error === null ? (
          <>
            <span className="text-strong tabular-nums">
              {objectiveValue(candidate.mean, objective)} ±{" "}
              {objectiveValue(candidate.std, objective)}
            </span>
            <span className="text-dim tabular-nums">
              train {objectiveValue(candidate.train_mean, objective)}
            </span>
          </>
        ) : (
          <span className="text-negative">failed</span>
        )}
        {leader && (
          <span className="rounded-full bg-now/15 px-2 py-px text-[12px] text-now">new best</span>
        )}
        {candidate.ai && <AiMark />}
      </div>
      <span className="truncate text-dim">{changesText(candidate)}</span>
    </div>
  );
}

/** O agente de IA a pensar, agora: uma bolha aberta, a crescer, com o cursor a piscar. */
function Thinking({ thought }: { thought: Thought }) {
  return (
    <li className="mt-5 flex max-w-[88%] gap-3 self-start" aria-live="polite" aria-busy="true">
      <Avatar icon={Sparkles} color={AI_AGENT_COLOR} dashed hidden={false} />
      <div className="flex min-w-0 flex-col gap-1.5">
        <p className="text-xs">
          <span className="text-sm font-medium" style={{ color: AI_AGENT_COLOR }}>
            {thought.author}
          </span>
          <span className="ml-2 text-dim">
            is thinking about candidate #{formatCount(thought.candidate)}
          </span>
        </p>
        <div
          className="max-h-80 overflow-y-auto rounded-2xl rounded-tl-md border border-dashed bg-[#141414] px-4 py-2.5 font-mono text-xs leading-relaxed whitespace-pre-wrap text-soft"
          style={{ borderColor: `${AI_AGENT_COLOR}66` }}
        >
          {thought.text === "" ? "…" : thought.text}
          <span
            aria-hidden="true"
            className="ml-0.5 inline-block h-3 w-1.5 animate-pulse align-middle"
            style={{ background: AI_AGENT_COLOR }}
          />
        </div>
      </div>
    </li>
  );
}

/**
 * A linha do tempo, na vertical: uma marca por mensagem, na cor de quem a disse (mais comprida
 * quando decide mais), as fases à esquerda, e a janela da parte que se está a ler. É um cursor:
 * clica-se ou arrasta-se, e também anda com as setas, o Page Up/Down, o Home e o End.
 */
function TimelineRail({
  messages,
  visible,
  at,
  onJump,
  label,
}: {
  messages: readonly AgentMessage[];
  visible: readonly [number, number];
  /** A mensagem que se está a ler: as teclas andam a partir dela. */
  at: number;
  onJump: (index: number) => void;
  label: ReturnType<typeof useLabel>;
}) {
  const strip = useRef<HTMLDivElement>(null);
  const count = messages.length;
  const [first, last] = visible;
  const current = messages[at];
  const pick = (clientY: number) => {
    const box = strip.current?.getBoundingClientRect();
    if (box === undefined || box.height === 0) {
      return;
    }
    onJump(Math.floor(((clientY - box.top) / box.height) * count));
  };
  const keys: Readonly<Record<string, number>> = {
    ArrowUp: at - 1,
    ArrowLeft: at - 1,
    ArrowDown: at + 1,
    ArrowRight: at + 1,
    PageUp: at - 10,
    PageDown: at + 10,
    Home: 0,
    End: count - 1,
  };
  // As fases por ordem, cada uma com quantas mensagens tem (a altura do seu troço).
  const phases: { phase: SearchPhase; size: number }[] = [];
  for (const message of messages) {
    const previous = phases.at(-1);
    if (previous?.phase === message.phase) {
      previous.size += 1;
    } else {
      phases.push({ phase: message.phase, size: 1 });
    }
  }
  const spaced = count <= SPACED_TICKS;
  return (
    <div className="flex min-h-0 flex-1 gap-3">
      <div
        aria-hidden="true"
        className={cn("flex w-[5.5rem] shrink-0 flex-col", spaced && "gap-px")}
      >
        {phases.map(({ phase, size }, index) => (
          <div
            key={`${phase}-${String(index)}`}
            className="min-h-0 overflow-hidden border-t border-[#2e2e2e] pt-1 text-[12px] leading-tight text-dim"
            style={{ flex: size }}
          >
            <span className="block truncate">{label("SearchPhase", phase)}</span>
          </div>
        ))}
      </div>
      <div
        ref={strip}
        role="slider"
        tabIndex={0}
        aria-orientation="vertical"
        aria-label="The moment of the search"
        aria-valuemin={1}
        aria-valuemax={count}
        aria-valuenow={at + 1}
        aria-valuetext={
          current === undefined
            ? undefined
            : `Message ${String(at + 1)} of ${String(count)}: ${current.author}, ${label("MessageKind", current.kind).toLowerCase()}`
        }
        onKeyDown={(event: KeyboardEvent<HTMLDivElement>) => {
          const target = keys[event.key];
          if (target !== undefined) {
            event.preventDefault();
            onJump(target);
          }
        }}
        onPointerDown={(event: PointerEvent<HTMLDivElement>) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          pick(event.clientY);
        }}
        onPointerMove={(event: PointerEvent<HTMLDivElement>) => {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) {
            pick(event.clientY);
          }
        }}
        className={cn(
          "relative flex min-h-0 flex-1 cursor-pointer touch-none flex-col rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-4 focus-visible:ring-offset-[#0f0f0f]",
          spaced && "gap-px",
        )}
      >
        {messages.map((message, index) => {
          const color = messageColor(message);
          const inView = index >= first && index <= last;
          return (
            <span
              key={message.number}
              aria-hidden="true"
              className="min-h-0 flex-1 rounded-[2px] transition-opacity duration-200"
              style={{
                width: TICK_WIDTH[message.kind],
                background: color,
                opacity: inView ? 1 : 0.28,
                boxShadow: inView && message.kind === "decision" ? `0 0 10px ${color}` : undefined,
              }}
            />
          );
        })}
        {/* A janela: a parte da conversa que está à vista. */}
        <span
          aria-hidden="true"
          className="pointer-events-none absolute -inset-x-1.5 rounded-md border border-white/25 bg-white/[0.04] transition-[top,height] duration-150"
          style={{
            top: `${String((first / count) * 100)}%`,
            height: `${String(((last - first + 1) / count) * 100)}%`,
          }}
        />
      </div>
    </div>
  );
}

/**
 * Quem falou, cada um na sua cor, com quantas mensagens disse. Clicar num agente leva à
 * próxima mensagem dele (depois da que está à vista), para se seguir um agente pela conversa.
 */
function Speakers({
  messages,
  from,
  onJump,
}: {
  messages: readonly AgentMessage[];
  from: number;
  onJump: (index: number) => void;
}) {
  const speakers = useMemo(() => {
    const byAuthor = new Map<string, { author: string; color: string; count: number }>();
    for (const message of messages) {
      const entry = byAuthor.get(message.author);
      if (entry === undefined) {
        byAuthor.set(message.author, {
          author: message.author,
          color: messageColor(message),
          count: 1,
        });
      } else {
        entry.count += 1;
      }
    }
    return [...byAuthor.values()];
  }, [messages]);
  if (speakers.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-col gap-2">
      <p className="small-label">Who spoke</p>
      <ul className="flex flex-col gap-0.5">
        {speakers.map((speaker) => (
          <li key={speaker.author}>
            <button
              type="button"
              onClick={() => {
                const after = messages.findIndex(
                  (message, index) => index > from && message.author === speaker.author,
                );
                onJump(
                  after === -1
                    ? messages.findIndex((message) => message.author === speaker.author)
                    : after,
                );
              }}
              className="flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left text-sm text-soft outline-none hover:bg-[#181818] hover:text-strong focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span
                aria-hidden="true"
                className="size-2 shrink-0 rounded-full"
                style={{ background: speaker.color }}
              />
              <span className="min-w-0 flex-1 truncate">{speaker.author}</span>
              <span className="font-mono text-xs text-dim tabular-nums">
                {formatCount(speaker.count)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Pôr a conversa em ecrã inteiro a sério (a API Fullscreen do browser), e sair dele. */
function useFullscreen(target: RefObject<HTMLElement | null>) {
  const supported = typeof document !== "undefined" && document.fullscreenEnabled;
  const [active, setActive] = useState(false);
  useEffect(() => {
    const onChange = () => {
      setActive(document.fullscreenElement !== null);
    };
    document.addEventListener("fullscreenchange", onChange);
    return () => {
      document.removeEventListener("fullscreenchange", onChange);
      // Ao fechar a conversa, sai-se do ecrã inteiro (senão ficava preso a uma página vazia).
      if (document.fullscreenElement !== null) {
        void document.exitFullscreen();
      }
    };
  }, []);
  const toggle = useCallback(() => {
    if (document.fullscreenElement !== null) {
      void document.exitFullscreen();
      return;
    }
    void target.current?.requestFullscreen();
  }, [target]);
  return { supported, active, toggle };
}
