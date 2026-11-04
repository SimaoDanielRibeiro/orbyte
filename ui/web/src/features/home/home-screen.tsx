/**
 * A entrada: o nome, uma frase, e as duas maneiras de trabalhar, cada uma num cartão com o seu
 * desenho animado, o nome, o que se faz nela e o botão para começar.
 *
 * Simples de propósito: o título grande, um fundo discreto de órbitas (a marca), e os dois
 * cartões. O desenho do Manual mostra os passos que se fazem à mão e um modelo a aprender; o dos
 * Agentes, o coordenador a dar a vez aos agentes de cada família, numa órbita. Os dois andam em
 * ciclo e são gerais (não dependem da API); com "reduzir movimento", mostram o fim, parados.
 * Não há barra lateral: ela é de cada interface.
 */
import { Link, useNavigate } from "@tanstack/react-router";
import { Hand, Workflow, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { Logo } from "@/components/shell/logo";
import { INTERFACE_NAMES } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import { useApiDown } from "@/hooks/use-api-down";
import { useSession, type WorkPath } from "@/state/session";

import { AgentsVisual } from "./agents-visual";
import { ManualVisual } from "./manual-visual";

export function HomeScreen() {
  const navigate = useNavigate();
  const choosePath = useSession((state) => state.choosePath);

  function start(path: WorkPath) {
    choosePath(path);
    void navigate({ to: "/data" });
  }

  return (
    <div className="relative min-h-svh overflow-hidden bg-ink">
      <Backdrop />
      {/* A largura toda, com as margens das outras páginas (3 cm num computador). */}
      <div className="relative flex w-full flex-col px-4 pb-20 md:px-8 xl:px-[3cm]">
        <header className="flex h-16 items-center justify-between">
          <span className="flex items-center gap-2.5">
            <Logo className="h-5 w-6" />
            <span
              translate="no"
              className="font-display text-lg font-medium tracking-[-0.02em] text-strong"
            >
              Orbyte
            </span>
          </span>
          <ApiDown />
        </header>

        <main>
          {/* O nome é o desenho: a Unbounded, larga como uma órbita, num degradê de luz. O que é
              o Orbyte diz-se na frase, não num rótulo por cima. */}
          <div className="pt-16 pb-14 md:pt-28 md:pb-20">
            <h1
              translate="no"
              className="bg-linear-to-b from-white via-[#e8e8e8] to-[#8f8f8f] bg-clip-text pb-2 font-display text-[clamp(3.25rem,10vw,7rem)] leading-[0.95] font-medium tracking-[-0.045em] text-transparent"
            >
              Orbyte
            </h1>
            <p className="mt-6 max-w-[52ch] text-lg text-soft">
              A machine learning lab. Train models by hand, or let a team of agents search for you,
              and see how every one of them learns.
            </p>
          </div>

          <div className="grid gap-5 md:grid-cols-2">
            <InterfaceCard
              path="manual"
              icon={Hand}
              text="You choose the data, the preparation and the model, and watch every step of the training."
              action="Start by hand"
              onStart={() => {
                start("manual");
              }}
            >
              <ManualVisual />
            </InterfaceCard>
            <InterfaceCard
              path="agents"
              icon={Workflow}
              text="Tell the agents what to predict: they try the models, tune the best ones and explain why one wins."
              action="Start with the agents"
              onStart={() => {
                start("agents");
              }}
            >
              <AgentsVisual />
            </InterfaceCard>
          </div>

          <Continue />
        </main>
      </div>
    </div>
  );
}

/**
 * O fundo: um brilho azul muito ténue por cima e três órbitas finas que rodam devagar (paradas,
 * com "reduzir movimento"). É só decoração: os leitores de ecrã não o veem.
 */
function Backdrop() {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 top-0 h-[44rem]">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_0%,rgb(105_150_212/0.14),transparent_70%)]" />
      <svg
        viewBox="0 0 1200 700"
        className="absolute top-[-6rem] left-1/2 h-[52rem] w-[90rem] max-w-none -translate-x-1/2 opacity-70"
      >
        <g
          className="origin-center motion-safe:animate-[spin_160s_linear_infinite]"
          style={{ transformBox: "fill-box" }}
        >
          {[
            [520, 170],
            [400, 128],
            [280, 88],
          ].map(([rx, ry]) => (
            <ellipse
              key={rx}
              cx={600}
              cy={330}
              rx={rx}
              ry={ry}
              transform="rotate(-14 600 330)"
              fill="none"
              stroke="rgb(255 255 255 / 0.055)"
            />
          ))}
        </g>
      </svg>
      <div className="absolute inset-x-0 bottom-0 h-40 bg-linear-to-b from-transparent to-ink" />
    </div>
  );
}

interface InterfaceCardProps {
  path: WorkPath;
  icon: LucideIcon;
  text: string;
  action: string;
  onStart: () => void;
  /** O desenho da interface. */
  children: ReactNode;
}

/** Uma interface: o desenho, o nome e o que se faz nela (os passos estão na barra de cada uma). */
function InterfaceCard({ path, icon: Icon, text, action, onStart, children }: InterfaceCardProps) {
  return (
    <article className="group relative flex min-w-0 flex-col overflow-hidden rounded-2xl border border-[#262626] bg-[#141414]/90 shadow-[inset_0_1px_0_rgb(255_255_255/0.04)] backdrop-blur transition-colors hover:border-[#3a3a3a]">
      <div className="relative border-b border-[#222] bg-[radial-gradient(circle,#222_1px,transparent_1.2px)] bg-size-[20px_20px] px-5 py-6">
        {children}
      </div>
      <div className="flex flex-1 flex-col gap-4 p-6">
        <div className="flex items-center gap-2.5">
          <Icon aria-hidden="true" className="size-5 text-soft" />
          <h2 className="text-2xl">{INTERFACE_NAMES[path]}</h2>
        </div>
        <p className="max-w-[46ch] text-base text-soft">{text}</p>
        <div className="mt-auto pt-2">
          {/* O botão cobre o cartão todo (o "after"), mas o nome dele é só a ação. */}
          <Button
            size="lg"
            onClick={onStart}
            className="px-5 after:absolute after:inset-0 after:content-['']"
          >
            {action}
          </Button>
        </div>
      </div>
    </article>
  );
}

/** A API caiu: só então aparece, no canto (ligada, é o normal e não precisa de texto). */
function ApiDown() {
  const down = useApiDown();
  if (!down) {
    return null;
  }
  return (
    <span role="alert" className="flex items-center gap-2 text-sm text-negative">
      <span aria-hidden="true" className="size-2 rounded-full bg-negative" />
      API offline
    </span>
  );
}

/** O que ficou aberto da última vez: o último treino e a última procura dos agentes. */
function Continue() {
  const runId = useSession((state) => state.runId);
  const searchId = useSession((state) => state.searchId);
  if (runId === null && searchId === null) {
    return null;
  }
  return (
    <section aria-labelledby="continue" className="mt-10 flex flex-wrap items-center gap-3">
      <h2 id="continue" className="mr-2 text-sm font-normal text-soft">
        Pick up where you left off
      </h2>
      {runId !== null && (
        <Button asChild variant="outline" size="sm">
          <Link to="/train/$runId" params={{ runId }}>
            Open the last training
          </Link>
        </Button>
      )}
      {searchId !== null && (
        <Button asChild variant="outline" size="sm">
          <Link to="/agents/$searchId" params={{ searchId }}>
            Open the last search
          </Link>
        </Button>
      )}
    </section>
  );
}
