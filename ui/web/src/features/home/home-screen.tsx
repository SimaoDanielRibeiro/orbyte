/**
 * A landing: o nome da app e as duas interfaces, cada uma num cartão com o seu desenho.
 *
 * O cartão do Manual mostra um modelo a aprender (o campo de decisão de uma rede neuronal, da
 * demonstração do catálogo), que se reproduz uma vez e fica no fim; com "reduzir movimento",
 * mostra-se logo o fim. O dos Agentes mostra o coordenador a distribuir o trabalho pelos agentes
 * das famílias de modelos, cada um na sua cor. Não há barra lateral: ela é de cada interface.
 */
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { ChevronRight, Hand, Workflow, type LucideIcon } from "lucide-react";
import { useId, type ReactNode } from "react";

import { demoQuery } from "@/api/demo";
import type { ModelFamily } from "@/api/generated";
import { metaHealthOptions } from "@/api/generated/@tanstack/react-query.gen";
import { FieldPlayer } from "@/components/charts/field-player";
import { GlowFilter } from "@/components/charts/chart";
import { Logo } from "@/components/shell/logo";
import { INTERFACE_NAMES, stepsOf } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { familyColor } from "@/lib/families";
import { useSession, type WorkPath } from "@/state/session";

/** O modelo do cartão Manual: o que melhor mostra uma fronteira a formar-se. */
const HERO_MODEL = "mlp";
const VISUAL_HEIGHT = 248;

export function HomeScreen() {
  const navigate = useNavigate();
  const choosePath = useSession((state) => state.choosePath);

  function start(path: WorkPath) {
    choosePath(path);
    void navigate({ to: "/dados" });
  }

  return (
    <div className="min-h-svh bg-ink">
      <div className="mx-auto flex max-w-6xl flex-col px-4 pb-16 md:px-8">
        <header className="flex h-16 items-center justify-between">
          <span className="flex items-center gap-2.5">
            <Logo className="h-5 w-6" />
            <span translate="no" className="text-[1.0625rem] font-semibold text-strong">
              ML Lab
            </span>
          </span>
          <ApiPill />
        </header>

        <main>
          <div className="pt-14 pb-12 md:pt-24 md:pb-16">
            <h1
              translate="no"
              className="text-[clamp(3.25rem,9vw,6rem)] leading-[0.95] font-semibold tracking-[-0.045em]"
            >
              ML Lab
            </h1>
            <p className="mt-5 max-w-xl text-lg text-soft">
              Treina modelos de machine learning e vê-os aprender, passo a passo.
            </p>
          </div>

          <div className="grid gap-5 md:grid-cols-2">
            <InterfaceCard
              path="manual"
              icon={Hand}
              text="Escolhes os dados, a preparação e o modelo, e vês cada passo do treino."
              action="Começar à mão"
              onStart={() => {
                start("manual");
              }}
            >
              <ManualVisual />
            </InterfaceCard>
            <InterfaceCard
              path="agents"
              icon={Workflow}
              text="Dizes o que queres prever, e os agentes experimentam os modelos por ti."
              action="Começar com os agentes"
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

interface InterfaceCardProps {
  path: WorkPath;
  icon: LucideIcon;
  text: string;
  action: string;
  onStart: () => void;
  /** O desenho da interface. */
  children: ReactNode;
}

/** Uma interface: o desenho, o nome, o que se faz nela e os passos (as secções da barra). */
function InterfaceCard({ path, icon: Icon, text, action, onStart, children }: InterfaceCardProps) {
  const flow = stepsOf(path);
  return (
    <article className="group relative flex min-w-0 flex-col overflow-hidden panel transition-colors hover:border-[#333]">
      <div
        className="relative flex items-center border-b bg-ground px-5 pt-5 pb-4"
        style={{ minHeight: VISUAL_HEIGHT + 36 }}
      >
        <div className="w-full">{children}</div>
      </div>
      <div className="flex flex-1 flex-col gap-5 p-6">
        <div className="flex items-center gap-2.5">
          <Icon aria-hidden="true" className="size-5 text-soft" />
          <h2 className="text-2xl">{INTERFACE_NAMES[path]}</h2>
        </div>
        <p className="max-w-[46ch] text-[0.9375rem] text-soft">{text}</p>
        <ol
          aria-label={`As secções da interface ${INTERFACE_NAMES[path]}`}
          className="flex flex-wrap items-center gap-y-1.5"
        >
          {flow.map((screen, index) => (
            <li key={screen.to} className="flex items-center">
              <span className="mono-label">{screen.title}</span>
              {index < flow.length - 1 && (
                <ChevronRight aria-hidden="true" className="mx-1 size-3 text-faint" />
              )}
            </li>
          ))}
        </ol>
        <div className="mt-auto pt-1">
          {/* O botão cobre o cartão todo (o "after"), mas o nome dele é só a ação. */}
          <Button
            size="lg"
            onClick={onStart}
            className="px-4 after:absolute after:inset-0 after:content-['']"
          >
            {action}
          </Button>
        </div>
      </div>
    </article>
  );
}

/** O Manual: uma rede neuronal a aprender as luas, ao vivo. */
function ManualVisual() {
  const demo = useQuery(demoQuery(HERO_MODEL));
  if (demo.data?.animation) {
    return (
      <FieldPlayer
        animation={demo.data.animation}
        autoPlay
        controls={false}
        // A legenda por cima ocupa 36 px: o campo fica com o resto, e os dois cartões alinham.
        height={VISUAL_HEIGHT - 36}
        description={`O campo de decisão de ${demo.data.model_display_name} em ${demo.data.example.title}, passo a passo.`}
      />
    );
  }
  return (
    <div style={{ height: VISUAL_HEIGHT }} className="flex flex-col justify-end gap-3">
      <Skeleton className="w-full flex-1" />
      <p role="status" className="mono-label">
        {demo.error === null ? "A treinar a rede neuronal…" : "A demonstração não abriu."}
      </p>
    </div>
  );
}

/**
 * Os agentes de cada família, como no fluxograma do ecrã Agentes (os nomes curtos), cada um com a
 * cor da sua família (a mesma em toda a interface).
 */
const FAMILIES = [
  ["linear", "Lineares"],
  ["tree", "Árvores"],
  ["ensemble", "Ensembles"],
  ["kernel", "Kernel"],
  ["neural", "Redes neuronais"],
] as const satisfies readonly (readonly [ModelFamily, string])[];

/** Os Agentes: o coordenador distribui o trabalho, e o melhor candidato é o vencedor. */
function AgentsVisual() {
  const id = useId();
  const glow = `${id}-brilho`;
  const top = 26;
  const gap = (VISUAL_HEIGHT - 2 * top) / (FAMILIES.length - 1);
  const middle = VISUAL_HEIGHT / 2;
  return (
    <svg
      viewBox={`0 0 520 ${String(VISUAL_HEIGHT)}`}
      role="img"
      aria-label="O coordenador distribui o trabalho pelos agentes de cada família de modelos, e o melhor candidato é o vencedor."
      className="block h-auto w-full font-mono"
      style={{ maxHeight: VISUAL_HEIGHT }}
    >
      <defs>
        <GlowFilter id={glow} blur={4} />
      </defs>
      {FAMILIES.map(([family, name], index) => {
        const y = top + index * gap;
        const color = familyColor(family);
        const toFamily = `M150 ${String(middle)} C 200 ${String(middle)}, 200 ${String(y)}, 250 ${String(y)}`;
        const toWinner = `M376 ${String(y)} C 410 ${String(y)}, 410 ${String(middle)}, 440 ${String(middle)}`;
        return (
          <g key={family}>
            <path
              d={toFamily}
              fill="none"
              stroke={color}
              strokeWidth={5}
              strokeOpacity={0.35}
              filter={`url(#${glow})`}
            />
            <path d={toFamily} fill="none" stroke={color} strokeWidth={1.75} />
            <path d={toWinner} fill="none" stroke="#3a3a3a" strokeWidth={1.25} />
            <rect
              x={250.5}
              y={y - 13.5}
              width={125}
              height={27}
              rx={8}
              fill="#191919"
              stroke="#2a2a2a"
            />
            <circle cx={265} cy={y} r={4} fill={color} />
            <text x={276} y={y} dy="0.34em" fontSize={11.5} fill="#d4d4d4">
              {name}
            </text>
          </g>
        );
      })}
      <Node x={4} label="Dados" middle={middle} />
      <Node x={66} label="Coordenador" middle={middle} width={84} />
      <path d={`M60 ${String(middle)} H66`} stroke="#3a3a3a" />
      <Node x={440} label="Vencedor" middle={middle} width={76} strong />
    </svg>
  );
}

interface NodeProps {
  x: number;
  label: string;
  middle: number;
  width?: number;
  strong?: boolean;
}

function Node({ x, label, middle, width = 56, strong = false }: NodeProps) {
  return (
    <g>
      <rect
        x={x + 0.5}
        y={middle - 15.5}
        width={width}
        height={31}
        rx={9}
        fill={strong ? "#232323" : "#191919"}
        stroke={strong ? "#4a4a4a" : "#2a2a2a"}
      />
      <text
        x={x + width / 2}
        y={middle}
        dy="0.34em"
        textAnchor="middle"
        fontSize={11.5}
        fill={strong ? "#fafafa" : "#d4d4d4"}
      >
        {label}
      </text>
    </g>
  );
}

/** Se a API está ligada, num ponto pequeno no canto (como na barra lateral). */
function ApiPill() {
  const health = useQuery({ ...metaHealthOptions(), refetchInterval: 15_000, retry: false });
  const down = health.isError;
  return (
    <span role="status" className="flex items-center gap-2 text-sm text-soft">
      <span
        aria-hidden="true"
        className={down ? "size-2 rounded-full bg-negative" : "size-2 rounded-full bg-positive"}
      />
      {health.isPending ? "A ligar à API…" : down ? "API desligada" : "API ligada"}
    </span>
  );
}

/** O que ficou aberto da última vez: o último treino e a última procura dos agentes. */
function Continue() {
  const { runId, searchId } = useSession();
  if (runId === null && searchId === null) {
    return null;
  }
  return (
    <section aria-labelledby="continuar" className="mt-10 flex flex-wrap items-center gap-3">
      <h2 id="continuar" className="mr-2 text-sm font-normal text-soft">
        Continuar onde ficaste
      </h2>
      {runId !== null && (
        <Button asChild variant="outline" size="sm">
          <Link to="/treinar/$runId" params={{ runId }}>
            Abrir o último treino
          </Link>
        </Button>
      )}
      {searchId !== null && (
        <Button asChild variant="outline" size="sm">
          <Link to="/agentes/$searchId" params={{ searchId }}>
            Abrir a última procura
          </Link>
        </Button>
      )}
    </section>
  );
}
