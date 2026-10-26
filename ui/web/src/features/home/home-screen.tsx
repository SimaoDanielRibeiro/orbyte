/**
 * O Início: abre com um modelo a aprender (o campo de decisão de uma rede neuronal, passo a
 * passo), mostra o que ficou aberto da última vez e pergunta como se quer trabalhar.
 *
 * O campo é o momento de abertura da interface (ADR 0029): reproduz-se uma vez, sozinho, e
 * depois fica parado na última fotografia. Com "reduzir movimento", mostra-se logo o fim.
 */
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import type { ReactNode } from "react";

import { demoQuery } from "@/api/demo";
import { datasetsSummaryOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { FieldPlayer } from "@/components/charts/field-player";
import { ErrorNote } from "@/components/shell/blocks";
import { Page } from "@/components/shell/page";
import { AgentsDiagram, ManualDiagram } from "@/components/shell/path-diagrams";
import { PATH_NAMES } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession, type WorkPath } from "@/state/session";

/** O modelo da abertura: o que melhor mostra uma fronteira a formar-se. */
const HERO_MODEL = "mlp";
const HERO_HEIGHT = 340;

export function HomeScreen() {
  const navigate = useNavigate();
  const choosePath = useSession((state) => state.choosePath);

  function start(path: WorkPath) {
    choosePath(path);
    void navigate({ to: "/dados" });
  }

  return (
    <Page>
      <Hero />
      <Continue />
      <section className="mt-16" aria-labelledby="caminhos">
        <h2 id="caminhos" className="text-3xl">
          Como queres trabalhar?
        </h2>
        <p className="mt-3 max-w-[62ch] font-serif text-reading text-muted-foreground">
          Os dois caminhos começam nos dados e acabam no mesmo sítio: um run que se treina, avalia e
          explica.
        </p>
        <div className="mt-10 grid gap-12 @4xl/main:grid-cols-2">
          <PathOption
            title={PATH_NAMES.manual}
            text="Escolhes os dados, a preparação e o modelo, e vês cada passo do treino."
            action="Começar à mão"
            onStart={() => {
              start("manual");
            }}
          >
            <ManualDiagram />
          </PathOption>
          <PathOption
            title={PATH_NAMES.agents}
            text="Os agentes experimentam modelos com validação cruzada e mostram cada passo do que fazem."
            action="Começar com os agentes"
            onStart={() => {
              start("agents");
            }}
          >
            <AgentsDiagram />
          </PathOption>
        </div>
      </section>
    </Page>
  );
}

/** A abertura: um modelo a aprender, ao vivo. */
function Hero() {
  const demo = useQuery(demoQuery(HERO_MODEL));
  return (
    <section
      aria-labelledby="abertura"
      className="grid items-center gap-10 @4xl/main:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]"
    >
      <div className="max-w-[34rem]">
        <h1 id="abertura" className="text-5xl leading-[1.08] @4xl/main:text-6xl">
          Vê um modelo aprender.
        </h1>
        <p className="mt-6 font-serif text-reading text-muted-foreground">
          Cada fotografia é um passo do treino. As cores são o que o modelo responde em cada ponto
          do plano, e a fronteira entre elas muda à medida que ele aprende com os pontos. É assim em
          todos os modelos do laboratório: o treino por dentro, os dados, as métricas e o porquê de
          cada escolha.
        </p>
      </div>
      <div className="min-w-0">
        {demo.error !== null ? (
          <ErrorNote error={demo.error} />
        ) : demo.data?.animation ? (
          <FieldPlayer
            animation={demo.data.animation}
            autoPlay
            height={HERO_HEIGHT}
            description={`O campo de decisão de ${demo.data.model_display_name} em ${demo.data.example.title}, passo a passo.`}
            caption={
              <>
                {demo.data.model_display_name} em {demo.data.example.title}, com os valores por
                omissão.{" "}
                <Link
                  to="/modelos/$name"
                  params={{ name: HERO_MODEL }}
                  className="underline decoration-input underline-offset-4 hover:decoration-foreground"
                >
                  Ver como funciona
                </Link>
              </>
            }
          />
        ) : (
          <div className="flex flex-col gap-3" aria-busy="true">
            <Skeleton className="w-full" style={{ height: HERO_HEIGHT }} />
            <p role="status" className="text-sm text-muted-foreground">
              A treinar a rede neuronal para a abertura…
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

/** O que ficou aberto da última vez: o dataset, a preparação, o run e a procura dos agentes. */
function Continue() {
  const label = useLabel();
  const { datasetId, preparation, runId, searchId } = useSession();
  const dataset = useQuery({
    ...datasetsSummaryOptions({ path: { dataset_id: datasetId ?? "" } }),
    enabled: datasetId !== null,
  });
  const items: { key: string; title: string; detail: string; link: ReactNode }[] = [];
  if (dataset.data !== undefined) {
    items.push({
      key: "dados",
      title: dataset.data.name,
      detail:
        preparation?.datasetId === dataset.data.id
          ? `Preparado para ${label("TaskType", preparation.taskType).toLowerCase()}`
          : "Dados abertos",
      link:
        preparation?.datasetId === dataset.data.id ? (
          <Link to="/modelos">Escolher o modelo</Link>
        ) : (
          <Link to="/explorar">Explorar</Link>
        ),
    });
  }
  if (runId !== null) {
    items.push({
      key: "run",
      title: "O último treino",
      detail: runId,
      link: (
        <Link to="/treinar/$runId" params={{ runId }}>
          Abrir o treino
        </Link>
      ),
    });
  }
  if (searchId !== null) {
    items.push({
      key: "procura",
      title: "A última procura dos agentes",
      detail: searchId,
      link: (
        <Link to="/agentes/$searchId" params={{ searchId }}>
          Abrir a procura
        </Link>
      ),
    });
  }
  if (items.length === 0) {
    return null;
  }
  return (
    <section aria-labelledby="continuar" className="mt-14 border-t pt-6">
      <h2 id="continuar" className="text-sm text-muted-foreground">
        Continuar onde ficaste
      </h2>
      <ul className="mt-4 grid gap-6 @3xl/main:grid-cols-3">
        {items.map((item) => (
          <li key={item.key} className="flex flex-col items-start gap-1">
            <span className="text-lg">{item.title}</span>
            <span className="text-sm text-muted-foreground">{item.detail}</span>
            <Button asChild variant="outline" size="sm" className="mt-2">
              {item.link}
            </Button>
          </li>
        ))}
      </ul>
    </section>
  );
}

interface PathOptionProps {
  title: string;
  text: string;
  action: string;
  onStart: () => void;
  /** O desenho do caminho. */
  children: ReactNode;
}

function PathOption({ title, text, action, onStart, children }: PathOptionProps) {
  return (
    <section className="flex flex-col gap-5 border-t pt-6">
      <div>
        <h3 className="text-2xl">{title}</h3>
        <p className="mt-2 max-w-[60ch] font-serif text-reading text-muted-foreground">{text}</p>
      </div>
      {children}
      <Button variant="outline" size="lg" className="self-start" onClick={onStart}>
        {action}
      </Button>
    </section>
  );
}
