/**
 * A página de um modelo: como funciona, a trabalhar.
 *
 * Em cima, o modelo ao vivo num conjunto de exemplo pequeno, ao lado dos hiperparâmetros: volta
 * a treinar quando se muda um (a fotografia anterior fica à vista, esbatida, até chegar a nova).
 * Depois, o que ele aprendeu; a ficha, uma secção de cada vez (a intuição, a matemática, quando
 * usar, as armadilhas); e as demonstrações do Laboratório que o mostram.
 *
 * O nome chega da lista dos modelos a deslizar para o título (o mesmo `layoutId`). A família tem o
 * contorno com a cor dela, a mesma da lista e do agente da família (ADR 0038).
 */
import { keepPreviousData, useQuery, useSuspenseQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { cn } from "cn";
import { ArrowRight, ChevronLeft, LoaderCircle } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";

import { demoQuery } from "@/api/demo";
import type { ModelCard, ModelDemo } from "@/api/generated";
import { catalogModelOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { FieldPlayer } from "@/components/charts/field-player";
import { Explain } from "@/components/explain/explain";
import { schemaDefaults, type Values } from "@/components/forms/schema";
import { SchemaForm } from "@/components/forms/schema-form";
import { Prose } from "@/components/markdown/prose";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Panel, Stat } from "@/components/shell/blocks";
import { Page } from "@/components/shell/page";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useStartTraining } from "@/features/training/use-start-training";
import { useDebounced } from "@/hooks/use-debounced";
import { familyColor } from "@/lib/families";
import { formatList, formatMetric } from "@/lib/format";

import { EmbeddingView } from "./embedding-view";
import { LearnedViewPanel } from "./learned-view";
import { nameLayoutId } from "./names";

/** O tempo sem mexer nos hiperparâmetros antes de voltar a treinar o exemplo. */
const SETTLE_MS = 300;
const FIELD_HEIGHT = 360;

export function ModelScreen() {
  const { name } = useParams({ from: "/modelos/$name" });
  const card = useSuspenseQuery({
    ...catalogModelOptions({ path: { name } }),
    staleTime: "static",
  }).data;
  return <ModelPage key={card.name} card={card} />;
}

function ModelPage({ card }: { card: ModelCard }) {
  const label = useLabel();
  const [example, setExample] = useState<string | null>(card.examples[0]?.name ?? null);
  const [values, setValues] = useState<Values>(() => ({
    ...schemaDefaults(card.hyperparameters_schema),
    ...card.defaults,
  }));
  const [step, setStep] = useState<number | undefined>(undefined);
  // O pedido vai como texto: só muda quando o conteúdo muda.
  const settled = JSON.parse(useDebounced(JSON.stringify({ values, example }), SETTLE_MS)) as {
    values: Values;
    example: string | null;
  };
  const demo = useQuery({
    ...demoQuery(card.name, settled.values, settled.example),
    placeholderData: keepPreviousData,
  });
  return (
    <Page>
      <Link
        to="/modelos"
        className="-ml-1 inline-flex items-center gap-1 text-sm text-soft hover:text-strong"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        Modelos
      </Link>
      <header className="mt-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <motion.h1
            layoutId={nameLayoutId(card.name)}
            className="inline-block text-[2rem] leading-tight tracking-[-0.025em]"
          >
            {card.display_name}
          </motion.h1>
          <WhyButton title={card.display_name} description={card.summary} topics={card.topics} />
        </div>
        <ul className="flex flex-wrap gap-2" aria-label="A ficha técnica">
          {[
            label("ModelFamily", card.family),
            formatList(card.task_types.map((task) => label("TaskType", task))),
            label("Implementation", card.implementation),
            card.iterative ? "Treina passo a passo" : "Treina de uma vez",
          ].map((fact, index) => (
            <li
              key={fact}
              className={cn(
                "rounded-full border border-[#2a2a2a] bg-surface px-3 py-1 font-mono text-xs",
                index === 0 ? "text-strong" : "text-soft",
              )}
              // A família, a primeira, leva a cor dela no contorno.
              style={index === 0 ? { borderColor: familyColor(card.family) } : undefined}
            >
              {fact}
            </li>
          ))}
        </ul>
      </header>

      <div className="mt-8 grid gap-x-12 gap-y-10 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,23rem)]">
        <Panel id="ao-vivo" title="Ao vivo" as="h2" className="min-w-0">
          <Example
            card={card}
            example={example}
            onExample={setExample}
            demo={demo.data}
            reloading={demo.isPlaceholderData || demo.isFetching}
            onStepChange={setStep}
          />
        </Panel>
        <Controls card={card} values={values} onValues={setValues} error={demo.error} />
      </div>

      {demo.data !== undefined && (
        <section aria-labelledby="aprendeu-titulo" className="mt-10">
          <h2 id="aprendeu-titulo" className="mb-4 text-xl">
            O que aprendeu
          </h2>
          <LearnedViewPanel learned={demo.data.learned} network={demo.data.network} step={step} />
        </section>
      )}

      {card.sections.length > 0 ? (
        <DocTabs sections={card.sections} />
      ) : (
        card.doc !== "" && (
          <div className="mt-10">
            <Prose text={card.doc} />
          </div>
        )
      )}

      {card.laboratory.length > 0 && (
        <Panel
          title="No Laboratório"
          explain="Mexe num controlo e vê o efeito no modelo."
          className="mt-4"
        >
          <ul className="flex flex-wrap gap-2">
            {card.laboratory.map((demoName) => (
              <li key={demoName}>
                <Button asChild variant="outline" size="sm">
                  <Link to="/laboratorio" search={{ demo: demoName }}>
                    {label("LaboratoryDemo", demoName)}
                  </Link>
                </Button>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </Page>
  );
}

/** A ficha didática, uma secção de cada vez: a intuição primeiro, como num capítulo. */
function DocTabs({ sections }: { sections: ModelCard["sections"] }) {
  const first = sections[0]?.key ?? "";
  return (
    <section aria-label="A ficha" className="mt-10">
      <Tabs defaultValue={first}>
        <TabsList
          aria-label="As secções da ficha"
          className="flex-wrap group-data-horizontal/tabs:h-auto"
        >
          {sections.map((section) => (
            <TabsTrigger key={section.key} value={section.key}>
              {section.title}
            </TabsTrigger>
          ))}
        </TabsList>
        {sections.map((section) => (
          // O nome da secção já está no separador: não se repete por cima do texto.
          <TabsContent key={section.key} value={section.key} className="mt-6">
            <Prose text={section.text} />
          </TabsContent>
        ))}
      </Tabs>
    </section>
  );
}

interface ExampleProps {
  card: ModelCard;
  example: string | null;
  onExample: (example: string) => void;
  demo: ModelDemo | undefined;
  reloading: boolean;
  onStepChange: (step: number) => void;
}

/** O modelo num conjunto de exemplo: o campo passo a passo, ou a projeção (na redução). */
function Example({ card, example, onExample, demo, reloading, onStepChange }: ExampleProps) {
  const vocabulary = useVocabulary();
  const chosen = card.examples.find((one) => one.name === example) ?? card.examples[0];
  const primary = vocabulary.tasks.find(
    (task) => task.task_type === demo?.task_type,
  )?.primary_metric;
  const report = demo?.reports[demo.main_partition];
  const train = demo?.reports.train;
  const metric =
    report?.metrics.find((one) => one.name === primary) ?? report?.metrics[0] ?? undefined;
  const trainMetric = train?.metrics.find((one) => one.name === metric?.name);

  return (
    <div className="flex flex-col gap-4">
      {card.examples.length > 1 && (
        <div className="flex flex-wrap items-center gap-2">
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={chosen?.name ?? ""}
            onValueChange={(value) => {
              if (value !== "") {
                onExample(value);
              }
            }}
            aria-label="Conjunto de exemplo"
            className="flex-wrap justify-start"
          >
            {card.examples.map((one) => (
              <ToggleGroupItem key={one.name} value={one.name} className="px-3">
                {one.title}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          {chosen !== undefined && <Explain title={chosen.title}>{chosen.description}</Explain>}
        </div>
      )}
      <div className={cn("relative", reloading && "stale")} aria-busy={reloading}>
        {demo === undefined ? (
          <div
            className="flex items-center justify-center rounded-xl border border-dashed font-mono text-xs text-dim"
            style={{ height: FIELD_HEIGHT }}
          >
            <LoaderCircle aria-hidden="true" className="mr-2 size-4 animate-spin" />A treinar o
            exemplo
          </div>
        ) : (
          <DemoView demo={demo} onStepChange={onStepChange} />
        )}
        {reloading && demo !== undefined && (
          <span
            role="status"
            className="absolute top-10 right-2 flex items-center gap-1.5 rounded-lg border border-[#2a2a2a] bg-popover px-2.5 py-1 font-mono text-xs text-soft"
          >
            <LoaderCircle aria-hidden="true" className="size-3.5 animate-spin" />A treinar com os
            valores novos
          </span>
        )}
      </div>
      {metric !== undefined && (
        <div className="flex flex-wrap gap-x-10 gap-y-3">
          <Stat label={`${metric.display_name} no teste`} value={formatMetric(metric)} />
          {trainMetric !== undefined && report?.partition !== "train" && (
            <Stat label="No treino" value={formatMetric(trainMetric)} />
          )}
        </div>
      )}
      {demo !== undefined && <NoticeList notices={demo.notices} />}
    </div>
  );
}

function DemoView({
  demo,
  onStepChange,
}: {
  demo: ModelDemo;
  onStepChange: (step: number) => void;
}) {
  if (demo.animation !== null) {
    return (
      <FieldPlayer
        // Um treino novo (outros valores, outro exemplo) volta a reproduzir-se do princípio.
        key={JSON.stringify([demo.example.name, demo.hyperparameters])}
        animation={demo.animation}
        autoPlay={demo.animation.frames.length > 1}
        height={FIELD_HEIGHT}
        onStepChange={onStepChange}
        description={`O que ${demo.model_display_name} responde em cada ponto do plano, em ${demo.example.title}.`}
        caption={demo.animation.how_to_read}
      />
    );
  }
  if (demo.embedding !== null) {
    return (
      <EmbeddingView
        embedding={demo.embedding}
        height={FIELD_HEIGHT}
        description={`Os pontos de ${demo.example.title} nas duas dimensões que ${demo.model_display_name} encontrou.`}
        caption={<TopicText topic="embedding" />}
      />
    );
  }
  return null;
}

interface ControlsProps {
  card: ModelCard;
  values: Values;
  onValues: (values: Values) => void;
  error: unknown;
}

/** Os hiperparâmetros e o treino com os dados do utilizador. */
function Controls({ card, values, onValues, error }: ControlsProps) {
  const defaults = { ...schemaDefaults(card.hyperparameters_schema), ...card.defaults };
  const changed = JSON.stringify(values) !== JSON.stringify(defaults);
  return (
    <Panel
      title="Hiperparâmetros"
      className="self-start"
      actions={
        changed && (
          <Button
            variant="ghost"
            size="sm"
            className="-my-1"
            onClick={() => {
              onValues(defaults);
            }}
          >
            Repor os de omissão
          </Button>
        )
      }
    >
      <SchemaForm
        schema={card.hyperparameters_schema}
        values={values}
        onChange={onValues}
        idPrefix={`hiperparametro-${card.name}`}
      />
      <ErrorNote error={error} />
      <TrainHere card={card} values={values} />
    </Panel>
  );
}

/** "Treinar com estes valores": com a preparação da sessão, se a tarefa for deste modelo. */
function TrainHere({ card, values }: { card: ModelCard; values: Values }) {
  const label = useLabel();
  const training = useStartTraining();
  const { preparation } = training;
  return (
    <div className="flex flex-col items-start gap-3">
      {preparation === null ? (
        <>
          <p className="text-sm text-soft">
            Para treinar este modelo com os teus dados, prepara-os primeiro.
          </p>
          <Button asChild variant="outline">
            <Link to="/dados">Escolher os dados</Link>
          </Button>
        </>
      ) : !card.task_types.includes(preparation.taskType) ? (
        <>
          <p className="text-sm text-soft">
            Os dados que preparaste são para {label("TaskType", preparation.taskType).toLowerCase()}
            , e este modelo faz{" "}
            {formatList(card.task_types.map((task) => label("TaskType", task).toLowerCase()))}.
          </p>
          <Button asChild variant="outline">
            <Link to="/modelos" search={{ tarefa: preparation.taskType }}>
              Ver os modelos para esta tarefa
            </Link>
          </Button>
        </>
      ) : (
        <>
          <div className="flex items-center gap-1.5">
            <Button
              variant="proceed"
              size="proceed"
              disabled={!training.ready || training.pending}
              onClick={() => {
                training.start(card.name, values, card.display_name);
              }}
            >
              {training.pending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
              Treinar com estes valores
              {!training.pending && <ArrowRight aria-hidden="true" />}
            </Button>
            {/* Com que dados treina fica no "?", não numa linha por baixo. */}
            {training.datasetName !== null && (
              <Explain title="Com que dados treina">
                Com {training.datasetName}, preparado para{" "}
                {label("TaskType", preparation.taskType).toLowerCase()}, e a seed {preparation.seed}
                , a mesma da divisão: o resultado repete-se.
              </Explain>
            )}
          </div>
          <ErrorNote error={training.error} />
        </>
      )}
    </div>
  );
}
