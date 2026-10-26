/**
 * A página de um modelo: como funciona, a trabalhar.
 *
 * Lê-se como um capítulo: a intuição primeiro; depois o modelo ao vivo num conjunto de exemplo
 * pequeno, que volta a treinar quando se muda um hiperparâmetro (a fotografia anterior fica à
 * vista, esbatida, até chegar a nova); o que ele aprendeu; o resto da ficha (a matemática,
 * quando usar, as armadilhas); e as demonstrações do Laboratório que o mostram.
 *
 * O nome chega da lista dos modelos a deslizar para o título (o mesmo `layoutId`).
 */
import { keepPreviousData, useQuery, useSuspenseQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { cn } from "cn";
import { ChevronLeft, LoaderCircle } from "lucide-react";
import { motion } from "motion/react";
import { useState } from "react";

import { demoQuery } from "@/api/demo";
import type { ModelCard, ModelDemo } from "@/api/generated";
import { catalogModelOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { FieldPlayer } from "@/components/charts/field-player";
import { schemaDefaults, type Values } from "@/components/forms/schema";
import { SchemaForm } from "@/components/forms/schema-form";
import { Prose } from "@/components/markdown/prose";
import { TopicText } from "@/components/markdown/topic-text";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, Facts } from "@/components/shell/blocks";
import { Page } from "@/components/shell/page";
import { WhyButton } from "@/components/shell/why-button";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useStartTraining } from "@/features/training/use-start-training";
import { useDebounced } from "@/hooks/use-debounced";
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
  const [intuition, ...rest] = card.sections;
  const hasLearned = demo.data !== undefined;

  const anchors = [
    ...(intuition === undefined ? [] : [{ id: `ficha-${intuition.key}`, title: intuition.title }]),
    { id: "ao-vivo", title: "Ao vivo" },
    ...(hasLearned ? [{ id: "aprendeu", title: "O que aprendeu" }] : []),
    ...rest.map((section) => ({ id: `ficha-${section.key}`, title: section.title })),
    ...(card.laboratory.length > 0 ? [{ id: "laboratorio", title: "No Laboratório" }] : []),
  ];

  return (
    <Page>
      <Link
        to="/modelos"
        className="-ml-1 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ChevronLeft aria-hidden="true" className="size-4" />
        Modelos
      </Link>
      <header className="mt-4 flex flex-col gap-4 @3xl/main:flex-row @3xl/main:items-start @3xl/main:justify-between">
        <div className="max-w-[68ch]">
          <motion.h1
            layoutId={nameLayoutId(card.name)}
            className="inline-block text-4xl leading-tight"
          >
            {card.display_name}
          </motion.h1>
          <p className="mt-3 font-serif text-reading text-muted-foreground">{card.summary}</p>
        </div>
        <WhyButton topics={card.topics} />
      </header>
      <Facts
        size="sm"
        className="mt-6"
        facts={[
          { label: "Família", value: label("ModelFamily", card.family) },
          {
            label: "Tarefas",
            value: formatList(card.task_types.map((task) => label("TaskType", task))),
          },
          { label: "Implementação", value: label("Implementation", card.implementation) },
          { label: "Treino", value: card.iterative ? "Passo a passo" : "De uma vez" },
        ]}
      />
      <nav aria-label="Nesta página" className="mt-8 border-y py-3">
        <ul className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
          {anchors.map((anchor) => (
            <li key={anchor.id}>
              <a
                href={`#${anchor.id}`}
                className="text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
              >
                {anchor.title}
              </a>
            </li>
          ))}
        </ul>
      </nav>

      {intuition !== undefined && (
        <DocBlock id={`ficha-${intuition.key}`} title={intuition.title} text={intuition.text} />
      )}

      <section id="ao-vivo" aria-labelledby="ao-vivo-titulo" className="mt-14 scroll-mt-6">
        <h2 id="ao-vivo-titulo" className="text-2xl">
          Ao vivo
        </h2>
        <div className="mt-6 grid gap-10 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
          <div className="min-w-0 @5xl/main:sticky @5xl/main:top-6 @5xl/main:self-start">
            <Example
              card={card}
              example={example}
              onExample={setExample}
              demo={demo.data}
              reloading={demo.isPlaceholderData || demo.isFetching}
              onStepChange={setStep}
            />
          </div>
          <Controls card={card} values={values} onValues={setValues} error={demo.error} />
        </div>
      </section>

      {demo.data !== undefined && (
        <section id="aprendeu" aria-labelledby="aprendeu-titulo" className="mt-14 scroll-mt-6">
          <h2 id="aprendeu-titulo" className="mb-6 text-2xl">
            O que aprendeu
          </h2>
          <LearnedViewPanel learned={demo.data.learned} network={demo.data.network} step={step} />
        </section>
      )}

      {rest.map((section) => (
        <DocBlock
          key={section.key}
          id={`ficha-${section.key}`}
          title={section.title}
          text={section.text}
        />
      ))}
      {card.sections.length === 0 && card.doc !== "" && <Prose text={card.doc} className="mt-14" />}

      {card.laboratory.length > 0 && (
        <section
          id="laboratorio"
          aria-labelledby="laboratorio-titulo"
          className="mt-14 scroll-mt-6"
        >
          <h2 id="laboratorio-titulo" className="text-2xl">
            No Laboratório
          </h2>
          <p className="mt-2 max-w-[62ch] font-serif text-reading text-muted-foreground">
            Mexe num controlo e vê o efeito no modelo.
          </p>
          <ul className="mt-4 flex flex-wrap gap-3">
            {card.laboratory.map((demoName) => (
              <li key={demoName}>
                <Button asChild variant="outline">
                  <Link to="/laboratorio">{label("LaboratoryDemo", demoName)}</Link>
                </Button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </Page>
  );
}

function DocBlock({ id, title, text }: { id: string; title: string; text: string }) {
  return (
    <section id={id} aria-labelledby={`${id}-titulo`} className="mt-14 scroll-mt-6">
      <h2 id={`${id}-titulo`} className="text-2xl">
        {title}
      </h2>
      <Prose text={text} className="mt-3" />
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
      )}
      {chosen !== undefined && (
        <p className="max-w-[62ch] font-serif text-sm leading-relaxed text-muted-foreground">
          {chosen.description}
        </p>
      )}
      <div
        className={cn("relative transition-opacity", reloading && "opacity-50")}
        aria-busy={reloading}
      >
        {demo === undefined ? (
          <div
            className="flex items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground"
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
            className="absolute top-2 right-2 flex items-center gap-1.5 rounded-md bg-popover px-2 py-1 text-xs"
          >
            <LoaderCircle aria-hidden="true" className="size-3.5 animate-spin" />A treinar com os
            valores novos
          </span>
        )}
      </div>
      {metric !== undefined && (
        <Facts
          size="sm"
          facts={[
            {
              label: `${metric.display_name} no teste`,
              value: formatMetric(metric),
            },
            ...(trainMetric === undefined || report?.partition === "train"
              ? []
              : [{ label: "No treino", value: formatMetric(trainMetric) }]),
          ]}
        />
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
    <div className="flex flex-col gap-6">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="text-lg">Hiperparâmetros</h3>
        {changed && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              onValues(defaults);
            }}
          >
            Repor os de omissão
          </Button>
        )}
      </div>
      <SchemaForm
        schema={card.hyperparameters_schema}
        values={values}
        onChange={onValues}
        idPrefix={`hiperparametro-${card.name}`}
      />
      <ErrorNote error={error} />
      <TrainHere card={card} values={values} />
    </div>
  );
}

/** "Treinar com estes valores": com a preparação da sessão, se a tarefa for deste modelo. */
function TrainHere({ card, values }: { card: ModelCard; values: Values }) {
  const label = useLabel();
  const training = useStartTraining();
  const { preparation } = training;
  return (
    <div className="flex flex-col items-start gap-3 border-t pt-5">
      {preparation === null ? (
        <>
          <p className="font-serif text-sm leading-relaxed text-muted-foreground">
            Para treinar este modelo com os teus dados, prepara-os primeiro.
          </p>
          <Button asChild variant="outline">
            <Link to="/dados">Escolher os dados</Link>
          </Button>
        </>
      ) : !card.task_types.includes(preparation.taskType) ? (
        <>
          <p className="font-serif text-sm leading-relaxed text-muted-foreground">
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
          <Button
            disabled={!training.ready || training.pending}
            onClick={() => {
              training.start(card.name, values, card.display_name);
            }}
          >
            {training.pending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
            Treinar com estes valores
          </Button>
          {training.datasetName !== null && (
            <p className="font-serif text-sm leading-relaxed text-muted-foreground">
              Com {training.datasetName}, preparado para{" "}
              {label("TaskType", preparation.taskType).toLowerCase()}, e a seed {preparation.seed},
              a mesma da divisão: o resultado repete-se.
            </p>
          )}
          <ErrorNote error={training.error} />
        </>
      )}
    </div>
  );
}
