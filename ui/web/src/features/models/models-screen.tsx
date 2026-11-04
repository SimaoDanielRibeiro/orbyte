/**
 * O ecrã Modelos: todos os modelos do laboratório, arrumados pelo problema que resolvem
 * (ADR 0048), em dois andares:
 *
 * 1. o paradigma: primeiro os supervisionados (aprendem com um alvo), depois os não
 *    supervisionados (procuram estrutura sem alvo), cada um com o seu "?";
 * 2. o problema (a classificação, a regressão, o clustering…), com a cor dele: fria nos
 *    supervisionados, quente nos não supervisionados (`lib/problems.ts`). A cor diz o que o
 *    modelo resolve; o nome da família, por baixo do nome, diz como aprende.
 *
 * Cada modelo é um cartão com um desenho, a mexer, do que ele faz (`model-glyph.tsx`), o nome e
 * a família. O nome é o que se leva para a página do modelo: ao abri-la, ele desliza para o
 * título (uma transição partilhada do Motion, com o mesmo `layoutId` nos dois sítios).
 *
 * Abre com todas as tarefas; o filtro fica no endereço (`?task=…`), e pode escolher-se uma tarefa
 * (a página de um modelo leva de volta aos modelos da tarefa preparada).
 */
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { ArrowUpRight } from "lucide-react";
import { motion } from "motion/react";
import type { CSSProperties } from "react";

import { useTopic } from "@/api/topics";
import type { ModelCard, Paradigm, Problem, TaskType } from "@/api/generated";
import { catalogModelsOptions } from "@/api/generated/@tanstack/react-query.gen";
import { useLabel, useVocabulary } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { ErrorNote } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { withAlpha } from "@/lib/colors";
import { formatCount, formatQuantity } from "@/lib/format";
import { problemColor } from "@/lib/problems";
import { useSession } from "@/state/session";

import { glyphOf } from "./glyphs";
import { ModelGlyph } from "./model-glyph";
import { ALL_TASKS, nameLayoutId } from "./names";

/** Os paradigmas com modelos, pela ordem em que se aprendem, e o tema que explica cada um. */
const PARADIGMS = [
  { paradigm: "supervised", topic: "supervised" },
  { paradigm: "unsupervised", topic: "unsupervised" },
] as const satisfies readonly { paradigm: Paradigm; topic: string }[];

/** Quanto se adianta o desenho de cada modelo em relação ao anterior (em segundos). */
const GLYPH_STAGGER = 0.7;

export function ModelsScreen() {
  const label = useLabel();
  const vocabulary = useVocabulary();
  const navigate = useNavigate();
  const search = useSearch({ from: "/models/" });
  const preparation = useSession((state) => state.preparation);
  // Abre sempre com todas as tarefas (ADR 0051); só um endereço com `?task=…` filtra. Antes abria
  // na tarefa da preparação, e quem só queria ver os modelos encontrava a lista cortada.
  const task: TaskType | null =
    search.task === undefined || search.task === ALL_TASKS ? null : (search.task as TaskType);
  const models = useQuery({
    ...catalogModelsOptions({ query: task === null ? undefined : { task_type: task } }),
    staleTime: "static",
  });

  return (
    <Page>
      <PageHeader
        title={SCREENS.models.title}
        description={SCREENS.models.description}
        topics={["model_selection", "bias_variance"]}
      />
      <div className="mt-10 flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-2">
          <label htmlFor="modelos-tarefa" className="small-label">
            Task
          </label>
          <Select
            value={task ?? ALL_TASKS}
            onValueChange={(value) => {
              void navigate({ to: "/models", search: { task: value } });
            }}
          >
            <SelectTrigger id="modelos-tarefa" className="min-w-64">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_TASKS}>All tasks</SelectItem>
              {vocabulary.tasks.map((one) => (
                <SelectItem key={one.task_type} value={one.task_type}>
                  {label("TaskType", one.task_type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {models.data !== undefined && (
          <p className="text-sm text-dim tabular-nums" role="status">
            {formatQuantity(models.data.length, "model", "models")}
            {preparation !== null && task === preparation.taskType
              ? ", the ones for the task you prepared"
              : ""}
          </p>
        )}
      </div>
      <ErrorNote error={models.error} className="mt-8" />
      {models.data === undefined ? (
        models.error === null && <ListSkeleton />
      ) : (
        <Catalog models={models.data} />
      )}
    </Page>
  );
}

interface ProblemGroup {
  problem: Problem;
  members: ModelCard[];
  /** A posição do primeiro modelo do problema na página (para desencontrar os desenhos). */
  first: number;
}

interface ParadigmGroup {
  paradigm: Paradigm;
  topic: string;
  problems: ProblemGroup[];
}

/** Os modelos de cada paradigma, por problema, pela ordem das tarefas e do catálogo. */
function Catalog({ models }: { models: readonly ModelCard[] }) {
  const vocabulary = useVocabulary();
  // O paradigma e o problema de um modelo são os das suas tarefas (regras do backend, no
  // vocabulário): as tarefas de um modelo resolvem todas o mesmo problema.
  const infoOf = (model: ModelCard) =>
    vocabulary.tasks.find((info) => info.task_type === model.task_types[0]);
  const order = vocabulary.tasks.map((info) => info.problem);
  return (
    <>
      {arrange(models, infoOf, order).map((section) => (
        <ParadigmSection key={section.paradigm} section={section} />
      ))}
    </>
  );
}

/** Os paradigmas com modelos e, em cada um, os problemas, pela ordem das tarefas. */
function arrange(
  models: readonly ModelCard[],
  infoOf: (model: ModelCard) => { paradigm: Paradigm; problem: Problem } | undefined,
  order: readonly Problem[],
): ParadigmGroup[] {
  const sections: ParadigmGroup[] = [];
  let position = 0;
  for (const { paradigm, topic } of PARADIGMS) {
    const problems: ProblemGroup[] = [];
    for (const model of models) {
      const info = infoOf(model);
      if (info?.paradigm !== paradigm) {
        continue;
      }
      const found = problems.find((group) => group.problem === info.problem);
      if (found === undefined) {
        problems.push({ problem: info.problem, members: [model], first: 0 });
      } else {
        found.members.push(model);
      }
    }
    problems.sort((a, b) => order.indexOf(a.problem) - order.indexOf(b.problem));
    for (const group of problems) {
      group.first = position;
      position += group.members.length;
    }
    if (problems.length > 0) {
      sections.push({ paradigm, topic, problems });
    }
  }
  return sections;
}

/** Um paradigma: o título (o do tema do backend) e o "?", e os problemas dele, um a um. */
function ParadigmSection({ section }: { section: ParadigmGroup }) {
  const label = useLabel();
  const { paradigm, topic, problems } = section;
  const explanation = useTopic(topic);
  return (
    <section
      aria-labelledby={`paradigma-${paradigm}`}
      className="mt-20 border-t border-[#202020] pt-10 first-of-type:mt-12"
    >
      <div className="mb-10 flex items-center gap-1.5">
        <h2 id={`paradigma-${paradigm}`} className="text-2xl">
          {explanation?.title ?? label("Paradigm", paradigm)}
        </h2>
        <Explain topic={topic} />
      </div>
      <div className="flex flex-col gap-14">
        {problems.map((group) => (
          <ProblemSection key={group.problem} group={group} />
        ))}
      </div>
    </section>
  );
}

/** Um problema: o nome com a cor dele e quantos modelos o resolvem, e os cartões. */
function ProblemSection({ group }: { group: ProblemGroup }) {
  const label = useLabel();
  const { problem, members, first } = group;
  const color = problemColor(problem);
  const id = `problema-${problem}`;
  return (
    <section aria-labelledby={id}>
      <div className="mb-5 flex items-center gap-3">
        <span
          aria-hidden="true"
          className="size-2.5 rounded-full"
          style={{ backgroundColor: color, boxShadow: `0 0 12px ${withAlpha(color, 0.55)}` }}
        />
        <h3 id={id} className="text-lg font-medium tracking-normal text-strong">
          {label("Problem", problem)}
        </h3>
        <span className="font-mono text-sm text-dim tabular-nums">
          {formatCount(members.length)}
        </span>
      </div>
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(15.5rem,1fr))] gap-3">
        {members.map((model, offset) => (
          <ModelTile
            key={model.name}
            model={model}
            color={color}
            delay={((first + offset) * GLYPH_STAGGER) % 6}
          />
        ))}
      </ul>
    </section>
  );
}

interface ModelTileProps {
  model: ModelCard;
  /** A cor do problema que o modelo resolve. */
  color: string;
  /** O avanço do desenho no ciclo, para os cartões não mexerem todos ao mesmo tempo. */
  delay: number;
}

/**
 * Um modelo: o desenho em cima, o nome e a família em baixo. Ao passar o rato, o cartão sobe um
 * nada, o contorno ganha a cor do problema e uma luz dela acende por trás do desenho.
 *
 * A ligação é só o nome (é o nome dela, e o que se lê), e cobre o cartão todo com o `after`: a
 * família fica fora dela, como descrição. Com a família dentro, o texto à vista da ligação não
 * batia com o nome dela.
 */
function ModelTile({ model, color, delay }: ModelTileProps) {
  const label = useLabel();
  const familyId = `modelo-familia-${model.name}`;
  return (
    <li
      // `isolate`: a luz (atrás, com `-z-10`) fica por cima do fundo do cartão e por baixo do resto.
      className="group relative isolate flex min-w-0 flex-col gap-6 overflow-hidden rounded-2xl border border-[#242424] bg-surface p-4 pb-5 shadow-[inset_0_1px_0_rgb(255_255_255/0.025)] transition-[border-color,background-color,translate,box-shadow] duration-500 ease-out-expo hover:-translate-y-0.5 hover:[border-color:var(--tone-line)] hover:bg-[#1c1c1c] hover:shadow-[0_12px_32px_-16px_var(--tone-shadow)] has-[a:focus-visible]:[border-color:var(--tone-line)] has-[a:focus-visible]:ring-2 has-[a:focus-visible]:ring-ring motion-reduce:hover:translate-y-0"
      style={
        {
          "--tone": color,
          "--tone-line": withAlpha(color, 0.5),
          "--tone-glow": withAlpha(color, 0.16),
          "--tone-shadow": withAlpha(color, 0.35),
        } as CSSProperties
      }
    >
      <span
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-28 bg-[radial-gradient(ellipse_70%_100%_at_30%_0%,var(--tone-glow),transparent_75%)] opacity-0 transition-opacity duration-500 ease-out-expo group-hover:opacity-100 group-has-[a:focus-visible]:opacity-100"
      />
      <span aria-hidden="true" className="flex items-start justify-between">
        <span className="flex h-16 w-24 items-center justify-center rounded-xl border border-[#262626] bg-[#141414] transition-[border-color] duration-500 ease-out-expo group-hover:[border-color:var(--tone-line)]">
          <ModelGlyph
            kind={glyphOf(model)}
            color={color}
            delay={delay}
            className="glyph h-12 w-[4.75rem]"
          />
        </span>
        <ArrowUpRight className="size-4 -translate-x-1 translate-y-1 text-dim opacity-0 transition-[opacity,translate,color] duration-500 ease-out-expo group-hover:translate-x-0 group-hover:translate-y-0 group-hover:text-strong group-hover:opacity-100 group-has-[a:focus-visible]:translate-x-0 group-has-[a:focus-visible]:translate-y-0 group-has-[a:focus-visible]:opacity-100" />
      </span>
      <span className="mt-auto flex min-w-0 flex-col gap-1">
        <Link
          to="/models/$name"
          params={{ name: model.name }}
          aria-describedby={familyId}
          className="outline-none after:absolute after:inset-0 after:content-['']"
        >
          <motion.span
            layoutId={nameLayoutId(model.name)}
            className="inline-block text-base leading-snug font-medium text-strong"
          >
            {model.display_name}
          </motion.span>
        </Link>
        <span id={familyId} className="text-sm text-dim">
          {label("ModelFamily", model.family)}
        </span>
      </span>
    </li>
  );
}

function ListSkeleton() {
  return (
    <div
      className="mt-12 grid grid-cols-[repeat(auto-fill,minmax(15.5rem,1fr))] gap-3"
      role="status"
      aria-busy="true"
      aria-label="Loading"
    >
      {Array.from({ length: 8 }, (_, index) => (
        <Skeleton key={index} className="h-40 w-full rounded-2xl" />
      ))}
    </div>
  );
}
