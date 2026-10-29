/**
 * O ecrã Modelos: todos os modelos do laboratório, arrumados em dois andares (ADR 0037):
 *
 * 1. o paradigma: primeiro os supervisionados (aprendem com um alvo), depois os não
 *    supervisionados (procuram estrutura sem alvo), cada um com o seu "?";
 * 2. a família, num cartão com a cor dela: a mesma cor do agente da família no ecrã Agentes.
 *
 * Cada modelo é uma linha com um desenho pequeno, a mexer, do que ele faz (`model-glyph.tsx`), o
 * nome e uma seta. O nome é o que se leva para a página do modelo: ao abri-la, ele desliza para o
 * título (uma transição partilhada do Motion, com o mesmo `layoutId` nos dois sítios).
 *
 * Com uma preparação feita, a lista mostra só os modelos da tarefa dela (o filtro fica no
 * endereço, `?tarefa=…`, e pode mudar-se para outra tarefa ou para todas).
 */
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { ChevronRight } from "lucide-react";
import { motion } from "motion/react";
import type { CSSProperties } from "react";

import { useTopic } from "@/api/topics";
import type { ModelCard, Paradigm, TaskType } from "@/api/generated";
import {
  catalogModelsOptions,
  catalogSuggestedModelOptions,
} from "@/api/generated/@tanstack/react-query.gen";
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
import { familyColor } from "@/lib/families";
import { formatQuantity } from "@/lib/format";
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
  const search = useSearch({ from: "/modelos/" });
  const preparation = useSession((state) => state.preparation);
  const task: TaskType | null =
    search.tarefa === ALL_TASKS
      ? null
      : ((search.tarefa as TaskType | undefined) ?? preparation?.taskType ?? null);
  const models = useQuery({
    ...catalogModelsOptions({ query: task === null ? undefined : { task_type: task } }),
    staleTime: "static",
  });
  const suggested = useQuery({
    ...catalogSuggestedModelOptions({ query: { task_type: task ?? "binary_classification" } }),
    enabled: task !== null,
    staleTime: "static",
  });

  return (
    <Page>
      <PageHeader
        title={SCREENS.models.title}
        description={SCREENS.models.description}
        topics={["model_selection", "bias_variance"]}
      />
      <div className="mt-8 flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1.5">
          <label htmlFor="modelos-tarefa" className="mono-label">
            Tarefa
          </label>
          <Select
            value={task ?? ALL_TASKS}
            onValueChange={(value) => {
              void navigate({ to: "/modelos", search: { tarefa: value } });
            }}
          >
            <SelectTrigger id="modelos-tarefa" className="min-w-64">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_TASKS}>Todas as tarefas</SelectItem>
              {vocabulary.tasks.map((one) => (
                <SelectItem key={one.task_type} value={one.task_type}>
                  {label("TaskType", one.task_type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {models.data !== undefined && (
          <p className="font-mono text-xs text-dim" role="status">
            {formatQuantity(models.data.length, "modelo", "modelos")}
            {preparation !== null && task === preparation.taskType
              ? ", os da tarefa que preparaste"
              : ""}
          </p>
        )}
      </div>
      <ErrorNote error={models.error} className="mt-8" />
      {models.data === undefined ? (
        models.error === null && <ListSkeleton />
      ) : (
        <Catalog models={models.data} suggested={suggested.data ?? null} />
      )}
    </Page>
  );
}

interface FamilyGroup {
  family: ModelCard["family"];
  members: ModelCard[];
  /** A posição do primeiro modelo da família na página (para desencontrar os desenhos). */
  first: number;
}

interface ParadigmGroup {
  paradigm: Paradigm;
  topic: string;
  families: FamilyGroup[];
}

/** Os modelos de cada paradigma, por família, pela ordem do catálogo. */
function Catalog({
  models,
  suggested,
}: {
  models: readonly ModelCard[];
  suggested: string | null;
}) {
  const vocabulary = useVocabulary();
  // O paradigma de um modelo é o das suas tarefas (uma regra do backend, no vocabulário).
  const paradigmOf = (model: ModelCard): Paradigm =>
    vocabulary.tasks.find((info) => info.task_type === model.task_types[0])?.paradigm ??
    "supervised";
  return (
    <>
      {arrange(models, paradigmOf).map((section) => (
        <ParadigmSection key={section.paradigm} section={section} suggested={suggested} />
      ))}
    </>
  );
}

/** Os paradigmas com modelos, e em cada um as famílias, pela ordem em que aparecem no catálogo. */
function arrange(
  models: readonly ModelCard[],
  paradigmOf: (model: ModelCard) => Paradigm,
): ParadigmGroup[] {
  const sections: ParadigmGroup[] = [];
  let position = 0;
  for (const { paradigm, topic } of PARADIGMS) {
    const families: FamilyGroup[] = [];
    for (const model of models.filter((one) => paradigmOf(one) === paradigm)) {
      const found = families.find((group) => group.family === model.family);
      if (found === undefined) {
        families.push({ family: model.family, members: [model], first: 0 });
      } else {
        found.members.push(model);
      }
    }
    for (const group of families) {
      group.first = position;
      position += group.members.length;
    }
    if (families.length > 0) {
      sections.push({ paradigm, topic, families });
    }
  }
  return sections;
}

/** Um paradigma: o título (o do tema do backend) e o "?", e as famílias em colunas. */
function ParadigmSection({
  section,
  suggested,
}: {
  section: ParadigmGroup;
  suggested: string | null;
}) {
  const label = useLabel();
  const { paradigm, topic, families } = section;
  const explanation = useTopic(topic);
  return (
    <section aria-labelledby={`paradigma-${paradigm}`} className="mt-14 first-of-type:mt-10">
      <div className="mb-6 flex items-center gap-1.5">
        <h2 id={`paradigma-${paradigm}`} className="text-xl">
          {explanation?.title ?? label("Paradigm", paradigm)}
        </h2>
        <Explain topic={topic} />
      </div>
      <div className="gap-6 @4xl/main:columns-2 @[96rem]/main:columns-3">
        {families.map((group) => (
          <FamilyCard key={group.family} group={group} suggested={suggested} />
        ))}
      </div>
    </section>
  );
}

/** Uma família: um cartão com a cor dela, e um modelo por linha, cada um com o seu desenho. */
function FamilyCard({ group, suggested }: { group: FamilyGroup; suggested: string | null }) {
  const label = useLabel();
  const { family, members, first } = group;
  const color = familyColor(family);
  const id = `familia-${family}-${String(first)}`;
  return (
    // Um grupo, não uma região: uma família pode aparecer nos dois paradigmas, e as regiões da
    // página (os paradigmas) precisam de nomes únicos.
    <div
      role="group"
      aria-labelledby={id}
      // A cor da família é o contorno do cartão (ADR 0038).
      className="relative mb-6 break-inside-avoid overflow-hidden rounded-2xl border bg-surface"
      style={{ "--family": color, borderColor: withAlpha(color, 0.45) } as CSSProperties}
    >
      <h3
        id={id}
        className="px-5 pt-5 pb-3 text-[0.9375rem] font-medium tracking-normal text-strong"
      >
        {label("ModelFamily", family)}
      </h3>
      <ul className="relative flex flex-col gap-0.5 px-2 pb-2">
        {members.map((model, offset) => (
          <li key={model.name}>
            <Link
              to="/modelos/$name"
              params={{ name: model.name }}
              className="group flex items-center gap-4 rounded-xl px-3 py-2 transition-colors outline-none hover:bg-surface-high focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span
                aria-hidden="true"
                className="flex h-11 w-[4.25rem] shrink-0 items-center justify-center rounded-lg border border-[#262626] bg-[#141414] transition-[border-color,box-shadow] duration-300 group-hover:[border-color:var(--family)] group-focus-visible:[border-color:var(--family)]"
              >
                <ModelGlyph
                  kind={glyphOf(model)}
                  color={color}
                  delay={((first + offset) * GLYPH_STAGGER) % 6}
                />
              </span>
              <motion.span
                layoutId={nameLayoutId(model.name)}
                className="inline-block min-w-0 flex-1 text-[0.9375rem] text-strong"
              >
                {model.display_name}
              </motion.span>
              <span className="flex shrink-0 items-center gap-3">
                {model.name === suggested && (
                  <span className="rounded-full border border-white/20 bg-white/[0.06] px-2 py-0.5 font-mono text-[0.6875rem] text-strong">
                    sugerido para começar
                  </span>
                )}
                <ChevronRight
                  aria-hidden="true"
                  className="size-4 text-dim transition-transform group-hover:translate-x-0.5 group-hover:text-strong motion-reduce:transition-none"
                />
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ListSkeleton() {
  return (
    <div
      className="mt-10 grid gap-6 @4xl/main:grid-cols-2"
      role="status"
      aria-busy="true"
      aria-label="A carregar"
    >
      {[0, 1, 2, 3].map((index) => (
        <Skeleton key={index} className="h-48 w-full rounded-2xl" />
      ))}
    </div>
  );
}
