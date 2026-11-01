/**
 * O ecrã Guardar: o pacote de um run (o que se preparou e o modelo, num ficheiro `.mllab`), visto
 * antes de existir, e a biblioteca dos modelos guardados, onde também se carrega um modelo já
 * treinado (ADR 0039).
 *
 * Em cima, numa faixa, escolhe-se o que vai no ficheiro: o run, o nome e as partes que se podem
 * deixar de fora (o backend diz quais são, o que levam e porque é que um run não tem alguma). Em
 * baixo está o próprio ficheiro: as partes à esquerda, que mudam quando muda uma opção, e a
 * escolhida à direita, como é. A pré-visualização e o guardar saem do mesmo código no backend,
 * por isso o que se vê é o que se grava.
 *
 * "Guardar o modelo" põe o pacote na biblioteca (e o run no histórico, com o modelo) e descarrega
 * o ficheiro.
 */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { Check, Download, LoaderCircle } from "lucide-react";
import { useEffect, useId, useState, type ReactNode } from "react";
import { toast } from "sonner";

import { ApiRequestError } from "@/api/errors";
import type {
  PackageCandidate,
  PackageChoice,
  PackageField,
  PackageOptions,
  PackageRequest,
  SavedPackage,
} from "@/api/generated";
import {
  packagesCandidatesOptions,
  packagesOptionsOptions,
  packagesSaveMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { packagesPreview } from "@/api/generated/sdk.gen";
import { Explain } from "@/components/explain/explain";
import { NoticeList } from "@/components/notices/notice-list";
import { ErrorNote, NeedsStep, NextStep, Panel, Proceed } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useDebounced } from "@/hooks/use-debounced";
import { formatBytes, formatCount, formatMoment } from "@/lib/format";
import { useSession } from "@/state/session";

import { invalidatePackages, packageFileUrl } from "./cache";
import { Library } from "./library";
import { PartContent, PartList } from "./package-viewer";

/** As ideias por trás do ecrã (os temas do "Porquê?"). */
const TOPICS = [
  "model_package",
  "model_card",
  "pickle_safety",
  "onnx_format",
  "memorized_rows",
  "package_versions",
] as const;

/** A pausa antes de pedir a pré-visualização outra vez, depois de mudar uma opção. */
const SETTLE_MS = 300;

/** Os erros de um run que não se pode guardar (e que se resolvem noutro ecrã). */
const BLOCKED = new Set(["run_not_found", "run_not_in_memory", "training_in_progress"]);

export function SaveScreen({ runId }: { runId: string | null }) {
  return (
    <Page>
      {runId === null ? <NoRun /> : <SaveRun runId={runId} />}
      <Library />
      <NextStep>
        <Proceed to="/experiencias">Comparar no Histórico</Proceed>
      </NextStep>
    </Page>
  );
}

/** O cabeçalho, com a ação do ecrã quando há um run para guardar. */
function Header({ children }: { children?: ReactNode }) {
  return (
    <PageHeader title={SCREENS.save.title} description={SCREENS.save.description} topics={TOPICS}>
      {children}
    </PageHeader>
  );
}

/** Sem um run aberto: escolhe-se um do histórico, ou diz-se como treinar um. */
function NoRun() {
  const candidates = useQuery(packagesCandidatesOptions());
  const available = candidates.data?.filter((candidate) => candidate.available) ?? [];
  return (
    <>
      <Header />
      <ErrorNote error={candidates.error} className="mt-10" />
      {candidates.data === undefined ? (
        candidates.error === null && <Skeleton className="mt-10 h-24 w-full max-w-md" />
      ) : available.length === 0 ? (
        <NeedsStep
          message="Para guardar um modelo, primeiro é preciso treiná-lo. Um modelo que já foi treinado (um ficheiro .mllab) carrega-se em baixo, na biblioteca."
          to="/modelos"
          action="Escolher o modelo"
        />
      ) : (
        <div className="mt-10 max-w-md">
          <RunPicker runId={null} candidates={available} />
        </div>
      )}
    </>
  );
}

function SaveRun({ runId }: { runId: string }) {
  const openRun = useSession((state) => state.openRun);
  const options = useQuery(packagesOptionsOptions({ path: { run_id: runId } }));

  // O run que se guarda passa a ser o último (o Treinar e o Explicar voltam a ele).
  useEffect(() => {
    openRun(runId);
  }, [runId, openRun]);

  if (options.error !== null) {
    return (
      <>
        <Header />
        <Blocked error={options.error} runId={runId} />
      </>
    );
  }
  if (options.data === undefined) {
    return (
      <>
        <Header />
        <Skeleton className="mt-10 h-[32rem] w-full" />
      </>
    );
  }
  return <SaveForm options={options.data} />;
}

/** Um run que não se pode guardar: diz porquê (a mensagem é a da API) e leva onde se resolve. */
function Blocked({ error, runId }: { error: unknown; runId: string }) {
  const candidates = useQuery(packagesCandidatesOptions());
  if (!(error instanceof ApiRequestError) || !BLOCKED.has(error.code)) {
    return <ErrorNote error={error} className="mt-10" />;
  }
  const code = error.code;
  const available = candidates.data?.filter((candidate) => candidate.available) ?? [];
  return (
    <div className="mt-10 flex max-w-2xl flex-col items-start gap-6">
      <div className="flex flex-col items-start gap-4 panel p-6">
        <p className="text-reading text-soft">{error.message}</p>
        <Button asChild variant="outline">
          {code === "training_in_progress" ? (
            <Link to="/treinar/$runId" params={{ runId }}>
              Acompanhar o treino
            </Link>
          ) : (
            <Link to="/experiencias">Abrir as experiências</Link>
          )}
        </Button>
      </div>
      {available.length > 0 && (
        <div className="w-full max-w-md">
          <RunPicker runId={null} candidates={available} />
        </div>
      )}
    </div>
  );
}

/** O run a guardar: os do histórico que têm o modelo à mão. Escolher outro abre o dele. */
function RunPicker({
  runId,
  candidates,
  current,
}: {
  runId: string | null;
  candidates: readonly PackageCandidate[];
  /** O nome do run aberto, quando ele não está na lista (por exemplo, ainda não registado). */
  current?: string;
}) {
  const navigate = useNavigate();
  const listed = candidates.some((candidate) => candidate.run_id === runId);
  return (
    <Field label="Run" htmlFor="guardar-run">
      <Select
        value={runId ?? undefined}
        onValueChange={(chosen) => {
          void navigate({ to: "/guardar/$runId", params: { runId: chosen } });
        }}
      >
        <SelectTrigger id="guardar-run" className="w-full">
          <SelectValue placeholder="Escolhe um run do histórico" />
        </SelectTrigger>
        <SelectContent>
          {!listed && runId !== null && <SelectItem value={runId}>{current ?? runId}</SelectItem>}
          {candidates.map((candidate) => (
            <SelectItem key={candidate.run_id} value={candidate.run_id}>
              {candidate.name}
              <span className="font-mono text-xs text-dim tabular-nums">
                {formatMoment(candidate.created_at)}
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </Field>
  );
}

type Included = Record<PackageField, boolean>;

/** O que vai por omissão: o que o backend sugere, e só o que o run tem para dar. */
function suggested(choices: readonly PackageChoice[]): Included {
  const included: Included = {
    include_model: false,
    include_examples: false,
    include_training: false,
  };
  for (const choice of choices) {
    included[choice.key] = choice.available && choice.included;
  }
  return included;
}

/** O que guardar, à esquerda, e o ficheiro que isso dá, à direita. */
function SaveForm({ options }: { options: PackageOptions }) {
  const queryClient = useQueryClient();
  const candidates = useQuery(packagesCandidatesOptions());
  const [name, setName] = useState(options.name);
  const [included, setIncluded] = useState(() => suggested(options.choices));
  const [examples, setExamples] = useState(options.examples);

  const request: PackageRequest = {
    name: name.trim() === "" ? null : name.trim(),
    ...included,
    examples,
  };
  const asked = JSON.stringify(request);
  const settled = useDebounced(asked, SETTLE_MS);
  const preview = useQuery({
    queryKey: ["packages-preview", options.run_id, settled],
    queryFn: async ({ signal }) => {
      const { data } = await packagesPreview({
        path: { run_id: options.run_id },
        body: JSON.parse(settled) as PackageRequest,
        signal,
        throwOnError: true,
      });
      return data;
    },
    placeholderData: keepPreviousData,
    staleTime: "static",
  });

  const save = useMutation({
    ...packagesSaveMutation(),
    onSuccess: (pkg) => {
      invalidatePackages(queryClient);
      download(pkg);
      toast.success(`«${pkg.name}» está na biblioteca (${formatBytes(pkg.size)}).`);
    },
  });
  // A parte aberta: começa na ficha (a primeira) e mantém-se enquanto existir no ficheiro.
  const [chosen, setChosen] = useState<string | null>(null);
  const parts = preview.data?.parts ?? [];
  const current = parts.find((part) => part.path === chosen) ?? parts[0];
  // O visto fica enquanto o pedido for o que se guardou: mudar uma opção pede outro ficheiro.
  const saved = save.isSuccess && JSON.stringify(save.variables.body) === asked;
  const available = candidates.data?.filter((candidate) => candidate.available) ?? [];

  return (
    <>
      <Header>
        <div className="flex flex-col items-start gap-2 @3xl/main:items-end">
          <Button
            variant="proceed"
            size="proceed"
            disabled={save.isPending || saved}
            onClick={() => {
              save.mutate({ path: { run_id: options.run_id }, body: request });
            }}
          >
            {save.isPending ? (
              <LoaderCircle
                aria-hidden="true"
                className="animate-spin motion-reduce:animate-none"
              />
            ) : saved ? (
              <Check aria-hidden="true" />
            ) : (
              <Download aria-hidden="true" />
            )}
            {saved ? "Guardado" : "Guardar o modelo"}
          </Button>
          <ErrorNote error={save.error} />
        </div>
      </Header>

      {/*
        Em cima, numa faixa, o que vai no ficheiro; em baixo, o ficheiro: as partes à esquerda e a
        escolhida à direita, lado a lado, para se verem sempre as duas juntas.
      */}
      <div className="mt-10 flex flex-col gap-12">
        <Panel title="O que guardar" as="h2">
          <div className="flex flex-col gap-6">
            {/* `grid-cols-1` encolhe: sem ele, o nome comprido de um run alargava a página. */}
            <div className="grid grid-cols-1 gap-6 @3xl/main:grid-cols-2">
              <RunPicker runId={options.run_id} candidates={available} current={options.name} />
              <Field label="Nome" htmlFor="guardar-nome">
                <Input
                  id="guardar-nome"
                  name="nome"
                  autoComplete="off"
                  maxLength={120}
                  value={name}
                  onChange={(event) => {
                    setName(event.target.value);
                  }}
                />
              </Field>
            </div>
            <div className="grid grid-cols-1 gap-x-10 gap-y-4 @3xl/main:grid-cols-3 @3xl/main:items-start">
              {options.choices.map((choice) => (
                <ChoiceRow
                  key={choice.key}
                  choice={choice}
                  on={included[choice.key]}
                  onChange={(on) => {
                    setIncluded((current) => ({ ...current, [choice.key]: on }));
                  }}
                >
                  {choice.key === "include_examples" &&
                    included.include_examples &&
                    options.max_examples > 1 && (
                      <ExampleCount
                        value={examples}
                        max={options.max_examples}
                        onChange={setExamples}
                      />
                    )}
                </ChoiceRow>
              ))}
            </div>
            {preview.data !== undefined && <NoticeList notices={preview.data.notices} />}
          </div>
        </Panel>

        <Panel
          title="O ficheiro"
          as="h2"
          actions={
            preview.data !== undefined && (
              <p className="font-mono text-xs text-dim tabular-nums">
                <span className="text-strong">{preview.data.file_name}</span> ·{" "}
                {formatBytes(preview.data.size)} sem comprimir ·{" "}
                {formatCount(preview.data.parts.length)} ficheiros
              </p>
            )
          }
        >
          <ErrorNote error={preview.error} />
          {preview.data === undefined || current === undefined ? (
            preview.error === null && <Skeleton className="h-[32rem] w-full" />
          ) : (
            <div className="grid grid-cols-1 gap-4 @4xl/main:grid-cols-[18rem_minmax(0,1fr)] @4xl/main:items-start">
              <PartList
                parts={preview.data.parts}
                chosen={current.path}
                onChoose={setChosen}
                label="As partes do ficheiro"
              />
              <PartContent part={current} scrollClassName="max-h-[36rem]" />
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}

/** Uma parte que se pode deixar de fora: o nome, o "?" com o que leva, e o interruptor. */
function ChoiceRow({
  choice,
  on,
  onChange,
  children,
}: {
  choice: PackageChoice;
  on: boolean;
  onChange: (on: boolean) => void;
  children?: ReactNode;
}) {
  const id = `guardar-${choice.key.replaceAll("_", "-")}`;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-4">
        <div className="flex min-w-0 items-center gap-1">
          <Label htmlFor={id} className={choice.available ? "text-sm" : "text-sm text-dim"}>
            {choice.title}
          </Label>
          <Explain title={choice.title}>
            <p>{choice.description}</p>
            {choice.reason !== null && <p className="mt-2">{choice.reason}</p>}
          </Explain>
        </div>
        <Switch id={id} checked={on} disabled={!choice.available} onCheckedChange={onChange} />
      </div>
      {children}
    </div>
  );
}

/** Quantas linhas de exemplo vão no ficheiro. */
function ExampleCount({
  value,
  max,
  onChange,
}: {
  value: number;
  max: number;
  onChange: (value: number) => void;
}) {
  const labelId = useId();
  const text = value === 1 ? "1 linha" : `${formatCount(value)} linhas`;
  return (
    <div className="flex items-center gap-4 pl-0.5">
      <span id={labelId} className="sr-only">
        Quantas linhas de exemplo
      </span>
      <Slider
        min={1}
        max={max}
        step={1}
        value={[value]}
        onValueChange={([next]) => {
          if (next !== undefined) {
            onChange(next);
          }
        }}
        aria-labelledby={labelId}
        valueText={text}
        className="flex-1"
      />
      <span className="w-20 shrink-0 text-right font-mono text-xs text-soft tabular-nums">
        {text}
      </span>
    </div>
  );
}

interface FieldProps {
  label: string;
  htmlFor: string;
  children: ReactNode;
}

function Field({ label, htmlFor, children }: FieldProps) {
  return (
    <div className="flex flex-col gap-2">
      <Label htmlFor={htmlFor} className="text-sm text-soft">
        {label}
      </Label>
      {children}
    </div>
  );
}

/** Descarrega o ficheiro de um pacote da biblioteca (uma ligação normal, clicada por nós). */
function download(pkg: SavedPackage): void {
  const link = document.createElement("a");
  link.href = packageFileUrl(pkg.package_id);
  link.download = pkg.file_name;
  document.body.append(link);
  link.click();
  link.remove();
}
