/**
 * O ecrã Guardar: o pacote de um run (o que se preparou e o modelo, num ficheiro `.mllab`), visto
 * antes de existir, e a biblioteca dos modelos guardados, onde também se carrega um modelo já
 * treinado (ADR 0039).
 *
 * O guardar faz-se em três passos numerados (ADR 0051): o run e o nome; as partes que se podem
 * deixar de fora, cada uma com o que leva à vista (o backend diz quais são, o que levam e porque
 * é que um run não tem alguma); e o próprio ficheiro, as partes à esquerda (que mudam quando muda
 * uma opção) e a escolhida à direita, como é. Ao lado dos dois primeiros passos, o resumo do
 * pacote diz o que ele permite e o que convém saber. A pré-visualização e o guardar saem do
 * mesmo código no backend, por isso o que se vê é o que se grava.
 *
 * "Guardar o modelo" põe o pacote na biblioteca (e o run no histórico, com o modelo) e descarrega
 * o ficheiro.
 */
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { cn } from "cn";
import { Check, Download, LoaderCircle, Minus } from "lucide-react";
import { useEffect, useId, useState, type ReactNode } from "react";
import { toast } from "sonner";

import { ApiRequestError } from "@/api/errors";
import type {
  PackageCandidate,
  PackageChoice,
  PackageField,
  PackageOptions,
  PackagePreview,
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
import { ErrorNote, NeedsStep } from "@/components/shell/blocks";
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
          message="To save a model, you need to train it first. A model that was already trained (a .mllab file) is loaded below, in the library."
          to="/models"
          action="Choose the model"
        />
      ) : (
        <div className="mt-10 max-w-xl">
          <Step number={1} title="Choose the run to save" topic="model_package">
            <RunPicker runId={null} candidates={available} />
          </Step>
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
            <Link to="/train/$runId" params={{ runId }}>
              Follow the training
            </Link>
          ) : (
            <Link to="/history">Open the history</Link>
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
          void navigate({ to: "/save/$runId", params: { runId: chosen } });
        }}
      >
        <SelectTrigger id="guardar-run" className="w-full">
          <SelectValue placeholder="Choose a run from the history" />
        </SelectTrigger>
        <SelectContent>
          {!listed && runId !== null && <SelectItem value={runId}>{current ?? runId}</SelectItem>}
          {candidates.map((candidate) => (
            <SelectItem key={candidate.run_id} value={candidate.run_id}>
              {/* O espaço separa a data no botão, onde o texto corre numa linha com reticências. */}
              {candidate.name}{" "}
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
      toast.success(`“${pkg.name}” is in the library (${formatBytes(pkg.size)}).`);
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
          {/* A ação do ecrã e, ao lado, o passo seguinte: comparar o run no Histórico. */}
          <div className="flex flex-wrap items-center gap-2">
            <Button asChild variant="outline" size="proceed">
              <Link to="/history">Compare in History</Link>
            </Button>
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
              {saved ? "Saved" : "Save the model"}
            </Button>
          </div>
          <ErrorNote error={save.error} />
        </div>
      </Header>

      {/*
        Três passos, pela ordem em que se decide: o run e o nome, o que vai no ficheiro, e o
        próprio ficheiro (as partes à esquerda e a escolhida à direita). Ao lado dos dois
        primeiros, o resumo do pacote acompanha cada escolha.
      */}
      <div className="mt-10 grid grid-cols-1 gap-x-12 gap-y-12 @5xl/main:grid-cols-[minmax(0,1fr)_minmax(0,22rem)]">
        <div className="flex min-w-0 flex-col gap-12">
          <Step number={1} title="The run and its name">
            {/* `grid-cols-1` encolhe: sem ele, o nome comprido de um run alargava a página. */}
            <div className="grid grid-cols-1 gap-5">
              <RunPicker runId={options.run_id} candidates={available} current={options.name} />
              <Field label="Name" htmlFor="guardar-nome">
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
          </Step>
          <Step number={2} title="What goes in the file" topic="model_package">
            <ul className="flex flex-col divide-y divide-border rounded-2xl border bg-surface">
              {options.choices.map((choice) => (
                <li key={choice.key}>
                  <ChoiceRow
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
                </li>
              ))}
            </ul>
          </Step>
        </div>
        <PackageSummary preview={preview.data} error={preview.error} />
      </div>

      <Step
        number={3}
        title="Check the file"
        className="mt-14"
        actions={
          preview.data !== undefined && (
            <p className="font-mono text-xs text-dim tabular-nums">
              {formatCount(preview.data.parts.length)} files · {formatBytes(preview.data.size)}{" "}
              uncompressed
            </p>
          )
        }
      >
        {preview.data === undefined || current === undefined ? (
          preview.error === null && <Skeleton className="h-[32rem] w-full" />
        ) : (
          <div className="grid grid-cols-1 gap-4 @4xl/main:grid-cols-[18rem_minmax(0,1fr)] @4xl/main:items-start">
            <PartList
              parts={preview.data.parts}
              chosen={current.path}
              onChoose={setChosen}
              label="The file's parts"
            />
            <PartContent part={current} scrollClassName="max-h-[36rem]" />
          </div>
        )}
      </Step>
    </>
  );
}

interface StepProps {
  number: number;
  title: string;
  /** O tema do backend que explica o passo (no "?" ao lado do título). */
  topic?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}

/** Um passo do guardar: o número, o título e o que se decide nele. */
function Step({ number, title, topic, actions, children, className }: StepProps) {
  const id = useId();
  return (
    <section aria-labelledby={id} className={cn("flex min-w-0 flex-col gap-5", className)}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span
            aria-hidden="true"
            className="grid size-7 shrink-0 place-items-center rounded-full border border-border font-mono text-xs text-soft tabular-nums"
          >
            {number}
          </span>
          <h2 id={id} className="text-lg">
            {title}
          </h2>
          {topic !== undefined && <Explain topic={topic} />}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

/**
 * O resumo do pacote: o nome do ficheiro e o que ele permite (prever, correr fora do Orbyte), e
 * os avisos do backend. Acompanha a página num ecrã largo, ao lado das escolhas.
 */
function PackageSummary({
  preview,
  error,
}: {
  preview: PackagePreview | undefined;
  error: unknown;
}) {
  const id = useId();
  return (
    <section
      aria-labelledby={id}
      className="flex min-w-0 flex-col gap-5 self-start panel p-5 @5xl/main:sticky @5xl/main:top-6"
    >
      <h2 id={id} className="text-sm font-medium tracking-normal text-soft">
        The package
      </h2>
      <ErrorNote error={error} />
      {preview === undefined ? (
        error === null && <Skeleton className="h-40 w-full" />
      ) : (
        <>
          <div className="flex flex-col gap-1">
            <p className="font-mono text-sm break-words text-strong">{preview.file_name}</p>
            <p className="font-mono text-xs text-dim tabular-nums">
              {formatBytes(preview.size)} uncompressed
            </p>
          </div>
          <ul className="flex flex-col gap-2.5 text-sm">
            <Capability on={preview.has_model} topic="pickle_safety">
              {preview.has_model ? "Predicts new data" : "No model: it can only be read"}
            </Capability>
            <Capability on={preview.has_onnx} topic="onnx_format">
              {preview.has_onnx ? "Also runs outside Orbyte (ONNX)" : "No ONNX copy"}
            </Capability>
          </ul>
          {preview.notices.length > 0 && <NoticeList notices={preview.notices} />}
        </>
      )}
    </section>
  );
}

/** O que o pacote permite, ou não: o visto (ou o traço), o texto e o "?" que o explica. */
function Capability({ on, topic, children }: { on: boolean; topic: string; children: ReactNode }) {
  const Icon = on ? Check : Minus;
  return (
    <li className="flex items-center gap-2.5">
      <Icon aria-hidden="true" className={cn("size-4 shrink-0", on ? "text-strong" : "text-dim")} />
      <span className={on ? "text-foreground" : "text-soft"}>{children}</span>
      <Explain topic={topic} />
    </li>
  );
}

/** Uma parte que se pode deixar de fora: o nome, o que leva (à vista), e o interruptor. */
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
    <div className="flex flex-col gap-4 px-5 py-4">
      <div className="flex items-start justify-between gap-6">
        <div className="flex min-w-0 flex-col gap-1">
          <Label
            htmlFor={id}
            className={cn("text-base font-medium", !choice.available && "text-dim")}
          >
            {choice.title}
          </Label>
          <p id={`${id}-descricao`} className="text-sm leading-relaxed text-soft">
            {choice.description}
            {choice.reason !== null && <> {choice.reason}</>}
          </p>
        </div>
        <Switch
          id={id}
          checked={on}
          disabled={!choice.available}
          onCheckedChange={onChange}
          aria-describedby={`${id}-descricao`}
          className="mt-1"
        />
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
  const text = value === 1 ? "1 row" : `${formatCount(value)} rows`;
  return (
    <div className="flex items-center gap-4 pl-0.5">
      <span id={labelId} className="sr-only">
        How many example rows
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
