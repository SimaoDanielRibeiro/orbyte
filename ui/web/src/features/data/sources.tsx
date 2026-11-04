/**
 * As três fontes de dados do ecrã Dados, umas por baixo das outras (ADR 0037): os datasets de
 * exemplo, um ficheiro do computador (numa zona grande, onde se larga o ficheiro) e os geradores
 * de dados sintéticos. Cada exemplo é um cartão com os seus números; a descrição e a fonte ficam
 * no "?".
 */
import { useMutation, useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { FileSpreadsheet, FileUp, LoaderCircle } from "lucide-react";
import { useState, type DragEvent } from "react";

import type { SourceSpec } from "@/api/generated";
import {
  datasetsGenerateMutation,
  datasetsGeneratorsOptions,
  datasetsLoadSampleMutation,
  datasetsSamplesOptions,
  datasetsUploadMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { schemaDefaults, type Values } from "@/components/forms/schema";
import { SchemaForm } from "@/components/forms/schema-form";
import { ErrorNote, Facts } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatBytes, formatCount } from "@/lib/format";
import { isValidSeed, MAX_SEED } from "@/lib/seed";

import { useDatasetLoaded } from "./use-dataset-loaded";

/** Os ficheiros que a leitura do backend aceita (`core/data/io/file_source.py`). */
const ACCEPTED = ".csv,.tsv,.txt,.gz,.xlsx,.xlsm,.xls,.json,.jsonl,.ndjson,.parquet,.pq";

/** Os datasets de exemplo, cada um com o que tem e de onde vem. */
export function SampleSource({ open }: { open: SourceSpec | undefined }) {
  const label = useLabel();
  const loaded = useDatasetLoaded();
  const samples = useQuery({ ...datasetsSamplesOptions(), staleTime: "static" });
  const load = useMutation({ ...datasetsLoadSampleMutation(), onSuccess: loaded });

  if (samples.error !== null) {
    return <ErrorNote error={samples.error} />;
  }
  if (samples.data === undefined) {
    return <ListSkeleton />;
  }
  return (
    <div className="flex flex-col gap-5">
      <ul className="grid gap-4 @3xl/main:grid-cols-2">
        {samples.data.map((sample) => {
          const pending = load.isPending && load.variables.path.name === sample.name;
          const isOpen = open?.kind === "sample" && open.params?.name === sample.name;
          return (
            <li
              key={sample.name}
              // O exemplo aberto tem o contorno verde (o botão diz "Carregar outra vez").
              className={cn(
                "flex flex-col gap-5 panel p-5 transition-colors hover:border-[#333]",
                isOpen && "border-positive/70 hover:border-positive",
              )}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex min-w-0 flex-wrap items-center gap-x-1 gap-y-1">
                  <h4 className="text-base font-medium text-strong">{sample.title}</h4>
                  <Explain title={sample.title}>
                    <p>{sample.description}</p>
                    <p className="mt-2 text-xs text-dim">Source: {sample.source}</p>
                  </Explain>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={load.isPending}
                  onClick={() => {
                    load.mutate({ path: { name: sample.name } });
                  }}
                  aria-label={`Load ${sample.title}`}
                >
                  {pending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
                  {isOpen ? "Load again" : "Load"}
                </Button>
              </div>
              <Facts
                size="sm"
                facts={[
                  { label: "Rows", value: formatCount(sample.n_rows) },
                  { label: "Columns", value: formatCount(sample.n_columns) },
                  { label: "Usual target", value: sample.target },
                  { label: "Task", value: label("TaskType", sample.task_type) },
                ]}
              />
            </li>
          );
        })}
      </ul>
      <ErrorNote error={load.error} />
    </div>
  );
}

/**
 * Um ficheiro do computador: uma zona grande, onde se larga o ficheiro ou se clica para o escolher.
 * Com o ficheiro escolhido, a zona mostra o nome e o tamanho, e o botão de o carregar.
 */
export function FileSource() {
  const loaded = useDatasetLoaded();
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const upload = useMutation({
    ...datasetsUploadMutation(),
    onSuccess: (dataset) => {
      loaded(dataset);
      setFile(null);
    },
  });

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files.item(0);
    if (dropped !== null) {
      setFile(dropped);
      upload.reset();
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => {
          setDragging(false);
        }}
        onDrop={drop}
        className={cn(
          "relative flex min-h-64 flex-col items-center justify-center gap-6 rounded-2xl border border-dashed px-6 py-12 text-center transition-colors",
          "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50",
          dragging
            ? "border-strong/70 bg-surface-high"
            : "border-[#3a3a3a] bg-ground hover:border-[#5a5a5a] hover:bg-[#151515]",
        )}
      >
        {/* A zona toda escolhe o ficheiro; só o botão de carregar fica por cima dela. */}
        <label htmlFor="dados-ficheiro" className="absolute inset-0 cursor-pointer rounded-2xl">
          <span className="sr-only">Choose a file from your computer</span>
        </label>
        <span
          aria-hidden="true"
          className={cn(
            "pointer-events-none flex size-16 items-center justify-center rounded-2xl border bg-surface shadow-[inset_0_1px_0_rgb(255_255_255/0.05)] transition-transform duration-300 motion-reduce:transition-none",
            dragging ? "scale-110 border-[#5a5a5a]" : "border-[#333]",
          )}
        >
          {file === null ? (
            <FileUp className="size-7 text-soft" />
          ) : (
            <FileSpreadsheet className="size-7 text-strong" />
          )}
        </span>
        <div className="pointer-events-none flex flex-col gap-2">
          <p className="text-lg text-strong">
            {file === null
              ? dragging
                ? "Drop the file here"
                : "Drag a file here, or click to choose one"
              : file.name}
          </p>
          <p className="font-mono text-xs text-dim tabular-nums">
            {file === null
              ? "CSV, TSV, TXT, GZ, Excel (XLSX, XLS), JSON or Parquet"
              : `${fileSize(file.size)}, ready to load`}
          </p>
        </div>
        {file !== null && (
          <Button
            size="lg"
            className="relative z-10 px-5"
            disabled={upload.isPending}
            onClick={() => {
              upload.mutate({ body: { file } });
            }}
          >
            {upload.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
            Load file
          </Button>
        )}
        <input
          id="dados-ficheiro"
          type="file"
          accept={ACCEPTED}
          className="sr-only"
          onChange={(event) => {
            setFile(event.target.files?.item(0) ?? null);
            upload.reset();
          }}
        />
      </div>
      <ErrorNote error={upload.error} />
    </div>
  );
}

/** Os geradores de dados sintéticos: escolhe-se a forma dos dados, o tamanho e o ruído. */
export function SyntheticSource() {
  const label = useLabel();
  const loaded = useDatasetLoaded();
  const generators = useQuery({ ...datasetsGeneratorsOptions(), staleTime: "static" });
  const [chosen, setChosen] = useState<string | null>(null);
  // Os parâmetros de cada gerador ficam guardados ao trocar de um para outro.
  const [parameters, setParameters] = useState<Record<string, Values>>({});
  const [seed, setSeed] = useState(42);
  const generate = useMutation({ ...datasetsGenerateMutation(), onSuccess: loaded });

  if (generators.error !== null) {
    return <ErrorNote error={generators.error} />;
  }
  if (generators.data === undefined) {
    return <ListSkeleton />;
  }
  const generator =
    generators.data.find((item) => item.name === chosen) ?? generators.data[0] ?? null;
  if (generator === null) {
    return null;
  }
  const values = parameters[generator.name] ?? schemaDefaults(generator.parameters_schema);
  const validSeed = isValidSeed(seed);

  return (
    <div className="grid gap-x-12 gap-y-10 @3xl/main:grid-cols-[16rem_minmax(0,1fr)] @6xl/main:grid-cols-[18rem_minmax(0,44rem)]">
      <ToggleGroup
        type="single"
        orientation="vertical"
        spacing={1}
        value={generator.name}
        onValueChange={(name) => {
          if (name !== "") {
            setChosen(name);
            generate.reset();
          }
        }}
        aria-label="Generator"
        className="w-full gap-0.5 panel p-2"
      >
        {generators.data.map((item) => (
          <ToggleGroupItem
            key={item.name}
            value={item.name}
            className="h-auto w-full flex-col items-start gap-0.5 border border-transparent px-3 py-2 text-left data-[state=on]:border-[#333]"
          >
            <span className="text-sm">{item.title}</span>
            <span className="text-xs font-normal text-dim">
              {label("TaskType", item.task_type)}
            </span>
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <div className="flex min-w-0 flex-col gap-6">
        <div className="flex items-center gap-1">
          <h4 className="text-base font-medium">{generator.title}</h4>
          <Explain title={generator.title}>{generator.description}</Explain>
        </div>
        <SchemaForm
          schema={generator.parameters_schema}
          values={values}
          onChange={(next) => {
            setParameters((current) => ({ ...current, [generator.name]: next }));
          }}
          idPrefix={`gerador-${generator.name}`}
        />
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-1">
            <Label htmlFor="gerador-seed" className="text-sm">
              Seed
            </Label>
            <Explain title="Seed">The same seed always gives the same points.</Explain>
          </div>
          <Input
            id="gerador-seed"
            name="seed"
            autoComplete="off"
            type="number"
            inputMode="numeric"
            min={0}
            max={MAX_SEED}
            step={1}
            value={Number.isNaN(seed) ? "" : seed}
            onChange={(event) => {
              setSeed(event.target.valueAsNumber);
            }}
            aria-invalid={!validSeed}
            className="w-40 tabular-nums"
          />
        </div>
        <div>
          <Button
            disabled={generate.isPending || !validSeed}
            onClick={() => {
              generate.mutate({
                body: {
                  generator: generator.name,
                  parameters: numbersOnly(values),
                  seed,
                },
              });
            }}
          >
            {generate.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
            Generate data
          </Button>
        </div>
        <ErrorNote error={generate.error} />
      </div>
    </div>
  );
}

function ListSkeleton() {
  return (
    <div
      className="grid gap-x-12 gap-y-10 @3xl/main:grid-cols-2"
      role="status"
      aria-busy="true"
      aria-label="Loading"
    >
      {[0, 1].map((index) => (
        <div key={index} className="flex flex-col gap-3">
          <Skeleton className="h-5 w-40" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      ))}
    </div>
  );
}

/** Os parâmetros de um gerador são todos números (o pedido só aceita números). */
function numbersOnly(values: Values): Record<string, number> {
  return Object.fromEntries(
    Object.entries(values).filter(
      (entry): entry is [string, number] => typeof entry[1] === "number",
    ),
  );
}

function fileSize(bytes: number): string {
  return formatBytes(Math.max(bytes, 1024));
}
