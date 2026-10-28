/**
 * As três fontes de dados do ecrã Dados: os datasets de exemplo, um ficheiro do computador e os
 * geradores de dados sintéticos. Cada exemplo é um cartão com os seus números; a descrição e a
 * fonte ficam no "?".
 */
import { useMutation, useQuery } from "@tanstack/react-query";
import { cn } from "cn";
import { FileUp, LoaderCircle } from "lucide-react";
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
import { formatCount, formatNumber } from "@/lib/format";
import { isValidSeed, MAX_SEED } from "@/lib/seed";

import { useDatasetLoaded } from "./use-dataset-loaded";

/** Os ficheiros que a leitura do backend aceita (`core/data/io/file_source.py`). */
const ACCEPTED = ".csv,.tsv,.txt,.gz,.xlsx,.xlsm";

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
            <li key={sample.name} className="flex flex-col gap-5 border-t pt-5">
              <div className="flex items-start justify-between gap-4">
                <div className="flex min-w-0 items-center gap-1">
                  <h3 className="text-base font-medium">{sample.title}</h3>
                  <Explain title={sample.title}>
                    <p>{sample.description}</p>
                    <p className="mt-2 font-mono text-xs text-dim">Fonte: {sample.source}</p>
                  </Explain>
                  {isOpen && (
                    <span className="ml-2 flex items-center gap-1.5 text-xs text-positive">
                      <span aria-hidden="true" className="size-1.5 rounded-full bg-positive" />
                      aberto
                    </span>
                  )}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={load.isPending}
                  onClick={() => {
                    load.mutate({ path: { name: sample.name } });
                  }}
                  aria-label={`Carregar ${sample.title}`}
                >
                  {pending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
                  {isOpen ? "Carregar outra vez" : "Carregar"}
                </Button>
              </div>
              <Facts
                size="sm"
                facts={[
                  { label: "Linhas", value: formatCount(sample.n_rows) },
                  { label: "Colunas", value: formatCount(sample.n_columns) },
                  { label: "Alvo habitual", value: sample.target },
                  { label: "Tarefa", value: label("TaskType", sample.task_type) },
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

/** Um ficheiro do computador: arrasta-se para a zona ou escolhe-se com o botão. */
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

  function drop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    const dropped = event.dataTransfer.files.item(0);
    if (dropped !== null) {
      setFile(dropped);
      upload.reset();
    }
  }

  return (
    <div className="flex max-w-2xl flex-col gap-5 border-t pt-5">
      <label
        htmlFor="dados-ficheiro"
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => {
          setDragging(false);
        }}
        onDrop={drop}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-2 rounded-xl border border-dashed border-input bg-ground px-6 py-12 text-center transition-colors",
          "focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50 hover:border-[#555]",
          dragging && "border-strong/60 bg-surface-high",
        )}
      >
        <FileUp aria-hidden="true" className="size-6 text-dim" />
        <span className="text-[0.9375rem] text-strong">
          {file === null ? "Arrasta um ficheiro para aqui, ou escolhe-o no computador" : file.name}
        </span>
        <span className="font-mono text-xs text-dim tabular-nums">
          {file === null ? "CSV, TSV, TXT, GZ, XLSX ou XLSM" : fileSize(file.size)}
        </span>
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
      </label>
      <div className="flex items-center gap-2">
        <Button
          disabled={file === null || upload.isPending}
          onClick={() => {
            if (file !== null) {
              upload.mutate({ body: { file } });
            }
          }}
        >
          {upload.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
          Carregar ficheiro
        </Button>
        <Explain topic="upload" />
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
        aria-label="Gerador"
        className="w-full gap-0.5 panel p-2"
      >
        {generators.data.map((item) => (
          <ToggleGroupItem
            key={item.name}
            value={item.name}
            className="h-auto w-full flex-col items-start gap-0.5 border border-transparent px-3 py-2 text-left data-[state=on]:border-[#333]"
          >
            <span className="text-sm">{item.title}</span>
            <span className="font-mono text-[0.6875rem] font-normal text-dim">
              {label("TaskType", item.task_type)}
            </span>
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <div className="flex min-w-0 flex-col gap-6 border-t pt-5">
        <div className="flex items-center gap-1">
          <h3 className="text-base font-medium">{generator.title}</h3>
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
        <div className="flex flex-col gap-2 border-t pt-5">
          <Label htmlFor="gerador-seed" className="text-sm">
            Seed
          </Label>
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
            aria-describedby="gerador-seed-ajuda"
            className="w-40 tabular-nums"
          />
          <p id="gerador-seed-ajuda" className="mono-label">
            A mesma seed dá sempre os mesmos pontos.
          </p>
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
            Gerar dados
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
      aria-label="A carregar"
    >
      {[0, 1].map((index) => (
        <div key={index} className="flex flex-col gap-3 border-t pt-5">
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
  if (bytes < 1024 * 1024) {
    return `${formatCount(Math.max(1, Math.round(bytes / 1024)))} KB`;
  }
  return `${formatNumber(bytes / 1024 ** 2, 1)} MB`;
}
