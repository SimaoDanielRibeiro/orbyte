/**
 * O ecrã Preparar: o alvo e a tarefa, a limpeza, a divisão e as transformações, e o que sai.
 *
 * A preparação volta a correr sempre que um controlo muda (depois de uma pausa curta), para se
 * ver logo o efeito de cada escolha na linhagem, nas partes e na projeção. A que corre bem fica
 * na sessão: é com ela que se treina. Num ecrã largo, o resultado fica ao lado dos controlos e
 * acompanha a página.
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { useEffect } from "react";

import type { DatasetSummary, PrepareRequest } from "@/api/generated";
import {
  datasetsSuggestCleaningOptions,
  datasetsSuggestTaskOptions,
  datasetsTaskOptionsOptions,
} from "@/api/generated/@tanstack/react-query.gen";
import { datasetsPrepare } from "@/api/generated/sdk.gen";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { Proceed } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import { RequireDataset } from "@/features/data/require-dataset";
import { useDebounced } from "@/hooks/use-debounced";
import { useSession } from "@/state/session";

import { buildConfig, effectiveTask, initialChoices, type PrepareChoices } from "./choices";
import { PrepareForm } from "./prepare-form";
import { PrepareResult } from "./prepare-result";

/** O tempo sem mexer nos controlos antes de preparar outra vez. */
const SETTLE_MS = 350;

export function PrepareScreen() {
  const preparation = useSession((state) => state.preparation);
  const datasetId = useSession((state) => state.datasetId);
  const ready = preparation !== null && preparation.datasetId === datasetId;
  return (
    <Page>
      <PageHeader
        title={SCREENS.prepare.title}
        description={SCREENS.prepare.description}
        topics={[
          "cleaning",
          "task",
          "split",
          "transform",
          "leakage",
          "projection",
          "output",
          "lineage",
          "split_classes",
          "pca_loadings",
        ]}
      >
        {/* Daqui, os modelos abrem já com a tarefa preparada (ADR 0051): é para ela que se escolhe.
            Pela barra, abrem com todas. */}
        {ready ? (
          <Button asChild variant="proceed" size="proceed">
            <Link to="/models" search={{ task: preparation.taskType }}>
              Choose the model
            </Link>
          </Button>
        ) : (
          <Proceed to="/models" disabled>
            Choose the model
          </Proceed>
        )}
      </PageHeader>
      <RequireDataset>{(dataset) => <Prepare key={dataset.id} dataset={dataset} />}</RequireDataset>
    </Page>
  );
}

function Prepare({ dataset }: { dataset: DatasetSummary }) {
  const draft = useSession((state) => state.draft);
  const setDraft = useSession((state) => state.setDraft);
  const setPreparation = useSession((state) => state.setPreparation);
  const choices =
    draft?.datasetId === dataset.id ? draft.choices : initialChoices(dataset.suggested_target);
  const change = (next: PrepareChoices) => {
    setDraft({ datasetId: dataset.id, choices: next });
  };

  const path = { dataset_id: dataset.id };
  const target = choices.target === null ? undefined : { target: choices.target };
  // Ao trocar de alvo, as sugestões anteriores ficam à vista até chegarem as novas; mas a
  // configuração só se monta com as sugestões do alvo certo (`isPlaceholderData`).
  const suggestion = useQuery({
    ...datasetsSuggestTaskOptions({ path, query: target }),
    placeholderData: keepPreviousData,
  });
  const cleaning = useQuery({
    ...datasetsSuggestCleaningOptions({ path, query: target }),
    placeholderData: keepPreviousData,
  });
  const taskType = suggestion.data === undefined ? null : effectiveTask(choices, suggestion.data);
  const options = useQuery({
    ...datasetsTaskOptionsOptions({
      query: { task_type: taskType ?? "clustering", target: choices.target },
    }),
    enabled: taskType !== null,
    placeholderData: keepPreviousData,
  });

  const fresh =
    suggestion.data !== undefined &&
    !suggestion.isPlaceholderData &&
    cleaning.data !== undefined &&
    !cleaning.isPlaceholderData &&
    options.data !== undefined &&
    !options.isPlaceholderData;
  const config = fresh ? buildConfig(choices, suggestion.data, cleaning.data, options.data) : null;
  // O pedido vai como texto: assim só muda quando o conteúdo muda (um objeto novo em cada
  // desenho faria o adiamento recomeçar sem parar).
  const request = config === null ? null : JSON.stringify({ config, seed: choices.seed });
  const settled = useDebounced(request, SETTLE_MS);
  const summary = useQuery({
    queryKey: ["preparation", dataset.id, settled],
    queryFn: async ({ signal }) => {
      const { data } = await datasetsPrepare({
        path,
        body: JSON.parse(settled ?? "{}") as PrepareRequest,
        signal,
        throwOnError: true,
      });
      return data;
    },
    enabled: settled !== null,
    placeholderData: keepPreviousData,
  });

  // A preparação que corre bem fica na sessão (é com ela que se treina); a que falha, ou um
  // alvo sem tarefa possível, tira-a: treinar com uma preparação que não está à vista enganava.
  const current = summary.data !== undefined && !summary.isPlaceholderData ? summary.data : null;
  useEffect(() => {
    if (current !== null && settled !== null) {
      const body = JSON.parse(settled) as Required<PrepareRequest>;
      setPreparation({
        datasetId: dataset.id,
        config: body.config,
        seed: body.seed,
        taskType: current.task_type,
      });
    }
  }, [current, settled, dataset.id, setPreparation]);
  const impossible = fresh && config === null;
  useEffect(() => {
    if (summary.isError || impossible) {
      setPreparation(null);
    }
  }, [summary.isError, impossible, setPreparation]);

  const pending = request !== settled || summary.isFetching;
  return (
    <>
      {/* Duas colunas a partir de um portátil (1440 px): os controlos numa coluna estreita e o
          resultado ao lado. Numa só coluna, os controlos não esticam: um interruptor longe do
          seu nome lê-se mal. */}
      <div className="mt-8 grid gap-x-10 gap-y-10 @5xl/main:grid-cols-[minmax(0,24rem)_minmax(0,1fr)] @7xl/main:gap-x-14">
        <div className="max-w-2xl min-w-0 @5xl/main:max-w-none">
          <PrepareForm
            dataset={dataset}
            choices={choices}
            onChange={change}
            suggestion={suggestion.data}
            cleaning={cleaning.data}
            options={options.data}
            taskType={taskType}
          />
        </div>
        <div className="min-w-0 @5xl/main:sticky @5xl/main:top-6 @5xl/main:max-h-[calc(100svh-3rem)] @5xl/main:self-start @5xl/main:overflow-y-auto @5xl/main:pr-2">
          {impossible ? (
            <p className="text-sm text-soft">
              No task is possible with this target: choose another target column.
            </p>
          ) : (
            <PrepareResult
              summary={summary.data}
              pending={pending && request !== null}
              error={summary.error}
            />
          )}
        </div>
      </div>
    </>
  );
}
