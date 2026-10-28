/**
 * O ecrã Preparar: o alvo e a tarefa, a limpeza, a divisão e as transformações, e o que sai.
 *
 * A preparação volta a correr sempre que um controlo muda (depois de uma pausa curta), para se
 * ver logo o efeito de cada escolha na linhagem, nas partes e na projeção. A que corre bem fica
 * na sessão: é com ela que se treina. Num ecrã largo, o resultado fica ao lado dos controlos e
 * acompanha a página.
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query";
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
import { NextStep, Proceed } from "@/components/shell/blocks";
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
        <Proceed to="/modelos" disabled={!ready}>
          Escolher o modelo
        </Proceed>
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
      <div className="mt-8 grid gap-x-12 gap-y-10 @6xl/main:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <PrepareForm
          dataset={dataset}
          choices={choices}
          onChange={change}
          suggestion={suggestion.data}
          cleaning={cleaning.data}
          options={options.data}
          taskType={taskType}
        />
        <div className="min-w-0 @6xl/main:sticky @6xl/main:top-6 @6xl/main:max-h-[calc(100svh-3rem)] @6xl/main:self-start @6xl/main:overflow-y-auto @6xl/main:pr-2">
          {impossible ? (
            <p className="text-sm text-soft">
              Com este alvo não há nenhuma tarefa possível: escolhe outra coluna-alvo.
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
      {current !== null && (
        <NextStep text="A preparação fica guardada para o treino. A seguir, escolhe o modelo que vai aprender com estes dados.">
          <Proceed to="/modelos">Escolher o modelo</Proceed>
        </NextStep>
      )}
    </>
  );
}
