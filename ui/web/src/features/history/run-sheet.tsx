/**
 * A gaveta de um run do histórico: o que ele é e o que se pode fazer com ele.
 *
 * - **Abrir**: no ecrã que é o dele (Treinar e avaliar, ou o RL);
 * - **Reproduzir**: treinar outra vez com a mesma configuração e a mesma seed (o run novo entra
 *   no histórico, para os comparar);
 * - **Guardar**: juntar o modelo e o treino passo a passo (só nos runs desta sessão da API);
 * - **Mudar o nome** e **apagar** (com confirmação: não se desfaz).
 */
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Play, RotateCcw, Save, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import type { RunEntry } from "@/api/generated";
import {
  historyDeleteMutation,
  historyRenameMutation,
  historyReproduceMutation,
  historySaveMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { ErrorNote, Facts, Stat } from "@/components/shell/blocks";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { formatMoment } from "@/lib/format";

import { invalidateHistory } from "./cache";
import { useOpenRun } from "./open-run";
import { primaryValue } from "./values";

interface RunSheetProps {
  /** O run aberto (`null`: a gaveta fechada). */
  entry: RunEntry | null;
  onClose: () => void;
  /** Depois de apagar o run (para o tirar da comparação). */
  onDeleted: (runId: string) => void;
}

export function RunSheet({ entry, onClose, onDeleted }: RunSheetProps) {
  return (
    <Sheet
      open={entry !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-lg">
        {entry !== null && (
          <RunDetails key={entry.run_id} entry={entry} onClose={onClose} onDeleted={onDeleted} />
        )}
      </SheetContent>
    </Sheet>
  );
}

function RunDetails({
  entry,
  onClose,
  onDeleted,
}: {
  entry: RunEntry;
  onClose: () => void;
  onDeleted: (runId: string) => void;
}) {
  const label = useLabel();
  const queryClient = useQueryClient();
  const openRun = useOpenRun();
  const [name, setName] = useState(entry.name);
  const path = { run_id: entry.run_id };

  const reproduce = useMutation({
    ...historyReproduceMutation(),
    onSuccess: (job) => {
      onClose();
      openRun(job.run_id, entry.task_type, entry.model_display_name);
    },
  });
  const save = useMutation({
    ...historySaveMutation(),
    onSuccess: () => {
      invalidateHistory(queryClient);
      toast.success("Experiência guardada.");
    },
  });
  const rename = useMutation({
    ...historyRenameMutation(),
    onSuccess: () => {
      invalidateHistory(queryClient);
    },
  });
  const remove = useMutation({
    ...historyDeleteMutation(),
    onSuccess: () => {
      invalidateHistory(queryClient);
      onDeleted(entry.run_id);
      onClose();
      toast.success("Run apagado do histórico.");
    },
  });

  return (
    <>
      <SheetHeader className="gap-2 border-b p-6">
        <SheetTitle className="pr-8 text-xl font-medium text-strong">{entry.name}</SheetTitle>
        <SheetDescription className="text-sm leading-relaxed text-soft">
          {entry.model_display_name} em {entry.dataset_name},{" "}
          {label("TaskType", entry.task_type).toLowerCase()}.
        </SheetDescription>
      </SheetHeader>
      <div className="flex flex-col gap-6 p-6">
        <div className="flex flex-col gap-5 border-t pt-5">
          <Stat
            label={entry.primary_label ?? "Métrica principal"}
            value={primaryValue(entry)}
            detail={label("Partition", entry.partition)}
          />
          <Facts
            size="sm"
            facts={[
              { label: "Quando", value: formatMoment(entry.created_at) },
              { label: "Seed", value: entry.seed },
              { label: "Run", value: <span className="font-mono text-xs">{entry.run_id}</span> },
              { label: "Guardado", value: entry.saved ? "Sim" : "Não" },
            ]}
          />
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Button
            onClick={() => {
              onClose();
              openRun(entry.run_id, entry.task_type, entry.model_display_name);
            }}
          >
            <Play aria-hidden="true" />
            Abrir
          </Button>
          <Button
            variant="outline"
            disabled={reproduce.isPending}
            onClick={() => {
              reproduce.mutate({ path });
            }}
          >
            <RotateCcw aria-hidden="true" />
            Reproduzir
          </Button>
          <Button
            variant="outline"
            disabled={entry.saved || save.isPending}
            onClick={() => {
              save.mutate({ path, body: { name: null } });
            }}
          >
            {entry.saved ? <Check aria-hidden="true" /> : <Save aria-hidden="true" />}
            {entry.saved ? "Guardado" : "Guardar"}
          </Button>
          <Explain title="Reproduzir e guardar" align="end">
            Reproduzir treina outra vez com a mesma configuração e a mesma seed: o run novo entra no
            histórico, para os comparares. Guardar só funciona nos runs que a API ainda tem em
            memória (os últimos).
          </Explain>
        </div>
        <ErrorNote error={reproduce.error ?? save.error} />

        <form
          className="flex flex-col gap-2 border-t pt-6"
          onSubmit={(event) => {
            event.preventDefault();
            rename.mutate({ path, body: { name } });
          }}
        >
          <label htmlFor="run-nome" className="text-sm text-soft">
            Nome
          </label>
          <div className="flex gap-2">
            <Input
              id="run-nome"
              name="nome"
              autoComplete="off"
              value={name}
              onChange={(event) => {
                setName(event.target.value);
              }}
            />
            <Button
              type="submit"
              variant="outline"
              disabled={rename.isPending || name.trim() === entry.name}
            >
              Mudar o nome
            </Button>
          </div>
          <ErrorNote error={rename.error} />
        </form>

        <div className="flex flex-col items-start gap-2 border-t pt-6">
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive">
                <Trash2 aria-hidden="true" />
                Apagar o run
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Apagar «{entry.name}»?</AlertDialogTitle>
                <AlertDialogDescription>
                  O run sai do histórico com os seus ficheiros (o modelo e o treino passo a passo,
                  se estiver guardado). Não se pode desfazer.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancelar</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => {
                    remove.mutate({ path });
                  }}
                >
                  Apagar
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <ErrorNote error={remove.error} />
        </div>
      </div>
    </>
  );
}
