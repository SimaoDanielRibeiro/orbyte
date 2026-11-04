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
      <SheetContent side="right" className="w-full overflow-y-auto data-[side=right]:sm:max-w-lg">
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
      toast.success("Run saved.");
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
      toast.success("Run deleted from the history.");
    },
  });

  return (
    <>
      <SheetHeader className="gap-2 border-b p-6">
        <SheetTitle className="pr-8 text-xl font-medium text-strong">{entry.name}</SheetTitle>
        <SheetDescription className="text-sm leading-relaxed text-soft">
          {entry.model_display_name} on {entry.dataset_name},{" "}
          {label("TaskType", entry.task_type).toLowerCase()}.
        </SheetDescription>
      </SheetHeader>
      <div className="flex flex-col gap-6 p-6">
        <div className="flex flex-col gap-5">
          <Stat
            label={entry.primary_label ?? "Main metric"}
            value={primaryValue(entry)}
            detail={label("Partition", entry.partition)}
          />
          <Facts
            size="sm"
            facts={[
              { label: "When", value: formatMoment(entry.created_at) },
              { label: "Seed", value: entry.seed },
              { label: "Run", value: <span className="font-mono text-xs">{entry.run_id}</span> },
              { label: "Saved", value: entry.saved ? "Yes" : "No" },
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
            Open
          </Button>
          <Button
            variant="outline"
            disabled={reproduce.isPending}
            onClick={() => {
              reproduce.mutate({ path });
            }}
          >
            <RotateCcw aria-hidden="true" />
            Replay
          </Button>
          <Button
            variant="outline"
            disabled={entry.saved || save.isPending}
            onClick={() => {
              save.mutate({ path, body: { name: null } });
            }}
          >
            {entry.saved ? <Check aria-hidden="true" /> : <Save aria-hidden="true" />}
            {entry.saved ? "Saved" : "Save"}
          </Button>
          <Explain title="Replay and save" align="end">
            Replay trains again with the same configuration and the same seed: the new run goes into
            the history, so you can compare them. Save only works on the runs the API still has in
            memory (the latest ones).
          </Explain>
        </div>
        <ErrorNote error={reproduce.error ?? save.error} />

        <form
          className="flex flex-col gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            rename.mutate({ path, body: { name } });
          }}
        >
          <label htmlFor="run-nome" className="text-sm text-soft">
            Name
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
              Rename
            </Button>
          </div>
          <ErrorNote error={rename.error} />
        </form>

        <div className="flex flex-col items-start gap-2">
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive">
                <Trash2 aria-hidden="true" />
                Delete the run
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete “{entry.name}”?</AlertDialogTitle>
                <AlertDialogDescription>
                  The run leaves the history with its files (the model and the step-by-step
                  training, if it was saved). This can&apos;t be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => {
                    remove.mutate({ path });
                  }}
                >
                  Delete
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
