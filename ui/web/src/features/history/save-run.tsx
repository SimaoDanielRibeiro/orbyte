/**
 * "Guardar experiência": junta ao histórico o modelo (ou o agente) e o treino passo a passo de
 * um run desta sessão da API. Cada run já fica registado sozinho quando acaba; guardar é o que o
 * faz voltar com tudo, mesmo depois de a API recomeçar.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Save } from "lucide-react";
import { toast } from "sonner";

import {
  historyEntryOptions,
  historySaveMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { ErrorNote } from "@/components/shell/blocks";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

import { invalidateHistory } from "./cache";

export function SaveRunButton({ runId }: { runId: string }) {
  const queryClient = useQueryClient();
  // O registo dá-se quando o run acaba: logo a seguir, a entrada pode ainda não estar lá.
  const entry = useQuery({
    ...historyEntryOptions({ path: { run_id: runId } }),
    retry: 3,
    retryDelay: 600,
  });
  const save = useMutation({
    ...historySaveMutation(),
    onSuccess: () => {
      invalidateHistory(queryClient);
      toast.success("Experiência guardada: está no ecrã Experiências.");
    },
  });
  if (entry.error !== null) {
    return null; // o run não ficou no histórico (falhou, ou foi apagado)
  }
  const saved = entry.data?.saved === true;
  return (
    <div className="flex flex-col items-start gap-2">
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            disabled={saved || save.isPending || entry.data === undefined}
            onClick={() => {
              save.mutate({ path: { run_id: runId }, body: { name: null } });
            }}
          >
            {saved ? <Check aria-hidden="true" /> : <Save aria-hidden="true" />}
            {saved ? "Experiência guardada" : "Guardar experiência"}
          </Button>
        </TooltipTrigger>
        <TooltipContent className="max-w-72">
          Cada run já fica no histórico. Guardar junta-lhe o modelo treinado e o treino passo a
          passo: volta a abrir com tudo, mesmo depois de a API recomeçar.
        </TooltipContent>
      </Tooltip>
      <ErrorNote error={save.error} />
    </div>
  );
}
