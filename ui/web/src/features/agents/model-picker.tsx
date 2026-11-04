/**
 * A escolha dos modelos que os agentes procuram (ADR 0049): todos à partida, e cada um pode
 * sair (ou voltar). Arrumam-se pela família, porque é o agente da família que os experimenta;
 * a caixa da família marca ou desmarca os dela de uma vez, e fica a meio quando só alguns
 * estão escolhidos.
 *
 * O que entra decide-o o backend: a lista é a das opções (`AutoMLOptions.agents`, os modelos
 * que se podem medir com a métrica escolhida), e os que não se podem medir vêm à parte, com a
 * razão (`skipped`). Aqui só se escolhe entre eles.
 */
import type { ModelAgentCard, SkippedModel } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { withAlpha } from "@/lib/colors";
import { familyColor } from "@/lib/families";
import { formatQuantity } from "@/lib/format";

interface ModelPickerProps {
  /** Os agentes, com os modelos que podem entrar (pela ordem do catálogo). */
  agents: readonly ModelAgentCard[];
  /** Os modelos da tarefa que não se podem medir com esta métrica, com a razão. */
  unavailable: readonly SkippedModel[];
  /** Os modelos escolhidos, pelo nome no registo. */
  chosen: readonly string[];
  onChange: (models: string[]) => void;
  /** O id do título, para a secção que o usa se nomear por ele. */
  titleId: string;
}

export function ModelPicker({ agents, unavailable, chosen, onChange, titleId }: ModelPickerProps) {
  const all = agents.flatMap((agent) => agent.models.map((model) => model.name));
  const picked = new Set(chosen);
  const count = all.filter((name) => picked.has(name)).length;

  /** Marca ou desmarca `names`, e devolve a escolha pela ordem do catálogo. */
  function toggle(names: readonly string[], on: boolean) {
    const next = new Set(picked);
    for (const name of names) {
      if (on) {
        next.add(name);
      } else {
        next.delete(name);
      }
    }
    onChange(all.filter((name) => next.has(name)));
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex items-center gap-1.5">
          <h2 id={titleId} className="text-xl">
            The models
          </h2>
          <Explain title="The models">
            Each family has an agent, and the agent tries the models of its family that are ticked
            here. They all start ticked: untick the ones you want to leave out of the search.
          </Explain>
        </div>
        <div className="flex items-center gap-1">
          <span role="status" className="mr-2 text-sm text-dim tabular-nums">
            {count} of {formatQuantity(all.length, "model", "models")}
          </span>
          <Button
            variant="ghost"
            size="sm"
            disabled={count === all.length}
            onClick={() => {
              onChange(all);
            }}
          >
            All
          </Button>
          <Button
            variant="ghost"
            size="sm"
            disabled={count === 0}
            onClick={() => {
              onChange([]);
            }}
          >
            None
          </Button>
        </div>
      </div>

      <ul className="flex flex-col gap-2.5">
        {agents.map((agent) => (
          <FamilyGroup key={agent.family} agent={agent} picked={picked} onToggle={toggle} />
        ))}
      </ul>

      {count === 0 && (
        <p role="alert" className="text-sm text-warning">
          Tick at least one model: the agents need something to search.
        </p>
      )}
      {unavailable.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="small-label">Can&apos;t join with this metric</h3>
          <ul className="flex flex-wrap gap-2">
            {unavailable.map((model) => (
              <li
                key={model.model}
                className="flex items-center gap-1 rounded-full border border-[#2a2a2a] py-0.5 pr-1 pl-3 text-sm text-dim"
              >
                {model.display_name}
                <Explain title={model.display_name}>{`${model.reason}.`}</Explain>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

interface FamilyGroupProps {
  agent: ModelAgentCard;
  picked: ReadonlySet<string>;
  onToggle: (names: readonly string[], on: boolean) => void;
}

/** Uma família: a caixa que marca os modelos dela todos, e um modelo por linha. */
function FamilyGroup({ agent, picked, onToggle }: FamilyGroupProps) {
  const label = useLabel();
  const color = familyColor(agent.family);
  const names = agent.models.map((model) => model.name);
  const on = names.filter((name) => picked.has(name)).length;
  const id = `modelos-familia-${agent.family}`;
  return (
    <li
      className="rounded-2xl border bg-surface p-1.5 transition-[border-color] duration-500"
      // A cor da família no contorno: inteira com modelos escolhidos, fraca sem nenhum.
      style={{ borderColor: withAlpha(color, on > 0 ? 0.45 : 0.12) }}
    >
      <div className="flex items-center gap-3 rounded-xl px-3 py-2.5">
        <Checkbox
          id={id}
          checked={on === names.length ? true : on === 0 ? false : "indeterminate"}
          onCheckedChange={(value) => {
            onToggle(names, value === true);
          }}
        />
        <span
          aria-hidden="true"
          className="size-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: color }}
        />
        <label htmlFor={id} className="flex-1 cursor-pointer text-base font-medium text-strong">
          {label("ModelFamily", agent.family)}
        </label>
        <span className="font-mono text-sm text-dim tabular-nums">
          {on}/{names.length}
        </span>
      </div>
      <ul className="flex flex-col pb-1">
        {agent.models.map((model) => {
          const modelId = `modelos-${model.name}`;
          return (
            <li
              key={model.name}
              className="flex items-center gap-2 rounded-lg pr-2 transition-colors hover:bg-surface-high"
            >
              <label
                htmlFor={modelId}
                className="flex flex-1 cursor-pointer items-center gap-3 py-2 pl-10"
              >
                <Checkbox
                  id={modelId}
                  checked={picked.has(model.name)}
                  onCheckedChange={(value) => {
                    onToggle([model.name], value === true);
                  }}
                />
                <span className="text-sm text-soft">{model.display_name}</span>
              </label>
              <span className="text-xs text-dim">
                {model.searches.length === 0
                  ? "defaults only"
                  : formatQuantity(model.searches.length, "setting", "settings")}
              </span>
              <Explain title={model.display_name}>
                {model.searches.length === 0
                  ? "Nothing to tune: the agent measures it with its default settings."
                  : `What the agent searches: ${model.searches.join("; ")}.`}
              </Explain>
            </li>
          );
        })}
      </ul>
    </li>
  );
}
