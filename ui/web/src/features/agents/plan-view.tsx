/**
 * O plano de uma procura: as linhas que os agentes vão usar, o que o agente de dados decidiu,
 * os agentes de modelo e o que cada um procura, e os modelos que ficam de fora.
 *
 * Fica à vista o que se decidiu; o porquê de cada decisão, e o que cada modelo procura, ficam
 * nos "?" (ADR 0032). Cada agente de modelo tem o contorno com a cor da família (ADR 0038).
 */
import { Check } from "lucide-react";

import type { AutoMlPlan } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { Stat } from "@/components/shell/blocks";
import { withAlpha } from "@/lib/colors";
import { familyColor } from "@/lib/families";
import { formatCount } from "@/lib/format";

interface PlanViewProps {
  plan: AutoMlPlan;
  /**
   * Se mostra os agentes, os modelos de cada um e os que ficam de fora. No pedido não: a escolha
   * dos modelos já os mostra, ao lado (ADR 0049).
   */
  models?: boolean;
}

export function PlanView({ plan, models = true }: PlanViewProps) {
  const sampled = plan.search_rows < plan.train_rows;
  return (
    <div className="flex flex-col gap-7">
      <div className="flex flex-wrap gap-x-10 gap-y-4">
        <Stat label="Training rows" value={formatCount(plan.train_rows)} />
        <Stat
          label={
            <Labelled title="Test rows">
              Kept aside to measure only the winner, at the end.
            </Labelled>
          }
          value={formatCount(plan.test_rows)}
        />
        {sampled && (
          <Stat
            label={
              <Labelled title="Search sample">
                Part of the training set, so each candidate's cross-validation is quick.
              </Labelled>
            }
            value={formatCount(plan.search_rows)}
          />
        )}
      </div>

      <section className="flex flex-col gap-3">
        <h3 className="small-label">
          {plan.adopted ? "Your preparation" : "What the data agent decided"}
        </h3>
        <ul className="flex flex-col gap-2">
          {plan.decisions.map((decision) => (
            <li key={decision.title} className="flex items-center gap-2 text-sm">
              <Check aria-hidden="true" className="size-4 shrink-0 text-positive" />
              <span className="text-strong">{decision.title}</span>
              <Explain title={decision.title}>{decision.reason}</Explain>
            </li>
          ))}
        </ul>
      </section>

      {models && <ModelAgents plan={plan} />}
    </div>
  );
}

/** Os agentes de modelo e o que cada um procura, e os modelos que ficam de fora. */
function ModelAgents({ plan }: { plan: AutoMlPlan }) {
  const label = useLabel();
  return (
    <>
      <section className="flex flex-col gap-3">
        <h3 className="small-label">The model agents and what they search</h3>
        <ul className="grid gap-3 @4xl/main:grid-cols-2">
          {plan.agents.map((agent) => (
            <li
              key={agent.family}
              className="rounded-xl border bg-ground p-4"
              style={{ borderColor: withAlpha(familyColor(agent.family), 0.5) }}
            >
              <p className="text-sm text-strong">{label("ModelFamily", agent.family)}</p>
              <ul className="mt-2 flex flex-col gap-1">
                {agent.models.map((model) => (
                  <li key={model.name} className="flex items-center gap-1 text-sm text-soft">
                    {model.display_name}
                    <Explain title={model.display_name}>
                      {model.searches.length === 0
                        ? "Only the default values."
                        : model.searches.join("; ")}
                    </Explain>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      </section>

      {plan.skipped.length > 0 && (
        <section className="flex flex-col gap-3">
          <h3 className="small-label">Left out</h3>
          <ul className="flex flex-wrap gap-2">
            {plan.skipped.map((skipped) => (
              <li
                key={skipped.model}
                className="flex items-center gap-1 rounded-full border border-[#2a2a2a] py-0.5 pr-1 pl-3 text-xs text-soft"
              >
                {skipped.display_name}
                <Explain title={skipped.display_name}>{skipped.reason}.</Explain>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

/** O nome de um número com o "?" ao lado, onde fica o que ele quer dizer. */
function Labelled({ title, children }: { title: string; children: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      {title}
      <Explain title={title}>{children}</Explain>
    </span>
  );
}
