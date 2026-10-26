/**
 * O plano de uma procura: as linhas que os agentes vão usar, o que o agente de dados decidiu,
 * os agentes de modelo e o que cada um procura, e os modelos que ficam de fora.
 */
import type { AutoMlPlan } from "@/api/generated";
import { useLabel } from "@/api/vocabulary";
import { formatCount } from "@/lib/format";

export function PlanView({ plan }: { plan: AutoMlPlan }) {
  const label = useLabel();
  const sampled = plan.search_rows < plan.train_rows;
  return (
    <div className="flex flex-col gap-8">
      <p className="max-w-[68ch] font-serif text-reading text-muted-foreground">
        {formatCount(plan.train_rows)} linhas de treino e {formatCount(plan.test_rows)} de teste,
        guardadas até ao fim para o vencedor.
        {sampled &&
          ` A validação cruzada usa uma amostra de ${formatCount(plan.search_rows)} linhas do treino, para a procura ser rápida.`}
      </p>
      <section className="flex flex-col gap-3">
        <h3 className="text-lg">
          {plan.adopted ? "A tua preparação" : "O que o agente de dados decidiu"}
        </h3>
        <ul className="flex max-w-[68ch] flex-col gap-2 font-serif text-reading">
          {plan.decisions.map((decision) => (
            <li key={decision.title}>
              <strong className="font-semibold">{decision.title}.</strong> {decision.reason}
            </li>
          ))}
        </ul>
      </section>
      <section className="flex flex-col gap-3">
        <h3 className="text-lg">Os agentes de modelo e o que procuram</h3>
        <ul className="grid gap-x-10 gap-y-5 @4xl/main:grid-cols-2">
          {plan.agents.map((agent) => (
            <li key={agent.family} className="border-t pt-3">
              <p className="font-medium">{label("ModelFamily", agent.family)}</p>
              <ul className="mt-2 flex flex-col gap-2">
                {agent.models.map((model) => (
                  <li key={model.name} className="text-sm">
                    <span>{model.display_name}</span>
                    <span className="block font-serif text-muted-foreground">
                      {model.searches.length === 0
                        ? "Só os valores por omissão."
                        : model.searches.join("; ")}
                    </span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      </section>
      {plan.skipped.length > 0 && (
        <section className="flex flex-col gap-3">
          <h3 className="text-lg">Ficam de fora</h3>
          <ul className="flex max-w-[68ch] flex-col gap-1.5 font-serif text-sm text-muted-foreground">
            {plan.skipped.map((skipped) => (
              <li key={skipped.model}>
                <span className="text-foreground">{skipped.display_name}</span>: {skipped.reason}.
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
