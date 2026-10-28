/**
 * O ecrã Reinforcement learning, antes de treinar: o ambiente (com os seus parâmetros), o
 * agente (com a ficha didática), os hiperparâmetros e a seed.
 *
 * Os formulários geram-se do JSON Schema dos cartões, como os hiperparâmetros de um modelo. O
 * treino corre em segundo plano na API; daqui segue-se para o ecrã do treino, que o acompanha.
 */
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "@tanstack/react-router";
import { LoaderCircle } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { AgentCard, EnvironmentCard } from "@/api/generated";
import {
  reinforcementAgentsOptions,
  reinforcementEnvironmentsOptions,
  reinforcementStartMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import { useLabel } from "@/api/vocabulary";
import { Explain } from "@/components/explain/explain";
import { schemaDefaults, type Values } from "@/components/forms/schema";
import { SchemaForm } from "@/components/forms/schema-form";
import { Prose } from "@/components/markdown/prose";
import { ErrorNote } from "@/components/shell/blocks";
import { Page, PageHeader } from "@/components/shell/page";
import { SCREENS } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { isValidSeed, MAX_SEED } from "@/lib/seed";
import { useSession } from "@/state/session";

export function SetupScreen() {
  const environments = useQuery({ ...reinforcementEnvironmentsOptions(), staleTime: "static" });
  const agents = useQuery({ ...reinforcementAgentsOptions(), staleTime: "static" });
  const lastRun = useSession((state) => state.agentRunId);

  return (
    <Page>
      <PageHeader
        title={SCREENS.reinforcement.title}
        description={SCREENS.reinforcement.description}
        topics={[
          "reinforcement_learning",
          "q_values",
          "temporal_difference",
          "exploration",
          "discount",
          "deep_q_network",
        ]}
      />
      {lastRun !== null && (
        <p className="mt-4 text-sm text-soft">
          <Link
            to="/rl/$runId"
            params={{ runId: lastRun }}
            className="underline-offset-4 hover:text-strong hover:underline"
          >
            Voltar ao último agente treinado
          </Link>
        </p>
      )}
      <ErrorNote error={environments.error ?? agents.error} className="mt-8" />
      {environments.data === undefined || agents.data === undefined ? (
        environments.error === null &&
        agents.error === null && <Skeleton className="mt-10 h-96 w-full" />
      ) : (
        <Setup environments={environments.data} agents={agents.data} />
      )}
    </Page>
  );
}

function Setup({
  environments,
  agents,
}: {
  environments: readonly EnvironmentCard[];
  agents: readonly AgentCard[];
}) {
  const label = useLabel();
  const navigate = useNavigate();
  const openAgentRun = useSession((state) => state.openAgentRun);
  const [environmentName, setEnvironmentName] = useState(environments[0]?.name ?? "");
  const [agentName, setAgentName] = useState(agents[0]?.name ?? "");
  // Os valores de cada ambiente e de cada agente ficam guardados ao trocar entre eles.
  const [parameters, setParameters] = useState<Record<string, Values>>({});
  const [hyperparameters, setHyperparameters] = useState<Record<string, Values>>({});
  const [seed, setSeed] = useState(42);
  const environment = environments.find((one) => one.name === environmentName);
  const agent = agents.find((one) => one.name === agentName);
  const start = useMutation({
    ...reinforcementStartMutation(),
    onSuccess: (job) => {
      openAgentRun(job.run_id);
      void navigate({ to: "/rl/$runId", params: { runId: job.run_id } });
    },
  });
  if (environment === undefined || agent === undefined) {
    return null;
  }
  const environmentValues = parameters[environment.name] ?? {
    ...schemaDefaults(environment.parameters_schema),
    ...environment.defaults,
  };
  const agentValues = hyperparameters[agent.name] ?? {
    ...schemaDefaults(agent.hyperparameters_schema),
    ...agent.defaults,
  };
  const validSeed = isValidSeed(seed);

  return (
    <div className="mt-8 grid gap-x-12 gap-y-10 @5xl/main:grid-cols-2">
      <Group title="O ambiente">
        {environments.length > 1 && (
          <Field label="Ambiente" htmlFor="rl-ambiente">
            <Select value={environment.name} onValueChange={setEnvironmentName}>
              <SelectTrigger id="rl-ambiente" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {environments.map((one) => (
                  <SelectItem key={one.name} value={one.name}>
                    {one.display_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </Field>
        )}
        <div className="flex items-center gap-1">
          <h3 className="text-lg">{environment.display_name}</h3>
          <Explain title={environment.display_name}>{environment.summary}</Explain>
        </div>
        <SchemaForm
          schema={environment.parameters_schema}
          values={environmentValues}
          onChange={(values) => {
            setParameters((previous) => ({ ...previous, [environment.name]: values }));
          }}
          idPrefix="rl-ambiente"
        />
      </Group>

      <Group title="O agente">
        <ToggleGroup
          aria-label="O agente"
          type="single"
          variant="outline"
          value={agent.name}
          onValueChange={(chosen) => {
            if (chosen !== "") {
              setAgentName(chosen);
            }
          }}
          className="flex-wrap justify-start"
        >
          {agents.map((one) => (
            <ToggleGroupItem key={one.name} value={one.name} className="px-3">
              {one.display_name}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-[#2a2a2a] px-3 py-1 font-mono text-xs text-soft">
            {label("Implementation", agent.implementation)}
          </span>
          <Explain title={agent.display_name}>{agent.summary}</Explain>
        </div>
        {agent.doc !== "" && (
          <Collapsible>
            <CollapsibleTrigger className="rounded-md text-sm text-soft underline-offset-4 outline-none hover:text-strong hover:underline focus-visible:ring-2 focus-visible:ring-ring">
              Ler a ficha didática do {agent.display_name}
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-4">
              <Prose text={agent.doc} />
            </CollapsibleContent>
          </Collapsible>
        )}
        <section className="flex flex-col gap-3 border-t pt-5">
          <h3 className="text-sm font-normal text-soft">Os hiperparâmetros</h3>
          <SchemaForm
            schema={agent.hyperparameters_schema}
            values={agentValues}
            onChange={(values) => {
              setHyperparameters((previous) => ({ ...previous, [agent.name]: values }));
            }}
            idPrefix={`rl-agente-${agent.name}`}
          />
        </section>
        <Field
          label="Seed"
          htmlFor="rl-seed"
          help="A mesma seed dá o mesmo treino: a mesma exploração e o mesmo acaso do ambiente."
        >
          <Input
            id="rl-seed"
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
        </Field>
        <div className="flex flex-col items-start gap-3 border-t pt-5">
          <Button
            size="lg"
            disabled={!validSeed || start.isPending}
            onClick={() => {
              start.mutate({
                body: {
                  environment: { name: environment.name, parameters: environmentValues },
                  agent: { name: agent.name, hyperparameters: agentValues },
                  seed,
                },
              });
            }}
          >
            {start.isPending && <LoaderCircle aria-hidden="true" className="animate-spin" />}
            Treinar o agente
          </Button>
          <ErrorNote error={start.error} />
        </div>
      </Group>
    </div>
  );
}

/** Uma das duas partes do pedido, num cartão. */
function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="flex min-w-0 flex-col gap-6 self-start border-t pt-5">
      <h2 className="text-[0.9375rem] font-normal tracking-normal text-soft">{title}</h2>
      {children}
    </section>
  );
}

function Field({
  label,
  htmlFor,
  help,
  children,
}: {
  label: string;
  htmlFor: string;
  /** A explicação do controlo: fica no "?" ao lado do nome. */
  help?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex min-h-6 items-center gap-1">
        <Label htmlFor={htmlFor} className="text-sm text-soft">
          {label}
        </Label>
        {help !== undefined && <Explain title={label}>{help}</Explain>}
      </div>
      {children}
    </div>
  );
}
