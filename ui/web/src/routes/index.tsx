import { createFileRoute, useNavigate } from "@tanstack/react-router";
import type { ReactNode } from "react";

import { Page, PageHeader } from "@/components/shell/page";
import { AgentsDiagram, ManualDiagram } from "@/components/shell/path-diagrams";
import { PATH_NAMES } from "@/components/shell/screens";
import { Button } from "@/components/ui/button";
import { useSession, type WorkPath } from "@/state/session";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  const navigate = useNavigate();
  const choosePath = useSession((state) => state.choosePath);

  function start(path: WorkPath) {
    choosePath(path);
    void navigate({ to: "/dados" });
  }

  return (
    <Page>
      <PageHeader
        title="Como queres trabalhar?"
        description="Os dois caminhos começam nos dados e acabam no mesmo sítio: um run que se treina, avalia e explica."
      />
      <div className="mt-12 grid gap-12 @4xl/main:grid-cols-2">
        <PathOption
          title={PATH_NAMES.manual}
          text="Escolhes os dados, a preparação e o modelo, e vês cada passo do treino."
          action="Começar à mão"
          onStart={() => {
            start("manual");
          }}
        >
          <ManualDiagram />
        </PathOption>
        <PathOption
          title={PATH_NAMES.agents}
          text="Os agentes experimentam modelos com validação cruzada e mostram cada passo do que fazem."
          action="Começar com os agentes"
          onStart={() => {
            start("agents");
          }}
        >
          <AgentsDiagram />
        </PathOption>
      </div>
    </Page>
  );
}

interface PathOptionProps {
  title: string;
  text: string;
  action: string;
  onStart: () => void;
  /** O desenho do caminho. */
  children: ReactNode;
}

function PathOption({ title, text, action, onStart, children }: PathOptionProps) {
  return (
    <section className="flex flex-col gap-5 border-t pt-6">
      <div>
        <h2 className="text-2xl">{title}</h2>
        <p className="mt-2 max-w-[60ch] font-serif text-reading text-muted-foreground">{text}</p>
      </div>
      {children}
      <Button variant="outline" size="lg" className="self-start" onClick={onStart}>
        {action}
      </Button>
    </section>
  );
}
